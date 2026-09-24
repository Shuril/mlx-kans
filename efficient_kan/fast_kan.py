"""
FastKAN: Kolmogorov-Arnold Networks with Gaussian Radial Basis Functions (RBF).
Based on the FastKAN formulation by Ziyao Li (2024).
Provides ~3x speedup over B-splines and smooth infinite differentiability.
"""

from __future__ import annotations
import math
from typing import Callable, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from .metal_kernels import metal_rbf_basis, is_metal_available


@mx.compile
def compute_rbf_basis(x: mx.array, centers: mx.array, inv_h: float) -> mx.array:
    """Compiled Gaussian RBF basis evaluation."""
    diff = (x[..., None] - centers[None, ...]) * inv_h
    return mx.exp(-diff * diff)


def _kaiming_uniform(shape: Sequence[int], fan_in: int, a: float = math.sqrt(5.0)) -> mx.array:
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(-bound, bound, shape)


class FastKANLinear(nn.Module):
    """
    FastKAN Layer using Gaussian Radial Basis Functions instead of piecewise B-splines.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_grids: Number of Gaussian RBF centers (default: 8)
        grid_range: Initial [min, max] interval (default: [-1, 1])
        base_activation: Activation function for base branch (default: nn.silu)
        use_metal_kernel: If True, uses custom MSL kernel on Apple Silicon (default: False)
        bias: Whether to add a bias term (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grids: int = 8,
        grid_range: Sequence[float] = (-1.0, 1.0),
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        use_metal_kernel: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        self.grid_range = list(grid_range)
        self.use_metal_kernel = use_metal_kernel and is_metal_available()

        if isinstance(base_activation, type):
            self.base_activation = base_activation()
        else:
            self.base_activation = base_activation

        # Grid centers and width
        h = (self.grid_range[1] - self.grid_range[0]) / max(num_grids - 1, 1)
        self.inv_h = 1.0 / h
        grid_steps = mx.linspace(self.grid_range[0], self.grid_range[1], num_grids)
        self.centers = mx.broadcast_to(grid_steps[None, :], (in_features, num_grids))
        self.freeze(keys=["centers"])

        # Base linear weight & RBF spline weights
        self.base_weight = _kaiming_uniform((out_features, in_features), in_features)
        self.spline_weight = _kaiming_uniform((out_features, in_features * num_grids), in_features)

        self.bias = mx.zeros((out_features,)) if bias else None

    def rbf_basis(self, x: mx.array) -> mx.array:
        """Evaluate Gaussian RBF bases on input x."""
        if self.use_metal_kernel and x.ndim == 2:
            return metal_rbf_basis(x, self.centers, self.inv_h)
        return compute_rbf_basis(x, self.centers, self.inv_h)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Base branch
        base_out = self.base_activation(x_flat) @ self.base_weight.T

        # RBF branch
        bases = self.rbf_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        spline_out = bases_flat @ self.spline_weight.T

        out = base_out + spline_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    forward = __call__

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ):
        """Return a quantized approximation of this FastKAN layer."""
        from .quantized import QuantizedFastKANLinear
        return QuantizedFastKANLinear.from_layer(self, group_size=group_size, bits=bits, mode=mode, **kwargs)

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        l1 = mx.mean(mx.abs(self.spline_weight), axis=-1)
        reg_l1 = mx.sum(l1)
        p = l1 / (reg_l1 + 1e-8)
        entropy = -mx.sum(p * mx.log(p + 1e-8))
        return regularize_activation * reg_l1 + regularize_entropy * entropy

    def update_grid(self, x: mx.array, margin: float = 0.01):
        """
        Adaptively update RBF grid centers and width to fit the input data distribution.

        Args:
            x: Input array of shape (*batch_dims, in_features)
            margin: Safety boundary added to the min/max range
        """
        x_flat = x.reshape(-1, self.in_features)
        x_min = mx.min(x_flat, axis=0) - margin
        x_max = mx.max(x_flat, axis=0) + margin

        grid_steps = mx.linspace(0.0, 1.0, self.num_grids)
        new_centers = x_min[:, None] + grid_steps[None, :] * (x_max - x_min)[:, None]

        new_h = (x_max - x_min) / max(self.num_grids - 1, 1)
        self.inv_h = (1.0 / mx.maximum(new_h, 1e-6))[None, :, None]
        self.centers = new_centers
        self.freeze(keys=["centers"])


class FastKAN(nn.Module):
    """Multi-layer FastKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        num_grids: int = 8,
        grid_range: Sequence[float] = (-1.0, 1.0),
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        use_metal_kernel: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            FastKANLinear(
                in_f,
                out_f,
                num_grids=num_grids,
                grid_range=grid_range,
                base_activation=base_activation,
                use_metal_kernel=use_metal_kernel,
                bias=bias,
            )
            for in_f, out_f in zip(self.layers_hidden, self.layers_hidden[1:])
        ]

    def update_grid(self, x: mx.array, margin: float = 0.01):
        """Adaptively update RBF grids layer-by-layer."""
        for layer in self.layers:
            layer.update_grid(x, margin=margin)
            x = layer(x)

    def __call__(self, x: mx.array, update_grid: bool = False) -> mx.array:
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    forward = __call__

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.regularization_loss(regularize_activation, regularize_entropy)
        return total
