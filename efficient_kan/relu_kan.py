"""
ReLUKAN: Kolmogorov-Arnold Networks with Rectified Linear Units.
Based on Qiu et al. (2024).
Replaces transcendental functions and splines with piecewise-linear tent functions,
optimizing execution on GPU tensor cores and Apple Silicon Neural Engine / Metal.
"""

from __future__ import annotations
import math
from typing import Callable, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from .metal_kernels import metal_relu_basis, is_metal_available


@mx.compile
def compute_relu_tent_basis(x: mx.array, centers: mx.array, inv_h: float) -> mx.array:
    """Compiled piecewise-linear tent basis evaluation via ReLU."""
    diff = mx.abs(x[..., None] - centers[None, ...]) * inv_h
    return mx.maximum(0.0, 1.0 - diff)


def _kaiming_uniform(shape: Sequence[int], fan_in: int, a: float = math.sqrt(5.0)) -> mx.array:
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(-bound, bound, shape)


class ReLUKANLinear(nn.Module):
    """
    ReLUKAN Layer using piecewise linear tent functions.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_grids: Number of knot centers (default: 8)
        grid_range: Knot interval (default: [-1, 1])
        base_activation: Base branch activation (default: nn.silu)
        use_metal_kernel: If True, uses custom MSL kernel (default: False)
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

        h = (self.grid_range[1] - self.grid_range[0]) / max(num_grids - 1, 1)
        self.inv_h = 1.0 / h
        grid_steps = mx.linspace(self.grid_range[0], self.grid_range[1], num_grids)
        self.centers = mx.broadcast_to(grid_steps[None, :], (in_features, num_grids))
        self.freeze(keys=["centers"])

        self.base_weight = _kaiming_uniform((out_features, in_features), in_features)
        self.spline_weight = _kaiming_uniform((out_features, in_features * num_grids), in_features)
        self.bias = mx.zeros((out_features,)) if bias else None

    def relu_basis(self, x: mx.array) -> mx.array:
        if self.use_metal_kernel and x.ndim == 2:
            return metal_relu_basis(x, self.centers, self.inv_h)
        return compute_relu_tent_basis(x, self.centers, self.inv_h)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.base_activation(x_flat) @ self.base_weight.T

        bases = self.relu_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        spline_out = bases_flat @ self.spline_weight.T

        out = base_out + spline_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    forward = __call__

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        l1 = mx.mean(mx.abs(self.spline_weight), axis=-1)
        reg_l1 = mx.sum(l1)
        p = l1 / (reg_l1 + 1e-8)
        entropy = -mx.sum(p * mx.log(p + 1e-8))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class ReLUKAN(nn.Module):
    """Multi-layer ReLUKAN Network."""

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
            ReLUKANLinear(
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

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x

    forward = __call__

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.regularization_loss(regularize_activation, regularize_entropy)
        return total
