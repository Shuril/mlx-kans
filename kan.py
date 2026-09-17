"""
Efficient-KAN implementation in Apple MLX.

Kolmogorov-Arnold Network (KAN) optimized for Apple Silicon (M-series / Metal GPU).
Based on the efficient-kan reformulation by Blealtan (Ziyao Li).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence, Union

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_b_splines(x: mx.array, grid: mx.array, spline_order: int) -> mx.array:
    """
    Compute Cox-de Boor B-spline basis functions for input x and given knot grid.
    Compiled with JIT (mx.compile) for kernel fusion on Metal GPU.

    Args:
        x: Input array of shape (..., in_features)
        grid: Knot grid array of shape (in_features, grid_size + 2 * spline_order + 1)
        spline_order: Order of the B-spline (e.g. 3 for cubic splines)

    Returns:
        B-spline bases of shape (..., in_features, grid_size + spline_order)
    """
    x_exp = x[..., None]

    grid_left = grid[:, :-1][None, ...]
    grid_right = grid[:, 1:][None, ...]

    bases = ((x_exp >= grid_left) & (x_exp < grid_right)).astype(x.dtype)

    for k in range(1, spline_order + 1):
        g_k_neg = grid[:, : -(k + 1)][None, ...]
        g_k_pos = grid[:, k:-1][None, ...]
        g_right_1 = grid[:, k + 1 :][None, ...]
        g_right_2 = grid[:, 1 : -k][None, ...]

        term1 = ((x_exp - g_k_neg) / (g_k_pos - g_k_neg)) * bases[..., :-1]
        term2 = ((g_right_1 - x_exp) / (g_right_1 - g_right_2)) * bases[..., 1:]
        bases = term1 + term2

    return bases


def _kaiming_uniform(shape: Sequence[int], fan_in: int, a: float = math.sqrt(5.0)) -> mx.array:
    """Kaiming uniform weight initialization."""
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(-bound, bound, shape)


class KANLinear(nn.Module):
    """
    Single Kolmogorov-Arnold Network (KAN) Layer implemented in MLX.

    Reformulates spline evaluations on inputs first, turning the spline transformation
    into an efficient linear projection (GEMM) suitable for Metal acceleration on Apple Silicon.

    Args:
        in_features: Number of input dimensions.
        out_features: Number of output dimensions.
        grid_size: Number of grid intervals (knots = grid_size + 2 * spline_order + 1).
        spline_order: Order of B-splines (default 3 for cubic splines).
        scale_noise: Magnitude of uniform noise for spline weight initialization.
        scale_base: Scaling factor for base linear weight initialization.
        scale_spline: Scaling factor for spline weights / scaler.
        enable_standalone_scale_spline: Whether to include a separate learnable scaling parameter.
        base_activation: Activation function applied to the base linear branch (default: nn.silu).
        grid_eps: Mixing factor between uniform and adaptive grid during grid updates.
        grid_range: Initial [min, max] domain for the knot grid (default: [-1, 1]).
        bias: Whether to add a learnable bias term (default: False).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        grid_eps: float = 0.02,
        grid_range: Sequence[float] = (-1.0, 1.0),
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.grid_eps = grid_eps
        self.grid_range = list(grid_range)

        # Base activation function
        if isinstance(base_activation, type):
            self.base_activation = base_activation()
        else:
            self.base_activation = base_activation

        # Initialize uniform knot grid
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        grid_steps = (
            mx.arange(-spline_order, grid_size + spline_order + 1, dtype=mx.float32) * h
            + self.grid_range[0]
        )
        self.grid = mx.broadcast_to(grid_steps[None, :], (in_features, grid_steps.shape[0]))
        self.freeze(keys=["grid"])

        # Optional bias
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters with Kaiming uniform and curve interpolation."""
        # Base linear branch weights: (out_features, in_features)
        self.base_weight = _kaiming_uniform(
            (self.out_features, self.in_features),
            self.in_features,
            a=math.sqrt(5.0) * self.scale_base,
        )

        # Standalone spline scaler
        if self.enable_standalone_scale_spline:
            self.spline_scaler = _kaiming_uniform(
                (self.out_features, self.in_features),
                self.in_features,
                a=math.sqrt(5.0) * self.scale_spline,
            )

        # Initial curve noise for spline weights
        noise = (
            mx.random.uniform(
                0.0, 1.0, (self.grid_size + 1, self.in_features, self.out_features)
            )
            - 0.5
        ) * (self.scale_noise / self.grid_size)

        points_x = self.grid.T[self.spline_order : -self.spline_order]
        coeff = self.curve2coeff(points_x, noise)
        scale_factor = self.scale_spline if not self.enable_standalone_scale_spline else 1.0
        self.spline_weight = coeff * scale_factor

    def b_splines(self, x: mx.array) -> mx.array:
        """
        Evaluate B-spline basis functions for input x.

        Args:
            x: Input array of shape (..., in_features)

        Returns:
            B-spline bases of shape (..., in_features, grid_size + spline_order)
        """
        return compute_b_splines(x, self.grid, self.spline_order)

    def curve2coeff(self, x: mx.array, y: mx.array) -> mx.array:
        """
        Compute spline coefficients interpolating (x, y) via least squares.
        Uses Moore-Penrose pseudo-inverse executed on CPU stream via Apple Unified Memory.

        Args:
            x: (batch_size, in_features)
            y: (batch_size, in_features, out_features)

        Returns:
            (out_features, in_features, grid_size + spline_order)
        """
        assert x.ndim == 2 and x.shape[1] == self.in_features
        assert y.ndim == 3 and y.shape[:2] == x.shape and y.shape[2] == self.out_features

        bases = self.b_splines(x)
        A = mx.swapaxes(bases, 0, 1)  # (in, batch, num_bases)
        B = mx.swapaxes(y, 0, 1)      # (in, batch, out)

        with mx.stream(mx.cpu):
            pinv_A = mx.linalg.pinv(A)
            solution = pinv_A @ B
            mx.eval(solution)

        return mx.transpose(solution, (2, 0, 1))

    @property
    def scaled_spline_weight(self) -> mx.array:
        """Spline weight scaled by spline_scaler if enabled."""
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler[..., None]
        return self.spline_weight

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of KANLinear.

        Args:
            x: Array of shape (*batch_dims, in_features)

        Returns:
            Array of shape (*batch_dims, out_features)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Base branch: y_base = SiLU(x) @ W_base^T
        base_out = self.base_activation(x_flat) @ self.base_weight.T

        # Spline branch: evaluate bases first, then GEMM
        bases = self.b_splines(x_flat)  # (batch, in_features, num_bases)
        bases_flat = bases.reshape(x_flat.shape[0], -1)  # (batch, in_features * num_bases)

        scaled_w = self.scaled_spline_weight.reshape(self.out_features, -1)
        spline_out = bases_flat @ scaled_w.T

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
        """Return a quantized approximation of this KAN layer."""
        from efficient_kan.quantized import QuantizedKANLinear
        return QuantizedKANLinear.from_layer(self, group_size=group_size, bits=bits, mode=mode, **kwargs)

    def update_grid(self, x: mx.array, margin: float = 0.01):
        """
        Adaptively update grid knots to fit the input data distribution while
        preserving existing spline outputs.

        Args:
            x: Input array of shape (*batch_dims, in_features)
            margin: Safety boundary added to the min/max range
        """
        x_flat = x.reshape(-1, self.in_features)
        batch = x_flat.shape[0]
        assert batch >= self.grid_size + 1, (
            f"Batch size ({batch}) must be >= grid_size + 1 ({self.grid_size + 1}) to update grid."
        )

        splines = self.b_splines(x_flat)  # (batch, in, coeff)
        splines = mx.swapaxes(splines, 0, 1)  # (in, batch, coeff)
        orig_coeff = mx.transpose(self.scaled_spline_weight, (1, 2, 0))  # (in, coeff, out)
        unreduced_spline_output = mx.matmul(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = mx.swapaxes(unreduced_spline_output, 0, 1)  # (batch, in, out)

        x_sorted = mx.sort(x_flat, axis=0)

        adaptive_indices = mx.linspace(0, batch - 1, self.grid_size + 1).astype(mx.int32)
        grid_adaptive = x_sorted[adaptive_indices]  # (grid_size + 1, in_features)

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2.0 * margin) / self.grid_size
        grid_uniform = (
            mx.arange(self.grid_size + 1, dtype=mx.float32)[:, None] * uniform_step[None, :]
            + x_sorted[0][None, :]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1.0 - self.grid_eps) * grid_adaptive

        order_down = mx.arange(self.spline_order, 0, -1, dtype=mx.float32)[:, None]
        order_up = mx.arange(1, self.spline_order + 1, dtype=mx.float32)[:, None]

        grid_prefix = grid[:1] - uniform_step[None, :] * order_down
        grid_postfix = grid[-1:] + uniform_step[None, :] * order_up

        new_grid = mx.concatenate([grid_prefix, grid, grid_postfix], axis=0)

        self.grid = new_grid.T
        self.freeze(keys=["grid"])

        new_coeff = self.curve2coeff(x_flat, unreduced_spline_output)
        if self.enable_standalone_scale_spline:
            new_coeff = new_coeff / (self.spline_scaler[..., None] + 1e-8)
        self.spline_weight = new_coeff

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> mx.array:
        """
        Regularization loss (L1 on spline weights + entropy sparsity penalty).

        Args:
            regularize_activation: Weight for L1 magnitude regularizer.
            regularize_entropy: Weight for entropy sparsity regularizer.

        Returns:
            Scalar penalty array.
        """
        l1_fake = mx.mean(mx.abs(self.spline_weight), axis=-1)
        reg_activation = mx.sum(l1_fake)
        p = l1_fake / (reg_activation + 1e-8)
        reg_entropy = -mx.sum(p * mx.log(p + 1e-8))
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy


class KAN(nn.Module):
    """
    Multi-layer Kolmogorov-Arnold Network (KAN) in Apple MLX.

    Args:
        layers_hidden: Sequence of layer widths, e.g. [2, 5, 1].
        grid_size: Number of grid intervals (default: 5).
        spline_order: Order of B-splines (default: 3).
        scale_noise: Noise scale for initial spline weights (default: 0.1).
        scale_base: Scaling factor for base linear weights (default: 1.0).
        scale_spline: Scaling factor for spline weights (default: 1.0).
        enable_standalone_scale_spline: Whether to enable separate spline scalers (default: True).
        base_activation: Activation function for base branch (default: nn.silu).
        grid_eps: Adaptive grid interpolation parameter (default: 0.02).
        grid_range: Initial grid interval (default: [-1, 1]).
        bias: Whether to add bias terms in layers (default: False).
    """

    def __init__(
        self,
        layers_hidden: Sequence[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        grid_eps: float = 0.02,
        grid_range: Sequence[float] = (-1.0, 1.0),
        bias: bool = False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers_hidden = list(layers_hidden)

        self.layers = [
            KANLinear(
                in_features=in_f,
                out_features=out_f,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                bias=bias,
            )
            for in_f, out_f in zip(self.layers_hidden, self.layers_hidden[1:])
        ]

    def __call__(self, x: mx.array, update_grid: bool = False) -> mx.array:
        """
        Forward pass through all KAN layers.

        Args:
            x: Input array of shape (*batch_dims, layers_hidden[0])
            update_grid: If True, updates knot grid for each layer before passing data.

        Returns:
            Output array of shape (*batch_dims, layers_hidden[-1])
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    forward = __call__

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> mx.array:
        """Compute accumulated regularization loss across all layers."""
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.regularization_loss(regularize_activation, regularize_entropy)
        return total
