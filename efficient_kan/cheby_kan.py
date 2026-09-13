"""
ChebyKAN: Kolmogorov-Arnold Networks with Chebyshev Polynomials.
Replaces B-splines with orthogonal Chebyshev polynomials of the 1st kind.
Optimal minimax polynomial approximation on [-1, 1] without grid management.
"""

from __future__ import annotations
import math
from typing import Callable, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from .metal_kernels import metal_cheby_basis, is_metal_available


@mx.compile
def compute_cheby_basis(x: mx.array, degree: int) -> mx.array:
    """Compiled Chebyshev polynomial recurrence evaluation."""
    x_clamped = mx.clip(x, -1.0, 1.0)
    bases = [mx.ones_like(x_clamped[..., None])]
    if degree > 1:
        bases.append(x_clamped[..., None])
        for k in range(2, degree):
            t_next = 2.0 * x_clamped[..., None] * bases[-1] - bases[-2]
            bases.append(t_next)
    return mx.concatenate(bases, axis=-1)


def _kaiming_uniform(shape: Sequence[int], fan_in: int, a: float = math.sqrt(5.0)) -> mx.array:
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(-bound, bound, shape)


class ChebyKANLinear(nn.Module):
    """
    ChebyKAN Layer using Chebyshev polynomials of the 1st kind.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        degree: Number of polynomial basis terms (default: 4)
        base_activation: Base branch activation (default: nn.silu)
        use_metal_kernel: If True, uses custom MSL kernel (default: False)
        bias: Whether to add a bias term (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        use_metal_kernel: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.use_metal_kernel = use_metal_kernel and is_metal_available()

        if isinstance(base_activation, type):
            self.base_activation = base_activation()
        else:
            self.base_activation = base_activation

        self.base_weight = _kaiming_uniform((out_features, in_features), in_features)
        self.cheby_weight = _kaiming_uniform((out_features, in_features * degree), in_features)
        self.bias = mx.zeros((out_features,)) if bias else None

    def cheby_basis(self, x: mx.array) -> mx.array:
        if self.use_metal_kernel and x.ndim == 2:
            return metal_cheby_basis(x, self.degree)
        return compute_cheby_basis(x, self.degree)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.base_activation(x_flat) @ self.base_weight.T

        bases = self.cheby_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        cheby_out = bases_flat @ self.cheby_weight.T

        out = base_out + cheby_out
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
        """Return a quantized approximation of this ChebyKAN layer."""
        from .quantized import QuantizedChebyKANLinear
        return QuantizedChebyKANLinear.from_layer(self, group_size=group_size, bits=bits, mode=mode)

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        l1 = mx.mean(mx.abs(self.cheby_weight), axis=-1)
        reg_l1 = mx.sum(l1)
        p = l1 / (reg_l1 + 1e-8)
        entropy = -mx.sum(p * mx.log(p + 1e-8))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class ChebyKAN(nn.Module):
    """Multi-layer ChebyKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        degree: int = 4,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        use_metal_kernel: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            ChebyKANLinear(
                in_f,
                out_f,
                degree=degree,
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
