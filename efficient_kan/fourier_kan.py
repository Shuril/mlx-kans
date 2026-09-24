"""
FourierKAN: Kolmogorov-Arnold Networks with Fourier Series.
Harmonic decomposition with trigonometric basis cos(k*pi*x), sin(k*pi*x).
Particularly effective for periodic systems, PDEs, audio, and physics-informed ML (PINNs).
"""

from __future__ import annotations
import math
from typing import Callable, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from .metal_kernels import metal_fourier_basis, is_metal_available


@mx.compile
def compute_fourier_basis(x: mx.array, freqs: mx.array) -> mx.array:
    """Compiled Fourier series basis evaluation."""
    # x: (..., in_features)
    # freqs: (num_frequencies,)
    # kx: (..., in_features, num_frequencies)
    kx = x[..., None] * freqs[None, None, :]
    cos_terms = mx.cos(kx)
    sin_terms = mx.sin(kx)
    const_term = mx.ones_like(x[..., None])
    return mx.concatenate([const_term, cos_terms, sin_terms], axis=-1)


def _kaiming_uniform(shape: Sequence[int], fan_in: int, a: float = math.sqrt(5.0)) -> mx.array:
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(-bound, bound, shape)


class FourierKANLinear(nn.Module):
    """
    FourierKAN Layer.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_frequencies: Number of harmonic frequencies K (bases = 2*K + 1) (default: 4)
        base_activation: Base branch activation (default: nn.silu)
        bias: Whether to add a bias term (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_frequencies: int = 4,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
        use_metal_kernel: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.num_bases = 2 * num_frequencies + 1
        self.use_metal_kernel = use_metal_kernel and is_metal_available()

        if isinstance(base_activation, type):
            self.base_activation = base_activation()
        else:
            self.base_activation = base_activation

        # Frequencies 1*pi, 2*pi, ..., K*pi
        self.freqs = mx.arange(1, num_frequencies + 1, dtype=mx.float32) * mx.pi
        self.freeze(keys=["freqs"])

        self.base_weight = _kaiming_uniform((out_features, in_features), in_features)
        self.fourier_weight = _kaiming_uniform((out_features, in_features * self.num_bases), in_features)
        self.bias = mx.zeros((out_features,)) if bias else None

    def fourier_basis(self, x: mx.array) -> mx.array:
        if self.use_metal_kernel and x.ndim == 2:
            return metal_fourier_basis(x, self.freqs)
        return compute_fourier_basis(x, self.freqs)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.base_activation(x_flat) @ self.base_weight.T

        bases = self.fourier_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        fourier_out = bases_flat @ self.fourier_weight.T

        out = base_out + fourier_out
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
        """Return a quantized approximation of this FourierKAN layer."""
        from .quantized import QuantizedFourierKANLinear
        return QuantizedFourierKANLinear.from_layer(self, group_size=group_size, bits=bits, mode=mode, **kwargs)

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        l1 = mx.mean(mx.abs(self.fourier_weight), axis=-1)
        reg_l1 = mx.sum(l1)
        p = l1 / (reg_l1 + 1e-8)
        entropy = -mx.sum(p * mx.log(p + 1e-8))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class FourierKAN(nn.Module):
    """Multi-layer FourierKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        num_frequencies: int = 4,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            FourierKANLinear(
                in_f,
                out_f,
                num_frequencies=num_frequencies,
                base_activation=base_activation,
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
