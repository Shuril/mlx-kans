"""
Wav-KAN: Wavelet Kolmogorov-Arnold Networks.
Based on Bozorgasl & Chen (2024).
Uses continuous wavelets (Mexican Hat / Ricker, Morlet, DOG) with multiresolution localization
in both time and frequency, mitigating catastrophic forgetting.
"""

from __future__ import annotations
import math
from typing import Callable, Literal, Sequence, Union

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_wavelet_basis(
    x: mx.array,
    translation: mx.array,
    scale: mx.array,
    wavelet_type: str = "mexican_hat",
) -> mx.array:
    """Compiled multiresolution wavelet basis evaluation."""
    # z: (..., in_features, num_wavelets)
    z = (x[..., None] - translation[None, ...]) / (mx.abs(scale[None, ...]) + 1e-4)

    if wavelet_type == "morlet":
        return mx.cos(5.0 * z) * mx.exp(-0.5 * z * z)
    elif wavelet_type == "dog":
        return -z * mx.exp(-0.5 * z * z)
    else:
        # Default: Mexican Hat / Ricker wavelet
        return (1.0 - z * z) * mx.exp(-0.5 * z * z)


def _kaiming_uniform(shape: Sequence[int], fan_in: int, a: float = math.sqrt(5.0)) -> mx.array:
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(-bound, bound, shape)


class WavKANLinear(nn.Module):
    """
    Wavelet KAN Layer.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_wavelets: Number of wavelet scales/translations (default: 8)
        wavelet_type: "mexican_hat", "morlet", or "dog"
        learnable_scales: Whether scales & translations are trainable parameters (default: True)
        base_activation: Base branch activation (default: nn.silu)
        bias: Whether to add a bias term (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_wavelets: int = 8,
        wavelet_type: Literal["mexican_hat", "morlet", "dog"] = "mexican_hat",
        learnable_scales: bool = True,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type
        self.learnable_scales = learnable_scales

        if isinstance(base_activation, type):
            self.base_activation = base_activation()
        else:
            self.base_activation = base_activation

        # Initialize translations evenly across [-1, 1]
        trans_steps = mx.linspace(-1.0, 1.0, num_wavelets)
        self.translation = mx.broadcast_to(trans_steps[None, :], (in_features, num_wavelets))

        # Initialize scales
        scale_init = 2.0 / max(num_wavelets, 1)
        self.scale = mx.full((in_features, num_wavelets), scale_init, dtype=mx.float32)

        if not learnable_scales:
            self.freeze(keys=["translation", "scale"])

        self.base_weight = _kaiming_uniform((out_features, in_features), in_features)
        self.wav_weight = _kaiming_uniform((out_features, in_features * num_wavelets), in_features)
        self.bias = mx.zeros((out_features,)) if bias else None

    def wavelet_basis(self, x: mx.array) -> mx.array:
        return compute_wavelet_basis(x, self.translation, self.scale, self.wavelet_type)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.base_activation(x_flat) @ self.base_weight.T

        bases = self.wavelet_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        wav_out = bases_flat @ self.wav_weight.T

        out = base_out + wav_out
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
        """Return a quantized approximation of this WavKAN layer."""
        from .quantized import QuantizedWavKANLinear
        return QuantizedWavKANLinear.from_layer(self, group_size=group_size, bits=bits, mode=mode, **kwargs)

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        l1 = mx.mean(mx.abs(self.wav_weight), axis=-1)
        reg_l1 = mx.sum(l1)
        p = l1 / (reg_l1 + 1e-8)
        entropy = -mx.sum(p * mx.log(p + 1e-8))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class WavKAN(nn.Module):
    """Multi-layer Wav-KAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        num_wavelets: int = 8,
        wavelet_type: Literal["mexican_hat", "morlet", "dog"] = "mexican_hat",
        learnable_scales: bool = True,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            WavKANLinear(
                in_f,
                out_f,
                num_wavelets=num_wavelets,
                wavelet_type=wavelet_type,
                learnable_scales=learnable_scales,
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
