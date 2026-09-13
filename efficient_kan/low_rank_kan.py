"""
LowRankKAN (Bottleneck KAN / LoRA-KAN):
Kolmogorov-Arnold Networks with Low-Rank Factorized Spline Projections.
Reduces parameter complexity from O(d_out * d_in * K) to O((d_out + d_in * K) * rank),
delivering up to 10x memory savings and faster GEMMs on Apple Silicon.
"""

from __future__ import annotations
import math
from typing import Callable, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from .fast_kan import compute_rbf_basis


def _kaiming_uniform(shape: Sequence[int], fan_in: int, a: float = math.sqrt(5.0)) -> mx.array:
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(-bound, bound, shape)


class LowRankKANLinear(nn.Module):
    """
    LowRankKAN Layer with rank-factorized spline projection (LoRA / Bottleneck).

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Bottleneck rank r << min(in_features, out_features) (default: 8)
        num_grids: Number of basis centers (default: 8)
        grid_range: Basis interval (default: [-1, 1])
        base_activation: Base branch activation (default: nn.silu)
        bias: Whether to add a bias term (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        num_grids: int = 8,
        grid_range: Sequence[float] = (-1.0, 1.0),
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, out_features, in_features * num_grids)
        self.num_grids = num_grids
        self.grid_range = list(grid_range)

        if isinstance(base_activation, type):
            self.base_activation = base_activation()
        else:
            self.base_activation = base_activation

        h = (self.grid_range[1] - self.grid_range[0]) / max(num_grids - 1, 1)
        self.inv_h = 1.0 / h
        grid_steps = mx.linspace(self.grid_range[0], self.grid_range[1], num_grids)
        self.centers = mx.broadcast_to(grid_steps[None, :], (in_features, num_grids))
        self.freeze(keys=["centers"])

        # Base branch weight
        self.base_weight = _kaiming_uniform((out_features, in_features), in_features)

        # Low-rank factorized spline weights:
        # V projects from (in_features * num_grids) down to rank
        # U projects from rank up to out_features
        self.spline_V = _kaiming_uniform((self.rank, in_features * num_grids), in_features * num_grids)
        self.spline_U = mx.zeros((out_features, self.rank))

        self.scaling = 1.0 / self.rank
        self.bias = mx.zeros((out_features,)) if bias else None

    def rbf_basis(self, x: mx.array) -> mx.array:
        return compute_rbf_basis(x, self.centers, self.inv_h)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Base branch
        base_out = self.base_activation(x_flat) @ self.base_weight.T

        # Low-rank spline branch: Two sequential small GEMMs
        # (batch, in_features * num_grids) @ V.T -> (batch, rank)
        bases = self.rbf_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)

        bottleneck = bases_flat @ self.spline_V.T
        spline_out = (bottleneck @ self.spline_U.T) * self.scaling

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
        """Return a quantized approximation of this LowRankKAN layer."""
        from .quantized import QuantizedLowRankKANLinear
        return QuantizedLowRankKANLinear.from_layer(self, group_size=group_size, bits=bits, mode=mode)

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        l1_u = mx.mean(mx.abs(self.spline_U))
        l1_v = mx.mean(mx.abs(self.spline_V))
        return regularize_activation * (l1_u + l1_v)


class LowRankKAN(nn.Module):
    """Multi-layer LowRankKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        rank: int = 8,
        num_grids: int = 8,
        grid_range: Sequence[float] = (-1.0, 1.0),
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            LowRankKANLinear(
                in_f,
                out_f,
                rank=rank,
                num_grids=num_grids,
                grid_range=grid_range,
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
