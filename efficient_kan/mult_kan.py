"""
MultKAN (KAN 2.0): Kolmogorov-Arnold Networks with Multiplication Nodes.
Based on Ziming Liu et al. (MIT / Harvard / Caltech, 2024).
Introduces explicit multiplicative nodes u * v alongside standard additive nodes,
enabling exact representations of product terms (e.g. physical conservation laws, polynomials).
"""

from __future__ import annotations
import math
from typing import Callable, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from .fast_kan import FastKANLinear


class MultKANLinear(nn.Module):
    """
    MultKAN (KAN 2.0) Layer with additive and multiplicative channels.

    Given input of size `in_features`, produces `num_add` additive outputs and
    `num_mult` multiplicative outputs (each formed by u * v), resulting in
    total output size `num_add + num_mult`.

    Args:
        in_features: Input dimension
        out_features: Total output dimension (num_add + num_mult)
        num_mult: Number of multiplicative nodes (default: max(1, out_features // 2))
        num_grids: Number of RBF centers (default: 8)
        base_activation: Base branch activation (default: nn.silu)
        bias: Whether to add a bias term (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_mult: int | None = None,
        num_grids: int = 8,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if num_mult is None:
            # Default to half multiplication nodes if out_features >= 2
            self.num_mult = max(1, out_features // 2) if out_features > 1 else 0
        else:
            self.num_mult = min(num_mult, out_features)

        self.num_add = out_features - self.num_mult

        # The internal layer produces num_add + 2 * num_mult channels
        internal_out = self.num_add + 2 * self.num_mult

        self.sub_layer = FastKANLinear(
            in_features=in_features,
            out_features=internal_out,
            num_grids=num_grids,
            base_activation=base_activation,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # z: (*batch_dims, num_add + 2 * num_mult)
        z = self.sub_layer(x)

        if self.num_mult == 0:
            return z

        # Split into additive and multiplicative parts
        if self.num_add > 0:
            add_part = z[..., : self.num_add]
            mult_raw = z[..., self.num_add :]
        else:
            add_part = None
            mult_raw = z

        # Reshape multiplicative part into pairs (u, v) and multiply
        # mult_raw shape: (*batch_dims, 2 * num_mult)
        orig_shape = mult_raw.shape
        mult_pairs = mult_raw.reshape(*orig_shape[:-1], self.num_mult, 2)
        mult_part = mult_pairs[..., 0] * mult_pairs[..., 1]

        if add_part is not None:
            return mx.concatenate([add_part, mult_part], axis=-1)
        return mult_part

    forward = __call__

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ):
        """Return a quantized approximation of this MultKAN layer."""
        from .quantized import QuantizedMultKANLinear
        return QuantizedMultKANLinear.from_layer(self, group_size=group_size, bits=bits, mode=mode, **kwargs)

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        return self.sub_layer.regularization_loss(regularize_activation, regularize_entropy)


class MultKAN(nn.Module):
    """Multi-layer MultKAN (KAN 2.0) Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        num_grids: int = 8,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            MultKANLinear(
                in_f,
                out_f,
                num_grids=num_grids,
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
