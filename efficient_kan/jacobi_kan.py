"""
JacobiKAN: Kolmogorov-Arnold Networks with Jacobi Polynomials.
General orthogonal polynomials on [-1, 1] with weighting parameters (alpha, beta).
Includes Legendre (alpha=0, beta=0) and Gegenbauer as special cases.
"""

from __future__ import annotations
import math
from typing import Callable, Sequence, Union

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_jacobi_basis(x: mx.array, degree: int, alpha: float = 0.0, beta: float = 0.0) -> mx.array:
    """Compiled Jacobi polynomial recurrence evaluation."""
    x_c = mx.clip(x, -1.0, 1.0)
    bases = [mx.ones_like(x_c[..., None])]
    if degree > 1:
        p1 = 0.5 * (alpha - beta + (alpha + beta + 2.0) * x_c[..., None])
        bases.append(p1)
        a_b = alpha + beta
        for n in range(2, degree):
            an = 2.0 * n * (n + a_b) * (2.0 * n + a_b - 2.0)
            bn_term1 = (2.0 * n + a_b - 1.0) * (2.0 * n + a_b) * (2.0 * n + a_b - 2.0)
            bn_term2 = (2.0 * n + a_b - 1.0) * (alpha**2 - beta**2)
            bn = bn_term1 * x_c[..., None] + bn_term2
            cn = 2.0 * (n + alpha - 1.0) * (n + beta - 1.0) * (2.0 * n + a_b)
            pn = (bn * bases[-1] - cn * bases[-2]) / an
            bases.append(pn)
    return mx.concatenate(bases, axis=-1)


def _kaiming_uniform(shape: Sequence[int], fan_in: int, a: float = math.sqrt(5.0)) -> mx.array:
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(-bound, bound, shape)


class JacobiKANLinear(nn.Module):
    """
    JacobiKAN Layer.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        degree: Degree of Jacobi polynomials (default: 4)
        alpha: First Jacobi parameter > -1 (default: 0.0, Legendre)
        beta: Second Jacobi parameter > -1 (default: 0.0, Legendre)
        base_activation: Base branch activation (default: nn.silu)
        bias: Whether to add a bias term (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        alpha: float = 0.0,
        beta: float = 0.0,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.alpha = alpha
        self.beta = beta

        if isinstance(base_activation, type):
            self.base_activation = base_activation()
        else:
            self.base_activation = base_activation

        self.base_weight = _kaiming_uniform((out_features, in_features), in_features)
        self.jacobi_weight = _kaiming_uniform((out_features, in_features * degree), in_features)
        self.bias = mx.zeros((out_features,)) if bias else None

    def jacobi_basis(self, x: mx.array) -> mx.array:
        return compute_jacobi_basis(x, self.degree, self.alpha, self.beta)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.base_activation(x_flat) @ self.base_weight.T

        bases = self.jacobi_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        jacobi_out = bases_flat @ self.jacobi_weight.T

        out = base_out + jacobi_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    forward = __call__

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> mx.array:
        l1 = mx.mean(mx.abs(self.jacobi_weight), axis=-1)
        reg_l1 = mx.sum(l1)
        p = l1 / (reg_l1 + 1e-8)
        entropy = -mx.sum(p * mx.log(p + 1e-8))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class JacobiKAN(nn.Module):
    """Multi-layer JacobiKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        degree: int = 4,
        alpha: float = 0.0,
        beta: float = 0.0,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            JacobiKANLinear(
                in_f,
                out_f,
                degree=degree,
                alpha=alpha,
                beta=beta,
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
