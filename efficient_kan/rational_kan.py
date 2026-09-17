"""
RationalKAN: Kolmogorov-Arnold Networks with Padé Rational Activations in Apple MLX.

Replaces polynomial or spline bases with trainable rational Padé approximants:
    phi(x) = P_p(x) / (1 + |Q_q(x)|)

Rational functions achieve exponential convergence on functions with poles,
asymptotes, boundary layers, and rational kinetics (Michaelis-Menten, Voigt,
compressibility factor) using dramatically fewer parameters than pure polynomials.
"""

from __future__ import annotations
import math
from typing import Callable, Sequence, Union, Optional

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_cheby_basis(x: mx.array, degree: int) -> mx.array:
    """Compiled Chebyshev polynomial recurrence evaluation on clamped [-1, 1]."""
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


class RationalKANLinear(nn.Module):
    """
    Rational Padé-Chebyshev KAN Layer.

    Approximates edge activation functions as rational functions:
        phi_{i,j}(x) = P_{i,j}(x) / (1 + |Q_{i,j}(x)|)
    where P and Q are Chebyshev polynomial expansions of degree p and q.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        p_degree: Degree of numerator polynomial P(x) (default: 4)
        q_degree: Degree of denominator polynomial Q(x) (default: 2)
        base_activation: Activation for the residual base linear branch (default: nn.silu)
        bias: Whether to add a bias term (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        p_degree: int = 4,
        q_degree: int = 2,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p_degree = p_degree
        self.q_degree = q_degree
        self.max_degree = max(p_degree, q_degree) + 1

        if isinstance(base_activation, type):
            self.base_activation = base_activation()
        else:
            self.base_activation = base_activation

        # Base branch
        self.base_weight = _kaiming_uniform((out_features, in_features), in_features)

        # Numerator weights: (out_features, in_features, p_degree + 1)
        bound_num = math.sqrt(3.0 / (in_features * (p_degree + 1)))
        self.num_weight = mx.random.uniform(-bound_num, bound_num, (out_features, in_features, p_degree + 1))

        # Denominator weights: (out_features, in_features, q_degree)
        # Initialized close to zero so denominator starts near 1.0 (stable training start)
        self.den_weight = mx.random.uniform(-0.02, 0.02, (out_features, in_features, q_degree))

        self.bias = mx.zeros((out_features,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # 1. Base branch
        base_out = self.base_activation(x_flat) @ self.base_weight.T

        # 2. Chebyshev recurrence evaluation
        T = compute_cheby_basis(x_flat, self.max_degree)  # (B, in_features, max_degree)

        # Numerator: P(x) = sum_{k=0}^p a_k T_k(x)
        T_num = T[:, :, : self.p_degree + 1]  # (B, in_f, p+1)
        P = mx.einsum("bik,oik->boi", T_num, self.num_weight)  # (B, out_f, in_f)

        # Denominator: Q(x) = sum_{m=1}^q b_m T_m(x)
        T_den = T[:, :, 1 : self.q_degree + 1]  # (B, in_f, q)
        Q = mx.einsum("bim,oim->boi", T_den, self.den_weight)  # (B, out_f, in_f)

        # Rational activation: phi(x) = P(x) / (1 + |Q(x)|)
        phi = P / (1.0 + mx.abs(Q))

        # Sum over input dimension
        rational_out = mx.sum(phi, axis=-1)  # (B, out_f)

        out = base_out + rational_out
        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    forward = __call__


class RationalKAN(nn.Module):
    """
    Multi-layer Kolmogorov-Arnold Network with Padé-Chebyshev rational activations.

    Args:
        layers_hidden: Sequence of layer widths, e.g. [2, 8, 1]
        p_degree: Numerator polynomial degree (default: 4)
        q_degree: Denominator polynomial degree (default: 2)
        base_activation: Base branch activation (default: nn.silu)
        bias: Whether to add bias to each layer (default: False)
    """

    def __init__(
        self,
        layers_hidden: Sequence[int],
        p_degree: int = 4,
        q_degree: int = 2,
        base_activation: Union[Callable[[mx.array], mx.array], type] = nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.p_degree = p_degree
        self.q_degree = q_degree

        self.layers = [
            RationalKANLinear(
                in_f,
                out_f,
                p_degree=p_degree,
                q_degree=q_degree,
                base_activation=base_activation,
                bias=bias,
            )
            for in_f, out_f in zip(layers_hidden[:-1], layers_hidden[1:])
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x

    forward = __call__
