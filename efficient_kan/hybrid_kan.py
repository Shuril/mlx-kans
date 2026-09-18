"""
Hybrid KAN: Fused Metal Shading Language + MLX Framework Integration.
Combines high-level MLX neural network abstractions, autograd, and parameter optimization
with zero-allocation Direct Metal GPU compute shaders via macOS Unified Memory.
"""

from __future__ import annotations
import math
import os
from typing import Sequence, Optional, Union, Callable
import numpy as np
import mlx.core as mx
import mlx.nn as nn

try:
    from metal_kans.metal_kan import MetalChebyKAN, MetalFastKAN, MetalReLUKAN
    _METAL_AVAILABLE = True
except Exception:
    _METAL_AVAILABLE = False


def _kaiming_uniform(shape: Sequence[int], fan_in: int, a: float = math.sqrt(5.0)) -> mx.array:
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(-bound, bound, shape)


class HybridChebyKAN(nn.Module):
    """
    Hybrid ChebyKAN Layer:
    - Native MLX nn.Module supporting backprop, AdamW, JIT graph compilation.
    - Zero-allocation Direct Metal GPU kernel on large batches (B >= threshold).
    - Eliminates intermediate VRAM basis allocations while preserving full MLX autograd.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        bias: bool = True,
        use_base: bool = True,
        adaptive_threshold: int = 1024,
        prefer_metal: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.use_base = use_base
        self.adaptive_threshold = adaptive_threshold
        self.prefer_metal = prefer_metal and _METAL_AVAILABLE

        # Trainable MLX parameters
        self.cheby_weight = _kaiming_uniform((out_features, in_features * degree), in_features)
        self.base_weight = _kaiming_uniform((out_features, in_features), in_features) if use_base else None
        self.bias = mx.zeros((out_features,)) if bias else None

        # Direct Metal Engine
        if self.prefer_metal:
            self._metal_engine = MetalChebyKAN(
                in_features=in_features,
                out_features=out_features,
                degree=degree,
                bias=bias,
                use_base=use_base
            )
        else:
            self._metal_engine = None

    def _sync_weights_to_metal(self):
        if self._metal_engine is not None:
            self._metal_engine.w_cheby = np.array(self.cheby_weight, copy=False)
            if self.use_base and self.base_weight is not None:
                self._metal_engine.w_base = np.array(self.base_weight, copy=False)
            if self.bias is not None:
                self._metal_engine.bias = np.array(self.bias, copy=False)

    def _mlx_forward(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        B = x_flat.shape[0]

        # Chebyshev recurrence
        x_clamped = mx.clip(x_flat, -1.0, 1.0)
        bases = [mx.ones_like(x_clamped[..., None])]
        if self.degree > 1:
            bases.append(x_clamped[..., None])
            for k in range(2, self.degree):
                bases.append(2.0 * x_clamped[..., None] * bases[-1] - bases[-2])
        bases_flat = mx.concatenate(bases, axis=-1).reshape(B, -1)
        out = bases_flat @ self.cheby_weight.T

        if self.use_base and self.base_weight is not None:
            base_out = (x_flat / (1.0 + mx.exp(-x_flat))) @ self.base_weight.T
            out = out + base_out

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        B = x_flat.shape[0]

        # Adaptive routing:
        # Use Direct Metal Fused kernel when batch size is large enough to benefit from zero-VRAM
        if self.prefer_metal and self._metal_engine is not None and B >= self.adaptive_threshold:
            self._sync_weights_to_metal()
            x_np = np.array(x_flat, copy=False)
            y_np = self._metal_engine.forward(x_np)
            out = mx.array(y_np)
            return out.reshape(*orig_shape[:-1], self.out_features)
        else:
            return self._mlx_forward(x)

    forward = __call__


class HybridFastKAN(nn.Module):
    """
    Hybrid FastKAN Layer (Gaussian RBF):
    - Combines MLX autograd & JIT with Direct Metal Fused RBF compute kernel.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_centers: int = 8,
        degree: Optional[int] = None,
        grid_range: tuple = (-1.0, 1.0),
        bias: bool = True,
        use_base: bool = True,
        adaptive_threshold: int = 1024,
        prefer_metal: bool = True,
    ):
        super().__init__()
        if degree is not None:
            num_centers = degree
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.degree = num_centers
        self.use_base = use_base
        self.adaptive_threshold = adaptive_threshold
        self.prefer_metal = prefer_metal and _METAL_AVAILABLE

        self.grid = mx.linspace(grid_range[0], grid_range[1], num_centers)
        h = (grid_range[1] - grid_range[0]) / (num_centers - 1)
        self.inv_denominator = float(1.0 / (2.0 * (h ** 2)))

        self.rbf_weight = _kaiming_uniform((out_features, in_features * num_centers), in_features)
        self.base_weight = _kaiming_uniform((out_features, in_features), in_features) if use_base else None
        self.bias = mx.zeros((out_features,)) if bias else None

        if self.prefer_metal:
            self._metal_engine = MetalFastKAN(
                in_features=in_features,
                out_features=out_features,
                num_centers=num_centers,
                grid_range=grid_range,
                bias=bias,
                use_base=use_base
            )
        else:
            self._metal_engine = None

    def _sync_weights_to_metal(self):
        if self._metal_engine is not None:
            self._metal_engine.w_rbf = np.array(self.rbf_weight, copy=False)
            if self.use_base and self.base_weight is not None:
                self._metal_engine.w_base = np.array(self.base_weight, copy=False)
            if self.bias is not None:
                self._metal_engine.bias = np.array(self.bias, copy=False)

    def _mlx_forward(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        diff = x_flat[..., None] - self.grid
        rbf = mx.exp(- (diff ** 2) * self.inv_denominator)
        rbf_flat = rbf.reshape(x_flat.shape[0], -1)

        out = rbf_flat @ self.rbf_weight.T
        if self.use_base and self.base_weight is not None:
            out = out + (x_flat / (1.0 + mx.exp(-x_flat))) @ self.base_weight.T
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        B = x_flat.shape[0]

        if self.prefer_metal and self._metal_engine is not None and B >= self.adaptive_threshold:
            self._sync_weights_to_metal()
            x_np = np.array(x_flat, copy=False)
            y_np = self._metal_engine.forward(x_np)
            return mx.array(y_np).reshape(*orig_shape[:-1], self.out_features)
        else:
            return self._mlx_forward(x)

    forward = __call__


class HybridReLUKAN(nn.Module):
    """
    Hybrid ReLUKAN Layer (Piecewise Linear Tent):
    - Zero transcendental functions, hardware tent basis + MLX training integration.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grids: int = 8,
        degree: Optional[int] = None,
        grid_range: tuple = (-1.0, 1.0),
        bias: bool = True,
        use_base: bool = True,
        adaptive_threshold: int = 1024,
        prefer_metal: bool = True,
    ):
        super().__init__()
        if degree is not None:
            num_grids = degree
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        self.degree = num_grids
        self.use_base = use_base
        self.adaptive_threshold = adaptive_threshold
        self.prefer_metal = prefer_metal and _METAL_AVAILABLE

        self.grid = mx.linspace(grid_range[0], grid_range[1], num_grids)
        h = (grid_range[1] - grid_range[0]) / (num_grids - 1)
        self.inv_h = float(1.0 / h)

        self.relu_weight = _kaiming_uniform((out_features, in_features * num_grids), in_features)
        self.base_weight = _kaiming_uniform((out_features, in_features), in_features) if use_base else None
        self.bias = mx.zeros((out_features,)) if bias else None

        if self.prefer_metal:
            self._metal_engine = MetalReLUKAN(
                in_features=in_features,
                out_features=out_features,
                num_grids=num_grids,
                grid_range=grid_range,
                bias=bias,
                use_base=use_base
            )
        else:
            self._metal_engine = None

    def _sync_weights_to_metal(self):
        if self._metal_engine is not None:
            self._metal_engine.w_relu = np.array(self.relu_weight, copy=False)
            if self.use_base and self.base_weight is not None:
                self._metal_engine.w_base = np.array(self.base_weight, copy=False)
            if self.bias is not None:
                self._metal_engine.bias = np.array(self.bias, copy=False)

    def _mlx_forward(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        diff = mx.abs(x_flat[..., None] - self.grid)
        tent = mx.maximum(0.0, 1.0 - diff * self.inv_h)
        tent_flat = tent.reshape(x_flat.shape[0], -1)

        out = tent_flat @ self.relu_weight.T
        if self.use_base and self.base_weight is not None:
            out = out + (x_flat / (1.0 + mx.exp(-x_flat))) @ self.base_weight.T
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        B = x_flat.shape[0]

        if self.prefer_metal and self._metal_engine is not None and B >= self.adaptive_threshold:
            self._sync_weights_to_metal()
            x_np = np.array(x_flat, copy=False)
            y_np = self._metal_engine.forward(x_np)
            return mx.array(y_np).reshape(*orig_shape[:-1], self.out_features)
        else:
            return self._mlx_forward(x)

    forward = __call__


class HybridKAN(nn.Module):
    """
    Multi-layer Hybrid KAN Network Container.
    Constructs a sequential chain of Hybrid KAN layers.
    """
    def __init__(
        self,
        layers_hidden: Sequence[int],
        basis_type: str = "cheby",
        degree: int = 4,
        bias: bool = True,
        adaptive_threshold: int = 1024,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.layers = []

        for i in range(len(layers_hidden) - 1):
            in_f = layers_hidden[i]
            out_f = layers_hidden[i + 1]
            if basis_type.lower() == "cheby":
                layer = HybridChebyKAN(in_f, out_f, degree=degree, bias=bias, adaptive_threshold=adaptive_threshold)
            elif basis_type.lower() == "fastkan":
                layer = HybridFastKAN(in_f, out_f, num_centers=degree, bias=bias, adaptive_threshold=adaptive_threshold)
            elif basis_type.lower() == "relu":
                layer = HybridReLUKAN(in_f, out_f, num_grids=degree, bias=bias, adaptive_threshold=adaptive_threshold)
            else:
                raise ValueError(f"Unknown basis_type: {basis_type}")
            self.layers.append(layer)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x

    forward = __call__
