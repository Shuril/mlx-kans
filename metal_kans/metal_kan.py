"""
Direct Metal Shading Language (MSL) KAN Implementation for Apple Silicon GPU.
Fuses basis function evaluation and matrix multiplication into a single GPU compute pass
with zero intermediate VRAM allocation and register blocking.
"""

import os
import ctypes
import numpy as np
from typing import Optional, Union, Tuple

# Locate and load libmetal_kan.dylib
_DIR = os.path.dirname(os.path.abspath(__file__))
_DYLIB_PATH = os.path.join(_DIR, "libmetal_kan.dylib")
_SHADER_PATH = os.path.join(_DIR, "fused_kan.metal")
_MM_PATH = os.path.join(_DIR, "metal_kan_bridge.mm")

_lib = None
if not os.path.exists(_DYLIB_PATH) and os.path.exists(_MM_PATH):
    import subprocess
    try:
        cmd = [
            "clang++", "-O3", "-dynamiclib",
            "-framework", "Metal", "-framework", "Foundation",
            _MM_PATH, "-o", _DYLIB_PATH
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:
        pass

if os.path.exists(_DYLIB_PATH):
    _lib = ctypes.CDLL(_DYLIB_PATH)
    # Set argtypes and restype
    _lib.metal_kan_init.argtypes = [ctypes.c_char_p]
    _lib.metal_kan_init.restype = ctypes.c_int

    _lib.metal_kan_cheby_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _lib.metal_kan_cheby_forward.restype = ctypes.c_int

    _lib.metal_kan_fastkan_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    _lib.metal_kan_fastkan_forward.restype = ctypes.c_int

    _lib.metal_kan_relu_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    _lib.metal_kan_relu_forward.restype = ctypes.c_int

    _lib.benchmark_metal_cheby.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    _lib.benchmark_metal_cheby.restype = ctypes.c_double

    # Initialize Metal shaders
    res = _lib.metal_kan_init(_SHADER_PATH.encode("utf-8"))
    if res != 0:
        raise RuntimeError(f"Failed to initialize Metal KAN shaders (code: {res})")


class MetalChebyKAN:
    """
    Direct Metal Fused ChebyKAN Layer.
    Executes Chebyshev polynomials in GPU registers, fusing basis and GEMM into 1 pass.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4, bias: bool = True, use_base: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        # Initialize weights with Kaiming uniform
        bound = 1.0 / np.sqrt(in_features)
        self.w_cheby = np.random.uniform(-bound, bound, (out_features, in_features * degree)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x, dtype=np.float32)
        B, D_in = x.shape
        assert D_in == self.in_features, f"Expected input feature dim {self.in_features}, got {D_in}"

        y = np.empty((B, self.out_features), dtype=np.float32)

        _lib.metal_kan_cheby_forward(
            x.ctypes.data,
            self.w_cheby.ctypes.data,
            self.w_base.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.degree,
            self.has_base, self.has_bias
        )
        return y

    __call__ = forward

    def benchmark(self, x: np.ndarray, warmup: int = 10, iters: int = 50) -> float:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x, dtype=np.float32)
        B, D_in = x.shape
        y = np.empty((B, self.out_features), dtype=np.float32)

        return _lib.benchmark_metal_cheby(
            x.ctypes.data,
            self.w_cheby.ctypes.data,
            self.w_base.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.degree,
            self.has_base, self.has_bias,
            warmup, iters
        )


class MetalFastKAN:
    """
    Direct Metal Fused FastKAN Layer (Gaussian RBF).
    Fuses RBF basis evaluation and linear projection into a single GPU compute pass.
    """
    def __init__(self, in_features: int, out_features: int, num_centers: int = 8, grid_range: Tuple[float, float] = (-1.0, 1.0), bias: bool = True, use_base: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        self.grid = np.linspace(grid_range[0], grid_range[1], num_centers, dtype=np.float32)
        h = (grid_range[1] - grid_range[0]) / (num_centers - 1)
        self.inv_denominator = float(1.0 / (2.0 * (h ** 2)))

        bound = 1.0 / np.sqrt(in_features)
        self.w_rbf = np.random.uniform(-bound, bound, (out_features, in_features * num_centers)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x, dtype=np.float32)
        B, D_in = x.shape
        y = np.empty((B, self.out_features), dtype=np.float32)

        _lib.metal_kan_fastkan_forward(
            x.ctypes.data,
            self.w_rbf.ctypes.data,
            self.w_base.ctypes.data,
            self.grid.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.num_centers, self.inv_denominator,
            self.has_base, self.has_bias
        )
        return y

    __call__ = forward


class MetalReLUKAN:
    """
    Direct Metal Fused ReLUKAN Layer (Piecewise Linear Tent).
    Zero transcendental operations, fused piecewise-linear basis and GEMM.
    """
    def __init__(self, in_features: int, out_features: int, num_grids: int = 8, grid_range: Tuple[float, float] = (-1.0, 1.0), bias: bool = True, use_base: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        self.grid = np.linspace(grid_range[0], grid_range[1], num_grids, dtype=np.float32)
        h = (grid_range[1] - grid_range[0]) / (num_grids - 1)
        self.inv_h = float(1.0 / h)

        bound = 1.0 / np.sqrt(in_features)
        self.w_relu = np.random.uniform(-bound, bound, (out_features, in_features * num_grids)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x, dtype=np.float32)
        B, D_in = x.shape
        y = np.empty((B, self.out_features), dtype=np.float32)

        _lib.metal_kan_relu_forward(
            x.ctypes.data,
            self.w_relu.ctypes.data,
            self.w_base.ctypes.data,
            self.grid.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.num_grids, self.inv_h,
            self.has_base, self.has_bias
        )
        return y

    __call__ = forward


class MetalKAN:
    """
    Multi-layer Pure Metal KAN Network.
    Executes sequentially on Apple Silicon GPU without PyTorch or MLX dependencies.
    """
    def __init__(self, layers_hidden: list[int], basis_type: str = "cheby", degree: int = 4, bias: bool = True):
        self.layers_hidden = list(layers_hidden)
        self.basis_type = basis_type.lower()
        self.layers = []
        for i in range(len(layers_hidden) - 1):
            in_f = layers_hidden[i]
            out_f = layers_hidden[i + 1]
            if self.basis_type == "cheby":
                layer = MetalChebyKAN(in_f, out_f, degree=degree, bias=bias)
            elif self.basis_type in ("fastkan", "rbf"):
                layer = MetalFastKAN(in_f, out_f, num_centers=degree, bias=bias)
            elif self.basis_type in ("relukan", "relu"):
                layer = MetalReLUKAN(in_f, out_f, num_grids=degree, bias=bias)
            else:
                raise ValueError(f"Unknown basis_type: {basis_type}")
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    __call__ = forward


