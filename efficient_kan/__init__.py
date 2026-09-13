"""
Efficient-KAN Suite for Apple Silicon MLX.
A unified, high-performance Kolmogorov-Arnold Networks library optimized for Apple Silicon Metal GPU.
"""

from kan import KAN, KANLinear, compute_b_splines
from .fast_kan import FastKAN, FastKANLinear
from .relu_kan import ReLUKAN, ReLUKANLinear
from .cheby_kan import ChebyKAN, ChebyKANLinear
from .wav_kan import WavKAN, WavKANLinear
from .fourier_kan import FourierKAN, FourierKANLinear
from .jacobi_kan import JacobiKAN, JacobiKANLinear
from .mult_kan import MultKAN, MultKANLinear
from .low_rank_kan import LowRankKAN, LowRankKANLinear
from .utils import build_train_step, count_parameters, to_fp16, to_bf16
from .metal_kernels import (
    metal_rbf_basis,
    metal_cheby_basis,
    metal_relu_basis,
    is_metal_available,
)
from .quantized import (
    QuantizedWeight,
    QuantizedKANLinear,
    QuantizedFastKANLinear,
    QuantizedReLUKANLinear,
    QuantizedChebyKANLinear,
    QuantizedWavKANLinear,
    QuantizedFourierKANLinear,
    QuantizedJacobiKANLinear,
    QuantizedLowRankKANLinear,
    QuantizedMultKANLinear,
    quantize,
    to_int8,
    to_int4,
    to_fp8,
    get_model_size,
    is_fp8_hardware_supported,
    get_chip_name,
    HardwareNotSupportedError,
)

__all__ = [
    # Core B-Spline KAN
    "KAN",
    "KANLinear",
    "compute_b_splines",
    # FastKAN (Gaussian RBF)
    "FastKAN",
    "FastKANLinear",
    # ReLUKAN
    "ReLUKAN",
    "ReLUKANLinear",
    # ChebyKAN (Chebyshev polynomials)
    "ChebyKAN",
    "ChebyKANLinear",
    # Wav-KAN (Continuous Wavelets)
    "WavKAN",
    "WavKANLinear",
    # FourierKAN (Trigonometric series)
    "FourierKAN",
    "FourierKANLinear",
    # JacobiKAN (Orthogonal Jacobi polynomials)
    "JacobiKAN",
    "JacobiKANLinear",
    # MultKAN (KAN 2.0 with multiplication nodes)
    "MultKAN",
    "MultKANLinear",
    # LowRankKAN (LoRA / Bottleneck)
    "LowRankKAN",
    "LowRankKANLinear",
    # Utilities & Optimizations
    "build_train_step",
    "count_parameters",
    "to_fp16",
    "to_bf16",
    # Custom Metal Kernels
    "metal_rbf_basis",
    "metal_cheby_basis",
    "metal_relu_basis",
    "is_metal_available",
    # Native Metal INT8 & INT4 Quantization
    "QuantizedWeight",
    "QuantizedKANLinear",
    "QuantizedFastKANLinear",
    "QuantizedReLUKANLinear",
    "QuantizedChebyKANLinear",
    "QuantizedWavKANLinear",
    "QuantizedFourierKANLinear",
    "QuantizedJacobiKANLinear",
    "QuantizedLowRankKANLinear",
    "QuantizedMultKANLinear",
    "quantize",
    "to_int8",
    "to_int4",
    "to_fp8",
    "get_model_size",
    "is_fp8_hardware_supported",
    "get_chip_name",
    "HardwareNotSupportedError",
]
