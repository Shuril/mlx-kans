"""
Native INT8, INT4 & FP8 Quantization for Kolmogorov-Arnold Networks (KAN) in Apple MLX.

Leverages Apple Silicon Metal GPU hardware acceleration via `mx.quantize` and `mx.quantized_matmul`.
Provides:
  - INT8 / INT4: Native hardware execution on all Apple Silicon chips (M1, M2, M3, M4, M5+).
  - FP8 (mxfp8 / E4M3): Native hardware tensor acceleration on Apple Silicon M4/M5+ (Apple GPU Family 9+).
    On earlier chips (M1/M2/M3), raises HardwareNotSupportedError unless `allow_emulation=True`.
"""

from __future__ import annotations

import math
import re
import subprocess
from typing import Callable, Dict, Sequence, Union, Optional

import mlx.core as mx
import mlx.nn as nn

from kan import compute_b_splines
from .fast_kan import compute_rbf_basis
from .relu_kan import compute_relu_tent_basis
from .cheby_kan import compute_cheby_basis
from .wav_kan import compute_wavelet_basis
from .fourier_kan import compute_fourier_basis
from .jacobi_kan import compute_jacobi_basis
from .metal_kernels import (
    metal_rbf_basis,
    metal_cheby_basis,
    metal_relu_basis,
    is_metal_available,
)


class HardwareNotSupportedError(RuntimeError):
    """
    Raised when an operation requires hardware features not physically supported
    by the current Apple Silicon GPU architecture (such as hardware FP8 tensor units on M1/M2/M3).
    """
    pass


def get_chip_name() -> str:
    """Return the marketing name of the current Apple Silicon chip (e.g. 'Apple M1', 'Apple M4 Pro')."""
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        name = info.get("device_name", "")
        if name and name != "Unknown":
            return name
    except Exception:
        pass
    try:
        brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        if brand:
            return brand
    except Exception:
        pass
    return "Apple Silicon"


def is_fp8_hardware_supported() -> bool:
    """
    Check if the current Apple Silicon GPU has native hardware support for FP8 (mxfp8 / E4M3).
    Hardware FP8 execution units were introduced in Apple Silicon with M4 / M5 (Apple GPU Family 9+).
    Earlier architectures (M1, M2, M3) do not have hardware FP8 execution units in the silicon.
    """
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        device_name = info.get("device_name", "")
        arch = info.get("architecture", "")

        # M1, M2, M3 do not have hardware FP8 execution units
        for gen in ["M1", "M2", "M3"]:
            if gen in device_name:
                return False

        # Architecture string check: g13 (M1), g14 (M2), g15 (M3)
        if any(g in arch for g in ["g13", "g14", "g15"]):
            return False

        # M4, M5, M6+ have hardware FP8 execution units
        if any(gen in device_name for gen in ["M4", "M5", "M6", "M7", "A18"]):
            return True

        match = re.search(r"Apple M(\d+)", device_name)
        if match:
            return int(match.group(1)) >= 4

        brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        match = re.search(r"Apple M(\d+)", brand)
        if match:
            return int(match.group(1)) >= 4

        return False
    except Exception:
        return False


class QuantizedWeight(nn.Module):
    """
    Quantized linear projection weight wrapper using Apple MLX native Metal GPU kernels.

    Packs floating-point weights into 32-bit words (`uint32`) containing:
      - 4 INT8 values (for `bits=8, mode="affine"`)
      - 8 INT4 values (for `bits=4, mode="affine"`)
      - 4 FP8 values (for `bits=8, mode="mxfp8"`)
    Executes matrix multiplication directly on Apple Silicon GPU without dequantization to FP32.

    Automatically handles non-standard input dimensions by zero-padding to the nearest
    multiple of `group_size` before quantization and inference.

    Args:
        weight: Unquantized floating-point weight matrix of shape (out_features, in_features).
        group_size: Number of weight elements per quantization group (default: 64, or 32 for mxfp8).
        bits: Bit-width per parameter (default: 8).
        mode: Quantization mode ("affine", "mxfp8", or "fp8").
        allow_emulation: If True, allows running FP8 on M1/M2/M3 via software emulation despite
            lack of hardware tensor acceleration.
    """

    def __init__(
        self,
        weight: mx.array,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        allow_emulation: bool = False,
    ):
        super().__init__()
        if mode in ["fp8", "mxfp8"]:
            mode = "mxfp8"
            if not is_fp8_hardware_supported() and not allow_emulation:
                chip = get_chip_name()
                raise HardwareNotSupportedError(
                    f"Hardware FP8 (mxfp8 / E4M3) quantization is not supported in hardware on {chip}. "
                    f"Apple Silicon chips prior to M4/M5 (M1, M2, M3) do not have native hardware FP8 tensor/ALU execution units in the GPU silicon. "
                    f"For native hardware-accelerated quantization on {chip}, use INT8 via `kans.to_int8(model)` instead. "
                    f"(To force software-emulated FP8 anyway, pass `allow_emulation=True`)."
                )
            if group_size is None or group_size != 32:
                group_size = 32

        out_features, in_features = weight.shape
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # If in_features is not divisible by group_size, pad columns to nearest multiple
        pad_len = (group_size - (in_features % group_size)) % group_size
        self.pad_len = pad_len
        if pad_len > 0:
            weight = mx.pad(weight, [(0, 0), (0, pad_len)])

        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size=group_size, bits=bits, mode=mode
        )
        self.biases = biases[0] if biases else None
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        """
        Compute Y = X @ W^T directly on Metal GPU using packed weights.

        Args:
            x: Input array of shape (*batch_dims, in_features).

        Returns:
            Output array of shape (*batch_dims, out_features).
        """
        if self.pad_len > 0:
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, self.pad_len)])

        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    @property
    def num_bytes(self) -> int:
        """Memory footprint of this quantized weight in bytes."""
        total = self["weight"].size * self["weight"].itemsize
        total += self["scales"].size * self["scales"].itemsize
        if "biases" in self and self["biases"] is not None:
            total += self["biases"].size * self["biases"].itemsize
        return total


# ---------------------------------------------------------------------------
# Quantized KAN Layers
# ---------------------------------------------------------------------------

class QuantizedKANLinear(nn.Module):
    """Quantized B-Spline KAN Layer (Cox-de Boor)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        spline_order: int,
        grid: mx.array,
        quant_base: QuantizedWeight,
        quant_spline: QuantizedWeight,
        base_activation: Callable[[mx.array], mx.array] = nn.silu,
        bias: Optional[mx.array] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid = grid
        self.freeze(keys=["grid"])
        self.quant_base = quant_base
        self.quant_spline = quant_spline
        self.base_activation = base_activation
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Base linear branch
        base_out = self.quant_base(self.base_activation(x_flat))

        # Spline basis evaluation + quantized projection
        bases = compute_b_splines(x_flat, self.grid, self.spline_order)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        spline_out = self.quant_spline(bases_flat)

        out = base_out + spline_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_layer(
        cls,
        layer,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ) -> QuantizedKANLinear:
        quant_base = QuantizedWeight(layer.base_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        spline_w = layer.scaled_spline_weight.reshape(layer.out_features, -1)
        quant_spline = QuantizedWeight(spline_w, group_size=group_size, bits=bits, mode=mode, **kwargs)
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            grid_size=layer.grid_size,
            spline_order=layer.spline_order,
            grid=layer.grid,
            quant_base=quant_base,
            quant_spline=quant_spline,
            base_activation=layer.base_activation,
            bias=layer.bias,
        )


class QuantizedFastKANLinear(nn.Module):
    """Quantized FastKAN Layer (Gaussian Radial Basis Functions)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grids: int,
        centers: mx.array,
        inv_h: float,
        quant_base: QuantizedWeight,
        quant_spline: QuantizedWeight,
        base_activation: Callable[[mx.array], mx.array] = nn.silu,
        use_metal_kernel: bool = False,
        bias: Optional[mx.array] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        self.centers = centers
        self.inv_h = inv_h
        self.freeze(keys=["centers"])
        self.quant_base = quant_base
        self.quant_spline = quant_spline
        self.base_activation = base_activation
        self.use_metal_kernel = use_metal_kernel and is_metal_available()
        self.bias = bias

    def rbf_basis(self, x: mx.array) -> mx.array:
        if self.use_metal_kernel and x.ndim == 2:
            return metal_rbf_basis(x, self.centers, self.inv_h)
        return compute_rbf_basis(x, self.centers, self.inv_h)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.quant_base(self.base_activation(x_flat))

        bases = self.rbf_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        spline_out = self.quant_spline(bases_flat)

        out = base_out + spline_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_layer(
        cls,
        layer,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ) -> QuantizedFastKANLinear:
        quant_base = QuantizedWeight(layer.base_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        quant_spline = QuantizedWeight(layer.spline_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            num_grids=layer.num_grids,
            centers=layer.centers,
            inv_h=layer.inv_h,
            quant_base=quant_base,
            quant_spline=quant_spline,
            base_activation=layer.base_activation,
            use_metal_kernel=getattr(layer, "use_metal_kernel", False),
            bias=layer.bias,
        )


class QuantizedReLUKANLinear(nn.Module):
    """Quantized ReLUKAN Layer (Piecewise Linear / Tent Basis)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grids: int,
        centers: mx.array,
        inv_h: float,
        quant_base: QuantizedWeight,
        quant_spline: QuantizedWeight,
        base_activation: Callable[[mx.array], mx.array] = nn.silu,
        use_metal_kernel: bool = False,
        bias: Optional[mx.array] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        self.centers = centers
        self.inv_h = inv_h
        self.freeze(keys=["centers"])
        self.quant_base = quant_base
        self.quant_spline = quant_spline
        self.base_activation = base_activation
        self.use_metal_kernel = use_metal_kernel and is_metal_available()
        self.bias = bias

    def relu_basis(self, x: mx.array) -> mx.array:
        if self.use_metal_kernel and x.ndim == 2:
            return metal_relu_basis(x, self.centers, self.inv_h)
        return compute_relu_tent_basis(x, self.centers, self.inv_h)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.quant_base(self.base_activation(x_flat))

        bases = self.relu_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        spline_out = self.quant_spline(bases_flat)

        out = base_out + spline_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_layer(
        cls,
        layer,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ) -> QuantizedReLUKANLinear:
        quant_base = QuantizedWeight(layer.base_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        quant_spline = QuantizedWeight(layer.spline_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            num_grids=layer.num_grids,
            centers=layer.centers,
            inv_h=layer.inv_h,
            quant_base=quant_base,
            quant_spline=quant_spline,
            base_activation=layer.base_activation,
            use_metal_kernel=getattr(layer, "use_metal_kernel", False),
            bias=layer.bias,
        )


class QuantizedChebyKANLinear(nn.Module):
    """Quantized ChebyKAN Layer (Orthogonal Chebyshev Polynomials)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int,
        quant_base: QuantizedWeight,
        quant_cheby: QuantizedWeight,
        base_activation: Callable[[mx.array], mx.array] = nn.silu,
        use_metal_kernel: bool = False,
        bias: Optional[mx.array] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.quant_base = quant_base
        self.quant_cheby = quant_cheby
        self.base_activation = base_activation
        self.use_metal_kernel = use_metal_kernel and is_metal_available()
        self.bias = bias

    def cheby_basis(self, x: mx.array) -> mx.array:
        if self.use_metal_kernel and x.ndim == 2:
            return metal_cheby_basis(x, self.degree)
        return compute_cheby_basis(x, self.degree)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.quant_base(self.base_activation(x_flat))

        bases = self.cheby_basis(x_flat)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        cheby_out = self.quant_cheby(bases_flat)

        out = base_out + cheby_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_layer(
        cls,
        layer,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ) -> QuantizedChebyKANLinear:
        quant_base = QuantizedWeight(layer.base_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        quant_cheby = QuantizedWeight(layer.cheby_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            degree=layer.degree,
            quant_base=quant_base,
            quant_cheby=quant_cheby,
            base_activation=layer.base_activation,
            use_metal_kernel=getattr(layer, "use_metal_kernel", False),
            bias=layer.bias,
        )


class QuantizedWavKANLinear(nn.Module):
    """Quantized WavKAN Layer (Continuous Wavelets)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        translation: mx.array,
        scale: mx.array,
        wavelet_type: str,
        quant_base: QuantizedWeight,
        quant_wav: QuantizedWeight,
        base_activation: Callable[[mx.array], mx.array] = nn.silu,
        bias: Optional[mx.array] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.translation = translation
        self.scale = scale
        self.wavelet_type = wavelet_type
        self.freeze(keys=["translation", "scale"])
        self.quant_base = quant_base
        self.quant_wav = quant_wav
        self.base_activation = base_activation
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.quant_base(self.base_activation(x_flat))

        bases = compute_wavelet_basis(x_flat, self.translation, self.scale, self.wavelet_type)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        wav_out = self.quant_wav(bases_flat)

        out = base_out + wav_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_layer(
        cls,
        layer,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ) -> QuantizedWavKANLinear:
        quant_base = QuantizedWeight(layer.base_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        quant_wav = QuantizedWeight(layer.wav_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            translation=layer.translation,
            scale=layer.scale,
            wavelet_type=layer.wavelet_type,
            quant_base=quant_base,
            quant_wav=quant_wav,
            base_activation=layer.base_activation,
            bias=layer.bias,
        )


class QuantizedFourierKANLinear(nn.Module):
    """Quantized FourierKAN Layer (Fourier Trigonometric Series)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        freqs: mx.array,
        quant_base: QuantizedWeight,
        quant_fourier: QuantizedWeight,
        base_activation: Callable[[mx.array], mx.array] = nn.silu,
        bias: Optional[mx.array] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.freqs = freqs
        self.freeze(keys=["freqs"])
        self.quant_base = quant_base
        self.quant_fourier = quant_fourier
        self.base_activation = base_activation
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.quant_base(self.base_activation(x_flat))

        bases = compute_fourier_basis(x_flat, self.freqs)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        fourier_out = self.quant_fourier(bases_flat)

        out = base_out + fourier_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_layer(
        cls,
        layer,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ) -> QuantizedFourierKANLinear:
        quant_base = QuantizedWeight(layer.base_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        quant_fourier = QuantizedWeight(layer.fourier_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            freqs=layer.freqs,
            quant_base=quant_base,
            quant_fourier=quant_fourier,
            base_activation=layer.base_activation,
            bias=layer.bias,
        )


class QuantizedJacobiKANLinear(nn.Module):
    """Quantized JacobiKAN Layer (Orthogonal Jacobi Polynomials)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int,
        alpha: float,
        beta: float,
        quant_base: QuantizedWeight,
        quant_jacobi: QuantizedWeight,
        base_activation: Callable[[mx.array], mx.array] = nn.silu,
        bias: Optional[mx.array] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        self.quant_base = quant_base
        self.quant_jacobi = quant_jacobi
        self.base_activation = base_activation
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.quant_base(self.base_activation(x_flat))

        bases = compute_jacobi_basis(x_flat, self.degree, self.alpha, self.beta)
        bases_flat = bases.reshape(x_flat.shape[0], -1)
        jacobi_out = self.quant_jacobi(bases_flat)

        out = base_out + jacobi_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_layer(
        cls,
        layer,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ) -> QuantizedJacobiKANLinear:
        quant_base = QuantizedWeight(layer.base_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        quant_jacobi = QuantizedWeight(layer.jacobi_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            degree=layer.degree,
            alpha=layer.alpha,
            beta=layer.beta,
            quant_base=quant_base,
            quant_jacobi=quant_jacobi,
            base_activation=layer.base_activation,
            bias=layer.bias,
        )


class QuantizedLowRankKANLinear(nn.Module):
    """Quantized LowRankKAN Layer (LoRA / Bottleneck KAN)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
        centers: mx.array,
        inv_h: float,
        quant_base: QuantizedWeight,
        quant_V: QuantizedWeight,
        quant_U: QuantizedWeight,
        base_activation: Callable[[mx.array], mx.array] = nn.silu,
        bias: Optional[mx.array] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling
        self.centers = centers
        self.inv_h = inv_h
        self.freeze(keys=["centers"])
        self.quant_base = quant_base
        self.quant_V = quant_V
        self.quant_U = quant_U
        self.base_activation = base_activation
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        base_out = self.quant_base(self.base_activation(x_flat))

        bases = compute_rbf_basis(x_flat, self.centers, self.inv_h)
        bases_flat = bases.reshape(x_flat.shape[0], -1)

        bottleneck = self.quant_V(bases_flat)
        spline_out = self.quant_U(bottleneck) * self.scaling

        out = base_out + spline_out
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_layer(
        cls,
        layer,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ) -> QuantizedLowRankKANLinear:
        quant_base = QuantizedWeight(layer.base_weight, group_size=group_size, bits=bits, mode=mode, **kwargs)
        quant_V = QuantizedWeight(layer.spline_V, group_size=group_size, bits=bits, mode=mode, **kwargs)
        quant_U = QuantizedWeight(layer.spline_U, group_size=group_size, bits=bits, mode=mode, **kwargs)
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            rank=layer.rank,
            scaling=layer.scaling,
            centers=layer.centers,
            inv_h=layer.inv_h,
            quant_base=quant_base,
            quant_V=quant_V,
            quant_U=quant_U,
            base_activation=layer.base_activation,
            bias=layer.bias,
        )


class QuantizedMultKANLinear(nn.Module):
    """Quantized MultKAN (KAN 2.0) Layer with multiplication nodes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_add: int,
        num_mult: int,
        sub_layer: nn.Module,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_add = num_add
        self.num_mult = num_mult
        self.sub_layer = sub_layer

    def __call__(self, x: mx.array) -> mx.array:
        z = self.sub_layer(x)
        if self.num_mult == 0:
            return z

        if self.num_add > 0:
            add_part = z[..., : self.num_add]
            mult_raw = z[..., self.num_add :]
        else:
            add_part = None
            mult_raw = z

        orig_shape = mult_raw.shape
        mult_pairs = mult_raw.reshape(*orig_shape[:-1], self.num_mult, 2)
        mult_part = mult_pairs[..., 0] * mult_pairs[..., 1]

        if add_part is not None:
            return mx.concatenate([add_part, mult_part], axis=-1)
        return mult_part

    @classmethod
    def from_layer(
        cls,
        layer,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        **kwargs,
    ) -> QuantizedMultKANLinear:
        quant_sub = QuantizedFastKANLinear.from_layer(
            layer.sub_layer, group_size=group_size, bits=bits, mode=mode, **kwargs
        )
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            num_add=layer.num_add,
            num_mult=layer.num_mult,
            sub_layer=quant_sub,
        )


# ---------------------------------------------------------------------------
# High-Level Quantization API & Utilities
# ---------------------------------------------------------------------------

def quantize(
    model: nn.Module,
    group_size: int = 64,
    bits: int = 8,
    mode: str = "affine",
    allow_emulation: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Quantize all KAN layers (and standard MLX layers) in a model to INT8, INT4, or FP8.

    Args:
        model: MLX model (e.g. KAN, FastKAN, or custom nn.Module containing KAN layers).
        group_size: Quantization group size (32, 64, or 128; must be 32 for mxfp8).
        bits: Bit-width per parameter (8 or 4).
        mode: Quantization mode ("affine", "mxfp8", or "fp8").
        allow_emulation: If True, allows running FP8 on M1/M2/M3 via software emulation despite
            lack of hardware tensor acceleration.

    Returns:
        The quantized model (updated in-place).
    """
    if mode in ["fp8", "mxfp8"]:
        mode = "mxfp8"
        if not is_fp8_hardware_supported() and not allow_emulation:
            chip = get_chip_name()
            raise HardwareNotSupportedError(
                f"Hardware FP8 (mxfp8 / E4M3) quantization is not supported in hardware on {chip}. "
                f"Apple Silicon chips prior to M4/M5 (M1, M2, M3) do not have native hardware FP8 tensor/ALU execution units in the GPU silicon. "
                f"For native hardware-accelerated quantization on {chip}, use INT8 via `kans.to_int8(model)` instead. "
                f"(To force software-emulated FP8 anyway, pass `allow_emulation=True`)."
            )
        if group_size is None or group_size != 32:
            group_size = 32

    def predicate(path, m):
        if hasattr(m, "to_quantized"):
            params = {
                "group_size": group_size,
                "bits": bits,
                "mode": mode,
                "allow_emulation": allow_emulation,
            }
            params.update(kwargs)
            return params
        return False

    nn.quantize(model, class_predicate=predicate)
    return model


def to_int8(
    model: nn.Module,
    group_size: int = 64,
    mode: str = "affine",
    **kwargs,
) -> nn.Module:
    """
    Quantize model weights to 8-bit integers (INT8) natively on Apple Silicon Metal GPU.
    Supported on all Apple Silicon generations (M1, M2, M3, M4, M5+).
    Achieves ~3.5x memory reduction with negligible accuracy loss.
    """
    return quantize(model, group_size=group_size, bits=8, mode=mode, **kwargs)


def to_int4(
    model: nn.Module,
    group_size: int = 64,
    mode: str = "affine",
    **kwargs,
) -> nn.Module:
    """
    Quantize model weights to 4-bit integers (INT4) natively on Apple Silicon Metal GPU.
    Supported on all Apple Silicon generations (M1, M2, M3, M4, M5+).
    Achieves ~6.5x memory reduction for memory-constrained Apple Silicon devices.
    """
    return quantize(model, group_size=group_size, bits=4, mode=mode, **kwargs)


def to_fp8(
    model: nn.Module,
    group_size: int = 32,
    allow_emulation: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Quantize model weights to 8-bit floating point (FP8 / mxfp8, E4M3) on Apple Silicon GPU.

    Note:
        Native hardware FP8 tensor acceleration requires Apple Silicon M4 / M5 or newer
        (Apple GPU Family 9+). On earlier architectures (M1, M2, M3), hardware FP8 execution
        units are not present in the silicon, and this function will raise `HardwareNotSupportedError`
        unless `allow_emulation=True` is explicitly passed.

    Args:
        model: MLX model (e.g. KAN, FastKAN, etc.)
        group_size: Quantization group size (must be 32 for mxfp8, default: 32).
        allow_emulation: If True, allows running on M1/M2/M3 via software emulation despite
            lack of hardware tensor acceleration.
    """
    if not is_fp8_hardware_supported() and not allow_emulation:
        chip = get_chip_name()
        raise HardwareNotSupportedError(
            f"Hardware FP8 (mxfp8 / E4M3) quantization is not supported in hardware on {chip}. "
            f"Apple Silicon chips prior to M4/M5 (M1, M2, M3) do not have native hardware FP8 tensor/ALU execution units in the GPU silicon. "
            f"For native hardware-accelerated quantization on {chip}, use INT8 via `kans.to_int8(model)` instead. "
            f"(To force software-emulated FP8 anyway, pass `allow_emulation=True`)."
        )
    return quantize(model, group_size=group_size, bits=8, mode="mxfp8", allow_emulation=allow_emulation, **kwargs)


def get_model_size(model: nn.Module) -> Dict[str, Union[int, float, str]]:
    """
    Calculate total parameter count and memory consumption in bytes/megabytes
    for any MLX model (FP32, FP16, or quantized INT8/INT4/FP8).

    Returns:
        dict with keys: 'total_params', 'total_bytes', 'mb', 'summary'.
    """
    def _leaves(tree):
        if isinstance(tree, dict):
            for v in tree.values():
                yield from _leaves(v)
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                yield from _leaves(v)
        elif isinstance(tree, mx.array):
            yield tree

    total_bytes = 0
    total_params = 0

    for arr in _leaves(model.parameters()):
        total_bytes += arr.size * arr.itemsize
        total_params += arr.size

    mb = total_bytes / (1024.0 * 1024.0)
    summary = f"{mb:.3f} MB ({total_bytes:,} bytes, {total_params:,} elements)"

    return {
        "total_params": total_params,
        "total_bytes": total_bytes,
        "mb": mb,
        "summary": summary,
    }
