"""
Convolutional Kolmogorov-Arnold Networks (ConvKAN) for Apple Silicon MLX.
Implements 1D and 2D Convolutional KAN layers with high-performance patch unfolding
and Metal GPU acceleration across multiple non-linear polynomial, RBF, and spline bases.
"""

from typing import Union, Tuple, Optional, Any
import mlx.core as mx
import mlx.nn as nn

from .fast_kan import FastKANLinear
from .cheby_kan import ChebyKANLinear
from .relu_kan import ReLUKANLinear
from .wav_kan import WavKANLinear
from .fourier_kan import FourierKANLinear
from .jacobi_kan import JacobiKANLinear
from .rational_kan import RationalKANLinear
try:
    from kan import KANLinear
except ImportError:
    KANLinear = None


def _to_2d_tuple(val: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(val, (int, float)):
        return (int(val), int(val))
    return (int(val[0]), int(val[1]))


def extract_patches_1d(
    x: mx.array,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0
) -> mx.array:
    """
    Extracts 1D sliding patches from input tensor of shape (B, L, C).
    Returns tensor of shape (B, L_out, C * kernel_size).
    """
    if padding > 0:
        x = mx.pad(x, [(0, 0), (padding, padding), (0, 0)])

    B, L, C = x.shape
    if L < kernel_size:
        raise ValueError(f"Input sequence length {L} smaller than kernel_size {kernel_size}")

    out_l = (L - kernel_size) // stride + 1
    slices = [x[:, di : di + out_l * stride : stride, :] for di in range(kernel_size)]
    return mx.concatenate(slices, axis=-1)


def extract_patches_2d(
    x: mx.array,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0)
) -> mx.array:
    """
    Extracts 2D sliding image patches from input tensor of shape (B, H, W, C).
    Returns tensor of shape (B, H_out, W_out, C * K_h * K_w).
    """
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding

    if ph > 0 or pw > 0:
        x = mx.pad(x, [(0, 0), (ph, ph), (pw, pw), (0, 0)])

    B, H, W, C = x.shape
    if H < kh or W < kw:
        raise ValueError(f"Spatial dimensions ({H}, {W}) smaller than kernel size ({kh}, {kw})")

    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1

    slices = []
    for di in range(kh):
        for dj in range(kw):
            p = x[:, di : di + out_h * sh : sh, dj : dj + out_w * sw : sw, :]
            slices.append(p)

    return mx.concatenate(slices, axis=-1)


def _create_kan_layer(
    basis: str,
    in_features: int,
    out_features: int,
    degree: int,
    bias: bool
) -> nn.Module:
    b = basis.lower()
    if b in ("fastkan", "rbf"):
        return FastKANLinear(in_features, out_features, num_grids=degree, bias=bias)
    elif b == "cheby":
        return ChebyKANLinear(in_features, out_features, degree=degree, bias=bias)
    elif b in ("relu", "relukan"):
        return ReLUKANLinear(in_features, out_features, num_grids=degree, bias=bias)
    elif b in ("bspline", "kan"):
        if KANLinear is None:
            raise ImportError("kan.KANLinear is not available for BSpline basis.")
        return KANLinear(in_features, out_features, grid_size=degree, bias=bias)
    elif b in ("wav", "wavelet"):
        return WavKANLinear(in_features, out_features, num_wavelets=degree, bias=bias)
    elif b == "fourier":
        return FourierKANLinear(in_features, out_features, num_frequencies=degree, bias=bias)
    elif b == "jacobi":
        return JacobiKANLinear(in_features, out_features, degree=degree, bias=bias)
    elif b == "rational":
        return RationalKANLinear(in_features, out_features, p_degree=degree, bias=bias)
    else:
        raise ValueError(f"Unsupported KAN basis for convolution: {basis}")


class ConvKAN1d(nn.Module):
    """
    1D Convolutional Kolmogorov-Arnold Network Layer for sequential data (B, L, C).

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel length
        stride: Stride length
        padding: Zero-padding added to both sides
        basis: Basis type ("fastkan", "cheby", "bspline", "relu", "wav", "fourier", "jacobi", "rational")
        degree: Degree or grid size of the basis
        bias: Whether to include bias term
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        basis: str = "fastkan",
        degree: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.basis = basis
        self.degree = degree
        self.use_bias = bias

        in_features = in_channels * kernel_size
        self.kan = _create_kan_layer(basis, in_features, out_channels, degree, bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Input shape: (B, L, C)
        Output shape: (B, L_out, C_out)
        """
        patches = extract_patches_1d(x, self.kernel_size, self.stride, self.padding)
        return self.kan(patches)


Conv1dKAN = ConvKAN1d


class ConvKAN2d(nn.Module):
    """
    2D Convolutional Kolmogorov-Arnold Network Layer for spatial data (B, H, W, C).

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel dimensions (int or (H, W))
        stride: Stride dimensions (int or (H, W))
        padding: Padding dimensions (int or (H, W))
        basis: Basis type ("fastkan", "cheby", "bspline", "relu", "wav", "fourier", "jacobi", "rational")
        degree: Degree or grid size of the basis
        bias: Whether to include bias term
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        basis: str = "fastkan",
        degree: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_2d_tuple(kernel_size)
        self.stride = _to_2d_tuple(stride)
        self.padding = _to_2d_tuple(padding)
        self.basis = basis
        self.degree = degree
        self.use_bias = bias

        kh, kw = self.kernel_size
        in_features = in_channels * kh * kw
        self.kan = _create_kan_layer(basis, in_features, out_channels, degree, bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Input shape: (B, H, W, C)
        Output shape: (B, H_out, W_out, C_out)
        """
        patches = extract_patches_2d(x, self.kernel_size, self.stride, self.padding)
        return self.kan(patches)


Conv2dKAN = ConvKAN2d


class FastConv2dKAN(ConvKAN2d):
    """Gaussian RBF Convolutional KAN 2D Layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]] = 3, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, num_grids: int = 8, num_centers: Optional[int] = None, bias: bool = True):
        degree = num_centers if num_centers is not None else num_grids
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, basis="fastkan", degree=degree, bias=bias)


class ChebyConv2dKAN(ConvKAN2d):
    """Chebyshev Polynomial Convolutional KAN 2D Layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]] = 3, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, degree: int = 4, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, basis="cheby", degree=degree, bias=bias)


class ReLUConv2dKAN(ConvKAN2d):
    """Piecewise Linear Tent ReLUKAN Convolutional 2D Layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]] = 3, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, num_grids: int = 8, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, basis="relu", degree=num_grids, bias=bias)
