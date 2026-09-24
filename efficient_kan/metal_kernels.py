"""
Custom Metal Shading Language (MSL) kernels for Apple Silicon acceleration in MLX.
Provides hardware-fused basis evaluations executing directly on the M-series GPU.
"""

from __future__ import annotations
from typing import Union, Any, Dict, Tuple
import mlx.core as mx
import mlx.core.fast as fast


def is_metal_available() -> bool:
    """Check whether Metal GPU device is available."""
    try:
        return mx.default_device().type == mx.DeviceType.gpu
    except Exception:
        return False


_SCALAR_CACHE: Dict[Tuple[Any, Any], mx.array] = {}


def _scalar(val: Union[int, float], dtype: Any) -> mx.array:
    """Caches scalar dimension and parameter arrays to avoid per-call allocation overhead."""
    key = (val, dtype)
    arr = _SCALAR_CACHE.get(key)
    if arr is None:
        arr = mx.array([val], dtype=dtype)
        _SCALAR_CACHE[key] = arr
    return arr


# Gaussian RBF basis kernel (FastKAN).
_RBF_SOURCE = """
    uint g = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint n = thread_position_in_grid.z;
    uint G = G_dim[0];
    uint D = D_dim[0];
    uint N = N_dim[0];
    if (n < N && d < D && g < G) {
        T val = x[n * D + d];
        T c = centers[d * G + g];
        T diff = (val - c) * inv_h[0];
        out[(n * D + d) * G + g] = metal::fast::exp2(T(-1.4426950408889634) * diff * diff);
    }
"""

_rbf_metal_kernel = fast.metal_kernel(
    name="fastkan_rbf_basis",
    input_names=["x", "centers", "inv_h", "N_dim", "D_dim", "G_dim"],
    output_names=["out"],
    source=_RBF_SOURCE,
)


def metal_rbf_basis(x: mx.array, centers: mx.array, inv_h: float) -> mx.array:
    """
    Compute Gaussian RBF basis functions using custom Metal GPU kernel.

    Args:
        x: (*batch_dims, D) input array
        centers: (D, G) grid centers
        inv_h: 1.0 / grid_step

    Returns:
        (*batch_dims, D, G) basis evaluation array
    """
    orig_shape = x.shape
    if len(orig_shape) != 2:
        x = x.reshape(-1, orig_shape[-1])

    N, D = x.shape
    G = centers.shape[1]

    inv_h_arr = _scalar(inv_h, x.dtype)
    N_dim = _scalar(N, mx.uint32)
    D_dim = _scalar(D, mx.uint32)
    G_dim = _scalar(G, mx.uint32)

    outputs = _rbf_metal_kernel(
        inputs=[x, centers, inv_h_arr, N_dim, D_dim, G_dim],
        template=[("T", x.dtype)],
        grid=(G, D, N),
        threadgroup=(min(G, 32), min(D, 8), 1),
        output_shapes=[(N, D, G)],
        output_dtypes=[x.dtype],
    )
    res = outputs[0]
    if len(orig_shape) != 2:
        res = res.reshape(*orig_shape[:-1], D, G)
    return res


# Chebyshev polynomials kernel (ChebyKAN).
_CHEBY_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint total = N_dim[0] * D_dim[0];
    uint deg = deg_dim[0];
    if (elem < total) {
        T val = x[elem];
        // Clamp to [-1, 1] for numerical stability
        val = metal::max(T(-1.0), metal::min(T(1.0), val));

        uint base_idx = elem * deg;
        T t0 = T(1.0);
        out[base_idx] = t0;
        if (deg > 1) {
            T t1 = val;
            out[base_idx + 1] = t1;
            #pragma unroll
            for (uint k = 2; k < deg; ++k) {
                T t_next = metal::fma(T(2.0) * val, t1, -t0);
                out[base_idx + k] = t_next;
                t0 = t1;
                t1 = t_next;
            }
        }
    }
"""

_cheby_metal_kernel = fast.metal_kernel(
    name="cheby_polynomial_basis",
    input_names=["x", "N_dim", "D_dim", "deg_dim"],
    output_names=["out"],
    source=_CHEBY_SOURCE,
)


def metal_cheby_basis(x: mx.array, degree: int) -> mx.array:
    """
    Compute Chebyshev polynomial basis T_0(x), ..., T_{degree-1}(x) in GPU registers.

    Args:
        x: (*batch_dims, D) input array
        degree: number of polynomial terms

    Returns:
        (*batch_dims, D, degree) basis evaluation array
    """
    orig_shape = x.shape
    if len(orig_shape) != 2:
        x = x.reshape(-1, orig_shape[-1])

    N, D = x.shape
    N_dim = _scalar(N, mx.uint32)
    D_dim = _scalar(D, mx.uint32)
    deg_dim = _scalar(degree, mx.uint32)

    total_threads = N * D
    outputs = _cheby_metal_kernel(
        inputs=[x, N_dim, D_dim, deg_dim],
        template=[("T", x.dtype)],
        grid=(total_threads, 1, 1),
        threadgroup=(min(total_threads, 256), 1, 1),
        output_shapes=[(N, D, degree)],
        output_dtypes=[x.dtype],
    )
    res = outputs[0]
    if len(orig_shape) != 2:
        res = res.reshape(*orig_shape[:-1], D, degree)
    return res


# Shifted ReLU tent basis kernel (ReLUKAN).
_RELU_SOURCE = """
    uint g = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint n = thread_position_in_grid.z;
    uint G = G_dim[0];
    uint D = D_dim[0];
    uint N = N_dim[0];
    if (n < N && d < D && g < G) {
        T val = x[n * D + d];
        T c = centers[d * G + g];
        T diff = metal::abs(val - c) * inv_h[0];
        out[(n * D + d) * G + g] = metal::max(T(0.0), T(1.0) - diff);
    }
"""

_relu_metal_kernel = fast.metal_kernel(
    name="relukan_tent_basis",
    input_names=["x", "centers", "inv_h", "N_dim", "D_dim", "G_dim"],
    output_names=["out"],
    source=_RELU_SOURCE,
)


def metal_relu_basis(x: mx.array, centers: mx.array, inv_h: float) -> mx.array:
    """
    Compute piece-wise linear (tent) basis functions via ReLU using custom Metal kernel.

    Args:
        x: (*batch_dims, D) input array
        centers: (D, G) knot centers
        inv_h: 1.0 / grid_step

    Returns:
        (*batch_dims, D, G) basis evaluation array
    """
    orig_shape = x.shape
    if len(orig_shape) != 2:
        x = x.reshape(-1, orig_shape[-1])

    N, D = x.shape
    G = centers.shape[1]

    inv_h_arr = _scalar(inv_h, x.dtype)
    N_dim = _scalar(N, mx.uint32)
    D_dim = _scalar(D, mx.uint32)
    G_dim = _scalar(G, mx.uint32)

    outputs = _relu_metal_kernel(
        inputs=[x, centers, inv_h_arr, N_dim, D_dim, G_dim],
        template=[("T", x.dtype)],
        grid=(G, D, N),
        threadgroup=(min(G, 32), min(D, 8), 1),
        output_shapes=[(N, D, G)],
        output_dtypes=[x.dtype],
    )
    res = outputs[0]
    if len(orig_shape) != 2:
        res = res.reshape(*orig_shape[:-1], D, G)
    return res


# Cubic B-spline basis kernel (Cox-de Boor).
_BSPLINE_SOURCE = """
    uint d = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    uint N = N_dim[0];
    uint D = D_dim[0];
    uint GS = grid_size_dim[0];
    uint num_bases = GS + 3;

    if (n < N && d < D) {
        T val = x[n * D + d];
        T gmin = grid_min[0];
        T invh = inv_h[0];
        T pos = (val - gmin) * invh;

        int span = metal::clamp(int(metal::floor(pos)), 0, int(GS - 1));
        T u = metal::clamp(pos - T(span), T(0.0), T(1.0));

        T one_u = T(1.0) - u;
        constexpr float inv6 = 0.16666667f;
        T b0 = (one_u * one_u * one_u) * T(inv6);
        T b1 = (T(3.0) * u * u * u - T(6.0) * u * u + T(4.0)) * T(inv6);
        T b2 = (T(-3.0) * u * u * u + T(3.0) * u * u + T(3.0) * u + T(1.0)) * T(inv6);
        T b3 = (u * u * u) * T(inv6);

        uint out_base = (n * D + d) * num_bases;
        for (uint i = 0; i < num_bases; ++i) {
            out[out_base + i] = T(0.0);
        }
        out[out_base + span + 0] = b0;
        out[out_base + span + 1] = b1;
        out[out_base + span + 2] = b2;
        out[out_base + span + 3] = b3;
    }
"""

_bspline_metal_kernel = fast.metal_kernel(
    name="bspline_cubic_basis",
    input_names=["x", "grid_min", "inv_h", "N_dim", "D_dim", "grid_size_dim"],
    output_names=["out"],
    source=_BSPLINE_SOURCE,
)


def metal_bspline_basis(x: mx.array, grid_min: float, grid_max: float, grid_size: int) -> mx.array:
    """
    Compute cubic B-spline basis functions using custom Metal GPU kernel.

    Args:
        x: (*batch_dims, D) input array
        grid_min: minimum bound of domain
        grid_max: maximum bound of domain
        grid_size: number of intervals

    Returns:
        (*batch_dims, D, grid_size + 3) B-spline basis evaluation array
    """
    orig_shape = x.shape
    if len(orig_shape) != 2:
        x = x.reshape(-1, orig_shape[-1])

    N, D = x.shape
    num_bases = grid_size + 3
    inv_h = float(grid_size / max(grid_max - grid_min, 1e-6))

    inv_h_arr = _scalar(inv_h, x.dtype)
    grid_min_arr = _scalar(grid_min, x.dtype)
    N_dim = _scalar(N, mx.uint32)
    D_dim = _scalar(D, mx.uint32)
    grid_size_dim = _scalar(grid_size, mx.uint32)

    outputs = _bspline_metal_kernel(
        inputs=[x, grid_min_arr, inv_h_arr, N_dim, D_dim, grid_size_dim],
        template=[("T", x.dtype)],
        grid=(D, N, 1),
        threadgroup=(min(D, 16), min(N, 16), 1),
        output_shapes=[(N, D, num_bases)],
        output_dtypes=[x.dtype],
    )
    res = outputs[0]
    if len(orig_shape) != 2:
        res = res.reshape(*orig_shape[:-1], D, num_bases)
    return res


# Continuous wavelet basis kernel (WavKAN).
_WAV_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint n = thread_position_in_grid.z;
    uint N = N_dim[0];
    uint D = D_dim[0];
    uint K = K_dim[0];
    uint w_type = type_dim[0];

    if (n < N && d < D && k < K) {
        T val = x[n * D + d];
        T t = trans[d * K + k];
        T s = scale[d * K + k];
        T z = (val - t) / (metal::abs(s) + T(1e-4));
        T z2 = z * z;
        T exp_term = metal::fast::exp2(T(-0.5 * 1.4426950408889634) * z2);

        T res = T(0.0);
        if (w_type == 1) {
            res = metal::fast::cos(T(5.0) * z) * exp_term;
        } else if (w_type == 2) {
            res = -z * exp_term;
        } else {
            res = (T(1.0) - z2) * exp_term;
        }
        out[(n * D + d) * K + k] = res;
    }
"""

_wav_metal_kernel = fast.metal_kernel(
    name="wavkan_basis",
    input_names=["x", "trans", "scale", "N_dim", "D_dim", "K_dim", "type_dim"],
    output_names=["out"],
    source=_WAV_SOURCE,
)


def metal_wav_basis(x: mx.array, translation: mx.array, scale: mx.array, wavelet_type: str = "mexican_hat") -> mx.array:
    """
    Compute continuous wavelet basis functions using custom Metal GPU kernel.
    """
    orig_shape = x.shape
    if len(orig_shape) != 2:
        x = x.reshape(-1, orig_shape[-1])

    N, D = x.shape
    K = translation.shape[-1]
    type_code = 1 if wavelet_type == "morlet" else (2 if wavelet_type == "dog" else 0)

    N_dim = _scalar(N, mx.uint32)
    D_dim = _scalar(D, mx.uint32)
    K_dim = _scalar(K, mx.uint32)
    type_dim = _scalar(type_code, mx.uint32)

    outputs = _wav_metal_kernel(
        inputs=[x, translation, scale, N_dim, D_dim, K_dim, type_dim],
        template=[("T", x.dtype)],
        grid=(K, D, N),
        threadgroup=(min(K, 16), min(D, 16), 1),
        output_shapes=[(N, D, K)],
        output_dtypes=[x.dtype],
    )
    res = outputs[0]
    if len(orig_shape) != 2:
        res = res.reshape(*orig_shape[:-1], D, K)
    return res


# Fourier series basis kernel (FourierKAN).
_FOURIER_SOURCE = """
    uint f_idx = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint n = thread_position_in_grid.z;
    uint N = N_dim[0];
    uint D = D_dim[0];
    uint num_freqs = K_dim[0];
    uint total_bases = 2 * num_freqs + 1;

    if (n < N && d < D && f_idx < num_freqs) {
        T val = x[n * D + d];
        uint base_offset = (n * D + d) * total_bases;
        if (f_idx == 0) {
            out[base_offset] = T(1.0);
        }
        T f = freqs[f_idx];
        T angle = val * f;
        out[base_offset + 1 + f_idx] = metal::fast::cos(angle);
        out[base_offset + 1 + num_freqs + f_idx] = metal::fast::sin(angle);
    }
"""

_fourier_metal_kernel = fast.metal_kernel(
    name="fourier_basis",
    input_names=["x", "freqs", "N_dim", "D_dim", "K_dim"],
    output_names=["out"],
    source=_FOURIER_SOURCE,
)


def metal_fourier_basis(x: mx.array, freqs: mx.array) -> mx.array:
    """
    Compute harmonic Fourier series basis using custom Metal GPU kernel.
    """
    orig_shape = x.shape
    if len(orig_shape) != 2:
        x = x.reshape(-1, orig_shape[-1])

    N, D = x.shape
    num_freqs = len(freqs)
    total_bases = 2 * num_freqs + 1

    N_dim = _scalar(N, mx.uint32)
    D_dim = _scalar(D, mx.uint32)
    K_dim = _scalar(num_freqs, mx.uint32)

    outputs = _fourier_metal_kernel(
        inputs=[x, freqs, N_dim, D_dim, K_dim],
        template=[("T", x.dtype)],
        grid=(num_freqs, D, N),
        threadgroup=(min(num_freqs, 16), min(D, 16), 1),
        output_shapes=[(N, D, total_bases)],
        output_dtypes=[x.dtype],
    )
    res = outputs[0]
    if len(orig_shape) != 2:
        res = res.reshape(*orig_shape[:-1], D, total_bases)
    return res


# Jacobi polynomial basis kernel (JacobiKAN).
_JACOBI_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint total = N_dim[0] * D_dim[0];
    uint deg = deg_dim[0];
    T alpha = alpha_dim[0];
    T beta = beta_dim[0];
    T a_b = alpha + beta;

    if (elem < total) {
        T val = x[elem];
        val = metal::max(T(-1.0), metal::min(T(1.0), val));
        uint base_idx = elem * deg;

        T p0 = T(1.0);
        out[base_idx + 0] = p0;

        if (deg > 1) {
            T p1 = T(0.5) * (alpha - beta + (a_b + T(2.0)) * val);
            out[base_idx + 1] = p1;

            T prev2 = p0;
            T prev1 = p1;

            #pragma unroll
            for (uint n = 2; n < deg; ++n) {
                T fn = T(n);
                T an = T(2.0) * fn * (fn + a_b) * (T(2.0) * fn + a_b - T(2.0));
                T bn_term1 = (T(2.0) * fn + a_b - T(1.0)) * (T(2.0) * fn + a_b) * (T(2.0) * fn + a_b - T(2.0));
                T bn_term2 = (T(2.0) * fn + a_b - T(1.0)) * (alpha * alpha - beta * beta);
                T bn = bn_term1 * val + bn_term2;
                T cn = T(2.0) * (fn + alpha - T(1.0)) * (fn + beta - T(1.0)) * (T(2.0) * fn + a_b);

                T pn = (bn * prev1 - cn * prev2) / an;
                out[base_idx + n] = pn;
                prev2 = prev1;
                prev1 = pn;
            }
        }
    }
"""

_jacobi_metal_kernel = fast.metal_kernel(
    name="jacobi_polynomial_basis",
    input_names=["x", "N_dim", "D_dim", "deg_dim", "alpha_dim", "beta_dim"],
    output_names=["out"],
    source=_JACOBI_SOURCE,
)


def metal_jacobi_basis(x: mx.array, degree: int, alpha: float = 0.0, beta: float = 0.0) -> mx.array:
    """
    Compute orthogonal Jacobi polynomial basis using custom Metal GPU kernel.
    """
    orig_shape = x.shape
    if len(orig_shape) != 2:
        x = x.reshape(-1, orig_shape[-1])

    N, D = x.shape
    total_threads = N * D

    N_dim = _scalar(N, mx.uint32)
    D_dim = _scalar(D, mx.uint32)
    deg_dim = _scalar(degree, mx.uint32)
    alpha_dim = _scalar(alpha, x.dtype)
    beta_dim = _scalar(beta, x.dtype)

    outputs = _jacobi_metal_kernel(
        inputs=[x, N_dim, D_dim, deg_dim, alpha_dim, beta_dim],
        template=[("T", x.dtype)],
        grid=(total_threads, 1, 1),
        threadgroup=(min(total_threads, 256), 1, 1),
        output_shapes=[(N, D, degree)],
        output_dtypes=[x.dtype],
    )
    res = outputs[0]
    if len(orig_shape) != 2:
        res = res.reshape(*orig_shape[:-1], D, degree)
    return res

