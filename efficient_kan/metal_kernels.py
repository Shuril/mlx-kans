"""
Custom Metal Shading Language (MSL) kernels for Apple Silicon acceleration in MLX.
Provides hardware-fused basis evaluations executing directly on the M-series GPU.
"""

from __future__ import annotations
import mlx.core as mx
import mlx.core.fast as fast


def is_metal_available() -> bool:
    """Check whether Metal GPU device is available."""
    try:
        return mx.default_device().type == mx.DeviceType.gpu
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. Metal Kernel for Gaussian RBF Basis (FastKAN)
# ---------------------------------------------------------------------------
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
        out[(n * D + d) * G + g] = metal::exp(-diff * diff);
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
        x: (N, D) input array
        centers: (D, G) grid centers
        inv_h: 1.0 / grid_step

    Returns:
        (N, D, G) basis evaluation array
    """
    N, D = x.shape
    G = centers.shape[1]

    inv_h_arr = mx.array([inv_h], dtype=x.dtype)
    N_dim = mx.array([N], dtype=mx.uint32)
    D_dim = mx.array([D], dtype=mx.uint32)
    G_dim = mx.array([G], dtype=mx.uint32)

    outputs = _rbf_metal_kernel(
        inputs=[x, centers, inv_h_arr, N_dim, D_dim, G_dim],
        template=[("T", x.dtype)],
        grid=(G, D, N),
        threadgroup=(min(G, 32), min(D, 8), 1),
        output_shapes=[(N, D, G)],
        output_dtypes=[x.dtype],
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# 2. Metal Kernel for Chebyshev Polynomials (ChebyKAN)
# ---------------------------------------------------------------------------
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
            for (uint k = 2; k < deg; ++k) {
                T t_next = T(2.0) * val * t1 - t0;
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
        x: (N, D) input array
        degree: number of polynomial terms

    Returns:
        (N, D, degree) basis evaluation array
    """
    N, D = x.shape
    N_dim = mx.array([N], dtype=mx.uint32)
    D_dim = mx.array([D], dtype=mx.uint32)
    deg_dim = mx.array([degree], dtype=mx.uint32)

    total_threads = N * D
    outputs = _cheby_metal_kernel(
        inputs=[x, N_dim, D_dim, deg_dim],
        template=[("T", x.dtype)],
        grid=(total_threads, 1, 1),
        threadgroup=(min(total_threads, 256), 1, 1),
        output_shapes=[(N, D, degree)],
        output_dtypes=[x.dtype],
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# 3. Metal Kernel for Shifted ReLU / Tent Basis (ReLUKAN)
# ---------------------------------------------------------------------------
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
        x: (N, D) input array
        centers: (D, G) knot centers
        inv_h: 1.0 / grid_step

    Returns:
        (N, D, G) basis evaluation array
    """
    N, D = x.shape
    G = centers.shape[1]

    inv_h_arr = mx.array([inv_h], dtype=x.dtype)
    N_dim = mx.array([N], dtype=mx.uint32)
    D_dim = mx.array([D], dtype=mx.uint32)
    G_dim = mx.array([G], dtype=mx.uint32)

    outputs = _relu_metal_kernel(
        inputs=[x, centers, inv_h_arr, N_dim, D_dim, G_dim],
        template=[("T", x.dtype)],
        grid=(G, D, N),
        threadgroup=(min(G, 32), min(D, 8), 1),
        output_shapes=[(N, D, G)],
        output_dtypes=[x.dtype],
    )
    return outputs[0]
