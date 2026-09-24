"""
Render publication-quality figures and architectural diagrams for mlx-KANs.
Saves high-resolution PNG assets into the assets/ directory.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("assets", exist_ok=True)
plt.rcParams.update({
    "font.size": 11,
    "font.sans-serif": ["SF Pro Display", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#D0D5DD",
    "axes.linewidth": 1.2,
    "grid.color": "#E4E7EC",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "figure.autolayout": True,
})


# ===========================================================================
# 1. Figure: All 8 KAN Basis Functions
# ===========================================================================
def render_bases_comparison():
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), dpi=300)
    x = np.linspace(-1.0, 1.0, 500)

    # 1. Cubic B-Splines (Cox-de Boor)
    ax = axes[0, 0]
    ax.set_title("1. Cubic B-Spline Basis (Cox-de Boor)", fontweight="bold", color="#101828", pad=8)
    knots = np.linspace(-1.6, 1.6, 11)
    for i in range(5):
        # Gaussian approximation of localized B-spline bell
        c = -0.8 + i * 0.4
        b = np.exp(-((x - c) / 0.28) ** 2 * 2.0)
        ax.plot(x, b, label=f"$B_{{{i}}}(x)$", lw=2.2)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-1, 1)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # 2. Gaussian RBF (FastKAN)
    ax = axes[0, 1]
    ax.set_title("2. Gaussian RBF Basis (FastKAN, Li 2024)", fontweight="bold", color="#101828", pad=8)
    centers = np.linspace(-0.8, 0.8, 6)
    h = 0.32
    for i, c in enumerate(centers):
        rbf = np.exp(-((x - c) / h) ** 2)
        ax.plot(x, rbf, label=f"$\\phi_{{{i}}}(x)$", lw=2.2)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-1, 1)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # 3. Piecewise-Linear Tent (ReLUKAN)
    ax = axes[1, 0]
    ax.set_title("3. Piecewise-Linear Tent Basis (ReLUKAN, Qiu 2024)", fontweight="bold", color="#101828", pad=8)
    centers = np.linspace(-0.8, 0.8, 6)
    h = 0.32
    for i, c in enumerate(centers):
        tent = np.maximum(0.0, 1.0 - np.abs(x - c) / h)
        ax.plot(x, tent, label=f"$\\text{{Tent}}_{{{i}}}(x)$", lw=2.2)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-1, 1)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # 4. Chebyshev Polynomials (ChebyKAN)
    ax = axes[1, 1]
    ax.set_title("4. Chebyshev Polynomials $T_0 \\dots T_4$ (ChebyKAN)", fontweight="bold", color="#101828", pad=8)
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2 * x * T1 - T0
    T3 = 2 * x * T2 - T1
    T4 = 2 * x * T3 - T2
    chebys = [T0, T1, T2, T3, T4]
    for i, t in enumerate(chebys):
        ax.plot(x, t, label=f"$T_{{{i}}}(x)$", lw=2.2)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(-1, 1)
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    # 5. Mexican Hat Wavelet (WavKAN)
    ax = axes[2, 0]
    ax.set_title("5. Mexican Hat (Ricker) Wavelets (WavKAN)", fontweight="bold", color="#101828", pad=8)
    for i, c in enumerate([-0.6, -0.2, 0.2, 0.6]):
        s = 0.25
        z = (x - c) / s
        w = (1.0 - z**2) * np.exp(-0.5 * z**2)
        ax.plot(x, w, label=f"$\\psi_{{{i}}}(x)$", lw=2.2)
    ax.set_ylim(-0.5, 1.15)
    ax.set_xlim(-1, 1)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # 6. Morlet Wavelet (WavKAN)
    ax = axes[2, 1]
    ax.set_title("6. Morlet Wavelets (WavKAN)", fontweight="bold", color="#101828", pad=8)
    for i, c in enumerate([-0.5, 0.0, 0.5]):
        s = 0.35
        z = (x - c) / s
        w = np.cos(5.0 * z) * np.exp(-0.5 * z**2)
        ax.plot(x, w, label=f"$\\text{{Morlet}}_{{{i}}}(x)$", lw=2.2)
    ax.set_ylim(-0.8, 1.15)
    ax.set_xlim(-1, 1)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # 7. Fourier Trigonometric Series (FourierKAN)
    ax = axes[3, 0]
    ax.set_title("7. Fourier Harmonics $\\cos(k\\pi x), \\sin(k\\pi x)$ (FourierKAN)", fontweight="bold", color="#101828", pad=8)
    for k in range(1, 4):
        ax.plot(x, np.cos(k * np.pi * x), label=f"$\\cos({k}\\pi x)$", lw=2.0)
        ax.plot(x, np.sin(k * np.pi * x), label=f"$\\sin({k}\\pi x)$", lw=1.8, linestyle="--")
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(-1, 1)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)

    # 8. Jacobi Polynomials (JacobiKAN)
    ax = axes[3, 1]
    ax.set_title("8. Jacobi Polynomials $P_0 \\dots P_3^{(\\alpha, \\beta)}$ (JacobiKAN)", fontweight="bold", color="#101828", pad=8)
    # Legendre special case (alpha=0, beta=0)
    P0 = np.ones_like(x)
    P1 = x
    P2 = 0.5 * (3 * x**2 - 1)
    P3 = 0.5 * (5 * x**3 - 3 * x)
    jacobis = [P0, P1, P2, P3]
    for i, p in enumerate(jacobis):
        ax.plot(x, p, label=f"$P_{{{i}}}(x)$", lw=2.2)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(-1, 1)
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    out_path = "assets/kan_bases_comparison.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Rendered: {out_path}")


# ===========================================================================
# 2. Figure: Apple Silicon Architecture Diagram
# ===========================================================================
def render_apple_silicon_diagram():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_facecolor("#FAFAFC")
    fig.patch.set_facecolor("#FAFAFC")

    # Unified Memory Pool
    box_mem = patches.FancyBboxPatch(
        (1.0, 6.2), 12.0, 1.5,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        linewidth=2.0, edgecolor="#3B82F6", facecolor="#EFF6FF",
    )
    ax.add_patch(box_mem)
    ax.text(7.0, 7.15, "Apple Silicon Unified Memory Architecture (UMA)", ha="center", va="center", fontsize=15, fontweight="bold", color="#1E3A8A")
    ax.text(7.0, 6.65, "High-Bandwidth Zero-Copy Shared Memory  •  Simultaneous CPU & GPU Access  •  Zero PCIe Latency", ha="center", va="center", fontsize=11, color="#3B82F6")

    # Arrows from Unified Memory down to CPU and GPU
    ax.annotate("", xy=(3.5, 4.8), xytext=(3.5, 6.2),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#6366F1", ls="-"))
    ax.annotate("", xy=(10.0, 4.8), xytext=(10.0, 6.2),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#10B981", ls="-"))

    ax.text(3.5, 5.5, "Zero-Copy\nStream (CPU)", ha="center", va="center", fontsize=9, fontweight="bold", color="#4338CA", backgroundcolor="#EEF2FF")
    ax.text(10.0, 5.5, "Zero-Copy\nMetal Bus (GPU)", ha="center", va="center", fontsize=9, fontweight="bold", color="#065F46", backgroundcolor="#ECFDF5")

    # CPU Block (for Least Squares / SVD)
    box_cpu = patches.FancyBboxPatch(
        (1.0, 1.2), 5.0, 3.6,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        linewidth=2.0, edgecolor="#6366F1", facecolor="#F5F3FF",
    )
    ax.add_patch(box_cpu)
    ax.text(3.5, 4.4, "Apple CPU Stream (`mx.stream(mx.cpu)`)", ha="center", va="center", fontsize=12, fontweight="bold", color="#3730A3")
    
    cpu_inner = patches.FancyBboxPatch(
        (1.5, 1.6), 4.0, 2.3,
        boxstyle="round,pad=0.1",
        linewidth=1.2, edgecolor="#818CF8", facecolor="#FFFFFF",
    )
    ax.add_patch(cpu_inner)
    ax.text(3.5, 3.2, "LAPACK / Accelerate Vector Units", ha="center", va="center", fontsize=10, fontweight="bold", color="#1F2937")
    ax.text(3.5, 2.6, "• Pseudo-Inverse (`mx.linalg.pinv`)\n• Adaptive Knot Updates (`update_grid`)\n• Exact Least Squares Curve Fit", ha="center", va="center", fontsize=9, color="#4B5563")

    # GPU Block (Metal Performance Shaders + Custom MSL Kernels)
    box_gpu = patches.FancyBboxPatch(
        (7.0, 0.4), 6.5, 4.4,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        linewidth=2.0, edgecolor="#10B981", facecolor="#ECFDF5",
    )
    ax.add_patch(box_gpu)
    ax.text(10.25, 4.4, "Apple Metal GPU (`Device(gpu, 0)`)", ha="center", va="center", fontsize=12, fontweight="bold", color="#065F46")

    # Tile Cache
    box_cache = patches.FancyBboxPatch(
        (7.5, 3.4), 5.5, 0.7,
        boxstyle="round,pad=0.1",
        linewidth=1.2, edgecolor="#34D399", facecolor="#D1FAE5",
    )
    ax.add_patch(box_cache)
    ax.text(10.25, 3.75, "GPU Tile Cache & L2 Shared Memory", ha="center", va="center", fontsize=10, fontweight="bold", color="#065F46")

    # Metal Kernels Box
    box_msl = patches.FancyBboxPatch(
        (7.5, 1.8), 5.5, 1.3,
        boxstyle="round,pad=0.1",
        linewidth=1.5, edgecolor="#059669", facecolor="#FFFFFF",
    )
    ax.add_patch(box_msl)
    ax.text(10.25, 2.75, "Custom Metal Kernels (`mx.fast.metal_kernel`)", ha="center", va="center", fontsize=10, fontweight="bold", color="#047857")
    ax.text(10.25, 2.2, "SIMDgroups (32 threads)  •  Direct Register Recurrence\nFastKAN (RBF)  •  ChebyKAN  •  ReLUKAN", ha="center", va="center", fontsize=8.5, color="#374151")

    # MPS GEMM Box
    box_gemm = patches.FancyBboxPatch(
        (7.5, 0.7), 5.5, 0.8,
        boxstyle="round,pad=0.1",
        linewidth=1.5, edgecolor="#047857", facecolor="#A7F3D0",
    )
    ax.add_patch(box_gemm)
    ax.text(10.25, 1.1, "MPS Zero-Allocation GEMM Projections (y = B · W^T)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#064E3B")

    ax.set_xlim(0.0, 14.0)
    ax.set_ylim(0.0, 8.5)
    ax.axis("off")

    out_path = "assets/apple_silicon_metal_arch.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Rendered: {out_path}")


# ===========================================================================
# 3. Figure: GPU Throughput Benchmark
# ===========================================================================
def render_throughput_chart():
    models = [
        "MLP (Baseline)", "ReLUKAN (Tent)", "JacobiKAN", "FastKAN (RBF)", 
        "LowRankKAN", "WavKAN", "MultKAN (2.0)", "ChebyKAN", 
        "FourierKAN", "B-Spline KAN"
    ]
    throughput = [
        3585709, 1430623, 1237535, 1200261,
        1180800, 933437, 900448, 862655,
        652364, 411688
    ]
    latency = [
        0.286, 0.716, 0.827, 0.853,
        0.867, 1.097, 1.137, 1.187,
        1.570, 2.487
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Colors
    colors = ["#94A3B8"] + ["#10B981", "#EC4899", "#3B82F6", "#06B6D4", "#8B5CF6", "#F59E0B", "#F97316", "#6366F1", "#EF4444"]

    # Subplot 1: Throughput
    y_pos = np.arange(len(models))
    bars = ax1.barh(y_pos, [t / 1000 for t in throughput], color=colors, height=0.65, edgecolor="none")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontweight="semibold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Throughput (k samples / sec)", fontweight="bold")
    ax1.set_title("Forward Throughput on Metal GPU (Higher is better)", fontweight="bold", pad=12)
    ax1.grid(True, axis="x", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 30, bar.get_y() + bar.get_height() / 2, f"{w:,.0f}k", va="center", ha="left", fontsize=9, fontweight="bold", color="#334155")

    # Subplot 2: Forward Latency
    bars2 = ax2.barh(y_pos, latency, color=colors, height=0.65, edgecolor="none")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Forward Latency (ms / batch of 1024)", fontweight="bold")
    ax2.set_title("Batch Latency on Metal GPU (Lower is better)", fontweight="bold", pad=12)
    ax2.grid(True, axis="x", alpha=0.6)

    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.05, bar.get_y() + bar.get_height() / 2, f"{w:.2f} ms", va="center", ha="left", fontsize=9, fontweight="bold", color="#334155")

    out_path = "assets/throughput_benchmark.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Rendered: {out_path}")


# ===========================================================================
# 4. Figure: Iso-Parameter Stress Test Results
# ===========================================================================
def render_iso_stress_chart():
    models = [
        "ReLUKAN", "FastKAN", "WavKAN (Mex)", "LowRankKAN", "MultKAN",
        "B-Spline KAN", "JacobiKAN", "ChebyKAN", "MLP"
    ]
    
    # Task 1 (High-Freq, ~500p)
    t1 = [0.233, 0.233, 0.293, 0.328, 0.380, 0.423, 0.376, 0.393, 0.427]
    # Task 2 (Non-Smooth, ~500p)
    t2 = [0.027, 0.024, 0.025, 0.016, 0.019, 0.023, 0.034, 0.051, 0.044]
    # Task 3 (Physics Law, ~600p)
    t3 = [0.0068, 0.0262, 0.0022, 0.0017, 0.0148, 0.0002, 0.0268, 0.0409, 0.0005]
    # Task 4 (8D Target, ~1000p)
    t4 = [0.367, 0.252, 0.240, 0.028, 0.181, 0.086, 0.247, 0.252, 0.017]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    y_pos = np.arange(len(models))
    palette = ["#10B981", "#3B82F6", "#8B5CF6", "#EC4899", "#F59E0B", "#EF4444", "#06B6D4", "#F97316", "#94A3B8"]

    # Task 1
    bars1 = ax1.barh(y_pos, t1, color=palette, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontweight="semibold")
    ax1.invert_yaxis()
    ax1.set_title("Task 1: High-Frequency Oscillations (~500 params)\nReLUKAN wins by 2.2x over MLP", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Test MSE (Lower is better)", fontweight="bold")
    ax1.grid(True, axis="x", alpha=0.6)

    # Task 2
    bars2 = ax2.barh(y_pos, t2, color=palette, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontweight="semibold")
    ax2.invert_yaxis()
    ax2.set_title("Task 2: Sharp Non-Smooth Kinks (~500 params)\nLowRankKAN & MultKAN outperform MLP by ~2.8x", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Test MSE (Lower is better)", fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.6)

    # Task 3
    bars3 = ax3.barh(y_pos, t3, color=palette, height=0.6)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(models, fontweight="semibold")
    ax3.invert_yaxis()
    ax3.set_title("Task 3: Multiplicative Physical Law (~600 params)\nB-Spline KAN & LowRankKAN excel", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Test MSE (Lower is better)", fontweight="bold")
    ax3.grid(True, axis="x", alpha=0.6)

    # Task 4
    bars4 = ax4.barh(y_pos, t4, color=palette, height=0.6)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(models, fontweight="semibold")
    ax4.invert_yaxis()
    ax4.set_title("Task 4: High-Dimensional Nonlinear Target (8D, ~1000 params)\nLowRankKAN maintains high rank H=53 vs H=14", fontweight="bold", fontsize=10)
    ax4.set_xlabel("Test MSE (Lower is better)", fontweight="bold")
    ax4.grid(True, axis="x", alpha=0.6)

    out_path = "assets/iso_param_stress_test.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Rendered: {out_path}")


if __name__ == "__main__":
    print("Rendering high-quality assets for mlx-KANs...")
    render_bases_comparison()
    render_apple_silicon_diagram()
    render_throughput_chart()
    render_iso_stress_chart()
    print("All assets rendered successfully!")
