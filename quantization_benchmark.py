"""
Hardware Benchmark: Native INT8 & INT4 Quantization on Apple Silicon Metal GPU.
Evaluates:
  1. Memory compression ratio (FP32 vs FP16 vs INT8 vs INT4)
  2. Forward inference latency (ms) & throughput (samples/sec)
  3. Numerical accuracy retention (MAE vs FP32)
  4. Generates visual chart assets/quantization_benchmark.png
"""

import os
import time
import mlx.core as mx
import mlx.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from efficient_kan import (
    KAN,
    FastKAN,
    ReLUKAN,
    ChebyKAN,
    WavKAN,
    FourierKAN,
    JacobiKAN,
    MultKAN,
    LowRankKAN,
    to_fp16,
    to_int8,
    to_int4,
    get_model_size,
)


def get_model_builders(hidden_dims):
    return {
        "FastKAN (RBF)": lambda: FastKAN(hidden_dims, num_grids=8),
        "ReLUKAN (Tent)": lambda: ReLUKAN(hidden_dims, num_grids=8),
        "ChebyKAN": lambda: ChebyKAN(hidden_dims, degree=6),
        "WavKAN (Wavelet)": lambda: WavKAN(hidden_dims, num_wavelets=8),
        "FourierKAN": lambda: FourierKAN(hidden_dims, num_frequencies=6),
        "JacobiKAN": lambda: JacobiKAN(hidden_dims, degree=6),
        "MultKAN (2.0)": lambda: MultKAN(hidden_dims, num_grids=8),
        "LowRankKAN": lambda: LowRankKAN(hidden_dims, rank=16, num_grids=8),
        "B-Spline KAN": lambda: KAN(hidden_dims, grid_size=6),
    }


def measure_inference(model, x, warmups=10, iters=40):
    for _ in range(warmups):
        mx.eval(model(x))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model(x)
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    avg_sec = float(np.mean(times))
    lat_ms = avg_sec * 1000.0
    fps = x.shape[0] / avg_sec
    return lat_ms, fps, out


def run_benchmark():
    hidden = [128, 256, 128]
    batch_size = 256
    x_fp32 = mx.random.normal((batch_size, hidden[0]))

    builders = get_model_builders(hidden)

    results = []

    print("=" * 100)
    print(f"  mlx-KANs: Native Metal INT8 & INT4 Quantization Benchmark on Apple Silicon ({mx.default_device()})")
    print(f"  Topology: {hidden} | Batch: {batch_size} | Device: Unified Memory Metal GPU")
    print("=" * 100)

    for name, builder in builders.items():
        # 1. FP32
        m_fp32 = builder()
        s_fp32 = get_model_size(m_fp32)
        lat_fp32, fps_fp32, out_fp32 = measure_inference(m_fp32, x_fp32)

        # 2. FP16
        m_fp16 = builder()
        to_fp16(m_fp16)
        s_fp16 = get_model_size(m_fp16)
        x_fp16 = x_fp32.astype(mx.float16)
        lat_fp16, fps_fp16, out_fp16 = measure_inference(m_fp16, x_fp16)

        # 3. INT8
        m_int8 = builder()
        to_int8(m_int8, group_size=64)
        s_int8 = get_model_size(m_int8)
        lat_int8, fps_int8, out_int8 = measure_inference(m_int8, x_fp32)
        mae_int8 = mx.mean(mx.abs(out_fp32 - out_int8)).item()

        # 4. INT4
        m_int4 = builder()
        to_int4(m_int4, group_size=64)
        s_int4 = get_model_size(m_int4)
        lat_int4, fps_int4, out_int4 = measure_inference(m_int4, x_fp32)
        mae_int4 = mx.mean(mx.abs(out_fp32 - out_int4)).item()

        comp_int8 = s_fp32["total_bytes"] / max(s_int8["total_bytes"], 1)
        comp_int4 = s_fp32["total_bytes"] / max(s_int4["total_bytes"], 1)

        results.append({
            "name": name,
            "fp32_mb": s_fp32["mb"],
            "fp16_mb": s_fp16["mb"],
            "int8_mb": s_int8["mb"],
            "int4_mb": s_int4["mb"],
            "comp_int8": comp_int8,
            "comp_int4": comp_int4,
            "lat_fp32": lat_fp32,
            "lat_fp16": lat_fp16,
            "lat_int8": lat_int8,
            "lat_int4": lat_int4,
            "mae_int8": mae_int8,
            "mae_int4": mae_int4,
        })

        print(f"[{name:<17}] FP32: {s_fp32['mb']:.2f} MB ({lat_fp32:.2f} ms) | INT8: {s_int8['mb']:.2f} MB ({comp_int8:.2f}x, MAE={mae_int8:.4f}) | INT4: {s_int4['mb']:.2f} MB ({comp_int4:.2f}x)")

    # Print Table
    print("\n" + "=" * 105)
    print(f"| {'Model':<17} | {'FP32 Mem':<9} | {'INT8 Mem':<9} | {'Comp (INT8)':<11} | {'INT8 MAE':<10} | {'INT4 Mem':<9} | {'Comp (INT4)':<11} |")
    print("|" + "-" * 19 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 13 + "|" + "-" * 12 + "|" + "-" * 11 + "|" + "-" * 13 + "|")
    for r in results:
        print(f"| {r['name']:<17} | {r['fp32_mb']:>6.2f} MB | {r['int8_mb']:>6.2f} MB | {r['comp_int8']:>9.2f}x | {r['mae_int8']:>10.4f} | {r['int4_mb']:>6.2f} MB | {r['comp_int4']:>9.2f}x |")
    print("=" * 105)

    # Plot figure
    os.makedirs("assets", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    names = [r["name"] for r in results]
    x_indices = np.arange(len(names))
    width = 0.22

    # Plot 1: Memory Footprint (MB)
    ax1.bar(x_indices - 1.5 * width, [r["fp32_mb"] for r in results], width, label="FP32 (4 bytes)", color="#4A90E2", alpha=0.9)
    ax1.bar(x_indices - 0.5 * width, [r["fp16_mb"] for r in results], width, label="FP16 (2 bytes)", color="#50E3C2", alpha=0.9)
    ax1.bar(x_indices + 0.5 * width, [r["int8_mb"] for r in results], width, label="INT8 Native Metal (~1 byte)", color="#F5A623", alpha=0.9)
    ax1.bar(x_indices + 1.5 * width, [r["int4_mb"] for r in results], width, label="INT4 Native Metal (~0.5 byte)", color="#D0021B", alpha=0.9)

    ax1.set_title("VRAM Memory Footprint by Precision", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Memory (MB)", fontsize=11)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Plot 2: Compression Ratio vs FP32
    ax2.bar(x_indices - 0.5 * width, [r["comp_int8"] for r in results], width * 1.5, label="INT8 Compression (~3.5x)", color="#F5A623", alpha=0.9)
    ax2.bar(x_indices + 0.5 * width, [r["comp_int4"] for r in results], width * 1.5, label="INT4 Compression (~6.5x)", color="#D0021B", alpha=0.9)

    ax2.axhline(1.0, color="gray", linestyle=":", label="FP32 Baseline (1.0x)")
    ax2.set_title("Memory Compression Multiplier vs FP32", fontsize=14, fontweight="bold", pad=12)
    ax2.set_ylabel("Compression Ratio (higher is better)", fontsize=11)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    for i, r in enumerate(results):
        ax2.text(i - 0.5 * width, r["comp_int8"] + 0.1, f"{r['comp_int8']:.1f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax2.text(i + 0.5 * width, r["comp_int4"] + 0.1, f"{r['comp_int4']:.1f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("mlx-KANs: Native Apple Silicon Metal GPU Quantization (INT8 / INT4)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    chart_path = "assets/quantization_benchmark.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\n[+] Rendered visualization chart saved to: {chart_path}")


if __name__ == "__main__":
    run_benchmark()
