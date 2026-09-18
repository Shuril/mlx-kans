"""
Head-to-Head Benchmark: Direct Metal Shading Language (MSL) vs Apple MLX
Comparing:
1. Standard MLX KAN (ChebyKAN)
2. MLX Custom Metal Kernel (mx.fast.metal_kernel)
3. Direct Metal Fused MSL Kernel (libmetal_kan.dylib)
"""

import time
import numpy as np
import mlx.core as mx
from efficient_kan.cheby_kan import ChebyKANLinear
from metal_kans.metal_kan import MetalChebyKAN

def sync_mlx():
    mx.eval()

def run_benchmark():
    print("=" * 95)
    print(" HEAD-TO-HEAD BENCHMARK: DIRECT METAL SHADING LANGUAGE (MSL) vs APPLE MLX")
    print(" Hardware: Apple Silicon M1 Metal GPU | Unified Memory")
    print("=" * 95)

    scenarios = [
        ("Small Batch (B=1024, D=64)",       1024,  64,  64, 4),
        ("Medium Batch (B=4096, D=64)",      4096,  64,  64, 4),
        ("Large Batch (B=16384, D=64)",     16384,  64,  64, 4),
        ("High-Throughput (B=65536, D=64)", 65536,  64,  64, 4),
        ("Wide Layer (B=4096, D=128)",       4096, 128, 128, 4),
        ("Deep Layer (B=8192, D=128)",       8192, 128, 128, 4),
    ]

    WARMUP = 15
    ITERS = 60

    print(f"\n{'Scenario':32s} | {'MLX Std (ms)':13s} | {'MLX MSL (ms)':13s} | {'Direct MSL (ms)':15s} | {'Speedup':10s}")
    print("-" * 95)

    results = []

    for name, B, D_in, D_out, deg in scenarios:
        # 1. Setup MLX Standard
        mlx_std = ChebyKANLinear(D_in, D_out, degree=deg, bias=True, use_metal_kernel=False)
        mx.eval(mlx_std.parameters())

        # 2. Setup MLX MSL Kernel
        mlx_msl = ChebyKANLinear(D_in, D_out, degree=deg, bias=True, use_metal_kernel=True)
        mx.eval(mlx_msl.parameters())

        # 3. Setup Direct Metal
        metal_layer = MetalChebyKAN(D_in, D_out, degree=deg, bias=True)
        metal_layer.w_cheby = np.array(mlx_std.cheby_weight)
        metal_layer.w_base = np.array(mlx_std.base_weight)
        metal_layer.bias = np.array(mlx_std.bias)

        x_np = np.random.uniform(-1.0, 1.0, (B, D_in)).astype(np.float32)
        x_mx = mx.array(x_np)

        # -----------------------
        # Bench MLX Standard
        # -----------------------
        for _ in range(WARMUP):
            out = mlx_std(x_mx)
            mx.eval(out)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = mlx_std(x_mx)
            mx.eval(out)
        t1 = time.perf_counter()
        ms_mlx_std = (t1 - t0) / ITERS * 1000.0

        # -----------------------
        # Bench MLX MSL Kernel
        # -----------------------
        for _ in range(WARMUP):
            out = mlx_msl(x_mx)
            mx.eval(out)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = mlx_msl(x_mx)
            mx.eval(out)
        t1 = time.perf_counter()
        ms_mlx_msl = (t1 - t0) / ITERS * 1000.0

        # -----------------------
        # Bench Direct Metal Fused
        # -----------------------
        ms_direct_metal = metal_layer.benchmark(x_np, warmup=WARMUP, iters=ITERS)

        speedup = ms_mlx_std / ms_direct_metal
        throughput_samples_sec = (B / (ms_direct_metal / 1000.0))

        results.append({
            "name": name,
            "B": B,
            "D_in": D_in,
            "D_out": D_out,
            "ms_mlx_std": ms_mlx_std,
            "ms_mlx_msl": ms_mlx_msl,
            "ms_direct_metal": ms_direct_metal,
            "speedup": speedup,
            "throughput": throughput_samples_sec
        })

        print(f"{name:32s} | {ms_mlx_std:10.3f} ms | {ms_mlx_msl:10.3f} ms | {ms_direct_metal:12.3f} ms | {speedup:8.2f}x")

    print("=" * 95)
    print("\n DETAILED THROUGHPUT COMPARISON:")
    print(f"{'Scenario':32s} | {'MLX Std (spl/s)':16s} | {'Direct MSL (spl/s)':19s} | {'VRAM Traffic Saved':18s}")
    print("-" * 95)
    for r in results:
        t_mlx = r["B"] / (r["ms_mlx_std"] / 1000.0)
        t_direct = r["throughput"]
        # Intermediate basis tensor size: B * D_in * K * 4 bytes (read + written = 2x)
        mem_saved_mb = (2 * r["B"] * r["D_in"] * 4 * 4) / (1024 * 1024)
        print(f"{r['name']:32s} | {t_mlx:13,.0f} spl/s | {t_direct:16,.0f} spl/s | {mem_saved_mb:12.2f} MB / pass")
    print("=" * 95)

if __name__ == "__main__":
    run_benchmark()
