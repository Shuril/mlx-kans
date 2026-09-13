# mlx-KANs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.10+-red.svg)](https://github.com/ml-explore/mlx)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20--%20M4%20Metal%20GPU-purple.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**mlx-KANs** is a high-performance suite of **Kolmogorov-Arnold Networks (KAN)** built on top of [Apple MLX](https://github.com/ml-explore/mlx), specifically optimized for **Apple Silicon (M1 / M2 / M3 / M4 / Pro / Max / Ultra)** with custom Metal Shading Language (MSL) kernels, mixed-precision FP16/BF16, zero-allocation GEMM projections, and unified memory acceleration.

> 🇷🇺 *Русскоязычная версия документации доступна в [README_RU.md](README_RU.md).*

---

## Key Highlights

- **9 Modern KAN Architectures in One Package**:
  - `FastKAN` (Gaussian RBF basis) — ~3x faster than B-splines with $C^\infty$ smoothness.
  - `ReLUKAN` (Piecewise-linear tent basis) — zero transcendental operations, >1.17M samples/s.
  - `LowRankKAN` (LoRA / Bottleneck factorization) — **up to 10x parameter compression** for deep & wide layers.
  - `WavKAN` (Continuous Wavelets: Mexican Hat, Morlet, DOG) — multiresolution frequency-time localization, lowest MSE.
  - `ChebyKAN` (Chebyshev Polynomials 1st kind) — minimax polynomial optimality on $[-1, 1]$ without grid updates.
  - `FourierKAN` (Harmonic Fourier series) — trigonometric basis for periodic signals, audio, and PINNs.
  - `JacobiKAN` (Jacobi / Legendre / Gegenbauer orthogonal polynomials) — parametrized by $(\alpha, \beta)$.
  - `MultKAN` (KAN 2.0 with Multiplication Nodes) — exact analytical product terms $u \cdot v$ for physical conservation laws.
  - `KAN` (Cubic B-splines / Efficient-KAN) — classical Cox-de Boor reformulation with adaptive grid updates.
- **Hardware-Fused Metal Kernels (`mx.fast.metal_kernel`)**:
  - Handcrafted Metal Shading Language (MSL) kernels executing RBF, tent-ReLU, and Chebyshev recurrences directly in GPU thread registers without VRAM memory traffic.
- **Unified Memory Stream Orchestration**:
  - Zero-copy CPU/GPU streaming via macOS unified memory for least-squares pseudo-inverses (`pinv`).
- **JIT Graph Compilation**:
  - Integrated `build_train_step` compiling forward pass, autograd backpropagation, and optimizer updates into a single monolithic Metal execution graph.
- **Mixed Precision**:
  - Native `to_fp16()` and `to_bf16()` casting doubling GPU memory bandwidth and throughput.

<p align="center">
  <img src="assets/kan_bases_comparison.png" alt="KAN Basis Functions Comparison" width="95%"/>
</p>

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Shuril/mlx-kans.git
cd mlx-kans

# Install in editable mode
pip install -e .

# Or using uv (recommended for ultra-fast setup):
uv pip install -e .
```

---

## Quick Start

```python
import mlx.core as mx
import mlx.optimizers as optim
import mlx_kans as kans

# 1. Instantiate any KAN model
model = kans.FastKAN(layers_hidden=[2, 16, 1], num_grids=8)

# 2. Setup optimizer and loss
optimizer = optim.Adam(learning_rate=0.02)

def loss_fn(m, x, y):
    pred = m(x)
    return mx.mean((pred - y) ** 2)

# 3. Compile training step directly into Metal GPU graph
train_step = kans.build_train_step(model, optimizer, loss_fn)

# 4. Generate data & train
x_train = mx.random.uniform(-1.0, 1.0, (1000, 2))
y_train = mx.sin(4.0 * mx.pi * x_train[:, 0:1]) + (x_train[:, 1:2] ** 2)

for epoch in range(100):
    loss = train_step(x_train, y_train)
    mx.eval(model.parameters(), optimizer.state)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.5f}")
```

---

## Performance Benchmarks on Apple Silicon Metal GPU

Benchmarked on **Apple Silicon GPU (`Device(gpu, 0)`)**, architecture `[64, 64, 64]`, batch size **1024**:

```text
========================================================================================
Model              | Trainable Params | Forward (ms)   | Throughput (samples/s) | Train Step (ms)
----------------------------------------------------------------------------------------
MLP (Baseline)     | 8,320            |      0.339 ms  |        3,016,866       |       0.691 ms
LowRankKAN         | 15,360           |      0.828 ms  |        1,236,498       |       1.951 ms
ReLUKAN (Tent)     | 57,344           |      0.871 ms  |        1,175,727       |       2.072 ms
WavKAN (Wavelet)   | 58,880           |      0.983 ms  |        1,041,365       |       3.028 ms
JacobiKAN          | 40,960           |      0.992 ms  |        1,032,210       |       1.762 ms
MultKAN (2.0)      | 86,016           |      1.016 ms  |        1,008,141       |       2.290 ms
FastKAN (RBF)      | 57,344           |      1.142 ms  |          896,675       |       2.087 ms
FourierKAN         | 65,536           |      1.397 ms  |          733,221       |       2.365 ms
ChebyKAN           | 40,960           |      2.186 ms  |          468,365       |       1.799 ms
B-Spline KAN       | 81,920           |      2.392 ms  |          428,156       |       5.193 ms
========================================================================================
```

<p align="center">
  <img src="assets/throughput_benchmark.png" alt="Metal GPU Throughput Benchmark" width="95%"/>
</p>

---

## Iso-Parameter Stress Test (Identical Parameter Budget $\pm 1\%$)

To ensure strict scientific fairness, all models were calibrated to have the **exact same parameter budget ($\pm 1\%$)** across 4 diverse stress scenarios:

- **Task 1: High-Frequency Oscillations (2D)**: $f(x) = \sin(8\pi x_1)\cos(6\pi x_2) + 0.5\sin(16\pi x_1 x_2)$
- **Task 2: Sharp Non-Smooth Transitions (2D)**: $f(x) = |x_1| - 2\max(0, x_2) + \text{sign}(x_1 x_2)|x_1 - x_2|^{0.7}$
- **Task 3: Multiplicative Physical Law (4D)**: $f(x) = (x_1 x_2) e^{-x_3^2} + x_3 x_4^2$
- **Task 4: High-Dimensional Nonlinear Target (8D)**: $f(x) = \exp(-\sum x^2 / 4) \sin(\pi \sum_{1}^4 x_i)$

### Results (Test MSE, 150 epochs on Metal GPU):

| Model | Budget | Task 1 (High-Freq, 2D) | Task 2 (Non-Smooth, 2D) | Task 3 (Physics, 4D) | Task 4 (Target, 8D) | Train Time / Epoch |
|---|---|---|---|---|---|---|
| **MLP (Baseline)** | $\approx$ const | 0.42251 | 0.04385 | **0.00055** 🥈 | **0.01713** 🥇 | **0.72 – 0.93 ms** |
| **`ReLUKAN` (Tent)** | $\approx$ const | **0.18538** 🏆 | 0.02744 | 0.00679 | 0.36655 | **0.79 – 0.88 ms** ⚡ |
| **`FastKAN` (RBF)** | $\approx$ const | **0.30261** 🥈 | 0.02387 | 0.02616 | 0.25177 | **0.84 – 0.93 ms** ⚡ |
| **`WavKAN` (MexHat)**| $\approx$ const | **0.31261** 🥉 | 0.02515 | 0.00222 | 0.23987 | 1.05 – 1.38 ms |
| **`LowRankKAN`** | $\approx$ const | 0.35542 | **0.01555** 🏆 | **0.00174** 🥉 | **0.02772** 🥈 | 1.05 – 1.49 ms |
| **`MultKAN` (2.0)** | $\approx$ const | 0.37618 | **0.01930** 🥈 | 0.01482 | 0.18123 | 0.90 – 0.97 ms |
| **`B-Spline KAN`** | $\approx$ const | 0.43318 | **0.02300** 🥉 | **0.00017** 🏆 | **0.08623** 🥉 | 2.07 – 2.91 ms |
| **`JacobiKAN`** | $\approx$ const | 0.35679 | 0.03450 | 0.02681 | 0.24693 | 0.90 – 0.98 ms |
| **`ChebyKAN`** | $\approx$ const | 0.35915 | 0.05112 | 0.04086 | 0.25154 | 0.84 – 1.02 ms |
| **`FourierKAN`** | $\approx$ const | 0.92909 | 1.47926 | 0.25044 | 0.38723 | 0.89 – 0.90 ms |

<p align="center">
  <img src="assets/iso_param_stress_test.png" alt="Iso-Parameter Stress Test Results" width="95%"/>
</p>

### Key Takeaways:
1. **High-Frequency Details**: `ReLUKAN` outperforms standard MLP by **over 2.2x** (MSE 0.185 vs 0.422) by eliminating spectral bias.
2. **Non-Smooth Boundaries**: `LowRankKAN` and `MultKAN` outperform MLP by **2.3x–2.8x** because univariate edge functions isolate kinks without global ringing artifacts.
3. **High-Dimensional Scaling (8D)**: `LowRankKAN` achieves near-parity with MLP (MSE 0.027 vs 0.017) thanks to rank-4 factorization while standard full-grid KANs degrade under tight parameter limits.

---

## Documentation

Detailed guides and references are available in the [`docs/`](docs/) directory:
- [**API Reference**](docs/API_REFERENCE.md): Full documentation for all classes, methods, and constructor arguments.
- [**Architecture & Mathematics**](docs/ARCHITECTURE.md): Mathematical formulations for each basis family and the Kolmogorov-Arnold representation theorem.
- [**Metal GPU Kernels Guide**](docs/METAL_KERNELS.md): Custom MSL shader implementation, register utilization, and threadgroup tiling.

---

## Running Tests & Benchmarks

```bash
# Run the test suite (14 unit tests, ~0.1s on Metal GPU)
uv run --with mlx python -m unittest test_kan.py

# Run the throughput benchmark
uv run --with mlx python benchmark.py

# Run the iso-parameter stress test
uv run --with mlx python iso_param_stress_test.py

# Run the comparative nonlinear regression demo
uv run --with mlx python example.py
```

---

## License

Released under the [MIT License](LICENSE).
