"""
Comprehensive Benchmark Suite for the Efficient-KAN Family on Apple Silicon MLX.
Compares:
  - B-spline KAN (Original Efficient-KAN)
  - FastKAN (Gaussian RBF)
  - ReLUKAN (Piecewise linear / Tent)
  - ChebyKAN (Chebyshev Polynomials)
  - WavKAN (Continuous Wavelets)
  - FourierKAN (Fourier Series)
  - JacobiKAN (Jacobi / Legendre)
  - MultKAN (KAN 2.0 with Multiplication Nodes)
  - LowRankKAN (LoRA / Bottleneck KAN)
  - Standard MLP (Baseline)
"""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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
    count_parameters,
    build_train_step,
)


def run_benchmark(batch_size=1024, hidden_dims=[64, 64, 64], n_iters=40):
    print("=" * 88, flush=True)
    print(f"  Бенчмарк семейства KAN на Apple Silicon Metal GPU ({mx.default_device()})", flush=True)
    print(f"  Топология: {hidden_dims} | Batch Size: {batch_size} | Итераций: {n_iters}", flush=True)
    print("=" * 88, flush=True)

    # Define standard MLP baseline
    class StandardMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.l2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        def __call__(self, x):
            return self.l2(nn.silu(self.l1(x)))

    models = {
        "MLP (Baseline)" : StandardMLP(),
        "FastKAN (RBF)"  : FastKAN(hidden_dims, num_grids=6),
        "ReLUKAN (Tent)" : ReLUKAN(hidden_dims, num_grids=6),
        "ChebyKAN"       : ChebyKAN(hidden_dims, degree=4),
        "WavKAN (Wavelet)": WavKAN(hidden_dims, num_wavelets=6),
        "FourierKAN"     : FourierKAN(hidden_dims, num_frequencies=3),
        "JacobiKAN"      : JacobiKAN(hidden_dims, degree=4),
        "MultKAN (2.0)"  : MultKAN(hidden_dims, num_grids=6),
        "LowRankKAN"     : LowRankKAN(hidden_dims, rank=8, num_grids=6),
        "B-Spline KAN"   : KAN(hidden_dims, grid_size=5, spline_order=3),
    }

    x = mx.random.normal((batch_size, hidden_dims[0]))
    y = mx.random.normal((batch_size, hidden_dims[-1]))

    print(f"{'Модель':<18} | {'Параметры':<11} | {'Forward (ms)':<14} | {'Fwd Thput (smp/s)':<19} | {'Train Step (ms)':<15}", flush=True)
    print("-" * 88, flush=True)

    for name, model in models.items():
        params_info = count_parameters(model)
        n_params = params_info["trainable"]

        # Warmup forward
        for _ in range(5):
            mx.eval(model(x))

        # Benchmark Forward
        t0 = time.perf_counter()
        for _ in range(n_iters):
            mx.eval(model(x))
        t1 = time.perf_counter()

        fwd_ms = (t1 - t0) / n_iters * 1000
        throughput = batch_size / ((t1 - t0) / n_iters)

        # Benchmark Train Step (Forward + Backward + Adam Update)
        optimizer = optim.Adam(learning_rate=0.01)

        def loss_fn(m, x_val, y_val):
            return mx.mean((m(x_val) - y_val) ** 2)

        train_step = build_train_step(model, optimizer, loss_fn)

        # Warmup train
        for _ in range(5):
            l = train_step(x, y)
            mx.eval(model.parameters(), optimizer.state)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            l = train_step(x, y)
            mx.eval(model.parameters(), optimizer.state)
        t1 = time.perf_counter()

        train_ms = (t1 - t0) / n_iters * 1000

        print(
            f"{name:<18} | {n_params:<11,d} | {fwd_ms:10.3f} ms  | {throughput:16.1f}   | {train_ms:11.3f} ms",
            flush=True,
        )

    print("=" * 88, flush=True)


if __name__ == "__main__":
    run_benchmark(batch_size=1024, hidden_dims=[64, 64, 64], n_iters=40)
