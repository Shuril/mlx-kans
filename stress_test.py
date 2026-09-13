"""
Comprehensive Stress Test Suite for the Entire KAN Family in Apple MLX.

Tasks:
1. High-Frequency Multiscale Oscillations:
   f(x1, x2) = sin(8*pi*x1) * cos(6*pi*x2) + 0.5 * sin(16*pi*x1*x2)
   (Stress test for spectral bias and frequency resolution)

2. Sharp Non-Smooth Transition:
   f(x1, x2) = |x1| - 2*max(0, x2) + sign(x1 * x2) * |x1 - x2|^0.7
   (Stress test for Gibbs oscillations, Runge phenomenon, piecewise linearity)

3. Multiplicative Law (Physics):
   f(x1, x2, x3, x4) = (x1 * x2) * exp(-x3^2) + (x3 * x4^2)
   (Stress test for multiplicative nodes vs additive expansions)

4. High-Dimensional Multichannel Interaction (D=8):
   f(x) = exp(-sum(x^2)/4) * sin(pi * (x1 + x2 + x3 + x4))
   (Stress test for parameter explosion and curse of dimensionality)
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
    build_train_step,
    count_parameters,
)


def get_dataset_task1(n_samples, seed=42):
    """Task 1: High-Frequency Multiscale Oscillations (2D -> 1D)"""
    mx.random.seed(seed)
    x = mx.random.uniform(-1.0, 1.0, (n_samples, 2))
    x1, x2 = x[:, 0:1], x[:, 1:2]
    y = mx.sin(8.0 * mx.pi * x1) * mx.cos(6.0 * mx.pi * x2) + 0.5 * mx.sin(16.0 * mx.pi * x1 * x2)
    return x, y


def get_dataset_task2(n_samples, seed=43):
    """Task 2: Sharp Non-Smooth Transitions (2D -> 1D)"""
    mx.random.seed(seed)
    x = mx.random.uniform(-1.0, 1.0, (n_samples, 2))
    x1, x2 = x[:, 0:1], x[:, 1:2]
    term1 = mx.abs(x1) - 2.0 * mx.maximum(0.0, x2)
    term2 = mx.sign(x1 * x2) * (mx.abs(x1 - x2) ** 0.7)
    y = term1 + term2
    return x, y


def get_dataset_task3(n_samples, seed=44):
    """Task 3: Multiplicative Physical Law (4D -> 1D)"""
    mx.random.seed(seed)
    x = mx.random.uniform(-1.0, 1.0, (n_samples, 4))
    x1, x2, x3, x4 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
    y = (x1 * x2) * mx.exp(-(x3 ** 2)) + (x3 * (x4 ** 2))
    return x, y


def get_dataset_task4(n_samples, seed=45):
    """Task 4: High-Dimensional Nonlinear Target (8D -> 1D)"""
    mx.random.seed(seed)
    x = mx.random.uniform(-1.0, 1.0, (n_samples, 8))
    norm_sq = mx.sum(x ** 2, axis=-1, keepdims=True)
    sum_first4 = mx.sum(x[:, :4], axis=-1, keepdims=True)
    y = mx.exp(-norm_sq / 4.0) * mx.sin(mx.pi * sum_first4)
    return x, y


class StandardMLP(nn.Module):
    def __init__(self, in_f, hidden_f, out_f):
        super().__init__()
        self.l1 = nn.Linear(in_f, hidden_f)
        self.l2 = nn.Linear(hidden_f, hidden_f)
        self.l3 = nn.Linear(hidden_f, out_f)
    def __call__(self, x):
        return self.l3(nn.silu(self.l2(nn.silu(self.l1(x)))))


def create_models(in_f, hidden_f=16, out_f=1):
    hidden = [in_f, hidden_f, out_f]
    return {
        "MLP (Baseline)" : StandardMLP(in_f, hidden_f, out_f),
        "FastKAN (RBF)"  : FastKAN(hidden, num_grids=8),
        "ReLUKAN (Tent)" : ReLUKAN(hidden, num_grids=8),
        "ChebyKAN"       : ChebyKAN(hidden, degree=5),
        "WavKAN (MexHat)": WavKAN(hidden, num_wavelets=8, wavelet_type="mexican_hat"),
        "WavKAN (Morlet)": WavKAN(hidden, num_wavelets=8, wavelet_type="morlet"),
        "FourierKAN"     : FourierKAN(hidden, num_frequencies=5),
        "JacobiKAN"      : JacobiKAN(hidden, degree=5, alpha=0.0, beta=0.0),
        "MultKAN (2.0)"  : MultKAN(hidden, num_grids=8),
        "LowRankKAN"     : LowRankKAN(hidden, rank=4, num_grids=8),
        "B-Spline KAN"   : KAN(hidden, grid_size=5, spline_order=3),
    }


def train_single_model(model, x_train, y_train, x_test, y_test, epochs=150, lr=0.02):
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(m, x_val, y_val):
        pred = m(x_val)
        return mx.mean((pred - y_val) ** 2)

    train_step = build_train_step(model, optimizer, loss_fn)
    n_params = count_parameters(model)["trainable"]

    # Warmup
    for _ in range(3):
        train_step(x_train[:32], y_train[:32])
        mx.eval(model.parameters(), optimizer.state)

    t0 = time.perf_counter()
    for _ in range(epochs):
        train_step(x_train, y_train)
        mx.eval(model.parameters(), optimizer.state)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    test_preds = model(x_test)
    test_mse = mx.mean((test_preds - y_test) ** 2).item()

    return {
        "params": n_params,
        "test_mse": test_mse,
        "time_ms": elapsed_ms,
        "ms_per_epoch": elapsed_ms / epochs,
    }


def run_stress_test():
    print("=" * 96, flush=True)
    print("                STRESS TEST: ВСЁ СЕМЕЙСТВО KAN НА APPLE SILICON METAL GPU", flush=True)
    print(f"                Устройство: {mx.default_device()}", flush=True)
    print("=" * 96, flush=True)

    tasks = [
        ("Task 1: High-Frequency Oscillations", get_dataset_task1, 2, 16, 150, 0.02),
        ("Task 2: Sharp Non-Smooth Transitions", get_dataset_task2, 2, 16, 150, 0.02),
        ("Task 3: Multiplicative Physics Law", get_dataset_task3, 4, 16, 150, 0.02),
        ("Task 4: High-Dimensional Nonlinear (8D)", get_dataset_task4, 8, 24, 150, 0.02),
    ]

    all_results = {}

    for task_name, dataset_fn, in_f, hidden_f, epochs, lr in tasks:
        print(f"\n>>> Запуск {task_name} (Входов: {in_f}, Скрытых: {hidden_f}, Эпох: {epochs})", flush=True)
        print("-" * 96, flush=True)
        print(f"{'Модель':<18} | {'Параметры':<11} | {'Test MSE':<12} | {'Время (мс)':<12} | {'мс / эпоха':<12}", flush=True)
        print("-" * 96, flush=True)

        x_train, y_train = dataset_fn(1200, seed=100)
        x_test, y_test = dataset_fn(300, seed=200)

        models = create_models(in_f=in_f, hidden_f=hidden_f, out_f=1)
        task_res = {}

        for m_name, model in models.items():
            res = train_single_model(model, x_train, y_train, x_test, y_test, epochs=epochs, lr=lr)
            task_res[m_name] = res
            print(
                f"{m_name:<18} | {res['params']:<11,d} | {res['test_mse']:<12.6f} | {res['time_ms']:<9.1f} ms | {res['ms_per_epoch']:<7.2f} ms",
                flush=True,
            )

        all_results[task_name] = task_res

    print("\n" + "=" * 96, flush=True)
    print("                     ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА СТРЕСС-ТЕСТА (TEST MSE)", flush=True)
    print("=" * 96, flush=True)

    header = f"{'Модель':<18} | {'Task 1 (HighFreq)':<17} | {'Task 2 (NonSmooth)':<18} | {'Task 3 (Physics)':<16} | {'Task 4 (8D Target)':<17}"
    print(header, flush=True)
    print("-" * 96, flush=True)

    model_names = list(models.keys())
    for m_name in model_names:
        mse1 = all_results["Task 1: High-Frequency Oscillations"][m_name]["test_mse"]
        mse2 = all_results["Task 2: Sharp Non-Smooth Transitions"][m_name]["test_mse"]
        mse3 = all_results["Task 3: Multiplicative Physics Law"][m_name]["test_mse"]
        mse4 = all_results["Task 4: High-Dimensional Nonlinear (8D)"][m_name]["test_mse"]
        print(f"{m_name:<18} | {mse1:<17.5f} | {mse2:<18.5f} | {mse3:<16.5f} | {mse4:<17.5f}", flush=True)

    print("=" * 96, flush=True)


if __name__ == "__main__":
    run_stress_test()
