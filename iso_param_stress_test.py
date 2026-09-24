"""
Iso-Parameter Stress Test for the KAN Family on Apple Silicon Metal GPU.
All models are strictly calibrated to have the same parameter budget (+- 1%):
  - Tasks 1 & 2 (2D inputs): ~500 parameters
  - Task 3 (4D inputs):      ~600 parameters
  - Task 4 (8D inputs):      ~1000 parameters
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
    def __init__(self, in_f, h1, h2, out_f=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, out_f)
        )
    def __call__(self, x):
        return self.net(x)


def get_iso_models_2d():
    """Calibrated for in_f=2, target ~500 parameters (all within 498-504 params)"""
    return {
        "MLP (Baseline)" : StandardMLP(2, 9, 43, 1),             # 501 params
        "FastKAN (RBF)"  : FastKAN([2, 21, 1], num_grids=7),      # 504 params
        "ReLUKAN (Tent)" : ReLUKAN([2, 21, 1], num_grids=7),      # 504 params
        "ChebyKAN"       : ChebyKAN([2, 24, 1], degree=6),        # 504 params
        "WavKAN (MexHat)": WavKAN([2, 21, 1], num_wavelets=4, wavelet_type="mexican_hat"), # 499 params
        "WavKAN (Morlet)": WavKAN([2, 21, 1], num_wavelets=4, wavelet_type="morlet"),     # 499 params
        "FourierKAN"     : FourierKAN([2, 14, 1], num_frequencies=5), # 504 params
        "JacobiKAN"      : JacobiKAN([2, 24, 1], degree=6, alpha=0.0, beta=0.0), # 504 params
        "MultKAN (2.0)"  : MultKAN([2, 21, 1], num_grids=5),      # 498 params
        "LowRankKAN"     : LowRankKAN([2, 29, 1], rank=4, num_grids=8), # 500 params
        "B-Spline KAN"   : KAN([2, 21, 1], grid_size=3, spline_order=3), # 504 params
    }


def get_iso_models_4d():
    """Calibrated for in_f=4, target ~600 parameters (all within 600-601 params)"""
    return {
        "MLP (Baseline)" : StandardMLP(4, 19, 24, 1),             # 600 params
        "FastKAN (RBF)"  : FastKAN([4, 15, 1], num_grids=7),      # 600 params
        "ReLUKAN (Tent)" : ReLUKAN([4, 15, 1], num_grids=7),      # 600 params
        "ChebyKAN"       : ChebyKAN([4, 20, 1], degree=5),        # 600 params
        "WavKAN (MexHat)": WavKAN([4, 14, 1], num_wavelets=5, wavelet_type="mexican_hat"), # 600 params
        "WavKAN (Morlet)": WavKAN([4, 14, 1], num_wavelets=5, wavelet_type="morlet"),     # 600 params
        "FourierKAN"     : FourierKAN([4, 10, 1], num_frequencies=5), # 600 params
        "JacobiKAN"      : JacobiKAN([4, 20, 1], degree=5, alpha=0.0, beta=0.0), # 600 params
        "MultKAN (2.0)"  : MultKAN([4, 11, 1], num_grids=7),      # 600 params
        "LowRankKAN"     : LowRankKAN([4, 46, 1], rank=3, num_grids=4), # 601 params
        "B-Spline KAN"   : KAN([4, 12, 1], grid_size=5, spline_order=3), # 600 params
    }


def get_iso_models_8d():
    """Calibrated for in_f=8, target ~1000 parameters (all within 990-1008 params)"""
    return {
        "MLP (Baseline)" : StandardMLP(8, 27, 26, 1),             # 998 params
        "FastKAN (RBF)"  : FastKAN([8, 14, 1], num_grids=7),      # 1008 params
        "ReLUKAN (Tent)" : ReLUKAN([8, 14, 1], num_grids=7),      # 1008 params
        "ChebyKAN"       : ChebyKAN([8, 16, 1], degree=6),        # 1008 params
        "WavKAN (MexHat)": WavKAN([8, 12, 1], num_wavelets=6, wavelet_type="mexican_hat"), # 996 params
        "WavKAN (Morlet)": WavKAN([8, 12, 1], num_wavelets=6, wavelet_type="morlet"),     # 996 params
        "FourierKAN"     : FourierKAN([8, 14, 1], num_frequencies=3), # 1008 params
        "JacobiKAN"      : JacobiKAN([8, 16, 1], degree=6, alpha=0.0, beta=0.0), # 1008 params
        "MultKAN (2.0)"  : MultKAN([8, 13, 1], num_grids=5),      # 990 params
        "LowRankKAN"     : LowRankKAN([8, 53, 1], rank=2, num_grids=6), # 998 params
        "B-Spline KAN"   : KAN([8, 14, 1], grid_size=3, spline_order=3), # 1008 params
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


def run_iso_stress_test():
    print("=" * 96, flush=True)
    print("      ИЗО-ПАРАМЕТРИЧЕСКИЙ СТРЕСС-ТЕСТ: ОДИНАКОВЫЙ БЮДЖЕТ ПАРАМЕТРОВ (ISO-PARAMS +-1%)", flush=True)
    print(f"      Устройство: {mx.default_device()}", flush=True)
    print("=" * 96, flush=True)

    tasks = [
        ("Task 1: High-Frequency Oscillations", get_dataset_task1, get_iso_models_2d, 150, 0.02),
        ("Task 2: Sharp Non-Smooth Transitions", get_dataset_task2, get_iso_models_2d, 150, 0.02),
        ("Task 3: Multiplicative Physics Law", get_dataset_task3, get_iso_models_4d, 150, 0.02),
        ("Task 4: High-Dimensional Target (8D)", get_dataset_task4, get_iso_models_8d, 150, 0.02),
    ]

    all_results = {}

    for task_name, dataset_fn, models_fn, epochs, lr in tasks:
        models = models_fn()
        sample_params = list(models.values())[0]
        p_count = count_parameters(sample_params)["trainable"]
        print(f"\n>>> {task_name} | Бюджет параметров: ~{p_count} | Эпох: {epochs}", flush=True)
        print("-" * 96, flush=True)
        print(f"{'Модель':<18} | {'Параметры':<11} | {'Test MSE':<12} | {'Время (мс)':<12} | {'мс / эпоха':<12}", flush=True)
        print("-" * 96, flush=True)

        x_train, y_train = dataset_fn(1200, seed=100)
        x_test, y_test = dataset_fn(300, seed=200)

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
    print("            СВОДНАЯ ТАБЛИЦА ISO-ПАРАМЕТРИЧЕСКОГО СТРЕСС-ТЕСТА (TEST MSE)", flush=True)
    print("=" * 96, flush=True)

    header = f"{'Модель':<18} | {'Task 1 (~500p)':<16} | {'Task 2 (~500p)':<16} | {'Task 3 (~600p)':<16} | {'Task 4 (~1000p)':<16}"
    print(header, flush=True)
    print("-" * 96, flush=True)

    model_names = list(models.keys())
    for m_name in model_names:
        mse1 = all_results["Task 1: High-Frequency Oscillations"][m_name]["test_mse"]
        mse2 = all_results["Task 2: Sharp Non-Smooth Transitions"][m_name]["test_mse"]
        mse3 = all_results["Task 3: Multiplicative Physics Law"][m_name]["test_mse"]
        mse4 = all_results["Task 4: High-Dimensional Target (8D)"][m_name]["test_mse"]
        print(f"{m_name:<18} | {mse1:<16.5f} | {mse2:<16.5f} | {mse3:<16.5f} | {mse4:<16.5f}", flush=True)

    print("=" * 96, flush=True)


if __name__ == "__main__":
    run_iso_stress_test()
