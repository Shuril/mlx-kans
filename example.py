"""
Comparative Demo: Training Different KAN Variants in Apple MLX.
Fits the classic nonlinear function: f(x, y) = exp(sin(pi * x) + y^2).
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


def target_function(x: mx.array) -> mx.array:
    """f(x, y) = exp(sin(pi * x) + y^2)"""
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    return mx.exp(mx.sin(mx.pi * x1) + (x2 ** 2))


def train_and_eval(name: str, model: nn.Module, x_tr, y_tr, x_te, y_te, epochs=100, lr=0.03):
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(m, x, y):
        pred = m(x)
        mse = mx.mean((pred - y) ** 2)
        reg = 1e-4 * m.regularization_loss() if hasattr(m, "regularization_loss") else 0.0
        return mse + reg

    train_step = build_train_step(model, optimizer, loss_fn)
    n_params = count_parameters(model)["trainable"]

    t0 = time.perf_counter()
    for _ in range(epochs):
        train_step(x_tr, y_tr)
        mx.eval(model.parameters(), optimizer.state)
    elapsed = time.perf_counter() - t0

    preds = model(x_te)
    test_mse = mx.mean((preds - y_te) ** 2).item()

    print(f"{name:<18} | Params: {n_params:<6d} | Test MSE: {test_mse:<8.5f} | Time: {elapsed*1000:<7.1f} ms ({elapsed/epochs*1000:.2f} ms/ep)")
    return test_mse


def main():
    print("=" * 80)
    print("  Сравнение сходимости семейства KAN на задаче exp(sin(pi*x) + y^2)")
    print("  Device:", mx.default_device())
    print("=" * 80)

    mx.random.seed(42)
    n_train, n_test = 1000, 200

    x_tr = mx.random.uniform(-1.0, 1.0, (n_train, 2))
    y_tr = target_function(x_tr)

    x_te = mx.random.uniform(-1.0, 1.0, (n_test, 2))
    y_te = target_function(x_te)

    hidden = [2, 5, 1]

    models = [
        ("FastKAN (RBF)", FastKAN(hidden, num_grids=6)),
        ("ReLUKAN (Tent)", ReLUKAN(hidden, num_grids=6)),
        ("ChebyKAN", ChebyKAN(hidden, degree=4)),
        ("WavKAN (Mexican)", WavKAN(hidden, num_wavelets=6, wavelet_type="mexican_hat")),
        ("WavKAN (Morlet)", WavKAN(hidden, num_wavelets=6, wavelet_type="morlet")),
        ("FourierKAN", FourierKAN(hidden, num_frequencies=3)),
        ("JacobiKAN", JacobiKAN(hidden, degree=4)),
        ("MultKAN (2.0)", MultKAN(hidden, num_grids=6)),
        ("LowRankKAN", LowRankKAN(hidden, rank=4, num_grids=6)),
        ("B-Spline KAN", KAN(hidden, grid_size=5, spline_order=3)),
    ]

    for name, model in models:
        train_and_eval(name, model, x_tr, y_tr, x_te, y_te, epochs=100)

    print("=" * 80)


if __name__ == "__main__":
    main()
