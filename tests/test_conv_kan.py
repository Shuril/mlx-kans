import os
import tempfile
import pytest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from efficient_kan import (
    ConvKAN1d,
    ConvKAN2d,
    FastConv2dKAN,
    ChebyConv2dKAN,
    ReLUConv2dKAN,
    extract_patches_1d,
    extract_patches_2d,
    save_pretrained,
    load_pretrained,
    build_train_step,
    FastKAN,
)


def test_extract_patches_1d():
    B, L, C = 2, 10, 4
    x = mx.random.normal((B, L, C))
    kernel_size = 3
    patches = extract_patches_1d(x, kernel_size=kernel_size, stride=1, padding=1)
    assert patches.shape == (B, L, C * kernel_size)


def test_extract_patches_2d():
    B, H, W, C = 2, 8, 8, 3
    x = mx.random.normal((B, H, W, C))
    patches = extract_patches_2d(x, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    assert patches.shape == (B, 8, 8, 3 * 9)


def test_conv_kan_1d():
    layer = ConvKAN1d(in_channels=4, out_channels=8, kernel_size=3, padding=1, basis="fastkan")
    x = mx.random.normal((2, 16, 4))
    y = layer(x)
    assert y.shape == (2, 16, 8)


def test_conv_kan_2d_bases():
    x = mx.random.normal((2, 8, 8, 4))

    # FastKAN basis
    conv_fast = FastConv2dKAN(in_channels=4, out_channels=8, kernel_size=3, padding=1)
    y_fast = conv_fast(x)
    assert y_fast.shape == (2, 8, 8, 8)

    # ChebyKAN basis
    conv_cheby = ChebyConv2dKAN(in_channels=4, out_channels=6, kernel_size=3, padding=1, degree=3)
    y_cheby = conv_cheby(x)
    assert y_cheby.shape == (2, 8, 8, 6)

    # ReLUKAN basis
    conv_relu = ReLUConv2dKAN(in_channels=4, out_channels=4, kernel_size=3, padding=1)
    y_relu = conv_relu(x)
    assert y_relu.shape == (2, 8, 8, 4)


def test_conv_kan_2d_training():
    class SimpleConvKANNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = FastConv2dKAN(in_channels=2, out_channels=4, kernel_size=3, padding=1)
            self.head = nn.Linear(4, 1)

        def __call__(self, x):
            h = self.conv(x)
            return self.head(mx.mean(h, axis=(1, 2)))

    model = SimpleConvKANNet()
    optimizer = optim.Adam(learning_rate=0.01)

    def loss_fn(m, x, y):
        pred = m(x)
        return mx.mean((pred - y) ** 2)

    train_step = build_train_step(model, optimizer, loss_fn)

    x = mx.random.normal((4, 8, 8, 2))
    target = mx.random.normal((4, 1))

    initial_loss = float(loss_fn(model, x, target))
    for _ in range(20):
        loss = float(train_step(x, target))

    assert loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {loss}"


def test_save_and_load_pretrained():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = FastKAN([4, 8, 2], num_grids=5)
        x = mx.random.normal((2, 4))
        y_before = model(x)

        # Save model
        save_path = save_pretrained(model, tmpdir, config={"architectures": ["FastKAN"], "hidden": [4, 8, 2]})
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(tmpdir, "config.json"))

        # Create fresh model with different initial weights
        model_loaded = FastKAN([4, 8, 2], num_grids=5)
        # Load weights
        load_pretrained(model_loaded, tmpdir)
        y_after = model_loaded(x)

        max_diff = float(mx.max(mx.abs(y_before - y_after)))
        assert max_diff < 1e-6, f"Reloaded model outputs differ: max diff = {max_diff}"
