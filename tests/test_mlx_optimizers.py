import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx_kans import (
    ChebyKANLinear,
    FastKANLinear,
    Optimizer,
    Adam,
    AdamW,
    Muon,
    Lion,
    RMSprop,
    SGD,
    AdaDelta,
    Adafactor,
    Adagrad,
)


class TestMLXOptimizers(unittest.TestCase):
    def test_imports_and_types(self):
        for opt_cls in [Adam, AdamW, Muon, Lion, RMSprop, SGD, AdaDelta, Adafactor, Adagrad]:
            self.assertTrue(issubclass(opt_cls, Optimizer))

    def test_training_step_cheby(self):
        model = ChebyKANLinear(4, 2, degree=4)
        x = mx.random.normal((16, 4))
        y = mx.random.normal((16, 2))

        optimizers = [
            Adam(learning_rate=1e-3),
            AdamW(learning_rate=1e-3),
            Muon(learning_rate=0.01),
            Lion(learning_rate=1e-4),
            RMSprop(learning_rate=1e-3),
            SGD(learning_rate=1e-2),
        ]

        def loss_fn(m, x, y):
            return mx.mean((m(x) - y) ** 2)

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        for opt in optimizers:
            loss, grads = loss_and_grad_fn(model, x, y)
            opt.update(model, grads)
            mx.eval(model.parameters())
            self.assertFalse(mx.isnan(loss).item())


if __name__ == "__main__":
    unittest.main()
