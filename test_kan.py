import math
import os
import tempfile
import unittest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from efficient_kan import (
    KAN,
    KANLinear,
    compute_b_splines,
    FastKAN,
    FastKANLinear,
    ReLUKAN,
    ReLUKANLinear,
    ChebyKAN,
    ChebyKANLinear,
    WavKAN,
    WavKANLinear,
    FourierKAN,
    FourierKANLinear,
    JacobiKAN,
    JacobiKANLinear,
    MultKAN,
    MultKANLinear,
    LowRankKAN,
    LowRankKANLinear,
    RationalKAN,
    RationalKANLinear,
    build_train_step,
    count_parameters,
    to_fp16,
    metal_rbf_basis,
    metal_cheby_basis,
    metal_relu_basis,
    is_metal_available,
    QuantizedWeight,
    quantize,
    to_int8,
    to_int4,
    get_model_size,
    checkpoint_kan,
    CheckpointedKAN,
    prune,
    compact_kan,
    compute_node_importance,
    to_symbolic,
    SymbolicKAN,
)


class TestEfficientKANSuite(unittest.TestCase):
    def setUp(self):
        mx.random.seed(42)

    # -----------------------------------------------------------------------
    # 1. B-spline KAN Tests
    # -----------------------------------------------------------------------
    def test_b_splines_partition_of_unity(self):
        in_features, grid_size, spline_order = 4, 5, 3
        grid_range = (-1.0, 1.0)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid_steps = (
            mx.arange(-spline_order, grid_size + spline_order + 1, dtype=mx.float32) * h
            + grid_range[0]
        )
        grid = mx.broadcast_to(grid_steps[None, :], (in_features, grid_steps.shape[0]))
        x = mx.random.uniform(-0.95, 0.95, (40, in_features))
        bases = compute_b_splines(x, grid, spline_order)
        sums = mx.sum(bases, axis=-1)
        max_err = mx.max(mx.abs(sums - 1.0)).item()
        self.assertLess(max_err, 1e-5)

    def test_kan_linear_multidim_shapes(self):
        layer = KANLinear(6, 3)
        x2d = mx.random.normal((16, 6))
        self.assertEqual(layer(x2d).shape, (16, 3))
        x3d = mx.random.normal((4, 8, 6))
        self.assertEqual(layer(x3d).shape, (4, 8, 3))

    def test_kan_update_grid(self):
        layer = KANLinear(4, 4, grid_size=5)
        x = mx.random.uniform(-0.8, 0.8, (50, 4))
        y_before = layer(x)
        mx.eval(y_before)
        layer.update_grid(x)
        y_after = layer(x)
        mx.eval(y_after)
        self.assertLess(mx.max(mx.abs(y_before - y_after)).item(), 1e-3)

    # -----------------------------------------------------------------------
    # 2. FastKAN (Gaussian RBF) Tests
    # -----------------------------------------------------------------------
    def test_fast_kan(self):
        model = FastKAN([4, 8, 2], num_grids=6)
        x = mx.random.uniform(-1.0, 1.0, (16, 4))
        out = model(x)
        self.assertEqual(out.shape, (16, 2))

        # Test gradients
        def loss(m, x_val):
            return mx.mean(m(x_val) ** 2)

        loss_and_grad = nn.value_and_grad(model, loss)
        l, g = loss_and_grad(model, x)
        mx.eval(l, g)
        self.assertFalse(mx.isnan(l).item())
        self.assertIn("layers", g)

    # -----------------------------------------------------------------------
    # 3. ReLUKAN (Piecewise Linear / Tent) Tests
    # -----------------------------------------------------------------------
    def test_relu_kan(self):
        model = ReLUKAN([3, 6, 1], num_grids=6)
        x = mx.random.uniform(-1.0, 1.0, (12, 3))
        out = model(x)
        self.assertEqual(out.shape, (12, 1))

        # Train 5 steps
        optimizer = optim.Adam(0.01)
        for _ in range(5):
            def loss(m):
                return mx.mean(m(x) ** 2)
            l, g = nn.value_and_grad(model, loss)(model)
            optimizer.update(model, g)
            mx.eval(model.parameters())
        self.assertFalse(mx.isnan(l).item())

    # -----------------------------------------------------------------------
    # 4. ChebyKAN (Chebyshev Polynomials) Tests
    # -----------------------------------------------------------------------
    def test_cheby_kan(self):
        model = ChebyKAN([4, 6, 2], degree=5)
        x = mx.random.uniform(-1.0, 1.0, (10, 4))
        out = model(x)
        self.assertEqual(out.shape, (10, 2))

        # Check clamp / stability on out-of-bounds input
        x_large = mx.array([[5.0, -10.0, 0.5, -0.2]])
        out_large = model(x_large)
        mx.eval(out_large)
        self.assertFalse(mx.any(mx.isnan(out_large)).item())

    # -----------------------------------------------------------------------
    # 5. WavKAN (Continuous Wavelets) Tests
    # -----------------------------------------------------------------------
    def test_wav_kan(self):
        # Mexican Hat
        model_mex = WavKAN([3, 5, 2], num_wavelets=6, wavelet_type="mexican_hat")
        # Morlet
        model_mor = WavKAN([3, 5, 2], num_wavelets=6, wavelet_type="morlet")

        x = mx.random.uniform(-1.0, 1.0, (16, 3))
        out_mex = model_mex(x)
        out_mor = model_mor(x)
        self.assertEqual(out_mex.shape, (16, 2))
        self.assertEqual(out_mor.shape, (16, 2))

    # -----------------------------------------------------------------------
    # 6. FourierKAN (Fourier Series) Tests
    # -----------------------------------------------------------------------
    def test_fourier_kan(self):
        model = FourierKAN([2, 8, 1], num_frequencies=4)
        x = mx.random.uniform(-1.0, 1.0, (20, 2))
        out = model(x)
        self.assertEqual(out.shape, (20, 1))

    # -----------------------------------------------------------------------
    # 7. JacobiKAN (Jacobi Polynomials) Tests
    # -----------------------------------------------------------------------
    def test_jacobi_kan(self):
        # Legendre (alpha=0, beta=0)
        model_leg = JacobiKAN([3, 6, 1], degree=4, alpha=0.0, beta=0.0)
        # Gegenbauer (alpha=0.5, beta=0.5)
        model_geg = JacobiKAN([3, 6, 1], degree=4, alpha=0.5, beta=0.5)

        x = mx.random.uniform(-1.0, 1.0, (8, 3))
        out_leg = model_leg(x)
        out_geg = model_geg(x)
        self.assertEqual(out_leg.shape, (8, 1))
        self.assertEqual(out_geg.shape, (8, 1))

    # -----------------------------------------------------------------------
    # 8. MultKAN (KAN 2.0 Multiplication Nodes) Tests
    # -----------------------------------------------------------------------
    def test_mult_kan(self):
        # Layer producing 4 outputs (2 additive, 2 multiplicative)
        layer = MultKANLinear(in_features=4, out_features=4, num_mult=2)
        x = mx.random.normal((10, 4))
        out = layer(x)
        self.assertEqual(out.shape, (10, 4))

        # Multi-layer MultKAN
        model = MultKAN([3, 6, 2])
        out_m = model(mx.random.normal((16, 3)))
        self.assertEqual(out_m.shape, (16, 2))

    # -----------------------------------------------------------------------
    # 9. LowRankKAN (Bottleneck / LoRA Factorization) Tests
    # -----------------------------------------------------------------------
    def test_low_rank_kan(self):
        standard_layer = FastKANLinear(32, 32, num_grids=8)
        low_rank_layer = LowRankKANLinear(32, 32, rank=4, num_grids=8)

        params_std = count_parameters(standard_layer)["trainable"]
        params_lr = count_parameters(low_rank_layer)["trainable"]

        # LowRank must have significantly fewer parameters
        self.assertLess(params_lr, params_std)
        self.assertLess(params_lr, params_std * 0.4)

        x = mx.random.normal((16, 32))
        out = low_rank_layer(x)
        self.assertEqual(out.shape, (16, 32))

    # -----------------------------------------------------------------------
    # 10. Custom Metal Kernels Accuracy Verification
    # -----------------------------------------------------------------------
    @unittest.skipUnless(is_metal_available(), "Metal GPU required")
    def test_custom_metal_kernels(self):
        N, D, G = 16, 8, 6
        x = mx.random.uniform(-1.0, 1.0, (N, D))
        centers = mx.random.uniform(-1.0, 1.0, (D, G))
        inv_h = 2.0

        # RBF Metal Kernel vs MLX reference
        rbf_metal = metal_rbf_basis(x, centers, inv_h)
        rbf_ref = mx.exp(-(((x[..., None] - centers[None, ...]) * inv_h) ** 2))
        self.assertLess(mx.max(mx.abs(rbf_metal - rbf_ref)).item(), 1e-5)

        # Cheby Metal Kernel vs MLX reference
        deg = 5
        cheby_metal = metal_cheby_basis(x, deg)
        # reference
        x_c = mx.clip(x, -1.0, 1.0)
        b0 = mx.ones_like(x_c[..., None])
        b1 = x_c[..., None]
        b2 = 2.0 * x_c[..., None] * b1 - b0
        b3 = 2.0 * x_c[..., None] * b2 - b1
        b4 = 2.0 * x_c[..., None] * b3 - b2
        cheby_ref = mx.concatenate([b0, b1, b2, b3, b4], axis=-1)
        self.assertLess(mx.max(mx.abs(cheby_metal - cheby_ref)).item(), 1e-5)

        # ReLU Tent Metal Kernel vs MLX reference
        relu_metal = metal_relu_basis(x, centers, inv_h)
        relu_ref = mx.maximum(0.0, 1.0 - mx.abs(x[..., None] - centers[None, ...]) * inv_h)
        self.assertLess(mx.max(mx.abs(relu_metal - relu_ref)).item(), 1e-5)

    # -----------------------------------------------------------------------
    # 11. Utilities: build_train_step and to_fp16
    # -----------------------------------------------------------------------
    def test_build_train_step_compiled(self):
        model = FastKAN([2, 5, 1], num_grids=6)
        optimizer = optim.Adam(0.01)

        def loss_fn(m, x, y):
            return mx.mean((m(x) - y) ** 2)

        train_step = build_train_step(model, optimizer, loss_fn)
        x = mx.random.normal((32, 2))
        y = mx.random.normal((32, 1))

        initial_loss = None
        for _ in range(10):
            l = train_step(x, y)
            mx.eval(model.parameters(), optimizer.state)
            if initial_loss is None:
                initial_loss = l.item()

        self.assertLess(l.item(), initial_loss)

    def test_to_fp16(self):
        model = ChebyKAN([3, 6, 2], degree=4)
        to_fp16(model)
        for k, p in model.parameters().items():
            if isinstance(p, list):
                for sub in p:
                    for arr in sub.values():
                        if mx.issubdtype(arr.dtype, mx.floating):
                            self.assertEqual(arr.dtype, mx.float16)

        x_fp16 = mx.random.uniform(-1.0, 1.0, (8, 3)).astype(mx.float16)
        out = model(x_fp16)
        self.assertEqual(out.dtype, mx.float16)

    # -----------------------------------------------------------------------
    # 12. Native Metal INT8 & INT4 Quantization Tests
    # -----------------------------------------------------------------------
    def test_quantized_weight_auto_padding(self):
        # Odd dimensions that don't divide by 32
        w = mx.random.normal((17, 13))
        qw = QuantizedWeight(w, group_size=32, bits=8)
        x = mx.random.normal((5, 13))
        y_ref = x @ w.T
        y_quant = qw(x)
        self.assertEqual(y_quant.shape, (5, 17))
        # Accuracy retention: max difference between FP32 and INT8 matmul
        self.assertLess(mx.max(mx.abs(y_quant - y_ref)).item(), 0.15)

    def test_quantize_int8_all_architectures(self):
        models = [
            KAN([6, 12, 2], grid_size=5),
            FastKAN([6, 12, 2], num_grids=6),
            ReLUKAN([6, 12, 2], num_grids=6),
            ChebyKAN([6, 12, 2], degree=4),
            WavKAN([6, 12, 2], num_wavelets=6),
            FourierKAN([6, 12, 2], num_frequencies=4),
            JacobiKAN([6, 12, 2], degree=4),
            MultKAN([6, 12, 2]),
            LowRankKAN([6, 12, 2], rank=4),
        ]
        x = mx.random.normal((8, 6))

        for m in models:
            y_fp32 = m(x)
            mx.eval(y_fp32)
            to_int8(m, group_size=32)
            y_int8 = m(x)
            mx.eval(y_int8)

            self.assertEqual(y_int8.shape, (8, 2))
            mae = mx.mean(mx.abs(y_fp32 - y_int8)).item()
            self.assertLess(mae, 0.1, f"High MAE on {type(m).__name__}: {mae}")

    def test_quantize_int4(self):
        m = FastKAN([8, 16, 2], num_grids=6)
        x = mx.random.normal((4, 8))
        y_fp32 = m(x)
        mx.eval(y_fp32)

        to_int4(m, group_size=32)
        y_int4 = m(x)
        mx.eval(y_int4)

        self.assertEqual(y_int4.shape, (4, 2))
        mae = mx.mean(mx.abs(y_fp32 - y_int4)).item()
        self.assertLess(mae, 0.1)

    def test_nn_quantize_compat(self):
        # Official MLX nn.quantize compatibility
        m = KAN([8, 16, 2], grid_size=5)
        x = mx.random.normal((4, 8))
        nn.quantize(m, group_size=32, bits=8)
        y = m(x)
        mx.eval(y)
        self.assertEqual(y.shape, (4, 2))

    def test_model_size_compression(self):
        m = FastKAN([64, 128, 64], num_grids=8)
        s_fp32 = get_model_size(m)
        to_int8(m, group_size=64)
        s_int8 = get_model_size(m)

        # Expect ~3.4x memory reduction for INT8
        compression = s_fp32["total_bytes"] / s_int8["total_bytes"]
        self.assertGreater(compression, 3.0)

    # -----------------------------------------------------------------------
    # 13. Advanced Optimizations: Checkpointing, Pruning & Symbolic
    # -----------------------------------------------------------------------
    def test_checkpoint_kan(self):
        m = FastKAN([8, 16, 8, 4], num_grids=6)
        m_ckpt = checkpoint_kan(m)

        x = mx.random.normal((4, 8))
        y_target = mx.random.normal((4, 4))

        # Check forward pass matches
        y1 = m(x)
        y2 = m_ckpt(x)
        mx.eval(y1, y2)
        self.assertTrue(mx.allclose(y1, y2, atol=1e-5))

        # Check gradient computation through checkpointing
        loss_fn = lambda model, a, b: mx.mean((model(a) - b) ** 2)
        loss, grads = nn.value_and_grad(m_ckpt, loss_fn)(m_ckpt, x, y_target)
        mx.eval(loss, grads)
        self.assertFalse(math.isnan(loss.item()))

    def test_structural_pruning(self):
        m = FastKAN([4, 16, 2], num_grids=6)
        # Artificially zero out neurons 4..15 to test exact pruning
        mask = mx.zeros((16,))
        mask = mask.at[:4].add(1.0)
        m.layers[0].base_weight = m.layers[0].base_weight * mask[:, None]
        m.layers[0].spline_weight = m.layers[0].spline_weight * mask[:, None]
        m.layers[1].base_weight = m.layers[1].base_weight * mask[None, :]

        compact_model, stats = prune(m, threshold=0.01, min_active=1)
        self.assertEqual(stats["new_dims"], [4, 4, 2])
        self.assertEqual(stats["pruned_neurons"], 12)
        self.assertAlmostEqual(stats["percent_neurons_pruned"], 75.0)

        # Check execution of compacted model
        x = mx.random.normal((6, 4))
        y = compact_model(x)
        mx.eval(y)
        self.assertEqual(y.shape, (6, 2))

    def test_symbolic_extraction(self):
        m = FastKAN([2, 1], num_grids=6)
        sym = to_symbolic(m, sample_points=100)

        self.assertIsInstance(sym, SymbolicKAN)
        formula = sym.formula()
        latex = sym.latex()
        self.assertIn("y0 =", formula)
        self.assertIn("x0", formula)
        self.assertIn("y_{0} =", latex)

        # Evaluate symbolic model on sample inputs
        x = mx.random.uniform(-0.8, 0.8, (8, 2))
        y_sym = sym(x)
        mx.eval(y_sym)
        self.assertEqual(y_sym.shape, (8, 1))

    # -----------------------------------------------------------------------
    # 7. RationalKAN & C Export Tests
    # -----------------------------------------------------------------------
    def test_rational_kan(self):
        # Multi-layer RationalKAN
        m = RationalKAN([4, 8, 2], p_degree=3, q_degree=2)
        x2d = mx.random.uniform(-0.9, 0.9, (16, 4))
        out2d = m(x2d)
        mx.eval(out2d)
        self.assertEqual(out2d.shape, (16, 2))

        # 3D input shape support
        x3d = mx.random.uniform(-0.9, 0.9, (4, 6, 4))
        out3d = m(x3d)
        mx.eval(out3d)
        self.assertEqual(out3d.shape, (4, 6, 2))

        # Gradient update check
        opt = optim.Adam(0.01)
        step = build_train_step(m, opt, lambda model, a, b: mx.mean((model(a) - b) ** 2))
        target = mx.ones((16, 2))
        loss = step(x2d, target)
        mx.eval(loss)
        self.assertGreater(loss.item(), 0.0)

        # Symbolic extraction on RationalKAN
        sym = to_symbolic(m, sample_points=100)
        self.assertIsInstance(sym, SymbolicKAN)
        self.assertIn("y0 =", sym.formula())

    def test_symbolic_c_export_and_compilation(self):
        import subprocess
        m = FastKAN([2, 1], num_grids=6)
        sym = to_symbolic(m, sample_points=100)

        # 1. Test C code and Header generation string
        c_code = sym.to_c_code(func_name="kan_model_eval")
        self.assertIn("void kan_model_eval", c_code)
        self.assertIn("y[0] =", c_code)

        c_header = sym.to_c_header(guard="MY_KAN_H", func_name="kan_model_eval")
        self.assertIn("#ifndef MY_KAN_H", c_header)
        self.assertIn("#include <math.h>", c_header)

        # 2. Test writing to file and compiling with system C compiler
        with tempfile.TemporaryDirectory() as tmpdir:
            header_path = os.path.join(tmpdir, "kan_model.h")
            sym.export_c(header_path, guard="TEST_KAN_H", func_name="kan_predict")
            self.assertTrue(os.path.exists(header_path))

            # Create a test runner C program
            main_c_path = os.path.join(tmpdir, "main.c")
            main_c_code = """
            #include <stdio.h>
            #include "kan_model.h"

            int main() {
                float x[2] = {0.5f, -0.25f};
                float y[1] = {0.0f};
                kan_predict(x, y);
                printf("OUT: %f\\n", y[0]);
                return 0;
            }
            """
            with open(main_c_path, "w") as f:
                f.write(main_c_code)

            # Compile with clang/gcc
            bin_path = os.path.join(tmpdir, "test_kan_bin")
            cc = os.environ.get("CC", "clang")
            res = subprocess.run([cc, "-O2", main_c_path, "-o", bin_path, "-lm"], capture_output=True, text=True)
            if res.returncode == 0:
                run_res = subprocess.run([bin_path], capture_output=True, text=True)
                self.assertEqual(run_res.returncode, 0)
                self.assertIn("OUT:", run_res.stdout)
                c_out = float(run_res.stdout.strip().split("OUT:")[1])

                # Compare with Python SymbolicKAN output on same input
                py_out = sym(mx.array([[0.5, -0.25]])).item()
                self.assertAlmostEqual(c_out, py_out, places=4)


if __name__ == "__main__":
    unittest.main()
