"""
Unit tests for custom Metal Shading Language (MSL) GPU kernels in MLX.
Tests numerical accuracy against compiled reference implementations.
"""

import unittest
import numpy as np
import mlx.core as mx

from efficient_kan.metal_kernels import (
    is_metal_available,
    metal_cheby_basis,
    metal_rbf_basis,
    metal_relu_basis,
    metal_bspline_basis,
    metal_wav_basis,
    metal_fourier_basis,
    metal_jacobi_basis,
)
from efficient_kan.cheby_kan import compute_cheby_basis
from efficient_kan.fast_kan import compute_rbf_basis
from efficient_kan.relu_kan import compute_relu_tent_basis
from efficient_kan.wav_kan import compute_wavelet_basis
from efficient_kan.fourier_kan import compute_fourier_basis
from efficient_kan.jacobi_kan import compute_jacobi_basis
from kan import compute_b_splines, KANLinear
from efficient_kan.wav_kan import WavKANLinear
from efficient_kan.fourier_kan import FourierKANLinear
from efficient_kan.jacobi_kan import JacobiKANLinear


class TestMLXMetalKernels(unittest.TestCase):
    def setUp(self):
        if not is_metal_available():
            self.skipTest("Metal GPU is not available.")

    def test_bspline_metal_kernel(self):
        N, D, grid_size = 64, 16, 5
        x = mx.random.uniform(-0.95, 0.95, (N, D))
        out_metal = metal_bspline_basis(x, -1.0, 1.0, grid_size)
        mx.eval(out_metal)

        self.assertEqual(out_metal.shape, (N, D, grid_size + 3))
        # Partition of unity test: sum across bases should be 1.0
        sums = mx.sum(out_metal, axis=-1)
        mx.eval(sums)
        diff = np.max(np.abs(np.array(sums) - 1.0))
        self.assertLess(diff, 1e-5, f"B-spline partition of unity failed, diff={diff}")

        # Test KANLinear with use_metal_kernel=True
        layer_metal = KANLinear(D, 32, grid_size=grid_size, use_metal_kernel=True)
        y = layer_metal(x)
        mx.eval(y)
        self.assertEqual(y.shape, (N, 32))
        self.assertFalse(np.isnan(np.array(y)).any())

    def test_wavelet_metal_kernel(self):
        N, D, K = 32, 8, 4
        x = mx.random.uniform(-1.0, 1.0, (N, D))
        trans = mx.random.uniform(-1.0, 1.0, (D, K))
        scale = mx.random.uniform(0.5, 1.5, (D, K))

        for w_type in ["mexican_hat", "morlet", "dog"]:
            ref = compute_wavelet_basis(x, trans, scale, wavelet_type=w_type)
            metal_res = metal_wav_basis(x, trans, scale, wavelet_type=w_type)
            mx.eval(ref, metal_res)
            diff = np.max(np.abs(np.array(ref) - np.array(metal_res)))
            self.assertLess(diff, 1e-4, f"Wavelet {w_type} discrepancy: {diff}")

        # Test WavKANLinear with use_metal_kernel=True
        layer = WavKANLinear(D, 16, num_wavelets=K, use_metal_kernel=True)
        y = layer(x)
        mx.eval(y)
        self.assertEqual(y.shape, (N, 16))
        self.assertFalse(np.isnan(np.array(y)).any())

    def test_fourier_metal_kernel(self):
        N, D = 32, 8
        freqs = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        x = mx.random.uniform(-1.0, 1.0, (N, D))

        ref = compute_fourier_basis(x, freqs)
        metal_res = metal_fourier_basis(x, freqs)
        mx.eval(ref, metal_res)
        diff = np.max(np.abs(np.array(ref) - np.array(metal_res)))
        self.assertLess(diff, 1e-4, f"Fourier discrepancy: {diff}")

        # Test FourierKANLinear with use_metal_kernel=True
        layer = FourierKANLinear(D, 16, num_frequencies=4, use_metal_kernel=True)
        y = layer(x)
        mx.eval(y)
        self.assertEqual(y.shape, (N, 16))
        self.assertFalse(np.isnan(np.array(y)).any())

    def test_jacobi_metal_kernel(self):
        N, D, deg = 32, 8, 4
        x = mx.random.uniform(-0.95, 0.95, (N, D))
        alpha, beta = 0.5, -0.2

        ref = compute_jacobi_basis(x, deg, alpha=alpha, beta=beta)
        metal_res = metal_jacobi_basis(x, deg, alpha=alpha, beta=beta)
        mx.eval(ref, metal_res)
        diff = np.max(np.abs(np.array(ref) - np.array(metal_res)))
        self.assertLess(diff, 1e-4, f"Jacobi discrepancy: {diff}")

        # Test JacobiKANLinear with use_metal_kernel=True
        layer = JacobiKANLinear(D, 16, degree=deg, alpha=alpha, beta=beta, use_metal_kernel=True)
        y = layer(x)
        mx.eval(y)
        self.assertEqual(y.shape, (N, 16))
        self.assertFalse(np.isnan(np.array(y)).any())

    def test_nd_shapes_and_caching(self):
        """Verify that MSL basis kernels support 3D/ND inputs (B, T, D) and leverage scalar cache."""
        B, T, D = 4, 16, 8
        x = mx.random.uniform(-0.9, 0.9, (B, T, D))

        # Test Chebyshev 3D
        out_cheby = metal_cheby_basis(x, degree=4)
        mx.eval(out_cheby)
        self.assertEqual(out_cheby.shape, (B, T, D, 4))

        # Test RBF 3D
        centers = mx.random.uniform(-1.0, 1.0, (D, 6))
        out_rbf = metal_rbf_basis(x, centers, inv_h=1.5)
        mx.eval(out_rbf)
        self.assertEqual(out_rbf.shape, (B, T, D, 6))

        # Test B-spline 3D
        out_bspline = metal_bspline_basis(x, -1.0, 1.0, grid_size=5)
        mx.eval(out_bspline)
        self.assertEqual(out_bspline.shape, (B, T, D, 8))

    def test_fastkan_grid_adaptation(self):
        """Verify FastKAN adaptive grid updates."""
        from efficient_kan.fast_kan import FastKANLinear, FastKAN

        layer = FastKANLinear(in_features=4, out_features=8, num_grids=6)
        x_dense = mx.random.uniform(2.0, 5.0, (32, 4))
        orig_centers = mx.array(layer.centers)
        layer.update_grid(x_dense)
        mx.eval(layer.centers)

        # Centers should now be centered around [2, 5] instead of default [-1, 1]
        self.assertGreater(float(mx.min(layer.centers)), 1.5)
        self.assertLessEqual(float(mx.max(layer.centers)), 5.5)

        # Multi-layer FastKAN sequential update
        model = FastKAN([4, 8, 2], num_grids=6)
        y = model(x_dense, update_grid=True)
        mx.eval(y)
        self.assertEqual(y.shape, (32, 2))


if __name__ == "__main__":
    unittest.main()
