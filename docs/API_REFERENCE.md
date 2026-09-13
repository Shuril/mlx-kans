# mlx-KANs API Reference

Complete documentation for all classes, layers, functions, and utilities in **mlx-KANs**.

All components can be imported from `mlx_kans`:
```python
import mlx_kans as kans
# or
from mlx_kans import FastKAN, ReLUKAN, LowRankKAN, WavKAN, ChebyKAN, FourierKAN, JacobiKAN, MultKAN, KAN
```

---

## Table of Contents
1. [Core Models & Layers](#core-models--layers)
   - [FastKAN / FastKANLinear](#fastkan--fastkanlinear)
   - [ReLUKAN / ReLUKANLinear](#relukan--relukanlinear)
   - [LowRankKAN / LowRankKANLinear](#lowrankkan--lowrankkanlinear)
   - [WavKAN / WavKANLinear](#wavkan--wavkanlinear)
   - [ChebyKAN / ChebyKANLinear](#chebykan--chebykanlinear)
   - [FourierKAN / FourierKANLinear](#fourierkan--fourierkanlinear)
   - [JacobiKAN / JacobiKANLinear](#jacobikan--jacobikanlinear)
   - [MultKAN / MultKANLinear](#multkan--multkanlinear)
   - [KAN / KANLinear (B-spline)](#kan--kanlinear-b-spline)
2. [Metal Kernels (MSL)](#metal-kernels-msl)
3. [Optimization & Training Utilities](#optimization--training-utilities)
4. [Native INT8 & INT4 Quantization](#native-int8--int4-quantization)

---

## Core Models & Layers

All multi-layer KAN models inherit from `mlx.nn.Module`, take a list of layer dimensions `layers_hidden` (e.g. `[2, 16, 1]`), and support:
- `__call__(x: mx.array) -> mx.array`: forward pass supporting arbitrary batch shapes `(*batch_dims, in_features)`.
- `forward(x: mx.array) -> mx.array`: PyTorch-compatible alias for `__call__`.
- `regularization_loss(regularize_activation=1.0, regularize_entropy=1.0) -> mx.array`: computes L1 sparsity and entropy regularization penalties.
- `save_weights(path: str)` / `load_weights(path: str)`: saves/loads model weights in `.safetensors` or `.npz` format.

---

### FastKAN / FastKANLinear
Uses Gaussian Radial Basis Functions: $\phi_i(x) = \exp\left(-\left(\frac{x - c_i}{h}\right)^2\right)$.

#### `FastKANLinear(in_features, out_features, num_grids=8, grid_range=(-1.0, 1.0), base_activation=nn.silu, use_metal_kernel=False, bias=False)`
- **Parameters**:
  - `in_features` (*int*): Input dimension.
  - `out_features` (*int*): Output dimension.
  - `num_grids` (*int*, default `8`): Number of Gaussian RBF centers along each input dimension.
  - `grid_range` (*tuple[float, float]*, default `(-1.0, 1.0)`): Domain interval for RBF centers.
  - `base_activation` (*Callable*, default `nn.silu`): Activation function applied to the residual linear base branch.
  - `use_metal_kernel` (*bool*, default `False`): If `True`, runs fused Metal Shading Language kernel for basis evaluation.
  - `bias` (*bool*, default `False`): Whether to include an additive learnable bias term.

#### `FastKAN(layers_hidden, num_grids=8, grid_range=(-1.0, 1.0), base_activation=nn.silu, use_metal_kernel=False, bias=False)`
- Sequence of `FastKANLinear` layers connecting dimensions in `layers_hidden`.

---

### ReLUKAN / ReLUKANLinear
Uses piecewise-linear tent functions formed by Rectified Linear Units: $\phi_i(x) = \max\left(0, 1 - \frac{|x - c_i|}{h}\right)$. Zero transcendental operations.

#### `ReLUKANLinear(in_features, out_features, num_grids=8, grid_range=(-1.0, 1.0), base_activation=nn.silu, use_metal_kernel=False, bias=False)`
- **Parameters**:
  - `in_features` (*int*): Input dimension.
  - `out_features` (*int*): Output dimension.
  - `num_grids` (*int*, default `8`): Number of knot centers.
  - `grid_range` (*tuple[float, float]*, default `(-1.0, 1.0)`): Interval range.
  - `base_activation` (*Callable*, default `nn.silu`): Base branch activation.
  - `use_metal_kernel` (*bool*, default `False`): Run custom tent-ReLU Metal kernel.
  - `bias` (*bool*, default `False`): Add learnable bias.

#### `ReLUKAN(layers_hidden, num_grids=8, ...)`
- Multi-layer stack of `ReLUKANLinear`.

---

### LowRankKAN / LowRankKANLinear
Decomposes the spline weight tensor via a low-rank bottleneck $r \ll \min(d_{in}, d_{out})$:
$$W_{spline} \approx U \cdot V, \quad V \in \mathbb{R}^{r \times (d_{in} \cdot G)}, \; U \in \mathbb{R}^{d_{out} \times r}$$
Reduces parameter complexity from $\mathcal{O}(d_{out} \cdot d_{in} \cdot G)$ to $\mathcal{O}((d_{out} + d_{in} \cdot G) \cdot r)$.

#### `LowRankKANLinear(in_features, out_features, rank=8, num_grids=8, grid_range=(-1.0, 1.0), base_activation=nn.silu, bias=False)`
- **Parameters**:
  - `in_features` (*int*): Input dimension.
  - `out_features` (*int*): Output dimension.
  - `rank` (*int*, default `8`): Bottleneck rank $r$.
  - `num_grids` (*int*, default `8`): Number of basis centers.
  - `base_activation` (*Callable*, default `nn.silu`): Base branch activation.
  - `bias` (*bool*, default `False`): Add learnable bias.

#### `LowRankKAN(layers_hidden, rank=8, num_grids=8, ...)`
- Multi-layer stack of `LowRankKANLinear`.

---

### WavKAN / WavKANLinear
Continuous multiresolution wavelet basis:
$$z = \frac{x - t}{s}, \quad \psi(z) = (1 - z^2) \exp\left(-\frac{z^2}{2}\right) \quad \text{(Mexican Hat)}$$
or $\psi(z) = \cos(5z) \exp(-z^2/2)$ (Morlet).

#### `WavKANLinear(in_features, out_features, num_wavelets=8, wavelet_type="mexican_hat", learnable_scales=True, base_activation=nn.silu, bias=False)`
- **Parameters**:
  - `num_wavelets` (*int*, default `8`): Number of wavelet translations/scales per dimension.
  - `wavelet_type` (*str*, default `"mexican_hat"`): `"mexican_hat"`, `"morlet"`, or `"dog"`.
  - `learnable_scales` (*bool*, default `True`): If `True`, optimizes translation $t$ and scale $s$ via gradient descent.

#### `WavKAN(layers_hidden, num_wavelets=8, wavelet_type="mexican_hat", ...)`
- Multi-layer stack of `WavKANLinear`.

---

### ChebyKAN / ChebyKANLinear
Orthogonal Chebyshev polynomials of the 1st kind on $[-1, 1]$:
$$T_0(x) = 1, \quad T_1(x) = x, \quad T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)$$

#### `ChebyKANLinear(in_features, out_features, degree=4, base_activation=nn.silu, use_metal_kernel=False, bias=False)`
- **Parameters**:
  - `degree` (*int*, default `4`): Number of polynomial terms ($T_0, \dots, T_{degree-1}$).
  - `use_metal_kernel` (*bool*, default `False`): Use custom MSL recurrence kernel.

#### `ChebyKAN(layers_hidden, degree=4, ...)`
- Multi-layer stack of `ChebyKANLinear`.

---

### FourierKAN / FourierKANLinear
Harmonic Fourier series expansion:
$$\mathcal{B}(x) = [1, \cos(\pi x), \sin(\pi x), \dots, \cos(K \pi x), \sin(K \pi x)]$$

#### `FourierKANLinear(in_features, out_features, num_frequencies=4, base_activation=nn.silu, bias=False)`
- **Parameters**:
  - `num_frequencies` (*int*, default `4`): Number of harmonics $K$. Generates $2K + 1$ basis terms per input dimension.

#### `FourierKAN(layers_hidden, num_frequencies=4, ...)`
- Multi-layer stack of `FourierKANLinear`.

---

### JacobiKAN / JacobiKANLinear
Orthogonal Jacobi polynomials $P_n^{(\alpha, \beta)}(x)$ with weighting $(1-x)^\alpha (1+x)^\beta$.
Special cases:
- $\alpha = 0, \beta = 0$: Legendre polynomials
- $\alpha = 0.5, \beta = 0.5$: Chebyshev polynomials of the 2nd kind
- $\alpha = -0.5, \beta = -0.5$: Chebyshev polynomials of the 1st kind

#### `JacobiKANLinear(in_features, out_features, degree=4, alpha=0.0, beta=0.0, base_activation=nn.silu, bias=False)`
- **Parameters**:
  - `degree` (*int*, default `4`): Number of polynomial terms.
  - `alpha` (*float*, default `0.0`): First Jacobi exponent $> -1$.
  - `beta` (*float*, default `0.0`): Second Jacobi exponent $> -1$.

#### `JacobiKAN(layers_hidden, degree=4, alpha=0.0, beta=0.0, ...)`
- Multi-layer stack of `JacobiKANLinear`.

---

### MultKAN / MultKANLinear
KAN 2.0 architecture with explicit multiplicative interaction nodes $u \cdot v$.

#### `MultKANLinear(in_features, out_features, num_mult=None, num_grids=8, base_activation=nn.silu, bias=False)`
- **Parameters**:
  - `num_mult` (*int | None*, default `None`): Number of multiplication nodes. If `None`, defaults to $\lfloor d_{out} / 2 \rfloor$. Total output channels = $n_{add} + n_{mult} = d_{out}$.

#### `MultKAN(layers_hidden, num_grids=8, ...)`
- Multi-layer stack of `MultKANLinear`.

---

### KAN / KANLinear (B-spline)
Original efficient Kolmogorov-Arnold network with cubic B-splines and adaptive knot relocation.

#### `KANLinear(in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=nn.silu, grid_eps=0.02, grid_range=(-1.0, 1.0), bias=False)`
- **Methods**:
  - `update_grid(x: mx.array, margin: float = 0.01)`: Adaptively redistributes knot grid according to input quantiles while preserving existing function outputs via CPU-stream pseudo-inverse.

#### `KAN(layers_hidden, grid_size=5, spline_order=3, ...)`
- Multi-layer stack of `KANLinear`.
- `__call__(x: mx.array, update_grid: bool = False)`: If `update_grid=True`, updates grids across all layers sequentially.

---

## Metal Kernels (MSL)

Located in `mlx_kans.metal_kernels`:

### `metal_rbf_basis(x: mx.array, centers: mx.array, inv_h: float) -> mx.array`
- Custom MSL kernel computing Gaussian RBF values in 3D grid $(G, D, N)$.

### `metal_cheby_basis(x: mx.array, degree: int) -> mx.array`
- Custom MSL kernel computing all $T_0(x), \dots, T_{degree-1}(x)$ in thread registers with clamping to $[-1, 1]$.

### `metal_relu_basis(x: mx.array, centers: mx.array, inv_h: float) -> mx.array`
- Custom MSL kernel computing linear tent basis functions via hardware ALU instructions.

### `is_metal_available() -> bool`
- Returns `True` if running on Apple Silicon GPU.

---

## Optimization & Training Utilities

Located in `mlx_kans.utils`:

### `build_train_step(model, optimizer, loss_fn)`
Compiles the full training cycle into an optimized Metal execution graph:
```python
from mlx_kans import FastKAN, build_train_step
import mlx.optimizers as optim

model = FastKAN([2, 16, 1])
optimizer = optim.Adam(learning_rate=0.01)

def loss_fn(m, x, y):
    return mx.mean((m(x) - y) ** 2)

# Returns compiled callable: train_step(x, y) -> loss
train_step = build_train_step(model, optimizer, loss_fn)
```

### `count_parameters(model: nn.Module) -> dict[str, int]`
Returns a dictionary:
```python
{
    "trainable": 504,
    "frozen": 14,
    "total": 518
}
```

### `to_fp16(model: nn.Module) -> nn.Module`
In-place casts all floating-point parameters to `mx.float16` for maximum Apple Silicon GPU throughput.

### `to_bf16(model: nn.Module) -> nn.Module`
In-place casts all floating-point parameters to `mx.bfloat16`.

---

## Native INT8 & INT4 Quantization

Located in `mlx_kans.quantized`:

### `to_int8(model: nn.Module, group_size: int = 64, mode: str = "affine", **kwargs) -> nn.Module`
Quantizes all KAN layers in `model` to 8-bit integers (`uint32` packing 4 INT8 values per word) with group scales and biases. Matrix multiplications execute natively on Apple Silicon GPU without dequantization.
- **Parameters**:
  - `model` (*nn.Module*): Any model containing KAN layers.
  - `group_size` (*int*, default `64`): Quantization block size (32, 64, or 128).
  - `mode` (*str*, default `"affine"`): Quantization scheme (`"affine"` or `"symmetric"`).

### `to_int4(model: nn.Module, group_size: int = 64, mode: str = "affine", **kwargs) -> nn.Module`
Quantizes all KAN layers in `model` to 4-bit integers (`uint32` packing 8 INT4 values per word) with group scales and biases. Achieves up to 6.8x memory reduction for ultra-compact deployments.

### `to_fp8(model: nn.Module, group_size: int = 32, allow_emulation: bool = False, **kwargs) -> nn.Module`
Quantizes all KAN layers in `model` to 8-bit floating point (`mxfp8` / E4M3 with E8M0 scale per group of 32).
- **Hardware Requirement**: Native hardware FP8 execution units require Apple Silicon M4 / M5 or newer (Apple GPU Family 9+).
- **Error Behavior**: On earlier chips (M1, M2, M3), calling `to_fp8` raises `HardwareNotSupportedError` unless `allow_emulation=True` is explicitly specified.

### `is_fp8_hardware_supported() -> bool`
Returns `True` if the current Apple Silicon GPU possesses physical hardware tensor/ALU execution units for FP8 (M4 / M5+). Returns `False` on M1, M2, M3.

### `get_chip_name() -> str`
Returns the detected marketing name of the active Apple Silicon chip (e.g. `'Apple M1'`, `'Apple M4 Pro'`).

### `HardwareNotSupportedError(RuntimeError)`
Exception raised when an operation requires hardware features not physically present in the GPU silicon of the host machine.

### `quantize(model: nn.Module, group_size: int = 64, bits: int = 8, mode: str = "affine", allow_emulation: bool = False, **kwargs) -> nn.Module`
General quantization driver supporting any custom bit-width (4 or 8), group size, and mode (`"affine"` or `"mxfp8"` / `"fp8"`).

### `get_model_size(model: nn.Module) -> dict`
Returns memory statistics for the model:
```python
{
    "total_params": 592896,
    "total_bytes": 671744,
    "mb": 0.640625,
    "summary": "0.641 MB (671,744 bytes, 592,896 elements)"
}
```

### `QuantizedWeight(weight: mx.array, group_size: int = 64, bits: int = 8, mode: str = "affine")`
Low-level wrapper encapsulating packed `uint32` weight matrices, per-group scales, and biases. Automatically pads non-divisible dimensions. Forward pass invokes `mx.quantized_matmul(..., transpose=True)`.

### Layer Classes:
- `QuantizedKANLinear`
- `QuantizedFastKANLinear`
- `QuantizedReLUKANLinear`
- `QuantizedChebyKANLinear`
- `QuantizedWavKANLinear`
- `QuantizedFourierKANLinear`
- `QuantizedJacobiKANLinear`
- `QuantizedLowRankKANLinear`
- `QuantizedMultKANLinear`

Each quantized layer provides a `.from_layer(layer, group_size, bits, mode)` factory method and conforms to standard MLX `to_quantized(...)` interface.
