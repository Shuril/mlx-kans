# mlx-KANs

**mlx-KANs**: библиотека сетей Колмогорова-Арнольда (**Kolmogorov-Arnold Networks, KAN**) на базе [Apple MLX](https://github.com/ml-explore/mlx), аппаратно оптимизированная для процессоров **Apple Silicon (серии M1 / M2 / M3 / M4 / Pro / Max / Ultra)** с поддержкой кастомных Metal-шейдеров (MSL), смешанной точности FP16/BF16 и JIT-компиляции.

---

## Установка

```bash
# Установка в текущее окружение
pip install -e .

# Либо через uv
uv pip install -e .
```

Импорт в Python:
```python
import mlx_kans as kans
# или:
from mlx_kans import FastKAN, ReLUKAN, WavKAN, ChebyKAN, LowRankKAN, KAN
```

---

## Поддерживаемые архитектуры KAN

В состав **mlx-KANs** входят все современные оптимизированные варианты KAN:

| Архитектура | Базис | Преимущество | Рекомендуемое применение |
|---|---|---|---|
| **`ReLUKAN`** | Piecewise-linear (Tent) | Ноль трансцендентных функций, 1.17M семплов/с | Высокочастотные сигналы, INT8/FP16 квантование |
| **`FastKAN`** | Гауссовы RBF ($\exp(-d^2/h^2)$) | В 2.5-3.5x быстрее B-сплайнов, гладкие градиенты | Замена MLP, компьютерное зрение |
| **`LowRankKAN`** | LoRA / Bottleneck факторизация | **В 5-10x меньше параметров**, 1.23M семплов/с | Большие размерности, многомерные входы |
| **`WavKAN`** | Вейвлеты (Mexican Hat, Morlet, DOG) | Мультимасштабная локализация, наименьшая MSE | Временные ряды, аудио, защита от забывания |
| **`ChebyKAN`** | Полиномы Чебышёва 1-го рода | Минимаксная оптимальность на $[-1, 1]$, нет сетки | Физика, аппроксимация гладких полей |
| **`FourierKAN`** | Тригонометрические гармоники ($\cos, \sin$) | Отличный спектральный анализ | Периодические процессы, PINNs, PDEs |
| **`RationalKAN`** | Рациональные функции Паде-Чебышёва ($P/Q$) | Точное моделирование полюсов и сингулярностей | Пограничные слои, жесткие ДУ |
| **`JacobiKAN`** | Ортогональные полиномы Якоби ($\alpha, \beta$) | Обобщение Лежандра и Гегенбауэра | Краевые задачи, математическая физика |
| **`MultKAN` (2.0)** | Сплайны/RBF + узлы умножения ($u \cdot v$) | Точное представление произведений | Аналитическая регрессия, законы сохранения |
| **`KAN`** | B-сплайны де Боора | Классический KAN с адаптивной сеткой | Интерпретируемость, классический KAN |
| **`HybridKAN`** | MSL Fused + MLX JIT | **3.36x ускорение (10.58M семплов/с)**, 0 промежуточных аллокаций | Продакшн-инференс, большие батчи ($B \ge 1024$) |

---

## Механизмы оптимизации под Apple Silicon (MPS / Metal)

1. **Собственные Metal-кернелы (`mlx_kans.metal_kernels`)**:
   - Шейдеры на **Metal Shading Language (MSL)** через `mx.fast.metal_kernel` для RBF, кусочно-линейного ReLU и полиномов Чебышёва. Исполняются непосредственно в регистрах потоков GPU.
2. **Zero-allocation MPS GEMM**:
   - Вычисления сплайнов сведены к стандартному GEMM, исполняемому на аппаратно ускоренных блоках Metal Performance Shaders.
3. **Unified Memory Stream Orchestration**:
   - Операции псевдообращения матриц для сеток выполняются на потоке `mx.cpu` с **нулевыми затратами копирования** благодаря объединённой памяти macOS.
4. **Низкоранговая факторизация (`LowRankKAN`)**:
   - Сжатие весов сплайнов $W \approx U \cdot V$, позволяющее масштабировать KAN на высокие размерности без взрыва памяти.
5. **JIT-компиляция графа обучения (`build_train_step`)**:
   - Компилирует прямой проход, вычисление градиентов и шаг оптимизатора в единый монолитный граф Metal.
6. **Смешанная точность (`to_fp16`, `to_bf16`)**:
   - Аппаратное ускорение на ядрах Apple GPU и двукратная экономия полосы пропускания памяти.
7. **Встроенный набор оптимизаторов MLX (`from mlx_kans import Muon, AdamW`)**:
   - Прямой экспорт современных оптимизаторов: `Muon` (ортогонализация шага полиномами Ньютона-Шульца 5-го порядка), `AdamW` (с разделенным затуханием весов), `Lion` (знаковый импульс), `RMSprop`, `Adam` и `SGD`.

---

## Изо-параметрический стресс-тест (Одинаковый бюджет весов $\pm 1\%$)

Все 11 моделей были протестированы при абсолютно одинаковом числе параметров:
- **Задачи 1 и 2 (2D входы):** бюджет $\approx 500$ параметров (все модели имеют 498-504 параметров).
- **Задача 3 (4D входы, физ. закон):** бюджет $\approx 600$ параметров.
- **Задача 4 (8D входы, многомерность):** бюджет $\approx 1000$ параметров.

Результаты Test MSE на Metal GPU (`Device(gpu, 0)`), 150 эпох:

| Модель | Task 1 (High-Freq, 2D) | Task 2 (Non-Smooth, 2D) | Task 3 (Physics, 4D) | Task 4 (Target, 8D) | Время / эпоха |
|---|---|---|---|---|---|
| **MLP (Baseline)** | 0.42668 | 0.04385 | **0.00055** | **0.01713** | **0.74 ms** |
| **`ReLUKAN` (Tent)** | **0.23316** | 0.02744 | 0.00679 | 0.36655 | **0.82 ms** |
| **`FastKAN` (RBF)** | **0.23344** | 0.02387 | 0.02616 | 0.25177 | **0.89 ms** |
| **`WavKAN` (MexHat)**| 0.29318 | 0.02515 | 0.00222 | 0.23987 | 1.09 ms |
| **`LowRankKAN`** | 0.32824 | **0.01555** | **0.00174** | **0.02772** | 1.16 ms |
| **`MultKAN` (2.0)** | 0.37980 | **0.01930** | 0.01482 | 0.18123 | 0.95 ms |
| **`JacobiKAN`** | 0.37621 | 0.03450 | 0.02681 | 0.24693 | 1.03 ms |
| **`ChebyKAN`** | 0.39325 | 0.05112 | 0.04086 | 0.25154 | 1.35 ms |
| **`B-Spline KAN`** | 0.42266 | **0.02300** | **0.00017** | **0.08623** | 2.40 ms |
| **`FourierKAN`** | 0.91594 | 1.47926 | 0.25044 | 0.38723 | 0.94 ms |

---

### Сравнение 3 библиотек на GPU: MLX vs metal-KANs (v0.3.1) vs slang-KANs (v0.2.0)

Полный физический бенчмарк всех 10 архитектур на **Apple Silicon GPU**, слой `64 -> 64`:

| Архитектура | Батч | MLX (ms) | metal-KANs (Pure Metal AMX) | slang-KANs (Shared-Memory GEMM) | Победитель (Ускорение) |
|---|---|---|---|---|---|
| **ChebyKAN** | 128 | 0.387 ms | 0.295 ms | **0.226 ms** | **slang-KANs (1.30x)** |
| | 1024 | 0.526 ms | 0.372 ms | **0.324 ms** | **slang-KANs (1.15x)** |
| | 4096 | 1.380 ms | **0.484 ms** | 0.885 ms | **metal-KANs (1.83x)** |
| **BSplineKAN** | 128 | 0.354 ms | 0.258 ms | **0.136 ms** | **slang-KANs (1.90x)** |
| | 1024 | 0.835 ms | **0.404 ms** | 0.541 ms | **metal-KANs (1.34x)** |
| | 4096 | 2.190 ms | **0.826 ms** | 1.400 ms | **metal-KANs (1.70x)** |
| **FastKAN** | 128 | 0.270 ms | **0.254 ms** | 0.692 ms | **metal-KANs (1.06x)** |
| | 1024 | 0.398 ms | **0.352 ms** | 1.439 ms | **metal-KANs (1.13x)** |
| | 4096 | 1.475 ms | **1.006 ms** | 4.010 ms | **metal-KANs (1.47x)** |
| **WavKAN** | 128 | **0.354 ms** | 0.367 ms | 0.869 ms | **MLX (1.04x)** |
| | 1024 | 1.845 ms | **0.541 ms** | 1.941 ms | **metal-KANs (3.41x)** |
| | 4096 | 1.844 ms | **1.189 ms** | 4.132 ms | **metal-KANs (1.55x)** |
| **ReLUKAN** | 128 | **0.311 ms** | 0.364 ms | 0.852 ms | **MLX (1.17x)** |
| | 1024 | 0.542 ms | **0.511 ms** | 1.888 ms | **metal-KANs (1.06x)** |
| | 4096 | 1.254 ms | **1.062 ms** | 4.134 ms | **metal-KANs (1.18x)** |
| **FourierKAN** | 128 | **0.351 ms** | 0.384 ms | 0.883 ms | **MLX (1.09x)** |
| | 1024 | 0.975 ms | **0.605 ms** | 2.098 ms | **metal-KANs (1.61x)** |
| | 4096 | 2.385 ms | **0.868 ms** | 2.868 ms | **metal-KANs (2.75x)** |
| **JacobiKAN** | 128 | 0.405 ms | 0.294 ms | **0.144 ms** | **slang-KANs (2.04x)** |
| | 1024 | 0.632 ms | 0.415 ms | **0.325 ms** | **slang-KANs (1.28x)** |
| | 4096 | 1.413 ms | **0.515 ms** | 0.722 ms | **metal-KANs (1.40x)** |
| **RationalKAN**| 128 | 0.691 ms | 0.385 ms | **0.166 ms** | **slang-KANs (2.32x)** |
| | 1024 | 4.354 ms | 0.822 ms | **0.648 ms** | **slang-KANs (1.27x)** |
| | 4096 | 18.225 ms| **2.300 ms** | 2.468 ms | **metal-KANs (1.07x)** |
| **MultKAN** | 128 | 0.378 ms | **0.278 ms** | 0.783 ms | **metal-KANs (1.36x)** |
| | 1024 | 0.494 ms | **0.415 ms** | 3.738 ms | **metal-KANs (1.19x)** |
| | 4096 | 2.179 ms | **1.526 ms** | 7.734 ms | **metal-KANs (1.43x)** |
| **LowRankKAN** | 128 | **0.514 ms** | 0.546 ms | 0.811 ms | **MLX (1.06x)** |
| | 1024 | 0.866 ms | **0.612 ms** | 8.050 ms | **metal-KANs (1.41x)** |
| | 4096 | 1.722 ms | **1.101 ms** | 27.890 ms| **metal-KANs (1.56x)** |

*Репозитории библиотек*:
- **[metal-KANs (v0.3.1)](https://github.com/Shuril/metal-kans)**: Чистый Metal Shading Language (MSL) с аппаратным матричным умножением на сопроцессоре Apple AMX и векторными SIMD базисными кернелами.
- **[slang-KANs (v0.2.0)](https://github.com/Shuril/slang-kans)**: Кросс-платформенные Slang шейдеры с 16x16 shared-memory тайлингом GEMM и поддержкой Vulkan/Metal/CUDA.

---

## Гибридный движок: MLX + Прямой Metal Shading Language (MSL)

Для высоконагруженного инференса и больших батчей в библиотеку встроен **гибридный модуль** (`HybridChebyKAN`, `HybridFastKAN`, `HybridReLUKAN`, `HybridKAN`).

### Архитектурные преимущества:
Стандартные графы MLX вычисляют слой KAN в два этапа:
1. **Генерация базиса** ($[B, D_{\text{in}}, \text{degree}]$) с сохранением тензоров в промежуточную видеопамять.
2. **Линейная проекция GEMM** по материализованным базисам.

На больших батчах ($B \ge 1024$) аллокация промежуточных буферов занимает большую часть времени и расходует полосу пропускания unified memory.

**Гибридный слой автоматически маршрутизирует вызовы:**
- **Малые батчи ($B < 1024$)**: исполняются через **MLX JIT граф**, имеющий минимальные накладные расходы хоста.
- **Большие батчи ($B \ge 1024$)**: исполняются через **Direct Metal Fused шейдер** с нулевым копированием (zero-copy через указатели unified memory `newBufferWithBytesNoCopy`).

Прямой кернел Metal вычисляет базис в **аппаратных регистрах потоков и SRAM-памяти тредгруппы**, сразу накапливая результат в выходную матрицу: **0 байт промежуточных аллокаций VRAM**.

### Микро-бенчмарк на Apple M1 GPU ($64 \to 64$, степень 4):

| Размер батча ($B$) | MLX JIT Forward | Прямой Fused Metal | Ускорение | Промежуточные аллокации |
|---|---|---|---|---|
| **16** | **0.06 ms** (16 µs граф) | 0.08 ms | MLX быстрее | 0 (регистры) |
| **256** | 0.28 ms | **0.19 ms** | **1.47x** | 0 против 131 КБ |
| **1,024** | 0.74 ms | **0.32 ms** | **2.31x** | 0 против 524 КБ |
| **4,096** | 2.12 ms | **0.78 ms** | **2.72x** | 0 против 2.1 МБ |
| **16,384** | 5.21 ms (3.14M семплов/с) | **1.54 ms (10.58M семплов/с)** | **3.36x** | **0 МБ против 8.4 МБ** |

### Пример использования:
```python
import mlx.core as mx
import mlx.optimizers as optim
from mlx_kans import HybridChebyKAN, build_train_step

# Автоматическое переключение между MLX JIT и Direct Metal Fused
model = HybridChebyKAN(
    in_features=64, 
    out_features=64, 
    degree=4, 
    adaptive_threshold=1024
)

# Малый батч -> MLX JIT
x_small = mx.random.normal((32, 64))
y_small = model(x_small)

# Большой батч -> Direct Metal Fused MSL (10.58M семплов/с)
x_large = mx.random.normal((4096, 64))
y_large = model(x_large)

# Полная поддержка autograd и обучения MLX
optimizer = optim.Adam(learning_rate=1e-3)
def loss_fn(m, x, y):
    return mx.mean((m(x) - y) ** 2)

step = build_train_step(model, optimizer, loss_fn)
loss = step(x_small, mx.random.normal((32, 64)))
```

---

## Примеры использования

### 1. Быстрый старт с FastKAN
```python
import mlx.core as mx
from mlx_kans import FastKAN

# Создание FastKAN с топологией 4 -> 16 -> 2
model = FastKAN([4, 16, 2], num_grids=8)
x = mx.random.normal((32, 4))
y = model(x)
print(y.shape)  # (32, 2)
```

### 2. Обучение ReLUKAN с JIT-компиляцией
```python
import mlx.core as mx
import mlx.optimizers as optim
from mlx_kans import ReLUKAN, build_train_step

model = ReLUKAN([2, 21, 1], num_grids=7)
optimizer = optim.Adam(learning_rate=0.02)

def loss_fn(m, x, y):
    return mx.mean((m(x) - y) ** 2)

train_step = build_train_step(model, optimizer, loss_fn)

x = mx.random.uniform(-1, 1, (1000, 2))
y = mx.sin(8.0 * mx.pi * x[:, 0:1]) + (x[:, 1:2] ** 2)

for epoch in range(100):
    loss = train_step(x, y)
    mx.eval(model.parameters(), optimizer.state)
```

### 3. Нативное квантование весов в INT8 и INT4 на Metal GPU
```python
import mlx_kans as kans
import mlx.core as mx

# 1. Создание модели KAN
model = kans.FastKAN([128, 256, 128], num_grids=8)

# Проверка размера в FP32
print("FP32 размер:", kans.get_model_size(model)["summary"])
# -> 2.262 MB (592,896 параметров)

# 2. Квантование весов в INT8 нативно под Apple Silicon Metal
kans.to_int8(model, group_size=64)

# Проверка размера после квантования
print("INT8 размер:", kans.get_model_size(model)["summary"])
# -> 0.641 MB, сокращение памяти в 3.53 раза

# 3. Квантование в INT4 для большего сжатия памяти
kans.to_int4(model, group_size=64)
print("INT4 размер:", kans.get_model_size(model)["summary"])
# -> 0.364 MB, сжатие в 6.23 раза
```

### Матрица аппаратной поддержки (INT8 vs INT4):
- **INT8 / INT4 (`affine`)**: Аппаратно поддерживается **на всех поколениях Apple Silicon (M1, M2, M3, M4, M5+)**.
- Выполняет матричное умножение напрямую на GPU Metal без промежуточной деквантования в FP32.

---

## Продвинутые оптимизации памяти и деплоя

### 1. Градиентный чекпоинтинг (`checkpoint_kan`)
Экономит **~57% пикового VRAM** при обратном распространении ошибки в глубоких KAN на Unified Memory Apple Silicon. За счет отбрасывания промежуточных сплайн-базисов и их пересчета на backward pass устраняет троттлинг шины памяти и подкачку страниц (ускоряя обучение **до 7.4 раз на M1**):

```python
import mlx_kans as kans

# Оборачиваем любую глубокую KAN в чекпоинтинг активаций
model = kans.FastKAN([128] * 9, num_grids=8)
checkpointed_model = kans.checkpoint_kan(model)

# Обучаем стандартно через nn.value_and_grad
loss, grads = nn.value_and_grad(checkpointed_model, loss_fn)(checkpointed_model, x, y)
```

### 2. Структурный прунинг и компактизация узлов (`prune`)
Сети KAN обладают естественной разреженностью узлов при обучении с L1-регуляризацией. В отличие от MLP, требующих разреженных масок, неактивные нейроны KAN **физически вырезаются** из матриц весов, схлопывая размерности модели: сокращение параметров **до 87%** и ускорение инференса **в 3.3 раза**:

```python
# Вычисляем важность и удаляем нейроны с вкладом < 5% от максимального
compact_model, stats = kans.prune(trained_model, threshold=0.05, min_active=2)

print("Исходные размеры:", stats["orig_dims"])
print("Сжатые размеры:", stats["new_dims"])
print(f"Удалено {stats['pruned_neurons']} нейронов ({stats['percent_neurons_pruned']:.1f}%)")

# compact_model: физически уменьшенная модель FastKAN без накладных расходов
y = compact_model(x_test)
```

### 3. Точная символическая экстракция формул (`to_symbolic`)
Одномерные сплайновые ребра KAN сопоставляются с библиотекой аналитических функций ($x^2, \sin, \cos, \exp$, полиномы) методом наименьших квадратов ($R^2 > 0.90$). Сеть преобразуется в замкнутую математическую формулу:

```python
# Извлечение аналитической символической формулы
sym_kan = kans.to_symbolic(trained_model, r2_threshold=0.90)

# Печать человекочитаемого математического уравнения
print(sym_kan.formula())
# Вывод: y0 = 0.998*x0^2 + 1.001*sin(pi*x1)

# Экспорт в LaTeX
print(sym_kan.latex())
# Вывод: y_{0} = 0.998 x_{0}^2 + 1.001 \sin(\pi x_{1})

# Прямое вычисление формулы на CPU без аллокаций памяти GPU
y_pred = sym_kan(x_test)
```

### 4. Автономный экспорт в ANSI C99 / C++ Header-Only (`export_c`)
Экспорт обученной и дистиллированной KAN в чистый заголовочный C99-файл без каких-либо внешних зависимостей (без MLX, Python и BLAS). Готово к прямой вставке в гидродинамические и физические симуляторы (**OpenFOAM, MODFLOW, SU2**), микроконтроллеры (**STM32, ESP32**) и медицинские прошивки:

```python
# Экспорт модели прямо в C-заголовок
sym_kan.export_c("kan_model.h", function_name="kan_evaluate")

# Или получение C-кода в виде строки
c_code = sym_kan.to_c_code(function_name="kan_evaluate")
```
- **Производительность**: **$> 1,000,000,000$ точек/сек** ($10^7$ вычислений за $9.9\text{ мс}$ при `clang -O3`).
- **Память**: $0\text{ MB}$ RAM, $0\text{ MB}$ VRAM. Чистые скалярные регистры процессора.

### Результаты бенчмарка квантования (топология `[128, 256, 128]`, Metal GPU):

| Модель | FP32 Память | INT8 Память | Сжатие INT8 | INT8 MAE | INT4 Память | Сжатие INT4 |
|---|---|---|---|---|---|---|
| **`FastKAN` (RBF)** | 2.26 MB | 0.64 MB | **3.51x** | 0.757 | 0.36 MB | **6.23x** |
| **`ReLUKAN` (Tent)** | 2.26 MB | 0.64 MB | **3.51x** | 0.559 | 0.36 MB | **6.23x** |
| **`ChebyKAN`** | 1.75 MB | 0.49 MB | **3.56x** | 1.441 | 0.27 MB | **6.40x** |
| **`WavKAN` (Wavelet)** | 2.27 MB | 0.66 MB | **3.46x** | 0.706 | 0.38 MB | **6.06x** |
| **`FourierKAN`** | 3.50 MB | 0.98 MB | **3.56x** | 1.851 | 0.55 MB | **6.40x** |
| **`JacobiKAN`** | 1.75 MB | 0.49 MB | **3.56x** | 1.220 | 0.27 MB | **6.40x** |
| **`MultKAN` (2.0)** | 3.39 MB | 0.96 MB | **3.52x** | 0.578 | 0.54 MB | **6.28x** |
| **`LowRankKAN`** | 0.47 MB | 0.16 MB | **2.93x** | 0.115 | 0.09 MB | **4.99x** |
| **`B-Spline KAN`** | 2.77 MB | 0.72 MB | **3.83x** | 0.115 | 0.41 MB | **6.76x** |

<p align="center">
  <img src="assets/quantization_benchmark.png" alt="Бенчмарк квантования на Metal GPU" width="95%"/>
</p>

---

### 5. Запуск тестов и бенчмарков
```bash
# 1. Запуск 19 модульных тестов (включая квантование INT8/INT4):
uv run --with mlx python -m unittest test_kan.py

# 2. Изо-параметрический стресс-тест:
uv run --with mlx python iso_param_stress_test.py

# 3. Сравнительный замер скорости (throughput):
uv run --with mlx python benchmark.py

# 4. Бенчмарк квантования INT8/INT4 на Metal GPU:
uv run --with mlx,matplotlib python quantization_benchmark.py
```

---

## Лицензия
MIT License.
