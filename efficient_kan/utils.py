"""
Utility functions for Apple Silicon MLX training and optimization.
Includes training step JIT compiler, parameter counters, and mixed precision helpers.
"""

from __future__ import annotations
from typing import Callable, Any
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def build_train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: Callable[[nn.Module, mx.array, mx.array], mx.array],
) -> Callable[[mx.array, mx.array], mx.array]:
    """
    Build a JIT-compiled Metal training step capturing model and optimizer state.

    Args:
        model: MLX nn.Module
        optimizer: MLX Optimizer
        loss_fn: Callable(model, x, y) -> scalar loss

    Returns:
        Compiled train_step(x, y) -> loss
    """
    step_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    def step(x: mx.array, y: mx.array) -> mx.array:
        loss, grads = step_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    return mx.compile(step, inputs=state, outputs=state)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count total and trainable parameters in an MLX module.

    Returns:
        dict with 'trainable', 'frozen', and 'total' parameter counts.
    """
    def _count(d: Any) -> int:
        total = 0
        if isinstance(d, dict):
            for v in d.values():
                total += _count(v)
        elif isinstance(d, list):
            for v in d:
                total += _count(v)
        elif isinstance(d, mx.array):
            total += d.size
        return total

    trainable = _count(model.trainable_parameters())
    total = _count(model.parameters())
    return {
        "trainable": trainable,
        "frozen": total - trainable,
        "total": total,
    }


def to_fp16(model: nn.Module) -> nn.Module:
    """Cast all floating point parameters in model to float16 for Metal acceleration."""
    def _cast_tree(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: _cast_tree(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [_cast_tree(v) for v in d]
        elif isinstance(d, mx.array) and mx.issubdtype(d.dtype, mx.floating):
            return d.astype(mx.float16)
        return d

    model.update(_cast_tree(model.parameters()))
    return model


def to_bf16(model: nn.Module) -> nn.Module:
    """
    Cast all floating point parameters in model to bfloat16.

    Hardware execution:
      - Native hardware execution units require Apple Silicon M3 or newer (Apple GPU Family 8+).
      - On M1 / M2, executed via software emulation/upcasting (halves memory and avoids underflow/NaN,
        but compute runs at FP32 throughput).
    """
    def _cast_tree(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: _cast_tree(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [_cast_tree(v) for v in d]
        elif isinstance(d, mx.array) and mx.issubdtype(d.dtype, mx.floating):
            return d.astype(mx.bfloat16)
        return d

    model.update(_cast_tree(model.parameters()))
    return model
