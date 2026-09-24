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


def save_pretrained(
    model: nn.Module,
    save_directory: str,
    config: Optional[dict] = None,
    weights_name: str = "model.safetensors",
) -> str:
    """
    Saves model weights in HuggingFace / SafeTensors format along with optional config.json.

    Args:
        model: MLX nn.Module
        save_directory: Target directory or direct filepath (.safetensors or .npz)
        config: Optional metadata dictionary saved as config.json
        weights_name: File name if save_directory is a folder (default "model.safetensors")

    Returns:
        Absolute filepath of saved weights.
    """
    import os
    import json
    from mlx.utils import tree_flatten

    if save_directory.endswith(".safetensors") or save_directory.endswith(".npz"):
        filepath = os.path.abspath(save_directory)
        out_dir = os.path.dirname(filepath)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.abspath(save_directory)
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, weights_name)

    flat = dict(tree_flatten(model.parameters()))
    if filepath.endswith(".safetensors"):
        mx.save_safetensors(filepath, flat)
    else:
        mx.savez(filepath, **flat)

    if config is not None:
        cfg_path = os.path.join(out_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    return filepath


def load_pretrained(
    model: nn.Module,
    path_or_dir: str,
    weights_name: str = "model.safetensors",
) -> nn.Module:
    """
    Loads model weights from SafeTensors or NPZ file or folder.

    Args:
        model: MLX nn.Module to load weights into
        path_or_dir: Directory containing weights or direct path to weights file
        weights_name: Default weights filename to check in directory

    Returns:
        The updated model.
    """
    import os
    from mlx.utils import tree_unflatten

    if os.path.isdir(path_or_dir):
        candidate_st = os.path.join(path_or_dir, weights_name)
        candidate_npz = os.path.join(path_or_dir, "model.npz")
        if os.path.exists(candidate_st):
            filepath = candidate_st
        elif os.path.exists(candidate_npz):
            filepath = candidate_npz
        else:
            raise FileNotFoundError(f"No weights file found in {path_or_dir}")
    else:
        filepath = path_or_dir
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found at {filepath}")

    loaded = mx.load(filepath)
    model.update(tree_unflatten(list(loaded.items())))
    return model
