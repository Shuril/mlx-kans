"""
Structural Pruning and Node Compaction for Kolmogorov-Arnold Networks (KAN).

Unlike standard MLPs that require unstructured sparse masks, KANs exhibit natural node-level
sparsity when trained with L1/entropy regularization. Inactive neurons whose incoming and
outgoing spline coefficients decay to zero can be mathematically sliced and eliminated,
producing a physically smaller model with reduced parameters and accelerated Metal GPU execution.
"""

from __future__ import annotations
from typing import Sequence, Union, List, Dict, Any, Tuple
import mlx.core as mx
import mlx.nn as nn


def _get_layer_weights(layer: nn.Module) -> Tuple[mx.array, mx.array, int]:
    """Extract base weight, basis weight, and basis dimension from any KAN layer."""
    base_w = getattr(layer, 'base_weight', None)
    if base_w is None:
        raise ValueError(f'Layer {type(layer).__name__} does not have a base_weight attribute.')

    basis_keys = ['spline_weight', 'cheby_weight', 'wav_weight', 'fourier_weight', 'jacobi_weight']
    basis_w = None
    basis_attr_name = None
    for k in basis_keys:
        if hasattr(layer, k):
            basis_w = getattr(layer, k)
            basis_attr_name = k
            break

    if basis_w is None:
        raise ValueError(f'Layer {type(layer).__name__} does not have any recognized basis weights.')

    if basis_w.ndim == 2:
        in_dim = layer.in_features
        num_basis = basis_w.shape[1] // in_dim
    elif basis_w.ndim == 3:
        num_basis = basis_w.shape[2]
    else:
        num_basis = 1

    return base_w, basis_w, num_basis, basis_attr_name


def compute_node_importance(model: nn.Module) -> List[mx.array]:
    """
    Calculate the importance score for each hidden neuron across all hidden layers.

    The importance score of hidden neuron j is given by the product of its outgoing
    coupling magnitude from layer l and its incoming coupling magnitude to layer l+1:
        I_j = magnitude_out(layer_l, j) * magnitude_in(layer_{l+1}, j)

    Args:
        model: Trained KAN model (e.g. FastKAN, ChebyKAN, KAN, etc.).

    Returns:
        List of 1D mx.arrays, one per hidden layer, containing the importance of each neuron.
    """
    if not hasattr(model, 'layers') or len(model.layers) < 2:
        raise ValueError('Model must possess at least 2 layers to compute hidden node importance.')

    layers = model.layers
    num_hidden_layers = len(layers) - 1
    importances = []

    for l_idx in range(num_hidden_layers):
        l_curr = layers[l_idx]
        l_next = layers[l_idx + 1]

        base_curr, basis_curr, _, _ = _get_layer_weights(l_curr)
        base_next, basis_next, n_basis_next, _ = _get_layer_weights(l_next)

        # Outgoing magnitude from l_curr (shape: [out_features])
        out_base = mx.mean(mx.abs(base_curr), axis=1)
        if basis_curr.ndim == 2:
            out_basis = mx.mean(mx.abs(basis_curr), axis=1)
        else:
            out_basis = mx.mean(mx.abs(basis_curr), axis=(1, 2))
        out_mag = out_base + out_basis

        # Incoming magnitude to l_next (shape: [in_features])
        in_base = mx.mean(mx.abs(base_next), axis=0)
        if basis_next.ndim == 2:
            b_reshaped = basis_next.reshape(l_next.out_features, l_next.in_features, n_basis_next)
            in_basis = mx.mean(mx.abs(b_reshaped), axis=(0, 2))
        else:
            in_basis = mx.mean(mx.abs(basis_next), axis=(0, 2))
        in_mag = in_base + in_basis

        importance = out_mag * in_mag
        importances.append(importance)

    return importances


def compact_kan(model: nn.Module, active_indices: Sequence[Sequence[int]]) -> nn.Module:
    """
    Construct a compacted KAN model by slicing weights according to active neuron indices.

    Args:
        model: Original trained KAN model.
        active_indices: List of integer sequences, one per hidden layer, specifying which
            neurons to keep.

    Returns:
        A new compacted model instance with sliced weights and reduced parameter dimensions.
    """
    from .fast_kan import FastKAN
    from .cheby_kan import ChebyKAN

    layers = model.layers
    num_hidden = len(layers) - 1
    if len(active_indices) != num_hidden:
        raise ValueError(f'Expected {num_hidden} active index sets for hidden layers, got {len(active_indices)}')

    orig_dims = [layers[0].in_features] + [l.out_features for l in layers]
    new_dims = [orig_dims[0]] + [len(idx) for idx in active_indices] + [orig_dims[-1]]
    idx_mx = [mx.array(sorted(idx)) for idx in active_indices]

    if isinstance(model, FastKAN):
        compact_model = FastKAN(new_dims, num_grids=layers[0].num_grids, grid_range=layers[0].grid_range)
        for l_idx, (old_l, new_l) in enumerate(zip(layers, compact_model.layers)):
            g = old_l.num_grids
            if l_idx == 0:
                act_out = idx_mx[0]
                new_l.base_weight = old_l.base_weight[act_out]
                new_l.spline_weight = old_l.spline_weight[act_out]
                if old_l.bias is not None:
                    new_l.bias = old_l.bias[act_out]
            elif l_idx == len(layers) - 1:
                act_in = idx_mx[-1]
                new_l.base_weight = old_l.base_weight[:, act_in]
                s = old_l.spline_weight.reshape(old_l.out_features, old_l.in_features, g)
                new_l.spline_weight = s[:, act_in, :].reshape(new_l.out_features, new_l.in_features * g)
                new_l.centers = old_l.centers[act_in]
                if old_l.bias is not None:
                    new_l.bias = old_l.bias
            else:
                act_in = idx_mx[l_idx - 1]
                act_out = idx_mx[l_idx]
                new_l.base_weight = old_l.base_weight[act_out][:, act_in]
                s = old_l.spline_weight.reshape(old_l.out_features, old_l.in_features, g)
                new_l.spline_weight = s[act_out][:, act_in, :].reshape(new_l.out_features, new_l.in_features * g)
                new_l.centers = old_l.centers[act_in]
                if old_l.bias is not None:
                    new_l.bias = old_l.bias[act_out]

        mx.eval(compact_model.parameters())
        return compact_model

    elif isinstance(model, ChebyKAN):
        compact_model = ChebyKAN(new_dims, degree=layers[0].degree)
        for l_idx, (old_l, new_l) in enumerate(zip(layers, compact_model.layers)):
            deg = old_l.degree
            if l_idx == 0:
                act_out = idx_mx[0]
                new_l.base_weight = old_l.base_weight[act_out]
                new_l.cheby_weight = old_l.cheby_weight[act_out]
                if old_l.bias is not None:
                    new_l.bias = old_l.bias[act_out]
            elif l_idx == len(layers) - 1:
                act_in = idx_mx[-1]
                new_l.base_weight = old_l.base_weight[:, act_in]
                c = old_l.cheby_weight.reshape(old_l.out_features, old_l.in_features, deg)
                new_l.cheby_weight = c[:, act_in, :].reshape(new_l.out_features, new_l.in_features * deg)
                if old_l.bias is not None:
                    new_l.bias = old_l.bias
            else:
                act_in = idx_mx[l_idx - 1]
                act_out = idx_mx[l_idx]
                new_l.base_weight = old_l.base_weight[act_out][:, act_in]
                c = old_l.cheby_weight.reshape(old_l.out_features, old_l.in_features, deg)
                new_l.cheby_weight = c[act_out][:, act_in, :].reshape(new_l.out_features, new_l.in_features * deg)
                if old_l.bias is not None:
                    new_l.bias = old_l.bias[act_out]

        mx.eval(compact_model.parameters())
        return compact_model

    else:
        raise NotImplementedError(f'Compaction not yet implemented for model type {type(model).__name__}')


def prune(
    model: nn.Module,
    threshold: float = 0.05,
    min_active: int = 1,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Automatically compute node importance, prune inactive neurons, and return a compacted model.

    Args:
        model: Trained KAN model (e.g. FastKAN, ChebyKAN).
        threshold: Relative threshold for active neurons (fraction of maximum node importance,
            e.g. 0.05 means keep nodes with > 5% of peak importance).
        min_active: Minimum number of active neurons to keep in each hidden layer.

    Returns:
        Tuple of (compact_model, stats_dict) where stats_dict contains:
            - 'orig_dims': list of original layer widths
            - 'new_dims': list of compacted layer widths
            - 'pruned_neurons': count of pruned hidden units
            - 'percent_neurons_pruned': percentage of hidden neurons pruned
            - 'active_indices': list of kept neuron indices per hidden layer
    """
    importances = compute_node_importance(model)
    active_indices = []
    orig_hidden_count = 0
    new_hidden_count = 0

    for imp in importances:
        imp_list = imp.tolist()
        max_val = max(imp_list) if imp_list else 1.0
        cutoff = threshold * max_val

        active = [i for i, val in enumerate(imp_list) if val > cutoff]
        if len(active) < min_active:
            indexed = sorted(enumerate(imp_list), key=lambda p: p[1], reverse=True)
            active = sorted([idx for idx, _ in indexed[:min_active]])

        active_indices.append(active)
        orig_hidden_count += len(imp_list)
        new_hidden_count += len(active)

    compact_model = compact_kan(model, active_indices)

    pruned = orig_hidden_count - new_hidden_count
    orig_dims = [model.layers[0].in_features] + [l.out_features for l in model.layers]
    new_dims = [compact_model.layers[0].in_features] + [l.out_features for l in compact_model.layers]

    stats = {
        'orig_dims': orig_dims,
        'new_dims': new_dims,
        'pruned_neurons': pruned,
        'percent_neurons_pruned': (pruned / orig_hidden_count * 100.0) if orig_hidden_count > 0 else 0.0,
        'active_indices': active_indices,
    }

    return compact_model, stats
