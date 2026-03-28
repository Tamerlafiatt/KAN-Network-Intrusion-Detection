"""
evaluate.py  —  metrics, inference timing, memory, model size
Fixed:
  - remaining_params and sparsity now use pruning.count_nonzero_params()
    which measures only prunable weights — matches what pruning.py actually zeros
  - per-class F1 breakdown added (shows attack-type breakdown clearly)
  - inference time measured over 10 reps for stability
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
import psutil


@torch.no_grad()
def get_predictions(model, X, batch_size=1024, device_str='cpu'):
    device = torch.device(device_str)
    model  = model.to(device); model.eval()
    ld = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    for (Xb,) in ld:
        probs.append(torch.softmax(model(Xb.to(device)), 1).cpu().numpy())
    probs = np.vstack(probs)
    return probs.argmax(1), probs


@torch.no_grad()
def measure_inference_time(model, X, batch_size=512, n_reps=10, device_str='cpu'):
    device = torch.device(device_str)
    model  = model.to(device); model.eval()
    n      = min(batch_size, len(X))
    Xb     = torch.tensor(X[:n], dtype=torch.float32).to(device)
    # Warm-up
    for _ in range(3): model(Xb)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model(Xb)
        times.append(time.perf_counter() - t0)
    return dict(
        mean_ms       = float(np.mean(times)) * 1e3,
        std_ms        = float(np.std(times))  * 1e3,
        us_per_sample = float(np.mean(times)) / n * 1e6,
    )


def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2


def ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def evaluate_model(model, data, split='test', batch_size=1024,
                   device_str='cpu', verbose=True):
    from pruning import count_nonzero_params   # use honest counter

    X, y  = data[f'X_{split}'], data[f'y_{split}']
    n_cl  = data['n_classes']

    preds, probs = get_predictions(model, X, batch_size, device_str)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, average='weighted', zero_division=0)
    rec  = recall_score(y, preds, average='weighted', zero_division=0)
    f1   = f1_score(y, preds, average='weighted', zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(y, preds, average=None, zero_division=0)

    try:
        if n_cl == 2:
            auc = roc_auc_score(y, probs[:, 1])
        else:
            auc = roc_auc_score(y, probs, multi_class='ovr', average='weighted')
    except Exception:
        auc = float('nan')

    cm    = confusion_matrix(y, preds)
    infer = measure_inference_time(model, X, batch_size, device_str=device_str)

    # ── Honest parameter counts via pruning module ──────────────────────────
    stats = count_nonzero_params(model)

    res = dict(
        # Classification metrics
        accuracy      = acc,
        precision     = prec,
        recall        = rec,
        f1            = f1,
        f1_per_class  = f1_per_class,
        roc_auc       = auc,
        confusion_matrix = cm,
        predictions   = preds,
        probabilities = probs,
        y_true        = y,
        # Parameter counts (honest — prunable weights only)
        total_params     = stats['total_params'],
        prunable_params  = stats['prunable_params'],
        remaining_params = stats['remaining_params'],
        zero_params      = stats['zero_params'],
        sparsity         = stats['sparsity'],
        # Efficiency
        inference_ms     = infer['mean_ms'],
        inference_std_ms = infer['std_ms'],
        us_per_sample    = infer['us_per_sample'],
        model_size_mb    = model_size_mb(model),
        memory_mb        = 0.0,
    )

    if verbose:
        print(f'  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  '
              f'prunable={stats["prunable_params"]:,}  '
              f'remaining={stats["remaining_params"]:,}  '
              f'sparsity={stats["sparsity"]:.1%}  '
              f'{infer["mean_ms"]:.2f}±{infer["std_ms"]:.2f}ms  '
              f'{model_size_mb(model):.3f}MB')

    return res
