"""
pruning.py  —  magnitude, structured, and gradual pruning
Fixed:
  - Magnitude pruning now targets ONLY main weight tensors (not biases,
    LayerNorm params, scalers) so the threshold is honest
  - Sparsity measurement uses the same targeted-weight-only counter
  - KAN structured pruning correctly zeros both spline_weight AND base_weight
    for each dead edge, and reports true edge-level sparsity
  - MLP structured pruning reports the actual neuron-level sparsity
"""
import copy
import numpy as np
import torch
import torch.nn as nn


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_main_weight(name: str) -> bool:
    """
    Return True only for the core learnable weight tensors we want to prune.
    Excludes: bias, LayerNorm weight/bias, spline_scaler, BatchNorm params.
    """
    if 'bias' in name:                return False
    if 'norm' in name.lower():        return False
    if 'spline_scaler' in name:       return False
    if 'running_' in name:            return False
    return True


def _collect_main_weights(model: nn.Module):
    """
    Returns list of (param_name, param_tensor) for all prunable weights.
    """
    return [(n, p) for n, p in model.named_parameters() if _is_main_weight(n)]


def targeted_sparsity(model: nn.Module) -> float:
    """Sparsity computed only over prunable weight tensors (not biases etc.)."""
    t, z = 0, 0
    for n, p in _collect_main_weights(model):
        t += p.numel()
        z += (p.data == 0).sum().item()
    return z / (t + 1e-8)


def full_sparsity(model: nn.Module) -> float:
    """Total sparsity over ALL parameters (for reporting)."""
    t, z = 0, 0
    for p in model.parameters():
        t += p.numel()
        z += (p.data == 0).sum().item()
    return z / (t + 1e-8)


def count_nonzero_params(model: nn.Module) -> dict:
    """
    Returns dict with total_params, zero_params, remaining_params, sparsity
    counted over prunable weights only — gives the honest numbers.
    """
    total, zeros = 0, 0
    for n, p in _collect_main_weights(model):
        total += p.numel()
        zeros += (p.data == 0).sum().item()
    return dict(
        total_params     = sum(p.numel() for p in model.parameters()),
        prunable_params  = total,
        zero_params      = zeros,
        remaining_params = total - zeros,
        sparsity         = zeros / (total + 1e-8),
    )


# ── 1. Magnitude pruning (unstructured) ──────────────────────────────────────

def magnitude_prune(model: nn.Module, sparsity: float = 0.5) -> nn.Module:
    """
    Global magnitude pruning applied ONLY to main weight tensors.
    Threshold is the sparsity-th percentile of |weight| across all
    targeted tensors combined — so 50% really means 50% of weights zeroed.
    """
    model = copy.deepcopy(model)
    pairs = _collect_main_weights(model)

    if not pairs:
        print(f'  WARNING: no prunable weights found')
        return model

    # Pool all targeted weights to find global threshold
    all_vals = torch.cat([p.data.abs().flatten() for _, p in pairs])
    threshold = torch.quantile(all_vals, sparsity)

    # Zero out weights below threshold
    for name, p in pairs:
        mask = p.data.abs() >= threshold
        p.data.mul_(mask.float())

    stats = count_nonzero_params(model)
    print(f'  Magnitude prune: target={sparsity:.0%} | '
          f'actual={stats["sparsity"]:.1%} | '
          f'zeros={stats["zero_params"]:,}/{stats["prunable_params"]:,}')
    return model


# ── 2. Structured pruning ─────────────────────────────────────────────────────

def structured_prune_mlp(model: nn.Module, sparsity: float = 0.5) -> nn.Module:
    """
    Neuron-level structured pruning for MLP.
    Scores each neuron by the L2 norm of its outgoing weights,
    zeros out the bottom-sparsity fraction of neurons entirely
    (both the row in the weight matrix and the bias).
    Skips the final output layer.
    """
    model   = copy.deepcopy(model)
    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear)]

    for layer_name, layer in linears[:-1]:   # skip output layer
        W      = layer.weight.data           # (out_neurons, in_features)
        norms  = W.norm(dim=1)               # L2 norm per output neuron
        n_total = len(norms)
        n_keep  = max(1, int(n_total * (1.0 - sparsity)))
        _, keep_idx = torch.topk(norms, n_keep)

        mask = torch.zeros(n_total, dtype=torch.bool, device=W.device)
        mask[keep_idx] = True

        layer.weight.data[~mask] = 0.0
        if layer.bias is not None:
            layer.bias.data[~mask] = 0.0

        n_pruned = (~mask).sum().item()
        print(f'    {layer_name}: pruned {n_pruned}/{n_total} neurons')

    stats = count_nonzero_params(model)
    print(f'  Structured prune (MLP): target={sparsity:.0%} | '
          f'actual={stats["sparsity"]:.1%} | '
          f'zeros={stats["zero_params"]:,}/{stats["prunable_params"]:,}')
    return model


def structured_prune_kan(model: nn.Module, sparsity: float = 0.5) -> nn.Module:
    """
    Edge-level structured pruning for KAN.
    Scores each (out, in) edge by the L2 norm of its spline coefficient vector,
    zeroes out all spline coefficients AND the base weight for the weakest edges.
    """
    from model_kan import KANLayer, PolyKANLayer
    model = copy.deepcopy(model)

    for mod_name, m in model.named_modules():

        if isinstance(m, KANLayer):
            # spline_weight: (out_features, in_features, n_basis)
            sw     = m.spline_weight.data
            # Score each edge by L2 norm over its basis coefficients
            norms  = sw.norm(dim=-1)            # (out, in)
            flat   = norms.flatten()
            n_keep = max(1, int(len(flat) * (1.0 - sparsity)))
            thr    = torch.topk(flat, n_keep).values.min()

            # Build binary edge mask
            edge_mask = (norms >= thr)          # (out, in)  True = keep

            # Zero spline weights for dead edges
            m.spline_weight.data[~edge_mask] = 0.0

            # Zero base weight for dead edges too
            m.base_weight.data[~edge_mask] = 0.0

            n_dead = (~edge_mask).sum().item()
            n_total = edge_mask.numel()
            print(f'    {mod_name} (KANLayer): pruned {n_dead}/{n_total} edges')

        elif isinstance(m, PolyKANLayer):
            # poly_weight: (out_features, in_features, degree+1)
            pw     = m.poly_weight.data
            norms  = pw.norm(dim=-1)            # (out, in)
            flat   = norms.flatten()
            n_keep = max(1, int(len(flat) * (1.0 - sparsity)))
            thr    = torch.topk(flat, n_keep).values.min()

            edge_mask = (norms >= thr)
            m.poly_weight.data[~edge_mask] = 0.0

            n_dead = (~edge_mask).sum().item()
            print(f'    {mod_name} (PolyKANLayer): pruned {n_dead}/{edge_mask.numel()} edges')

    stats = count_nonzero_params(model)
    print(f'  Structured prune (KAN): target={sparsity:.0%} | '
          f'actual={stats["sparsity"]:.1%} | '
          f'zeros={stats["zero_params"]:,}/{stats["prunable_params"]:,}')
    return model


# ── 3. Gradual pruning schedule ───────────────────────────────────────────────

class GradualPruner:
    """
    Polynomial sparsity schedule applied during training.
    s(t) = s_final * (1 - (1 - (t-t0)/(T-t0))^3)
    Applied every `freq` optimizer steps.
    """
    def __init__(self, model, final_sparsity=0.5,
                 begin_step=2, end_step=10, freq=1):
        self.model  = model
        self.final  = final_sparsity
        self.begin  = begin_step
        self.end    = end_step
        self.freq   = freq
        self.step_n = 0

    def target(self, t):
        if t < self.begin:  return 0.0
        if t >= self.end:   return self.final
        p = (t - self.begin) / (self.end - self.begin)
        return self.final * (1 - (1 - p) ** 3)

    def step(self):
        self.step_n += 1
        if self.step_n % self.freq == 0:
            s = self.target(self.step_n)
            if s > 0:
                # Apply in-place without deepcopy (it's during training)
                pairs = _collect_main_weights(self.model)
                all_vals = torch.cat([p.data.abs().flatten() for _, p in pairs])
                thr = torch.quantile(all_vals, s)
                for _, p in pairs:
                    p.data.mul_((p.data.abs() >= thr).float())
        return self.target(self.step_n)


# ── summary helper ────────────────────────────────────────────────────────────

def pruning_summary(original: nn.Module, pruned: nn.Module) -> dict:
    orig  = count_nonzero_params(original)
    after = count_nonzero_params(pruned)
    return dict(
        original_prunable = orig['prunable_params'],
        total_params      = after['total_params'],
        prunable_params   = after['prunable_params'],
        zero_params       = after['zero_params'],
        remaining_params  = after['remaining_params'],
        sparsity          = after['sparsity'],
    )
