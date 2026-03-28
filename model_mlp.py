"""
model_mlp.py  —  Residual MLP baseline
"""
import torch
import torch.nn as nn
from typing import List


class MLPBlock(nn.Module):
    def __init__(self, in_f, out_f, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_f, out_f), nn.LayerNorm(out_f),
                                 nn.GELU(), nn.Dropout(dropout))
        self.proj = nn.Linear(in_f, out_f, bias=False) if in_f != out_f else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)


class MLPNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int], dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            *[MLPBlock(layer_sizes[i], layer_sizes[i+1], dropout)
              for i in range(len(layer_sizes) - 2)])
        self.output = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.output(self.blocks(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_sparsity(self):
        t, z = 0, 0
        for p in self.parameters():
            t += p.numel(); z += (p.data == 0).sum().item()
        return z / (t + 1e-8)


def build_mlp(n_features, n_classes, hidden_sizes=None, dropout=0.1):
    if hidden_sizes is None:
        hidden_sizes = [64, 32]
    sizes = [n_features] + hidden_sizes + [n_classes]
    m = MLPNetwork(sizes, dropout)
    print(f'  MLP built: {sizes} | params: {m.count_parameters():,}')
    return m
