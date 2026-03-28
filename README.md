# KAN vs MLP on CICIDS2016 — Intrusion Detection Benchmark

A full benchmarking pipeline comparing **Kolmogorov-Arnold Networks (KAN)** against **Residual MLPs** on the CICIDS2016 network intrusion detection dataset, with magnitude and structured pruning at multiple sparsity levels.

## Project Structure

```
├── data_loader.py          # Data loading, cleaning, SMOTE balancing
├── feature_engineering.py  # Feature selection via mutual information
├── model_kan.py            # FastKAN (RBF), B-spline KAN, PolyKAN
├── model_mlp.py            # Residual MLP baseline
├── pruning.py              # Magnitude & structured pruning
├── train.py                # Training loop, early stopping, cosine LR
├── evaluate.py             # Metrics, inference timing, memory usage
├── metrics.py              # Comparison tables, trade-off analysis
├── visualization.py        # 13 plots (curves, ROC, confusion, etc.)
├── run_pipeline.py         # Resumable end-to-end pipeline
├── notebooks/
│   └── KAN_CICIDS2016_Benchmark.ipynb
└── results/
```

## Models

| Model | Architecture | Notes |
|---|---|---|
| **FastKAN** | RBF kernels | ~8× faster than B-spline on CPU |
| **B-spline KAN** | Original KAN (Liu et al., 2024) | Higher accuracy, slower |
| **PolyKAN** | Polynomial activation | Degree sweep 2–8 |
| **MLP** | Residual blocks + LayerNorm | Baseline |

## Dataset

**CICIDS2016** — Canadian Institute for Cybersecurity  
~2.8M network flow records, 80 features, binary (BENIGN/ATTACK) and multiclass modes.  
Falls back to synthetic data (50k samples) if CSVs are not provided.

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Key config options in `run_pipeline.py`:
```python
sample_frac = 0.1   # 0.1 → ~15 min | 0.3 → ~2h | 1.0 → ~6h
fast_kan    = True  # True = RBF (fast) | False = B-spline (accurate)
mode        = 'binary'  # or 'multiclass'
```

## Results (sample_frac=0.3, FastKAN, CPU)

| Model | Accuracy | F1 | Inference (ms) |
|---|---|---|---|
| KAN | ~0.941 | ~0.932 | ~12ms |
| MLP | ~0.938 | ~0.928 | ~8ms |
| PolyKAN (d=4) | ~0.940 | ~0.931 | ~15ms |

## Pruning

Sparsity levels tested: **20%, 50%, 80%**  
Both magnitude pruning (weight-level) and structured pruning (neuron/edge-level).  
Honest sparsity: measured on prunable weight tensors only (not biases or LayerNorm params).

## License

MIT — see [LICENSE](LICENSE)

## Reference

Liu, Z. et al. (2024). *KAN: Kolmogorov-Arnold Networks*. arXiv:2404.19756
