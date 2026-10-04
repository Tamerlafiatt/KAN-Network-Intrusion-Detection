# KAN vs MLP + Exact Symbolic Reconstruction for Network Intrusion Detection

A research-oriented benchmarking project investigating **Kolmogorov–Arnold Networks (KANs)** for network intrusion detection from two complementary perspectives:

1. **Predictive performance:** KAN vs Residual MLP.
2. **Deployment efficiency:** Exact Symbolic Reconstruction vs FastKAN and EfficientKAN.

The first benchmark evaluates whether KAN-style learnable edge functions can improve intrusion-detection performance compared with a conventional neural network.

The second investigates whether a trained symbolic KAN can be transformed into a significantly faster and smaller representation **without retraining and without changing the symbolic function it computes**.

---

## Part I — KAN vs Residual MLP

The first experiment compares KAN architectures against a Residual MLP under the same intrusion-detection pipeline.

### Models

| Model | Representation | Role |
|---|---|---|
| **B-spline KAN** | Learnable spline functions on edges | Main KAN architecture |
| **FastKAN** | Gaussian RBF edge functions | Faster KAN variant |
| **PolyKAN** | Chebyshev polynomial functions | Polynomial KAN variant |
| **Residual MLP** | Linear layers + fixed nonlinearities | Neural-network baseline |

### KAN vs MLP Results

| Model | Accuracy | F1 | Inference |
|---|---:|---:|---:|
| **KAN** | **~0.941** | **~0.932** | ~12 ms |
| Residual MLP | ~0.938 | ~0.928 | **~8 ms** |
| PolyKAN (degree 4) | ~0.940 | ~0.931 | ~15 ms |

In this experiment, KAN achieves slightly higher predictive performance:

- Accuracy: **~94.1% vs ~93.8%**
- F1: **~93.2% vs ~92.8%**

The MLP remains faster at inference.

This illustrates the first trade-off investigated in the project:

> **KAN provides stronger predictive performance in the reported experiment, while the MLP retains a latency advantage.**

---

# Part II — Exact Symbolic Reconstruction

The second part focuses on a different limitation of KANs: **inference efficiency after symbolification**.

KANs are attractive because each connection contains a learnable one-dimensional function that can be replaced by an explicit symbolic expression.

However, evaluating these symbolic edge functions through a general-purpose deep-learning backend introduces considerable framework and dispatch overhead.

The proposed solution is **Exact Symbolic Reconstruction**.

---

## From Symbolic KAN to a Single Polynomial

After symbolification, each KAN edge is represented as

```text
φ(x) = c · f(a·x + b) + d
```

where the symbolic function is selected from a low-degree monomial library:

```text
{x, x², x³, x⁴, x⁵}
```

Instead of evaluating every symbolic edge separately, reconstruction:

```text
Symbolic KAN
     ↓
Read symbolic edge functions
     ↓
Substitute compositions
     ↓
Expand polynomial powers
     ↓
Collect identical monomials
     ↓
Single multivariate polynomial
     ↓
Direct arithmetic evaluation
```

The resulting output has the form:

```text
ŷ(x) = Σ β_α x^α
```

and therefore requires only **additions and multiplications** during inference.

No spline evaluation, RBF evaluation, neural-network backend, or edge-level dispatch is required.

---

## Exactness

This transformation is not a new approximation of the KAN.

It is an algebraic reconstruction of the already-symbolified network.

The reconstruction performs only:

- substitution,
- polynomial expansion,
- collection of identical terms.

No coefficients are retrained, approximated, truncated, or replaced.

Experimentally, the mean difference between reconstructed and symbolic KAN logits is approximately:

```text
2.4 × 10⁻⁷
```

with **no change in predicted labels**.

Therefore:

> **The reconstructed model preserves the symbolic KAN function up to floating-point round-off.**

---

# Reconstruction vs FastKAN vs EfficientKAN

The principal deployment benchmark compares:

- **Exact Reconstruction**
- **Safe Pruning + Reconstruction**
- **FastKAN**
- **EfficientKAN**

on CIC-IDS2017.

## CIC-IDS2017 Results

| Method | AUC | Accuracy | F1 | ms / batch | Parameters |
|---|---:|---:|---:|---:|---:|
| **Reconstruction** | 0.9952 ± 0.0014 | 0.9803 ± 0.0036 | 0.9755 ± 0.0044 | **0.3746 ± 0.0397** | **468** |
| **Safe + Reconstruction** | 0.9899 ± 0.0039 | 0.9720 ± 0.0068 | 0.9656 ± 0.0082 | **0.3694 ± 0.0157** | **358** |
| FastKAN | **0.9985 ± 0.0008** | **0.9911 ± 0.0039** | **0.9889 ± 0.0049** | 1.2543 ± 0.2263 | 3,773 |
| EfficientKAN | 0.9981 ± 0.0015 | 0.9873 ± 0.0031 | 0.9841 ± 0.0039 | 4.6201 ± 0.3690 | 4,000 |

### Main result

Compared with FastKAN, reconstruction is approximately:

```text
3.35× faster
8.1× fewer parameters
```

Compared with EfficientKAN:

```text
12.3× faster
8.5× fewer parameters
```

while retaining an AUC above **0.995**.

Safe pruning reduces the model even further:

```text
468 → 358 parameters
```

without retraining.

---

## What This Comparison Shows

The objective is not to claim that reconstruction has the highest classification score on CIC-IDS2017.

FastKAN achieves slightly higher AUC, Accuracy and F1.

Instead, reconstruction targets a different point on the performance–efficiency trade-off:

```text
                  Predictive       Inference       Model
                  quality          speed           size

FastKAN           ★★★★★            ★★★             ★★
EfficientKAN      ★★★★★            ★               ★★
Reconstruction    ★★★★★            ★★★★★           ★★★★★
```

The important result is that comparable discrimination can be retained while dramatically reducing computational and storage cost.

> **Exact reconstruction trades a very small amount of discrimination for a major improvement in latency, compactness and interpretability.**

---

# Safe Pruning

The reconstructed symbolic model can be compressed further using **safe pruning**.

Unlike standard magnitude pruning, an edge is not deleted simply because its weight is small.

For each symbolic edge, the method computes its mean contribution:

```text
cₑ = mean(|φₑ(x)|)
```

Candidate edges are ordered by normalized contribution.

Before an edge is permanently removed, the model is evaluated with that edge ablated.

Removal is accepted only if:

```text
ΔAUC ≤ τ_AUC
and
ΔAccuracy ≤ τ_Accuracy
```

This produces a pruning strategy that is:

- post-training,
- retraining-free,
- based on actual prediction impact,
- bounded by explicit quality tolerances.

---

# Cross-Dataset Results

The reconstruction method was also evaluated beyond intrusion detection.

| Dataset | Reconstruction AUC | FastKAN AUC | EfficientKAN AUC |
|---|---:|---:|---:|
| NSL-KDD | 0.975 | **0.990** | 0.985 |
| **Pima Diabetes** | **0.819** | 0.767 | 0.812 |
| Covertype | 0.849 | 0.857 | **0.858** |
| **MNIST** | **0.960** | 0.942 | 0.915 |

The results show that reconstruction is not merely a compression mechanism.

On some datasets it also achieves the strongest discrimination.

### Pima Diabetes

```text
Reconstruction AUC: 0.819
FastKAN AUC:        0.767
EfficientKAN AUC:   0.812
```

### MNIST

```text
Reconstruction AUC: 0.960
FastKAN AUC:        0.942
EfficientKAN AUC:   0.915
```

On MNIST, reconstruction also achieves:

```text
Accuracy: 0.904
F1:       0.901
```

compared with:

```text
FastKAN
Accuracy: 0.875
F1:       0.866

EfficientKAN
Accuracy: 0.840
F1:       0.834
```

The trade-off is that high-dimensional polynomial expansion makes reconstruction slower than FastKAN on MNIST.

---

# Controlled Same-Architecture Comparison

To determine whether the efficiency improvement comes simply from using a smaller architecture, an additional controlled experiment gives all models the identical topology:

```text
[8, 5, 1]
```

Under this controlled setting:

| Model | Parameters |
|---|---:|
| **Reconstruction** | **180** |
| FastKAN | 411 |
| EfficientKAN | 450 |

All three models contain the same number of edges.

The difference therefore comes from the representation itself.

A reconstructed symbolic edge requires only:

```text
(a, b, c, d)
```

while FastKAN requires several RBF coefficients and EfficientKAN requires multiple spline coefficients.

This makes reconstruction approximately:

```text
2.3–2.5× more compact
```

even when architecture size is held constant.

---

# Overall Findings

The project therefore evaluates KANs along two separate dimensions.

## 1. KAN vs MLP

```text
Question:
Can KAN-style learnable functions improve intrusion detection
compared with a traditional neural network?

Result:
KAN achieves slightly better Accuracy and F1 in the reported benchmark.
```

## 2. Reconstruction vs Existing KAN Accelerators

```text
Question:
Can an interpretable symbolic KAN also be efficient enough for deployment?

Result:
Yes.

Exact reconstruction:
• preserves the symbolic function
• requires no retraining
• runs 3.35× faster than FastKAN
• runs 12.3× faster than EfficientKAN
• stores roughly 8× fewer parameters
• maintains AUC ≈ 0.995
```

Together, these experiments investigate the complete KAN pipeline:

```text
                  MODEL QUALITY
                       │
                 KAN vs MLP
                       │
                       ▼
             Train interpretable KAN
                       │
                       ▼
                 Symbolification
                       │
                       ▼
             Safe edge pruning
                       │
                       ▼
          Exact symbolic reconstruction
                       │
                       ▼
             Fast polynomial inference
                       │
                       ▼
            DEPLOYMENT EFFICIENCY
```

The broader goal is therefore not only to ask:

> **Can KANs perform well?**

but also:

> **Can an interpretable KAN be transformed into a compact and computationally efficient model suitable for real deployment?**

The experimental results suggest that it can.
