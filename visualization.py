"""
visualization.py  —  all plots
Fixed/added:
  - sparsity_vs_accuracy now uses honest sparsity from evaluate
  - new plot: KAN vs MLP direct comparison bars (accuracy, F1, inference, params)
  - new plot: pruning impact waterfall (accuracy drop per sparsity level)
  - new plot: per-class F1 heatmap
  - network_structure uses prunable_params not total_params for honest bar
  - all plots saved at 130 dpi, larger fonts
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.preprocessing import label_binarize
import warnings; warnings.filterwarnings('ignore')

PALETTE = ['#2563EB','#DC2626','#16A34A','#D97706','#7C3AED','#0891B2','#BE185D','#0D9488']
plt.rcParams.update({
    'font.family':        'monospace',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'axes.titlesize':     12,
    'axes.labelsize':     10,
})


# ── Training curves ───────────────────────────────────────────────────────────
def plot_training_curves(histories, save_path='training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training & Validation Curves', fontsize=14, fontweight='bold')
    for i, (name, h) in enumerate(histories.items()):
        c  = PALETTE[i % len(PALETTE)]
        ep = range(1, len(h['train_loss']) + 1)
        axes[0].plot(ep, h['train_loss'], '--', c=c, alpha=0.45, lw=1.5)
        axes[0].plot(ep, h['val_loss'],   '-',  c=c, lw=2.2, label=name)
        axes[1].plot(ep, h['train_acc'],  '--', c=c, alpha=0.45, lw=1.5)
        axes[1].plot(ep, h['val_acc'],    '-',  c=c, lw=2.2, label=name)
    for ax, title, ylabel in zip(axes,
            ['Loss  (dashed=train, solid=val)', 'Accuracy  (dashed=train, solid=val)'],
            ['Cross-Entropy Loss', 'Accuracy']):
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── KAN vs MLP direct comparison ──────────────────────────────────────────────
def plot_kan_vs_mlp(results: dict, save_path='kan_vs_mlp_comparison.png'):
    """
    Side-by-side bar comparison of KAN, PolyKAN and MLP on key metrics.
    """
    keys    = ['KAN', 'PolyKAN', 'MLP']
    keys    = [k for k in keys if k in results]
    metrics = ['accuracy', 'f1', 'roc_auc']
    labels  = ['Accuracy', 'F1 Score', 'ROC-AUC']
    colors  = ['#2563EB', '#7C3AED', '#DC2626']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('KAN vs PolyKAN vs MLP — Direct Comparison', fontsize=14, fontweight='bold')

    x = np.arange(len(keys))
    for ax, metric, label in zip(axes, metrics, labels):
        vals  = [results[k][metric] for k in keys]
        bars  = ax.bar(keys, vals, color=colors[:len(keys)], alpha=0.85,
                       edgecolor='white', linewidth=1.5, width=0.5)
        # value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0003,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # zoom y-axis to show differences
        ymin = max(0, min(vals) - 0.005)
        ymax = min(1, max(vals) + 0.005)
        ax.set_ylim(ymin, ymax)
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel(label)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── Inference time & params comparison ───────────────────────────────────────
def plot_efficiency_comparison(results: dict, save_path='efficiency_comparison.png'):
    keys   = ['KAN', 'PolyKAN', 'MLP']
    keys   = [k for k in keys if k in results]
    colors = ['#2563EB', '#7C3AED', '#DC2626']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Model Efficiency Comparison', fontsize=14, fontweight='bold')

    infers = [results[k]['inference_ms']   for k in keys]
    params = [results[k]['total_params']   for k in keys]
    sizes  = [results[k]['model_size_mb']  for k in keys]

    for ax, vals, title, ylabel, fmt in zip(
            axes,
            [infers, params, sizes],
            ['Inference Time (ms/batch)', 'Total Parameters', 'Model Size (MB)'],
            ['ms', 'params', 'MB'],
            ['.2f', ',', '.3f']):
        bars = ax.bar(keys, vals, color=colors[:len(keys)], alpha=0.85,
                      edgecolor='white', linewidth=1.5, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.01,
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── Pruning impact waterfall ──────────────────────────────────────────────────
def plot_pruning_impact(results: dict, save_path='pruning_impact.png'):
    """
    For each architecture, show accuracy and F1 across sparsity levels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pruning Impact: Accuracy & F1 vs Sparsity', fontsize=14, fontweight='bold')

    for row, arch in enumerate(['KAN', 'MLP']):
        base = results.get(arch)
        if not base:
            continue

        # Collect base + all pruned variants
        relevant = {arch: base}
        for name, r in results.items():
            if name.startswith(arch + '_') and r['sparsity'] > 0.01:
                relevant[name] = r

        spar  = [r['sparsity']  for r in relevant.values()]
        accs  = [r['accuracy']  for r in relevant.values()]
        f1s   = [r['f1']        for r in relevant.values()]
        names = list(relevant.keys())

        # Sort by sparsity
        order = np.argsort(spar)
        spar  = [spar[i]  for i in order]
        accs  = [accs[i]  for i in order]
        f1s   = [f1s[i]   for i in order]
        names = [names[i] for i in order]

        color = '#2563EB' if arch == 'KAN' else '#DC2626'

        # Accuracy
        ax = axes[row, 0]
        ax.plot(spar, accs, 'o-', color=color, lw=2.5, ms=9, markeredgecolor='white', markeredgewidth=1.5)
        for s, a, n in zip(spar, accs, names):
            ax.annotate(n, (s, a), textcoords='offset points', xytext=(6, 4), fontsize=7.5)
        ax.set_xlabel('Sparsity (fraction of prunable weights zeroed)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{arch}: Accuracy vs Sparsity', fontweight='bold')
        # Mark base
        ax.axhline(base['accuracy'], color=color, lw=1, ls='--', alpha=0.4)

        # F1
        ax = axes[row, 1]
        ax.plot(spar, f1s, 's-', color=color, lw=2.5, ms=9, markeredgecolor='white', markeredgewidth=1.5)
        for s, f, n in zip(spar, f1s, names):
            ax.annotate(n, (s, f), textcoords='offset points', xytext=(6, 4), fontsize=7.5)
        ax.set_xlabel('Sparsity (fraction of prunable weights zeroed)')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{arch}: F1 vs Sparsity', fontweight='bold')
        ax.axhline(base['f1'], color=color, lw=1, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── Confusion matrices ────────────────────────────────────────────────────────
def plot_confusion_matrices(results, class_names=None, save_path='confusion_matrices.png'):
    n    = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.array(axes).flatten()
    cmap = LinearSegmentedColormap.from_list('cyber', ['#0f172a', '#1d4ed8', '#93c5fd'])

    for ax, (name, r) in zip(axes, results.items()):
        cm   = r['confusion_matrix'].astype(float)
        cm_n = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(cm_n, annot=True, fmt='.2f', cmap=cmap, ax=ax, cbar=False,
                    xticklabels=class_names or 'auto',
                    yticklabels=class_names or 'auto',
                    annot_kws={'size': 10})
        ax.set_title(f'{name}\nAcc={r["accuracy"]:.4f}  F1={r["f1"]:.4f}',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    for ax in axes[len(results):]:
        ax.set_visible(False)

    plt.suptitle('Confusion Matrices (Row-Normalized)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── ROC curves ────────────────────────────────────────────────────────────────
def plot_roc_curves(results, save_path='roc_curves.png'):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, lw=1.5, label='Random (AUC=0.500)')

    for i, (name, r) in enumerate(results.items()):
        y, probs = r['y_true'], r['probabilities']
        c        = PALETTE[i % len(PALETTE)]
        try:
            if probs.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y, probs[:, 1])
                roc_val     = sk_auc(fpr, tpr)
            else:
                nc  = probs.shape[1]
                yb  = label_binarize(y, classes=range(nc))
                fpr = np.linspace(0, 1, 300)
                tpr = np.mean(
                    [np.interp(fpr, *roc_curve(yb[:, c_], probs[:, c_])[:2])
                     for c_ in range(nc)], axis=0)
                roc_val = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=c, lw=2.2, label=f'{name} (AUC={roc_val:.4f})')
        except Exception as e:
            print(f'  ROC skip {name}: {e}')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── Per-class F1 heatmap ──────────────────────────────────────────────────────
def plot_per_class_f1(results: dict, class_names=None, save_path='per_class_f1.png'):
    model_names = list(results.keys())
    f1_matrix   = np.array([r['f1_per_class'] for r in results.values()])

    n_classes = f1_matrix.shape[1]
    ylabels   = class_names if class_names else [f'Class {i}' for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(8, len(model_names)*1.4), max(4, n_classes*0.6)))
    cmap = LinearSegmentedColormap.from_list('perf', ['#7f1d1d', '#fbbf24', '#16a34a'])
    sns.heatmap(f1_matrix.T, annot=True, fmt='.3f',
                xticklabels=model_names, yticklabels=ylabels,
                cmap=cmap, vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor='#1e293b',
                annot_kws={'size': 9})
    ax.set_title('Per-Class F1 Score — All Models', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Class')
    ax.tick_params(axis='x', rotation=35, labelsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── Accuracy vs inference time ────────────────────────────────────────────────
def plot_accuracy_vs_inference(results, save_path='accuracy_vs_inference.png'):
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, (name, r) in enumerate(results.items()):
        c    = PALETTE[i % len(PALETTE)]
        size = 250 * (1.0 - r['sparsity'] + 0.15)
        ax.scatter(r['inference_ms'], r['accuracy'],
                   s=size, color=c, alpha=0.85, zorder=5,
                   edgecolors='white', linewidths=2)
        ax.annotate(name, (r['inference_ms'], r['accuracy']),
                    textcoords='offset points', xytext=(9, 5),
                    fontsize=9, color=c, fontweight='bold')
    ax.set_xlabel('Inference Time (ms / 512-sample batch)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy vs Inference Time\n(bubble size ∝ parameter density)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── Parameters vs performance ─────────────────────────────────────────────────
def plot_params_vs_performance(results, save_path='params_vs_performance.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    names  = list(results.keys())
    params = [r['remaining_params'] for r in results.values()]
    accs   = [r['accuracy']         for r in results.values()]
    f1s    = [r['f1']               for r in results.values()]
    cs     = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    for ax, ys, ylabel, marker in zip(axes, [accs, f1s], ['Accuracy', 'F1'], ['o', 'D']):
        ax.scatter(params, ys, s=160, c=cs, zorder=5,
                   edgecolors='white', linewidths=1.5, marker=marker)
        for n, p, y, c in zip(names, params, ys, cs):
            ax.annotate(n, (p, y), textcoords='offset points',
                        xytext=(6, 4), fontsize=8, color=c)
        ax.set_xlabel('Remaining Prunable Parameters')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Remaining Parameters vs {ylabel}', fontweight='bold')

    plt.suptitle('Model Efficiency vs Performance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── KAN learned functions ─────────────────────────────────────────────────────
def visualize_kan_functions(model, feature_names,
                             n_features=8, save_path='kan_functions.png'):
    from model_kan import KANLayer
    kan_layer = next((m for m in model.modules() if isinstance(m, KANLayer)), None)
    if kan_layer is None:
        print('  No KANLayer found.'); return

    n_show = min(n_features, kan_layer.in_features)
    cols   = 4
    rows   = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()
    x_vals = torch.linspace(-3, 3, 300)

    with torch.no_grad():
        for fi in range(n_show):
            ax = axes[fi]
            Xp = torch.zeros(300, kan_layer.in_features)
            Xp[:, fi] = x_vals
            fn = kan_layer(Xp).sum(dim=1).numpy()
            ax.plot(x_vals.numpy(), fn,
                    color=PALETTE[fi % len(PALETTE)], lw=2.5)
            ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
            ax.axvline(0, color='gray', lw=0.6, alpha=0.5)
            nm = feature_names[fi] if fi < len(feature_names) else f'feat_{fi}'
            ax.set_title(nm[:24], fontsize=8, fontweight='bold')
            ax.set_xlabel('Normalised input', fontsize=7)
            ax.set_ylabel('φ(x)', fontsize=7)

    for ax in axes[n_show:]:
        ax.set_visible(False)

    plt.suptitle('KAN Learned Univariate Functions — Layer 1',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── Network structure (honest pruned vs active) ───────────────────────────────
def visualize_network_structure(results, save_path='network_structure.png'):
    fig, ax = plt.subplots(figsize=(13, 6))
    names      = list(results.keys())
    prunable   = [r['prunable_params']  for r in results.values()]
    remaining  = [r['remaining_params'] for r in results.values()]
    zeroed     = [p - r for p, r in zip(prunable, remaining)]
    x          = np.arange(len(names))

    b1 = ax.bar(x, remaining, 0.6, label='Active (non-zero)', color='#2563EB', alpha=0.88)
    b2 = ax.bar(x, zeroed,    0.6, bottom=remaining,
                label='Pruned (zero)', color='#DC2626', alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Prunable Parameter Count')
    ax.set_title('Active vs Pruned Parameters per Model\n(counts prunable weights only)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    for i, r in enumerate(results.values()):
        ax.text(i, prunable[i] * 1.015,
                f"{r['sparsity']:.1%}",
                ha='center', va='bottom', fontsize=9,
                color='#DC2626' if r['sparsity'] > 0.1 else 'gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')


# ── Summary dashboard ─────────────────────────────────────────────────────────
def plot_summary_dashboard(results, save_path='summary_dashboard.png'):
    fig = plt.figure(figsize=(22, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)
    names = list(results.keys())
    cs    = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    def bar(ax, vals, title, ylabel, fmt='.4f'):
        bars = ax.bar(names, vals, color=cs, alpha=0.85, edgecolor='white', linewidth=1.2)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', rotation=32, labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + abs(max(vals)) * 0.005,
                    f'{v:{fmt}}',
                    ha='center', va='bottom', fontsize=8)

    bar(fig.add_subplot(gs[0, 0]), [r['accuracy']      for r in results.values()], 'Accuracy',       'Acc')
    bar(fig.add_subplot(gs[0, 1]), [r['f1']            for r in results.values()], 'F1 Score',       'F1')
    bar(fig.add_subplot(gs[0, 2]), [r['roc_auc']       for r in results.values()], 'ROC-AUC',        'AUC')
    bar(fig.add_subplot(gs[1, 0]), [r['inference_ms']  for r in results.values()], 'Inference (ms)', 'ms',  '.2f')
    bar(fig.add_subplot(gs[1, 1]), [r['prunable_params']for r in results.values()],'Prunable Params','n',   ',')
    bar(fig.add_subplot(gs[1, 2]), [r['sparsity']*100  for r in results.values()], 'Sparsity (%)',   '%',   '.1f')

    fig.suptitle('KAN vs MLP Benchmark Summary — CICIDS2016',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_path}')
