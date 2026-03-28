"""
feature_engineering.py  —  selection, correlation, importance plots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.filterwarnings('ignore')


def remove_low_variance(X, feature_names, threshold=1e-4):
    mask = np.var(X, axis=0) > threshold
    X2   = X[:, mask]
    fn2  = [n for n, m in zip(feature_names, mask) if m]
    print(f'  Low-variance removed: {sum(~mask)} -> {len(fn2)} features remain')
    return X2, fn2, mask


def correlation_analysis(X, feature_names, threshold=0.95, plot=True):
    df   = pd.DataFrame(X, columns=feature_names)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    remaining = [f for f in feature_names if f not in to_drop]
    print(f'  Correlation threshold={threshold}: dropping {len(to_drop)}, keeping {len(remaining)}')
    if plot:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap='RdYlBu_r', xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=110, bbox_inches='tight')
        plt.show(); print('  Saved: correlation_heatmap.png')
    return to_drop, remaining


def mutual_info_selection(X_train, y_train, feature_names, top_k=40, plot=True):
    print(f'  Computing mutual info for {X_train.shape[1]} features...')
    scores  = mutual_info_classif(X_train, y_train, random_state=42)
    series  = pd.Series(scores, index=feature_names).sort_values(ascending=False)
    top_k   = min(top_k, len(feature_names))
    top_f   = series.head(top_k).index.tolist()
    indices = [feature_names.index(f) for f in top_f]
    if plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        series.head(top_k).plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
        ax.set_title(f'Top-{top_k} Features by Mutual Information', fontweight='bold')
        ax.tick_params(axis='x', labelsize=6)
        plt.tight_layout()
        plt.savefig('mutual_info_importance.png', dpi=110, bbox_inches='tight')
        plt.show(); print('  Saved: mutual_info_importance.png')
    return top_f, indices, series


def tree_importance(X_train, y_train, feature_names, top_k=40, plot=True, max_samples=20000):
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    print(f'  Training RF on {len(X_train):,} samples for feature importance...')
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    imp  = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    top_k = min(top_k, len(feature_names))
    top_f = imp.head(top_k).index.tolist()
    indices = [feature_names.index(f) for f in top_f]
    if plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        imp.head(top_k).plot(kind='bar', ax=ax, color='darkorange', edgecolor='white')
        ax.set_title(f'Top-{top_k} Features by RF Importance', fontweight='bold')
        ax.tick_params(axis='x', labelsize=6)
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png', dpi=110, bbox_inches='tight')
        plt.show(); print('  Saved: rf_feature_importance.png')
    return top_f, indices, imp


def plot_feature_distributions(X, y, feature_names, top_n=12):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    classes = np.unique(y)
    colors  = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    for i, (feat, ax) in enumerate(zip(feature_names[:top_n], axes)):
        for cls, col in zip(classes, colors):
            vals = X[y == cls, i]
            lbl  = 'BENIGN' if cls == 0 else 'ATTACK'
            ax.hist(vals, bins=50, alpha=0.6, color=col, label=lbl, density=True)
        ax.set_title(feat[:28], fontsize=8)
        ax.legend(fontsize=7)
    plt.suptitle('Feature Distributions by Class', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=110, bbox_inches='tight')
    plt.show(); print('  Saved: feature_distributions.png')


def run_feature_engineering(data, method='mutual_info', top_k=40, corr_threshold=0.95):
    print('\n' + '='*60 + '\n  FEATURE ENGINEERING\n' + '='*60)
    X_tr, X_val, X_te = data['X_train'], data['X_val'], data['X_test']
    y_tr = data['y_train']
    fn   = data['feature_names']

    X_tr, fn, mask = remove_low_variance(X_tr, fn)
    X_val = X_val[:, mask]; X_te = X_te[:, mask]

    to_drop, remaining = correlation_analysis(X_tr, fn, corr_threshold)
    keep = [i for i, n in enumerate(fn) if n not in to_drop]
    X_tr = X_tr[:, keep]; X_val = X_val[:, keep]; X_te = X_te[:, keep]; fn = remaining

    top_k = min(top_k, len(fn))
    if method == 'mutual_info':
        top_f, indices, scores = mutual_info_selection(X_tr, y_tr, fn, top_k)
    else:
        top_f, indices, scores = tree_importance(X_tr, y_tr, fn, top_k)

    X_tr = X_tr[:, indices]; X_val = X_val[:, indices]; X_te = X_te[:, indices]; fn = top_f
    plot_feature_distributions(X_tr, y_tr, fn, top_n=12)

    print(f'  Final features: {len(fn)}')
    print('='*60 + '\n')
    data.update(X_train=X_tr.astype(np.float32), X_val=X_val.astype(np.float32),
                X_test=X_te.astype(np.float32), n_features=len(fn),
                feature_names=fn, feature_scores=scores)
    return data
