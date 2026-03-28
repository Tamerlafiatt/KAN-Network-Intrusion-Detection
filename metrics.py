"""
metrics.py  —  comparison table and trade-off analysis
Updated to show prunable_params and remaining_params separately,
and to print per-class F1 breakdown.
"""
import numpy as np
import pandas as pd
from tabulate import tabulate


def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            'Model':            name,
            'Accuracy':         f"{r['accuracy']:.4f}",
            'F1':               f"{r['f1']:.4f}",
            'Precision':        f"{r['precision']:.4f}",
            'Recall':           f"{r['recall']:.4f}",
            'ROC-AUC':          f"{r['roc_auc']:.4f}",
            'Total Params':     f"{r['total_params']:,}",
            'Prunable Params':  f"{r['prunable_params']:,}",
            'Remaining':        f"{r['remaining_params']:,}",
            'Sparsity':         f"{r['sparsity']:.1%}",
            'Train(s)':         f"{r.get('train_time', 0):.1f}",
            'Infer(ms)':        f"{r['inference_ms']:.2f}±{r.get('inference_std_ms',0):.2f}",
            'Size(MB)':         f"{r['model_size_mb']:.3f}",
        })
    return pd.DataFrame(rows)


def print_comparison_table(results: dict) -> pd.DataFrame:
    df = build_comparison_table(results)
    print('\n' + '='*90)
    print('  BENCHMARK COMPARISON')
    print('='*90)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print('='*90)
    return df


def print_per_class_f1(results: dict, class_names: list = None):
    print('\n' + '='*70)
    print('  PER-CLASS F1 BREAKDOWN')
    print('='*70)
    for name, r in results.items():
        f1s = r.get('f1_per_class', [])
        if len(f1s) == 0:
            continue
        parts = []
        for i, v in enumerate(f1s):
            lbl = class_names[i] if class_names and i < len(class_names) else f'class_{i}'
            parts.append(f'{lbl}={v:.4f}')
        print(f'  {name:<20s}  {" | ".join(parts)}')
    print('='*70)


def print_pruning_impact(results: dict):
    """
    For each pruned model, show accuracy drop and speed change vs its base.
    """
    print('\n' + '='*70)
    print('  PRUNING IMPACT (vs unpruned base)')
    print('='*70)
    print(f'  {"Model":<22} {"Acc Drop":>10} {"F1 Drop":>10} '
          f'{"Sparsity":>10} {"Speed Δms":>12} {"SizeΔMB":>10}')
    print('  ' + '-'*75)
    for name, r in results.items():
        if r['sparsity'] < 0.01:
            continue
        # Find base: 'KAN_mag_50pct' -> base='KAN'
        base_name = name.split('_')[0]
        if base_name not in results:
            continue
        base = results[base_name]
        acc_drop   = base['accuracy']     - r['accuracy']
        f1_drop    = base['f1']           - r['f1']
        speed_gain = base['inference_ms'] - r['inference_ms']
        size_gain  = base['model_size_mb']- r['model_size_mb']
        print(f'  {name:<22} {acc_drop:>+10.4f} {f1_drop:>+10.4f} '
              f'{r["sparsity"]:>10.1%} {speed_gain:>+12.2f} {size_gain:>+10.3f}')
    print('='*70)


def print_tradeoff_analysis(results: dict):
    names = list(results.keys())
    rs    = list(results.values())

    def best(key, fn=max):
        r = fn(rs, key=lambda x: x[key])
        return names[rs.index(r)], r[key]

    print('\n' + '='*60)
    print('  TRADE-OFF ANALYSIS')
    print('='*60)

    n, v = best('accuracy');                print(f'  Best Accuracy:      {n} ({v:.4f})')
    n, v = best('f1');                      print(f'  Best F1:            {n} ({v:.4f})')
    n, v = best('roc_auc');                 print(f'  Best ROC-AUC:       {n} ({v:.4f})')
    n, v = best('inference_ms', fn=min);    print(f'  Fastest Inference:  {n} ({v:.2f}ms)')
    n, v = best('model_size_mb', fn=min);   print(f'  Smallest Model:     {n} ({v:.3f}MB)')
    n, v = best('sparsity');                print(f'  Most Sparse:        {n} ({v:.1%})')

    # KAN vs MLP base comparison
    base_kan = results.get('KAN')
    base_mlp = results.get('MLP')
    if base_kan and base_mlp:
        print(f'\n  KAN vs MLP (base models):')
        print(f'    Accuracy:   KAN={base_kan["accuracy"]:.4f}  MLP={base_mlp["accuracy"]:.4f}  '
              f'delta={base_kan["accuracy"]-base_mlp["accuracy"]:+.4f}')
        print(f'    F1:         KAN={base_kan["f1"]:.4f}  MLP={base_mlp["f1"]:.4f}  '
              f'delta={base_kan["f1"]-base_mlp["f1"]:+.4f}')
        print(f'    Inference:  KAN={base_kan["inference_ms"]:.2f}ms  '
              f'MLP={base_mlp["inference_ms"]:.2f}ms  '
              f'(MLP is {base_kan["inference_ms"]/base_mlp["inference_ms"]:.1f}x faster)')
        print(f'    Params:     KAN={base_kan["total_params"]:,}  '
              f'MLP={base_mlp["total_params"]:,}  '
              f'(KAN has {base_kan["total_params"]/base_mlp["total_params"]:.1f}x more)')

    # PolyKAN vs KAN
    base_poly = results.get('PolyKAN')
    if base_poly and base_kan:
        print(f'\n  PolyKAN vs KAN:')
        print(f'    Accuracy:   PolyKAN={base_poly["accuracy"]:.4f}  KAN={base_kan["accuracy"]:.4f}  '
              f'delta={base_poly["accuracy"]-base_kan["accuracy"]:+.4f}')
        print(f'    Inference:  PolyKAN={base_poly["inference_ms"]:.2f}ms  '
              f'KAN={base_kan["inference_ms"]:.2f}ms  '
              f'(PolyKAN is {base_kan["inference_ms"]/base_poly["inference_ms"]:.1f}x faster)')
        print(f'    Params:     PolyKAN={base_poly["total_params"]:,}  '
              f'KAN={base_kan["total_params"]:,}')

    print_pruning_impact(results)
    print('='*60 + '\n')
