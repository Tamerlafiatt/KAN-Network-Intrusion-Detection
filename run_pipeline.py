"""
run_pipeline.py  —  resumable pipeline, full dataset, fast KAN
================================================================
Speed strategy:
  - FastKAN (RBF kernels) instead of B-spline KAN: ~8x faster per epoch
  - sample_frac=0.3  — 30% of full dataset (~840k rows on real CICIDS2016)
    Still large enough for meaningful results, finishes in ~1-2h on CPU
  - batch_size=2048  — larger batches = fewer Python loop iterations
  - epochs=30        — enough for convergence with cosine LR + early stopping
  - finetune_ep=10   — enough recovery after pruning
  - DataLoader num_workers=2 — parallel data loading
  - torch.set_num_threads     — use all CPU cores

Resumable: re-run Cell 12 after any interruption.
"""

import os, sys, json, time, pickle, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '.')
os.makedirs('./data',        exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)

# Use all CPU cores
torch.set_num_threads(os.cpu_count() or 4)
print(f'Using {torch.get_num_threads()} CPU threads')

from data_loader         import prepare_data
from feature_engineering import run_feature_engineering
from model_kan           import build_kan
from model_mlp           import build_mlp
from pruning             import (magnitude_prune, structured_prune_mlp,
                                  structured_prune_kan, count_nonzero_params)
from train               import train_model, finetune_pruned
from evaluate            import evaluate_model
from metrics             import (print_comparison_table, print_tradeoff_analysis,
                                  print_per_class_f1)
from visualization       import (plot_training_curves,
                                  plot_kan_vs_mlp,
                                  plot_efficiency_comparison,
                                  plot_pruning_impact,
                                  plot_confusion_matrices,
                                  plot_roc_curves,
                                  plot_per_class_f1,
                                  plot_accuracy_vs_inference,
                                  plot_params_vs_performance,
                                  visualize_kan_functions,
                                  visualize_network_structure,
                                  plot_summary_dashboard)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir        = './data',
    mode            = 'binary',       # 'binary' or 'multiclass'
    sample_frac     = 0.3,            # 30% of dataset → real results, ~1-2h
                                      # set 1.0 for full run (~6h)
                                      # set 0.1 for quick test (~15min)
    imbalance       = 'smote',
    feature_method  = 'mutual_info',
    top_k           = 40,

    # ── Architecture ──────────────────────────────────────────────────────────
    hidden_sizes    = [128, 64],
    grid_size       = 8,              # RBF centres per feature (FastKAN)
    spline_order    = 3,              # only used if fast=False
    dropout         = 0.1,
    fast_kan        = True,           # True = RBF-KAN (~8x faster than B-spline)
                                      # False = original B-spline KAN (slower, slightly better)

    # ── Training ──────────────────────────────────────────────────────────────
    epochs          = 30,
    batch_size      = 2048,           # larger batch = faster epochs
    lr              = 1e-3,
    weight_decay    = 1e-4,
    patience        = 8,
    finetune_ep     = 10,
    num_workers     = 2,              # parallel data loading

    # ── Pruning ───────────────────────────────────────────────────────────────
    sparsity_levels = [0.2, 0.5, 0.8],

    # ── System ────────────────────────────────────────────────────────────────
    device          = 'cpu',
    seed            = 42,
)

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])

print('KAN vs MLP on CICIDS2016')
print(f'  sample_frac={CFG["sample_frac"]}  fast_kan={CFG["fast_kan"]}  '
      f'batch={CFG["batch_size"]}  epochs={CFG["epochs"]}')


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
CKPT = './checkpoints'
STATE_FILE = os.path.join(CKPT, 'pipeline_state.json')

def _state():
    return json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}

def _save(state):
    json.dump(state, open(STATE_FILE, 'w'), indent=2)

def ck(name):
    return os.path.join(CKPT, name)

def done(state, key):
    return state.get(key, False)

def mark(state, key):
    state[key] = True; _save(state)
    print(f'  ✓ {key}')

def save_data(d):
    payload = {k: v for k, v in d.items() if k not in ('label_encoder','scaler')}
    pickle.dump(payload,        open(ck('data.pkl'),   'wb'))
    if d.get('scaler'):         pickle.dump(d['scaler'],        open(ck('scaler.pkl'), 'wb'))
    if d.get('label_encoder'):  pickle.dump(d['label_encoder'], open(ck('le.pkl'),     'wb'))

def load_data():
    d = pickle.load(open(ck('data.pkl'), 'rb'))
    for key, f in [('scaler','scaler.pkl'),('label_encoder','le.pkl')]:
        if os.path.exists(ck(f)): d[key] = pickle.load(open(ck(f),'rb'))
    return d

def svm(model, name):   torch.save(model.state_dict(), ck(f'{name}.pt'))
def lvm(model, name):   model.load_state_dict(torch.load(ck(f'{name}.pt'), map_location='cpu')); return model
def svx(obj,   name):   pickle.dump(obj,  open(ck(f'{name}.pkl'),'wb'))
def ldx(name):          return pickle.load(open(ck(f'{name}.pkl'),'rb'))
def svt(name,  t):      json.dump({'t':t}, open(ck(f'{name}_t.json'),'w'))
def ldt(name):
    p = ck(f'{name}_t.json')
    return json.load(open(p))['t'] if os.path.exists(p) else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
state = _state()
done_list = [k for k, v in state.items() if v]
print(f'\nDone: {done_list if done_list else "none — fresh start"}\n')


# ── Step 1: Data ───────────────────────────────────────────────────────────────
if done(state, 'data'):
    print('SKIP  Step 1 — data')
    data = load_data()
else:
    print('RUN   Step 1 — data')
    data = prepare_data(
        data_dir=CFG['data_dir'], mode=CFG['mode'],
        sample_frac=CFG['sample_frac'], imbalance_strategy=CFG['imbalance'],
        random_state=CFG['seed'])
    data = run_feature_engineering(
        data, method=CFG['feature_method'], top_k=CFG['top_k'])
    save_data(data); mark(state, 'data')

N, C = data['n_features'], data['n_classes']
CLASS_NAMES = ['BENIGN', 'ATTACK'] if C == 2 else None
print(f'Features={N}  Classes={C}  '
      f'train={len(data["X_train"]):,}  val={len(data["X_val"]):,}  test={len(data["X_test"]):,}')


# ── Step 2: Build skeletons ────────────────────────────────────────────────────
print('\nRUN   Step 2 — build skeletons')

def _build_kan():
    return build_kan(N, C, CFG['hidden_sizes'], CFG['grid_size'],
                     CFG['spline_order'], CFG['dropout'],
                     use_poly=False, fast=CFG['fast_kan'])

def _build_poly():
    return build_kan(N, C, CFG['hidden_sizes'], dropout=CFG['dropout'],
                     use_poly=True, poly_degree=4)

def _build_mlp():
    return build_mlp(N, C, CFG['hidden_sizes'], CFG['dropout'])

kan_model      = _build_kan()
poly_kan_model = _build_poly()
mlp_model      = _build_mlp()


# ── Step 3: Train base models ──────────────────────────────────────────────────
print('\n--- Step 3: Train base models ---')
histories, train_times = {}, {}

for name, builder in [('KAN',_build_kan),('PolyKAN',_build_poly),('MLP',_build_mlp)]:
    key = f'train_{name}'
    if done(state, key):
        print(f'SKIP  {name}')
        model             = lvm(builder(), name)
        histories[name]   = ldx(f'{name}_hist')
        train_times[name] = ldt(name)
    else:
        print(f'RUN   {name}')
        model = builder()
        res = train_model(
            model, data,
            epochs       = CFG['epochs'],
            batch_size   = CFG['batch_size'],
            lr           = CFG['lr'],
            weight_decay = CFG['weight_decay'],
            patience     = CFG['patience'],
            device_str   = CFG['device'],
            num_workers  = CFG['num_workers'],
        )
        model             = res['model']
        histories[name]   = res['history']
        train_times[name] = res['train_time']
        svm(model, name); svx(res['history'], f'{name}_hist'); svt(name, res['train_time'])
        mark(state, key)

    if name == 'KAN':       kan_model      = model
    elif name == 'PolyKAN': poly_kan_model = model
    else:                   mlp_model      = model


# ── Step 4: Pruning ────────────────────────────────────────────────────────────
print('\n--- Step 4: Pruning ---')
pruned_models = {}

prune_jobs = []
for sp in CFG['sparsity_levels']:
    lbl = f'{int(sp*100)}pct'
    prune_jobs += [
        (f'KAN_mag_{lbl}', magnitude_prune,      sp, kan_model, 'KAN', _build_kan),
        (f'KAN_str_{lbl}', structured_prune_kan, sp, kan_model, 'KAN', _build_kan),
        (f'MLP_mag_{lbl}', magnitude_prune,      sp, mlp_model, 'MLP', _build_mlp),
        (f'MLP_str_{lbl}', structured_prune_mlp, sp, mlp_model, 'MLP', _build_mlp),
    ]

for name, prune_fn, sp, base, arch, builder in prune_jobs:
    key = f'prune_{name}'
    if done(state, key):
        print(f'SKIP  {name}')
        sk                = lvm(builder(), name)
        histories[name]   = ldx(f'{name}_hist')
        train_times[name] = ldt(name)
    else:
        print(f'RUN   {name}')
        pm  = prune_fn(base, sparsity=sp)
        ft  = finetune_pruned(pm, data, epochs=CFG['finetune_ep'],
                              batch_size=CFG['batch_size'],
                              device_str=CFG['device'],
                              num_workers=CFG['num_workers'])
        sk                = ft['model']
        histories[name]   = ft['history']
        train_times[name] = ft['train_time']
        svm(sk, name); svx(ft['history'], f'{name}_hist'); svt(name, ft['train_time'])
        mark(state, key)
    pruned_models[name] = sk


# ── Step 5: Evaluate ───────────────────────────────────────────────────────────
print('\n--- Step 5: Evaluate ---')
all_results  = {}
eval_targets = (
    [('KAN',kan_model),('PolyKAN',poly_kan_model),('MLP',mlp_model)]
    + list(pruned_models.items())
)

for name, model in eval_targets:
    key = f'eval_{name}'
    if done(state, key):
        print(f'SKIP  {name}')
        r = ldx(f'{name}_result')
    else:
        print(f'RUN   {name}')
        r               = evaluate_model(model, data, split='test',
                                         device_str=CFG['device'])
        r['train_time'] = train_times.get(name, 0)
        svx(r, f'{name}_result'); mark(state, key)
    all_results[name] = r


# ── Step 6: Tables ─────────────────────────────────────────────────────────────
print('\n--- Step 6: Results ---')
df = print_comparison_table(all_results)
print_tradeoff_analysis(all_results)
print_per_class_f1(all_results, class_names=CLASS_NAMES)
df.to_csv('benchmark_results.csv', index=False)
print('Saved: benchmark_results.csv')


# ── Step 7: Visualizations ─────────────────────────────────────────────────────
if done(state, 'viz'):
    print('\nSKIP  Step 7 — visualizations')
else:
    print('\n--- Step 7: Visualizations ---')

    plot_training_curves({k: histories[k] for k in ('KAN','PolyKAN','MLP')})
    plot_kan_vs_mlp(all_results)
    plot_efficiency_comparison(all_results)
    plot_pruning_impact(all_results)

    sp_mid = f'{int(CFG["sparsity_levels"][len(CFG["sparsity_levels"])//2]*100)}pct'
    cm_keys = ['KAN','MLP','PolyKAN',
               f'KAN_mag_{sp_mid}',f'KAN_str_{sp_mid}',
               f'MLP_mag_{sp_mid}',f'MLP_str_{sp_mid}']
    plot_confusion_matrices(
        {k: all_results[k] for k in cm_keys if k in all_results},
        class_names=CLASS_NAMES)
    plot_roc_curves({k: all_results[k] for k in cm_keys if k in all_results})
    plot_per_class_f1(all_results, class_names=CLASS_NAMES)
    plot_accuracy_vs_inference(all_results)
    plot_params_vs_performance(all_results)
    visualize_kan_functions(kan_model, data['feature_names'])
    visualize_network_structure(all_results)

    dash_keys = ['KAN','PolyKAN','MLP',
                 f'KAN_mag_{sp_mid}',f'KAN_str_{sp_mid}',
                 f'MLP_mag_{sp_mid}',f'MLP_str_{sp_mid}']
    plot_summary_dashboard({k: all_results[k] for k in dash_keys if k in all_results})
    mark(state, 'viz')


# ── Step 8: CryptoKAN degree sweep ────────────────────────────────────────────
print('\n--- Step 8: CryptoKAN degree sweep ---')
poly_res = {}
for deg in [2, 3, 4, 6, 8]:
    key  = f'PolyKAN_d{deg}'
    skey = f'crypto_{deg}'
    pm   = build_kan(N, C, CFG['hidden_sizes'], dropout=CFG['dropout'],
                     use_poly=True, poly_degree=deg)
    if done(state, skey):
        print(f'SKIP  degree={deg}')
        ev = ldx(f'{key}_result')
    else:
        print(f'RUN   degree={deg}')
        tr = train_model(pm, data, epochs=CFG['epochs'],
                         batch_size=CFG['batch_size'], lr=CFG['lr'],
                         patience=CFG['patience'], device_str=CFG['device'],
                         num_workers=CFG['num_workers'], verbose=False)
        ev               = evaluate_model(tr['model'], data, split='test',
                                          device_str=CFG['device'], verbose=False)
        ev['train_time'] = tr['train_time']
        svm(tr['model'], key); svx(ev, f'{key}_result'); mark(state, skey)
    poly_res[key] = ev
    print(f'  d={deg}: acc={ev["accuracy"]:.4f} f1={ev["f1"]:.4f} '
          f'{ev["inference_ms"]:.2f}ms sparsity={ev["sparsity"]:.1%}')

if not done(state, 'cryptokan_sweep'):
    mark(state, 'cryptokan_sweep')

degrees = [int(k.split('d')[1]) for k in poly_res]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric, ylabel, color in zip(axes,
        ['accuracy','f1','inference_ms'],
        ['Accuracy','F1','Inference(ms)'],
        ['#2563EB','#16A34A','#DC2626']):
    vals = [poly_res[f'PolyKAN_d{d}'][metric] for d in degrees]
    ax.plot(degrees, vals, 'o-', color=color, lw=2.5, ms=9,
            markeredgecolor='white', markeredgewidth=1.5)
    for d, v in zip(degrees, vals):
        ax.annotate(f'{v:.4f}', (d, v), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=8)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel(ylabel)
    ax.set_title(f'CryptoKAN: Degree vs {ylabel}', fontweight='bold')
plt.suptitle('CryptoKAN Polynomial Degree Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cryptokan_degree_analysis.png', dpi=130, bbox_inches='tight')
plt.show()
print_comparison_table(poly_res)

print('\n' + '='*60)
print('PIPELINE COMPLETE')
print('='*60)
print('Files:', sorted(glob.glob('*.png')) + sorted(glob.glob('*.csv')))
