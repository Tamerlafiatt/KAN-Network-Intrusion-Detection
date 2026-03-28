"""
train.py  —  training loop with num_workers support for faster data loading
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def make_loaders(data, batch_size=1024, num_workers=2):
    def _ld(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=False)
    return (_ld(data['X_train'], data['y_train'], True),
            _ld(data['X_val'],   data['y_val'],   False),
            _ld(data['X_test'],  data['y_test'],  False))


def _class_weights(y, n_classes):
    c = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0 / (c + 1)
    return torch.tensor(w / w.sum() * n_classes, dtype=torch.float32)


def _train_epoch(model, loader, optim, crit, device, clip, pruner):
    model.train()
    loss_sum = correct = total = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optim.zero_grad()
        out  = model(Xb)
        loss = crit(out, yb)
        loss.backward()
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()
        loss_sum += loss.item() * len(yb)
        correct  += (out.argmax(1) == yb).sum().item()
        total    += len(yb)
        if pruner: pruner.step()
    return loss_sum / total, correct / total


@torch.no_grad()
def _eval_epoch(model, loader, crit, device):
    model.eval()
    loss_sum = correct = total = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out  = model(Xb)
        loss = crit(out, yb)
        loss_sum += loss.item() * len(yb)
        correct  += (out.argmax(1) == yb).sum().item()
        total    += len(yb)
    return loss_sum / total, correct / total


def train_model(model, data,
                epochs=30, batch_size=1024, lr=1e-3, weight_decay=1e-4,
                patience=8, use_class_weights=True, grad_clip=1.0,
                scheduler_type='cosine', gradual_prune=False, final_sparsity=0.5,
                device_str='cpu', num_workers=2, verbose=True):

    device = torch.device(device_str)
    model  = model.to(device)
    tr_ld, val_ld, _ = make_loaders(data, batch_size, num_workers)

    cw    = (_class_weights(data['y_train'], data['n_classes']).to(device)
             if use_class_weights else None)
    crit  = nn.CrossEntropyLoss(weight=cw)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.01)
             if scheduler_type == 'cosine'
             else ReduceLROnPlateau(optim, patience=4, factor=0.5))

    pruner = None
    if gradual_prune:
        from pruning import GradualPruner
        pruner = GradualPruner(model, final_sparsity=final_sparsity,
                               begin_step=2, end_step=epochs - 2)

    hist = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], lr=[])
    best_val, best_state, no_imp = float('inf'), None, 0
    t0_total = time.time()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_l, tr_a   = _train_epoch(model, tr_ld, optim, crit, device, grad_clip, pruner)
        val_l, val_a = _eval_epoch(model, val_ld, crit, device)

        if scheduler_type == 'cosine': sched.step()
        else:                          sched.step(val_l)

        cur_lr = optim.param_groups[0]['lr']
        for k, v in zip(['train_loss','val_loss','train_acc','val_acc','lr'],
                        [tr_l, val_l, tr_a, val_a, cur_lr]):
            hist[k].append(v)

        if val_l < best_val - 1e-4:
            best_val = val_l
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1

        if verbose and (ep % 5 == 0 or ep == 1):
            elapsed = time.time() - t0_total
            remaining = elapsed / ep * (epochs - ep)
            print(f'  ep {ep:3d}/{epochs} | '
                  f'loss {tr_l:.4f}/{val_l:.4f} | '
                  f'acc {tr_a:.4f}/{val_a:.4f} | '
                  f'lr {cur_lr:.1e} | '
                  f'{time.time()-t0:.1f}s/ep | '
                  f'~{remaining/60:.0f}min left')

        if no_imp >= patience:
            if verbose: print(f'  Early stop @ epoch {ep}')
            break

    total_time = time.time() - t0_total
    if best_state: model.load_state_dict(best_state)
    if verbose: print(f'  Done in {total_time/60:.1f}min')
    return dict(model=model, history=hist, train_time=total_time)


def finetune_pruned(model, data, epochs=10, batch_size=1024,
                    lr=1e-4, device_str='cpu', num_workers=2):
    print('  Fine-tuning...')
    return train_model(model, data,
                       epochs=epochs, batch_size=batch_size,
                       lr=lr, patience=5, device_str=device_str,
                       num_workers=num_workers, verbose=False)
