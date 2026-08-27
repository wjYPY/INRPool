"""HA-PoolINR training & publishing pipeline.

End-to-end flow:
  1. Build one global shared parameter pool (the ONLY trainable tensors).
  2. Assemble one key-derived network per task (cover + secrets) that reads
     pool entries through a key-derived permutation and sign mask.
  3. Jointly optimize the pool with the equal-weight sum of the per-task
     MSE losses; the pool reached at the end of the budget is published.
  4. Publish ONLY the cover network as a normal-looking INR state dict.
  5. Verify: decode the published cover back into the pool with the cover key,
     re-assemble every task network from the recovered pool, evaluate
     reconstruction quality and wrong-key resistance.

Paper protocol (used for all reported numbers):
  epochs = 20000 (fixed, no early stop), full-batch, AdamW lr = 3e-3 with
  cosine annealing, gradient-norm clipping = 1.0. The joint loss is the
  equal-weight SUM of the per-task MSE losses (no task-count averaging),
  and the final pool at the end of the budget is published directly
  (no checkpoint selection).
"""
import os, torch, torch.optim as optim, torch.nn.functional as F
from pool    import TypedParamPool
from filler  import FillerProtocol
from models  import PoolNetwork
from configs import (detect_modality, MODALITY_CFG, build_arch,
                     fourier_args_from_cfg, count_params,
                     POOL_W_CAPACITY, POOL_B_CAPACITY)
from data_io import load_data, SAVERS
from metrics import EVALUATORS, pretty_print

MODALITY_WEIGHTS = {'cover': 1.0, 'image': 1.0, 'audio': 1.0, 'video': 1.0}


def _get_task_weight(task: dict) -> float:
    if task['role'] == 'cover':
        return MODALITY_WEIGHTS['cover']
    return MODALITY_WEIGHTS.get(task['modality'], 1.0)


def run(cover_path: str, secret_paths: list,
        epochs: int = 20000, batch_size: int = 10**12, lr: float = 3e-3,
        pool_w: int = POOL_W_CAPACITY, pool_b: int = POOL_B_CAPACITY,
        out_dir: str = './outputs', artifacts_dir: str = './artifacts',
        device: str = None, early_stop_loss: float = None, grad_clip: float = 1.0,
        init_scheme: str = 'kaiming_normal',
        cover_key: str = 'COVER_KEY_V1', secret_keys: list = None):
    """Train the shared pool and publish the cover network.

    Args:
        cover_path:     path to the cover media (image / audio / video).
        secret_paths:   paths to the secret media (any modality, mixed allowed).
        epochs:         total training epochs (paper protocol: 20000).
        batch_size:     samples per step; >= dataset size means full-batch.
        lr:             AdamW learning rate (paper protocol: 3e-3, cosine).
        early_stop_loss: None = fixed budget (paper protocol); set e.g. 1e-7
                        to stop early once the total loss is tiny.
        cover_key:      key string for the cover network. Use your own secret!
        secret_keys:    optional list of key strings, one per secret. Defaults
                        to 'SECRET_KEY_{i}_V1'.
    Returns:
        dict of per-task evaluation metrics.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"========== HA-PoolINR (device={device}) ==========")

    # ── task construction ─────────────────────────────────
    tasks = []
    c_mod = detect_modality(cover_path)
    tasks.append({'role': 'cover', 'key': cover_key,
                  'modality': c_mod, 'cfg': MODALITY_CFG[c_mod], 'path': cover_path})
    print(f"[Cover] {cover_path}  →  modality={c_mod}")

    for i, sp in enumerate(secret_paths):
        mod = detect_modality(sp)
        key = secret_keys[i] if secret_keys and i < len(secret_keys) else f'SECRET_KEY_{i}_V1'
        tasks.append({'role': f'secret{i}', 'key': key,
                      'modality': mod, 'cfg': MODALITY_CFG[mod], 'path': sp})
        print(f"[Secret{i}] {sp}  →  modality={mod}")

    N_W, N_b = pool_w, pool_b
    print(f"\n[Pool] capacity N_W={N_W:,}, N_b={N_b:,}")

    # ── architecture check & optional cover auto-widening ─
    for t in tasks:
        t['arch'] = build_arch(t['cfg'])
        nW, nb = count_params(t['arch'])
        print(f"  {t['role']:<9s}({t['modality']:<5}): nW={nW:<7,} nb={nb:<5}  "
              f"r_W={nW/N_W:.2f}  r_b={nb/N_b:.2f}")

    cover_nW, cover_nb = count_params(tasks[0]['arch'])
    if cover_nW < N_W or cover_nb < N_b:
        print(f"\n[Capacity Check] cover network smaller than pool, auto-widening (hidden += 32)...")
        while True:
            cover_nW, cover_nb = count_params(build_arch(tasks[0]['cfg']))
            if cover_nW >= N_W and cover_nb >= N_b:
                break
            tasks[0]['cfg']['hidden'] += 32
        tasks[0]['arch'] = build_arch(tasks[0]['cfg'])
        cover_nW, cover_nb = count_params(tasks[0]['arch'])
        print(f"   widened cover to hidden={tasks[0]['cfg']['hidden']} "
              f"(nW={cover_nW:,}, nb={cover_nb:,})")
    else:
        print(f"\n[Capacity Check] cover network covers the pool.")

    # ── data loading ──────────────────────────────────────
    for t in tasks:
        t['coords'], t['target'], t['meta'] = load_data(
            t['modality'], t['path'], t['cfg'], device)

    # ── pool & per-task key-derived networks ──────────────
    pool = TypedParamPool(n_w=N_W, n_b=N_b, device=device, init_scheme=init_scheme)
    pool.P_W.requires_grad_(True)
    pool.P_b.requires_grad_(True)

    nets = {
        t['role']: PoolNetwork(
            key=t['key'], arch=t['arch'], pool_w_size=N_W, pool_b_size=N_b,
            use_fourier=t['cfg']['use_fourier'],
            fourier_args=fourier_args_from_cfg(t['cfg']),
            is_siren=t['cfg']['is_siren'], device=device, leaky_slope=0.01,
        ).to(device)
        for t in tasks
    }

    # ── optimizer & schedule ──────────────────────────────
    optimizer = optim.AdamW([pool.P_W, pool.P_b], lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    task_weights = {t['role']: _get_task_weight(t) for t in tasks}

    log_interval = max(1, epochs // 20)

    # ── training loop ─────────────────────────────────────
    print("\n[Training] joint optimization (pure MSE, full-batch overfitting) ...")
    print(f"  loss = equal-weight sum over tasks | grad clip max_norm={grad_clip} | "
          f"early_stop={early_stop_loss}")

    for ep in range(epochs + 1):
        optimizer.zero_grad()
        losses = {}

        for t in tasks:
            n = t['coords'].shape[0]

            # sample indices (random subset for very large data)
            if batch_size >= n:
                idx = torch.arange(n, device=device)
            else:
                idx = torch.randperm(n, device=device)[:batch_size]

            pred = nets[t['role']](t['coords'][idx], pool.P_W, pool.P_b)
            losses[t['role']] = F.mse_loss(pred, t['target'][idx])

        # Joint loss: equal-weight sum of the per-task losses (paper Sec. 3.6).
        total = sum(task_weights[r] * losses[r] for r in losses)

        # skip update on non-finite loss (scheduler too, to keep epoch count in sync)
        if not torch.isfinite(total):
            print(f"!  Epoch {ep}: non-finite loss ({total.item():.4f}), skip update")
            continue

        total.backward()
        torch.nn.utils.clip_grad_norm_([pool.P_W, pool.P_b], max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        if ep % log_interval == 0:
            parts = " | ".join(f"{r}:{losses[r].item():.5f}" for r in losses)
            print(f"Epoch {ep:5d} [lr:{scheduler.get_last_lr()[0]:.1e}] "
                  f"| Total {total.item():.4f} | {parts}")

        if early_stop_loss is not None and ep > 1000 and total.item() < early_stop_loss:
            print(f"\n[EarlyStopping] Epoch {ep}: total loss={total.item():.2e} "
                  f"< {early_stop_loss:.0e}, stop early")
            break

    # ── publish stage (final pool at the end of the budget) ──
    filler = FillerProtocol(device=device)
    with torch.no_grad():
        cover_state = filler.fill_model(pool, tasks[0]['key'], tasks[0]['arch'])
    cover_publish = {k: v['data'].detach().cpu() for k, v in cover_state.items()}
    torch.save(cover_publish, os.path.join(artifacts_dir, 'cover_published.pth'))
    torch.save({'P_W': pool.P_W.detach().cpu(), 'P_b': pool.P_b.detach().cpu()},
               os.path.join(artifacts_dir, 'real_pool_GT.pth'))

    print("\n[Publish] saved cover_published.pth (public) and real_pool_GT.pth (sender-side reference)")

    # ── decode verification: public cover + key → pool ────
    cover_tensors = {
        n: {'data': cover_publish[n].to(device),
            'type': tasks[0]['arch'][n]['type'],
            's_tau': tasks[0]['arch'][n]['s_tau']}
        for n in cover_publish
    }
    rec_P_W, rec_P_b, cnt_W, cnt_b = filler.decode_to_pool(
        tensors=cover_tensors, key=tasks[0]['key'], arch=tasks[0]['arch'],
        pool_w_size=N_W, pool_b_size=N_b)

    gt = torch.load(os.path.join(artifacts_dir, 'real_pool_GT.pth'),
                    map_location=device)
    err_W = (rec_P_W - gt['P_W'].to(device)).abs().max().item()
    err_b = (rec_P_b - gt['P_b'].to(device)).abs().max().item()

    print(f"[Decode] coverage -> uncovered W: {(cnt_W==0).sum().item()}/{N_W}, "
          f"b: {(cnt_b==0).sum().item()}/{N_b}")
    print(f"[Decode] accuracy -> max|dP_W|={err_W:.3e}, max|dP_b|={err_b:.3e}")

    # ── re-assemble networks from recovered pool & evaluate ─
    rec_pool = TypedParamPool(n_w=N_W, n_b=N_b, device=device)
    rec_pool.P_W = rec_P_W
    rec_pool.P_b = rec_P_b

    print("\n[Reconstruct + Evaluate]")
    all_metrics = {}
    ext_map = {'image': '.jpg', 'audio': '.wav', 'video': '.mp4'}

    for t in tasks:
        with torch.no_grad():
            pred = nets[t['role']](t['coords'], rec_pool.P_W, rec_pool.P_b)
        out_path = os.path.join(
            out_dir, f"recovered_{t['role']}_{t['modality']}{ext_map[t['modality']]}")
        SAVERS[t['modality']](pred, t['meta'], out_path)
        m = EVALUATORS[t['modality']](t['target'], pred, t['meta'])
        all_metrics[t['role']] = m
        pretty_print(f"{t['role']} ({t['modality']}) vs "
                     f"{os.path.basename(t['path'])}", m)

    # ── wrong-key attack evaluation (all secrets) ─────────
    for t in tasks[1:]:
        wrong_net = PoolNetwork(
            key='WRONG_KEY_ATTACKER', arch=t['arch'],
            pool_w_size=N_W, pool_b_size=N_b,
            use_fourier=t['cfg']['use_fourier'],
            fourier_args=fourier_args_from_cfg(t['cfg']),
            is_siren=t['cfg']['is_siren'], device=device, leaky_slope=0.01,
        ).to(device)
        with torch.no_grad():
            wrong_pred = wrong_net(t['coords'], rec_pool.P_W, rec_pool.P_b)
        pretty_print(
            f"[Wrong-Key attack] {t['role']} ({t['modality']})",
            EVALUATORS[t['modality']](t['target'], wrong_pred, t['meta']))

    # ── save the trained model for further analysis ───────
    model_path = os.path.join(artifacts_dir, 'trained_model.pth')
    torch.save({
        'pool': {'P_W': pool.P_W.detach().cpu(), 'P_b': pool.P_b.detach().cpu()},
        'nets': {role: net.state_dict() for role, net in nets.items()},
        'tasks': [{
            'role': t['role'], 'key': t['key'], 'modality': t['modality'],
            'cfg': t['cfg'], 'arch': t['arch'], 'path': t['path']
        } for t in tasks],
        'pool_config': {'N_W': N_W, 'N_b': N_b},
        'metrics': all_metrics
    }, model_path)
    print(f"\n[Model] saved {model_path}")

    print("\n========== all done ==========")
    return all_metrics


if __name__ == "__main__":
    # Quick example — see demo.py for the full command-line interface.
    run(cover_path="./data/cover.jpg",
        secret_paths=["./data/secret.jpg"],
        epochs=20000)
