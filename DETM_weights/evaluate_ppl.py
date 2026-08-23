#!/usr/bin/env python
"""
Document-completion PPL evaluator for A/B/C/D/E checkpoints.
Uses the same math as main.py get_completion_ppl.
"""
from pathlib import Path
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data


def _norm_src(x):
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0: return ''
        x = x[0]
    if isinstance(x, bytes): x = x.decode()
    return str(x).strip().upper()


def _filter_test(d, idx):
    out = {}
    for k in ('tokens_1','counts_1','tokens_2','counts_2'):
        out[k] = [d[k][i] for i in idx]
    out['times'] = np.asarray(d['times'])[idx]
    return out


def build_eta(model, rnn_inp, device):
    with torch.no_grad():
        inp = model.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = model.init_hidden()
        output, _ = model.q_eta(inp, hidden)
        output = output.squeeze()
        T, K = model.num_times, model.num_topics
        etas = torch.zeros(T, K, device=device)
        etas[0] = model.mu_q_eta(torch.cat([output[0], torch.zeros(K, device=device)], dim=0))
        for t in range(1, T):
            etas[t] = model.mu_q_eta(torch.cat([output[t], etas[t-1]], dim=0))
    return etas


def get_theta(model, eta_td, norm_bows):
    with torch.no_grad():
        q = model.q_theta(torch.cat([norm_bows, eta_td], dim=1))
        return F.softmax(model.mu_q_theta(q), dim=-1)


def compute_ppl(model, test_data, V, emb_size, num_times, device,
                batch_size=1000, src_for_D=None):
    N = len(test_data['tokens_1'])
    if N == 0: return float('nan'), 0
    rnn_inp = data.get_rnn_input(
        test_data['tokens_1'], test_data['counts_1'],
        test_data['times'], num_times, V, N).to(device)
    eta = build_eta(model, rnn_inp, device)
    alpha = model.mu_q_alpha.to(device)            # (K, T, L)
    if src_for_D is not None:
        alpha = alpha + model.delta_alpha[src_for_D].to(device)  # (K, T, L) source-adapted
    total, cnt = 0.0, 0
    for ind in torch.split(torch.arange(N), batch_size):
        il = ind.tolist()
        h1, tb = data.get_batch(test_data['tokens_1'], test_data['counts_1'],
                                il, V, emb_size, temporal=True, times=test_data['times'])
        sums1 = h1.sum(1).unsqueeze(1).clamp(min=1e-10)
        norm1 = h1 / sums1
        tl = tb.long()
        theta = get_theta(model, eta[tl], norm1)
        h2, _ = data.get_batch(test_data['tokens_2'], test_data['counts_2'],
                               il, V, emb_size, temporal=True, times=test_data['times'])
        sums2 = h2.sum(1).clamp(min=1e-10)
        alpha_td = alpha[:, tl, :]
        beta = model.get_beta(alpha_td).permute(1, 0, 2)  # always use get_beta; alpha already source-adapted if D
        ll = torch.log((theta.unsqueeze(2) * beta).sum(1) + 1e-10)
        nll = -(ll * h2).sum(-1)
        total += (nll / sums2).mean().item()
        cnt += 1
    cur = min(total / max(cnt, 1), 100)
    return round(math.exp(cur), 2), N


def load_model(p, device):
    with open(p, 'rb') as f:
        m = torch.load(f, map_location=device, weights_only=False)
    return m.to(device).eval()


def eval_ckpt(ckpt, data_dir, device, bsz=1000, per_source=True, is_D=False):
    print(f"\n[DATA] {data_dir}")
    vocab, _, _, test = data.get_data(data_dir, temporal=True)
    V = len(vocab)
    model = load_model(ckpt, device)
    T = model.num_times
    E = model.rho.weight.shape[1]
    rows = []
    if not is_D:
        ppl, n = compute_ppl(model, test, V, E, T, device, bsz)
        rows.append({'split': 'ALL', 'ppl': ppl, 'n_docs': n})
        print(f"  ALL: PPL={ppl} (n={n})")
    if per_source and 'sources' in test:
        srcs = np.array([_norm_src(s) for s in test['sources']])
        name2id = {'COHA': 0, 'HBR': 1, 'ILR': 2}
        for sname in sorted(set(srcs)):
            if sname not in name2id: continue
            idx = np.where(srcs == sname)[0]
            sub = _filter_test(test, idx)
            sid = name2id[sname] if is_D else None
            ppl, n = compute_ppl(model, sub, V, E, T, device, bsz, src_for_D=sid)
            rows.append({'split': sname, 'ppl': ppl, 'n_docs': int(n)})
            print(f"  {'D@' if is_D else ''}{sname}: PPL={ppl} (n={n})")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=1000)
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")
    cfg = json.load(open(args.config))
    results = []

    for cond in ('A', 'B'):
        for sname, sub in cfg.get(cond, {}).items():
            print(f"\n{'='*60}\n[{cond}-{sname}]\n{'='*60}")
            for r in eval_ckpt(sub['ckpt'], sub['data_dir'], device,
                               args.batch_size, per_source=False):
                results.append({'condition': cond, 'model_source': sname,
                                'eval_split': r['split'], 'ppl': r['ppl'],
                                'n_docs': r['n_docs'], 'ckpt': sub['ckpt']})

    if 'C' in cfg:
        print(f"\n{'='*60}\n[C-ALL]\n{'='*60}")
        for r in eval_ckpt(cfg['C']['ckpt'], cfg['C']['data_dir'], device,
                           args.batch_size, per_source=True):
            results.append({'condition': 'C-ALL', 'model_source': 'ALL',
                            'eval_split': r['split'], 'ppl': r['ppl'],
                            'n_docs': r['n_docs'], 'ckpt': cfg['C']['ckpt']})

    if 'D' in cfg:
        print(f"\n{'='*60}\n[D]\n{'='*60}")
        for r in eval_ckpt(cfg['D']['ckpt'], cfg['D']['data_dir'], device,
                           args.batch_size, per_source=True, is_D=True):
            results.append({'condition': 'D', 'model_source': r['split'],
                            'eval_split': r['split'], 'ppl': r['ppl'],
                            'n_docs': r['n_docs'], 'ckpt': cfg['D']['ckpt']})

    if 'E' in cfg:
        print(f"\n{'='*60}\n[E]\n{'='*60}")
        dd = cfg['E']['data_dir']
        for sname in ('COHA', 'HBR', 'ILR'):
            if sname not in cfg['E']: continue
            print(f"\n--- E-{sname} ---")
            for r in eval_ckpt(cfg['E'][sname], dd, device,
                               args.batch_size, per_source=True):
                results.append({'condition': 'E', 'model_source': sname,
                                'eval_split': r['split'], 'ppl': r['ppl'],
                                'n_docs': r['n_docs'], 'ckpt': cfg['E'][sname]})

    # Pretty table
    print(f"\n{'='*70}")
    print(f"{'Cond':<6} {'Model':<6} {'Eval':<6} {'PPL':>10} {'n_docs':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['condition']:<6} {r['model_source']:<6} "
              f"{r['eval_split']:<6} {r['ppl']:>10} {r['n_docs']:>8}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == '__main__':
    main()
