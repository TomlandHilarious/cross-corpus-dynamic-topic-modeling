#!/usr/bin/env python
"""
Unified TC evaluation with single-corpus reference for both D-* and E-* conditions.

For each (model, source) pair:
  - Extract β at each time slice (source-specific for D, global for E)
  - Use ONLY that source's documents at that time slice as the reference corpus
  - Compute TD (top-25), TC (top-10 PMI), TQ = TD * TC

This makes D-* and E-* directly comparable with A-*, B-* (all single-corpus reference).
"""
from pathlib import Path

import torch
import numpy as np
import argparse
import json
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data
from utils import get_topic_coherence


def _diversity_helper(beta_t, num_top):
    """TD: unique top-N words across topics / (K * num_top)"""
    K = beta_t.shape[0]
    top_words = set()
    for k in range(K):
        top_idx = np.argsort(beta_t[k])[-num_top:]
        top_words.update(top_idx.tolist())
    return len(top_words) / (K * num_top)


def evaluate_model(model, vocab, train, src_names, get_beta_fn,
                   num_tops_div=25, num_tops_coh=10):
    """
    Evaluate TD, TC, TQ for each source using ONLY that source's docs as reference.

    get_beta_fn(src_id) -> numpy array [K, T, V]
    """
    train_tokens = train['tokens']
    train_times = np.asarray(train['times'])

    # Normalize sources -> upper-case string -> src_id
    def _norm(x):
        if isinstance(x, bytes):
            x = x.decode()
        return str(x).strip().upper()
    train_sources_norm = [_norm(s) for s in train['sources']]
    name2id = {n: i for i, n in enumerate(src_names)}
    train_src_ids = np.array([name2id[s] for s in train_sources_norm])

    results = {}
    for src_id, src_name in enumerate(src_names):
        print(f"\n--- {src_name} (src_id={src_id}) ---")
        beta = get_beta_fn(src_id)  # [K, T, V]
        K, T, V = beta.shape

        td_per_t, tc_per_t = [], []
        for t in range(T):
            beta_t = beta[:, t, :]

            # TD
            td_t = _diversity_helper(beta_t, num_tops_div)
            td_per_t.append(td_t)

            # TC: filter by BOTH time AND source
            mask = (train_times == t) & (train_src_ids == src_id)
            doc_idx = np.where(mask)[0]
            if len(doc_idx) == 0:
                tc_per_t.append(float('nan'))
                continue
            docs_t = [train_tokens[i] for i in doc_idx]
            tc_t, _ = get_topic_coherence(beta_t, docs_t,
                                          vocab=vocab, top_n=num_tops_coh)
            tc_per_t.append(tc_t)

        td = float(np.nanmean(td_per_t))
        tc = float(np.nanmean(tc_per_t))
        tq = td * tc
        results[src_name] = {'TD': td, 'TC': tc, 'TQ': tq, 'n_times': T}
        print(f"  TD={td:.4f}  TC={tc:.4f}  TQ={tq:.4f}")
    return results


def load_model_cpu(ckpt_path):
    print(f"Loading: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        m = torch.load(f, map_location='cpu', weights_only=False)
    m = m.cpu().eval()
    for p in m.parameters():
        p.data = p.data.cpu()
    if hasattr(m, 'mu_q_alpha'):
        m.mu_q_alpha = m.mu_q_alpha.cpu()
    if hasattr(m, 'logsigma_q_alpha'):
        m.logsigma_q_alpha = m.logsigma_q_alpha.cpu()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['source_adaptation', 'full_finetune'], required=True)
    ap.add_argument('--data_dir', required=True,
                    help='Merged data dir (min_df_100) containing train tokens/times/src_ids')
    ap.add_argument('--checkpoint', help='SA checkpoint (mode=source_adaptation)')
    ap.add_argument('--coha_checkpoint', help='FF COHA checkpoint')
    ap.add_argument('--hbr_checkpoint', help='FF HBR checkpoint')
    ap.add_argument('--ilr_checkpoint', help='FF ILR checkpoint')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    print("Loading merged data ...")
    vocab, train, valid, test = data.get_data(args.data_dir, temporal=True)
    print(f"  vocab={len(vocab)}  train_docs={len(train['tokens'])}")

    src_names = ['COHA', 'HBR', 'ILR']

    if args.mode == 'source_adaptation':
        model = load_model_cpu(args.checkpoint)
        with torch.no_grad():
            alpha_global, _ = model.get_alpha()

        def get_beta_fn(src_id):
            with torch.no_grad():
                return model.get_beta_source(src_id, alpha_global).cpu().numpy()

        results = evaluate_model(model, vocab, train, src_names, get_beta_fn)
        results['_meta'] = {'mode': 'source_adaptation', 'checkpoint': args.checkpoint}

    else:  # full_finetune
        ckpts = {'COHA': args.coha_checkpoint,
                 'HBR':  args.hbr_checkpoint,
                 'ILR':  args.ilr_checkpoint}
        # Pre-compute betas for each model so we can use the per-source loop
        betas_cache = {}
        for sname, cp in ckpts.items():
            m = load_model_cpu(cp)
            with torch.no_grad():
                alpha_global, _ = m.get_alpha()
                beta = m.get_beta(alpha_global).cpu().numpy()  # [K, T, V]
            betas_cache[sname] = beta
            del m

        def get_beta_fn(src_id):
            return betas_cache[src_names[src_id]]

        results = evaluate_model(None, vocab, train, src_names, get_beta_fn)
        results['_meta'] = {'mode': 'full_finetune', 'checkpoints': ckpts}

    print("\n" + "=" * 70)
    print(f"FINAL: {args.mode} (single-corpus reference)")
    print("=" * 70)
    print(f"{'Source':<8} {'TD':>8} {'TC':>8} {'TQ':>8}")
    for s in src_names:
        r = results[s]
        print(f"{s:<8} {r['TD']:>8.4f} {r['TC']:>8.4f} {r['TQ']:>8.4f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == '__main__':
    main()
