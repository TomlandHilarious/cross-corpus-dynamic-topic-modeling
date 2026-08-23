#!/usr/bin/env python
"""Hungarian trajectory JSD for Ind-CS (3 corpus-specific models with DIFFERENT
vocabularies).

Approach: build the 3-way intersection vocabulary
    V_common = vocab_COHA ∩ vocab_HBR ∩ vocab_ILR
For each model, slice its beta down to V_common indices, then renormalize each
(k, t) row to a valid distribution. Run the existing trajectory JSD +
Hungarian matching code from alignment_metrics_trajectory.py on these
restricted distributions.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_metrics_trajectory import (
    load_model, get_beta_from_model,
    shared_backbone_trajectory_metrics, hungarian_trajectory_metrics,
    timestamp_match_change_rate,
)
from itertools import combinations


def load_vocab_pkl(path):
    with open(path, 'rb') as f:
        v = pickle.load(f)
    if isinstance(v, dict):
        if all(isinstance(k, str) for k in list(v.keys())[:5]):
            return [w for w, _ in sorted(v.items(), key=lambda x: x[1])]
        return [v[i] for i in sorted(v.keys())]
    return list(v)


def restrict_beta(beta_KTV_local, local_vocab, common_words):
    """beta (K, T, V_local) -> (K, T, V_common) by selecting columns whose words
    are in `common_words`, in the order of `common_words`. Renormalize per
    (k, t) row."""
    K, T, V_local = beta_KTV_local.shape
    word_to_idx = {w: i for i, w in enumerate(local_vocab)}
    sel = np.array([word_to_idx[w] for w in common_words], dtype=np.int64)
    out = beta_KTV_local[:, :, sel]
    print(f"    restricted: V_local={V_local}  ->  V_common={len(common_words)}")
    sums = out.sum(axis=2, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    out = out / sums
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--coha_checkpoint', required=True)
    p.add_argument('--hbr_checkpoint',  required=True)
    p.add_argument('--ilr_checkpoint',  required=True)
    p.add_argument('--coha_vocab', required=True)
    p.add_argument('--hbr_vocab',  required=True)
    p.add_argument('--ilr_vocab',  required=True)
    p.add_argument('--no_change_rate', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()

    names = ['COHA', 'HBR', 'ILR']
    ckpts = [a.coha_checkpoint, a.hbr_checkpoint, a.ilr_checkpoint]
    vpaths = [a.coha_vocab, a.hbr_vocab, a.ilr_vocab]

    # Load all 3 vocabs
    vocabs = {n: load_vocab_pkl(vp) for n, vp in zip(names, vpaths)}
    for n in names:
        print(f"  vocab[{n}] = {len(vocabs[n])}")

    # 3-way intersection (sorted for determinism)
    common = sorted(set(vocabs['COHA']) & set(vocabs['HBR']) & set(vocabs['ILR']))
    V_common = len(common)
    print(f"\n3-way intersection vocab: V_common = {V_common}")

    # Load models, extract betas, restrict to V_common
    betas = {}
    for n, c in zip(names, ckpts):
        print(f"\nLoading {n}: {c}")
        m = load_model(c)
        beta_local = get_beta_from_model(m)
        K, T, Vl = beta_local.shape
        print(f"  local beta: K={K} T={T} V={Vl}")
        assert Vl == len(vocabs[n]), (
            f"vocab.pkl size {len(vocabs[n])} != beta V {Vl} for {n}")
        betas[n] = restrict_beta(beta_local, vocabs[n], common)
    K, T, V = betas['COHA'].shape
    print(f"\nRestricted betas: K={K} T={T} V={V}")

    out = {
        'mode': 'ind_cs_trajectory_intersection',
        'checkpoints': dict(zip(names, ckpts)),
        'vocabs': dict(zip(names, vpaths)),
        'vocab_sizes': {n: len(vocabs[n]) for n in names},
        'V_common': V_common,
        'num_topics': K, 'num_times': T,
        'pairwise_metrics': []
    }

    with_change = not a.no_change_rate
    for ci, cj in combinations(names, 2):
        print(f"\n[{ci} <-> {cj}]")
        si = shared_backbone_trajectory_metrics(betas[ci], betas[cj])
        h  = hungarian_trajectory_metrics(betas[ci], betas[cj])
        rec = {'corpus_pair': f'{ci}<->{cj}', **si, **h}
        print(f"  same-index traj JSD        : {si['same_index_traj_jsd']:.4f}")
        print(f"  Hungarian-matched traj JSD : {h['hungarian_matched_traj_jsd']:.4f}")
        if with_change:
            cr = timestamp_match_change_rate(betas[ci], betas[cj])
            rec.update(cr)
            print(f"  timestamp match-change rate: {cr['timestamp_match_change_rate']:.3f}")
        out['pairwise_metrics'].append(rec)

    keys = ['same_index_traj_jsd', 'nearest_wrong_traj_jsd', 'traj_margin',
            'traj_retrieval_at_1', 'hungarian_matched_traj_jsd']
    if with_change:
        keys.append('timestamp_match_change_rate')
    out['overall_averages'] = {
        k: float(np.mean([p[k] for p in out['pairwise_metrics']])) for k in keys
    }
    print("\n--- AVG over corpus pairs ---")
    for k in keys:
        print(f"  {k:32s}: {out['overall_averages'][k]:.4f}")

    with open(a.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {a.output}")


if __name__ == '__main__':
    main()
