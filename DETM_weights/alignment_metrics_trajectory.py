#!/usr/bin/env python
"""
Trajectory-level alignment metrics.

A topic k from source s is treated as a joint distribution over (word, time):
    P_{s,k}(w, t) = (1/T) * beta^{(s)}_{k,t}(w),
i.e. uniform prior over time bins. Aligning two topics is then equivalent to
comparing two such joint distributions of dimension T*V.

Metrics
-------
For shared-backbone models (D source-adaptation):
  - same_index_traj_jsd : (1/K) sum_k JSD(P_{i,k}, P_{j,k})
  - nearest_wrong_traj_jsd: (1/K) sum_k min_{l != k} JSD(P_{i,k}, P_{j,l})
  - traj_margin         : nearest_wrong - same_index
  - traj_retrieval@1    : (1/K) sum_k 1[argmin_l JSD(P_{i,k}, P_{j,l}) == k]

For post-hoc alignment baseline (E full fine-tune, no shared index):
  - hungarian_matched_traj_jsd : after solving K x K linear assignment on the
    trajectory JSD cost matrix, mean JSD over matched pairs (lower = better)
  - timestamp_match_change_rate : do Hungarian per timestamp t on the
    per-time JSD matrix, count fraction of consecutive time pairs (t, t+1)
    where any matched index changed (instability diagnostic; lower = more stable)

Usage
-----
  # Shared-backbone (source adaptation):
  python alignment_metrics_trajectory.py --mode source_adaptation \
      --checkpoint /path/to/ckpt.pt \
      --output traj_alignment_sa.json

  # Post-hoc baseline (full fine-tune):
  python alignment_metrics_trajectory.py --mode full_finetune \
      --coha_checkpoint /path/to/coha.pt \
      --hbr_checkpoint /path/to/hbr.pt \
      --ilr_checkpoint /path/to/ilr.pt \
      --output traj_alignment_ff.json

  # Both:
  python alignment_metrics_trajectory.py --mode compare \
      --checkpoint ...  --coha_checkpoint ... --hbr_checkpoint ... --ilr_checkpoint ... \
      --output traj_alignment_cmp.json
"""
from pathlib import Path

import argparse
import json
import sys
from itertools import combinations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_metrics import (load_model,
                                get_beta_from_model,
                                get_source_beta_from_model)


# ---------------------------------------------------------------------------
# Core: trajectory JSD matrix and helpers
# ---------------------------------------------------------------------------
def beta_to_trajectory(beta_KTV):
    """
    Convert beta of shape (K, T, V) to trajectory distributions of shape (K, T*V).
    Each row is a valid prob dist over (w, t): p(w,t) = (1/T) * beta[k, t, w].
    """
    K, T, V = beta_KTV.shape
    P = beta_KTV / T                        # uniform t-prior
    P = P.reshape(K, T * V)
    # Numerical safety: renormalize each row (they should already sum to 1)
    P = P / P.sum(axis=1, keepdims=True).clip(min=1e-12)
    return P


def compute_trajectory_jsd_matrix(beta_i, beta_j):
    """
    K x K matrix of JSDs between trajectory distributions.
    Entry (k, l) = JSD(P_{i,k}, P_{j,l}) computed on T*V-dim vectors.
    """
    P_i = beta_to_trajectory(beta_i)          # (K, TV)
    P_j = beta_to_trajectory(beta_j)          # (K, TV)
    K = P_i.shape[0]
    M = np.zeros((K, K), dtype=np.float64)
    for k in range(K):
        for l in range(K):
            M[k, l] = jensenshannon(P_i[k], P_j[l], base=2)
    return M


# ---------------------------------------------------------------------------
# Shared-backbone (same-index) metrics on trajectory representation
# ---------------------------------------------------------------------------
def shared_backbone_trajectory_metrics(beta_i, beta_j):
    """
    Returns dict with: same_index_traj_jsd, nearest_wrong_traj_jsd,
    traj_margin, traj_retrieval_at_1 (bidirectional averages).
    """
    M = compute_trajectory_jsd_matrix(beta_i, beta_j)         # (K, K)
    K = M.shape[0]
    diag = np.diag(M)
    same_index = float(diag.mean())

    off = M.copy()
    np.fill_diagonal(off, np.inf)

    # i -> j direction
    nearest_wrong_ij = float(off.min(axis=1).mean())
    retrieval_ij = float((M.argmin(axis=1) == np.arange(K)).mean())

    # j -> i direction
    nearest_wrong_ji = float(off.min(axis=0).mean())
    retrieval_ji = float((M.argmin(axis=0) == np.arange(K)).mean())

    nearest_wrong = 0.5 * (nearest_wrong_ij + nearest_wrong_ji)
    retrieval = 0.5 * (retrieval_ij + retrieval_ji)
    margin = nearest_wrong - same_index

    return {
        'same_index_traj_jsd': same_index,
        'nearest_wrong_traj_jsd': nearest_wrong,
        'nearest_wrong_traj_jsd_ij': nearest_wrong_ij,
        'nearest_wrong_traj_jsd_ji': nearest_wrong_ji,
        'traj_margin': margin,
        'traj_retrieval_at_1': retrieval,
        'traj_retrieval_at_1_ij': retrieval_ij,
        'traj_retrieval_at_1_ji': retrieval_ji,
    }


# ---------------------------------------------------------------------------
# Post-hoc Hungarian matching on trajectory JSD
# ---------------------------------------------------------------------------
def hungarian_trajectory_metrics(beta_i, beta_j):
    """
    Solve K x K linear assignment on the trajectory JSD matrix.
    Return matched mean JSD + the permutation.
    """
    M = compute_trajectory_jsd_matrix(beta_i, beta_j)
    row_ind, col_ind = linear_sum_assignment(M)
    matched_jsd = float(M[row_ind, col_ind].mean())
    return {
        'hungarian_matched_traj_jsd': matched_jsd,
        'matching_i_to_j': col_ind.tolist(),
    }


# ---------------------------------------------------------------------------
# Per-timestamp Hungarian + match-change-rate (instability)
# ---------------------------------------------------------------------------
def timestamp_match_change_rate(beta_i, beta_j):
    """
    For each t, compute Hungarian matching on the K x K per-time JSD matrix.
    Then count, across consecutive (t, t+1), the average number of topics
    whose match changed, normalized by K.

    Lower = more temporally stable Hungarian assignment.
    """
    K, T, V = beta_i.shape
    matchings = np.zeros((T, K), dtype=np.int64)
    for t in range(T):
        M_t = np.zeros((K, K))
        for k in range(K):
            for l in range(K):
                M_t[k, l] = jensenshannon(beta_i[k, t], beta_j[l, t], base=2)
        row_ind, col_ind = linear_sum_assignment(M_t)
        matchings[t] = col_ind         # for row k, matched j-index = col_ind[k]

    # Fraction of indices whose match changes between consecutive t
    changes = (matchings[1:] != matchings[:-1]).mean(axis=1)  # (T-1,)
    return {
        'timestamp_match_change_rate': float(changes.mean()),
        'per_step_change_rate': changes.tolist(),
    }


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def evaluate_shared_backbone(checkpoint_path):
    print("="*80)
    print("SHARED-BACKBONE TRAJECTORY ALIGNMENT (source adaptation)")
    print("="*80)
    print(f"Loading {checkpoint_path}")
    model = load_model(checkpoint_path)
    names = ['COHA', 'HBR', 'ILR']
    betas = {n: get_source_beta_from_model(model, sid) for sid, n in enumerate(names)}
    K, T, V = betas['COHA'].shape

    out = {
        'mode': 'source_adaptation_trajectory',
        'checkpoint': checkpoint_path,
        'num_topics': K, 'num_times': T, 'vocab_size': V,
        'pairwise_metrics': []
    }
    for ci, cj in combinations(names, 2):
        print(f"\n[{ci} <-> {cj}]")
        m = shared_backbone_trajectory_metrics(betas[ci], betas[cj])
        h = hungarian_trajectory_metrics(betas[ci], betas[cj])
        m.update(h)
        m['corpus_pair'] = f'{ci}<->{cj}'
        out['pairwise_metrics'].append(m)
        print(f"  same-index traj JSD        : {m['same_index_traj_jsd']:.4f}  (lower=better)")
        print(f"  nearest-wrong traj JSD     : {m['nearest_wrong_traj_jsd']:.4f}  (higher=better)")
        print(f"  traj margin                : {m['traj_margin']:.4f}  (higher=better)")
        print(f"  traj Retrieval@1           : {m['traj_retrieval_at_1']:.4f} ({m['traj_retrieval_at_1']*100:.1f}%)")
        print(f"  Hungarian-matched traj JSD : {m['hungarian_matched_traj_jsd']:.4f}  (lower=better)")

    # averages over pairs
    keys = ['same_index_traj_jsd', 'nearest_wrong_traj_jsd', 'traj_margin',
            'traj_retrieval_at_1', 'hungarian_matched_traj_jsd']
    out['overall_averages'] = {k: float(np.mean([p[k] for p in out['pairwise_metrics']])) for k in keys}
    print("\n--- AVG over corpus pairs ---")
    for k in keys:
        print(f"  {k:25s}: {out['overall_averages'][k]:.4f}")
    return out


def evaluate_full_finetune(coha_ckpt, hbr_ckpt, ilr_ckpt, with_change_rate=True):
    print("="*80)
    print("POST-HOC HUNGARIAN ALIGNMENT (full fine-tune)")
    print("="*80)
    names = ['COHA', 'HBR', 'ILR']
    ckpts = [coha_ckpt, hbr_ckpt, ilr_ckpt]
    betas = {}
    for n, c in zip(names, ckpts):
        print(f"\nLoading {n} from {c}")
        m = load_model(c)
        betas[n] = get_beta_from_model(m)
    K, T, V = betas['COHA'].shape

    out = {
        'mode': 'full_finetune_trajectory',
        'checkpoints': dict(zip(names, ckpts)),
        'num_topics': K, 'num_times': T, 'vocab_size': V,
        'pairwise_metrics': []
    }
    for ci, cj in combinations(names, 2):
        print(f"\n[{ci} <-> {cj}]")
        # Same-index metrics: E ckpts all started from the same C-ALL backbone,
        # so topic index k is shared at initialization. This measures how much
        # alignment survives the per-source full fine-tuning.
        si = shared_backbone_trajectory_metrics(betas[ci], betas[cj])
        # Hungarian: best-case post-hoc matching (free permutation)
        h = hungarian_trajectory_metrics(betas[ci], betas[cj])
        rec = {'corpus_pair': f'{ci}<->{cj}', **si, **h}
        print(f"  same-index traj JSD        : {si['same_index_traj_jsd']:.4f}  (lower=better)")
        print(f"  nearest-wrong traj JSD     : {si['nearest_wrong_traj_jsd']:.4f}  (higher=better)")
        print(f"  traj margin                : {si['traj_margin']:.4f}  (higher=better)")
        print(f"  traj Retrieval@1           : {si['traj_retrieval_at_1']:.4f} ({si['traj_retrieval_at_1']*100:.1f}%)")
        print(f"  Hungarian-matched traj JSD : {h['hungarian_matched_traj_jsd']:.4f}  (lower=better)")
        if with_change_rate:
            cr = timestamp_match_change_rate(betas[ci], betas[cj])
            rec.update(cr)
            print(f"  timestamp match-change rate: {cr['timestamp_match_change_rate']:.3f}  (lower=stabler)")
        out['pairwise_metrics'].append(rec)

    keys = ['same_index_traj_jsd', 'nearest_wrong_traj_jsd', 'traj_margin',
            'traj_retrieval_at_1', 'hungarian_matched_traj_jsd']
    if with_change_rate:
        keys.append('timestamp_match_change_rate')
    out['overall_averages'] = {k: float(np.mean([p[k] for p in out['pairwise_metrics']])) for k in keys}
    print("\n--- AVG over corpus pairs ---")
    for k in keys:
        print(f"  {k:32s}: {out['overall_averages'][k]:.4f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['source_adaptation', 'full_finetune', 'compare'])
    p.add_argument('--checkpoint', help='source adaptation ckpt')
    p.add_argument('--coha_checkpoint')
    p.add_argument('--hbr_checkpoint')
    p.add_argument('--ilr_checkpoint')
    p.add_argument('--output', required=True)
    p.add_argument('--no_change_rate', action='store_true',
                   help='skip per-timestamp match change rate computation')
    args = p.parse_args()

    results = {}
    if args.mode in ('source_adaptation', 'compare'):
        if not args.checkpoint:
            raise SystemExit('--checkpoint required for source_adaptation mode')
        results['shared_backbone'] = evaluate_shared_backbone(args.checkpoint)

    if args.mode in ('full_finetune', 'compare'):
        if not (args.coha_checkpoint and args.hbr_checkpoint and args.ilr_checkpoint):
            raise SystemExit('--coha/hbr/ilr_checkpoint required for full_finetune mode')
        results['full_finetune'] = evaluate_full_finetune(
            args.coha_checkpoint, args.hbr_checkpoint, args.ilr_checkpoint,
            with_change_rate=not args.no_change_rate)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == '__main__':
    main()
