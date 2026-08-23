#!/usr/bin/env python
"""
Drift-from-backbone trajectory metric.

Argument
--------
The C-ALL merged DETM (condition C) is the *shared initialization* for both
D (source adaptation) and E (per-corpus full fine-tune). We measure how far
each per-source topic distribution has drifted from the merged backbone:

    drift_k^{(s)} = JSD_traj( beta^{(s)}_k ,  beta^{(0)}_k )

where beta^{(s)} is the per-source topic-word trajectory and beta^{(0)} is
the merged backbone. Topics are compared at the same index k (which is
canonical: D shares alpha^(0); E was initialized from C-ALL).

A topic-word trajectory beta_k of shape (T, V) is treated as a joint
distribution over (w, t):  P_k(w, t) = (1/T) * beta[k, t, w].

Usage
-----
  python drift_from_backbone.py \
      --backbone /shared/.../detm_merged_5year/.../*.pt \
      --d_checkpoint /shared/.../detm_source_adapted_5year/.../*.pt \
      --e_coha /shared/.../detm_full_finetune_baseline/coha/*.pt \
      --e_hbr  /shared/.../detm_full_finetune_baseline/hbr/*.pt \
      --e_ilr  /shared/.../detm_full_finetune_baseline/ilr/*.pt \
      --output drift.json
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_metrics import (load_model,
                                get_beta_from_model,
                                get_source_beta_from_model)


# ----------------------------------------------------------------------
def beta_to_trajectory(beta_KTV):
    K, T, V = beta_KTV.shape
    P = beta_KTV / T
    P = P.reshape(K, T * V)
    P = P / P.sum(axis=1, keepdims=True).clip(min=1e-12)
    return P


def per_topic_traj_jsd(beta_a, beta_b):
    """Return length-K array of trajectory JSDs at the same index k."""
    Pa = beta_to_trajectory(beta_a)
    Pb = beta_to_trajectory(beta_b)
    K = Pa.shape[0]
    out = np.zeros(K, dtype=np.float64)
    for k in range(K):
        out[k] = jensenshannon(Pa[k], Pb[k], base=2)
    return out


def per_timestep_jsd(beta_a, beta_b):
    """(K, T) per-(k,t) JSD between two beta tensors."""
    K, T, V = beta_a.shape
    out = np.zeros((K, T))
    for k in range(K):
        for t in range(T):
            out[k, t] = jensenshannon(beta_a[k, t], beta_b[k, t], base=2)
    return out


# ----------------------------------------------------------------------
def summarize_drift(name, drift_per_k, per_kt=None):
    rec = {
        'mean_traj_jsd': float(drift_per_k.mean()),
        'median_traj_jsd': float(np.median(drift_per_k)),
        'max_traj_jsd': float(drift_per_k.max()),
        'min_traj_jsd': float(drift_per_k.min()),
        'std_traj_jsd': float(drift_per_k.std()),
        'per_topic_traj_jsd': drift_per_k.tolist(),
    }
    if per_kt is not None:
        rec['mean_per_timestep_jsd'] = float(per_kt.mean())
        rec['per_timestep_mean_over_topics'] = per_kt.mean(axis=0).tolist()
    print(f"  [{name:14s}]  mean={rec['mean_traj_jsd']:.4f}  "
          f"median={rec['median_traj_jsd']:.4f}  "
          f"max={rec['max_traj_jsd']:.4f}  "
          f"min={rec['min_traj_jsd']:.4f}")
    return rec


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Default checkpoint paths for this project. Override via CLI flags.
# ----------------------------------------------------------------------
ROOT = str(Path(__file__).resolve().parent.parent)

# Two merged backbones exist with different vocabs; D ckpts pair with one of them.
BACKBONE_NEW = (  # V=19433
    f'{ROOT}/detm_merged_5year/merged_topic20_min_df100_delta0.01_20260325_001151/'
    'detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_'
    'Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt'
)
BACKBONE_OLD = (  # V=19461
    f'{ROOT}/detm_merged_5year/merged_topic20_min_df100_delta0.01_20260310_173832/'
    'detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_'
    'Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt'
)
DEFAULT_BACKBONE = BACKBONE_NEW

# D = source adaptation (frozen alpha^(0) + per-source delta_alpha)
_D_FNAME = (
    'detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_'
    'Lr_1e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_lora_r8.pt'
)
# Each D ckpt is paired with the backbone it was initialized from.
D_CHECKPOINTS = {
    # These are the EXACT ckpts used in the previous trajectory-alignment
    # report (all V=19433, paired with NEW backbone, same as E).
    '1e-3': (f'{ROOT}/detm_source_adapted_5year/adapt_kl0.3_anchor1e-3_20260325_015039/{_D_FNAME}', BACKBONE_NEW),
    '3e-4': (f'{ROOT}/detm_source_adapted_5year/ablate_a3e-04_s3e-04_e20_20260501_140344/{_D_FNAME}', BACKBONE_NEW),
    '1e-4': (f'{ROOT}/detm_source_adapted_5year/ablate_a1e-04_s1e-04_e20_20260501_142335/{_D_FNAME}', BACKBONE_NEW),
    # Optional: e40 ablation variants (longer training).
    '3e-4_e40': (f'{ROOT}/detm_source_adapted_5year/ablate_a3e-04_s3e-04_e40_20260501_152313/{_D_FNAME}', BACKBONE_NEW),
    '1e-4_e40': (f'{ROOT}/detm_source_adapted_5year/ablate_a1e-04_s1e-04_e40_20260501_160325/{_D_FNAME}', BACKBONE_NEW),
    # Legacy: March 17 D-main (V=19461, OLD backbone). Not used in main report.
    '1e-3_legacy': (f'{ROOT}/detm_source_adapted_5year/topic20_anchor1e-3_smooth1e-3_20260317_170735/{_D_FNAME}', BACKBONE_OLD),
}

# E = full fine-tune (per-corpus, started from C-ALL backbone)
DEFAULT_E_COHA = f'{ROOT}/detm_full_finetune_baseline/coha_fullft_topic20_20260412_122854/{_D_FNAME}'
DEFAULT_E_HBR  = f'{ROOT}/detm_full_finetune_baseline/hbr_fullft_topic20_20260412_122907/{_D_FNAME}'
DEFAULT_E_ILR  = f'{ROOT}/detm_full_finetune_baseline/ilr_fullft_topic20_20260412_122912/{_D_FNAME}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default=DEFAULT_BACKBONE,
                    help='C-ALL merged DETM checkpoint (.pt)')
    ap.add_argument('--d_lambda', default='1e-3', choices=list(D_CHECKPOINTS) + ['none'],
                    help='Which D ablation checkpoint to use (or "none" to skip D)')
    ap.add_argument('--d_checkpoint', default=None,
                    help='Override D checkpoint path (takes priority over --d_lambda)')
    ap.add_argument('--e_coha', default=DEFAULT_E_COHA)
    ap.add_argument('--e_hbr',  default=DEFAULT_E_HBR)
    ap.add_argument('--e_ilr',  default=DEFAULT_E_ILR)
    ap.add_argument('--skip_e', action='store_true', help='skip E entirely')
    ap.add_argument('--output', default=None,
                    help='output JSON path (auto-generated if not given)')
    ap.add_argument('--no_per_timestep', action='store_true',
                    help='skip per-(k,t) JSD computation (faster, smaller JSON)')
    args = ap.parse_args()

    # Resolve D checkpoint and its paired backbone.
    # If user did not override --backbone, use the one paired with this D.
    user_supplied_backbone = (args.backbone != DEFAULT_BACKBONE)
    if args.d_checkpoint is None and args.d_lambda != 'none':
        d_path, d_paired_backbone = D_CHECKPOINTS[args.d_lambda]
        args.d_checkpoint = d_path
        if not user_supplied_backbone:
            args.backbone = d_paired_backbone
    # Skip E if requested
    if args.skip_e:
        args.e_coha = args.e_hbr = args.e_ilr = None
    # Auto-generate output path
    if args.output is None:
        tag = f'_{args.d_lambda}' if args.d_lambda != 'none' else ''
        args.output = f'{ROOT}/DETM_weights/results/metrics/drift{tag}.json'

    print("=" * 80)
    print("DRIFT FROM C-ALL BACKBONE (trajectory JSD, same-index)")
    print("=" * 80)

    print(f"\nLoading backbone: {args.backbone}")
    bb_model = load_model(args.backbone)
    beta0 = get_beta_from_model(bb_model)
    K, T, V = beta0.shape
    print(f"  beta_0 shape = (K={K}, T={T}, V={V})")

    out = OrderedDict()
    out['backbone_ckpt'] = args.backbone
    out['shape'] = {'K': K, 'T': T, 'V': V}
    out['drift'] = OrderedDict()

    sources = [('COHA', 0), ('HBR', 1), ('ILR', 2)]

    # ---------------- D: source adaptation ----------------
    if args.d_checkpoint:
        print(f"\nLoading D (source adaptation): {args.d_checkpoint}")
        d_model = load_model(args.d_checkpoint)
        d_block = OrderedDict()
        print(f"\n[D vs backbone]")
        for sname, sid in sources:
            beta_s = get_source_beta_from_model(d_model, sid)
            assert beta_s.shape == beta0.shape, \
                f"shape mismatch D[{sname}] {beta_s.shape} vs backbone {beta0.shape}"
            drift = per_topic_traj_jsd(beta_s, beta0)
            per_kt = None if args.no_per_timestep else per_timestep_jsd(beta_s, beta0)
            d_block[sname] = summarize_drift(f"D-{sname}", drift, per_kt)
        # average across sources
        d_block['mean_over_sources'] = float(np.mean(
            [d_block[s]['mean_traj_jsd'] for s, _ in sources]))
        out['drift']['D'] = d_block
        print(f"  D mean over sources = {d_block['mean_over_sources']:.4f}")

    # ---------------- E: full fine-tune ----------------
    e_paths = {'COHA': args.e_coha, 'HBR': args.e_hbr, 'ILR': args.e_ilr}
    if any(e_paths.values()):
        print(f"\n[E (full fine-tune) vs backbone]")
        e_block = OrderedDict()
        for sname, _ in sources:
            p = e_paths[sname]
            if p is None:
                continue
            print(f"  Loading E-{sname}: {p}")
            m = load_model(p)
            beta_s = get_beta_from_model(m)
            assert beta_s.shape == beta0.shape, \
                f"shape mismatch E[{sname}] {beta_s.shape} vs backbone {beta0.shape}"
            drift = per_topic_traj_jsd(beta_s, beta0)
            per_kt = None if args.no_per_timestep else per_timestep_jsd(beta_s, beta0)
            e_block[sname] = summarize_drift(f"E-{sname}", drift, per_kt)
        if e_block:
            means = [v['mean_traj_jsd'] for v in e_block.values()
                     if isinstance(v, dict) and 'mean_traj_jsd' in v]
            e_block['mean_over_sources'] = float(np.mean(means))
            print(f"  E mean over sources = {e_block['mean_over_sources']:.4f}")
        out['drift']['E'] = e_block

    # ---------------- Side-by-side summary ----------------
    print("\n" + "=" * 80)
    print("SUMMARY (mean trajectory JSD vs backbone, same-index)")
    print("=" * 80)
    header = f"{'Source':<8} {'D':>10} {'E':>10} {'E - D':>10}"
    print(header)
    print('-' * len(header))
    def _g(side, s):
        if side not in out['drift']:
            return float('nan')
        return out['drift'][side].get(s, {}).get('mean_traj_jsd', float('nan'))
    def _fmt(v):
        return f"{v:>10.4f}" if not np.isnan(v) else f"{'-':>10}"
    def _fmt_diff(v):
        return f"{v:>+10.4f}" if not np.isnan(v) else f"{'-':>10}"
    for sname, _ in sources:
        d_v, e_v = _g('D', sname), _g('E', sname)
        diff = e_v - d_v if not (np.isnan(d_v) or np.isnan(e_v)) else float('nan')
        print(f"{sname:<8} {_fmt(d_v)} {_fmt(e_v)} {_fmt_diff(diff)}")
    d_m = out['drift'].get('D', {}).get('mean_over_sources', float('nan'))
    e_m = out['drift'].get('E', {}).get('mean_over_sources', float('nan'))
    diff_m = e_m - d_m if not (np.isnan(d_m) or np.isnan(e_m)) else float('nan')
    print(f"{'AVG':<8} {_fmt(d_m)} {_fmt(e_m)} {_fmt_diff(diff_m)}")

    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == '__main__':
    main()
