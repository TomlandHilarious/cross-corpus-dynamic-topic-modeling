#!/usr/bin/env python
"""
Train the missing no-reg checkpoint (if needed) and compute trajectory
alignment metrics for all 3 component-isolation combos:

  anchor_only  lambda_anchor=1e-3, lambda_smooth=0
  smooth_only  lambda_anchor=0,    lambda_smooth=1e-3
  no_reg       lambda_anchor=0,    lambda_smooth=0

Usage
-----
  python run_isolation_traj.py              # train no-reg + eval all 3
  python run_isolation_traj.py --eval_only  # skip training, eval existing ckpts
  python run_isolation_traj.py --dry_run    # print commands only
"""
from pathlib import Path

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO     = str(Path(__file__).resolve().parent)
PY       = '/user/rl3403/.conda/envs/nlp_kogut/bin/python'
MAIN     = f'{REPO}/main.py'
TRAJ     = f'{REPO}/alignment_metrics_trajectory.py'
DRIFT    = f'{REPO}/drift_from_backbone.py'
OUT_ROOT = f'{Path(__file__).resolve().parent.parent}/detm_source_adapted_5year'

DATA_DIR = f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/merged_v2_min100_5year_v2'
EMB_PATH = f'{DATA_DIR}/min_df_100/merged_embedding.npy'

BACKBONE = (f'{Path(__file__).resolve().parent.parent}/detm_merged_5year/'
            'merged_topic20_min_df100_delta0.01_20260325_001151/'
            'detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_'
            'Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt')

CKPT_FNAME = ('detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_'
              'Lr_1e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_lora_r8.pt')

COMBOS = {
    'anchor_only': {'anchor': 1e-3, 'smooth': 0.0,  'epochs': 20},
    'smooth_only': {'anchor': 0.0,  'smooth': 1e-3, 'epochs': 20},
    'no_reg':      {'anchor': 0.0,  'smooth': 0.0,  'epochs': 20},
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one(name, combo, dry_run=False):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fmt = lambda v: '0' if v == 0 else f'{v:.0e}'
    tag = f"a{fmt(combo['anchor'])}_s{fmt(combo['smooth'])}_e{combo['epochs']}"
    save_dir = f'{OUT_ROOT}/ablate_{tag}_{ts}'
    os.makedirs(save_dir, exist_ok=True)
    log_file = f'{save_dir}/train.log'

    cmd = [
        PY, '-u', MAIN,
        '--stage', 'lora',
        '--dataset', 'merged',
        '--data_path', DATA_DIR,
        '--emb_path',  EMB_PATH,
        '--save_path', save_dir,
        '--load_from', BACKBONE,
        '--num_topics', '20',
        '--emb_size',   '300',
        '--rho_size',   '300',
        '--batch_size', '500',
        '--delta',      '0.01',
        '--lr',         '1e-05',
        '--epochs',     str(combo['epochs']),
        '--min_df',     '100',
        '--train_embeddings',           '1',
        '--source_adaptation_mode',     '1',
        '--lambda_anchor',              str(combo['anchor']),
        '--lambda_smooth',              str(combo['smooth']),
        '--freeze_rho_in_adaptation',   '1',
        '--freeze_alpha_in_adaptation', '1',
        '--kl_alpha_scale',             '1e-6',
        '--adapt_kl_theta_max',         '0.3',
        '--adapt_warmup_epochs',        '5',
        '--visualize_every',            '5',
        '--save_checkpoint_every',      '5',
    ]

    print(f"\n{'='*70}")
    print(f"[TRAIN] {name}  save_dir={save_dir}")
    print(f"{'='*70}")
    print(' '.join(cmd))

    if dry_run:
        return None

    t0 = time.time()
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONUNBUFFERED': '1'})
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        proc.wait()

    if proc.returncode != 0:
        print(f"  TRAIN FAILED (exit={proc.returncode}). See {log_file}")
        return None

    print(f"  trained in {(time.time()-t0)/60:.1f} min")
    ckpt = os.path.join(save_dir, CKPT_FNAME)
    if not os.path.exists(ckpt):
        print(f"  WARNING: expected checkpoint not found at {ckpt}")
        return None
    return ckpt


# ---------------------------------------------------------------------------
# Trajectory alignment eval
# ---------------------------------------------------------------------------
def eval_traj(name, ckpt, dry_run=False):
    out = f'{REPO}/results/metrics/traj_align_D_{name}.json'
    cmd = [PY, TRAJ,
           '--mode',       'source_adaptation',
           '--checkpoint', ckpt,
           '--output',     out]
    print(f"\n{'='*70}")
    print(f"[TRAJ EVAL] {name}")
    print(f"  ckpt   : {ckpt}")
    print(f"  output : {out}")
    print(f"{'='*70}")
    print(' '.join(cmd))

    if dry_run:
        return None

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()

    if proc.returncode != 0:
        print(f"  EVAL FAILED (exit={proc.returncode})")
        return None

    with open(out) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Drift from backbone
# ---------------------------------------------------------------------------
def eval_drift(name, ckpt, dry_run=False):
    out = f'{REPO}/results/metrics/drift_{name}.json'
    cmd = [PY, DRIFT,
           '--d_checkpoint', ckpt,
           '--skip_e',
           '--output',       out]
    print(f"\n[DRIFT] {name}  ->  {out}")
    print(' '.join(cmd))

    if dry_run:
        return None

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()

    if proc.returncode != 0:
        print(f"  DRIFT FAILED (exit={proc.returncode})")
        return None

    with open(out) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(results):
    keys = ['same_index_traj_jsd', 'nearest_wrong_traj_jsd',
            'traj_margin', 'traj_retrieval_at_1', 'hungarian_matched_traj_jsd']
    header = f"\n{'Combo':<14}" + ''.join(f"{k:>26}" for k in keys)
    print("\n" + "="*80)
    print("TRAJECTORY ALIGNMENT SUMMARY (avg over corpus pairs)")
    print("="*80)
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        if r is None:
            print(f"{name:<14}  (failed)")
            continue
        avgs = r.get('shared_backbone', r).get('overall_averages', {})
        row = f"{name:<14}" + ''.join(f"{avgs.get(k, float('nan')):>26.4f}" for k in keys)
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_only', action='store_true',
                    help='Skip training; use KNOWN checkpoint for no_reg if available')
    ap.add_argument('--dry_run',   action='store_true',
                    help='Print commands without running them')
    ap.add_argument('--skip_drift', action='store_true',
                    help='Skip drift-from-backbone computation')
    args = ap.parse_args()

    ckpts = {}

    # --- train all 3 ---
    if not args.eval_only:
        for name, combo in COMBOS.items():
            ckpts[name] = train_one(name, combo, dry_run=args.dry_run)
    else:
        print("[--eval_only] skipping training; populate ckpts manually if needed")
        ckpts = {name: None for name in COMBOS}

    # --- eval each combo ---
    traj_results = {}
    for name in COMBOS:
        ckpt = ckpts.get(name)
        if ckpt is None:
            print(f"\n[SKIP] {name}: no checkpoint available")
            traj_results[name] = None
            continue
        if not os.path.exists(ckpt):
            print(f"\n[SKIP] {name}: checkpoint not found at {ckpt}")
            traj_results[name] = None
            continue
        traj_results[name] = eval_traj(name, ckpt, dry_run=args.dry_run)
        if not args.skip_drift:
            eval_drift(name, ckpt, dry_run=args.dry_run)

    if not args.dry_run:
        print_summary(traj_results)


if __name__ == '__main__':
    main()
