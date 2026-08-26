#!/usr/bin/env python
"""
D (source-adaptation) hyperparameter ablation.

For each (lambda_anchor, lambda_smooth, epochs) combination:
  1. Train source-adaptation from C-ALL backbone
  2. Evaluate the resulting D-* checkpoint on per-source single-corpus reference
     (both PMI and NPMI)
  3. Aggregate everything into one CSV / JSON

Trains sequentially on the same GPU (default).

Usage
-----
# Default reduced grid (anchor in {1e-3,3e-4,1e-4} x smooth=anchor x epochs={20,40})
python ablate_adaptation.py

# Custom combos via JSON list
python ablate_adaptation.py --combos_json combos.json
# combos.json = [{"anchor":1e-3,"smooth":1e-3,"epochs":20}, ...]

# Skip training (only re-evaluate checkpoints listed in --resume_dir)
python ablate_adaptation.py --eval_only --combos_json combos.json
"""
from pathlib import Path

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO  = str(Path(__file__).resolve().parent)
PY    = os.environ.get('PYTHON', 'python')
MAIN  = f'{REPO}/main.py'
EVAL  = f'{REPO}/evaluate_npmi_robustness.py'

BACKBONE_CKPT = (f'{Path(__file__).resolve().parent.parent}/detm_merged_5year/'
                 'merged_topic20_min_df100_delta0.01_20260325_001151/'
                 'detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_'
                 'Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt')

DATA_DIR  = f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/merged_v2_min100_5year_v2'
EMB_PATH  = f'{DATA_DIR}/min_df_100/merged_embedding.npy'
EVAL_DATA = f'{DATA_DIR}/min_df_100'

OUT_ROOT = f'{Path(__file__).resolve().parent.parent}/detm_source_adapted_5year'

# ---------------------------------------------------------------------------
# Paper-facing ablation (6 settings, all epochs=20):
#   - Full SB-RA at three strength levels (anchor==smooth)
#   - Component-isolation: anchor-only, smooth-only, no regularization
# ---------------------------------------------------------------------------
DEFAULT_COMBOS = [
    # Full SB-RA: both terms active, three strength levels
    {"anchor": 1e-3, "smooth": 1e-3, "epochs": 20},   # current setting
    {"anchor": 3e-4, "smooth": 3e-4, "epochs": 20},
    {"anchor": 1e-4, "smooth": 1e-4, "epochs": 20},
    # Component-isolation: drop one term at a time (anchor held at 1e-3 when kept)
    {"anchor": 1e-3, "smooth": 0,    "epochs": 20},   # anchor-only (no smoothness)
    {"anchor": 0,    "smooth": 1e-3, "epochs": 20},   # smooth-only (no anchor)
    {"anchor": 0,    "smooth": 0,    "epochs": 20},   # no regularization
]


def _fmt_lam(v):
    return "0" if v == 0 else f"{v:.0e}"


def combo_tag(c):
    return f"a{_fmt_lam(c['anchor'])}_s{_fmt_lam(c['smooth'])}_e{c['epochs']}"


def find_ckpt_in_dir(save_dir):
    for fn in os.listdir(save_dir):
        if fn.endswith('.pt') and 'lora_r' in fn:
            return os.path.join(save_dir, fn)
    return None


def train_one(combo, args, dry_run=False):
    """Launch one D-* training run; return save_dir + ckpt path."""
    tag = combo_tag(combo)
    save_dir = f"{args.out_root}/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    existing_ckpt = find_ckpt_in_dir(save_dir)
    if existing_ckpt is not None:
        print(f"\n[SKIP TRAIN] combo={tag} existing_ckpt={existing_ckpt}")
        return save_dir, existing_ckpt
    log_file = f"{save_dir}/train_seed{args.seed}.log"

    cmd = [
        PY, '-u', MAIN,
        '--stage', 'lora',
        '--dataset', 'merged',
        '--data_path', DATA_DIR,
        '--emb_path',  EMB_PATH,
        '--save_path', save_dir,
        '--load_from', args.backbone_ckpt,
        '--num_topics', '20',
        '--emb_size', '300',
        '--rho_size', '300',
        '--batch_size', '500',
        '--delta', '0.01',
        '--lr', '1e-05',
        '--epochs', str(combo['epochs']),
        '--min_df', '100',
        '--train_embeddings', '1',
        '--source_adaptation_mode', '1',
        '--lambda_anchor', f"{combo['anchor']}",
        '--lambda_smooth', f"{combo['smooth']}",
        '--freeze_rho_in_adaptation',   '1',
        '--freeze_alpha_in_adaptation', '1',
        '--kl_alpha_scale',     '1e-6',
        '--adapt_kl_theta_max', '0.3',
        '--adapt_warmup_epochs', '5',
        '--visualize_every', '5',
        '--save_checkpoint_every', '5',
        '--seed', str(args.seed),
    ]

    print(f"\n{'='*70}\n[TRAIN] combo={tag}  save_dir={save_dir}\n{'='*70}")
    print(' '.join(cmd))
    if dry_run:
        return save_dir, None

    t0 = time.time()
    # Stream stdout to BOTH terminal and log file (line-buffered)
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.cuda_device), 'PYTHONUNBUFFERED': '1'})
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        proc.wait()
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  TRAIN FAILED (exit={proc.returncode}). See {log_file}")
        return save_dir, None
    print(f"  trained in {dt/60:.1f} min")
    ck = find_ckpt_in_dir(save_dir)
    if ck is None:
        print(f"  WARNING: no .pt found in {save_dir}")
    return save_dir, ck


def eval_one(ckpt, output_json, cuda_device):
    """Run evaluate_npmi_robustness.py in mode D for one ckpt."""
    cmd = [PY, EVAL, '--mode', 'D',
           '--data_dir', EVAL_DATA,
           '--checkpoint', ckpt,
           '--output', output_json]
    print(f"\n[EVAL] {ckpt}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, text=True,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(cuda_device), 'PYTHONUNBUFFERED': '1'})
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        print(f"  EVAL FAILED (exit={proc.returncode})")
        return None
    with open(output_json) as f:
        return json.load(f)


def aggregate(rows, out_path):
    """Save CSV with one row per (combo, source)."""
    import csv
    keys = ['combo', 'anchor', 'smooth', 'epochs', 'seed', 'condition', 'source',
            'TD', 'TC', 'UMass', 'C_V', 'TQ', 'zero_pair_rate@10',
            'zero_pair_rate@15', 'avg_pair_count@10', 'n_ref_docs', 'ckpt']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f"\nAggregated -> {out_path}")

    # pretty print
    print(f"\n{'Combo':<22} {'Src':<5} {'TD':>7} {'TC':>9} {'UMass':>9} {'C_V':>8} {'TQ':>9}")
    print("-" * 80)
    for r in rows:
        print(f"{r['combo']:<22} {r['source']:<5} "
              f"{r['TD']:>7.4f} {r['TC']:>9.4f} {r['UMass']:>9.4f} "
              f"{r['C_V']:>8.4f} {r['TQ']:>9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combos_json', help='JSON list of {"anchor","smooth","epochs"}')
    ap.add_argument('--eval_only',  action='store_true',
                    help='Skip training; load ckpts from --resume_dir or combos_json')
    ap.add_argument('--resume_dir', help='Directory containing previously-trained ablation dirs')
    ap.add_argument('--out_csv', default=f'{REPO}/results/metrics/ablation_results.csv')
    ap.add_argument('--out_json', default=f'{REPO}/results/metrics/ablation_results.json')
    ap.add_argument('--out_root', default=OUT_ROOT)
    ap.add_argument('--backbone_ckpt', default=BACKBONE_CKPT)
    ap.add_argument('--seed', type=int, default=2019)
    ap.add_argument('--cuda_device', default='0')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    combos = json.load(open(args.combos_json)) if args.combos_json else DEFAULT_COMBOS
    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    rows = []
    for c in combos:
        tag = combo_tag(c)
        if args.eval_only:
            ckpt = c.get('ckpt')
            if ckpt is None and args.resume_dir:
                # try to find by tag
                for d in os.listdir(args.resume_dir):
                    if tag in d:
                        ckpt = find_ckpt_in_dir(os.path.join(args.resume_dir, d))
                        break
            if ckpt is None:
                print(f"[SKIP] no ckpt for {tag}")
                continue
        else:
            _, ckpt = train_one(c, args, dry_run=args.dry_run)
            if ckpt is None:
                continue

        if args.dry_run:
            continue

        eval_json = ckpt.replace('.pt', '_npmi_eval.json')
        results = eval_one(ckpt, eval_json, args.cuda_device)
        if not results:
            continue

        for r in results:
            rows.append({
                'combo':   tag,
                'anchor':  c['anchor'],
                'smooth':  c['smooth'],
                'epochs':  c['epochs'],
                'seed':    args.seed,
                'condition': r.get('condition', ''),
                'source':  r['source'],
                'TD':      r['TD'],
                'TC':      r['TC'],
                'UMass':   r['UMass'],
                'C_V':     r['C_V'],
                'TQ':      r['TQ'],
                'zero_pair_rate@10': r.get('zero_pair_rate@10', ''),
                'zero_pair_rate@15': r.get('zero_pair_rate@15', ''),
                'avg_pair_count@10': r.get('avg_pair_count@10', ''),
                'n_ref_docs': r.get('n_ref_docs', ''),
                'ckpt':    ckpt,
            })

    with open(args.out_json, 'w') as f:
        json.dump(rows, f, indent=2)
    aggregate(rows, args.out_csv)


if __name__ == '__main__':
    main()
