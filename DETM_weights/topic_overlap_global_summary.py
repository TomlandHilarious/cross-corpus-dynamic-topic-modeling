#!/usr/bin/env python
"""
Global Top-30 overlap summary across all topics and all five-year bins.

For each source s in {COHA, HBR, ILR}, topic k, and five-year bin t:

    Overlap^{(s)}_{k,t}          = |Top30(beta^{(s)}_{k,t}) ∩ Top30(beta^{(0)}_{k,t})| / 30
    SourceLocalCount^{(s)}_{k,t} = 30 - |Top30(beta^{(s)}_{k,t}) ∩ Top30(beta^{(0)}_{k,t})|

We then average across all (k, t) per source, reporting mean and SD.

This is a descriptive diagnostic intended to summarize whether one corpus
(e.g. COHA) stays closer to the shared backbone than others across the full
topic set. It is NOT a new evaluation metric.

Outputs (in --out_dir):
  - topic_overlap_global_summary.csv         : aggregated per-source table
  - topic_overlap_global_summary.md          : markdown table for the paper
  - topic_overlap_per_topic_bin.csv          : full audit, one row per (s, k, t)
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# ---- paths / config -------------------------------------------------------
DETM_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, DETM_DIR)
import detm  # noqa: F401

DEFAULT_CKPT = (
    f'{Path(__file__).resolve().parent.parent}/detm_source_adapted_5year/'
    'adapt_kl0.3_anchor1e-3_20260325_015039/'
    'detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_'
    'Lr_1e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_lora_r8.pt'
)
DEFAULT_VOCAB = (
    f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/'
    'merged_v2_min100_5year_v2/min_df_100/vocab.pkl'
)
DEFAULT_OUT = f'{Path(__file__).resolve().parent}/paper_figures/case_study_sbra'

SOURCE_NAMES = ['COHA', 'HBR', 'ILR']
K_TOP = 30


def load_model_and_vocab(ckpt_path, vocab_path, device):
    print(f"[load] ckpt:  {ckpt_path}")
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    print(f"[load] vocab: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"[load] K={model.num_topics} T={model.num_times} "
          f"V={model.vocab_size} S={model.num_sources}")
    return model, vocab


@torch.no_grad()
def extract_betas(model):
    alpha_global, _ = model.get_alpha()
    beta_shared = model.get_beta(alpha_global).cpu().numpy()  # (K, T, V)
    betas_src = np.stack(
        [model.get_beta_source(s, alpha_global).cpu().numpy()
         for s in range(model.num_sources)], axis=0,
    )  # (S, K, T, V)
    return beta_shared, betas_src


def top_k_set(row, K):
    cand = np.argpartition(-row, min(K, len(row) - 1))[:K]
    return set(int(i) for i in cand)


def compute_overlaps(beta_shared, betas_src, K=K_TOP):
    """Returns:
      audit (list of dicts): rows per (source, topic, t_idx).
      stats (dict): src -> {'overlap_mean','overlap_sd','local_mean','local_sd'}
    """
    S, K_top, T, V = betas_src.shape
    audit = []
    per_source = {s: {'overlap': [], 'local': []} for s in SOURCE_NAMES}

    for s_id, src in enumerate(SOURCE_NAMES):
        for k in range(K_top):
            for t in range(T):
                set_s = top_k_set(betas_src[s_id, k, t], K)
                set_0 = top_k_set(beta_shared[k, t], K)
                inter = len(set_s & set_0)
                overlap = inter / K
                local = K - inter
                audit.append({
                    'source': src, 'topic': k, 't_idx': t,
                    'overlap': overlap, 'source_local_count': local,
                })
                per_source[src]['overlap'].append(overlap)
                per_source[src]['local'].append(local)

    stats = {}
    for src, d in per_source.items():
        o = np.asarray(d['overlap'])
        c = np.asarray(d['local'])
        stats[src] = {
            'overlap_mean': float(o.mean()),
            'overlap_sd':   float(o.std(ddof=1)),
            'local_mean':   float(c.mean()),
            'local_sd':     float(c.std(ddof=1)),
            'n':            int(o.size),
        }
    return audit, stats


def write_audit_csv(audit, years, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Source', 'Topic', 'Year', 'Top30Overlap',
                    'SourceLocalCount', 'K'])
        for r in audit:
            yr = years[r['t_idx']]
            w.writerow([r['source'], r['topic'], yr,
                        f"{r['overlap']:.6f}", r['source_local_count'], K_TOP])
    print(f"[csv]  wrote {out_path}")


def write_summary_csv(stats, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Source', 'MeanTop30Overlap', 'SDTop30Overlap',
                    'MeanSourceLocalCount', 'SDSourceLocalCount',
                    'NObservations', 'K'])
        for src in SOURCE_NAMES:
            s = stats[src]
            w.writerow([src,
                        f"{s['overlap_mean']:.4f}",
                        f"{s['overlap_sd']:.4f}",
                        f"{s['local_mean']:.3f}",
                        f"{s['local_sd']:.3f}",
                        s['n'], K_TOP])
    print(f"[csv]  wrote {out_path}")


def write_summary_md(stats, out_path):
    lines = [
        "| Source | Mean Top-30 overlap | SD Top-30 overlap | "
        "Mean source-local count | SD source-local count |",
        "|--------|--------------------:|------------------:|"
        "-----------------------:|---------------------:|",
    ]
    for src in SOURCE_NAMES:
        s = stats[src]
        lines.append(
            f"| {src} | {s['overlap_mean']:.3f} | {s['overlap_sd']:.3f} | "
            f"{s['local_mean']:.2f} | {s['local_sd']:.2f} |"
        )
    out_path.write_text('\n'.join(lines) + '\n')
    print(f"[md]   wrote {out_path}")


def time_year_labels(T):
    return [1922 + 5 * i for i in range(T)]


def main():
    global K_TOP
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--vocab',      default=DEFAULT_VOCAB)
    ap.add_argument('--out_dir',    default=DEFAULT_OUT)
    ap.add_argument('--top_k',      type=int, default=K_TOP)
    args = ap.parse_args()
    K_TOP = args.top_k

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')

    model, _vocab = load_model_and_vocab(args.checkpoint, args.vocab, device)
    beta_shared, betas_src = extract_betas(model)
    K_topics, T, _V = beta_shared.shape
    years = time_year_labels(T)
    print(f"[info] K_topics={K_topics} T={T} K_TOP={K_TOP} "
          f"(n_obs per source = {K_topics * T})")

    audit, stats = compute_overlaps(beta_shared, betas_src, K=K_TOP)

    print("\nSource | Mean overlap | SD overlap | Mean local | SD local")
    for src in SOURCE_NAMES:
        s = stats[src]
        print(f"  {src:<5} | {s['overlap_mean']:.3f}        | "
              f"{s['overlap_sd']:.3f}      | {s['local_mean']:.2f}       | "
              f"{s['local_sd']:.2f}")

    write_audit_csv(audit, years,
                    out_dir / 'topic_overlap_per_topic_bin.csv')
    write_summary_csv(stats,
                      out_dir / 'topic_overlap_global_summary.csv')
    write_summary_md(stats,
                     out_dir / 'topic_overlap_global_summary.md')


if __name__ == '__main__':
    main()
