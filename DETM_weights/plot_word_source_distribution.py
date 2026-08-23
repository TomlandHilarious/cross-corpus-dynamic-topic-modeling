#!/usr/bin/env python
"""
Plot the evolution of selected words in **source‑specific** word distributions
after LoRA fine‑tuning. Each word gets its own figure, and years (1922-2019) 
are displayed on the x-axis instead of time indices.

Key fixes (v2)
--------------
* `alpha = model.mu_q_alpha` – was wrongly set to the model object.
* During `g_src_time` computation use **beta_ktv (LoRA β)** for each source,
  not the global `beta_tkv`.
* Removed unused `topk` arg, cleaned imports.
"""
import argparse
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

import data
from detm import DETM

SRC_NAMES = ["COHA", "HBR", "ILR"]
NAME2ID   = {s: i for i, s in enumerate(SRC_NAMES)}

def theta_bar_by_src_time(theta, counts, times, sources, T, device):
    counts = np.asarray(counts, dtype=object)
    out = {}
    for src in np.unique(sources):
        m = sources == src
        θ = theta[m].to(device)               # (D_s,K)
        t_idx = times[m]
        c_row = counts[m]
        K = θ.size(1)
        θ̄ = torch.zeros(T, K, device=device)
        for t in range(T):
            m_t = t_idx == t
            if not m_t.any():
                continue
            θ_t = θ[m_t]
            lens = torch.tensor([float(r.sum() if hasattr(r, "sum") else sum(r))
                                  for r in c_row[m_t]], device=device)
            θ̄[t] = (θ_t.T * lens).sum(1) / lens.sum()
        out[src] = θ̄
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--min_df", type=int, default=100)
    ap.add_argument("--words", nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default=".", help="Directory to save output images")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model: DETM = torch.load(args.ckpt, map_location=dev).to(dev).eval()

    # ----- data -----
    data_dir = Path(args.data_path) / f"min_df_{args.min_df}"
    vocab, train, _, _ = data.get_data(str(data_dir), temporal=True)
    vocab2id = {w: i for i, w in enumerate(vocab)}
    
    # Load actual year timestamps
    try:
        with open(data_dir / 'timestamps.pkl', 'rb') as f:
            timelist = pickle.load(f)
    except FileNotFoundError:
        # If timestamps.pkl is not in the min_df directory, try the parent data directory
        try:
            with open(Path(args.data_path) / 'timestamps.pkl', 'rb') as f:
                timelist = pickle.load(f)
        except FileNotFoundError:
            print("Warning: timestamps.pkl not found, using time indices instead of actual years")
            timelist = list(range(int(train["times"].max()) + 1))

    tokens, counts = train["tokens"], train["counts"]
    times   = train["times"]
    src_arr = np.array([s.upper().strip() for s in train["sources"]])
    T = int(times.max()) + 1
    V = len(vocab)

    # ----- infer θ -----
    with torch.no_grad():
        theta, _ = model.infer_all_theta(
            tokens, counts, times,
            data.get_rnn_input(tokens, counts, times, T, V, len(tokens)),
            src_ids=torch.tensor([NAME2ID[s] for s in src_arr], device=dev),
        )
        alpha = model.mu_q_alpha                     # << fix
        beta_src = {s: model.get_beta(alpha, src_id=NAME2ID[s]) for s in SRC_NAMES}

    θ̄_src_time = theta_bar_by_src_time(theta, counts, times, src_arr, T, dev)

    # ----- g_{src,t}(v) -----
    g_src = {}
    for src in SRC_NAMES:
        beta_tkv = beta_src[src].permute(1, 0, 2).contiguous()
        g_src[src] = (θ̄_src_time[src].unsqueeze(2) * beta_tkv).sum(1)  # (T,V)

    # ----- plot -----
    # Make sure output directory exists
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define styles for each source
    source_styles = {
        'COHA': {'linestyle': '-', 'marker': '^', 'color': '#457B9D', 'markersize': 6, 'linewidth': 2.5},
        'HBR':  {'linestyle': '-', 'marker': 'o', 'color': '#E63946', 'markersize': 6, 'linewidth': 2.5},
        'ILR':  {'linestyle': '--', 'marker': 's', 'color': '#1D3557', 'markersize': 6, 'linewidth': 2.5}
    }
    
    # Process each word in a separate figure
    for w in args.words:
        if w not in vocab2id:
            print(f"[WARN] '{w}' not in vocab – skipped")
            continue
            
        # Create a new figure for this word
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#F8F9FA')
        ax.set_facecolor('#F8F9FA')
        
        # Word ID
        wid = vocab2id[w]
        
        # Plot each source
        for src in SRC_NAMES:
            ax.plot(timelist, g_src[src][:, wid].cpu(), label=src, **source_styles[src])
        
        # Enhanced grid and spines
        ax.grid(True, linestyle='--', alpha=0.3, color='#9CA3AF')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#9CA3AF')
        
        # Add title and labels with better styling
        ax.set_title(f"Word Distribution: '{w}'", fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel('Probability', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_xlabel('Year', fontsize=13, fontweight='bold', labelpad=10)
        
        # Improved x-axis tick formatting - show years with proper spacing
        step = max(1, len(timelist) // 10)  # Show ~10 years on the axis
        ax.set_xticks(timelist[::step])
        ax.set_xticklabels(timelist[::step], rotation=45, ha='right')
        
        # Better legend
        legend = ax.legend(frameon=True, loc='best', fontsize=12, facecolor='white', edgecolor='#D1D5DB')
        
        # Add a subtle box around the plot
        ax.patch.set_edgecolor('#D1D5DB')
        ax.patch.set_linewidth(1)
        
        # Make sure everything fits well
        plt.tight_layout()
        
        # Save the figure for this word
        output_file = output_dir / f"word_{w}_distribution.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Close the figure to free memory
        plt.close(fig)
        
    print(f"Visualized {len(args.words)} words across {len(SRC_NAMES)} sources with actual years {timelist[0]}-{timelist[-1]}")


if __name__ == "__main__":
    main()
