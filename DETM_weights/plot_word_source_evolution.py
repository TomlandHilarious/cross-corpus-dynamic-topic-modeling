#!/usr/bin/env python
"""
Plot the evolution of selected words in topics across different sources.

This script combines functionality from plot_word_evolution.py and 
plot_word_source_distribution.py to create visualizations that show how
word probabilities evolve across time for different sources (COHA, HBR, ILR).

The x-axis uses actual years (1922-2019) instead of time indices.
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.cm as cm

import data
from detm import DETM

# Source names and mapping
SRC_NAMES = ["COHA", "HBR", "ILR"]
NAME2ID = {s: i for i, s in enumerate(SRC_NAMES)}

def theta_bar_by_src_time(theta, counts, times, sources, T, device):
    """
    Calculate average theta by source and time period.
    
    Args:
        theta: Document-topic distributions
        counts: Word counts
        times: Time indices
        sources: Source labels
        T: Number of time periods
        device: Computation device
        
    Returns:
        Dictionary mapping sources to average theta values
    """
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
    ap.add_argument("--ckpt", default=f"{Path(__file__).resolve().parent.parent}/detm_weighted/topic_50_min_df_10_delta_{0.05}_time_20250520_162831/lora_rank_16/detm_merged_K_50_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.0_Bsz_500_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_lora_r16.pt",
                    help="Path to the model checkpoint")
    ap.add_argument("--data_path", default=f"{Path(__file__).resolve().parent.parent}/merged_max_df_0.6",
                    help="Path to data directory")
    ap.add_argument("--min_df", type=int, default=10,
                    help="Minimum document frequency for vocabulary")
    ap.add_argument("--topic", type=int, default=5,
                    help="Topic ID to visualize")
    ap.add_argument("--topic_name", default="",
                    help="Optional name for the topic (will be displayed in the title)")
    ap.add_argument("--words", nargs="+", default=["woman", "strike", "union", "war"],
                    help="List of words to visualize")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Computation device")
    ap.add_argument("--out", default=f"{Path(__file__).resolve().parent.parent}/label_results/word_source_evolution.png",
                    help="Output file path")
    args = ap.parse_args()
    
    # Set up device
    dev = torch.device(args.device)
    
    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = torch.load(args.ckpt, map_location=dev).to(dev).eval()
    
    # Load data
    # The actual vocab.pkl is in a min_df_X subdirectory
    data_dir = Path(args.data_path)
    min_df_dir = data_dir / f"min_df_{args.min_df}"
    vocab, train, valid, test = data.get_data(str(min_df_dir), temporal=True)
    vocab_size = len(vocab)
    vocab2id = {w: i for i, w in enumerate(vocab)}
    
    # Load timestamps (actual years)
    try:
        with open(min_df_dir / 'timestamps.pkl', 'rb') as f:
            timelist = pickle.load(f)
    except FileNotFoundError:
        # If timestamps.pkl is not in the min_df directory, try the main data directory
        with open(data_dir / 'timestamps.pkl', 'rb') as f:
            timelist = pickle.load(f)
    T = len(timelist)
    
    # Extract tokens, counts, times, and sources
    tokens, counts = train["tokens"], train["counts"]
    times = train["times"]
    
    # Create source array if it exists in the data, otherwise use default
    if "sources" in train:
        src_arr = np.array([s.upper().strip() for s in train["sources"]])
    else:
        # Default to COHA if no source information
        src_arr = np.array(["COHA"] * len(tokens))
    
    # Check if words are in vocabulary
    for word in args.words:
        if word not in vocab2id:
            print(f"Warning: '{word}' not in vocabulary, skipping...")
            args.words.remove(word)
    
    if not args.words:
        print("No valid words to visualize. Exiting.")
        return
    
    # Get source-specific beta matrices
    print("Calculating source-specific beta matrices...")
    with torch.no_grad():
        # Get alpha and source-specific betas
        alpha = model.mu_q_alpha
        
        # Extract beta for each source
        beta_src = {}
        for src in SRC_NAMES:
            beta_src[src] = model.get_beta(alpha, src_id=NAME2ID[src]).cpu().numpy()
    
    # Setup plot
    plt.figure(figsize=(12, 8))
    
    # Define line styles and markers for each source
    styles = {
        'COHA': {'linestyle': '-', 'marker': 'o', 'color': 'blue'},
        'HBR': {'linestyle': '--', 'marker': 's', 'color': 'red'},
        'ILR': {'linestyle': ':', 'marker': '^', 'color': 'green'}
    }
    
    # Determine plot layout based on number of words
    nrows = len(args.words)
    if nrows == 0:
        print("No valid words to visualize. Exiting.")
        return
        
    # If only one word, use a single figure
    if nrows == 1:
        fig, axes = plt.subplots(figsize=(12, 6), facecolor='#F8F9FA')
        axes = [axes]  # Make it iterable
    else:
        # Create subplot grid for multiple words
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 5*nrows), sharex=True, facecolor='#F8F9FA')
        
    # Format the topic name for display
    topic_display = f"Topic {args.topic}"
    if args.topic_name:
        topic_display = f"Topic {args.topic}: {args.topic_name}"
        
    # Process each word
    for i, word in enumerate(args.words):
        word_id = vocab2id[word]
        ax = axes[i]
        
        # Set subplot background color
        ax.set_facecolor('#F8F9FA')
        
        # Extract word probability for each source and time
        for src_name in SRC_NAMES:
            # Extract from source-specific beta
            word_probs = beta_src[src_name][args.topic, :, word_id]
                
            # Plot the line for this source with enhanced styling
            ax.plot(timelist, word_probs, 
                   label=src_name,
                   **styles[src_name])
        
        # Add enhanced grid and spines
        ax.grid(True, linestyle='--', alpha=0.3, color='#9CA3AF')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#9CA3AF')
            
        # Add subplot title and decorations with better styling
        ax.set_title(f'Evolution of "{word}" in {topic_display}', 
                    fontsize=14, fontweight='bold', pad=12)
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold', labelpad=10)
        
        # Better legend
        ax.legend(frameon=True, loc='best', fontsize=11, facecolor='white', edgecolor='#D1D5DB')
        
        # Add a subtle box around the subplot
        ax.patch.set_edgecolor('#D1D5DB')
        ax.patch.set_linewidth(1)
        
    # Set x-axis properties on the last subplot with better styling
    axes[-1].set_xlabel('Year', fontsize=13, fontweight='bold', labelpad=10)
    
    # Improved x-axis tick formatting
    plt.xticks(timelist[::10], timelist[::10], rotation=45, ha='right')
    
    # Make sure all subplots are properly spaced
    plt.tight_layout()
    
    # Save the plot with higher resolution
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved enhanced visualization to {args.out}")
    print(f"Visualized {len(args.words)} words across {len(SRC_NAMES)} sources for {topic_display}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
