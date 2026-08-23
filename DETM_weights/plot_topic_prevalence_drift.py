#!/usr/bin/env python
"""
Batch processing version - generates prevalence and drift plots for all specified topics.
Each plot type is saved in a separate subfolder under the output directory.


Plot topic prevalence (theta) and drift (alpha) visualizations for a specific topic.

This script generates two visualizations:
1. Prevalence plot - yearly average theta values for HBR (solid) and ILR (dashed)
2. Global drift plot for the selected topic

The script takes a topic index as input and saves two PNG files:
- topic_k_prevalence.png
- topic_k_drift.png
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

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

def load_topic_labels(file_path):
    """Load topic labels from a CSV file"""
    topic_labels = {}
    try:
        with open(file_path, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split(',', 2)
                if len(parts) >= 3:
                    try:
                        topic_id = int(parts[1])
                        # Remove quotes if present
                        label = parts[2].strip('"\r\n ')
                        topic_labels[topic_id] = label
                    except (ValueError, IndexError):
                        pass
    except FileNotFoundError:
        print(f"Warning: Topic labels file not found at {file_path}")
    return topic_labels

def main():
    ap = argparse.ArgumentParser(description='Plot topic prevalence and drift for multiple topics.')
    ap.add_argument("--ckpt", default=f"{Path(__file__).resolve().parent.parent}/detm_weighted/topic_50_min_df_10_delta_{0.05}_time_20250520_162831/lora_rank_16/detm_merged_K_50_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.0_Bsz_500_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_lora_r16.pt",
                    help="Path to the model checkpoint")
    ap.add_argument("--data_path", default=f"{Path(__file__).resolve().parent.parent}/merged_max_df_0.6",
                    help="Path to data directory")
    ap.add_argument("--min_df", type=int, default=10,
                    help="Minimum document frequency for vocabulary")
    ap.add_argument("--topics", type=str, default="all",
                    help="Comma-separated list of topic IDs to visualize, or 'all' for all topics")
    ap.add_argument("--labels_file", default=f"{Path(__file__).resolve().parent.parent}/label_results/topic_labels_global.csv",
                    help="Path to the CSV file containing topic labels")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Computation device")
    ap.add_argument("--output_dir", default=f"{Path(__file__).resolve().parent.parent}/label_results",
                    help="Directory to save output images")
    args = ap.parse_args()
    
    # Set up device
    dev = torch.device(args.device)
    
    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = torch.load(args.ckpt, map_location=dev).to(dev).eval()
    
    # Load data
    data_dir = Path(args.data_path)
    min_df_dir = data_dir / f"min_df_{args.min_df}"
    vocab, train, valid, test = data.get_data(str(min_df_dir), temporal=True)
    vocab_size = len(vocab)
    
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
    
    # Load topic labels if available
    topic_labels = load_topic_labels(args.labels_file)
    print(f"Loaded {len(topic_labels)} topic labels from {args.labels_file}")
    
    # Get alpha (drift) and infer theta (topic prevalence)
    print("Calculating alpha and inferring theta...")
    with torch.no_grad():
        alpha = model.mu_q_alpha  # K × T × L
        theta, _ = model.infer_all_theta(
            tokens, counts, times,
            data.get_rnn_input(tokens, counts, times, T, vocab_size, len(tokens)),
            src_ids=torch.tensor([NAME2ID[s] for s in src_arr], device=dev),
        )
        
    # Calculate theta_bar for each source and time
    theta_bar = theta_bar_by_src_time(theta, counts, times, src_arr, T, dev)
    
    # Determine which topics to process
    K = model.num_topics  # Total number of topics in model
    if args.topics.lower() == "all":
        topic_ids = list(range(K))
    else:
        topic_ids = [int(t.strip()) for t in args.topics.split(",")]
    
    # Create the output directories
    base_output_dir = Path(args.output_dir)
    prevalence_dir = base_output_dir / "prevalence"
    drift_dir = base_output_dir / "drift"
    
    prevalence_dir.mkdir(exist_ok=True, parents=True)
    drift_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing {len(topic_ids)} topics...")
    
    # Process each topic
    for topic_id in topic_ids:
        if topic_id >= K:
            print(f"Warning: Topic {topic_id} exceeds the model's topic count ({K}). Skipping.")
            continue
            
        # Get topic name if available
        topic_name = topic_labels.get(topic_id, "") 
        topic_display = f"Topic {topic_id}"
        if topic_name:
            topic_display = f"Topic {topic_id}: {topic_name}"
            
        print(f"\nProcessing {topic_display}...")
        
        # ------ Create prevalence plot -------
        print(f"  Creating prevalence plot for {topic_display}...")
        plt.figure(figsize=(12, 7), facecolor='#F8F9FA')
        ax = plt.gca()
        ax.set_facecolor('#F8F9FA')
        
        # Plot theta (prevalence) for both sources
        sources_to_plot = ["HBR", "ILR"]
        source_styles = {
            'HBR': {'linestyle': '-', 'marker': 'o', 'color': '#E63946', 'markersize': 6, 'linewidth': 2.5},
            'ILR': {'linestyle': '--', 'marker': 's', 'color': '#1D3557', 'markersize': 6, 'linewidth': 2.5}
        }
        
        for src in sources_to_plot:
            if src in theta_bar:
                plt.plot(timelist, theta_bar[src][:, topic_id].cpu(), label=src, **source_styles[src])
        
        # Enhance grid and spines
        ax.grid(True, linestyle='--', alpha=0.3, color='#9CA3AF')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#9CA3AF')
        
        # Set title and labels
        plt.title(f"Topic Prevalence - {topic_display}", fontsize=16, fontweight='bold', pad=15)
        plt.ylabel('Topic Prevalence', fontsize=13, fontweight='bold', labelpad=10)
        plt.xlabel('Year', fontsize=13, fontweight='bold', labelpad=10)
        
        # X-axis formatting
        step = max(1, len(timelist) // 10)  # Show ~10 years on the axis
        plt.xticks(timelist[::step], timelist[::step], rotation=45, ha='right')
        
        # Better legend
        legend = plt.legend(frameon=True, loc='best', fontsize=12, facecolor='white', edgecolor='#D1D5DB')
        
        # Save the plot to the prevalence subfolder
        prevalence_file = prevalence_dir / f"topic_{topic_id}_prevalence.png"
        plt.tight_layout()
        plt.savefig(prevalence_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved prevalence plot to {prevalence_file}")
        
        # ------ Create drift plot -------
        print(f"  Creating drift plot for {topic_display}...")
        plt.figure(figsize=(12, 7), facecolor='#F8F9FA')
        ax = plt.gca()
        ax.set_facecolor('#F8F9FA')
        
        # Compute L2 norm of alpha (drift) for the selected topic
        topic_alpha = alpha[topic_id, :, :].cpu().numpy()  # T x L
        alpha_norm = np.sqrt(np.sum(topic_alpha**2, axis=1))  # T
        
        # Plot alpha drift norm
        plt.plot(timelist, alpha_norm, linestyle='-', marker='o', color='#2A9D8F', 
                 markersize=6, linewidth=2.5, label='Drift Magnitude')
        
        # Enhance grid and spines
        ax.grid(True, linestyle='--', alpha=0.3, color='#9CA3AF')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#9CA3AF')
        
        # Set title and labels
        plt.title(f"Topic Drift - {topic_display}\nL2 Norm of Alpha", fontsize=16, fontweight='bold', pad=15)
        plt.ylabel('Drift Magnitude', fontsize=13, fontweight='bold', labelpad=10)
        plt.xlabel('Year', fontsize=13, fontweight='bold', labelpad=10)
        
        # X-axis formatting
        plt.xticks(timelist[::step], timelist[::step], rotation=45, ha='right')
        
        # Better legend
        legend = plt.legend(frameon=True, loc='best', fontsize=12, facecolor='white', edgecolor='#D1D5DB')
        
        # Save the plot to the drift subfolder
        drift_file = drift_dir / f"topic_{topic_id}_drift.png"
        plt.tight_layout()
        plt.savefig(drift_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved drift plot to {drift_file}")
    
    print(f"\nCompleted processing {len(topic_ids)} topics.")
    print(f"Prevalence plots saved to: {prevalence_dir}")
    print(f"Drift plots saved to: {drift_dir}")
    
    plt.close('all')

if __name__ == "__main__":
    main()
