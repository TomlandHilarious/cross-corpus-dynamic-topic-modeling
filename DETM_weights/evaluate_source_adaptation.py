#!/usr/bin/env python
"""
Evaluate Source Adaptation Model: Per-Source TC/TD/TQ

Loads the source adaptation checkpoint and computes topic quality metrics
(Topic Diversity, Topic Coherence, Topic Quality) separately for each source.
"""

import torch
import pickle
import numpy as np
import argparse
from pathlib import Path
import sys

# Add DETM_weights to path to import data and utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data
from utils import get_topic_coherence

def load_model_and_data(checkpoint_path, data_dir):
    """Load model checkpoint and data"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        # If it's a state dict, we need to reconstruct the model
        # For now assume it's the full model
        model = checkpoint
    
    model.eval()
    model = model.cpu()
    
    # Load data (vocab, train, valid, test)
    print(f"Loading data: {data_dir}")
    vocab, train, valid, test = data.get_data(data_dir, temporal=True)
    
    return model, vocab, train

def get_source_beta(model, src_id):
    """
    Extract source-specific beta distributions for a given source
    Uses model's built-in get_beta_source method
    Returns: [K, T, V] numpy array
    """
    model.eval()
    
    with torch.no_grad():
        # Use model's built-in method to get source-specific beta
        beta_source = model.get_beta_source(src_id)
        return beta_source.detach().cpu().numpy()

def compute_topic_quality_per_source(model, vocab, train_data, src_id, src_name, num_top_words=25):
    """
    Compute TD, TC, TQ for a specific source
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {src_name} (source_id={src_id})")
    print(f"{'='*80}")
    
    # Get source-specific beta
    beta = get_source_beta(model, src_id)  # [K, T, V]
    K, T, V = beta.shape
    
    print(f"Beta shape: {beta.shape}")
    print(f"Vocab size: {len(vocab)}")
    
    # Compute Topic Diversity (average over time)
    td_scores = []
    for t in range(T):
        beta_t = beta[:, t, :]  # [K, V]
        
        # Get top words for each topic
        top_words = set()
        for k in range(K):
            top_indices = np.argsort(beta_t[k])[-num_top_words:]
            top_words.update(top_indices.tolist())
        
        # Diversity = unique words / (K * num_top_words)
        diversity_t = len(top_words) / (K * num_top_words)
        td_scores.append(diversity_t)
    
    td = np.mean(td_scores)
    
    # Compute Topic Coherence using NPMI
    # We need documents from this source to compute co-occurrence
    tc_scores = []
    
    train_tokens = train_data['tokens']
    train_counts = train_data['counts']
    train_times = train_data['times']
    
    for t in range(T):
        # Get documents from time t
        time_mask = (train_times == t)
        if not np.any(time_mask):
            continue
        
        time_docs_tokens = [train_tokens[i] for i in range(len(train_tokens)) if time_mask[i]]
        time_docs_counts = [train_counts[i] for i in range(len(train_counts)) if time_mask[i]]
        
        if len(time_docs_tokens) == 0:
            continue
        
        # Compute coherence for top-10 words per topic at time t
        beta_t = beta[:, t, :]
        tc_t, _ = get_topic_coherence(
            beta_or_top_ids=beta_t,
            data=time_docs_tokens,
            vocab=vocab,
            top_n=10
        )
        tc_scores.append(tc_t)
    
    tc = np.mean(tc_scores) if tc_scores else 0.0
    tq = td * tc
    
    print(f"\nResults for {src_name}:")
    print(f"  Topic Diversity (TD): {td:.6f}")
    print(f"  Topic Coherence (TC): {tc:.6f}")
    print(f"  Topic Quality (TQ):   {tq:.6f}")
    
    return {
        'source': src_name,
        'source_id': src_id,
        'td': td,
        'tc': tc,
        'tq': tq
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate source adaptation model per-source')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to source adaptation checkpoint')
    parser.add_argument('--merged_data', type=str, required=True,
                        help='Path to merged data directory (min_df_100 folder)')
    parser.add_argument('--output', type=str, default='source_adaptation_metrics.txt',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Load model and data
    model, vocab, train = load_model_and_data(
        args.checkpoint,
        args.merged_data
    )
    
    # Source names (order matters: must match source_id in training)
    sources = [
        (0, 'COHA'),
        (1, 'HBR'),
        (2, 'ILR')
    ]
    
    # Evaluate each source
    results = []
    for src_id, src_name in sources:
        result = compute_topic_quality_per_source(
            model, vocab, train, src_id, src_name
        )
        results.append(result)
    
    # Write results
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    with open(args.output, 'w') as f:
        f.write("Source Adaptation Model - Per-Source Topic Quality\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            line = f"{result['source']:10s}  TD: {result['td']:.6f}  TC: {result['tc']:.6f}  TQ: {result['tq']:.6f}\n"
            print(line.strip())
            f.write(line)
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
