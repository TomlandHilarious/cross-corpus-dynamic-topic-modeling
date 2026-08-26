#!/usr/bin/env python
"""
Cross-Corpus Topic Alignment Metrics

Computes alignment quality metrics between topic distributions across corpora:
1. Same-index JSD: Average divergence between same-index topics
2. Nearest-wrong JSD: Average divergence to nearest wrong-index topic
3. Alignment Margin: Difference between wrong and same (higher = better)
4. Retrieval@1: Proportion of times same-index topic is nearest neighbor

Works with both:
- Source adaptation models (single checkpoint with source-specific betas)
- Full fine-tune models (separate checkpoints per corpus)

Usage:
    # For source adaptation model:
    python alignment_metrics.py --mode source_adaptation \
        --checkpoint /path/to/source_adapted_model.pt \
        --output alignment_results.json

    # For full fine-tune models:
    python alignment_metrics.py --mode full_finetune \
        --coha_checkpoint /path/to/coha.pt \
        --hbr_checkpoint /path/to/hbr.pt \
        --ilr_checkpoint /path/to/ilr.pt \
        --output alignment_results.json
"""
from pathlib import Path

import torch
import numpy as np
import argparse
import json
import sys
from scipy.spatial.distance import jensenshannon
from itertools import combinations

# Add DETM_weights to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_model(checkpoint_path):
    """Load DETM model checkpoint"""
    device = torch.device('cpu')
    
    with open(checkpoint_path, 'rb') as f:
        model = torch.load(f, map_location=device)
    
    model = model.cpu()
    model.eval()
    
    # Move all parameters to CPU
    for param in model.parameters():
        param.data = param.data.cpu()
    
    # Move variational parameters
    if hasattr(model, 'mu_q_alpha'):
        model.mu_q_alpha = model.mu_q_alpha.cpu()
    if hasattr(model, 'logsigma_q_alpha'):
        model.logsigma_q_alpha = model.logsigma_q_alpha.cpu()
    
    return model


def get_beta_from_model(model):
    """
    Extract global beta from model (for full fine-tune models)
    Returns: numpy array [K, T, V]
    """
    with torch.no_grad():
        # Get alpha from variational parameters
        if hasattr(model, 'mu_q_alpha'):
            alpha = model.mu_q_alpha.detach().cpu()
        else:
            alpha, _ = model.get_alpha()
            alpha = alpha.detach().cpu()
        
        # Get rho (word embeddings)
        if hasattr(model.rho, 'weight'):
            rho = model.rho.weight.detach().cpu()
        else:
            rho = model.rho.detach().cpu()
        
        # Compute beta = softmax(alpha @ rho.T)
        K, T, L = alpha.shape
        V = rho.shape[0]
        beta = torch.zeros(K, T, V)
        
        for k in range(K):
            for t in range(T):
                logit = torch.matmul(alpha[k, t], rho.t())
                beta[k, t] = torch.softmax(logit, dim=0)
        
        return beta.numpy()


def get_source_beta_from_model(model, src_id):
    """
    Extract source-specific beta from source adaptation model
    Returns: numpy array [K, T, V]
    """
    with torch.no_grad():
        beta_source = model.get_beta_source(src_id)
        return beta_source.detach().cpu().numpy()


def compute_jsd_matrix(beta_i, beta_j, t):
    """
    Compute pairwise JSD matrix between topics at time t
    
    Args:
        beta_i: [K, T, V] topic-word distributions from corpus i
        beta_j: [K, T, V] topic-word distributions from corpus j
        t: time index
    
    Returns:
        jsd_matrix: [K, K] where entry (k, l) = JSD(beta_i[k,t], beta_j[l,t])
    """
    K = beta_i.shape[0]
    jsd_matrix = np.zeros((K, K))
    
    for k in range(K):
        for l in range(K):
            jsd_matrix[k, l] = jensenshannon(beta_i[k, t], beta_j[l, t])
    
    return jsd_matrix


def compute_same_index_jsd(beta_i, beta_j):
    """
    Compute Same-index JSD: average JSD between same-index topics
    
    JSD_same(i,j) = (1/KT) * sum_t sum_k JSD(beta_i[k,t], beta_j[k,t])
    
    Lower values = better alignment (same topics stay similar)
    """
    K, T, V = beta_i.shape
    total_jsd = 0.0
    
    for t in range(T):
        for k in range(K):
            jsd = jensenshannon(beta_i[k, t], beta_j[k, t])
            total_jsd += jsd
    
    return total_jsd / (K * T)


def compute_nearest_wrong_jsd(beta_i, beta_j):
    """
    Compute Nearest-wrong JSD: average min JSD to wrong-index topics
    
    JSD_wrong(i,j) = (1/KT) * sum_t sum_k min_{l != k} JSD(beta_i[k,t], beta_j[l,t])
    
    Higher values = better (wrong topics are far away)
    """
    K, T, V = beta_i.shape
    total_min_jsd = 0.0
    
    for t in range(T):
        jsd_matrix = compute_jsd_matrix(beta_i, beta_j, t)
        
        for k in range(K):
            # Get JSDs to all topics except same index
            wrong_jsds = [jsd_matrix[k, l] for l in range(K) if l != k]
            min_wrong_jsd = min(wrong_jsds)
            total_min_jsd += min_wrong_jsd
    
    return total_min_jsd / (K * T)


def compute_alignment_margin(jsd_same, jsd_wrong):
    """
    Compute Alignment Margin: JSD_wrong - JSD_same
    
    Higher values = better (clear separation between correct and wrong matches)
    """
    return jsd_wrong - jsd_same


def compute_retrieval_at_1(beta_i, beta_j):
    """
    Compute Retrieval@1: proportion of times same-index topic is nearest
    
    Retrieval@1(i,j) = (1/KT) * sum_t sum_k 1[k == argmin_l JSD(beta_i[k,t], beta_j[l,t])]
    
    Higher values = better (correct topic is retrieved most often)
    """
    K, T, V = beta_i.shape
    correct_retrievals = 0
    
    for t in range(T):
        jsd_matrix = compute_jsd_matrix(beta_i, beta_j, t)
        
        for k in range(K):
            # Find nearest topic in beta_j
            nearest_l = np.argmin(jsd_matrix[k])
            if nearest_l == k:
                correct_retrievals += 1
    
    return correct_retrievals / (K * T)


def compute_all_metrics(beta_i, beta_j, corpus_i, corpus_j):
    """
    Compute all alignment metrics between two corpora
    
    Returns dict with:
        - same_index_jsd: symmetric
        - nearest_wrong_jsd_ij: i -> j direction
        - nearest_wrong_jsd_ji: j -> i direction
        - nearest_wrong_jsd_bi: bidirectional average
        - alignment_margin_ij, _ji, _bi
        - retrieval_at_1_ij, _ji, _bi
    """
    # Same-index JSD (symmetric)
    jsd_same = compute_same_index_jsd(beta_i, beta_j)
    
    # Nearest-wrong JSD (asymmetric - compute both directions)
    jsd_wrong_ij = compute_nearest_wrong_jsd(beta_i, beta_j)
    jsd_wrong_ji = compute_nearest_wrong_jsd(beta_j, beta_i)
    jsd_wrong_bi = (jsd_wrong_ij + jsd_wrong_ji) / 2
    
    # Alignment Margin
    margin_ij = compute_alignment_margin(jsd_same, jsd_wrong_ij)
    margin_ji = compute_alignment_margin(jsd_same, jsd_wrong_ji)
    margin_bi = (margin_ij + margin_ji) / 2
    
    # Retrieval@1 (asymmetric)
    retrieval_ij = compute_retrieval_at_1(beta_i, beta_j)
    retrieval_ji = compute_retrieval_at_1(beta_j, beta_i)
    retrieval_bi = (retrieval_ij + retrieval_ji) / 2
    
    return {
        'corpus_pair': f'{corpus_i}<->{corpus_j}',
        'same_index_jsd': float(jsd_same),
        'nearest_wrong_jsd': {
            f'{corpus_i}->{corpus_j}': float(jsd_wrong_ij),
            f'{corpus_j}->{corpus_i}': float(jsd_wrong_ji),
            'bidirectional': float(jsd_wrong_bi)
        },
        'alignment_margin': {
            f'{corpus_i}->{corpus_j}': float(margin_ij),
            f'{corpus_j}->{corpus_i}': float(margin_ji),
            'bidirectional': float(margin_bi)
        },
        'retrieval_at_1': {
            f'{corpus_i}->{corpus_j}': float(retrieval_ij),
            f'{corpus_j}->{corpus_i}': float(retrieval_ji),
            'bidirectional': float(retrieval_bi)
        }
    }


def evaluate_source_adaptation(checkpoint_path):
    """
    Evaluate alignment metrics for source adaptation model
    Uses model's built-in get_beta_source() for each corpus
    """
    print("="*80)
    print("SOURCE ADAPTATION ALIGNMENT METRICS")
    print("="*80)
    
    print(f"\nLoading model: {checkpoint_path}")
    model = load_model(checkpoint_path)
    
    # Source IDs: 0=COHA, 1=HBR, 2=ILR
    corpus_names = ['COHA', 'HBR', 'ILR']
    source_ids = [0, 1, 2]
    
    # Extract betas for each source
    betas = {}
    for name, src_id in zip(corpus_names, source_ids):
        print(f"  Extracting beta for {name} (src_id={src_id})...")
        betas[name] = get_source_beta_from_model(model, src_id)
        print(f"    Shape: {betas[name].shape}")
    
    # Compute metrics for all pairs
    results = {
        'mode': 'source_adaptation',
        'checkpoint': checkpoint_path,
        'num_topics': betas['COHA'].shape[0],
        'num_times': betas['COHA'].shape[1],
        'vocab_size': betas['COHA'].shape[2],
        'pairwise_metrics': []
    }
    
    for corpus_i, corpus_j in combinations(corpus_names, 2):
        print(f"\nComputing metrics for {corpus_i} <-> {corpus_j}...")
        metrics = compute_all_metrics(
            betas[corpus_i], betas[corpus_j], 
            corpus_i, corpus_j
        )
        results['pairwise_metrics'].append(metrics)
        
        print(f"  Same-index JSD:      {metrics['same_index_jsd']:.6f}")
        print(f"  Nearest-wrong JSD:   {metrics['nearest_wrong_jsd']['bidirectional']:.6f}")
        print(f"  Alignment Margin:    {metrics['alignment_margin']['bidirectional']:.6f}")
        print(f"  Retrieval@1:         {metrics['retrieval_at_1']['bidirectional']:.4f} ({metrics['retrieval_at_1']['bidirectional']*100:.1f}%)")
    
    # Compute overall averages
    avg_same_jsd = np.mean([m['same_index_jsd'] for m in results['pairwise_metrics']])
    avg_wrong_jsd = np.mean([m['nearest_wrong_jsd']['bidirectional'] for m in results['pairwise_metrics']])
    avg_margin = np.mean([m['alignment_margin']['bidirectional'] for m in results['pairwise_metrics']])
    avg_retrieval = np.mean([m['retrieval_at_1']['bidirectional'] for m in results['pairwise_metrics']])
    
    results['overall_averages'] = {
        'same_index_jsd': float(avg_same_jsd),
        'nearest_wrong_jsd': float(avg_wrong_jsd),
        'alignment_margin': float(avg_margin),
        'retrieval_at_1': float(avg_retrieval)
    }
    
    print("\n" + "="*80)
    print("OVERALL AVERAGES (across all corpus pairs)")
    print("="*80)
    print(f"  Same-index JSD:      {avg_same_jsd:.6f}  (lower = better)")
    print(f"  Nearest-wrong JSD:   {avg_wrong_jsd:.6f}  (higher = better)")
    print(f"  Alignment Margin:    {avg_margin:.6f}  (higher = better)")
    print(f"  Retrieval@1:         {avg_retrieval:.4f} ({avg_retrieval*100:.1f}%)  (higher = better)")
    
    return results


def evaluate_full_finetune(coha_ckpt, hbr_ckpt, ilr_ckpt):
    """
    Evaluate alignment metrics for full fine-tune models
    Each model is loaded separately and global beta is extracted
    """
    print("="*80)
    print("FULL FINE-TUNE ALIGNMENT METRICS")
    print("="*80)
    
    corpus_names = ['COHA', 'HBR', 'ILR']
    checkpoints = [coha_ckpt, hbr_ckpt, ilr_ckpt]
    
    # Load models and extract betas
    betas = {}
    for name, ckpt in zip(corpus_names, checkpoints):
        print(f"\nLoading {name} model: {ckpt}")
        model = load_model(ckpt)
        betas[name] = get_beta_from_model(model)
        print(f"  Beta shape: {betas[name].shape}")
    
    # Compute metrics for all pairs
    results = {
        'mode': 'full_finetune',
        'checkpoints': {
            'COHA': coha_ckpt,
            'HBR': hbr_ckpt,
            'ILR': ilr_ckpt
        },
        'num_topics': betas['COHA'].shape[0],
        'num_times': betas['COHA'].shape[1],
        'vocab_size': betas['COHA'].shape[2],
        'pairwise_metrics': []
    }
    
    for corpus_i, corpus_j in combinations(corpus_names, 2):
        print(f"\nComputing metrics for {corpus_i} <-> {corpus_j}...")
        metrics = compute_all_metrics(
            betas[corpus_i], betas[corpus_j], 
            corpus_i, corpus_j
        )
        results['pairwise_metrics'].append(metrics)
        
        print(f"  Same-index JSD:      {metrics['same_index_jsd']:.6f}")
        print(f"  Nearest-wrong JSD:   {metrics['nearest_wrong_jsd']['bidirectional']:.6f}")
        print(f"  Alignment Margin:    {metrics['alignment_margin']['bidirectional']:.6f}")
        print(f"  Retrieval@1:         {metrics['retrieval_at_1']['bidirectional']:.4f} ({metrics['retrieval_at_1']['bidirectional']*100:.1f}%)")
    
    # Compute overall averages
    avg_same_jsd = np.mean([m['same_index_jsd'] for m in results['pairwise_metrics']])
    avg_wrong_jsd = np.mean([m['nearest_wrong_jsd']['bidirectional'] for m in results['pairwise_metrics']])
    avg_margin = np.mean([m['alignment_margin']['bidirectional'] for m in results['pairwise_metrics']])
    avg_retrieval = np.mean([m['retrieval_at_1']['bidirectional'] for m in results['pairwise_metrics']])
    
    results['overall_averages'] = {
        'same_index_jsd': float(avg_same_jsd),
        'nearest_wrong_jsd': float(avg_wrong_jsd),
        'alignment_margin': float(avg_margin),
        'retrieval_at_1': float(avg_retrieval)
    }
    
    print("\n" + "="*80)
    print("OVERALL AVERAGES (across all corpus pairs)")
    print("="*80)
    print(f"  Same-index JSD:      {avg_same_jsd:.6f}  (lower = better)")
    print(f"  Nearest-wrong JSD:   {avg_wrong_jsd:.6f}  (higher = better)")
    print(f"  Alignment Margin:    {avg_margin:.6f}  (higher = better)")
    print(f"  Retrieval@1:         {avg_retrieval:.4f} ({avg_retrieval*100:.1f}%)  (higher = better)")
    
    return results


def print_comparison_table(sa_results, ff_results):
    """Print a comparison table between source adaptation and full fine-tune"""
    print("\n" + "="*80)
    print("COMPARISON: SOURCE ADAPTATION vs FULL FINE-TUNE")
    print("="*80)
    
    print("\n{:<25} {:>15} {:>15} {:>10}".format(
        "Metric", "Source Adapt", "Full Finetune", "Winner"))
    print("-"*65)
    
    # Same-index JSD (lower = better)
    sa_same = sa_results['overall_averages']['same_index_jsd']
    ff_same = ff_results['overall_averages']['same_index_jsd']
    winner = "SA ✓" if sa_same < ff_same else "FF ✓"
    print("{:<25} {:>15.6f} {:>15.6f} {:>10}".format(
        "Same-index JSD ↓", sa_same, ff_same, winner))
    
    # Nearest-wrong JSD (higher = better)
    sa_wrong = sa_results['overall_averages']['nearest_wrong_jsd']
    ff_wrong = ff_results['overall_averages']['nearest_wrong_jsd']
    winner = "SA ✓" if sa_wrong > ff_wrong else "FF ✓"
    print("{:<25} {:>15.6f} {:>15.6f} {:>10}".format(
        "Nearest-wrong JSD ↑", sa_wrong, ff_wrong, winner))
    
    # Alignment Margin (higher = better)
    sa_margin = sa_results['overall_averages']['alignment_margin']
    ff_margin = ff_results['overall_averages']['alignment_margin']
    winner = "SA ✓" if sa_margin > ff_margin else "FF ✓"
    print("{:<25} {:>15.6f} {:>15.6f} {:>10}".format(
        "Alignment Margin ↑", sa_margin, ff_margin, winner))
    
    # Retrieval@1 (higher = better)
    sa_ret = sa_results['overall_averages']['retrieval_at_1']
    ff_ret = ff_results['overall_averages']['retrieval_at_1']
    winner = "SA ✓" if sa_ret > ff_ret else "FF ✓"
    print("{:<25} {:>14.1f}% {:>14.1f}% {:>10}".format(
        "Retrieval@1 ↑", sa_ret*100, ff_ret*100, winner))
    
    print("-"*65)
    print("\n↓ = lower is better, ↑ = higher is better")
    print("SA = Source Adaptation, FF = Full Fine-tune")


def save_results(results, output_path):
    """Save results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute cross-corpus topic alignment metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Source adaptation model:
    python alignment_metrics.py --mode source_adaptation \\
        --checkpoint /path/to/source_adapted.pt \\
        --output sa_alignment.json

    # Full fine-tune models:
    python alignment_metrics.py --mode full_finetune \\
        --coha_checkpoint /path/to/coha.pt \\
        --hbr_checkpoint /path/to/hbr.pt \\
        --ilr_checkpoint /path/to/ilr.pt \\
        --output ff_alignment.json

    # Compare both:
    python alignment_metrics.py --mode compare \\
        --checkpoint /path/to/source_adapted.pt \\
        --coha_checkpoint /path/to/coha.pt \\
        --hbr_checkpoint /path/to/hbr.pt \\
        --ilr_checkpoint /path/to/ilr.pt \\
        --output comparison.json
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['source_adaptation', 'full_finetune', 'compare'],
                        help='Evaluation mode')
    parser.add_argument('--checkpoint', type=str,
                        help='Source adaptation model checkpoint')
    parser.add_argument('--coha_checkpoint', type=str,
                        help='Full fine-tune COHA checkpoint')
    parser.add_argument('--hbr_checkpoint', type=str,
                        help='Full fine-tune HBR checkpoint')
    parser.add_argument('--ilr_checkpoint', type=str,
                        help='Full fine-tune ILR checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    if args.mode == 'source_adaptation':
        if not args.checkpoint:
            parser.error("--checkpoint required for source_adaptation mode")
        results = evaluate_source_adaptation(args.checkpoint)
        save_results(results, args.output)
        
    elif args.mode == 'full_finetune':
        if not all([args.coha_checkpoint, args.hbr_checkpoint, args.ilr_checkpoint]):
            parser.error("All three corpus checkpoints required for full_finetune mode")
        results = evaluate_full_finetune(
            args.coha_checkpoint, args.hbr_checkpoint, args.ilr_checkpoint)
        save_results(results, args.output)
        
    elif args.mode == 'compare':
        if not args.checkpoint:
            parser.error("--checkpoint required for compare mode")
        if not all([args.coha_checkpoint, args.hbr_checkpoint, args.ilr_checkpoint]):
            parser.error("All three corpus checkpoints required for compare mode")
        
        sa_results = evaluate_source_adaptation(args.checkpoint)
        print("\n")
        ff_results = evaluate_full_finetune(
            args.coha_checkpoint, args.hbr_checkpoint, args.ilr_checkpoint)
        
        print_comparison_table(sa_results, ff_results)
        
        combined_results = {
            'source_adaptation': sa_results,
            'full_finetune': ff_results
        }
        save_results(combined_results, args.output)
    
    print("\n" + "="*80)
    print("ALIGNMENT METRICS EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
