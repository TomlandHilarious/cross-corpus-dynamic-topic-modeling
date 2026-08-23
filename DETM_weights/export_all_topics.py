#!/usr/bin/env python3
"""
Export all topics with top words for all time periods and sources.
Generates comprehensive topic-word lists for manual inspection.
"""

import pickle
import numpy as np
import torch
import csv
from pathlib import Path

def load_model_and_vocab(checkpoint_path, vocab_path):
    """Load trained model and vocabulary"""
    with open(checkpoint_path, 'rb') as f:
        model = torch.load(f, map_location='cpu', weights_only=False)
    model.eval()
    
    with open(vocab_path, 'rb') as f:
       vocab = pickle.load(f)
    
    return model, vocab

def extract_topic_distributions(model):
    """Extract shared and source-specific topic-word distributions"""
    with torch.no_grad():
        # Get alpha
        if hasattr(model, 'mu_q_alpha'):
            alpha = model.mu_q_alpha.detach().cpu()
        else:
            alpha, _ = model.get_alpha()
            alpha = alpha.detach().cpu()
        
        # Get rho
        if hasattr(model.rho, 'weight'):
            rho = model.rho.weight.detach().cpu()
        else:
            rho = model.rho.detach().cpu()
        
        K, T, L = alpha.shape
        V = rho.shape[0]
        
        # Get delta_alpha - it's a 4D tensor [num_sources, K, T, L]
        if not hasattr(model, 'delta_alpha') or model.delta_alpha is None:
            print("WARNING: Model does not have source-specific delta_alpha parameters.")
            return None, None
        
        delta_alpha_tensor = model.delta_alpha.detach().cpu()
        num_sources = delta_alpha_tensor.shape[0]
        
        # Source names
        source_names = ['COHA', 'HBR', 'ILR']
        if num_sources != 3:
            print(f"WARNING: Model has {num_sources} sources, expected 3")
            source_names = [f'Source_{i}' for i in range(num_sources)]
        
        # Compute shared beta
        beta_shared = torch.zeros(K, T, V)
        for k in range(K):
            for t in range(T):
                logit = torch.matmul(alpha[k, t], rho.t())
                beta_shared[k, t] = torch.softmax(logit, dim=0)
        
        # Compute source-specific beta
        beta_sources = {}
        for src_id in range(num_sources):
            source_name = source_names[src_id]
            beta_source = torch.zeros(K, T, V)
            for k in range(K):
                for t in range(T):
                    alpha_source = alpha[k, t] + delta_alpha_tensor[src_id, k, t]
                    logit = torch.matmul(alpha_source, rho.t())
                    beta_source[k, t] = torch.softmax(logit, dim=0)
            beta_sources[source_name] = beta_source.numpy()
        
    return beta_shared.numpy(), beta_sources

def get_top_words(beta, vocab, n=15, stopwords=None):
    """Get top n words for a topic distribution, filtering stopwords"""
    if stopwords is None:
        stopwords = set()
    
    # Get more candidates to account for filtered words
    top_indices = np.argsort(beta)[-(n*3):][::-1]
    
    words = []
    for idx in top_indices:
        word = vocab[idx]
        if word not in stopwords:
            words.append(word)
            if len(words) >= n:
                break
    
    return words

def export_to_csv(beta_shared, beta_sources, vocab, output_path, stopwords=None):
    """Export all topics to CSV format"""
    K, T, V = beta_shared.shape
    sources = ['COHA', 'HBR', 'ILR']
    
    # Year labels
    years = [1922 + 5*i for i in range(T)]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Topic', 'Year', 'Source', 'Top_15_Words'])
        
        # For each topic
        for k in range(K):
            # For each time period
            for t in range(T):
                year = years[t]
                
                # Shared topic
                shared_words = get_top_words(beta_shared[k, t], vocab, 15, stopwords)
                writer.writerow([f'Topic_{k}', year, 'Shared', ', '.join(shared_words)])
                
                # Each source
                for source in sources:
                    source_words = get_top_words(beta_sources[source][k, t], vocab, 15, stopwords)
                    writer.writerow([f'Topic_{k}', year, source, ', '.join(source_words)])
    
    print(f"Exported to CSV: {output_path}")
    print(f"Total rows: {K * T * 4} (20 topics × 20 years × 4 versions)")

def export_to_markdown(beta_shared, beta_sources, vocab, output_dir, stopwords=None):
    """Export each topic to a separate markdown file"""
    K, T, V = beta_shared.shape
    sources = ['COHA', 'HBR', 'ILR']
    years = [1922 + 5*i for i in range(T)]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create one file per topic
    for k in range(K):
        md_path = output_dir / f'topic_{k:02d}_all_periods.md'
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Topic {k}: Complete Time Series\n\n")
            
            # For each time period
            for t in range(T):
                year = years[t]
                f.write(f"## Year {year}\n\n")
                
                # Shared
                shared_words = get_top_words(beta_shared[k, t], vocab, 15, stopwords)
                f.write(f"**Shared**: {', '.join(shared_words)}\n\n")
                
                # Each source
                for source in sources:
                    source_words = get_top_words(beta_sources[source][k, t], vocab, 15, stopwords)
                    f.write(f"**{source}**: {', '.join(source_words)}\n\n")
                
                f.write("---\n\n")
    
    print(f"Exported {K} markdown files to: {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export all topics with complete time series')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--format', type=str, default='both', choices=['csv', 'markdown', 'both'],
                        help='Output format (default: both)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, vocab = load_model_and_vocab(args.checkpoint, args.vocab)
    
    # Extract distributions
    print("Extracting topic distributions...")
    beta_shared, beta_sources = extract_topic_distributions(model)
    
    if beta_shared is None:
        print("ERROR: Model does not have source-specific parameters.")
        return
    
    K, T, V = beta_shared.shape
    print(f"Loaded: {K} topics, {T} time periods, {V} vocabulary size")
    
    # Define stopwords to filter
    stopwords = {
        "ion", "datum", "vol", "didst", "toolong", "www", "org", "br",
        "copyright", "reserve", "exhibit",
        "photo", "photograph", "chart", "table", "sidebar",
        "mg", "min", "taxis",
        "read", "date", "color"
    }
    print(f"Filtering {len(stopwords)} stopwords")
    
    # Export
    if args.format in ['csv', 'both']:
        csv_path = output_dir / 'all_topics_complete.csv'
        export_to_csv(beta_shared, beta_sources, vocab, csv_path, stopwords)
    
    if args.format in ['markdown', 'both']:
        md_dir = output_dir / 'topics_markdown'
        export_to_markdown(beta_shared, beta_sources, vocab, md_dir, stopwords)
    
    print("\n✓ Export complete!")

if __name__ == '__main__':
    main()
