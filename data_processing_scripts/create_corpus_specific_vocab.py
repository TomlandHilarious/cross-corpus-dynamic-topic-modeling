#!/user/rl3403/.conda/envs/nlp_kogut/bin/python
"""
Create corpus-specific vocabularies for individual DETM training.
This filters the merged vocabulary to only include words that appear in each corpus,
reducing vocabulary size and focusing the model on relevant terms.

Usage:
    python create_corpus_specific_vocab.py
"""

import os
import numpy as np
import scipy.io as sio
import pickle
from collections import defaultdict

# Paths
# Use NEW merged vocab (V=19433) and the v3 per-corpus split that was
# produced from the same source. Output to a fresh _v3 dir to avoid
# colliding with the legacy v1/v2 dirs (which were derived from the OLD
# V=19461 merged vocab).
MERGED_DIR = "/shared/share_hbr-ilr_nlp/data_processing_scripts/merged_v2_min100_5year_v2/min_df_100"
INDIVIDUAL_DIR = "/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_min100_5year_v3"
OUTPUT_DIR = "/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab_5year_v3"

CORPORA = ['coha', 'hbr', 'ilr']

def load_vocab(vocab_path):
    """Load vocabulary from file."""
    with open(vocab_path, 'r') as f:
        vocab = [line.strip() for line in f]
    return vocab

def load_mat_data(mat_path, key):
    """Load .mat file and extract data."""
    data = sio.loadmat(mat_path)[key]
    if data.dtype == 'object':
        return data.flatten()
    return data

def save_mat_data(mat_path, data, key):
    """Save data to .mat file."""
    sio.savemat(mat_path, {key: data}, do_compression=True)

def get_active_vocab_indices(corpus_name, merged_vocab_size):
    """
    Find which vocabulary indices are actually used in a corpus.
    Returns a set of active indices and their frequencies.
    """
    corpus_dir = os.path.join(INDIVIDUAL_DIR, corpus_name, 'min_df_100')
    
    # Track which vocab indices appear in the corpus
    active_indices = set()
    
    # Load all token files
    for split in ['tr', 'va', 'ts']:
        tokens_file = os.path.join(corpus_dir, f'bow_{split}_tokens.mat')
        if not os.path.exists(tokens_file):
            print(f"Warning: {tokens_file} not found, skipping")
            continue
            
        tokens = load_mat_data(tokens_file, 'tokens')
        
        # Collect all unique token indices from all documents
        for doc_tokens in tokens:
            if doc_tokens is not None and len(doc_tokens) > 0:
                active_indices.update(doc_tokens.flatten())
    
    # Also check test halves
    for half in ['h1', 'h2']:
        tokens_file = os.path.join(corpus_dir, f'bow_ts_{half}_tokens.mat')
        if os.path.exists(tokens_file):
            tokens = load_mat_data(tokens_file, 'tokens')
            for doc_tokens in tokens:
                if doc_tokens is not None and len(doc_tokens) > 0:
                    active_indices.update(doc_tokens.flatten())
    
    return sorted(active_indices)

def create_corpus_specific_vocab(corpus_name, merged_vocab, merged_embeddings):
    """
    Create a corpus-specific vocabulary by filtering the merged vocab
    to only include words that appear in this corpus.
    """
    print(f"\n{'='*80}")
    print(f"Processing {corpus_name.upper()}")
    print(f"{'='*80}")
    
    # Get active vocabulary indices for this corpus
    active_indices = get_active_vocab_indices(corpus_name, len(merged_vocab))
    
    print(f"Original merged vocab size: {len(merged_vocab)}")
    print(f"Active vocab size in {corpus_name}: {len(active_indices)}")
    print(f"Reduction: {len(merged_vocab) - len(active_indices)} words ({100*(1-len(active_indices)/len(merged_vocab)):.1f}%)")
    
    # Create mapping from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(active_indices)}
    
    # Extract corpus-specific vocab and embeddings
    corpus_vocab = [merged_vocab[i] for i in active_indices]
    corpus_embeddings = merged_embeddings[active_indices]
    
    # Create output directory
    output_corpus_dir = os.path.join(OUTPUT_DIR, corpus_name, 'min_df_100')
    os.makedirs(output_corpus_dir, exist_ok=True)
    
    # Save corpus-specific vocab (both .txt and .pkl)
    vocab_file = os.path.join(output_corpus_dir, 'vocab.txt')
    with open(vocab_file, 'w') as f:
        for word in corpus_vocab:
            f.write(f"{word}\n")
    print(f"Saved vocab to: {vocab_file}")
    
    vocab_pkl = os.path.join(output_corpus_dir, 'vocab.pkl')
    with open(vocab_pkl, 'wb') as f:
        pickle.dump(corpus_vocab, f)
    print(f"Saved vocab.pkl to: {vocab_pkl}")
    
    # Save corpus-specific embeddings
    emb_file = os.path.join(output_corpus_dir, 'embedding.npy')
    np.save(emb_file, corpus_embeddings)
    print(f"Saved embeddings to: {emb_file}")
    
    # Remap and copy data files
    input_corpus_dir = os.path.join(INDIVIDUAL_DIR, corpus_name, 'min_df_100')
    
    for split in ['tr', 'va', 'ts']:
        remap_bow_files(input_corpus_dir, output_corpus_dir, split, old_to_new)
    
    # Remap test halves
    for half in ['h1', 'h2']:
        remap_bow_files(input_corpus_dir, output_corpus_dir, f'ts_{half}', old_to_new)
    
    # Copy timestamp and source files (no remapping needed)
    for split in ['tr', 'va', 'ts']:
        for suffix in ['timestamps', 'sources']:
            src = os.path.join(input_corpus_dir, f'bow_{split}_{suffix}.mat')
            dst = os.path.join(output_corpus_dir, f'bow_{split}_{suffix}.mat')
            if os.path.exists(src):
                import shutil
                shutil.copy(src, dst)
    
    # Copy timestamps.pkl and sources.pkl from merged directory
    import shutil
    for pkl_file in ['timestamps.pkl', 'sources.pkl']:
        src = os.path.join(MERGED_DIR, pkl_file)
        dst = os.path.join(output_corpus_dir, pkl_file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {pkl_file}")
    
    print(f"Completed {corpus_name.upper()}")
    return len(corpus_vocab)

def remap_bow_files(input_dir, output_dir, split, old_to_new):
    """
    Remap token indices in BOW files from merged vocab to corpus-specific vocab.
    """
    tokens_file = os.path.join(input_dir, f'bow_{split}_tokens.mat')
    counts_file = os.path.join(input_dir, f'bow_{split}_counts.mat')
    
    if not os.path.exists(tokens_file):
        print(f"  Skipping {split} (file not found)")
        return
    
    # Load original data
    tokens_data = load_mat_data(tokens_file, 'tokens')
    counts_data = load_mat_data(counts_file, 'counts')
    
    # Remap tokens
    remapped_tokens = []
    remapped_counts = []
    
    for doc_tokens, doc_counts in zip(tokens_data, counts_data):
        if doc_tokens is None or len(doc_tokens) == 0:
            remapped_tokens.append(np.array([]))
            remapped_counts.append(np.array([]))
            continue
        
        # Flatten if needed
        doc_tokens = doc_tokens.flatten()
        doc_counts = doc_counts.flatten()
        
        # Remap indices
        new_tokens = []
        new_counts = []
        for old_idx, count in zip(doc_tokens, doc_counts):
            old_idx = int(old_idx)
            if old_idx in old_to_new:
                new_tokens.append(old_to_new[old_idx])
                new_counts.append(count)
            else:
                # This shouldn't happen if our active indices are correct
                print(f"    Warning: Token {old_idx} not in active vocab, skipping")
        
        remapped_tokens.append(np.array(new_tokens, dtype=np.int64))
        remapped_counts.append(np.array(new_counts, dtype=np.int64))
    
    # Save remapped data in (1, N) object array format
    remapped_tokens_array = np.empty((1, len(remapped_tokens)), dtype=object)
    remapped_counts_array = np.empty((1, len(remapped_counts)), dtype=object)
    
    for i, (tokens, counts) in enumerate(zip(remapped_tokens, remapped_counts)):
        remapped_tokens_array[0, i] = tokens
        remapped_counts_array[0, i] = counts
    
    # Save
    save_mat_data(os.path.join(output_dir, f'bow_{split}_tokens.mat'), 
                  remapped_tokens_array, 'tokens')
    save_mat_data(os.path.join(output_dir, f'bow_{split}_counts.mat'), 
                  remapped_counts_array, 'counts')
    
    print(f"  Remapped {split}: {len(remapped_tokens)} documents")

def main():
    """Main function to create corpus-specific vocabularies."""
    print("="*80)
    print("Creating Corpus-Specific Vocabularies")
    print("="*80)
    
    # Load merged vocabulary and embeddings
    print("\nLoading merged vocabulary and embeddings...")
    merged_vocab = load_vocab(os.path.join(MERGED_DIR, 'vocab.txt'))
    merged_embeddings = np.load(os.path.join(MERGED_DIR, 'merged_embedding.npy'))
    
    print(f"Merged vocab size: {len(merged_vocab)}")
    print(f"Embedding dimension: {merged_embeddings.shape[1]}")
    
    # Process each corpus
    results = {}
    for corpus in CORPORA:
        vocab_size = create_corpus_specific_vocab(corpus, merged_vocab, merged_embeddings)
        results[corpus] = vocab_size
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original merged vocab: {len(merged_vocab)} words")
    for corpus in CORPORA:
        reduction = len(merged_vocab) - results[corpus]
        pct = 100 * (1 - results[corpus] / len(merged_vocab))
        print(f"{corpus.upper():6s} specific vocab: {results[corpus]:5d} words (removed {reduction:5d}, {pct:4.1f}%)")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nTo use corpus-specific vocabularies, update your training scripts to use:")
    print("  DATA_DIR=\"/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab\"")

if __name__ == '__main__':
    main()
