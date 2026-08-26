#!/usr/bin/env python
"""
Split merged corpus data into individual COHA/HBR/ILR datasets.
Keeps the same vocab and embeddings for fair comparison.
"""

import os
import pickle
import numpy as np
import scipy.io

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_scipy(path, data, varname='data'):
    scipy.io.savemat(path, {varname: data}, do_compression=True)

def split_corpus_data(merged_dir, output_dir):
    """
    Split merged data by source while keeping shared vocab/embeddings.
    
    Args:
        merged_dir: path to merged_v2_min100_fixed/min_df_100/
        output_dir: base path for output (will create coha/, hbr/, ilr/ subdirs)
    """
    print("Loading merged data...")
    
    # Load source mapping
    sources = load_pickle(os.path.join(merged_dir, 'sources.pkl'))
    print(f"Total documents: {len(sources)}")
    unique_sources, counts = np.unique(sources, return_counts=True)
    print(f"Unique source values: {unique_sources}")
    print(f"Source counts: {counts}")
    for src, cnt in zip(unique_sources, counts):
        print(f"  Source {src}: {cnt} docs")
    
    # Load shared vocab and embeddings (will be copied to each corpus dir)
    vocab = load_pickle(os.path.join(merged_dir, 'vocab.pkl'))
    timestamps = load_pickle(os.path.join(merged_dir, 'timestamps.pkl'))
    embeddings = np.load(os.path.join(merged_dir, 'merged_embedding.npy'))
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Num timestamps: {len(timestamps)}")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Load train/val/test data
    # Note: scipy.io.loadmat returns arrays that might be (1, N) or (N, 1), need to handle properly
    train_tokens = scipy.io.loadmat(os.path.join(merged_dir, 'bow_tr_tokens.mat'))['tokens']
    train_counts = scipy.io.loadmat(os.path.join(merged_dir, 'bow_tr_counts.mat'))['counts']
    train_times = scipy.io.loadmat(os.path.join(merged_dir, 'bow_tr_timestamps.mat'))['timestamps']
    train_sources = scipy.io.loadmat(os.path.join(merged_dir, 'bow_tr_sources.mat'))['sources']
    
    val_tokens = scipy.io.loadmat(os.path.join(merged_dir, 'bow_va_tokens.mat'))['tokens']
    val_counts = scipy.io.loadmat(os.path.join(merged_dir, 'bow_va_counts.mat'))['counts']
    val_times = scipy.io.loadmat(os.path.join(merged_dir, 'bow_va_timestamps.mat'))['timestamps']
    val_sources = scipy.io.loadmat(os.path.join(merged_dir, 'bow_va_sources.mat'))['sources']
    
    test_tokens = scipy.io.loadmat(os.path.join(merged_dir, 'bow_ts_tokens.mat'))['tokens']
    test_counts = scipy.io.loadmat(os.path.join(merged_dir, 'bow_ts_counts.mat'))['counts']
    test_times = scipy.io.loadmat(os.path.join(merged_dir, 'bow_ts_timestamps.mat'))['timestamps']
    test_sources = scipy.io.loadmat(os.path.join(merged_dir, 'bow_ts_sources.mat'))['sources']
    
    # Load test half files (h1 and h2) for perplexity calculation
    test_h1_tokens = scipy.io.loadmat(os.path.join(merged_dir, 'bow_ts_h1_tokens.mat'))['tokens']
    test_h1_counts = scipy.io.loadmat(os.path.join(merged_dir, 'bow_ts_h1_counts.mat'))['counts']
    
    test_h2_tokens = scipy.io.loadmat(os.path.join(merged_dir, 'bow_ts_h2_tokens.mat'))['tokens']
    test_h2_counts = scipy.io.loadmat(os.path.join(merged_dir, 'bow_ts_h2_counts.mat'))['counts']
    
    print(f"Train tokens shape: {train_tokens.shape}, dtype: {train_tokens.dtype}")
    print(f"Train counts shape: {train_counts.shape}, dtype: {train_counts.dtype}")
    print(f"Train times shape: {train_times.shape}, dtype: {train_times.dtype}")
    print(f"Train sources shape: {train_sources.shape}, dtype: {train_sources.dtype}")
    
    # Check actual distribution in each split
    train_src_flat = train_sources.flatten()
    val_src_flat = val_sources.flatten()
    test_src_flat = test_sources.flatten()
    
    print("\nSource distribution in TRAIN:")
    train_unique, train_src_counts = np.unique(train_src_flat, return_counts=True)
    for src, cnt in zip(train_unique, train_src_counts):
        print(f"  {src}: {cnt} docs")
    
    print("\nSource distribution in VAL:")
    val_unique, val_src_counts = np.unique(val_src_flat, return_counts=True)
    for src, cnt in zip(val_unique, val_src_counts):
        print(f"  {src}: {cnt} docs")
    
    print("\nSource distribution in TEST:")
    test_unique, test_src_counts = np.unique(test_src_flat, return_counts=True)
    for src, cnt in zip(test_unique, test_src_counts):
        print(f"  {src}: {cnt} docs")
    
    # Source mapping - sources are strings like 'COHA', 'HBR', 'ILR'
    source_mapping = {'COHA': 'coha', 'HBR': 'hbr', 'ILR': 'ilr'}
    source_names = list(source_mapping.keys())
    
    for src_upper in source_names:
        src_lower = source_mapping[src_upper]
        print(f"\n{'='*80}")
        print(f"Processing {src_upper}...")
        print(f"{'='*80}")
        
        # Create output directory with min_df_100 subdirectory to match data structure
        src_dir = os.path.join(output_dir, src_lower, 'min_df_100')
        os.makedirs(src_dir, exist_ok=True)
        
        # Filter train data - sources are strings (strip whitespace!)
        train_sources_flat = train_sources.flatten()
        train_mask = np.array([s.strip() == src_upper for s in train_sources_flat])
        
        # tokens/counts/times are object arrays, need to flatten first
        train_tokens_flat = train_tokens.flatten()
        train_counts_flat = train_counts.flatten()
        train_times_flat = train_times.flatten()
        
        src_train_tokens = train_tokens_flat[train_mask]
        src_train_counts = train_counts_flat[train_mask]
        src_train_times = train_times_flat[train_mask]
        
        # Filter val data
        val_sources_flat = val_sources.flatten()
        val_mask = np.array([s.strip() == src_upper for s in val_sources_flat])
        
        val_tokens_flat = val_tokens.flatten()
        val_counts_flat = val_counts.flatten()
        val_times_flat = val_times.flatten()
        
        src_val_tokens = val_tokens_flat[val_mask]
        src_val_counts = val_counts_flat[val_mask]
        src_val_times = val_times_flat[val_mask]
        
        # Filter test data
        test_sources_flat = test_sources.flatten()
        test_mask = np.array([s.strip() == src_upper for s in test_sources_flat])
        
        test_tokens_flat = test_tokens.flatten()
        test_counts_flat = test_counts.flatten()
        test_times_flat = test_times.flatten()
        
        src_test_tokens = test_tokens_flat[test_mask]
        src_test_counts = test_counts_flat[test_mask]
        src_test_times = test_times_flat[test_mask]
        
        # Filter test h1 and h2 (same mask as test)
        test_h1_tokens_flat = test_h1_tokens.flatten()
        test_h1_counts_flat = test_h1_counts.flatten()
        
        test_h2_tokens_flat = test_h2_tokens.flatten()
        test_h2_counts_flat = test_h2_counts.flatten()
        
        src_test_h1_tokens = test_h1_tokens_flat[test_mask]
        src_test_h1_counts = test_h1_counts_flat[test_mask]
        
        src_test_h2_tokens = test_h2_tokens_flat[test_mask]
        src_test_h2_counts = test_h2_counts_flat[test_mask]
        
        print(f"Train: {len(src_train_tokens)} docs")
        print(f"Val:   {len(src_val_tokens)} docs")
        print(f"Test:  {len(src_test_tokens)} docs")
        
        # Save filtered data
        # Reshape back to (1, N) to match original format
        print("Saving data...")
        save_scipy(os.path.join(src_dir, 'bow_tr_tokens.mat'), src_train_tokens.reshape(1, -1), 'tokens')
        save_scipy(os.path.join(src_dir, 'bow_tr_counts.mat'), src_train_counts.reshape(1, -1), 'counts')
        save_scipy(os.path.join(src_dir, 'bow_tr_timestamps.mat'), src_train_times.reshape(1, -1), 'timestamps')
        # Save sources (all same label for individual corpus)
        src_train_sources = np.array([src_upper] * len(src_train_tokens), dtype=object).reshape(1, -1)
        save_scipy(os.path.join(src_dir, 'bow_tr_sources.mat'), src_train_sources, 'sources')
        
        save_scipy(os.path.join(src_dir, 'bow_va_tokens.mat'), src_val_tokens.reshape(1, -1), 'tokens')
        save_scipy(os.path.join(src_dir, 'bow_va_counts.mat'), src_val_counts.reshape(1, -1), 'counts')
        save_scipy(os.path.join(src_dir, 'bow_va_timestamps.mat'), src_val_times.reshape(1, -1), 'timestamps')
        src_val_sources = np.array([src_upper] * len(src_val_tokens), dtype=object).reshape(1, -1)
        save_scipy(os.path.join(src_dir, 'bow_va_sources.mat'), src_val_sources, 'sources')
        
        save_scipy(os.path.join(src_dir, 'bow_ts_tokens.mat'), src_test_tokens.reshape(1, -1), 'tokens')
        save_scipy(os.path.join(src_dir, 'bow_ts_counts.mat'), src_test_counts.reshape(1, -1), 'counts')
        save_scipy(os.path.join(src_dir, 'bow_ts_timestamps.mat'), src_test_times.reshape(1, -1), 'timestamps')
        src_test_sources = np.array([src_upper] * len(src_test_tokens), dtype=object).reshape(1, -1)
        save_scipy(os.path.join(src_dir, 'bow_ts_sources.mat'), src_test_sources, 'sources')
        
        # Save test h1 and h2 files
        save_scipy(os.path.join(src_dir, 'bow_ts_h1_tokens.mat'), src_test_h1_tokens.reshape(1, -1), 'tokens')
        save_scipy(os.path.join(src_dir, 'bow_ts_h1_counts.mat'), src_test_h1_counts.reshape(1, -1), 'counts')
        
        save_scipy(os.path.join(src_dir, 'bow_ts_h2_tokens.mat'), src_test_h2_tokens.reshape(1, -1), 'tokens')
        save_scipy(os.path.join(src_dir, 'bow_ts_h2_counts.mat'), src_test_h2_counts.reshape(1, -1), 'counts')
        
        # Copy shared vocab, timestamps, and embeddings
        print("Copying shared vocab and embeddings...")
        with open(os.path.join(src_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab, f)
        with open(os.path.join(src_dir, 'timestamps.pkl'), 'wb') as f:
            pickle.dump(timestamps, f)
        np.save(os.path.join(src_dir, 'embedding.npy'), embeddings)
        
        # Save vocab.txt for easy inspection
        with open(os.path.join(src_dir, 'vocab.txt'), 'w') as f:
            for word in vocab:
                f.write(f"{word}\n")
        
        print(f"✓ {src_upper} data saved to {src_dir}")
    
    print(f"\n{'='*80}")
    print("All corpora split successfully!")
    print(f"{'='*80}")

if __name__ == '__main__':
    # Use NEW merged vocab (V=19433); the same source data the NEW backbone
    # was trained on. Output goes to a fresh "_v3" tree to avoid colliding
    # with the legacy v1/v2 dirs (which were both built from the OLD V=19461
    # merged vocab despite the misleading "_v2" suffix).
    merged_dir = '/shared/share_hbr-ilr_nlp/data_processing_scripts/merged_v2_min100_5year_v2/min_df_100'
    output_dir = '/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_min100_5year_v3'

    split_corpus_data(merged_dir, output_dir)
