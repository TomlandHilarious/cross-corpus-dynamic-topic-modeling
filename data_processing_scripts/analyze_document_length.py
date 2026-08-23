#!/user/rl3403/.conda/envs/nlp_kogut/bin/python

"""
Analyze document length distributions from merged dataset by source.
Outputs statistics (mean, median, quartiles) for each source (COHA, HBR, ILR).
"""

import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ============== CONFIG ==============
# Same configuration as in merged_dataset_preprocessing.py
BASE_DIR = '/shared/share_hbr-ilr_nlp'
ROOT_FOLDER = os.path.join(BASE_DIR, 'Merged_1920plus')
YEAR_MIN, YEAR_MAX = 1922, 2019
OUTPUT_FILE = 'doc_length_stats.csv'
OUTPUT_PLOT = 'doc_length_distributions.png'
# ====================================

def remove_not_printable(in_str):
    """Remove non-printable characters from string."""
    return "".join([c for c in in_str if c in string.printable])

def process_document(file_path):
    """Process a document the same way as in merged_dataset_preprocessing.py."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        doc = f.read()
    
    # lowercase, replace newline with space, replace quotes with space, remove punctuation and digits
    words = doc.split()
    
    return len(words)

def main():
    print(f"Reading data from {ROOT_FOLDER}...")
    # Initialize data structures to store lengths by source
    doc_lengths = defaultdict(list)
    path_counts = defaultdict(int)
    
    # Read all documents
    for year_folder in sorted(os.listdir(ROOT_FOLDER)):
        # Skip years outside range
        if not (year_folder.isdigit() and YEAR_MIN <= int(year_folder) <= YEAR_MAX):
            continue
            
        year_dir = os.path.join(ROOT_FOLDER, year_folder)
        for fname in os.listdir(year_dir):
            if not fname.endswith('.txt'):
                continue
                
            full_path = os.path.join(year_dir, fname)
            # Extract source: COHA, HBR, or ILR
            src = fname.split('_', 1)[0]
            
            # Process document and get length
            length = process_document(full_path)
            doc_lengths[src].append(length)
            path_counts[src] += 1
    
    # Calculate statistics
    stats = {}
    for src in sorted(doc_lengths.keys()):
        lengths = np.array(doc_lengths[src])
        stats[src] = {
            'count': len(lengths),
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'q1': np.percentile(lengths, 25),
            'median': np.median(lengths),
            'q3': np.percentile(lengths, 75),
            'max': np.max(lengths),
            'lengths_below_50': sum(lengths < 50),
            'percent_below_50': sum(lengths < 50) / len(lengths) * 100,
            'lengths_below_100': sum(lengths < 100),
            'percent_below_100': sum(lengths < 100) / len(lengths) * 100,
            'lengths_below_200': sum(lengths < 200),
            'percent_below_200': sum(lengths < 200) / len(lengths) * 100
        }
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(OUTPUT_FILE)
    print(f"Statistics saved to {OUTPUT_FILE}")
    
    # Also print to console
    print("\nDocument Length Statistics by Source:")
    print(stats_df.round(2))
    
    # Create histograms
    plt.figure(figsize=(15, 10))
    
    # Plot histogram for each source
    for i, (src, lengths) in enumerate(sorted(doc_lengths.items())):
        plt.subplot(2, 2, i+1)
        plt.hist(lengths, bins=50, alpha=0.7)
        plt.title(f'{src} Document Lengths (n={len(lengths)})')
        plt.xlabel('Document Length (words)')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add vertical lines for reference thresholds
        plt.axvline(50, color='r', linestyle='--', label='50 words')
        plt.axvline(100, color='g', linestyle='--', label='100 words')
        plt.axvline(200, color='b', linestyle='--', label='200 words')
        plt.legend()
    
    # Combined plot with log scale
    plt.subplot(2, 2, 4)
    for src, lengths in sorted(doc_lengths.items()):
        plt.hist(lengths, bins=50, alpha=0.5, label=src)
    plt.title('Combined Document Lengths (Log Scale)')
    plt.xlabel('Document Length (words)')
    plt.ylabel('Frequency (log)')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plots saved to {OUTPUT_PLOT}")
    
    # Print recommendations
    print("\nRecommendations:")
    print("Based on the statistics, you might consider the following thresholds:")
    all_lengths = np.concatenate(list(doc_lengths.values()))
    
    # Suggest potential thresholds based on distribution
    suggestions = [
        np.percentile(all_lengths, 10),  # 10th percentile
        np.percentile(all_lengths, 25),  # 25th percentile
        50,  # common minimum threshold
        100  # another common threshold
    ]
    
    for threshold in suggestions:
        percent_removed = sum(all_lengths < threshold) / len(all_lengths) * 100
        print(f"  - Threshold {threshold:.0f} words: would remove {percent_removed:.2f}% of documents")

if __name__ == "__main__":
    main()