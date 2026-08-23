#!/user/rl3403/.conda/envs/nlp_kogut/bin/python
"""
Rebin temporal resolution of preprocessed DETM data.
Takes existing yearly data and creates N-year bins (e.g., 2-year, 5-year intervals).

Usage:
    python rebin_timestamps.py --bin_size 5 --input_dir <path> --output_dir <path>
    python rebin_timestamps.py --bin_size 2  # uses defaults
"""

import os
import sys
import argparse
import shutil
import numpy as np
import scipy.io as sio
import pickle

def load_mat_data(mat_path, key):
    """Load .mat file and extract data."""
    data = sio.loadmat(mat_path)[key]
    if data.dtype == 'object':
        return data.flatten()
    return data.flatten()

def save_mat_data(mat_path, data, key):
    """Save data to .mat file."""
    # Reshape to (1, N) for consistency
    if data.ndim == 1:
        data = data.reshape(1, -1)
    sio.savemat(mat_path, {key: data}, do_compression=True)

def rebin_timestamps(timestamps, bin_size):
    """
    Rebin timestamps into N-year intervals.
    
    Args:
        timestamps: array of time indices (0, 1, 2, ..., T-1)
        bin_size: number of original time steps per bin
    
    Returns:
        rebinned timestamps, number of new time bins
    """
    rebinned = timestamps // bin_size
    num_bins = int(rebinned.max()) + 1
    return rebinned, num_bins

def rebin_dataset(input_dir, output_dir, bin_size):
    """
    Rebin all timestamp files in a dataset.
    
    Args:
        input_dir: path to input data (e.g., merged_v2_min100_fixed/min_df_100)
        output_dir: path to output data (will be created)
        bin_size: number of years per bin
    """
    print(f"\n{'='*80}")
    print(f"Rebinning dataset: {bin_size}-year intervals")
    print(f"{'='*80}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original year labels from timestamps.txt or timestamps.pkl
    original_year_labels = None
    timestamps_txt_in = os.path.join(input_dir, 'timestamps.txt')
    timestamps_pkl_in = os.path.join(input_dir, 'timestamps.pkl')
    
    if os.path.exists(timestamps_txt_in):
        with open(timestamps_txt_in, 'r') as f:
            original_year_labels = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(original_year_labels)} year labels from timestamps.txt")
    elif os.path.exists(timestamps_pkl_in):
        with open(timestamps_pkl_in, 'rb') as f:
            original_year_labels = pickle.load(f)
        print(f"Loaded {len(original_year_labels)} year labels from timestamps.pkl")
    else:
        print("Warning: No timestamps file found, year mapping will use indices only")
    
    # Track original and new time range
    original_times = None
    rebinned_times = None
    num_new_bins = None
    
    # Process each split
    for split in ['tr', 'va', 'ts']:
        timestamps_file = os.path.join(input_dir, f'bow_{split}_timestamps.mat')
        
        if not os.path.exists(timestamps_file):
            print(f"Warning: {timestamps_file} not found, skipping")
            continue
        
        # Load timestamps
        timestamps = load_mat_data(timestamps_file, 'timestamps')
        
        if original_times is None:
            original_times = (int(timestamps.min()), int(timestamps.max()))
            print(f"Original time range: {original_times[0]} to {original_times[1]} ({original_times[1] - original_times[0] + 1} time steps)")
        
        # Rebin
        rebinned, num_bins = rebin_timestamps(timestamps, bin_size)
        
        if rebinned_times is None:
            rebinned_times = (int(rebinned.min()), int(rebinned.max()))
            num_new_bins = num_bins
            print(f"Rebinned time range: {rebinned_times[0]} to {rebinned_times[1]} ({num_bins} time bins)")
            
            # Calculate how many docs in last bin
            last_bin_years = (original_times[1] + 1) % bin_size
            if last_bin_years == 0:
                last_bin_years = bin_size
            print(f"Last bin contains {last_bin_years} year(s) (others contain {bin_size} years)")
        
        # Save rebinned timestamps
        output_file = os.path.join(output_dir, f'bow_{split}_timestamps.mat')
        save_mat_data(output_file, rebinned, 'timestamps')
        print(f"  Saved {split}: {len(rebinned)} documents")
    
    # Create year mapping
    year_mapping = {}
    if original_year_labels is not None:
        # Map each bin to its starting year
        for bin_idx in range(num_new_bins):
            # Get the first original time index in this bin
            first_original_idx = bin_idx * bin_size
            if first_original_idx < len(original_year_labels):
                # Try to parse as int (year), otherwise keep as string
                try:
                    year_mapping[bin_idx] = int(original_year_labels[first_original_idx])
                except (ValueError, TypeError):
                    year_mapping[bin_idx] = original_year_labels[first_original_idx]
            else:
                year_mapping[bin_idx] = bin_idx  # Fallback to index
    else:
        # No year labels, just use indices
        year_mapping = {i: i for i in range(num_new_bins)}
    
    # Save year mapping to file
    year_mapping_txt = os.path.join(output_dir, 'year_mapping.txt')
    with open(year_mapping_txt, 'w') as f:
        f.write("# Mapping from bin index to starting year\n")
        f.write("# Format: bin_index\tstarting_year\n")
        for bin_idx in range(num_new_bins):
            f.write(f"{bin_idx}\t{year_mapping[bin_idx]}\n")
    print(f"\nSaved year mapping to: {year_mapping_txt}")
    
    # Also save as pickle for easy loading
    year_mapping_pkl = os.path.join(output_dir, 'year_mapping.pkl')
    with open(year_mapping_pkl, 'wb') as f:
        pickle.dump(year_mapping, f)
    print(f"Saved year mapping to: {year_mapping_pkl}")
    
    # Update timestamps.pkl
    timestamps_pkl_out = os.path.join(output_dir, 'timestamps.pkl')
    # Create new timestamp list for rebinned data (just indices)
    new_timestamps = list(range(num_new_bins))
    with open(timestamps_pkl_out, 'wb') as f:
        pickle.dump(new_timestamps, f)
    print(f"\nUpdated timestamps.pkl: {len(new_timestamps)} bins")
    
    # Update timestamps.txt
    timestamps_txt_out = os.path.join(output_dir, 'timestamps.txt')
    with open(timestamps_txt_out, 'w') as f:
        for t in range(num_new_bins):
            f.write(f"{t}\n")
    print(f"Updated timestamps.txt: {num_new_bins} bins")
    
    # Copy all other files unchanged
    print("\nCopying other files...")
    files_to_copy = []
    for filename in os.listdir(input_dir):
        if 'timestamps' not in filename:  # Skip timestamp files (already handled)
            files_to_copy.append(filename)
    
    for filename in files_to_copy:
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.isfile(src):
            shutil.copy(src, dst)
    
    print(f"Copied {len(files_to_copy)} additional files")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Original resolution: {original_times[1] - original_times[0] + 1} time steps (yearly)")
    print(f"New resolution:      {num_new_bins} time bins ({bin_size}-year intervals)")
    print(f"Reduction factor:    ~{(original_times[1] - original_times[0] + 1) / num_new_bins:.1f}x")
    
    # Print year mapping
    if original_year_labels is not None and isinstance(year_mapping.get(0), int):
        print(f"\nYear mapping (bin → starting year):")
        for i in range(min(5, num_new_bins)):
            end_year = year_mapping.get(i+1, year_mapping[i] + bin_size) - 1
            if i == num_new_bins - 1:
                # Last bin might be shorter
                actual_end = int(original_year_labels[-1]) if len(original_year_labels) > 0 else end_year
                print(f"  Bin {i}: {year_mapping[i]}-{actual_end}")
            else:
                print(f"  Bin {i}: {year_mapping[i]}-{end_year}")
        if num_new_bins > 5:
            print(f"  ...")
            i = num_new_bins - 1
            actual_end = int(original_year_labels[-1]) if len(original_year_labels) > 0 else year_mapping[i] + bin_size - 1
            print(f"  Bin {i}: {year_mapping[i]}-{actual_end}")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"\nTo use rebinned data, update your training script:")
    print(f"  DATA_DIR=\"{os.path.dirname(output_dir)}\"")
    print(f"  --num_times {num_new_bins}")
    print(f"\nYear mapping saved to: {year_mapping_txt}")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description='Rebin temporal resolution of DETM data')
    parser.add_argument('--bin_size', type=int, required=True,
                        help='Number of years per bin (e.g., 2, 3, 5)')
    parser.add_argument('--input_dir', type=str,
                        default='/shared/share_hbr-ilr_nlp/data_processing_scripts/merged_v2_min100_fixed/min_df_100',
                        help='Input data directory (contains bow_*.mat files)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output data directory (default: auto-generated based on bin_size)')
    
    args = parser.parse_args()
    
    # Auto-generate output directory name if not specified
    if args.output_dir is None:
        base_dir = os.path.dirname(args.input_dir)
        parent_dir = os.path.dirname(base_dir)
        output_base = os.path.join(parent_dir, f'merged_v2_min100_{args.bin_size}year')
        args.output_dir = os.path.join(output_base, 'min_df_100')
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    if args.bin_size < 1:
        print(f"Error: bin_size must be >= 1, got {args.bin_size}")
        sys.exit(1)
    
    # Run rebinning
    rebin_dataset(args.input_dir, args.output_dir, args.bin_size)

if __name__ == '__main__':
    main()
