#!/bin/bash
###############################################################################
#   Rebin corpus-specific vocabulary data to use N-year temporal bins
#   Run this BEFORE training with the rebinned training scripts
###############################################################################

PYTHON="/user/rl3403/.conda/envs/nlp_kogut/bin/python"
REBIN_SCRIPT="/shared/share_hbr-ilr_nlp/data_processing_scripts/rebin_timestamps.py"

BIN_SIZE=5  # Change this to 2, 3, 5, etc.

echo "============================================"
echo "Rebinning corpus-specific vocab data"
echo "Bin size: ${BIN_SIZE} years"
echo "============================================"

# Rebin HBR
echo ""
echo "--- Rebinning HBR ---"
$PYTHON "$REBIN_SCRIPT" \
  --bin_size $BIN_SIZE \
  --input_dir "/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab/hbr/min_df_100" \
  --output_dir "/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab_${BIN_SIZE}year_v2/hbr/min_df_100"

# Rebin COHA
echo ""
echo "--- Rebinning COHA ---"
$PYTHON "$REBIN_SCRIPT" \
  --bin_size $BIN_SIZE \
  --input_dir "/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab/coha/min_df_100" \
  --output_dir "/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab_${BIN_SIZE}year_v2/coha/min_df_100"

# Rebin ILR
echo ""
echo "--- Rebinning ILR ---"
$PYTHON "$REBIN_SCRIPT" \
  --bin_size $BIN_SIZE \
  --input_dir "/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab/ilr/min_df_100" \
  --output_dir "/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab_${BIN_SIZE}year_v2/ilr/min_df_100"

echo ""
echo "============================================"
echo "Rebinning complete!"
echo "Output: individual_corpora_specific_vocab_${BIN_SIZE}year/"
echo ""
echo "Now you can run training scripts with:"
echo "  bash run_individual_hbr_specific_vocab_${BIN_SIZE}year.sh"
echo "  bash run_individual_coha_specific_vocab_${BIN_SIZE}year.sh"
echo "  bash run_individual_ilr_specific_vocab_${BIN_SIZE}year.sh"
echo "============================================"
