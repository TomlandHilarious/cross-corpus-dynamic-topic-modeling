#!/bin/bash
###############################################################################
# Rebin individual corpora (merged vocab) to 5-year bins
# Creates: individual_corpora_min100_5year/{hbr,coha,ilr}/min_df_100/
###############################################################################

PYTHON="${PYTHON:-python}"
REBIN_SCRIPT="/shared/share_hbr-ilr_nlp/data_processing_scripts/rebin_timestamps.py"
INPUT_BASE="/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_min100"
OUTPUT_BASE="/shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_min100_5year_v2"

BIN_SIZE=5

echo "============================================"
echo "Rebinning individual corpora (merged vocab) to ${BIN_SIZE}-year bins"
echo "============================================"

# HBR
echo ""
echo ">>> Processing HBR..."
$PYTHON "$REBIN_SCRIPT" \
  --bin_size $BIN_SIZE \
  --input_dir "${INPUT_BASE}/hbr/min_df_100" \
  --output_dir "${OUTPUT_BASE}/hbr/min_df_100"

# COHA
echo ""
echo ">>> Processing COHA..."
$PYTHON "$REBIN_SCRIPT" \
  --bin_size $BIN_SIZE \
  --input_dir "${INPUT_BASE}/coha/min_df_100" \
  --output_dir "${OUTPUT_BASE}/coha/min_df_100"

# ILR
echo ""
echo ">>> Processing ILR..."
$PYTHON "$REBIN_SCRIPT" \
  --bin_size $BIN_SIZE \
  --input_dir "${INPUT_BASE}/ilr/min_df_100" \
  --output_dir "${OUTPUT_BASE}/ilr/min_df_100"

echo ""
echo "============================================"
echo "All corpora rebinned successfully!"
echo "Output: ${OUTPUT_BASE}/"
echo "============================================"
