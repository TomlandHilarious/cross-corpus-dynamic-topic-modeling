#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
###############################################################################
#   Train vanilla DETM on ILR corpus only with MERGED vocabulary
#   Using 5-year temporal bins instead of yearly
###############################################################################

export CUDA_VISIBLE_DEVICES=5

# ---------- parameters ----------
NUM_TOPICS=20
EMB_DIM=300
BSZ=500
MIN_DF=100
NUM_EPOCHS=80
DELTA=0.01
KL_ALPHA_SCALE=1e-6
WARMUP_EPOCHS=50
KL_WEIGHT_MAX=0.9  # Cap KL weight to prevent TC plateau (default 1.0)
SAVE_CHECKPOINT_EVERY=5  # Save checkpoint every N epochs (0=only best)
# NOTE: num_times is auto-detected from data (will be ~20 for 5-year bins)

PYTHON="/user/rl3403/.conda/envs/nlp_kogut/bin/python"
MAIN_PY="$SCRIPT_DIR/main.py"

# Use merged vocabulary with 5-year rebinned timestamps (ILR only)
# NOTE: You must run rebin_individual_merged_vocab.sh first!
DATA_DIR="$PROJECT_ROOT/data_processing_scripts/individual_corpora_min100_5year_v3/ilr"
EMB_PATH="$DATA_DIR/min_df_${MIN_DF}/embedding.npy"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
SAVE_ROOT="$PROJECT_ROOT/detm_individual_merged_5year_v3/ilr_topic${NUM_TOPICS}_min_df${MIN_DF}_delta${DELTA}_${TIMESTAMP}"

###############################################################################
# Create output directory
mkdir -p "$SAVE_ROOT"

# Log file
LOG_FILE="$SAVE_ROOT/train_${TIMESTAMP}.log"

echo "============================================"
echo "Training ILR DETM with merged vocab (5-year bins)"
echo "============================================"
echo "Data path:   $DATA_DIR"
echo "Save path:   $SAVE_ROOT"
echo "Log file:    $LOG_FILE"
echo "Topics:      $NUM_TOPICS"
echo "Time slices: auto-detected from data (~20 for 5-year bins)"
echo "Epochs:      $NUM_EPOCHS"
echo "KL scale:    $KL_ALPHA_SCALE"
echo "Warmup:      $WARMUP_EPOCHS epochs"
echo "KL max:      $KL_WEIGHT_MAX"
echo "Checkpoint:  every $SAVE_CHECKPOINT_EVERY epochs"
echo "============================================"

# Run training
{
  $PYTHON -u "$MAIN_PY" \
  --stage pretrain \
  --dataset ilr \
  --data_path "$DATA_DIR" \
  --emb_path "$EMB_PATH" \
  --save_path "$SAVE_ROOT" \
  --num_topics $NUM_TOPICS \
  --emb_size $EMB_DIM \
  --rho_size $EMB_DIM \
  --batch_size $BSZ \
  --delta $DELTA \
  --lr 5e-05 \
  --epochs $NUM_EPOCHS \
  --min_df $MIN_DF \
  --train_embeddings 1 \
  --lora_rank 0 \
  --kl_alpha_scale $KL_ALPHA_SCALE \
  --warmup_epochs $WARMUP_EPOCHS \
  --kl_weight_max $KL_WEIGHT_MAX \
  --save_checkpoint_every $SAVE_CHECKPOINT_EVERY
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo "Training completed for ILR (merged vocab, 5-year bins)"
echo "Results saved to: $SAVE_ROOT"
echo "Log saved to:     $LOG_FILE"
echo "============================================"
