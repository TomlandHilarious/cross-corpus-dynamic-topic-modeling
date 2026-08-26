#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
###############################################################################
#   Full Fine-tuning Baseline: ILR
#   Start from merged backbone, unfreeze ALL parameters, train on ILR only
#   This serves as a baseline to compare against frozen backbone + delta_alpha
###############################################################################

export CUDA_VISIBLE_DEVICES=2

# ---------- Training parameters ----------
NUM_TOPICS=20
EMB_DIM=300
BSZ=500
MIN_DF=100
NUM_EPOCHS=20
LR=1e-5                         # Fine-tuning learning rate
DELTA=0.01
KL_ALPHA_SCALE=1e-6

# KL warmup for fine-tuning (match source adaptation config)
WARMUP_EPOCHS=5
KL_WEIGHT_MAX=0.3  # Match source adaptation: ADAPT_KL_THETA_MAX=0.3

# ---------- Merged backbone checkpoint ----------
MERGED_CKPT="$PROJECT_ROOT/detm_merged_5year/merged_topic20_min_df100_delta0.01_20260325_001151/detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt"

# ---------- Merged data (same vocab as backbone, filter to ILR docs) ----------
DATA_DIR="$PROJECT_ROOT/data_processing_scripts/merged_v2_min100_5year_v2"
EMB_PATH="$DATA_DIR/min_df_${MIN_DF}/merged_embedding.npy"
SOURCE_FILTER="ILR"  # Filter to only ILR documents

# ---------- Paths ----------
PYTHON="${PYTHON:-python}"
MAIN_PY="$SCRIPT_DIR/main.py"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
SAVE_ROOT="$PROJECT_ROOT/detm_full_finetune_baseline/ilr_fullft_topic${NUM_TOPICS}_${TIMESTAMP}"

###############################################################################
# Create output directory
mkdir -p "$SAVE_ROOT"

# Log file
LOG_FILE="$SAVE_ROOT/full_finetune_${TIMESTAMP}.log"

echo "============================================"
echo "Full Fine-tuning Baseline: ILR"
echo "============================================"
echo "Backbone checkpoint: $MERGED_CKPT"
echo "Data directory:      $DATA_DIR"
echo "Save directory:      $SAVE_ROOT"
echo "Epochs:              $NUM_EPOCHS"
echo "Learning rate:       $LR"
echo "Mode:                FULL FINE-TUNING (all parameters trainable)"
echo "============================================"

{
  $PYTHON "$MAIN_PY" \
      --mode train \
      --stage lora \
      --load_from "$MERGED_CKPT" \
      --full_finetune \
      --source_filter "$SOURCE_FILTER" \
      --dataset merged \
      --data_path "$DATA_DIR" \
      --emb_path "$EMB_PATH" \
      --save_path "$SAVE_ROOT" \
      --batch_size $BSZ \
      --lr $LR \
      --epochs $NUM_EPOCHS \
      --num_topics $NUM_TOPICS \
      --rho_size $EMB_DIM \
      --emb_size $EMB_DIM \
      --t_hidden_size 800 \
      --theta_act relu \
      --train_embeddings 1 \
      --enc_drop 0.0 \
      --clip 2.0 \
      --nonmono 10 \
      --min_df $MIN_DF \
      --optimizer adam \
      --delta $DELTA \
      --kl_alpha_scale $KL_ALPHA_SCALE \
      --warmup_epochs $WARMUP_EPOCHS \
      --kl_weight_max $KL_WEIGHT_MAX \
      --visualize_every 5 \
      --eval_batch_size 1000 \
      --bow_norm 1 \
      --num_words 10 \
      --log_interval 10 \
      --save_checkpoint_every 5

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Full fine-tuning completed!"
echo "Results saved to: $SAVE_ROOT"
echo "Log file: $LOG_FILE"

# Find the final checkpoint
FINAL_CKPT=$(ls -t "$SAVE_ROOT"/*.pt 2>/dev/null | head -1)

if [ -n "$FINAL_CKPT" ]; then
    echo ""
    echo "============================================"
    echo "Evaluating final checkpoint..."
    echo "Checkpoint: $FINAL_CKPT"
    echo "============================================"
    
    EVAL_OUTPUT="$SAVE_ROOT/evaluation_metrics.txt"
    
    $PYTHON "$MAIN_PY" \
        --mode eval \
        --load_from "$FINAL_CKPT" \
        --dataset "$DATA_DIR" \
        --data_path "$DATA_DIR" \
        --emb_path "$EMB_PATH" \
        --num_topics $NUM_TOPICS \
        --eval_batch_size 1000 \
        --min_df $MIN_DF \
        2>&1 | tee "$SAVE_ROOT/evaluation.log"
    
    echo "Evaluation completed! Check $SAVE_ROOT/evaluation.log"
else
    echo "WARNING: No checkpoint found in $SAVE_ROOT"
fi
