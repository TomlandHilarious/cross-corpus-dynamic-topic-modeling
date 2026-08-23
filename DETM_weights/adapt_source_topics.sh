#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
###############################################################################
# Source-Specific Topic Adaptation from Merged Backbone
# Learns delta_alpha residuals for each source while keeping shared backbone
###############################################################################

export CUDA_VISIBLE_DEVICES=0

# ---------- Merged backbone checkpoint ----------
MERGED_CKPT="$PROJECT_ROOT/detm_merged_5year/merged_topic20_min_df100_delta0.01_20260325_001151/detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt"

# ---------- Shared parameters (match merged training) ----------
NUM_TOPICS=20
EMB_DIM=300
BSZ=500
MIN_DF=100
DELTA=0.01
KL_ALPHA_SCALE=1e-6

# ---------- Adaptation-specific parameters ----------
ADAPT_LR=1e-05                    # Learning rate for delta_alpha and inference networks
ADAPT_EPOCHS=20                   # Number of adaptation epochs (shorter for sanity check)

# KL regularization (CRITICAL: different from pretraining!)
ADAPT_KL_THETA_MAX=0.3            # Maximum KL_theta weight (lower than pretraining)
ADAPT_WARMUP_EPOCHS=5             # KL warmup epochs for adaptation

# Delta-alpha regularization
LAMBDA_ANCHOR=1e-3                # Penalize deviation from backbone: ||delta_alpha||^2
LAMBDA_SMOOTH=1e-3                # Penalize temporal jumps: ||delta_alpha[t] - delta_alpha[t-1]||^2

# Freezing
FREEZE_RHO=1                      # 1=freeze word embeddings, 0=train
FREEZE_ALPHA=1                    # 1=freeze shared alpha_global, 0=train

# ---------- Paths ----------
PYTHON="/user/rl3403/.conda/envs/nlp_kogut/bin/python"
MAIN_PY="$SCRIPT_DIR/main.py"
DATA_DIR="$PROJECT_ROOT/data_processing_scripts/merged_v2_min100_5year_v2"
EMB_PATH="$DATA_DIR/min_df_${MIN_DF}/merged_embedding.npy"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
OUTPUT_ROOT="$PROJECT_ROOT/detm_source_adapted_5year"
SAVE_DIR="${OUTPUT_ROOT}/adapt_kl${ADAPT_KL_THETA_MAX}_anchor${LAMBDA_ANCHOR}_${TIMESTAMP}"

mkdir -p "$SAVE_DIR"

LOG_FILE="$SAVE_DIR/adaptation_${TIMESTAMP}.log"

echo "============================================"
echo "Source-Specific Topic Adaptation"
echo "============================================"
echo "Backbone:         $MERGED_CKPT"
echo "Data:             $DATA_DIR"
echo "Save to:          $SAVE_DIR"
echo "Learning rate:    $ADAPT_LR"
echo "Epochs:           $ADAPT_EPOCHS"
echo "KL Regularization:"
echo "  adapt_kl_theta_max: $ADAPT_KL_THETA_MAX"
echo "  warmup_epochs:      $ADAPT_WARMUP_EPOCHS"
echo "Delta-alpha Regularization:"
echo "  lambda_anchor:      $LAMBDA_ANCHOR"
echo "  lambda_smooth:      $LAMBDA_SMOOTH"
echo "Freezing:"
echo "  rho (embeddings):   $FREEZE_RHO"
echo "  alpha (backbone):   $FREEZE_ALPHA"
echo "============================================"

{
    $PYTHON -u "$MAIN_PY" \
      --stage lora \
      --dataset merged \
      --data_path "$DATA_DIR" \
      --emb_path "$EMB_PATH" \
      --save_path "$SAVE_DIR" \
      --load_from "$MERGED_CKPT" \
      --num_topics $NUM_TOPICS \
      --emb_size $EMB_DIM \
      --rho_size $EMB_DIM \
      --batch_size $BSZ \
      --delta $DELTA \
      --lr $ADAPT_LR \
      --epochs $ADAPT_EPOCHS \
      --min_df $MIN_DF \
      --train_embeddings 1 \
      --source_adaptation_mode 1 \
      --lambda_anchor $LAMBDA_ANCHOR \
      --lambda_smooth $LAMBDA_SMOOTH \
      --freeze_rho_in_adaptation $FREEZE_RHO \
      --freeze_alpha_in_adaptation $FREEZE_ALPHA \
      --kl_alpha_scale $KL_ALPHA_SCALE \
      --adapt_kl_theta_max $ADAPT_KL_THETA_MAX \
      --adapt_warmup_epochs $ADAPT_WARMUP_EPOCHS \
      --visualize_every 5 \
      --save_checkpoint_every 5
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo "Adaptation complete!"
echo "Log saved to: $LOG_FILE"
echo "============================================"
