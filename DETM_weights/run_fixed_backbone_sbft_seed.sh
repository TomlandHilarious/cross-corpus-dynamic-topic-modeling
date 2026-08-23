#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SEED="${1:?Usage: $0 SEED CUDA_DEVICE}"
CUDA_DEVICE="${2:?Usage: $0 SEED CUDA_DEVICE}"
SEED_PADDED=$(printf "%04d" "$SEED")

PYTHON="/user/rl3403/.conda/envs/nlp_kogut/bin/python"
MAIN_PY="$SCRIPT_DIR/main.py"

BACKBONE_CKPT="$PROJECT_ROOT/detm_merged_5year/merged_topic20_min_df100_delta0.01_20260325_001151/detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt"
DATA_DIR="$PROJECT_ROOT/data_processing_scripts/merged_v2_min100_5year_v2"
EMB_PATH="$DATA_DIR/min_df_100/merged_embedding.npy"
RUN_ROOT="$PROJECT_ROOT/multiseed_runs/fixed_backbone_ablation/seed_${SEED_PADDED}"
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints/sb_ft"
LOG_DIR="$RUN_ROOT/logs"
METRIC_DIR="$RUN_ROOT/metrics"
LOG_FILE="$LOG_DIR/fixed_backbone_sbft_seed${SEED_PADDED}_gpu${CUDA_DEVICE}.log"

NUM_TOPICS=20
EMB_DIM=300
BSZ=500
MIN_DF=100
NUM_EPOCHS=20
LR=1e-5
DELTA=0.01
KL_ALPHA_SCALE=1e-6
WARMUP_EPOCHS=5
KL_WEIGHT_MAX=0.3

mkdir -p "$CHECKPOINT_ROOT/coha" "$CHECKPOINT_ROOT/hbr" "$CHECKPOINT_ROOT/ilr" "$LOG_DIR" "$METRIC_DIR"

if [[ ! -f "$BACKBONE_CKPT" ]]; then
  echo "Missing backbone checkpoint: $BACKBONE_CKPT" >&2
  exit 1
fi

run_source() {
  local source="$1"
  local source_lc="$2"
  local save_root="$CHECKPOINT_ROOT/$source_lc"
  mkdir -p "$save_root"

  local existing_ckpt
  existing_ckpt="$(find "$save_root" -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1 || true)"
  if [[ -n "$existing_ckpt" ]]; then
    echo "[SKIP TRAIN] SB-FT source=${source} existing_ckpt=${existing_ckpt}"
    return 0
  fi

  echo "============================================"
  echo "SB-FT full fine-tuning seed=${SEED} source=${source}"
  echo "============================================"
  echo "Save root: $save_root"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -u "$MAIN_PY" \
    --mode train \
    --stage lora \
    --load_from "$BACKBONE_CKPT" \
    --full_finetune \
    --source_filter "$source" \
    --dataset merged \
    --data_path "$DATA_DIR" \
    --emb_path "$EMB_PATH" \
    --save_path "$save_root" \
    --batch_size "$BSZ" \
    --lr "$LR" \
    --epochs "$NUM_EPOCHS" \
    --num_topics "$NUM_TOPICS" \
    --rho_size "$EMB_DIM" \
    --emb_size "$EMB_DIM" \
    --t_hidden_size 800 \
    --theta_act relu \
    --train_embeddings 1 \
    --enc_drop 0.0 \
    --clip 2.0 \
    --nonmono 10 \
    --min_df "$MIN_DF" \
    --optimizer adam \
    --delta "$DELTA" \
    --kl_alpha_scale "$KL_ALPHA_SCALE" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --kl_weight_max "$KL_WEIGHT_MAX" \
    --visualize_every 5 \
    --eval_batch_size 1000 \
    --bow_norm 1 \
    --num_words 10 \
    --log_interval 10 \
    --save_checkpoint_every 5 \
    --seed "$SEED"
}

{
  echo "============================================"
  echo "Fixed-backbone SB-FT seed run"
  echo "============================================"
  echo "Seed:             $SEED"
  echo "CUDA device:      $CUDA_DEVICE"
  echo "Backbone:         $BACKBONE_CKPT"
  echo "Run root:         $RUN_ROOT"
  echo "Checkpoint root:  $CHECKPOINT_ROOT"
  echo "Metric dir:       $METRIC_DIR"
  echo "Started:          $(date --iso-8601=seconds)"
  echo "============================================"

  run_source "COHA" "coha"
  run_source "HBR" "hbr"
  run_source "ILR" "ilr"

  echo "============================================"
  echo "Finished:         $(date --iso-8601=seconds)"
  echo "Log saved to:     $LOG_FILE"
  echo "============================================"
} 2>&1 | tee "$LOG_FILE"
