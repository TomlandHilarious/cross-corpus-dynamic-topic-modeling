#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

NUM_TOPICS="${1:?Usage: $0 NUM_TOPICS CUDA_DEVICE}"
CUDA_DEVICE="${2:?Usage: $0 NUM_TOPICS CUDA_DEVICE}"
SEED=2019

EMB_DIM=300
BSZ=500
MIN_DF=100
DELTA=0.01
KL_ALPHA_SCALE=1e-6
PRETRAIN_EPOCHS=80
PRETRAIN_WARMUP_EPOCHS=50
PRETRAIN_KL_WEIGHT_MAX=0.9
ADAPT_EPOCHS=20
ADAPT_LR=1e-05
ADAPT_KL_THETA_MAX=0.3
ADAPT_WARMUP_EPOCHS=5
LAMBDA_ANCHOR=1e-3
LAMBDA_SMOOTH=1e-3
FT_EPOCHS=20
FT_LR=1e-5
FT_WARMUP_EPOCHS=5
FT_KL_WEIGHT_MAX=0.3

PYTHON="${PYTHON:-python}"
MAIN_PY="$SCRIPT_DIR/main.py"
DATA_DIR="$PROJECT_ROOT/data_processing_scripts/merged_v2_min100_5year_v2"
EMB_PATH="$DATA_DIR/min_df_${MIN_DF}/merged_embedding.npy"
RUN_ROOT="$PROJECT_ROOT/multiseed_runs/k_sensitivity_seed2019/K_${NUM_TOPICS}"
LOG_DIR="$RUN_ROOT/logs"
SB_JOINT_DIR="$RUN_ROOT/checkpoints/sb_joint"
SB_RA_DIR="$RUN_ROOT/checkpoints/sb_ra"
SB_FT_ROOT="$RUN_ROOT/checkpoints/sb_ft"
METRIC_DIR="$RUN_ROOT/metrics"
LOG_FILE="$LOG_DIR/k${NUM_TOPICS}_seed${SEED}_gpu${CUDA_DEVICE}.log"

mkdir -p "$LOG_DIR" "$SB_JOINT_DIR" "$SB_RA_DIR" "$SB_FT_ROOT/coha" "$SB_FT_ROOT/hbr" "$SB_FT_ROOT/ilr" "$METRIC_DIR"

find_ckpt() {
  local dir="$1"
  local pattern="$2"
  find "$dir" -maxdepth 1 -type f -name "$pattern" | sort | tail -1
}

run_stage() {
  echo ""
  echo "============================================"
  echo "$1"
  echo "============================================"
}

{
  echo "============================================"
  echo "K-sensitivity seed 2019"
  echo "============================================"
  echo "K:                $NUM_TOPICS"
  echo "Seed:             $SEED"
  echo "CUDA device:      $CUDA_DEVICE"
  echo "Data:             $DATA_DIR"
  echo "Run root:         $RUN_ROOT"
  echo "Started:          $(date --iso-8601=seconds)"
  echo "============================================"

  SB_JOINT_CKPT=$(find_ckpt "$SB_JOINT_DIR" "*_pretrain.pt")
  if [[ -z "$SB_JOINT_CKPT" ]]; then
    run_stage "SB-Joint pretraining K=$NUM_TOPICS"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -u "$MAIN_PY" \
      --stage pretrain \
      --dataset merged \
      --data_path "$DATA_DIR" \
      --emb_path "$EMB_PATH" \
      --save_path "$SB_JOINT_DIR" \
      --num_topics "$NUM_TOPICS" \
      --emb_size "$EMB_DIM" \
      --rho_size "$EMB_DIM" \
      --batch_size "$BSZ" \
      --delta "$DELTA" \
      --lr 5e-05 \
      --epochs "$PRETRAIN_EPOCHS" \
      --min_df "$MIN_DF" \
      --train_embeddings 1 \
      --lora_rank 0 \
      --kl_alpha_scale "$KL_ALPHA_SCALE" \
      --warmup_epochs "$PRETRAIN_WARMUP_EPOCHS" \
      --kl_weight_max "$PRETRAIN_KL_WEIGHT_MAX" \
      --save_checkpoint_every 5 \
      --seed "$SEED"
    SB_JOINT_CKPT=$(find_ckpt "$SB_JOINT_DIR" "*_pretrain.pt")
  else
    echo "[SKIP SB-Joint] existing checkpoint: $SB_JOINT_CKPT"
  fi

  if [[ -z "$SB_JOINT_CKPT" ]]; then
    echo "ERROR: SB-Joint checkpoint not found in $SB_JOINT_DIR" >&2
    exit 1
  fi

  SB_RA_CKPT=$(find_ckpt "$SB_RA_DIR" "*lora_r*.pt")
  if [[ -z "$SB_RA_CKPT" ]]; then
    run_stage "SB-RA main adaptation K=$NUM_TOPICS"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -u "$MAIN_PY" \
      --stage lora \
      --dataset merged \
      --data_path "$DATA_DIR" \
      --emb_path "$EMB_PATH" \
      --save_path "$SB_RA_DIR" \
      --load_from "$SB_JOINT_CKPT" \
      --num_topics "$NUM_TOPICS" \
      --emb_size "$EMB_DIM" \
      --rho_size "$EMB_DIM" \
      --batch_size "$BSZ" \
      --delta "$DELTA" \
      --lr "$ADAPT_LR" \
      --epochs "$ADAPT_EPOCHS" \
      --min_df "$MIN_DF" \
      --train_embeddings 1 \
      --source_adaptation_mode 1 \
      --lambda_anchor "$LAMBDA_ANCHOR" \
      --lambda_smooth "$LAMBDA_SMOOTH" \
      --freeze_rho_in_adaptation 1 \
      --freeze_alpha_in_adaptation 1 \
      --kl_alpha_scale "$KL_ALPHA_SCALE" \
      --adapt_kl_theta_max "$ADAPT_KL_THETA_MAX" \
      --adapt_warmup_epochs "$ADAPT_WARMUP_EPOCHS" \
      --visualize_every 5 \
      --save_checkpoint_every 5 \
      --seed "$SEED"
    SB_RA_CKPT=$(find_ckpt "$SB_RA_DIR" "*lora_r*.pt")
  else
    echo "[SKIP SB-RA] existing checkpoint: $SB_RA_CKPT"
  fi

  if [[ -z "$SB_RA_CKPT" ]]; then
    echo "ERROR: SB-RA checkpoint not found in $SB_RA_DIR" >&2
    exit 1
  fi

  for SOURCE in COHA HBR ILR; do
    source_lower=$(echo "$SOURCE" | tr '[:upper:]' '[:lower:]')
    FT_DIR="$SB_FT_ROOT/$source_lower"
    FT_CKPT=$(find_ckpt "$FT_DIR" "*.pt")
    if [[ -n "$FT_CKPT" ]]; then
      echo "[SKIP SB-FT $SOURCE] existing checkpoint: $FT_CKPT"
      continue
    fi

    run_stage "SB-FT full fine-tuning K=$NUM_TOPICS source=$SOURCE"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -u "$MAIN_PY" \
      --mode train \
      --stage lora \
      --load_from "$SB_JOINT_CKPT" \
      --full_finetune \
      --source_filter "$SOURCE" \
      --dataset merged \
      --data_path "$DATA_DIR" \
      --emb_path "$EMB_PATH" \
      --save_path "$FT_DIR" \
      --batch_size "$BSZ" \
      --lr "$FT_LR" \
      --epochs "$FT_EPOCHS" \
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
      --warmup_epochs "$FT_WARMUP_EPOCHS" \
      --kl_weight_max "$FT_KL_WEIGHT_MAX" \
      --visualize_every 5 \
      --eval_batch_size 1000 \
      --bow_norm 1 \
      --num_words 10 \
      --log_interval 10 \
      --save_checkpoint_every 5 \
      --seed "$SEED"
  done

  echo "============================================"
  echo "Finished:         $(date --iso-8601=seconds)"
  echo "SB-Joint ckpt:    $SB_JOINT_CKPT"
  echo "SB-RA ckpt:       $SB_RA_CKPT"
  echo "Log saved to:     $LOG_FILE"
  echo "============================================"
} 2>&1 | tee "$LOG_FILE"
