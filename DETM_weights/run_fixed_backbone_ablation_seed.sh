#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SEED="${1:?Usage: $0 SEED CUDA_DEVICE}"
CUDA_DEVICE="${2:?Usage: $0 SEED CUDA_DEVICE}"
SEED_PADDED=$(printf "%04d" "$SEED")

PYTHON="/user/rl3403/.conda/envs/nlp_kogut/bin/python"
ABLATE_PY="$SCRIPT_DIR/ablate_adaptation.py"

BACKBONE_CKPT="$PROJECT_ROOT/detm_merged_5year/merged_topic20_min_df100_delta0.01_20260325_001151/detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt"
RUN_ROOT="$PROJECT_ROOT/multiseed_runs/fixed_backbone_ablation/seed_${SEED_PADDED}"
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints/ablations"
LOG_DIR="$RUN_ROOT/logs"
METRIC_DIR="$RUN_ROOT/metrics"
LOG_FILE="$LOG_DIR/fixed_backbone_ablation_seed${SEED_PADDED}_gpu${CUDA_DEVICE}.log"

mkdir -p "$CHECKPOINT_ROOT" "$LOG_DIR" "$METRIC_DIR"

if [[ ! -f "$BACKBONE_CKPT" ]]; then
  echo "Missing backbone checkpoint: $BACKBONE_CKPT" >&2
  exit 1
fi

{
  echo "============================================"
  echo "Fixed-backbone SB-RA ablation seed run"
  echo "============================================"
  echo "Seed:             $SEED"
  echo "CUDA device:      $CUDA_DEVICE"
  echo "Backbone:         $BACKBONE_CKPT"
  echo "Run root:         $RUN_ROOT"
  echo "Checkpoint root:  $CHECKPOINT_ROOT"
  echo "Metric dir:       $METRIC_DIR"
  echo "Started:          $(date --iso-8601=seconds)"
  echo "============================================"

  if [[ -x /usr/bin/time ]]; then
    TIME_CMD=(/usr/bin/time -v)
  else
    echo "WARNING: /usr/bin/time not found; relying on main.py timing and CUDA memory logs only."
    TIME_CMD=()
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${TIME_CMD[@]}" "$PYTHON" -u "$ABLATE_PY" \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --backbone_ckpt "$BACKBONE_CKPT" \
    --out_root "$CHECKPOINT_ROOT" \
    --out_csv "$METRIC_DIR/table5_ablation_seed${SEED_PADDED}.csv" \
    --out_json "$METRIC_DIR/table5_ablation_seed${SEED_PADDED}.json"

  echo "============================================"
  echo "Finished:         $(date --iso-8601=seconds)"
  echo "Log saved to:     $LOG_FILE"
  echo "============================================"
} 2>&1 | tee "$LOG_FILE"
