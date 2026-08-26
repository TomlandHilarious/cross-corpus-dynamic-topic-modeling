#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_ROOT="${PROJECT_ROOT}/multiseed_runs/k_sensitivity_seed2019"
ALIGN_SCRIPT="${SCRIPT_DIR}/alignment_metrics_trajectory.py"
K_ARG="${1:-all}"

if [[ "${K_ARG}" == "all" ]]; then
  K_LIST=(10 30)
else
  K_LIST=("${K_ARG}")
fi

for K in "${K_LIST[@]}"; do
  ROOT="${RUN_ROOT}/K_${K}"
  METRICS_DIR="${ROOT}/metrics"
  mkdir -p "${METRICS_DIR}"

  SB_RA_CKPT="$(find "${ROOT}/checkpoints/sb_ra" -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1)"
  COHA_CKPT="$(find "${ROOT}/checkpoints/sb_ft/coha" -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1)"
  HBR_CKPT="$(find "${ROOT}/checkpoints/sb_ft/hbr" -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1)"
  ILR_CKPT="$(find "${ROOT}/checkpoints/sb_ft/ilr" -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1)"

  OUT_JSON="${METRICS_DIR}/k${K}_seed2019_reviewer_alignment_compare.json"
  OUT_LOG="${METRICS_DIR}/k${K}_seed2019_reviewer_alignment_compare.log"

  echo "============================================"
  echo "K-sensitivity reviewer alignment K=${K}"
  echo "============================================"
  echo "SB-RA: ${SB_RA_CKPT}"
  echo "COHA:  ${COHA_CKPT}"
  echo "HBR:   ${HBR_CKPT}"
  echo "ILR:   ${ILR_CKPT}"
  echo "Output: ${OUT_JSON}"

  "${PYTHON}" "${ALIGN_SCRIPT}" \
    --mode compare \
    --checkpoint "${SB_RA_CKPT}" \
    --coha_checkpoint "${COHA_CKPT}" \
    --hbr_checkpoint "${HBR_CKPT}" \
    --ilr_checkpoint "${ILR_CKPT}" \
    --output "${OUT_JSON}" 2>&1 | tee "${OUT_LOG}"
done

"${PYTHON}" - <<'PY'
from pathlib import Path
import csv
import json
import re

run_root = Path('/shared/share_hbr-ilr_nlp/shared_backbone_detm/multiseed_runs/k_sensitivity_seed2019')
metric_rows = []
for path in sorted(run_root.glob('K_*/metrics/k*_seed2019_reviewer_alignment_compare.json')):
    k = int(path.parents[1].name.split('_')[1])
    data = json.loads(path.read_text())
    for model_key, label in [('shared_backbone', 'SB-RA'), ('full_finetune', 'SB-FT')]:
        if model_key not in data:
            continue
        avg = data[model_key]['overall_averages']
        metric_rows.append({
            'K': k,
            'model': label,
            'same_jsd': avg.get('same_index_traj_jsd'),
            'hungarian_jsd': avg.get('hungarian_matched_traj_jsd'),
            'margin': avg.get('traj_margin'),
            'r1_pct': 100.0 * avg.get('traj_retrieval_at_1'),
            'nearest_wrong_jsd': avg.get('nearest_wrong_traj_jsd'),
            'timestamp_match_change_rate': avg.get('timestamp_match_change_rate', ''),
            'json': str(path),
        })
metric_out = run_root / 'k_sensitivity_reviewer_alignment_summary.csv'
with metric_out.open('w', newline='') as f:
    fieldnames = ['K', 'model', 'same_jsd', 'hungarian_jsd', 'margin', 'r1_pct', 'nearest_wrong_jsd', 'timestamp_match_change_rate', 'json']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metric_rows)

runtime_rows = []
for k_dir in sorted(run_root.glob('K_*')):
    k = int(k_dir.name.split('_')[1])
    logs = sorted((k_dir / 'logs').glob(f'k{k}_seed2019_gpu*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    completed = []
    for log in logs:
        text = log.read_text(errors='replace')
        if 'Finished:' in text:
            completed.append((log, text))
    if not completed:
        continue
    log, text = completed[0]
    lines = text.splitlines()
    gpu = ''
    for line in lines[:25]:
        if line.startswith('CUDA device:'):
            gpu = line.split(':', 1)[1].strip()
    current = None
    for i, line in enumerate(lines, 1):
        if 'SB-Joint pretraining' in line:
            current = ('SB-Joint', i)
        elif 'SB-RA main adaptation' in line:
            current = ('SB-RA', i)
        elif 'SB-FT full fine-tuning' in line:
            src = line.split('source=')[-1].strip() if 'source=' in line else ''
            current = (f'SB-FT {src}', i)
        elif '[run_timing] elapsed:' in line and current:
            match = re.search(r'elapsed:\s+([0-9:]+) \(([0-9.]+) seconds\)', line)
            if match:
                runtime_rows.append({
                    'K': k,
                    'stage': current[0],
                    'gpu': gpu,
                    'elapsed_hms': match.group(1),
                    'seconds': float(match.group(2)),
                    'log': str(log),
                    'stage_start_line': current[1],
                    'elapsed_line': i,
                })
            current = None
runtime_out = run_root / 'k_sensitivity_runtime_summary.csv'
with runtime_out.open('w', newline='') as f:
    fieldnames = ['K', 'stage', 'gpu', 'elapsed_hms', 'seconds', 'log', 'stage_start_line', 'elapsed_line']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(runtime_rows)
print(f'Saved metric summary: {metric_out}')
print(f'Saved runtime summary: {runtime_out}')
PY
