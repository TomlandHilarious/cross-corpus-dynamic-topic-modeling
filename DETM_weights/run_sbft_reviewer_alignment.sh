#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
ALIGN_SCRIPT="${SCRIPT_DIR}/alignment_metrics_trajectory.py"
BASELINE_ROOT="${PROJECT_ROOT}/detm_full_finetune_baseline"
METRICS_DIR="${SCRIPT_DIR}/results/metrics"
mkdir -p "${METRICS_DIR}"

COHA_CKPT="$(find "${BASELINE_ROOT}" -path '*coha_fullft_topic20_*/*.pt' -type f | sort | tail -n 1)"
HBR_CKPT="$(find "${BASELINE_ROOT}" -path '*hbr_fullft_topic20_*/*.pt' -type f | sort | tail -n 1)"
ILR_CKPT="$(find "${BASELINE_ROOT}" -path '*ilr_fullft_topic20_*/*.pt' -type f | sort | tail -n 1)"

OUT_JSON="${METRICS_DIR}/traj_align_E_fullft_current.json"
OUT_LOG="${METRICS_DIR}/traj_align_E_fullft_current.log"
OUT_CSV="${METRICS_DIR}/traj_align_E_fullft_current_summary.csv"

echo "============================================"
echo "SB-FT reviewer alignment metrics"
echo "============================================"
echo "COHA:   ${COHA_CKPT}"
echo "HBR:    ${HBR_CKPT}"
echo "ILR:    ${ILR_CKPT}"
echo "Output: ${OUT_JSON}"
echo "============================================"

"${PYTHON}" "${ALIGN_SCRIPT}" \
  --mode full_finetune \
  --coha_checkpoint "${COHA_CKPT}" \
  --hbr_checkpoint "${HBR_CKPT}" \
  --ilr_checkpoint "${ILR_CKPT}" \
  --output "${OUT_JSON}" 2>&1 | tee "${OUT_LOG}"

"${PYTHON}" - <<PY
from pathlib import Path
import csv
import json

out_json = Path('${OUT_JSON}')
out_csv = Path('${OUT_CSV}')
data = json.loads(out_json.read_text())['full_finetune']
rows = []
for rec in data['pairwise_metrics']:
    rows.append({
        'scope': rec['corpus_pair'],
        'same_jsd': rec['same_index_traj_jsd'],
        'hungarian_jsd': rec['hungarian_matched_traj_jsd'],
        'margin': rec['traj_margin'],
        'r1_pct': 100.0 * rec['traj_retrieval_at_1'],
        'nearest_wrong_jsd': rec['nearest_wrong_traj_jsd'],
        'timestamp_match_change_rate': rec.get('timestamp_match_change_rate', ''),
    })
avg = data['overall_averages']
rows.append({
    'scope': 'AVG',
    'same_jsd': avg['same_index_traj_jsd'],
    'hungarian_jsd': avg['hungarian_matched_traj_jsd'],
    'margin': avg['traj_margin'],
    'r1_pct': 100.0 * avg['traj_retrieval_at_1'],
    'nearest_wrong_jsd': avg['nearest_wrong_traj_jsd'],
    'timestamp_match_change_rate': avg.get('timestamp_match_change_rate', ''),
})
with out_csv.open('w', newline='') as f:
    fieldnames = ['scope', 'same_jsd', 'hungarian_jsd', 'margin', 'r1_pct', 'nearest_wrong_jsd', 'timestamp_match_change_rate']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f'Saved summary CSV: {out_csv}')
PY
