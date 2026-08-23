#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="/user/rl3403/.conda/envs/nlp_kogut/bin/python"
ALIGN_SCRIPT="${SCRIPT_DIR}/alignment_metrics_trajectory.py"
RUN_ROOT="${PROJECT_ROOT}/multiseed_runs/fixed_backbone_ablation"
AGG_DIR="${RUN_ROOT}/aggregate"
mkdir -p "${AGG_DIR}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(1 1013 2026)
fi

for SEED in ${SEEDS[@]}; do
  SEED_PADDED=$(printf "%04d" "$SEED")
  ROOT="${RUN_ROOT}/seed_${SEED_PADDED}"
  SBFT_ROOT="${ROOT}/checkpoints/sb_ft"
  METRICS_DIR="${ROOT}/metrics/sb_ft_alignment"
  mkdir -p "${METRICS_DIR}"

  COHA_CKPT="$(find "${SBFT_ROOT}/coha" -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1 || true)"
  HBR_CKPT="$(find "${SBFT_ROOT}/hbr" -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1 || true)"
  ILR_CKPT="$(find "${SBFT_ROOT}/ilr" -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1 || true)"

  if [[ -z "${COHA_CKPT}" || -z "${HBR_CKPT}" || -z "${ILR_CKPT}" ]]; then
    echo "[SKIP] seed=${SEED} missing SB-FT checkpoints under ${SBFT_ROOT}"
    continue
  fi

  OUT_JSON="${METRICS_DIR}/sbft_alignment_seed${SEED_PADDED}.json"
  OUT_LOG="${METRICS_DIR}/sbft_alignment_seed${SEED_PADDED}.log"

  echo "============================================"
  echo "SB-FT reviewer alignment seed=${SEED}"
  echo "============================================"
  echo "COHA: ${COHA_CKPT}"
  echo "HBR:  ${HBR_CKPT}"
  echo "ILR:  ${ILR_CKPT}"
  echo "Out:  ${OUT_JSON}"

  if [[ -f "${OUT_JSON}" ]]; then
    echo "[SKIP EVAL] existing ${OUT_JSON}"
  else
    "${PYTHON}" "${ALIGN_SCRIPT}" \
      --mode full_finetune \
      --coha_checkpoint "${COHA_CKPT}" \
      --hbr_checkpoint "${HBR_CKPT}" \
      --ilr_checkpoint "${ILR_CKPT}" \
      --output "${OUT_JSON}" 2>&1 | tee "${OUT_LOG}"
  fi
done

"${PYTHON}" - <<'PY'
from pathlib import Path
import csv
import json
import statistics

run_root = Path('/shared/share_hbr-ilr_nlp/shared_backbone_detm/multiseed_runs/fixed_backbone_ablation')
agg_dir = run_root / 'aggregate'
rows = []
for path in sorted(run_root.glob('seed_*/metrics/sb_ft_alignment/sbft_alignment_seed*.json')):
    seed_name = path.parents[2].name.replace('seed_', '')
    seed = int(seed_name)
    data = json.loads(path.read_text())['full_finetune']
    avg = data['overall_averages']
    rows.append({
        'seed': seed,
        'condition': 'SB-FT',
        'same_index_traj_jsd': avg['same_index_traj_jsd'],
        'hungarian_matched_traj_jsd': avg['hungarian_matched_traj_jsd'],
        'nearest_wrong_traj_jsd': avg['nearest_wrong_traj_jsd'],
        'traj_margin': avg['traj_margin'],
        'traj_retrieval_at_1': avg['traj_retrieval_at_1'],
        'timestamp_match_change_rate': avg.get('timestamp_match_change_rate', ''),
        'alignment_json': str(path),
    })

long_out = agg_dir / 'table5_sbft_alignment_long.csv'
summary_out = agg_dir / 'table5_sbft_alignment_mean_sd.csv'
fieldnames = ['seed', 'condition', 'same_index_traj_jsd', 'hungarian_matched_traj_jsd', 'nearest_wrong_traj_jsd', 'traj_margin', 'traj_retrieval_at_1', 'timestamp_match_change_rate', 'alignment_json']
with long_out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

metrics = ['same_index_traj_jsd', 'hungarian_matched_traj_jsd', 'traj_margin', 'traj_retrieval_at_1', 'nearest_wrong_traj_jsd', 'timestamp_match_change_rate']
summary = {'condition': 'SB-FT', 'n': len(rows), 'seeds': ';'.join(str(r['seed']) for r in sorted(rows, key=lambda x: x['seed']))}
for metric in metrics:
    vals = [float(r[metric]) for r in rows if r[metric] != '']
    summary[f'{metric}_mean'] = statistics.mean(vals) if vals else ''
    summary[f'{metric}_std'] = statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else ''
with summary_out.open('w', newline='') as f:
    fieldnames = list(summary.keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(summary)

print(f'Saved long CSV: {long_out}')
print(f'Saved summary CSV: {summary_out}')
if rows:
    print('SB-FT alignment mean±sd:')
    print(f"  Same JSD: {summary['same_index_traj_jsd_mean']:.4f} ± {summary['same_index_traj_jsd_std']:.4f}")
    print(f"  Hung JSD: {summary['hungarian_matched_traj_jsd_mean']:.4f} ± {summary['hungarian_matched_traj_jsd_std']:.4f}")
    print(f"  Margin:   {summary['traj_margin_mean']:.4f} ± {summary['traj_margin_std']:.4f}")
    print(f"  R@1:      {100*summary['traj_retrieval_at_1_mean']:.1f} ± {100*summary['traj_retrieval_at_1_std']:.1f}")
else:
    print('No SB-FT alignment rows found. Run run_fixed_backbone_sbft_seed_*.sh first.')
PY
