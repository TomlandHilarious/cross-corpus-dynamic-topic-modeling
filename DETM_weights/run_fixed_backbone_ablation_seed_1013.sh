#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-7}"
exec "$SCRIPT_DIR/run_fixed_backbone_ablation_seed.sh" 1013 "$GPU"
