#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-7}"
exec "$SCRIPT_DIR/run_k_sensitivity_seed2019.sh" 30 "$GPU"
