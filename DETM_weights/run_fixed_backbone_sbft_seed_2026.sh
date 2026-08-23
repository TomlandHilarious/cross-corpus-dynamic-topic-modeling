#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_fixed_backbone_sbft_seed.sh" 2026 "${1:-6}"
