#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
EXPERIMENT="${VF_FLYPPY_GATE2_EXPERIMENT:-artifacts/experiments/flyppy-v7-gate2-height-v2}"
VF_EXPERIMENT_DIR="$EXPERIMENT" \
VF_VIEWER_ENVIRONMENT_VERSION=v7 \
bash scripts/dev/view_flyppy.sh "$EXPERIMENT"
