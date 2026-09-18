#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_FLYPPY_V7_EXPERIMENT:-artifacts/experiments/flyppy-v7-envv7-frontier}"

VF_EXPERIMENT_DIR="$EXPERIMENT" \
VF_VIEWER_ENVIRONMENT_VERSION=v7 \
bash scripts/dev/view_flyppy.sh "$EXPERIMENT"
