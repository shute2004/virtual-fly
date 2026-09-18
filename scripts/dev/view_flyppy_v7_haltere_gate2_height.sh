#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export VF_FLYPPY_GATE2_EXPERIMENT="${VF_FLYPPY_GATE2_EXPERIMENT:-artifacts/experiments/flyppy-v7h-gate2-height}"
exec bash scripts/dev/view_flyppy_v7_gate2_height.sh
