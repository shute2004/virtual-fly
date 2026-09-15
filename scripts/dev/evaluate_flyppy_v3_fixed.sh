#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
PRODUCTION="${VF_EXPERIMENT_DIR:-$ROOT/artifacts/experiments/flyppy-v3}"
CALIBRATION="${VF_NEURAL_CALIBRATION:-$ROOT/artifacts/embodiment/neural-runtime-calibration-v1.json}"
POPULATION="${VF_FLYPPY_EVAL_POPULATION:-4}"
SEED_START="${VF_FLYPPY_EVAL_SEED_START:-0}"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }

[ -f "$PRODUCTION/checkpoint/manifest.json" ] || {
  echo "Flyppy production checkpoint not found at $PRODUCTION/checkpoint" >&2
  exit 2
}
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT" >&2
  exit 2
}
[ -f "$CALIBRATION" ] || {
  echo "neural runtime calibration not found at $CALIBRATION" >&2
  exit 2
}

[[ "$POPULATION" =~ ^[1-9][0-9]*$ ]] || {
  echo "VF_FLYPPY_EVAL_POPULATION must be a positive integer" >&2
  exit 2
}
if [ "$POPULATION" -gt 32 ]; then
  echo "VF_FLYPPY_EVAL_POPULATION=$POPULATION exceeds the GPU 32-slot limit" >&2
  exit 2
fi

cargo check -q -p vf-runner --bin population_neural_bridge
uv run python -m py_compile \
  src/virtual_fly/training/fixed_evaluation.py \
  scripts/analysis/evaluate_flyppy_fixed.py
uv run python tests/test_fixed_evaluation.py

uv run python scripts/analysis/evaluate_flyppy_fixed.py \
  --production "$PRODUCTION" \
  --snapshot "$SNAPSHOT" \
  --calibration "$CALIBRATION" \
  --population "$POPULATION" \
  --seed-start "$SEED_START" \
  "$@"
