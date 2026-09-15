#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
NEURAL_CALIBRATION="$ROOT/artifacts/embodiment/neural-runtime-calibration-v1.json"
EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v3}"
CHECKPOINT="$EXPERIMENT/checkpoint"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }

[ -f "$SNAPSHOT/manifest.json" ] || { echo "missing snapshot: $SNAPSHOT" >&2; exit 2; }
[ -f "$GROUPS" ] || { echo "missing groups: $GROUPS" >&2; exit 2; }
[ -f "$NEURAL_CALIBRATION" ] || { echo "missing neural calibration: $NEURAL_CALIBRATION" >&2; exit 2; }
[ -f "$CHECKPOINT/manifest.json" ] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }

VF_NEURAL_SYNAPSE_SCALE="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["synapse_scale"])' "$NEURAL_CALIBRATION")"
export VF_NEURAL_SYNAPSE_SCALE

printf 'population_smoke synapse_scale=%s checkpoint=%s\n' "$VF_NEURAL_SYNAPSE_SCALE" "$CHECKPOINT"

cargo test -q -p vf-neural transaction::tests
cargo check -q -p vf-runner --bin population_neural_bridge
uv run python -m py_compile \
  scripts/embodiment/population_neural_bridge_client.py \
  scripts/embodiment/train_flyppy_population.py \
  scripts/analysis/smoke_population_neural_bridge.py

uv run python scripts/analysis/smoke_population_neural_bridge.py \
  --snapshot "$SNAPSHOT" \
  --groups "$GROUPS" \
  --checkpoint "$CHECKPOINT"
