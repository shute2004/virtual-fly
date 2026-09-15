#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
RETINOTOPIC_MAP="$SNAPSHOT/retinotopic-vision-v1.json"
WING_MOTOR_MAP="$SNAPSHOT/wing-motor-neurons-v0.json"
BODY_MOTOR_MAP="$SNAPSHOT/body-motor-neurons-v0.json"
NEURAL_CALIBRATION="$ROOT/artifacts/embodiment/neural-runtime-calibration-v1.json"
VIEWER_GRAPH="$ROOT/artifacts/embodiment/neural-viewer-graph-v1.json"
EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v3}"
POPULATION="${VF_FLYPPY_POPULATION:-2}"
EPISODES="${VF_FLYPPY_EPISODES:-24}"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}

if [ "${VF_SKIP_V3_FLIGHT_PREFLIGHT:-0}" != "1" ]; then
  bash scripts/dev/preflight_flyppy_v3.sh
fi

if [ ! -f "$GROUPS" ]; then
  uv run python scripts/data/make_embodiment_groups.py \
    --snapshot "$SNAPSHOT" \
    --output "$GROUPS"
fi
if [ ! -f "$WING_MOTOR_MAP" ]; then
  uv run python scripts/data/prepare_wing_motor_map.py \
    --snapshot "$SNAPSHOT" \
    --output "$WING_MOTOR_MAP"
fi
if [ ! -f "$BODY_MOTOR_MAP" ]; then
  uv run python scripts/data/prepare_body_motor_map.py \
    --snapshot "$SNAPSHOT" \
    --output "$BODY_MOTOR_MAP"
fi
if [ ! -f "$RETINOTOPIC_MAP" ]; then
  uv run python scripts/data/prepare_retinotopic_vision.py \
    --snapshot "$SNAPSHOT" \
    --output "$RETINOTOPIC_MAP" \
    --download
fi

EXPECTED_VIEWER_GRAPH_SCHEMA=3
VIEWER_SCHEMA=0
if [ -f "$VIEWER_GRAPH" ]; then
  VIEWER_SCHEMA="$(uv run python -c 'import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get("schema_version", 0)))
except Exception:
    print(0)' "$VIEWER_GRAPH")"
fi
if [ "$VIEWER_SCHEMA" != "$EXPECTED_VIEWER_GRAPH_SCHEMA" ]; then
  uv run python scripts/data/prepare_neural_viewer_graph.py \
    --snapshot "$SNAPSHOT" \
    --groups "$GROUPS" \
    --motor-map "$WING_MOTOR_MAP" \
    --output "$VIEWER_GRAPH"
fi

EXPECTED_NEURAL_CALIBRATION_SCHEMA=2
CALIBRATION_SCHEMA=0
if [ -f "$NEURAL_CALIBRATION" ]; then
  CALIBRATION_SCHEMA="$(uv run python -c 'import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get("schema_version", 0)))
except Exception:
    print(0)' "$NEURAL_CALIBRATION")"
fi
if [ "$CALIBRATION_SCHEMA" != "$EXPECTED_NEURAL_CALIBRATION_SCHEMA" ]; then
  cargo check -q -p vf-runner --bin neural_stability_probe
  uv run python scripts/embodiment/calibrate_neural_runtime.py \
    --snapshot "$SNAPSHOT" \
    --mapping "$RETINOTOPIC_MAP" \
    --output "$NEURAL_CALIBRATION"
fi

VF_NEURAL_SYNAPSE_SCALE="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["synapse_scale"])' "$NEURAL_CALIBRATION")"
export VF_NEURAL_SYNAPSE_SCALE
unset VF_COURSE_START_GATE || true

cargo check -q -p vf-runner --bin population_neural_bridge
uv run python -m py_compile \
  scripts/embodiment/population_neural_bridge_client.py \
  scripts/embodiment/train_flyppy_population.py

TELEMETRY_ARGS=()
if [ "${VF_POPULATION_TELEMETRY:-0}" = "1" ]; then
  TELEMETRY_ARGS+=(--telemetry)
fi

printf 'shared_weight_population=%s episodes=%s experiment=%s synapse_scale=%s\n' \
  "$POPULATION" "$EPISODES" "$EXPERIMENT" "$VF_NEURAL_SYNAPSE_SCALE"

uv run python scripts/embodiment/train_flyppy_population.py \
  --episodes "$EPISODES" \
  --population "$POPULATION" \
  --snapshot "$SNAPSHOT" \
  --groups "$GROUPS" \
  --retinotopic-map "$RETINOTOPIC_MAP" \
  --wing-motor-map "$WING_MOTOR_MAP" \
  --body-motor-map "$BODY_MOTOR_MAP" \
  --viewer-graph "$VIEWER_GRAPH" \
  --output-dir "$EXPERIMENT" \
  "${TELEMETRY_ARGS[@]}" \
  "$@"

uv run python scripts/analysis/export_flyppy_report.py --summary "$EXPERIMENT/summary.json"
uv run python scripts/analysis/append_flyppy_run_history.py --summary "$EXPERIMENT/summary.json"

printf 'report_md=reports/flyppy/latest.md\n'
printf 'report_csv=reports/flyppy/latest.csv\n'
printf 'run_history=reports/flyppy/history.csv\n'
