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
EXPECTED_NEURAL_CALIBRATION_SCHEMA=2
EXPECTED_VIEWER_GRAPH_SCHEMA=3

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}

FULL_PREFLIGHT="${VF_FULL_PREFLIGHT:-0}"
FLYPPY_ARGS=()
REPORT_SUMMARY="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v1}/summary.json"
EXPECT_OUTPUT_DIR=0
for arg in "$@"; do
  if [ "$arg" = "--full-preflight" ]; then
    FULL_PREFLIGHT=1
    continue
  fi
  if [ "$EXPECT_OUTPUT_DIR" = "1" ]; then
    REPORT_SUMMARY="$arg/summary.json"
    EXPECT_OUTPUT_DIR=0
  elif [ "$arg" = "--output-dir" ]; then
    EXPECT_OUTPUT_DIR=1
  elif [[ "$arg" == --output-dir=* ]]; then
    REPORT_SUMMARY="${arg#--output-dir=}/summary.json"
  fi
  FLYPPY_ARGS+=("$arg")
done
if [ "$EXPECT_OUTPUT_DIR" = "1" ]; then
  echo '--output-dir requires a path' >&2
  exit 2
fi

unset VF_MOTOR_CALIBRATION || true
unset VF_NEURAL_SYNAPSE_SCALE || true

uv sync >/dev/null
uv run python -c 'import virtual_fly.training.curriculum, virtual_fly.physics'

cargo check -q -p vf-runner --bin neural_bridge
uv run python -m py_compile \
  scripts/embodiment/train_flyppy_curriculum.py \
  scripts/embodiment/flybody_biophysics.py \
  scripts/embodiment/flybody_runtime.py \
  scripts/embodiment/flybody_muscle_adapter.py \
  scripts/embodiment/flybody_neuromuscular_adapter.py \
  scripts/embodiment/whole_body_periphery.py \
  scripts/embodiment/flyppy_course.py \
  scripts/embodiment/flyppy_world.py \
  scripts/embodiment/evaluate_flyppy.py \
  scripts/embodiment/live_telemetry.py \
  scripts/embodiment/live_body_viewer.py \
  scripts/data/prepare_body_motor_map.py \
  scripts/data/prepare_neural_viewer_graph.py \
  scripts/analysis/export_flyppy_report.py

if [ ! -f "$GROUPS" ]; then
  printf '\n== Generate neural boundary groups ==\n'
  uv run python scripts/data/make_embodiment_groups.py \
    --snapshot "$SNAPSHOT" \
    --output "$GROUPS"
fi

if [ ! -f "$WING_MOTOR_MAP" ]; then
  printf '\n== Generate individual wing motor-neuron map ==\n'
  uv run python scripts/data/prepare_wing_motor_map.py \
    --snapshot "$SNAPSHOT" \
    --output "$WING_MOTOR_MAP"
fi

if [ ! -f "$BODY_MOTOR_MAP" ]; then
  printf '\n== Generate grounded whole-body motor-neuron map ==\n'
  uv run python scripts/data/prepare_body_motor_map.py \
    --snapshot "$SNAPSHOT" \
    --output "$BODY_MOTOR_MAP"
fi

if [ ! -f "$RETINOTOPIC_MAP" ]; then
  printf '\n== Generate retinotopic R1-R6 map ==\n'
  uv run python scripts/data/prepare_retinotopic_vision.py \
    --snapshot "$SNAPSHOT" \
    --output "$RETINOTOPIC_MAP" \
    --download
fi

VIEWER_SCHEMA=0
if [ -f "$VIEWER_GRAPH" ]; then
  VIEWER_SCHEMA="$(uv run python -c 'import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get("schema_version", 0)))
except Exception:
    print(0)' "$VIEWER_GRAPH")"
fi
if [ "$VIEWER_SCHEMA" != "$EXPECTED_VIEWER_GRAPH_SCHEMA" ]; then
  printf '\n== Prepare cached neural observer graph ==\n'
  uv run python scripts/data/prepare_neural_viewer_graph.py \
    --snapshot "$SNAPSHOT" \
    --groups "$GROUPS" \
    --motor-map "$WING_MOTOR_MAP" \
    --output "$VIEWER_GRAPH"
fi

CALIBRATION_SCHEMA=0
if [ -f "$NEURAL_CALIBRATION" ]; then
  CALIBRATION_SCHEMA="$(uv run python -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("schema_version", 0)))' "$NEURAL_CALIBRATION" 2>/dev/null || echo 0)"
fi

if [ "$CALIBRATION_SCHEMA" != "$EXPECTED_NEURAL_CALIBRATION_SCHEMA" ] || [ "$FULL_PREFLIGHT" = "1" ]; then
  printf '\n== Whole-CNS pulse stability calibration ==\n'
  cargo check -q -p vf-runner --bin neural_stability_probe
  uv run python scripts/embodiment/calibrate_neural_runtime.py \
    --snapshot "$SNAPSHOT" \
    --mapping "$RETINOTOPIC_MAP" \
    --output "$NEURAL_CALIBRATION"
fi

VF_NEURAL_SYNAPSE_SCALE="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["synapse_scale"])' "$NEURAL_CALIBRATION")"
export VF_NEURAL_SYNAPSE_SCALE
printf 'using cached VF_NEURAL_SYNAPSE_SCALE=%s\n' "$VF_NEURAL_SYNAPSE_SCALE"

if [ "$FULL_PREFLIGHT" = "1" ]; then
  printf '\n== MaleCNS retinal current smoke test ==\n'
  uv run python scripts/embodiment/malecns_retina_smoke.py \
    --snapshot "$SNAPSHOT" \
    --groups "$GROUPS" \
    --mapping "$RETINOTOPIC_MAP"

  printf '\n== Wing neuromuscular boundary smoke test ==\n'
  uv run python scripts/embodiment/wing_muscle_boundary_smoke.py \
    --motor-map "$WING_MOTOR_MAP"

  printf '\n== Independent virtual-muscle flight envelope ==\n'
  uv run python scripts/embodiment/flybody_muscle_flight_envelope.py
fi

printf '\n== Flyppy persistent curriculum learning ==\n'
printf 'live_viewer=separate-process command="bash scripts/dev/view_learning.sh"\n'
uv run python scripts/embodiment/train_flyppy_curriculum.py \
  --snapshot "$SNAPSHOT" \
  --groups "$GROUPS" \
  --retinotopic-map "$RETINOTOPIC_MAP" \
  --wing-motor-map "$WING_MOTOR_MAP" \
  --body-motor-map "$BODY_MOTOR_MAP" \
  --viewer-graph "$VIEWER_GRAPH" \
  --telemetry-stride 10 \
  --body-telemetry-stride 1 \
  --checkpoint-every 32 \
  "${FLYPPY_ARGS[@]}"

printf '\n== Export compact Git report ==\n'
uv run python scripts/analysis/export_flyppy_report.py --summary "$REPORT_SUMMARY"
printf 'report_md=reports/flyppy/latest.md\n'
printf 'report_csv=reports/flyppy/latest.csv\n'
