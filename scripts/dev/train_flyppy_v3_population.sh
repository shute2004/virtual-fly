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
# Logical population and body-process count are independent.  On the current M1
# reference machine, long packed-runtime calibration over N=4/8/12/16 selected
# N=4 for maximum aggregate control-step throughput.  Keep the calibrated value
# overrideable so other hardware can supply its own measured optimum.
POPULATION_REQUEST="${VF_FLYPPY_POPULATION:-auto}"
AUTO_POPULATION="${VF_FLYPPY_AUTO_POPULATION:-4}"
EPISODES="${VF_FLYPPY_EPISODES:-24}"

# Production curriculum is batch-based.  The older adaptive policy changed
# difficulty after every asynchronously completed episode, so short failing
# slots could move the spawn condition before slower successful slots finished.
# Boundary-band freezes one 24-attempt condition set, updates only at the batch
# boundary, and equal-weights per-slot/course-seed success rates for the update.
CURRICULUM_MODE="${VF_FLYPPY_CURRICULUM_MODE:-boundary-band}"
BOUNDARY_BATCH_SIZE="${VF_FLYPPY_BOUNDARY_BATCH_SIZE:-24}"
case "$CURRICULUM_MODE" in
  adaptive|boundary-band)
    ;;
  *)
    echo "VF_FLYPPY_CURRICULUM_MODE must be adaptive or boundary-band" >&2
    exit 2
    ;;
esac
[[ "$BOUNDARY_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "VF_FLYPPY_BOUNDARY_BATCH_SIZE must be an integer >= 5" >&2
  exit 2
}
if [ "$BOUNDARY_BATCH_SIZE" -lt 5 ]; then
  echo "VF_FLYPPY_BOUNDARY_BATCH_SIZE must be >= 5" >&2
  exit 2
fi

# Production vision is image-free by default.  K=13 was selected against the
# uncompressed direct receptive-field integral: light r=0.9990 and MaleCNS
# current r=0.9904, while the end-to-end learning benchmark ran 1.606x faster
# than the raster path on the M1 reference machine.  Raster remains available as
# an explicit compatibility/debug oracle.
VISION_MODE="${VF_FLYPPY_VISION_MODE:-direct-ray}"
OMMATIDIA_RAYS="${VF_FLYPPY_OMMATIDIA_RAYS:-13}"
case "$VISION_MODE" in
  direct|ray|direct-ray)
    VISION_MODE="direct-ray"
    ;;
  raster|flygym|reference)
    VISION_MODE="raster"
    ;;
  *)
    echo "VF_FLYPPY_VISION_MODE must be raster or direct-ray" >&2
    exit 2
    ;;
esac
[[ "$OMMATIDIA_RAYS" =~ ^[1-9][0-9]*$ ]] || {
  echo "VF_FLYPPY_OMMATIDIA_RAYS must be a positive integer" >&2
  exit 2
}
export VF_FLYPPY_VISION_MODE="$VISION_MODE"
export VF_FLYPPY_OMMATIDIA_RAYS="$OMMATIDIA_RAYS"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || {
  echo "VF_FLYPPY_EPISODES must be a positive integer" >&2
  exit 2
}
[[ "$AUTO_POPULATION" =~ ^[1-9][0-9]*$ ]] || {
  echo "VF_FLYPPY_AUTO_POPULATION must be an integer in 1..32" >&2
  exit 2
}
if [ "$AUTO_POPULATION" -gt 32 ]; then
  echo "VF_FLYPPY_AUTO_POPULATION=$AUTO_POPULATION exceeds the GPU runtime 32-slot active-mask limit" >&2
  exit 2
fi

if [ "$POPULATION_REQUEST" = "auto" ]; then
  POPULATION="$AUTO_POPULATION"
else
  [[ "$POPULATION_REQUEST" =~ ^[1-9][0-9]*$ ]] || {
    echo "VF_FLYPPY_POPULATION must be 'auto' or an integer in 1..32" >&2
    exit 2
  }
  if [ "$POPULATION_REQUEST" -gt 32 ]; then
    echo "VF_FLYPPY_POPULATION=$POPULATION_REQUEST exceeds the GPU runtime 32-slot active-mask limit" >&2
    exit 2
  fi
  POPULATION="$POPULATION_REQUEST"
fi

if [ "${VF_SKIP_V3_FLIGHT_PREFLIGHT:-0}" != "1" ]; then
  bash scripts/dev/preflight_flyppy_v3.sh
fi

uv run python scripts/dev/prepare_flyppy_population_inputs.py \
  --snapshot "$SNAPSHOT" \
  --groups "$GROUPS" \
  --retinotopic-map "$RETINOTOPIC_MAP" \
  --wing-motor-map "$WING_MOTOR_MAP" \
  --body-motor-map "$BODY_MOTOR_MAP"

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

cargo test -q -p vf-neural transaction::tests
cargo check -q -p vf-runner --bin population_neural_bridge
uv run python -m py_compile \
  scripts/embodiment/population_neural_bridge_client.py \
  scripts/embodiment/train_flyppy_population.py \
  scripts/embodiment/train_flyppy_population_process.py \
  scripts/embodiment/direct_ommatidia_sensor.py \
  scripts/embodiment/direct_ommatidia_sensor_bodyexclude.py \
  scripts/embodiment/flyppy_packed_body_worker.py \
  scripts/embodiment/flyppy_packed_slot_adapter.py \
  scripts/embodiment/train_flyppy_population_packed.py \
  scripts/analysis/export_flyppy_population_report.py

BODY_PROCESS_REQUEST="${VF_FLYPPY_BODY_PROCESSES:-auto}"
printf 'shared_weight_population=%s population_request=%s episodes=%s experiment=%s synapse_scale=%s body_runtime=packed body_processes=%s curriculum=%s boundary_batch=%s vision=%s rays_per_ommatidium=%s framebuffer=%s\n' \
  "$POPULATION" "$POPULATION_REQUEST" "$EPISODES" "$EXPERIMENT" "$VF_NEURAL_SYNAPSE_SCALE" "$BODY_PROCESS_REQUEST" "$CURRICULUM_MODE" "$BOUNDARY_BATCH_SIZE" "$VISION_MODE" "$OMMATIDIA_RAYS" "$([ "$VISION_MODE" = "raster" ] && printf true || printf false)"

TRAIN_ARGS=(
  uv run python scripts/embodiment/train_flyppy_population_packed.py
  --episodes "$EPISODES"
  --population "$POPULATION"
  --snapshot "$SNAPSHOT"
  --groups "$GROUPS"
  --retinotopic-map "$RETINOTOPIC_MAP"
  --wing-motor-map "$WING_MOTOR_MAP"
  --body-motor-map "$BODY_MOTOR_MAP"
  --viewer-graph "$VIEWER_GRAPH"
  --output-dir "$EXPERIMENT"
  --curriculum-mode "$CURRICULUM_MODE"
  --boundary-batch-size "$BOUNDARY_BATCH_SIZE"
)
if [ "${VF_POPULATION_TELEMETRY:-0}" = "1" ]; then
  TRAIN_ARGS+=(--telemetry)
fi
TRAIN_ARGS+=("$@")
"${TRAIN_ARGS[@]}"

uv run python scripts/analysis/export_flyppy_report.py --summary "$EXPERIMENT/summary.json"
uv run python scripts/analysis/export_flyppy_population_report.py --summary "$EXPERIMENT/summary.json"
uv run python scripts/analysis/append_flyppy_run_history.py --summary "$EXPERIMENT/summary.json"

printf 'report_md=reports/flyppy/latest.md\n'
printf 'population_report=reports/flyppy/population_latest.md\n'
printf 'report_csv=reports/flyppy/latest.csv\n'
printf 'run_history=reports/flyppy/history.csv\n'
