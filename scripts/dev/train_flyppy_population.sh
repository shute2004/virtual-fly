#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
RETINOTOPIC_MAP="$SNAPSHOT/retinotopic-vision-v1.json"
HALTERE_SENSORY_MAP="${VF_HALTERE_SENSORY_MAP:-$SNAPSHOT/haltere-campaniform-sensory-v1.json}"
HALTERE_SENSORY_KIND="${VF_HALTERE_SENSORY_KIND:-male-cns-haltere-campaniform-afferents}"
WING_MOTOR_MAP="$SNAPSHOT/wing-motor-neurons-v0.json"
BODY_MOTOR_MAP="$SNAPSHOT/body-motor-neurons-v0.json"
NEURAL_CALIBRATION="$ROOT/artifacts/embodiment/neural-runtime-calibration-v1.json"
VIEWER_GRAPH="$ROOT/artifacts/embodiment/neural-viewer-graph-v1.json"
EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v3}"
ENVIRONMENT_VERSION="${VF_FLYPPY_ENVIRONMENT_VERSION:-v3}"
FLIGHT_BODY_VERSION="${VF_FLYBODY_VERSION:-v3}"
HALTERE_CURRENT_GAIN="${VF_HALTERE_CURRENT_GAIN:-0.0}"
HALTERE_TRANSDUCTION="${VF_HALTERE_TRANSDUCTION:-angular-acceleration-v1}"
case "$ENVIRONMENT_VERSION" in
  v3|v4|v5|v6|v7) ;;
  *)
    echo "VF_FLYPPY_ENVIRONMENT_VERSION must be v3, v4, v5, v6, or v7" >&2
    exit 2
    ;;
esac
case "$FLIGHT_BODY_VERSION" in
  v3|v4|v5|v6|v7|v8) ;;
  *)
    echo "VF_FLYBODY_VERSION must be v3, v4, v5, v6, v7, or v8" >&2
    exit 2
    ;;
esac
# Logical population and body-process count are independent.  On the current M1
# reference machine, long packed-runtime calibration over N=4/8/12/16 selected
# N=4 for maximum aggregate control-step throughput.  Keep the calibrated value
# overrideable so other hardware can supply its own measured optimum.
POPULATION_REQUEST="${VF_FLYPPY_POPULATION:-auto}"
AUTO_POPULATION="${VF_FLYPPY_AUTO_POPULATION:-4}"
EPISODES="${VF_FLYPPY_EPISODES:-24}"
CHECKPOINT_EVERY="${VF_FLYPPY_CHECKPOINT_EVERY:-32}"
FRESH="${VF_FLYPPY_FRESH:-0}"
WRITE_LATEST_REPORTS="${VF_FLYPPY_WRITE_LATEST_REPORTS:-1}"

# Production curriculum is batch-based.  The older adaptive policy changed
# difficulty after every asynchronously completed episode, so short failing
# slots could move the spawn condition before slower successful slots finished.
# Boundary-band freezes one 24-attempt condition set, updates only at the batch
# boundary, and equal-weights per-slot/course-seed success rates for the update.
CURRICULUM_MODE="${VF_FLYPPY_CURRICULUM_MODE:-boundary-band}"
BOUNDARY_BATCH_SIZE="${VF_FLYPPY_BOUNDARY_BATCH_SIZE:-24}"
LAUNCH_MODE="${VF_FLYPPY_LAUNCH_MODE:-async}"
case "$CURRICULUM_MODE" in
  adaptive|boundary-band|gate2-height)
    ;;
  *)
    echo "VF_FLYPPY_CURRICULUM_MODE must be adaptive, boundary-band, or gate2-height" >&2
    exit 2
    ;;
esac
MIN_BOUNDARY_BATCH_SIZE=5
if [ "$CURRICULUM_MODE" = "gate2-height" ] && [ "${VF_FLYPPY_GATE2_ACQUISITION_ONLY:-0}" = "1" ]; then
  MIN_BOUNDARY_BATCH_SIZE=1
fi
[[ "$BOUNDARY_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "VF_FLYPPY_BOUNDARY_BATCH_SIZE must be a positive integer" >&2
  exit 2
}
if [ "$BOUNDARY_BATCH_SIZE" -lt "$MIN_BOUNDARY_BATCH_SIZE" ]; then
  echo "VF_FLYPPY_BOUNDARY_BATCH_SIZE must be >= $MIN_BOUNDARY_BATCH_SIZE for this curriculum mode" >&2
  exit 2
fi
case "$LAUNCH_MODE" in
  async|wave)
    ;;
  *)
    echo "VF_FLYPPY_LAUNCH_MODE must be async or wave" >&2
    exit 2
    ;;
esac

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

source "$ROOT/scripts/dev/lib/runtime.sh"
vf_resolve_cargo
vf_resolve_python "$ROOT"
CARGO_RUNNER=("${VF_CARGO[@]}")
PYTHON_RUNNER=("${VF_PYTHON[@]}")
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || {
  echo "VF_FLYPPY_EPISODES must be a positive integer" >&2
  exit 2
}
[[ "$CHECKPOINT_EVERY" =~ ^[1-9][0-9]*$ ]] || {
  echo "VF_FLYPPY_CHECKPOINT_EVERY must be a positive integer" >&2
  exit 2
}
case "$FRESH" in
  0|1) ;;
  *)
    echo "VF_FLYPPY_FRESH must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$WRITE_LATEST_REPORTS" in
  0|1) ;;
  *)
    echo "VF_FLYPPY_WRITE_LATEST_REPORTS must be 0 or 1" >&2
    exit 2
    ;;
esac
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

"${PYTHON_RUNNER[@]}" scripts/dev/prepare_flyppy_population_inputs.py \
  --snapshot "$SNAPSHOT" \
  --groups "$GROUPS" \
  --retinotopic-map "$RETINOTOPIC_MAP" \
  --haltere-sensory-map "$HALTERE_SENSORY_MAP" \
  --haltere-sensory-kind "$HALTERE_SENSORY_KIND" \
  --wing-motor-map "$WING_MOTOR_MAP" \
  --body-motor-map "$BODY_MOTOR_MAP"

NEUTRAL_PATTERN="${VF_NEUTRAL_TRIM_PATTERN:-$ROOT/artifacts/derived/wing-pattern-neutral-trim-v1.npy}"
NEUTRAL_METADATA="${VF_NEUTRAL_TRIM_METADATA:-${NEUTRAL_PATTERN%.npy}.json}"
if [ "$FLIGHT_BODY_VERSION" = "v7" ] || [ "$FLIGHT_BODY_VERSION" = "v8" ]; then
  if [ ! -e "$NEUTRAL_PATTERN" ] && [ ! -e "$NEUTRAL_METADATA" ]; then
    "${PYTHON_RUNNER[@]}" scripts/data/prepare_flybody_neutral_trim.py \
      --output-pattern "$NEUTRAL_PATTERN" \
      --output-metadata "$NEUTRAL_METADATA"
  fi
  "${PYTHON_RUNNER[@]}" - "$NEUTRAL_PATTERN" "$NEUTRAL_METADATA" <<'PY'
from pathlib import Path
import sys
from virtual_fly.reproducibility import validate_neutral_trim
validate_neutral_trim(Path(sys.argv[1]), Path(sys.argv[2]))
print("neutral_trim_provenance=PASS")
PY
fi

EXPECTED_VIEWER_GRAPH_SCHEMA=4
VIEWER_SCHEMA=0
if [ -f "$VIEWER_GRAPH" ]; then
  VIEWER_SCHEMA="$("${PYTHON_RUNNER[@]}" -c 'import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get("schema_version", 0)))
except Exception:
    print(0)' "$VIEWER_GRAPH")"
fi
if [ "$VIEWER_SCHEMA" != "$EXPECTED_VIEWER_GRAPH_SCHEMA" ]; then
  "${PYTHON_RUNNER[@]}" scripts/data/prepare_neural_viewer_graph.py \
    --snapshot "$SNAPSHOT" \
    --groups "$GROUPS" \
    --motor-map "$WING_MOTOR_MAP" \
    --output "$VIEWER_GRAPH"
fi
"${PYTHON_RUNNER[@]}" - "$VIEWER_GRAPH" "$SNAPSHOT" <<'PY'
from pathlib import Path
import sys
from virtual_fly.reproducibility import validate_derived_artifact
validate_derived_artifact(Path(sys.argv[1]), Path(sys.argv[2]))
print("viewer_graph_provenance=PASS")
PY

EXPECTED_NEURAL_CALIBRATION_SCHEMA=2
CALIBRATION_SCHEMA=0
if [ -f "$NEURAL_CALIBRATION" ]; then
  CALIBRATION_SCHEMA="$("${PYTHON_RUNNER[@]}" -c 'import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get("schema_version", 0)))
except Exception:
    print(0)' "$NEURAL_CALIBRATION")"
fi
if [ "$CALIBRATION_SCHEMA" != "$EXPECTED_NEURAL_CALIBRATION_SCHEMA" ]; then
  "${CARGO_RUNNER[@]}" check -q -p vf-runner --bin neural_stability_probe
  "${PYTHON_RUNNER[@]}" scripts/embodiment/calibrate_neural_runtime.py \
    --snapshot "$SNAPSHOT" \
    --mapping "$RETINOTOPIC_MAP" \
    --output "$NEURAL_CALIBRATION"
fi
"${PYTHON_RUNNER[@]}" - "$NEURAL_CALIBRATION" "$SNAPSHOT" <<'PY'
from pathlib import Path
import sys
from virtual_fly.reproducibility import validate_derived_artifact
validate_derived_artifact(Path(sys.argv[1]), Path(sys.argv[2]))
print("neural_calibration_provenance=PASS")
PY

VF_NEURAL_SYNAPSE_SCALE="$("${PYTHON_RUNNER[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["synapse_scale"])' "$NEURAL_CALIBRATION")"
export VF_NEURAL_SYNAPSE_SCALE
export VF_NEURAL_CALIBRATION_PATH="$NEURAL_CALIBRATION"
unset VF_COURSE_START_GATE || true

"${CARGO_RUNNER[@]}" test -q -p vf-neural transaction::tests
"${CARGO_RUNNER[@]}" check -q -p vf-runner --bin population_neural_bridge
"${PYTHON_RUNNER[@]}" -m py_compile \
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
printf 'shared_weight_population=%s population_request=%s episodes=%s checkpoint_every=%s fresh=%s experiment=%s environment=%s flight_body=%s haltere_gain=%s synapse_scale=%s body_runtime=packed body_processes=%s curriculum=%s boundary_batch=%s launch_mode=%s vision=%s rays_per_ommatidium=%s framebuffer=%s\n' \
  "$POPULATION" "$POPULATION_REQUEST" "$EPISODES" "$CHECKPOINT_EVERY" "$FRESH" "$EXPERIMENT" "$ENVIRONMENT_VERSION" "$FLIGHT_BODY_VERSION" "$HALTERE_CURRENT_GAIN" "$VF_NEURAL_SYNAPSE_SCALE" "$BODY_PROCESS_REQUEST" "$CURRICULUM_MODE" "$BOUNDARY_BATCH_SIZE" "$LAUNCH_MODE" "$VISION_MODE" "$OMMATIDIA_RAYS" "$([ "$VISION_MODE" = "raster" ] && printf true || printf false)"

TRAIN_ARGS=(
  "${PYTHON_RUNNER[@]}" scripts/embodiment/train_flyppy_population_packed.py
  --episodes "$EPISODES"
  --population "$POPULATION"
  --snapshot "$SNAPSHOT"
  --groups "$GROUPS"
  --retinotopic-map "$RETINOTOPIC_MAP"
  --wing-motor-map "$WING_MOTOR_MAP"
  --body-motor-map "$BODY_MOTOR_MAP"
  --viewer-graph "$VIEWER_GRAPH"
  --output-dir "$EXPERIMENT"
  --environment-version "$ENVIRONMENT_VERSION"
  --flight-body-version "$FLIGHT_BODY_VERSION"
  --neutral-trim-pattern "$NEUTRAL_PATTERN"
  --haltere-sensory-map "$HALTERE_SENSORY_MAP"
  --haltere-sensory-kind "$HALTERE_SENSORY_KIND"
  --haltere-current-gain "$HALTERE_CURRENT_GAIN"
  --haltere-transduction "$HALTERE_TRANSDUCTION"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --curriculum-mode "$CURRICULUM_MODE"
  --boundary-batch-size "$BOUNDARY_BATCH_SIZE"
  --launch-mode "$LAUNCH_MODE"
)
if [ "$FRESH" = "1" ]; then
  TRAIN_ARGS+=(--fresh)
fi
if [ "${VF_POPULATION_TELEMETRY:-0}" = "1" ]; then
  TRAIN_ARGS+=(--telemetry)
fi
if [ "${VF_FLYPPY_GATE2_ACQUISITION_ONLY:-0}" = "1" ]; then
  TRAIN_ARGS+=(--gate2-height-acquisition-only)
fi
TRAIN_ARGS+=("$@")
"${TRAIN_ARGS[@]}"

if [ "$WRITE_LATEST_REPORTS" = "1" ]; then
  "${PYTHON_RUNNER[@]}" scripts/analysis/export_flyppy_report.py --summary "$EXPERIMENT/summary.json"
  "${PYTHON_RUNNER[@]}" scripts/analysis/export_flyppy_population_report.py --summary "$EXPERIMENT/summary.json"
  printf 'report_md=reports/flyppy/latest.md\n'
  printf 'population_report=reports/flyppy/population_latest.md\n'
  printf 'report_csv=reports/flyppy/latest.csv\n'
fi
"${PYTHON_RUNNER[@]}" scripts/analysis/append_flyppy_run_history.py --summary "$EXPERIMENT/summary.json"
printf 'run_history=reports/flyppy/history.csv\n'
