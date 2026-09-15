#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="$ROOT/artifacts/malecns-v1.0"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
RETINOTOPIC_MAP="$SNAPSHOT/retinotopic-vision-v1.json"
WING_MOTOR_MAP="$SNAPSHOT/wing-motor-neurons-v0.json"
BODY_MOTOR_MAP="$SNAPSHOT/body-motor-neurons-v0.json"
NEURAL_CALIBRATION="$ROOT/artifacts/embodiment/neural-runtime-calibration-v1.json"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}

# Match the production trainer's prerequisite preparation. These files live under
# artifacts/ and may legitimately be absent after a clean checkout even when a
# training checkpoint still exists.
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

if [ ! -f "$NEURAL_CALIBRATION" ]; then
  printf '\n== Whole-CNS pulse stability calibration ==\n'
  cargo check -q -p vf-runner --bin neural_stability_probe
  uv run python scripts/embodiment/calibrate_neural_runtime.py \
    --snapshot "$SNAPSHOT" \
    --mapping "$RETINOTOPIC_MAP" \
    --output "$NEURAL_CALIBRATION"
fi

VF_NEURAL_SYNAPSE_SCALE="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["synapse_scale"])' "$NEURAL_CALIBRATION")"
export VF_NEURAL_SYNAPSE_SCALE
printf 'using production VF_NEURAL_SYNAPSE_SCALE=%s\n' "$VF_NEURAL_SYNAPSE_SCALE"

uv run python scripts/analysis/profile_flyppy_v3_p0.py \
  --warmup "${VF_P0_WARMUP_STEPS:-32}" \
  --steps "${VF_P0_MEASURE_STEPS:-256}" \
  --telemetry "${VF_P0_TELEMETRY:-on}" \
  "$@"

printf '\nP0 profile complete. Commit the requested report/log outputs from the A/B launcher.\n'
