#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
RETINOTOPIC_MAP="$SNAPSHOT/retinotopic-vision-v1.json"
WING_MOTOR_MAP="$SNAPSHOT/wing-motor-neurons-v0.json"
PRIOR_TRAJECTORY="$ROOT/artifacts/experiments/flyppy-v1/trajectory.jsonl"
MOTOR_CALIBRATION="$ROOT/artifacts/embodiment/flybody-muscle-calibration-v1.json"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}

uv sync

printf '\n== Neural bridge compile preflight ==\n'
cargo check -p vf-runner --bin neural_bridge

printf '\n== Experimental neural boundary groups ==\n'
uv run python scripts/data/make_embodiment_groups.py \
  --snapshot "$SNAPSHOT" \
  --output "$GROUPS"

printf '\n== Individual wing motor-neuron map ==\n'
uv run python scripts/data/prepare_wing_motor_map.py \
  --snapshot "$SNAPSHOT" \
  --output "$WING_MOTOR_MAP"

printf '\n== Retinotopic R1-R6 map ==\n'
uv run python scripts/data/prepare_retinotopic_vision.py \
  --snapshot "$SNAPSHOT" \
  --output "$RETINOTOPIC_MAP" \
  --download

printf '\n== MaleCNS retinal current smoke test ==\n'
uv run python scripts/embodiment/malecns_retina_smoke.py \
  --snapshot "$SNAPSHOT" \
  --groups "$GROUPS" \
  --mapping "$RETINOTOPIC_MAP"

printf '\n== Wing neuromuscular boundary smoke test ==\n'
uv run python scripts/embodiment/wing_muscle_boundary_smoke.py \
  --motor-map "$WING_MOTOR_MAP"

# If a stride-1 trajectory from the immediately preceding run exists, use it only
# to calibrate the static muscle->hinge seam. The search is body-only and runs
# before the trajectory is replaced by the new episode. Selected parameters are
# frozen for the entire subsequent process through VF_MOTOR_CALIBRATION.
if [ -f "$PRIOR_TRAJECTORY" ]; then
  printf '\n== Robust motor-interface calibration ==\n'
  uv run python scripts/embodiment/calibrate_motor_interface.py \
    --trajectory "$PRIOR_TRAJECTORY" \
    --output "$MOTOR_CALIBRATION"
  export VF_MOTOR_CALIBRATION="$MOTOR_CALIBRATION"

  printf '\n== Calibrated prior-motor replay preflight ==\n'
  uv run python scripts/embodiment/calibrated_motor_replay_smoke.py \
    --trajectory "$PRIOR_TRAJECTORY"
fi

PYTHON_LAUNCHER=(uv run python)
if [ "$(uname -s)" = "Darwin" ]; then
  for arg in "$@"; do
    if [ "$arg" = "--render" ]; then
      # MuJoCo's passive Cocoa viewer must run under mjpython on macOS.
      PYTHON_LAUNCHER=(uv run mjpython)
      break
    fi
  done
fi

printf '\n== Flyppy learning: individual wing MN -> muscle -> torque ==\n'
exec "${PYTHON_LAUNCHER[@]}" scripts/embodiment/flyppy_closed_loop.py \
  --snapshot "$SNAPSHOT" \
  --groups "$GROUPS" \
  --retinotopic-map "$RETINOTOPIC_MAP" \
  --wing-motor-map "$WING_MOTOR_MAP" \
  "$@"
