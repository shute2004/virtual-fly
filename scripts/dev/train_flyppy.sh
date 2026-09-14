#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
RETINOTOPIC_MAP="$SNAPSHOT/retinotopic-vision-v1.json"
WING_MOTOR_MAP="$SNAPSHOT/wing-motor-neurons-v0.json"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}

# A previous shell or diagnostic must never leak a trajectory-conditioned body
# calibration into learning. The virtual-muscle seam is fixed independently of
# Flyppy outcomes; crashes remain genuine neural/body outcomes and teaching events.
unset VF_MOTOR_CALIBRATION || true

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

# Body-only invariant: before involving the CNS, the fixed bootstrap
# muscle->hinge seam must possess at least one viable bilateral power operating
# point and the legacy FlyBody reference must remain viable. This diagnostic does
# not inspect any neural trajectory, reward, gate outcome, or learned state.
printf '\n== Independent virtual-muscle flight envelope ==\n'
uv run python scripts/embodiment/flybody_muscle_flight_envelope.py

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
