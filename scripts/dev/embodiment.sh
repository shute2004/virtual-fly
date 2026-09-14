#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
RETINOTOPIC_MAP="$SNAPSHOT/retinotopic-vision-v1.json"
WING_MOTOR_MAP="$SNAPSHOT/wing-motor-neurons-v0.json"

run() {
  printf '\n== %s ==\n' "$1"
  shift
  "$@"
  status=$?
  if [ "$status" -ne 0 ]; then
    printf '\nFAILED (exit %s)\n' "$status" >&2
    exit "$status"
  fi
}

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}

printf 'repository: %s\n' "$ROOT"
printf 'snapshot:   %s\n' "$SNAPSHOT"
printf 'motor path: individual released wing MN -> WingMusclePeriphery -> FlyBodyMuscleAdapter\n'

run "Embodiment Python dependencies" uv sync
run "Packaged curriculum import" uv run python -c 'import virtual_fly.training.curriculum'
run "Curriculum unit tests" uv run python -m unittest discover -s tests -p 'test_*.py'
run "FlyBody asset prefetch" uv run python scripts/dev/prefetch_flybody_assets.py
run "Python syntax preflight" uv run python -m compileall -q \
  src/virtual_fly tests scripts/embodiment scripts/data scripts/analysis scripts/dev/prefetch_flybody_assets.py
run "MaleCNS embodiment group discovery" uv run python scripts/data/make_embodiment_groups.py --snapshot "$SNAPSHOT" --output "$GROUPS"
run "MaleCNS retinotopic R1-R6 map" uv run python scripts/data/prepare_retinotopic_vision.py --snapshot "$SNAPSHOT" --output "$RETINOTOPIC_MAP" --download
run "Individual wing motor-neuron map" uv run python scripts/data/prepare_wing_motor_map.py --snapshot "$SNAPSHOT" --output "$WING_MOTOR_MAP"
run "Rust compile check" cargo check --workspace
run "Rust unit tests" cargo test --workspace
run "Flyppy course geometry" uv run python scripts/embodiment/flyppy_course.py
run "Flyppy eye rendering" uv run python scripts/embodiment/flyppy_vision_smoke.py
run "MaleCNS retinotopic photoreceptor input" uv run python scripts/embodiment/malecns_retina_smoke.py --snapshot "$SNAPSHOT" --groups "$GROUPS" --mapping "$RETINOTOPIC_MAP"
run "Individual wing MN peripheral boundary" uv run python scripts/embodiment/wing_muscle_boundary_smoke.py --motor-map "$WING_MOTOR_MAP"
run "Individual-MN virtual-muscle flight envelope" uv run python scripts/embodiment/flybody_muscle_flight_envelope.py

printf '\nembodiment=PASS\n'
printf 'groups: %s\n' "$GROUPS"
printf 'retinotopy: %s\n' "$RETINOTOPIC_MAP"
printf 'wing motor map: %s\n' "$WING_MOTOR_MAP"
printf 'artifacts: %s/artifacts/embodiment\n' "$ROOT"
printf 'legacy DNg02 diagnostics are intentionally excluded from the standard preflight.\n'
