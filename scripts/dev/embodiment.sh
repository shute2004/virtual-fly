#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"

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

prefetch_flybody_assets() {
  local attempt status
  for attempt in 1 2 3; do
    uv run python -c '
from flygym.compose.fly.flybody import FLYBODY_FULLSIZE_MESH_DIR
from flygym.utils.assets_lazy_loading import lazy_load_asset_dir
print(lazy_load_asset_dir(FLYBODY_FULLSIZE_MESH_DIR))
'
    status=$?
    if [ "$status" -eq 0 ]; then
      return 0
    fi
    if [ "$attempt" -eq 3 ]; then
      return "$status"
    fi
    printf 'FlyBody asset download failed (attempt %s/3); retrying with FlyGym loader...\n' "$attempt" >&2
    sleep "$attempt"
  done
}

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}

printf 'repository: %s\n' "$ROOT"
printf 'snapshot:   %s\n' "$SNAPSHOT"

run "Embodiment Python dependencies" uv sync
run "FlyBody asset prefetch" prefetch_flybody_assets
run "Python syntax preflight" uv run python -m compileall -q scripts/embodiment scripts/data scripts/analysis
run "MaleCNS embodiment group discovery" uv run python scripts/data/make_embodiment_groups.py --snapshot "$SNAPSHOT" --output "$GROUPS"
run "Rust compile check" cargo check --workspace
run "Rust unit tests" cargo test --workspace
run "Flyppy course geometry" uv run python scripts/embodiment/flyppy_course.py
run "Compound-eye vertical motion encoder" uv run python scripts/embodiment/visual_motion_encoder.py
run "FlyBody restored flight physics" uv run python scripts/embodiment/flybody_flight_physics_smoke.py
run "FlyBody wing actuation" uv run python scripts/embodiment/flybody_wing_smoke.py
run "FlyBody free-flight motor effect" uv run python scripts/embodiment/flybody_freeflight_smoke.py
run "FlyBody +x flight envelope" uv run python scripts/embodiment/flybody_flight_envelope.py
run "Flyppy compound-eye rendering" uv run python scripts/embodiment/flyppy_vision_smoke.py
run "MaleCNS to FlyBody motor bridge" uv run python scripts/embodiment/cns_flybody_smoke.py --snapshot "$SNAPSHOT" --groups "$GROUPS"

printf '\nembodiment=PASS\n'
printf 'groups: %s\n' "$GROUPS"
printf 'artifacts: %s/artifacts/embodiment\n' "$ROOT"
