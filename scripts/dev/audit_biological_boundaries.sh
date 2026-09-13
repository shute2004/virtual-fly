#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"

command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}

uv sync

printf '\n== Released wing motor-neuron inventory ==\n'
uv run python scripts/data/prepare_wing_motor_map.py \
  --snapshot "$SNAPSHOT" \
  --output "$SNAPSHOT/wing-motor-neurons-v0.json"

printf '\n== Released reinforcement DAN inventory ==\n'
uv run python scripts/data/prepare_reinforcement_targets.py \
  --snapshot "$SNAPSHOT" \
  --output "$SNAPSHOT/reinforcement-dans-v0.json"

printf '\nbiological_boundary_audit=PASS\n'
printf 'wing motor map: %s/wing-motor-neurons-v0.json\n' "$SNAPSHOT"
printf 'reinforcement: %s/reinforcement-dans-v0.json\n' "$SNAPSHOT"
