#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
LEARNED_CHECKPOINT="${VF_FLYPPY_CHECKPOINT:-$ROOT/artifacts/experiments/flyppy-v0/checkpoint}"

command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}
[ -d "$LEARNED_CHECKPOINT" ] || {
  echo "learned Flyppy checkpoint not found at $LEARNED_CHECKPOINT" >&2
  exit 2
}

if [ ! -f "$GROUPS" ]; then
  uv run python scripts/data/make_embodiment_groups.py \
    --snapshot "$SNAPSHOT" \
    --output "$GROUPS"
fi

exec uv run python scripts/embodiment/evaluate_flyppy.py \
  --snapshot "$SNAPSHOT" \
  --groups "$GROUPS" \
  --learned-checkpoint "$LEARNED_CHECKPOINT" \
  "$@"
