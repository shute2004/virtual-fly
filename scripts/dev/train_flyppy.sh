#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
RETINOTOPIC_MAP="$SNAPSHOT/retinotopic-vision-v1.json"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$SNAPSHOT/manifest.json" ] || {
  echo "MaleCNS snapshot not found at $SNAPSHOT; run scripts/dev/bootstrap.sh first" >&2
  exit 2
}

uv sync
uv run python scripts/data/make_embodiment_groups.py \
  --snapshot "$SNAPSHOT" \
  --output "$GROUPS"
uv run python scripts/data/prepare_retinotopic_vision.py \
  --snapshot "$SNAPSHOT" \
  --output "$RETINOTOPIC_MAP" \
  --download

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

exec "${PYTHON_LAUNCHER[@]}" scripts/embodiment/flyppy_closed_loop.py \
  --snapshot "$SNAPSHOT" \
  --groups "$GROUPS" \
  --retinotopic-map "$RETINOTOPIC_MAP" \
  "$@"
