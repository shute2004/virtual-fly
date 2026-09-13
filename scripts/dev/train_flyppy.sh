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

# The current Flyppy runner still reduces bilateral DNg02 population activity to
# spike fractions and then computes wing amplitudes outside the nervous system.
# That adapter remains useful for isolated physics smoke tests, but it is not an
# acceptable learning boundary for the target virtual-fly experiment. Refuse to
# run training by default until individual released wing motor neurons are wired
# through a peripheral motor/muscle model.
if [ "${VF_ALLOW_PROVISIONAL_DNG02_MOTOR:-0}" != "1" ]; then
  cat >&2 <<'EOF'
Flyppy learning is intentionally disabled at the current biological boundary.

Reason: the existing runner still averages DNg02 population spikes into an
external wing-amplitude command. virtual-fly must instead let the MaleCNS
premotor network reach individual released wing motor neurons and connect those
neurons to the peripheral muscle/body model without an external action decoder.

Run:
  bash scripts/dev/audit_biological_boundaries.sh

The provisional DNg02 adapter can still be exercised explicitly for legacy
physics diagnostics by setting VF_ALLOW_PROVISIONAL_DNG02_MOTOR=1, but results
from that mode must not be treated as target-faithful learning experiments.
EOF
  exit 2
fi

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
