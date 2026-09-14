#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v1}"
PORT="${VF_LIVE_VIEWER_PORT:-8765}"
SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
WING_MOTOR_MAP="$SNAPSHOT/wing-motor-neurons-v0.json"
VIEWER_GRAPH="$ROOT/artifacts/embodiment/neural-viewer-graph-v1.json"

if [ "$#" -gt 0 ]; then
  EXPERIMENT="$1"
fi

command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
command -v python3 >/dev/null 2>&1 || { echo 'python3 is required' >&2; exit 127; }

if [ ! -f "$VIEWER_GRAPH" ]; then
  [ -f "$GROUPS" ] || { echo "missing $GROUPS; run training/bootstrap first" >&2; exit 2; }
  [ -f "$WING_MOTOR_MAP" ] || { echo "missing $WING_MOTOR_MAP; run training/bootstrap first" >&2; exit 2; }
  printf 'preparing neural viewer graph...\n'
  uv run python scripts/data/prepare_neural_viewer_graph.py \
    --snapshot "$SNAPSHOT" \
    --groups "$GROUPS" \
    --motor-map "$WING_MOTOR_MAP" \
    --output "$VIEWER_GRAPH"
fi

URL="http://127.0.0.1:${PORT}/visualization/live-neural-viewer.html?experiment=${EXPERIMENT}"
LOG="/tmp/virtual-fly-live-viewer-${PORT}.log"

python3 -m http.server "$PORT" --bind 127.0.0.1 >"$LOG" 2>&1 &
SERVER_PID=$!
BODY_PID=""

cleanup() {
  if [ -n "$BODY_PID" ]; then
    kill "$BODY_PID" >/dev/null 2>&1 || true
  fi
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

sleep 0.25
if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
  cat "$LOG" >&2 || true
  exit 1
fi

BODY_LAUNCHER=(uv run python)
if [ "$(uname -s)" = "Darwin" ]; then
  command -v mjpython >/dev/null 2>&1 || {
    echo 'mjpython is required on macOS for the detached MuJoCo viewer' >&2
    exit 127
  }
  BODY_LAUNCHER=(uv run mjpython)
fi

"${BODY_LAUNCHER[@]}" scripts/embodiment/live_body_viewer.py \
  --experiment "$EXPERIMENT" &
BODY_PID=$!

case "$(uname -s)" in
  Darwin)
    open "$URL"
    ;;
  Linux)
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$URL" >/dev/null 2>&1 || true
    else
      echo "open in a browser: $URL"
    fi
    ;;
  *)
    echo "open in a browser: $URL"
    ;;
esac

printf 'body_viewer=MuJoCo detached observer\n'
printf 'neural_viewer=%s\n' "$URL"
printf 'telemetry=%s/live\n' "$EXPERIMENT"
printf 'Closing either viewer does not stop training. Ctrl-C here stops observers only.\n'

# Keep the local HTTP server available even if the MuJoCo window is closed;
# the neural viewer can remain open independently.
wait "$SERVER_PID"
