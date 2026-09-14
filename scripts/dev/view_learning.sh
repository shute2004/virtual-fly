#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v1}"
PORT="${VF_LIVE_VIEWER_PORT:-8765}"

if [ "$#" -gt 0 ]; then
  EXPERIMENT="$1"
fi

command -v python3 >/dev/null 2>&1 || { echo 'python3 is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }

URL="http://127.0.0.1:${PORT}/visualization/live-neural-viewer.html?experiment=${EXPERIMENT}"
LOG="/tmp/virtual-fly-live-viewer-${PORT}.log"
python3 -m http.server "$PORT" --bind 127.0.0.1 >"$LOG" 2>&1 &
SERVER_PID=$!
BODY_PID=""
cleanup() {
  if [ -n "$BODY_PID" ]; then kill "$BODY_PID" >/dev/null 2>&1 || true; fi
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

sleep 0.3
if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
  cat "$LOG" >&2 || true
  exit 1
fi

case "$(uname -s)" in
  Darwin)
    open "$URL"
    if command -v mjpython >/dev/null 2>&1; then
      uv run mjpython scripts/embodiment/live_body_viewer.py --experiment "$EXPERIMENT" &
    else
      echo 'mjpython is required for the detached MuJoCo body window on macOS' >&2
    fi
    ;;
  Linux)
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$URL" >/dev/null 2>&1 || true
    else
      echo "open in a browser: $URL"
    fi
    uv run python scripts/embodiment/live_body_viewer.py --experiment "$EXPERIMENT" &
    ;;
  *)
    echo "open in a browser: $URL"
    uv run python scripts/embodiment/live_body_viewer.py --experiment "$EXPERIMENT" &
    ;;
esac
BODY_PID=$!

printf 'neural viewer: %s\n' "$URL"
printf 'body viewer:   detached MuJoCo observer\n'
printf 'experiment:    %s\n' "$EXPERIMENT"
printf 'Closing either viewer does not stop training. Ctrl-C here closes observers only.\n'

if [ -n "$BODY_PID" ]; then
  wait "$BODY_PID" || true
else
  wait "$SERVER_PID"
fi
