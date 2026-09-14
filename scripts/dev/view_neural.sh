#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PORT="${VF_NEURAL_VIEWER_PORT:-8765}"
EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v0}"

if [ "$#" -gt 0 ]; then
  EXPERIMENT="$1"
fi

command -v python3 >/dev/null 2>&1 || { echo 'python3 is required' >&2; exit 127; }

URL="http://127.0.0.1:${PORT}/visualization/neural-viewer.html?experiment=${EXPERIMENT}"

python3 -m http.server "$PORT" --bind 127.0.0.1 >/tmp/virtual-fly-neural-viewer.log 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

sleep 0.35
if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
  cat /tmp/virtual-fly-neural-viewer.log >&2 || true
  exit 1
fi

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

printf 'neural viewer: %s\n' "$URL"
printf 'experiment:    %s\n' "$EXPERIMENT"
printf 'Press Ctrl-C here to stop the local viewer server.\n'
wait "$SERVER_PID"
