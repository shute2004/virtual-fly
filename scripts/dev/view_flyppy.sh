#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v2}"
PORT="${VF_LIVE_VIEWER_PORT:-8765}"
SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
WING_MOTOR_MAP="$SNAPSHOT/wing-motor-neurons-v0.json"
VIEWER_GRAPH="$ROOT/artifacts/embodiment/neural-viewer-graph-v1.json"
EXPECTED_VIEWER_GRAPH_SCHEMA=3
NATIVE_BODY_WINDOW="${VF_NATIVE_BODY_WINDOW:-0}"

if [ "$#" -gt 0 ]; then
  EXPERIMENT="$1"
fi

command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }

[ -f "$GROUPS" ] || { echo "missing $GROUPS; run training/bootstrap first" >&2; exit 2; }
[ -f "$WING_MOTOR_MAP" ] || { echo "missing $WING_MOTOR_MAP; run training/bootstrap first" >&2; exit 2; }

VIEWER_SCHEMA=0
if [ -f "$VIEWER_GRAPH" ]; then
  VIEWER_SCHEMA="$(uv run python -c 'import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get("schema_version", 0)))
except Exception:
    print(0)' "$VIEWER_GRAPH")"
fi
if [ "$VIEWER_SCHEMA" != "$EXPECTED_VIEWER_GRAPH_SCHEMA" ]; then
  printf 'preparing schematic CNS viewer graph...\n'
  uv run python scripts/data/prepare_neural_viewer_graph.py \
    --snapshot "$SNAPSHOT" \
    --groups "$GROUPS" \
    --motor-map "$WING_MOTOR_MAP" \
    --output "$VIEWER_GRAPH"
fi

uv run python -m py_compile \
  scripts/embodiment/live_telemetry.py \
  scripts/embodiment/live_viewer_server.py \
  scripts/embodiment/live_body_viewer.py

URL="http://127.0.0.1:${PORT}/visualization/live-neural-viewer.html?experiment=${EXPERIMENT}"
SERVER_LOG="/tmp/virtual-fly-live-viewer-${PORT}.log"
BODY_LOG="/tmp/virtual-fly-live-body-${PORT}.log"

: >"$SERVER_LOG"
: >"$BODY_LOG"

uv run python scripts/embodiment/live_viewer_server.py \
  --port "$PORT" \
  --bind 127.0.0.1 \
  --root "$ROOT" \
  --experiment "$EXPERIMENT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
BODY_PID=""

cleanup() {
  if [ -n "$BODY_PID" ]; then
    kill "$BODY_PID" >/dev/null 2>&1 || true
  fi
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

sleep 0.25
if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
  printf '%s\n' 'viewer HTTP server failed:' >&2
  cat "$SERVER_LOG" >&2 || true
  exit 1
fi

BODY_LAUNCHER=(uv run python)
if [ "$(uname -s)" = "Darwin" ]; then
  BODY_LAUNCHER=(uv run mjpython)
fi
BODY_ARGS=(--experiment "$EXPERIMENT")
if [ "$NATIVE_BODY_WINDOW" = "1" ]; then
  BODY_ARGS+=(--native-window)
fi

"${BODY_LAUNCHER[@]}" scripts/embodiment/live_body_viewer.py "${BODY_ARGS[@]}" >"$BODY_LOG" 2>&1 &
BODY_PID=$!

sleep 0.5
if ! kill -0 "$BODY_PID" >/dev/null 2>&1; then
  printf '%s\n' 'FlyBody renderer failed:' >&2
  cat "$BODY_LOG" >&2 || true
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

printf 'learning_viewer=%s\n' "$URL"
printf 'FlyBody=main interactive MuJoCo render (drag orbit / wheel zoom)\n'
printf 'MaleCNS=inset schematic 3D observer\n'
printf 'telemetry=%s/live\n' "$EXPERIMENT"
printf 'viewer_server_log=%s\n' "$SERVER_LOG"
printf 'body_renderer_log=%s\n' "$BODY_LOG"
printf 'Set VF_NATIVE_BODY_WINDOW=1 if you also want the native MuJoCo viewer.\n'
printf 'Ctrl-C here stops observers only; training is independent.\n'

# Bash 3.2 on macOS has no `wait -n`, so monitor both long-lived viewer
# processes explicitly. A dead body renderer must not leave a healthy-looking
# HTTP page behind, and a dead HTTP server must not leave the renderer orphaned.
while true; do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    printf '%s\n' 'viewer HTTP server exited unexpectedly:' >&2
    cat "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if ! kill -0 "$BODY_PID" >/dev/null 2>&1; then
    printf '%s\n' 'FlyBody renderer exited unexpectedly:' >&2
    cat "$BODY_LOG" >&2 || true
    exit 1
  fi
  sleep 0.5
done
