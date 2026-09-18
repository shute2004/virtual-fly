#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-}"
case "$MODE" in
  initial|learned) ;;
  *)
    echo "usage: bash scripts/dev/preview_flyppy_learning_compare.sh initial|learned" >&2
    exit 2
    ;;
esac

case "$MODE" in
  initial) DIR="artifacts/experiments/flyppy-preview-v7-fresh" ;;
  learned) DIR="artifacts/experiments/flyppy-preview-v7-learned" ;;
esac
PLAYBACK="$DIR/playback.json"

if [ ! -f "$PLAYBACK" ]; then
  echo "missing playback: $PLAYBACK" >&2
  exit 1
fi

python3 - "$PLAYBACK" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
print(f"comparison_preview={p.get('brain_state','unknown')}")
print(f"global_weight_version={p.get('global_weight_version')} checkpoint_step={p.get('checkpoint_neural_step')}")
print(f"environment={p.get('environment_version')} body={p.get('flight_body_version')} seed={p.get('course_seed')}")
print(f"passed_gates={p.get('passed_gates')} terminal={p.get('terminal_reason')} frames={p.get('frame_count')} fps={p.get('playback_fps')}")
print("video_recording=disabled")
PY

VF_SKIP_BODY_RENDERER=1 \
VF_EXPERIMENT_DIR="$DIR" \
bash scripts/dev/view_flyppy.sh "$DIR"
