#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/dev/lib/runtime.sh"
vf_resolve_python "$ROOT"

PRODUCTION="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v4}"
PLAYBACK_DIR="${VF_PREVIEW_DIR:-artifacts/experiments/flyppy-best-playback}"
PLAYBACK_JSON="$PLAYBACK_DIR/playback.json"

export VF_FLYPPY_VISION_MODE="${VF_FLYPPY_VISION_MODE:-direct-ray}"
export VF_FLYPPY_OMMATIDIA_RAYS="${VF_FLYPPY_OMMATIDIA_RAYS:-13}"
export VF_VIEWER_ENVIRONMENT_VERSION="v4"

SOURCE_STEP="$("${VF_PYTHON[@]}" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["step"]))' "$PRODUCTION/checkpoint/manifest.json")"
PLAYBACK_STEP="-1"
if [ -f "$PLAYBACK_JSON" ]; then
  PLAYBACK_STEP="$("${VF_PYTHON[@]}" -c 'import json,sys
try: print(int(json.load(open(sys.argv[1])).get("checkpoint_neural_step",-1)))
except Exception: print(-1)' "$PLAYBACK_JSON")"
fi

if [ "$PLAYBACK_STEP" != "$SOURCE_STEP" ]; then
  rm -rf "$PLAYBACK_DIR"
  "${VF_PYTHON[@]}" scripts/embodiment/capture_flyppy_preview_playback.py \
    --production "$PRODUCTION" \
    --output "$PLAYBACK_JSON"
  if [ "$(uname -s)" = "Darwin" ]; then
    uv run mjpython scripts/embodiment/render_flyppy_preview_playback.py --playback "$PLAYBACK_JSON"
  else
    "${VF_PYTHON[@]}" scripts/embodiment/render_flyppy_preview_playback.py --playback "$PLAYBACK_JSON"
  fi
fi

"${VF_PYTHON[@]}" - "$PLAYBACK_JSON" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(f"best_preview_source={p['source_experiment']}")
print(f"best_preview_environment={p['environment_version']} condition=part3_frontier seed={p['course_seed']} plasticity=false")
print(f"best_preview_checkpoint_step={p['checkpoint_neural_step']} global_v={p['global_weight_version']}")
print(f"best_preview_frames={p['frame_count']} fps={p['playback_fps']} passed_gates={p['passed_gates']} terminal={p['terminal_reason']}")
print("video_recording=disabled playback=pre-rendered-physics-substeps")
PY

VF_SKIP_BODY_RENDERER=1 \
VF_EXPERIMENT_DIR="$PLAYBACK_DIR" \
bash scripts/dev/view_flyppy.sh "$PLAYBACK_DIR"
