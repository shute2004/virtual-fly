#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LEARNED="${VF_FLYPPY_POST_LEARNED:-artifacts/experiments/flyppy-v7h-deadline-acquisition}"
BASELINE="${VF_FLYPPY_POST_BASELINE:-artifacts/experiments/flyppy-v7-envv7-frontier}"
OUT="${VF_FLYPPY_POST_OUT:-artifacts/experiments/flyppy-post-compare}"
TARGET="${VF_FLYPPY_POST_GATE2_Z_MM:-14.282297335224168}"
HALTERE_MAP="${VF_HALTERE_SENSORY_MAP:-artifacts/malecns-v1.0/haltere-timing-afferents-v1.json}"
HALTERE_GAIN="${VF_HALTERE_CURRENT_GAIN:-0.05}"
mkdir -p "$OUT/baseline" "$OUT/learned"

capture_one() {
  local name="$1" production="$2"
  local dir="$OUT/$name"
  rm -rf "$dir/frames"
  .venv/bin/python scripts/embodiment/capture_flyppy_preview_playback.py \
    --production "$production" \
    --output "$dir/playback.json" \
    --environment-version v7 \
    --flight-body-version v7 \
    --haltere-sensory-map "$HALTERE_MAP" \
    --haltere-sensory-kind male-cns-haltere-timing-afferents-inferred-v1 \
    --haltere-current-gain "$HALTERE_GAIN" \
    --haltere-transduction interaction-load-v2 \
    --course-seed 0 \
    --spawn-x-mm 8.83575 \
    --spawn-z-mm 11.2286819148691 \
    --initial-speed-mm-s 400.0 \
    --gate2-center-z-mm "$TARGET" \
    --physics-substep-stride 3 \
    --playback-fps 60
  .venv/bin/python scripts/embodiment/render_flyppy_preview_playback.py \
    --playback "$dir/playback.json" --width 960 --height 540
  ffmpeg -hide_banner -loglevel error -y \
    -framerate 60 -i "$dir/frames/frame-%05d.jpg" \
    -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart \
    "$dir/$name.mp4"
  .venv/bin/python - "$dir/playback.json" "$dir/$name.mp4" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(f"{sys.argv[2]}: global_v={p['global_weight_version']} passed={p['passed_gates']} terminal={p['terminal_reason']} controls={p['control_steps']}")
PY
}

capture_one baseline "$BASELINE"
capture_one learned "$LEARNED"
printf 'baseline_mp4=%s\n' "$OUT/baseline/baseline.mp4"
printf 'learned_mp4=%s\n' "$OUT/learned/learned.mp4"
