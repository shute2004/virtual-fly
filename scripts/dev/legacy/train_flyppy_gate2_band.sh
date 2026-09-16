#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v1}"
STATE="$EXPERIMENT/curriculum-state.json"
CHECKPOINT="$EXPERIMENT/checkpoint/manifest.json"

command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$CHECKPOINT" ] || { echo "checkpoint not found at $CHECKPOINT" >&2; exit 2; }
[ -f "$STATE" ] || { echo "curriculum state not found at $STATE" >&2; exit 2; }

# Gate 2 remains the focused training stage. Unlike the old one-episode
# success/failure ping-pong, this mode presents a shuffled band of conditions
# spanning the empirically measured 0%-success and 100%-success endpoints.
uv run python - "$STATE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
state["training_gate_index"] = 1
state["training_phase"] = "gate2_boundary_band"
state["curriculum_mode"] = "boundary-band"
state["curriculum_complete"] = False
# Keep an existing boundary_band payload so repeated runs continue the same
# 24-episode batch and retain batch-level difficulty adjustments.
temp = path.with_name(f".{path.name}.gate2-band.tmp")
temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
temp.replace(path)
PY

export VF_COURSE_START_GATE=1

bash scripts/dev/train_flyppy.sh \
  --curriculum-mode boundary-band \
  --curriculum-target-x-mm 9.0 \
  --curriculum-target-z-mm 5.0 \
  --curriculum-target-speed-mm-s 300.0 \
  --boundary-hard-x-mm 10.500 \
  --boundary-hard-z-mm 5.290 \
  --boundary-hard-speed-mm-s 350.0 \
  --boundary-easy-x-mm 10.625 \
  --boundary-easy-z-mm 5.3525 \
  --boundary-easy-speed-mm-s 356.25 \
  --boundary-batch-size 24 \
  --boundary-harden-success-rate 0.80 \
  --boundary-ease-success-rate 0.40 \
  --boundary-harden-x-step-mm 0.125 \
  --boundary-harden-z-step-mm 0.0625 \
  --boundary-harden-speed-step-mm-s 6.25 \
  --boundary-ease-x-step-mm 0.0625 \
  --boundary-ease-z-step-mm 0.03125 \
  --boundary-ease-speed-step-mm-s 3.125 \
  "$@"
