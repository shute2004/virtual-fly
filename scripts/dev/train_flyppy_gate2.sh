#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v1}"
STATE="$EXPERIMENT/curriculum-state.json"
CHECKPOINT="$EXPERIMENT/checkpoint/manifest.json"

command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$CHECKPOINT" ] || {
  echo "checkpoint not found at $CHECKPOINT; train gate 1 first" >&2
  exit 2
}
[ -f "$STATE" ] || {
  echo "curriculum state not found at $STATE" >&2
  exit 2
}

# First entry into this stage keeps the learned CNS checkpoint but replaces only
# episode-reset curriculum coordinates. Gate 2 is at x=15 mm for the default
# deterministic course. We start 2.5 mm before it at its own vertical center.
PYTHONPATH="$ROOT/scripts/embodiment${PYTHONPATH:+:$PYTHONPATH}" \
uv run python - "$STATE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from flyppy_course import FlyppyCourse

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
if int(state.get("training_gate_index", 0)) != 1:
    course = FlyppyCourse(seed=0, gate_count=6)
    gate = course.gates[1]
    state["training_gate_index"] = 1
    state["spawn_x_mm"] = float(gate.x_mm - 2.5)
    state["spawn_z_mm"] = float(gate.center_z_mm)
    state["initial_speed_mm_s"] = 450.0
    state["successful_first_gates"] = 0
    state["consecutive_failures"] = 0
    state["target_x_mm"] = 9.0
    state["target_z_mm"] = 5.0
    state["target_speed_mm_s"] = 300.0
    state["curriculum_complete"] = False
    temp = path.with_name(f".{path.name}.gate2.tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print(
        "gate2_curriculum_initialized "
        f"spawn=(x={state['spawn_x_mm']:.3f},z={state['spawn_z_mm']:.3f}) "
        f"vx={state['initial_speed_mm_s']:.1f}"
    )
else:
    print(
        "gate2_curriculum_resume "
        f"spawn=(x={float(state['spawn_x_mm']):.3f},z={float(state['spawn_z_mm']):.3f}) "
        f"vx={float(state['initial_speed_mm_s']):.1f}"
    )
PY

export VF_COURSE_START_GATE=1

exec bash scripts/dev/train_flyppy.sh \
  --curriculum-target-x-mm 9.0 \
  --curriculum-target-z-mm 5.0 \
  --curriculum-target-speed-mm-s 300.0 \
  --curriculum-x-step-mm 0.5 \
  --curriculum-z-step-mm 0.25 \
  --curriculum-speed-step-mm-s 25.0 \
  "$@"
