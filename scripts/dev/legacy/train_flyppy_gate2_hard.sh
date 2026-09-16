#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v1}"
STATE="$EXPERIMENT/curriculum-state.json"
CHECKPOINT="$EXPERIMENT/checkpoint/manifest.json"
BACKUP="$EXPERIMENT/curriculum-state.before-gate2-hard.json"

command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }
[ -f "$CHECKPOINT" ] || { echo "checkpoint not found at $CHECKPOINT" >&2; exit 2; }
[ -f "$STATE" ] || { echo "curriculum state not found at $STATE" >&2; exit 2; }

cp "$STATE" "$BACKUP"

# This is the immediately harder condition that failed 12/12 in episodes 72-95:
#   x=10.500, z=5.290, vx=350.0
# Keep it effectively fixed so repeated aversive/reward DAN events can change
# the same sensorimotor situation rather than bouncing between reset states.
uv run python - "$STATE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
state["training_gate_index"] = 1
state["training_phase"] = "gate2_hard_drill"
state["spawn_x_mm"] = 10.500
state["spawn_z_mm"] = 5.290
state["initial_speed_mm_s"] = 350.0
state["target_x_mm"] = 10.500
state["target_z_mm"] = 5.290
state["target_speed_mm_s"] = 350.0
state["consecutive_failures"] = 0
state["curriculum_complete"] = False
temp = path.with_name(f".{path.name}.hard.tmp")
temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
temp.replace(path)
print("gate2_hard_drill spawn=(x=10.500,z=5.290) vx=350.0")
PY

restore_state() {
  if [ -f "$BACKUP" ]; then
    uv run python - "$BACKUP" "$STATE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

backup_path = Path(sys.argv[1])
current_path = Path(sys.argv[2])
backup = json.loads(backup_path.read_text(encoding="utf-8"))
current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else {}
for key in ("curriculum_episodes", "successful_first_gates"):
    if key in current:
        backup[key] = current[key]
backup["consecutive_failures"] = 0
backup.pop("training_phase", None)
temp = current_path.with_name(f".{current_path.name}.restore.tmp")
temp.write_text(json.dumps(backup, indent=2) + "\n", encoding="utf-8")
temp.replace(current_path)
PY
    rm -f "$BACKUP"
  fi
}
trap restore_state EXIT INT TERM

export VF_COURSE_START_GATE=1

# Keep reset conditions effectively fixed. The CNS checkpoint remains plastic.
bash scripts/dev/train_flyppy.sh \
  --curriculum-target-x-mm 10.500 \
  --curriculum-target-z-mm 5.290 \
  --curriculum-target-speed-mm-s 350.0 \
  --curriculum-x-step-mm 0.000001 \
  --curriculum-z-step-mm 0.000001 \
  --curriculum-speed-step-mm-s 0.000001 \
  --curriculum-failure-x-step-mm 0.000001 \
  "$@"
