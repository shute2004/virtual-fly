#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Flyppy v2 is preserved as a reproducible historical baseline.  Independent
# body-mechanics diagnostics have now proven that its current virtual-muscle seam
# has no altitude-sustaining operating point, so additional learning would mix a
# known physical impossibility into the neural-learning results.  Keep an
# explicit escape hatch only for intentional historical reproduction/debugging.
if [ "${VF_ALLOW_KNOWN_NONFLYING_V2:-0}" != "1" ]; then
  cat >&2 <<'EOF'
Flyppy v2 training is blocked: the current body/wing seam is known to be unable
to sustain altitude (flybody_vertical_flight_capability: sustaining=0).

Use the reference-flight calibration diagnostics before further neural training.
For deliberate historical reproduction only, set:
  VF_ALLOW_KNOWN_NONFLYING_V2=1
EOF
  exit 2
fi

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v2}"
CHECKPOINT="$EXPERIMENT/checkpoint/manifest.json"
if [ ! -f "$CHECKPOINT" ]; then
  # Avoid empty-array expansion under macOS's Bash 3.2 with `set -u`.
  # Prepend --fresh only for the first v2 run; later runs resume the existing
  # v2 checkpoint and preserve all user-supplied arguments.
  set -- --fresh "$@"
fi

# Historical v2 task: 2.75 mm / 1.09 mg provisional body specification,
# full-body gate collisions, researched joint limits, and the grounded
# wing+leg+hDVM motor boundary.  It is intentionally retained unchanged for
# reproducibility while reference-flight physics is repaired in a successor task.
unset VF_COURSE_START_GATE || true

bash scripts/dev/train_flyppy.sh \
  --environment-version v2 \
  --motor-boundary whole-body \
  --output-dir "$EXPERIMENT" \
  --curriculum-mode adaptive \
  --curriculum-start-x-mm 8.25 \
  --curriculum-target-x-mm 0.0 \
  --curriculum-target-z-mm 8.25 \
  --curriculum-start-speed-mm-s 400.0 \
  --curriculum-target-speed-mm-s 300.0 \
  --curriculum-x-step-mm 0.275 \
  --curriculum-failure-x-step-mm 0.1375 \
  --curriculum-z-step-mm 0.1375 \
  --curriculum-speed-step-mm-s 12.5 \
  --curriculum-max-easy-speed-mm-s 450.0 \
  "$@"
