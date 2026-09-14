#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v2}"
CHECKPOINT="$EXPERIMENT/checkpoint/manifest.json"
if [ ! -f "$CHECKPOINT" ]; then
  # Avoid empty-array expansion under macOS's Bash 3.2 with `set -u`.
  # Prepend --fresh only for the first v2 run; later runs resume the existing
  # v2 checkpoint and preserve all user-supplied arguments.
  set -- --fresh "$@"
fi

# v2 is a new physical task: female FlyBody morphology at 2.75 mm / 1.09 mg,
# full-body gate collisions, researched joint limits, and the grounded
# wing+leg+hDVM motor boundary.  Do not silently import the wing-only v1 CNS
# checkpoint into this materially different embodiment.
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
