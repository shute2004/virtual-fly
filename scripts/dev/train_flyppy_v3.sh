#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v3}"
CHECKPOINT="$EXPERIMENT/checkpoint/manifest.json"

# v3 neural learning starts only after the source-equivalent body/course checks
# and the clamped source-to-production-muscle lift/tracking capability gate pass.
# Symmetric open-loop free-flight is retained by the preflight as a diagnostic,
# but is not a source-defined prerequisite for closed-loop neural stabilization.
if [ "${VF_SKIP_V3_FLIGHT_PREFLIGHT:-0}" != "1" ]; then
  bash scripts/dev/preflight_flyppy_v3.sh
fi

if [ ! -f "$CHECKPOINT" ]; then
  # macOS Bash 3.2-safe first-run argument prepend.
  set -- --fresh "$@"
fi

unset VF_COURSE_START_GATE || true

# v3: published FlyBody 2.97 mm / 0.983 mg scale, source-equivalent wing frame,
# -47.5 degree source flight root pose, measured hovering wingbeat baseline, and
# grounded whole-body MaleCNS motor boundary.
bash scripts/dev/train_flyppy.sh \
  --environment-version v3 \
  --motor-boundary whole-body \
  --output-dir "$EXPERIMENT" \
  --curriculum-mode adaptive \
  --curriculum-start-x-mm 8.91 \
  --curriculum-target-x-mm 0.0 \
  --curriculum-target-z-mm 8.91 \
  --curriculum-start-speed-mm-s 400.0 \
  --curriculum-target-speed-mm-s 300.0 \
  --curriculum-x-step-mm 0.297 \
  --curriculum-failure-x-step-mm 0.1485 \
  --curriculum-z-step-mm 0.1485 \
  --curriculum-speed-step-mm-s 12.5 \
  --curriculum-max-easy-speed-mm-s 450.0 \
  "$@"
