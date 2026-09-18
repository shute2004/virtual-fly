#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-start}"
case "$MODE" in
  start|resume) ;;
  *)
    echo "usage: bash scripts/dev/train_flyppy_v7_gate2_height.sh [start|resume]" >&2
    exit 2
    ;;
esac

SOURCE="${VF_FLYPPY_GATE2_SOURCE:-artifacts/experiments/flyppy-v7-envv7-frontier}"
EXPERIMENT="${VF_FLYPPY_GATE2_EXPERIMENT:-artifacts/experiments/flyppy-v7-gate2-height-v2}"
EPISODES="${VF_FLYPPY_GATE2_EPISODES:-240}"
CHECKPOINT_EVERY="${VF_FLYPPY_GATE2_CHECKPOINT_EVERY:-24}"
POPULATION="${VF_FLYPPY_GATE2_POPULATION:-4}"
BATCH_SIZE="${VF_FLYPPY_GATE2_BATCH_SIZE:-24}"

# Measured from the final v7 checkpoint on course seed 0.  The fixed spawn is a
# normal late-boundary evaluation condition, not a near-gate focus reset.  At
# this spawn the frozen checkpoint clears gate 2 at center 12.75 mm but fails at
# 13.50 mm, so 12.75 mm gives the curriculum a real success frontier.  Probe
# episodes test the next +0.25 mm level before it becomes the current level.
COURSE_SEED="${VF_FLYPPY_GATE2_COURSE_SEED:-0}"
SPAWN_X="${VF_FLYPPY_GATE2_SPAWN_X_MM:-8.83575}"
SPAWN_Z="${VF_FLYPPY_GATE2_SPAWN_Z_MM:-11.2286819148691}"
SPAWN_SPEED="${VF_FLYPPY_GATE2_SPAWN_SPEED_MM_S:-400.0}"
START_Z="${VF_FLYPPY_GATE2_START_Z_MM:-12.75}"
TARGET_Z="${VF_FLYPPY_GATE2_TARGET_Z_MM:-14.282297335224168}"
STEP_Z="${VF_FLYPPY_GATE2_STEP_MM:-0.125}"

if [ "$MODE" = "start" ]; then
  rm -rf "$EXPERIMENT"
  .venv/bin/python scripts/dev/fork_flyppy_experiment.py \
    "$SOURCE" "$EXPERIMENT" --launch-mode async
fi

printf 'gate2_height_training mode=%s source=%s experiment=%s episodes=%s\n' \
  "$MODE" "$SOURCE" "$EXPERIMENT" "$EPISODES"
printf 'course_seed=%s spawn=(%s,%s) speed=%s gate2_start=%s target=%s step=%s\n' \
  "$COURSE_SEED" "$SPAWN_X" "$SPAWN_Z" "$SPAWN_SPEED" "$START_Z" "$TARGET_Z" "$STEP_Z"
if [ "${VF_FLYPPY_GATE2_ACQUISITION_ONLY:-0}" = "1" ]; then
  printf 'batch=%s acquisition-only: all episodes current; advance when current>=80%%\n' "$BATCH_SIZE"
else
  printf 'batch=%s standard current/review mix; all episodes learn; advance when current>=80%%\n' "$BATCH_SIZE"
fi

VF_EXPERIMENT_DIR="$EXPERIMENT" \
VF_FLYPPY_ENVIRONMENT_VERSION=v7 \
VF_FLYBODY_VERSION=v7 \
VF_FLYPPY_EPISODES="$EPISODES" \
VF_FLYPPY_POPULATION="$POPULATION" \
VF_FLYPPY_CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
VF_FLYPPY_FRESH=0 \
VF_FLYPPY_CURRICULUM_MODE=gate2-height \
VF_FLYPPY_BOUNDARY_BATCH_SIZE="$BATCH_SIZE" \
VF_FLYPPY_LAUNCH_MODE=async \
VF_FLYPPY_VISION_MODE=direct-ray \
VF_FLYPPY_OMMATIDIA_RAYS=13 \
VF_FLYPPY_WRITE_LATEST_REPORTS="${VF_FLYPPY_WRITE_LATEST_REPORTS:-1}" \
bash scripts/dev/train_flyppy_population.sh \
  --fixed-course-seed "$COURSE_SEED" \
  --fixed-spawn-x-mm "$SPAWN_X" \
  --fixed-spawn-z-mm "$SPAWN_Z" \
  --fixed-spawn-speed-mm-s "$SPAWN_SPEED" \
  --gate2-height-start-z-mm "$START_Z" \
  --gate2-height-target-z-mm "$TARGET_Z" \
  --gate2-height-step-mm "$STEP_Z" \
  --gate2-height-current-success-rate 0.80 \
  --gate2-height-retention-success-rate 0.80
