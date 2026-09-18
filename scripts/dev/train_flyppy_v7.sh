#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-start}"
case "$MODE" in
  start|resume) ;;
  *)
    echo "usage: bash scripts/dev/train_flyppy_v7.sh [start|resume]" >&2
    exit 2
    ;;
esac

EXPERIMENT="${VF_FLYPPY_V7_EXPERIMENT:-artifacts/experiments/flyppy-v7-envv7-frontier}"
EPISODES="${VF_FLYPPY_V7_EPISODES:-240}"
CHECKPOINT_EVERY="${VF_FLYPPY_V7_CHECKPOINT_EVERY:-24}"
POPULATION="${VF_FLYPPY_V7_POPULATION:-4}"

if [ "$MODE" = "start" ]; then
  FRESH=1
else
  FRESH=0
fi

printf 'v7_training mode=%s experiment=%s episodes=%s population=%s checkpoint_every=%s\n' \
  "$MODE" "$EXPERIMENT" "$EPISODES" "$POPULATION" "$CHECKPOINT_EVERY"
printf '%s\n' 'curriculum=boundary-band frontier=focus+evaluation fixed_course=none fixed_spawn=none'

VF_EXPERIMENT_DIR="$EXPERIMENT" \
VF_FLYPPY_ENVIRONMENT_VERSION=v7 \
VF_FLYBODY_VERSION=v7 \
VF_FLYPPY_EPISODES="$EPISODES" \
VF_FLYPPY_POPULATION="$POPULATION" \
VF_FLYPPY_CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
VF_FLYPPY_FRESH="$FRESH" \
VF_FLYPPY_CURRICULUM_MODE=boundary-band \
VF_FLYPPY_BOUNDARY_BATCH_SIZE=24 \
VF_FLYPPY_LAUNCH_MODE=async \
VF_FLYPPY_VISION_MODE=direct-ray \
VF_FLYPPY_OMMATIDIA_RAYS=13 \
VF_FLYPPY_WRITE_LATEST_REPORTS=1 \
bash scripts/dev/train_flyppy_v3_population.sh
