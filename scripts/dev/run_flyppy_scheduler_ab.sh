#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SOURCE="${VF_FLYPPY_AB_SOURCE:-artifacts/experiments/flyppy-v3}"
AB_ROOT="${VF_FLYPPY_AB_ROOT:-artifacts/experiments/flyppy-v3-scheduler-ab}"
REPORT_ROOT="${VF_FLYPPY_AB_REPORT_ROOT:-reports/flyppy/experiments/scheduler-ab}"
EPISODES="${VF_FLYPPY_AB_EPISODES:-24}"
ASYNC_DIR="$AB_ROOT/async"
WAVE_DIR="$AB_ROOT/wave"
ASYNC_FIXED_JSON="$REPORT_ROOT/async_fixed.json"
ASYNC_FIXED_MD="$REPORT_ROOT/async_fixed.md"
WAVE_FIXED_JSON="$REPORT_ROOT/wave_fixed.json"
WAVE_FIXED_MD="$REPORT_ROOT/wave_fixed.md"
COMPARISON_JSON="$REPORT_ROOT/comparison.json"
COMPARISON_MD="$REPORT_ROOT/comparison.md"

[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || {
  echo "VF_FLYPPY_AB_EPISODES must be a positive integer" >&2
  exit 2
}

source "$ROOT/scripts/dev/lib/runtime.sh"
vf_resolve_python "$ROOT"
PYTHON_RUNNER=("${VF_PYTHON[@]}")

[ -f "$SOURCE/checkpoint/manifest.json" ] || {
  echo "source Flyppy checkpoint not found: $SOURCE/checkpoint/manifest.json" >&2
  exit 2
}
if [ -e "$AB_ROOT" ]; then
  echo "A/B output already exists: $AB_ROOT" >&2
  echo "Choose a new VF_FLYPPY_AB_ROOT; this script never deletes prior experiments." >&2
  exit 2
fi
if [ -e "$REPORT_ROOT" ]; then
  echo "A/B report output already exists: $REPORT_ROOT" >&2
  echo "Choose a new VF_FLYPPY_AB_REPORT_ROOT; this script never overwrites prior A/B reports." >&2
  exit 2
fi

printf '== Flyppy scheduler A/B shared preflight ==\n'
bash scripts/dev/preflight_flyppy_v3.sh

mkdir -p "$AB_ROOT" "$REPORT_ROOT"

"${PYTHON_RUNNER[@]}" scripts/dev/fork_flyppy_experiment.py \
  "$SOURCE" "$ASYNC_DIR" --launch-mode async
"${PYTHON_RUNNER[@]}" scripts/dev/fork_flyppy_experiment.py \
  "$SOURCE" "$WAVE_DIR" --launch-mode wave

run_training() {
  local mode="$1"
  local experiment="$2"
  echo
  echo "=== Flyppy scheduler A/B: $mode ==="
  echo "viewer_available_command=VF_EXPERIMENT_DIR=$experiment bash scripts/dev/view_flyppy_v3.sh"
  VF_EXPERIMENT_DIR="$experiment" \
  VF_FLYPPY_LAUNCH_MODE="$mode" \
  VF_FLYPPY_EPISODES="$EPISODES" \
  VF_FLYPPY_WRITE_LATEST_REPORTS=0 \
  VF_SKIP_V3_FLIGHT_PREFLIGHT=1 \
    bash scripts/dev/train_flyppy_v3_population.sh
}

run_fixed_eval() {
  local experiment="$1"
  local output_json="$2"
  local output_md="$3"
  "${PYTHON_RUNNER[@]}" scripts/analysis/evaluate_flyppy_fixed.py \
    --production "$experiment" \
    --output-json "$output_json" \
    --report "$output_md"
}

run_training async "$ASYNC_DIR"
run_fixed_eval "$ASYNC_DIR" "$ASYNC_FIXED_JSON" "$ASYNC_FIXED_MD"

run_training wave "$WAVE_DIR"
run_fixed_eval "$WAVE_DIR" "$WAVE_FIXED_JSON" "$WAVE_FIXED_MD"

"${PYTHON_RUNNER[@]}" scripts/analysis/compare_flyppy_scheduler_runs.py \
  --left-summary "$ASYNC_DIR/summary.json" \
  --right-summary "$WAVE_DIR/summary.json" \
  --left-fixed "$ASYNC_FIXED_JSON" \
  --right-fixed "$WAVE_FIXED_JSON" \
  --output-json "$COMPARISON_JSON" \
  --report "$COMPARISON_MD"

printf '\nFlyppy scheduler A/B complete\n'
printf 'async_experiment=%s\n' "$ASYNC_DIR"
printf 'wave_experiment=%s\n' "$WAVE_DIR"
printf 'comparison_report=%s\n' "$COMPARISON_MD"
