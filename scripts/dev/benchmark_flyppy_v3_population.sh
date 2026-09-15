#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
GROUPS="$SNAPSHOT/embodiment-groups-v0.json"
RETINOTOPIC_MAP="$SNAPSHOT/retinotopic-vision-v1.json"
WING_MOTOR_MAP="$SNAPSHOT/wing-motor-neurons-v0.json"
BODY_MOTOR_MAP="$SNAPSHOT/body-motor-neurons-v0.json"
NEURAL_CALIBRATION="$ROOT/artifacts/embodiment/neural-runtime-calibration-v1.json"
SOURCE_EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v3}"
BENCH_ROOT="$ROOT/artifacts/profiles/flyppy-v3-population-benchmark"
EPISODES="${VF_POP_BENCH_EPISODES:-8}"
MAX_STEPS="${VF_POP_BENCH_MAX_STEPS:-40}"
REPORT="$ROOT/reports/flyppy/population_benchmark.md"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }

for required in \
  "$SNAPSHOT/manifest.json" \
  "$GROUPS" \
  "$RETINOTOPIC_MAP" \
  "$WING_MOTOR_MAP" \
  "$BODY_MOTOR_MAP" \
  "$NEURAL_CALIBRATION" \
  "$SOURCE_EXPERIMENT/checkpoint/manifest.json" \
  "$SOURCE_EXPERIMENT/curriculum-state.json"
do
  [ -f "$required" ] || { echo "missing benchmark input: $required" >&2; exit 2; }
done

VF_NEURAL_SYNAPSE_SCALE="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["synapse_scale"])' "$NEURAL_CALIBRATION")"
export VF_NEURAL_SYNAPSE_SCALE

cargo test -q -p vf-neural transaction::tests
cargo check -q -p vf-runner --bin population_neural_bridge
uv run python -m py_compile \
  scripts/embodiment/population_neural_bridge_client.py \
  scripts/embodiment/train_flyppy_population.py

rm -rf "$BENCH_ROOT"
mkdir -p "$BENCH_ROOT"

run_population() {
  local population="$1"
  local out="$BENCH_ROOT/p${population}"
  mkdir -p "$out"
  cp -R "$SOURCE_EXPERIMENT/checkpoint" "$out/checkpoint"
  cp "$SOURCE_EXPERIMENT/curriculum-state.json" "$out/curriculum-state.json"

  printf '\n== population=%s ==\n' "$population"
  uv run python scripts/embodiment/train_flyppy_population.py \
    --episodes "$EPISODES" \
    --population "$population" \
    --snapshot "$SNAPSHOT" \
    --groups "$GROUPS" \
    --retinotopic-map "$RETINOTOPIC_MAP" \
    --wing-motor-map "$WING_MOTOR_MAP" \
    --body-motor-map "$BODY_MOTOR_MAP" \
    --output-dir "$out" \
    --max-control-steps "$MAX_STEPS" \
    --checkpoint-every 999999
}

run_population 1
run_population 2

mkdir -p "$(dirname "$REPORT")"
uv run python - "$BENCH_ROOT/p1/summary.json" "$BENCH_ROOT/p2/summary.json" "$REPORT" <<'PY'
import json
from pathlib import Path
import sys

p1 = json.loads(Path(sys.argv[1]).read_text())
p2 = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])

r1 = float(p1["aggregate_control_steps_per_second"])
r2 = float(p2["aggregate_control_steps_per_second"])
speedup = r2 / r1 if r1 > 0 else 0.0

lines = [
    "# Flyppy shared-weight population benchmark",
    "",
    f"- episodes per run: {p1['episodes_this_run']}",
    f"- max control steps per episode: {max(int(x['control_steps']) for x in p1['episode_results'])}",
    f"- population=1 aggregate control steps/s: {r1:.3f}",
    f"- population=2 aggregate control steps/s: {r2:.3f}",
    f"- population=2 / population=1 throughput ratio: {speedup:.3f}x",
    f"- population=1 elapsed: {float(p1['elapsed_seconds']):.3f} s",
    f"- population=2 elapsed: {float(p2['elapsed_seconds']):.3f} s",
    f"- population=2 mean staleness: {float(p2.get('mean_version_staleness', 0.0)):.3f}",
    f"- population=2 max staleness: {int(p2.get('max_version_staleness', 0))}",
    "- production checkpoint modified: false",
    "",
]
out.write_text("\n".join(lines), encoding="utf-8")
print(f"population_benchmark_report={out}")
print(f"population_1_control_steps_per_second={r1:.6f}")
print(f"population_2_control_steps_per_second={r2:.6f}")
print(f"population_2_speedup={speedup:.6f}")
PY
