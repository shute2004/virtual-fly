#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"
EXPERIMENT_DIR="${VF_EXPERIMENT_DIR:-$ROOT/artifacts/experiments/conditioning-v0}"
TRAIN_CYCLES="${VF_TRAIN_CYCLES:-24}"

run() {
  printf '\n== %s ==\n' "$1"
  shift
  "$@"
  status=$?
  if [ "$status" -ne 0 ]; then
    printf '\nFAILED (exit %s)\n' "$status" >&2
    exit "$status"
  fi
}

printf 'repository: %s\n' "$ROOT"
printf 'snapshot:   %s\n' "$SNAPSHOT"
printf 'experiment: %s\n' "$EXPERIMENT_DIR"
printf 'cycles:     %s\n' "$TRAIN_CYCLES"

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }

run "Python dependencies" uv sync

if [ -f "$SNAPSHOT/manifest.json" ]; then
  printf '\n== MaleCNS snapshot ==\nvalidating existing %s\n' "$SNAPSHOT"
  run "MaleCNS production snapshot semantics" uv run python - "$SNAPSHOT" <<'PY'
from pathlib import Path
import sys
from virtual_fly.reproducibility import validate_production_snapshot
info = validate_production_snapshot(Path(sys.argv[1]))
print(f"snapshot_semantics={info['runtime_semantics']}")
print(f"snapshot_sha256={info['snapshot_sha256']}")
PY
else
  run "MaleCNS v1.0 download and preprocessing" uv run python scripts/data/prepare_malecns.py --download --output "$SNAPSHOT"
fi

run "Conditioning group discovery" uv run python scripts/data/make_conditioning_config.py --snapshot "$SNAPSHOT" --output "$SNAPSHOT/conditioning-v0.json"
run "Rust compile check" cargo check --workspace
run "Rust tests" cargo test --workspace
run "GPU plasticity kernel smoke" cargo run -p vf-runner --bin vf-runner --release -- kernel-smoke --backend gpu --cycles 100
run "Real MaleCNS associative conditioning" cargo run -p vf-runner --bin malecns_conditioning --release -- --snapshot "$SNAPSHOT" --config "$SNAPSHOT/conditioning-v0.json" --output "$EXPERIMENT_DIR" --backend gpu --cycles "$TRAIN_CYCLES"
run "Conditioning analysis" uv run python scripts/analysis/analyze_conditioning.py --snapshot "$SNAPSHOT" --config "$SNAPSHOT/conditioning-v0.json" --experiment "$EXPERIMENT_DIR"

printf '\nbootstrap=PASS\n'
printf 'learned state: %s/learned_weights.f32le\n' "$EXPERIMENT_DIR"
printf 'result:        %s/result.json\n' "$EXPERIMENT_DIR"
printf 'analysis:      %s/analysis.json\n' "$EXPERIMENT_DIR"
