#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

SNAPSHOT="${VF_SNAPSHOT:-$ROOT/artifacts/malecns-v1.0}"

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

command -v cargo >/dev/null 2>&1 || { echo 'cargo is required' >&2; exit 127; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required' >&2; exit 127; }

run "Python dependencies" uv sync
run "MaleCNS v1.0 download and preprocessing" uv run python scripts/data/prepare_malecns.py --download --output "$SNAPSHOT"
run "MaleCNS modulatory candidate discovery" uv run python scripts/data/find_modulatory_neurons.py --snapshot "$SNAPSHOT" --output "$SNAPSHOT/modulatory-candidates.json"
run "Rust format check" cargo fmt --all -- --check
run "Rust compile check" cargo check --workspace
run "Rust tests" cargo test --workspace
run "CPU parallel plasticity smoke" cargo run -p vf-runner --release -- kernel-smoke --backend cpu --cycles 250 --flies 8
run "GPU plasticity smoke" cargo run -p vf-runner --release -- kernel-smoke --backend gpu --cycles 250
run "MaleCNS snapshot info" cargo run -p vf-runner --release -- snapshot-info --snapshot "$SNAPSHOT"
run "MaleCNS GPU scale benchmark" cargo run -p vf-runner --release -- snapshot-benchmark --snapshot "$SNAPSHOT" --backend gpu --steps 10

printf '\nbootstrap=PASS\n'
