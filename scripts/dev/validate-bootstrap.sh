#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

failures=0
run_step() {
  name="$1"
  shift
  printf '\n== %s ==\n' "$name"
  "$@"
  status=$?
  if [ "$status" -ne 0 ]; then
    printf 'FAIL: %s (exit %s)\n' "$name" "$status"
    failures=$((failures + 1))
  else
    printf 'PASS: %s\n' "$name"
  fi
}

printf 'repository: %s\n' "$ROOT"
git status --short --branch || true
rustc --version || true
cargo --version || true

run_step "cargo fmt" cargo fmt --all -- --check
run_step "cargo check" cargo check --workspace
run_step "cargo test" cargo test --workspace
run_step "CPU parallel plasticity smoke" cargo run -p vf-runner --release -- kernel-smoke --backend cpu --cycles 250 --flies 8

if [ "$(uname -s)" = "Darwin" ]; then
  run_step "Metal GPU plasticity smoke" env WGPU_BACKEND=metal cargo run -p vf-runner --release -- kernel-smoke --backend gpu --cycles 250
fi

if [ -n "${VF_SNAPSHOT:-}" ] && [ -d "${VF_SNAPSHOT}" ]; then
  run_step "MaleCNS snapshot info" cargo run -p vf-runner --release -- snapshot-info --snapshot "$VF_SNAPSHOT"
  run_step "MaleCNS GPU scale smoke" cargo run -p vf-runner --release -- snapshot-benchmark --snapshot "$VF_SNAPSHOT" --backend auto --steps 5
fi

printf '\n== validation summary ==\n'
if [ "$failures" -eq 0 ]; then
  printf 'PASS: all bootstrap checks passed\n'
  exit 0
fi
printf 'FAIL: %s check(s) failed\n' "$failures"
exit 1
