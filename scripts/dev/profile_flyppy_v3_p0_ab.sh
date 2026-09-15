#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

mkdir -p reports/flyppy

run_one() {
  local mode="$1"
  local report="reports/flyppy/profile_telemetry_${mode}.md"
  local json="reports/flyppy/profile_telemetry_${mode}.json"
  local log="reports/flyppy/profile_telemetry_${mode}.log"

  echo "=== Flyppy P0 telemetry=${mode} ===" | tee "$log"
  set +e
  VF_P0_TELEMETRY="$mode" \
    bash scripts/dev/profile_flyppy_v3_p0.sh \
      --report "$report" \
      --json "$json" \
      2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "exit_status=${status}" | tee -a "$log"
  return "$status"
}

set -e
on_status=0
off_status=0
run_one on || on_status=$?
run_one off || off_status=$?

{
  echo "# Flyppy P0 A/B run status"
  echo
  echo "- telemetry on exit: ${on_status}"
  echo "- telemetry off exit: ${off_status}"
} > reports/flyppy/profile_ab_status.md

printf '\nGenerated files:\n'
ls -lh reports/flyppy/profile_telemetry_on.* reports/flyppy/profile_telemetry_off.* reports/flyppy/profile_ab_status.md 2>/dev/null || true

if [ "$on_status" -ne 0 ] || [ "$off_status" -ne 0 ]; then
  echo "One or more profiler runs failed; commit the .log files and profile_ab_status.md so the failure can be diagnosed."
  exit 1
fi

echo "Both profiler runs completed successfully."
