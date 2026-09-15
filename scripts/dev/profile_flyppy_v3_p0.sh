#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

uv run python scripts/analysis/profile_flyppy_v3_p0.py \
  --warmup "${VF_P0_WARMUP_STEPS:-32}" \
  --steps "${VF_P0_MEASURE_STEPS:-256}" \
  --telemetry "${VF_P0_TELEMETRY:-on}" \
  "$@"

printf '\nP0 profile complete. Commit these compact reports after the run:\n'
printf '  reports/flyppy/profile_latest.md\n'
printf '  reports/flyppy/profile_latest.json\n'
