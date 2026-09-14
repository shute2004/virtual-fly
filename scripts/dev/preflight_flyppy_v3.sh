#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

printf '== Flyppy v3 syntax ==\n'
uv run python -m py_compile \
  scripts/embodiment/flybody_v3_adapter.py \
  scripts/embodiment/flybody_v3_physical_spec_smoke.py \
  scripts/embodiment/flybody_v3_muscle_flight_diagnosis.py \
  scripts/embodiment/flybody_v3_vertical_flight_capability.py \
  scripts/embodiment/flyppy_course.py \
  scripts/embodiment/train_flyppy_curriculum.py

printf '\n== Official measured wingbeat ==\n'
uv run python scripts/dev/prefetch_flybody_flight_data.py

printf '\n== Flyppy v3 course geometry ==\n'
uv run python scripts/embodiment/flyppy_course.py

printf '\n== Flyppy v3 physical scale / gate clearance ==\n'
uv run python scripts/embodiment/flybody_v3_physical_spec_smoke.py

printf '\n== Flyppy v3 source-to-muscle lift diagnosis ==\n'
uv run python scripts/embodiment/flybody_v3_muscle_flight_diagnosis.py

printf '\n== Flyppy v3 vertical flight capability ==\n'
uv run python scripts/embodiment/flybody_v3_vertical_flight_capability.py

printf '\nflyppy_v3_preflight=PASS\n'
printf 'physical_spec=%s/artifacts/embodiment/flybody-physical-spec-v3.json\n' "$ROOT"
printf 'muscle_diagnosis=%s/artifacts/embodiment/flybody-v3-muscle-flight-diagnosis.json\n' "$ROOT"
printf 'vertical_capability=%s/artifacts/embodiment/flybody-v3-vertical-flight-capability.json\n' "$ROOT"
