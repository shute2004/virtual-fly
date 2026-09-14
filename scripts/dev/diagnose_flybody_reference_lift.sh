#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

printf '== Official FlyBody measured wingbeat ==\n'
uv run python scripts/dev/prefetch_flybody_flight_data.py

printf '\n== FlyBody reference lift calibration ==\n'
uv run python scripts/embodiment/flybody_reference_lift_calibration.py

printf '\nreference_lift_diagnosis=COMPLETE\n'
printf 'result=%s/artifacts/embodiment/flybody-reference-lift-calibration.json\n' "$ROOT"
