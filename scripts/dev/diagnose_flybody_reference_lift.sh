#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

printf '== Official FlyBody measured wingbeat ==\n'
uv run python scripts/dev/prefetch_flybody_flight_data.py

printf '\n== Upstream FlyBody source assets ==\n'
uv run python scripts/dev/prefetch_upstream_flybody_source.py

printf '\n== FlyBody reference lift calibration ==\n'
uv run python scripts/embodiment/flybody_reference_lift_calibration.py

printf '\n== Exact-kinematic FlyBody aerodynamic support ==\n'
uv run python scripts/embodiment/flybody_kinematic_aero_support.py

printf '\n== FlyGym wing-frame aerodynamic A/B ==\n'
uv run python scripts/embodiment/flybody_wing_frame_aero_ab.py

printf '\n== Upstream XML exact-kinematic aerodynamic support ==\n'
uv run python scripts/embodiment/flybody_upstream_source_aero_reference.py

printf '\nreference_lift_diagnosis=COMPLETE\n'
printf 'result=%s/artifacts/embodiment/flybody-reference-lift-calibration.json\n' "$ROOT"
printf 'kinematic_result=%s/artifacts/embodiment/flybody-kinematic-aero-support.json\n' "$ROOT"
printf 'wing_frame_result=%s/artifacts/embodiment/flybody-wing-frame-aero-ab.json\n' "$ROOT"
printf 'upstream_source_result=%s/artifacts/embodiment/flybody-upstream-source-aero-reference.json\n' "$ROOT"
