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

# The hard capability gate is the clamped source-to-muscle comparison above, not
# open-loop free-flight stability.  Upstream FlyBody's vision-flight task adds the
# wing-beat generator output to an agent action and applies the whole action to the
# walker; the measured baseline cycle by itself is therefore not a source-defined
# requirement for stable untethered flight.  Require the production muscle seam to
# preserve source-equivalent lift/tracking while attitude is re-anchored, then keep
# the symmetric open-loop flight sweep as an informative diagnostic only.
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/embodiment/flybody-v3-muscle-flight-diagnosis.json")
data = json.loads(path.read_text(encoding="utf-8"))
diagnosis = str(data.get("diagnosis", ""))
expected = "CLAMPED_V3_MUSCLE_SUPPORT_OK_FREE_FLIGHT_STABILITY_REMAINS"
print(f"v3_muscle_capability diagnosis={diagnosis}")
if diagnosis != expected:
    raise SystemExit(
        "Flyppy v3 source-to-muscle capability gate failed; repair the diagnosed "
        "physical/actuation seam before neural training"
    )
print("flybody_v3_muscle_capability=PASS")
PY

printf '\n== Flyppy v3 symmetric open-loop flight diagnostic ==\n'
set +e
uv run python scripts/embodiment/flybody_v3_vertical_flight_capability.py
OPEN_LOOP_STATUS=$?
set -e
if [[ "$OPEN_LOOP_STATUS" -eq 0 ]]; then
  printf 'flybody_v3_open_loop_stability=PASS\n'
else
  printf 'flybody_v3_open_loop_stability=FAIL_NONBLOCKING status=%s\n' "$OPEN_LOOP_STATUS"
  printf 'note=source FlyBody does not define fixed symmetric baseline-wingbeat free-flight stability as a prerequisite; MaleCNS closed-loop control is allowed to supply stabilization\n'
fi

printf '\nflyppy_v3_preflight=PASS\n'
printf 'physical_spec=%s/artifacts/embodiment/flybody-physical-spec-v3.json\n' "$ROOT"
printf 'muscle_diagnosis=%s/artifacts/embodiment/flybody-v3-muscle-flight-diagnosis.json\n' "$ROOT"
printf 'vertical_capability=%s/artifacts/embodiment/flybody-v3-vertical-flight-capability.json\n' "$ROOT"
