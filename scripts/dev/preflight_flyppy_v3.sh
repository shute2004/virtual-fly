#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

printf '== Flyppy v3 syntax ==\n'
uv run python -m py_compile \
  scripts/embodiment/flybody_muscle_adapter.py \
  scripts/embodiment/flybody_v3_adapter.py \
  scripts/embodiment/flybody_v3_physical_spec_smoke.py \
  scripts/embodiment/flybody_v3_upstream_force_vector_ab.py \
  scripts/embodiment/flybody_v3_production_muscle_capability.py \
  scripts/embodiment/flybody_v3_vertical_flight_capability.py \
  scripts/embodiment/flyppy_course.py \
  scripts/embodiment/train_flyppy_curriculum.py

printf '\n== Official measured wingbeat ==\n'
uv run python scripts/dev/prefetch_flybody_flight_data.py

printf '\n== Upstream FlyBody source ==\n'
uv run python scripts/dev/prefetch_upstream_flybody_source.py

printf '\n== Flyppy v3 course geometry ==\n'
uv run python scripts/embodiment/flyppy_course.py

printf '\n== Flyppy v3 physical scale / gate clearance ==\n'
uv run python scripts/embodiment/flybody_v3_physical_spec_smoke.py

printf '\n== Flyppy v3 upstream fluid-force equivalence ==\n'
uv run python scripts/embodiment/flybody_v3_upstream_force_vector_ab.py
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/embodiment/flybody-v3-upstream-force-vector-ab.json")
data = json.loads(path.read_text(encoding="utf-8"))
diagnosis = str(data.get("diagnosis", ""))
expected = "V3_TOTAL_FLUID_VECTOR_APPROXIMATELY_SOURCE_EQUIVALENT"
print(f"v3_upstream_force_equivalence diagnosis={diagnosis}")
if diagnosis != expected:
    raise SystemExit(
        "Flyppy v3 body/fluid force vector differs materially from upstream FlyBody; "
        "repair the physical seam before neural training"
    )
print("flybody_v3_upstream_force_equivalence=PASS")
PY

printf '\n== Flyppy v3 production virtual-muscle capability ==\n'
uv run python scripts/embodiment/flybody_v3_production_muscle_capability.py

# Stable open-loop flight under a fixed symmetric muscle state is useful diagnostic
# information but is not an upstream FlyBody requirement.  The source vision-flight
# task applies policy action on top of the wing-beat generator, so closed-loop MaleCNS
# control is allowed to provide attitude stabilization.  Record this sweep without
# turning instability into a training veto or producing a misleading traceback.
printf '\n== Flyppy v3 symmetric open-loop flight diagnostic ==\n'
uv run python scripts/embodiment/flybody_v3_vertical_flight_capability.py --nonblocking

printf '\nflyppy_v3_preflight=PASS\n'
printf 'physical_spec=%s/artifacts/embodiment/flybody-physical-spec-v3.json\n' "$ROOT"
printf 'upstream_force_ab=%s/artifacts/embodiment/flybody-v3-upstream-force-vector-ab.json\n' "$ROOT"
printf 'production_muscle_capability=%s/artifacts/embodiment/flybody-v3-production-muscle-capability.json\n' "$ROOT"
printf 'vertical_capability=%s/artifacts/embodiment/flybody-v3-vertical-flight-capability.json\n' "$ROOT"
