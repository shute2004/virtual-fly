#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Preserve the v7 mechanical seam and learned motor semantics, but restore a
# conservative haltere -> MaleCNS timing-afferent feedback path.  Start forks
# the latest v7 gate-height experiment so the historical run remains immutable.
MODE="${1:-start}"
export VF_FLYBODY_VERSION=v7
export VF_HALTERE_SENSORY_MAP="${VF_HALTERE_SENSORY_MAP:-$ROOT/artifacts/malecns-v1.0/haltere-timing-afferents-v1.json}"
export VF_HALTERE_CURRENT_GAIN="${VF_HALTERE_CURRENT_GAIN:-0.05}"
export VF_HALTERE_TRANSDUCTION="${VF_HALTERE_TRANSDUCTION:-interaction-load-v2}"
export VF_FLYPPY_GATE2_SOURCE="${VF_FLYPPY_GATE2_SOURCE:-artifacts/experiments/flyppy-v7-gate2-height-v2}"
export VF_FLYPPY_GATE2_EXPERIMENT="${VF_FLYPPY_GATE2_EXPERIMENT:-artifacts/experiments/flyppy-v7h-gate2-height}"

exec bash scripts/dev/train_flyppy_v7_gate2_height.sh "$MODE"
