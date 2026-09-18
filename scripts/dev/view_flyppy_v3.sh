#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v3}"
# Let the viewer follow the experiment's stored environment version unless the
# caller explicitly sets VF_VIEWER_ENVIRONMENT_VERSION.
exec bash scripts/dev/view_flyppy.sh "$EXPERIMENT"
