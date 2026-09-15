#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${VF_EXPERIMENT_DIR:-artifacts/experiments/flyppy-v3}"
export VF_VIEWER_ENVIRONMENT_VERSION="${VF_VIEWER_ENVIRONMENT_VERSION:-v3}"
exec bash scripts/dev/view_flyppy.sh "$EXPERIMENT"
