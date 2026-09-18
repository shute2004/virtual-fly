#!/usr/bin/env bash
# Historical compatibility entry point. Production launcher is train_flyppy_population.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/scripts/dev/train_flyppy_population.sh" "$@"
