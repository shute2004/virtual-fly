#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VIEWER="$ROOT/visualization/neural-viewer.html"

[ -f "$VIEWER" ] || { echo "viewer not found: $VIEWER" >&2; exit 2; }

case "$(uname -s)" in
  Darwin)
    open "$VIEWER"
    ;;
  Linux)
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$VIEWER"
    else
      echo "open this file in a browser: $VIEWER"
    fi
    ;;
  *)
    echo "open this file in a browser: $VIEWER"
    ;;
esac
