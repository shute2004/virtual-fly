from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"

HISTORICAL_NEUTRAL_TRIM_PATTERN = ARTIFACTS_DIR / "embodiment/wing-pattern-neutral-trim-v1.npy"
PRODUCTION_NEUTRAL_TRIM_PATTERN = ARTIFACTS_DIR / "derived/wing-pattern-neutral-trim-v1.npy"
