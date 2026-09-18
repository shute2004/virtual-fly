"""Shared configuration helpers for deterministic Flyppy playback/evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path

from virtual_fly.paths import REPO_ROOT
from virtual_fly.semantics import PRODUCTION_DIRECT_RAY_COUNT

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v4")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_VIEWER_GRAPH = Path("artifacts/embodiment/neural-viewer-graph-v1.json")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
DEFAULT_SPAWN_X_MM = 3.267
DEFAULT_SPAWN_Z_MM = 11.467
DEFAULT_SPEED_MM_S = 450.0
DEFAULT_COURSE_SEED = 22


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_viewer_ids(path: Path) -> tuple[int, ...]:
    payload = load_json(path)
    return tuple(sorted({int(node["body_id"]) for node in payload.get("nodes", [])}))


def ensure_runtime_environment(calibration: Path) -> None:
    """Preserve the current frozen-playback runtime defaults.

    Environment variables remain accepted for historical compatibility. Defaults
    are the production direct-ray K13 sensory path and the calibrated synapse scale.
    """
    os.environ.setdefault("VF_FLYPPY_VISION_MODE", "direct-ray")
    os.environ.setdefault("VF_FLYPPY_OMMATIDIA_RAYS", str(PRODUCTION_DIRECT_RAY_COUNT))
    if not os.environ.get("VF_NEURAL_SYNAPSE_SCALE", "").strip():
        payload = load_json(calibration)
        os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(payload["synapse_scale"]))
