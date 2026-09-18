"""Adaptive Flyppy curriculum primitives and persisted state migration."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, MutableMapping

@dataclass(frozen=True)
class SpawnCondition:
    x_mm: float
    z_mm: float
    speed_mm_s: float


@dataclass(frozen=True)
class AdaptiveCurriculumConfig:
    target: SpawnCondition
    x_step_mm: float
    z_step_mm: float
    speed_step_mm_s: float
    max_easy_speed_mm_s: float
    failure_x_step_mm: float
    first_gate_x_mm: float
    first_gate_z_mm: float


def move_toward(value: float, target: float, step: float) -> float:
    if value < target:
        return min(target, value + step)
    if value > target:
        return max(target, value - step)
    return target


def _move_away_toward_cap(value: float, target: float, cap: float, step: float) -> float:
    """Ease one parameter away from target without moving beyond ``cap``."""

    if math.isclose(value, cap, abs_tol=1e-12):
        return cap
    direction = 1.0 if cap > target else -1.0
    candidate = value + direction * step
    if direction > 0:
        return min(cap, candidate)
    return max(cap, candidate)


def initial_state(start: SpawnCondition, target: SpawnCondition) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "spawn_x_mm": float(start.x_mm),
        "spawn_z_mm": float(start.z_mm),
        "initial_speed_mm_s": float(start.speed_mm_s),
        "successful_first_gates": 0,
        "consecutive_failures": 0,
        "curriculum_episodes": 0,
        "target_x_mm": float(target.x_mm),
        "target_z_mm": float(target.z_mm),
        "target_speed_mm_s": float(target.speed_mm_s),
        "curriculum_complete": False,
    }


def load_state(
    path: Path,
    *,
    start: SpawnCondition,
    target: SpawnCondition,
    checkpoint_exists: bool,
) -> dict[str, Any]:
    if not checkpoint_exists or not path.exists():
        return initial_state(start, target)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return initial_state(start, target)
    if int(state.get("schema_version", 0)) not in (2, 3, 4):
        return initial_state(start, target)
    state["schema_version"] = 4
    state.setdefault("spawn_x_mm", float(start.x_mm))
    state.setdefault("spawn_z_mm", float(start.z_mm))
    state.setdefault("initial_speed_mm_s", float(start.speed_mm_s))
    state.setdefault("successful_first_gates", 0)
    state.setdefault("consecutive_failures", 0)
    state.setdefault("curriculum_episodes", 0)
    state.setdefault("target_x_mm", float(target.x_mm))
    state.setdefault("target_z_mm", float(target.z_mm))
    state.setdefault("target_speed_mm_s", float(target.speed_mm_s))
    state.setdefault("curriculum_complete", False)
    return state


def current_adaptive_condition(state: MutableMapping[str, Any]) -> SpawnCondition:
    return SpawnCondition(
        float(state["spawn_x_mm"]),
        float(state["spawn_z_mm"]),
        float(state["initial_speed_mm_s"]),
    )


def record_adaptive_result(
    state: MutableMapping[str, Any],
    config: AdaptiveCurriculumConfig,
    *,
    success: bool,
) -> None:
    spawn = current_adaptive_condition(state)
    if success:
        state["successful_first_gates"] = int(state.get("successful_first_gates", 0)) + 1
        state["consecutive_failures"] = 0
        state["spawn_x_mm"] = move_toward(spawn.x_mm, config.target.x_mm, config.x_step_mm)
        state["spawn_z_mm"] = move_toward(spawn.z_mm, config.target.z_mm, config.z_step_mm)
        state["initial_speed_mm_s"] = move_toward(
            spawn.speed_mm_s, config.target.speed_mm_s, config.speed_step_mm_s
        )
    else:
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
        state["spawn_x_mm"] = min(
            config.first_gate_x_mm - 1.0,
            spawn.x_mm + config.failure_x_step_mm,
        )
        state["spawn_z_mm"] = move_toward(
            spawn.z_mm, config.first_gate_z_mm, config.z_step_mm
        )
        state["initial_speed_mm_s"] = min(
            config.max_easy_speed_mm_s,
            spawn.speed_mm_s + config.speed_step_mm_s,
        )

    state["target_x_mm"] = float(config.target.x_mm)
    state["target_z_mm"] = float(config.target.z_mm)
    state["target_speed_mm_s"] = float(config.target.speed_mm_s)
    state["curriculum_complete"] = bool(
        math.isclose(float(state["spawn_x_mm"]), config.target.x_mm, abs_tol=1e-9)
        and math.isclose(float(state["spawn_z_mm"]), config.target.z_mm, abs_tol=1e-9)
        and math.isclose(
            float(state["initial_speed_mm_s"]), config.target.speed_mm_s, abs_tol=1e-9
        )
    )
