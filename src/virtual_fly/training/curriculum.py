"""Curriculum policies for Flyppy training.

The neural learning rule lives in the CNS runtime.  Curriculum code may only
choose episode-reset initial conditions and record success/failure statistics;
it must never inject target actions, scalar reward values, or motor commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
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


@dataclass(frozen=True)
class BoundaryBandConfig:
    hard: SpawnCondition
    easy: SpawnCondition
    target: SpawnCondition
    batch_size: int = 24
    harden_success_rate: float = 0.80
    ease_success_rate: float = 0.40
    harden_step: SpawnCondition = SpawnCondition(0.125, 0.0625, 6.25)
    ease_step: SpawnCondition = SpawnCondition(0.0625, 0.03125, 3.125)
    seed: int = 0


def move_toward(value: float, target: float, step: float) -> float:
    if value < target:
        return min(target, value + step)
    if value > target:
        return max(target, value - step)
    return target


def _move_away_toward_cap(value: float, target: float, cap: float, step: float) -> float:
    """Ease one parameter away from target but never beyond its initial cap."""

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


def _condition_dict(condition: SpawnCondition) -> dict[str, float]:
    return {
        "x_mm": float(condition.x_mm),
        "z_mm": float(condition.z_mm),
        "speed_mm_s": float(condition.speed_mm_s),
    }


def _condition_from_dict(payload: MutableMapping[str, Any]) -> SpawnCondition:
    return SpawnCondition(
        float(payload["x_mm"]),
        float(payload["z_mm"]),
        float(payload["speed_mm_s"]),
    )


def _validate_boundary(config: BoundaryBandConfig) -> None:
    if config.batch_size < 5:
        raise ValueError("boundary batch_size must be >= 5")
    if not 0.0 <= config.ease_success_rate < config.harden_success_rate <= 1.0:
        raise ValueError("boundary success-rate thresholds are invalid")
    values = (
        config.hard.x_mm,
        config.hard.z_mm,
        config.hard.speed_mm_s,
        config.easy.x_mm,
        config.easy.z_mm,
        config.easy.speed_mm_s,
        config.target.x_mm,
        config.target.z_mm,
        config.target.speed_mm_s,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("boundary conditions must be finite")
    steps = (
        config.harden_step.x_mm,
        config.harden_step.z_mm,
        config.harden_step.speed_mm_s,
        config.ease_step.x_mm,
        config.ease_step.z_mm,
        config.ease_step.speed_mm_s,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in steps):
        raise ValueError("boundary shift steps must be finite and positive")


def ensure_boundary_state(
    state: MutableMapping[str, Any], config: BoundaryBandConfig
) -> MutableMapping[str, Any]:
    _validate_boundary(config)
    payload = state.get("boundary_band")
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        payload = {
            "schema_version": 1,
            "initial_hard": _condition_dict(config.hard),
            "initial_easy": _condition_dict(config.easy),
            "hard": _condition_dict(config.hard),
            "easy": _condition_dict(config.easy),
            "batch_number": 0,
            "attempts_in_batch": 0,
            "successes_in_batch": 0,
            "last_batch_success_rate": None,
            "last_adjustment": "initialized",
            "harder_shifts": 0,
            "easier_shifts": 0,
        }
        state["boundary_band"] = payload
    return payload


def _level_counts(batch_size: int) -> list[int]:
    # Target proportions mirror the desired 24-episode distribution 4/5/6/5/4.
    weights = (4, 5, 6, 5, 4)
    raw = [batch_size * weight / sum(weights) for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remaining = batch_size - sum(counts)
    order = sorted(range(5), key=lambda i: (raw[i] - counts[i], -abs(i - 2)), reverse=True)
    for i in order[:remaining]:
        counts[i] += 1
    return counts


def _shuffled_levels(batch_size: int, seed: int, batch_number: int) -> list[float]:
    counts = _level_counts(batch_size)
    levels: list[float] = []
    for level, count in zip((0.0, 0.25, 0.5, 0.75, 1.0), counts, strict=True):
        levels.extend([level] * count)
    rng = random.Random(int(seed) + int(batch_number) * 104729)
    rng.shuffle(levels)
    return levels


def _interpolate(hard: SpawnCondition, easy: SpawnCondition, ease_level: float) -> SpawnCondition:
    return SpawnCondition(
        hard.x_mm + (easy.x_mm - hard.x_mm) * ease_level,
        hard.z_mm + (easy.z_mm - hard.z_mm) * ease_level,
        hard.speed_mm_s + (easy.speed_mm_s - hard.speed_mm_s) * ease_level,
    )


def current_boundary_condition(
    state: MutableMapping[str, Any], config: BoundaryBandConfig
) -> tuple[SpawnCondition, float]:
    payload = ensure_boundary_state(state, config)
    batch_number = int(payload["batch_number"])
    attempt = int(payload["attempts_in_batch"])
    levels = _shuffled_levels(config.batch_size, config.seed, batch_number)
    level = levels[attempt % len(levels)]
    hard = _condition_from_dict(payload["hard"])
    easy = _condition_from_dict(payload["easy"])
    condition = _interpolate(hard, easy, level)
    state["spawn_x_mm"] = float(condition.x_mm)
    state["spawn_z_mm"] = float(condition.z_mm)
    state["initial_speed_mm_s"] = float(condition.speed_mm_s)
    state["boundary_ease_level"] = float(level)
    state["boundary_batch_number"] = batch_number
    state["target_x_mm"] = float(config.target.x_mm)
    state["target_z_mm"] = float(config.target.z_mm)
    state["target_speed_mm_s"] = float(config.target.speed_mm_s)
    return condition, level


def _harden(condition: SpawnCondition, config: BoundaryBandConfig) -> SpawnCondition:
    return SpawnCondition(
        move_toward(condition.x_mm, config.target.x_mm, config.harden_step.x_mm),
        move_toward(condition.z_mm, config.target.z_mm, config.harden_step.z_mm),
        move_toward(
            condition.speed_mm_s,
            config.target.speed_mm_s,
            config.harden_step.speed_mm_s,
        ),
    )


def _ease(
    condition: SpawnCondition,
    initial: SpawnCondition,
    config: BoundaryBandConfig,
) -> SpawnCondition:
    return SpawnCondition(
        _move_away_toward_cap(
            condition.x_mm, config.target.x_mm, initial.x_mm, config.ease_step.x_mm
        ),
        _move_away_toward_cap(
            condition.z_mm, config.target.z_mm, initial.z_mm, config.ease_step.z_mm
        ),
        _move_away_toward_cap(
            condition.speed_mm_s,
            config.target.speed_mm_s,
            initial.speed_mm_s,
            config.ease_step.speed_mm_s,
        ),
    )


def record_boundary_result(
    state: MutableMapping[str, Any],
    config: BoundaryBandConfig,
    *,
    success: bool,
) -> dict[str, Any] | None:
    payload = ensure_boundary_state(state, config)
    if success:
        state["successful_first_gates"] = int(state.get("successful_first_gates", 0)) + 1
        state["consecutive_failures"] = 0
    else:
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1

    payload["attempts_in_batch"] = int(payload["attempts_in_batch"]) + 1
    if success:
        payload["successes_in_batch"] = int(payload["successes_in_batch"]) + 1

    completed: dict[str, Any] | None = None
    if int(payload["attempts_in_batch"]) >= config.batch_size:
        attempts = int(payload["attempts_in_batch"])
        successes = int(payload["successes_in_batch"])
        rate = successes / attempts
        hard = _condition_from_dict(payload["hard"])
        easy = _condition_from_dict(payload["easy"])
        initial_hard = _condition_from_dict(payload["initial_hard"])
        initial_easy = _condition_from_dict(payload["initial_easy"])

        adjustment = "hold"
        if rate >= config.harden_success_rate:
            hard = _harden(hard, config)
            easy = _harden(easy, config)
            payload["harder_shifts"] = int(payload["harder_shifts"]) + 1
            adjustment = "harder"
        elif rate < config.ease_success_rate:
            hard = _ease(hard, initial_hard, config)
            easy = _ease(easy, initial_easy, config)
            payload["easier_shifts"] = int(payload["easier_shifts"]) + 1
            adjustment = "easier"

        payload["hard"] = _condition_dict(hard)
        payload["easy"] = _condition_dict(easy)
        payload["last_batch_success_rate"] = float(rate)
        payload["last_adjustment"] = adjustment
        payload["batch_number"] = int(payload["batch_number"]) + 1
        payload["attempts_in_batch"] = 0
        payload["successes_in_batch"] = 0
        completed = {
            "successes": successes,
            "attempts": attempts,
            "success_rate": rate,
            "adjustment": adjustment,
            "hard": _condition_dict(hard),
            "easy": _condition_dict(easy),
        }

    hard_now = _condition_from_dict(payload["hard"])
    easy_now = _condition_from_dict(payload["easy"])
    state["curriculum_complete"] = bool(
        all(
            math.isclose(value, target, abs_tol=1e-9)
            for value, target in (
                (hard_now.x_mm, config.target.x_mm),
                (hard_now.z_mm, config.target.z_mm),
                (hard_now.speed_mm_s, config.target.speed_mm_s),
                (easy_now.x_mm, config.target.x_mm),
                (easy_now.z_mm, config.target.z_mm),
                (easy_now.speed_mm_s, config.target.speed_mm_s),
            )
        )
    )
    return completed
