"""Curriculum policies for Flyppy training.

The neural learning rule lives in the CNS runtime. Curriculum code may only
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


def _recovery_caps(config: BoundaryBandConfig) -> tuple[SpawnCondition, SpawnCondition]:
    """Return one-band-easier caps for recovery after a poor batch.

    ``recovery_hard`` is the initial easy endpoint. ``recovery_easy`` extends
    the initial band by the same width once more. For the measured gate-2 band
    this gives an easy recovery endpoint near the previously observed
    x=10.75/z=5.415/vx=362.5 success condition.
    """

    recovery_hard = config.easy
    recovery_easy = SpawnCondition(
        config.easy.x_mm + (config.easy.x_mm - config.hard.x_mm),
        config.easy.z_mm + (config.easy.z_mm - config.hard.z_mm),
        config.easy.speed_mm_s + (config.easy.speed_mm_s - config.hard.speed_mm_s),
    )
    return recovery_hard, recovery_easy


def _validate_boundary(config: BoundaryBandConfig) -> None:
    if config.batch_size < 5:
        raise ValueError("boundary batch_size must be >= 5")
    if not 0.0 <= config.ease_success_rate < config.harden_success_rate <= 1.0:
        raise ValueError("boundary success-rate thresholds are invalid")
    recovery_hard, recovery_easy = _recovery_caps(config)
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
        recovery_hard.x_mm,
        recovery_hard.z_mm,
        recovery_hard.speed_mm_s,
        recovery_easy.x_mm,
        recovery_easy.z_mm,
        recovery_easy.speed_mm_s,
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
    recovery_hard, recovery_easy = _recovery_caps(config)
    payload = state.get("boundary_band")
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        payload = {
            "schema_version": 1,
            "initial_hard": _condition_dict(config.hard),
            "initial_easy": _condition_dict(config.easy),
            "recovery_hard": _condition_dict(recovery_hard),
            "recovery_easy": _condition_dict(recovery_easy),
            "hard": _condition_dict(config.hard),
            "easy": _condition_dict(config.easy),
            "batch_number": 0,
            "attempts_in_batch": 0,
            "successes_in_batch": 0,
            "completed_attempt_indices": [],
            "group_attempts": {},
            "group_successes": {},
            "last_batch_success_rate": None,
            "last_batch_raw_success_rate": None,
            "last_batch_group_success_rates": {},
            "last_batch_aggregation": "episode_mean",
            "last_adjustment": "initialized",
            "harder_shifts": 0,
            "easier_shifts": 0,
        }
        state["boundary_band"] = payload
    else:
        # Schema 1 existed briefly before explicit recovery caps were added.
        # Preserve any progress while adding the new fail-safe range.
        payload.setdefault("recovery_hard", _condition_dict(recovery_hard))
        payload.setdefault("recovery_easy", _condition_dict(recovery_easy))
        # Historical sequential boundary-band states only stored a count.  Their
        # consumed attempts were necessarily the prefix [0, attempts), so that
        # prefix is an exact migration to explicit attempt identities.
        payload.setdefault(
            "completed_attempt_indices",
            list(range(int(payload.get("attempts_in_batch", 0)))),
        )
        payload.setdefault("group_attempts", {})
        payload.setdefault("group_successes", {})
        payload.setdefault("last_batch_raw_success_rate", payload.get("last_batch_success_rate"))
        payload.setdefault("last_batch_group_success_rates", {})
        payload.setdefault("last_batch_aggregation", "episode_mean")
    return payload


def _level_counts(batch_size: int) -> list[int]:
    # Target proportions mirror the desired 24-episode distribution 4/5/6/5/4.
    weights = (4, 5, 6, 5, 4)
    raw = [batch_size * weight / sum(weights) for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remaining = batch_size - sum(counts)
    order = sorted(
        range(5),
        key=lambda i: (raw[i] - counts[i], -abs(i - 2)),
        reverse=True,
    )
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


def boundary_condition_for_attempt(
    state: MutableMapping[str, Any],
    config: BoundaryBandConfig,
    attempt_index: int,
) -> tuple[SpawnCondition, float]:
    """Return one fixed condition from the current boundary batch.

    ``attempt_index`` is a launch index, not a completion counter.  Keeping the
    two concepts separate lets an asynchronous population launch the complete
    batch up front while ``record_boundary_result`` later receives outcomes in
    any completion order.  The band itself is adjusted only after all outcomes
    in the batch have been recorded.
    """

    payload = ensure_boundary_state(state, config)
    attempt = int(attempt_index)
    if attempt < 0 or attempt >= config.batch_size:
        raise ValueError(
            f"boundary attempt_index must be in [0, {config.batch_size - 1}], got {attempt}"
        )
    batch_number = int(payload["batch_number"])
    levels = _shuffled_levels(config.batch_size, config.seed, batch_number)
    level = levels[attempt]
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


def next_boundary_attempt_for_group(
    state: MutableMapping[str, Any],
    config: BoundaryBandConfig,
    *,
    group_index: int,
    group_count: int,
    issued_attempts: set[int] | frozenset[int] = frozenset(),
) -> int | None:
    """Return the next unconsumed attempt statically assigned to one async group.

    Attempt ``i`` belongs to group ``i % group_count``.  This keeps the
    condition-to-course-seed mapping independent of which asynchronous slot
    finishes first.  ``issued_attempts`` covers in-flight episodes that have not
    yet reached ``record_boundary_result``.
    """

    if group_count < 1:
        raise ValueError("boundary group_count must be >= 1")
    if group_index < 0 or group_index >= group_count:
        raise ValueError(
            f"boundary group_index must be in [0, {group_count - 1}], got {group_index}"
        )
    payload = ensure_boundary_state(state, config)
    completed = {int(value) for value in payload.get("completed_attempt_indices", [])}
    issued = {int(value) for value in issued_attempts}
    return next(
        (
            attempt
            for attempt in range(group_index, config.batch_size, group_count)
            if attempt not in completed and attempt not in issued
        ),
        None,
    )


def current_boundary_condition(
    state: MutableMapping[str, Any], config: BoundaryBandConfig
) -> tuple[SpawnCondition, float]:
    payload = ensure_boundary_state(state, config)
    completed = {int(value) for value in payload.get("completed_attempt_indices", [])}
    attempt = next(
        (index for index in range(config.batch_size) if index not in completed),
        config.batch_size,
    )
    if attempt >= config.batch_size:
        raise RuntimeError("boundary batch has no remaining condition before adjustment")
    return boundary_condition_for_attempt(state, config, attempt)


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
    cap: SpawnCondition,
    config: BoundaryBandConfig,
) -> SpawnCondition:
    return SpawnCondition(
        _move_away_toward_cap(
            condition.x_mm, config.target.x_mm, cap.x_mm, config.ease_step.x_mm
        ),
        _move_away_toward_cap(
            condition.z_mm, config.target.z_mm, cap.z_mm, config.ease_step.z_mm
        ),
        _move_away_toward_cap(
            condition.speed_mm_s,
            config.target.speed_mm_s,
            cap.speed_mm_s,
            config.ease_step.speed_mm_s,
        ),
    )


def record_boundary_result(
    state: MutableMapping[str, Any],
    config: BoundaryBandConfig,
    *,
    success: bool,
    attempt_index: int | None = None,
    group: int | str | None = None,
) -> dict[str, Any] | None:
    payload = ensure_boundary_state(state, config)
    completed_indices = {
        int(value) for value in payload.get("completed_attempt_indices", [])
    }
    if attempt_index is None:
        attempt = next(
            (index for index in range(config.batch_size) if index not in completed_indices),
            config.batch_size,
        )
    else:
        attempt = int(attempt_index)
    if attempt < 0 or attempt >= config.batch_size:
        raise ValueError(
            f"boundary attempt_index must be in [0, {config.batch_size - 1}], got {attempt}"
        )
    if attempt in completed_indices:
        raise ValueError(f"boundary attempt {attempt} was already recorded")
    completed_indices.add(attempt)

    if success:
        state["successful_first_gates"] = int(state.get("successful_first_gates", 0)) + 1
    # ``consecutive_failures`` belongs to the historical per-episode adaptive
    # policy.  In an asynchronous population its value would depend on worker
    # completion order, so boundary-band deliberately keeps it neutral and uses
    # only commutative batch counters for difficulty updates.
    state["consecutive_failures"] = 0

    payload["completed_attempt_indices"] = sorted(completed_indices)
    payload["attempts_in_batch"] = len(completed_indices)
    if success:
        payload["successes_in_batch"] = int(payload["successes_in_batch"]) + 1
    if group is not None:
        key = str(group)
        group_attempts = dict(payload.get("group_attempts", {}))
        group_successes = dict(payload.get("group_successes", {}))
        group_attempts[key] = int(group_attempts.get(key, 0)) + 1
        if success:
            group_successes[key] = int(group_successes.get(key, 0)) + 1
        else:
            group_successes.setdefault(key, int(group_successes.get(key, 0)))
        payload["group_attempts"] = group_attempts
        payload["group_successes"] = group_successes

    completed: dict[str, Any] | None = None
    if int(payload["attempts_in_batch"]) >= config.batch_size:
        attempts = int(payload["attempts_in_batch"])
        successes = int(payload["successes_in_batch"])
        raw_rate = successes / attempts
        group_attempts = {
            str(key): int(value)
            for key, value in dict(payload.get("group_attempts", {})).items()
            if int(value) > 0
        }
        group_successes = {
            str(key): int(value)
            for key, value in dict(payload.get("group_successes", {})).items()
        }
        group_rates = {
            key: group_successes.get(key, 0) / count
            for key, count in sorted(group_attempts.items())
        }
        if group_rates:
            rate = sum(group_rates.values()) / len(group_rates)
            aggregation = "equal_group_mean"
        else:
            rate = raw_rate
            aggregation = "episode_mean"
        hard = _condition_from_dict(payload["hard"])
        easy = _condition_from_dict(payload["easy"])
        recovery_hard = _condition_from_dict(payload["recovery_hard"])
        recovery_easy = _condition_from_dict(payload["recovery_easy"])

        adjustment = "hold"
        if rate >= config.harden_success_rate:
            hard = _harden(hard, config)
            easy = _harden(easy, config)
            payload["harder_shifts"] = int(payload["harder_shifts"]) + 1
            adjustment = "harder"
        elif rate < config.ease_success_rate:
            hard = _ease(hard, recovery_hard, config)
            easy = _ease(easy, recovery_easy, config)
            payload["easier_shifts"] = int(payload["easier_shifts"]) + 1
            adjustment = "easier"

        payload["hard"] = _condition_dict(hard)
        payload["easy"] = _condition_dict(easy)
        payload["last_batch_success_rate"] = float(rate)
        payload["last_batch_raw_success_rate"] = float(raw_rate)
        payload["last_batch_group_success_rates"] = group_rates
        payload["last_batch_aggregation"] = aggregation
        payload["last_adjustment"] = adjustment
        payload["batch_number"] = int(payload["batch_number"]) + 1
        payload["attempts_in_batch"] = 0
        payload["successes_in_batch"] = 0
        payload["completed_attempt_indices"] = []
        payload["group_attempts"] = {}
        payload["group_successes"] = {}
        state["boundary_batch_number"] = int(payload["batch_number"])
        state["boundary_ease_level"] = None
        completed = {
            "successes": successes,
            "attempts": attempts,
            "success_rate": rate,
            "raw_success_rate": raw_rate,
            "group_success_rates": group_rates,
            "aggregation": aggregation,
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
