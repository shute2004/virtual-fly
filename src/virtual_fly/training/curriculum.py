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
class GateHeightCurriculumConfig:
    """Progress one gate opening while repeatedly rehearsing mastered heights."""

    gate_index: int
    start_center_z_mm: float
    target_center_z_mm: float
    step_mm: float = 0.25
    batch_size: int = 24
    current_success_rate: float = 0.80
    retention_success_rate: float = 0.80
    acquisition_only: bool = False
    seed: int = 0


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
            "frontier_gate_count": None,
            "frontier_mastered_gates": 0,
            "frontier_target_gates": 1,
            "gate_pass_counts_in_batch": {},
            "group_gate_pass_counts": {},
            "last_batch_gate_pass_rates": {},
            "last_batch_raw_gate_pass_rates": {},
            "frontier_advances": 0,
            "frontier_focus_target_gates": 1,
            "frontier_focus_level": 0.0,
            "frontier_focus_attempts_in_batch": 0,
            "frontier_focus_successes_in_batch": 0,
            "frontier_evaluation_attempts_in_batch": 0,
            "frontier_evaluation_successes_in_batch": 0,
            "last_frontier_focus_success_rate": None,
            "last_frontier_focus_level": 0.0,
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
        payload.setdefault("frontier_gate_count", None)
        payload.setdefault("frontier_mastered_gates", 0)
        payload.setdefault("frontier_target_gates", 1)
        payload.setdefault("gate_pass_counts_in_batch", {})
        payload.setdefault("group_gate_pass_counts", {})
        payload.setdefault("last_batch_gate_pass_rates", {})
        payload.setdefault("last_batch_raw_gate_pass_rates", {})
        payload.setdefault("frontier_advances", 0)
        payload.setdefault("frontier_focus_target_gates", int(payload.get("frontier_target_gates", 1)))
        payload.setdefault("frontier_focus_level", 0.0)
        payload.setdefault("frontier_focus_attempts_in_batch", 0)
        payload.setdefault("frontier_focus_successes_in_batch", 0)
        payload.setdefault("frontier_evaluation_attempts_in_batch", 0)
        payload.setdefault("frontier_evaluation_successes_in_batch", 0)
        payload.setdefault("last_frontier_focus_success_rate", None)
        payload.setdefault("last_frontier_focus_level", float(payload.get("frontier_focus_level", 0.0)))
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


def _group_balanced_levels(
    batch_size: int,
    seed: int,
    batch_number: int,
    group_count: int,
) -> list[float]:
    """Distribute the fixed level multiset as evenly as possible across groups.

    Attempt ``i`` is assigned to group ``i % group_count``.  A global shuffle can
    therefore give different course seeds very different difficulty mixtures even
    when every seed gets the same episode count.  This allocator preserves the
    exact global level counts while keeping each level's per-group count balanced
    as closely as the integer totals allow.
    """

    if group_count < 1:
        raise ValueError("boundary group_count must be >= 1")
    if group_count > batch_size:
        raise ValueError("boundary group_count cannot exceed batch_size")

    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    counts = _level_counts(batch_size)
    target_sizes = [len(range(group, batch_size, group_count)) for group in range(group_count)]
    buckets: list[list[float]] = [[] for _ in range(group_count)]
    per_level = [[0] * len(levels) for _ in range(group_count)]
    loads = [0] * group_count
    rng = random.Random(int(seed) + int(batch_number) * 104729 + int(group_count) * 1009)

    for level_index, (level, count) in enumerate(zip(levels, counts, strict=True)):
        tie_order = list(range(group_count))
        rng.shuffle(tie_order)
        priority = {group: rank for rank, group in enumerate(tie_order)}
        for _ in range(count):
            candidates = [
                group for group in range(group_count) if loads[group] < target_sizes[group]
            ]
            if not candidates:
                raise RuntimeError("boundary level allocator exhausted group capacity")
            min_level_count = min(per_level[group][level_index] for group in candidates)
            candidates = [
                group
                for group in candidates
                if per_level[group][level_index] == min_level_count
            ]
            min_load = min(loads[group] for group in candidates)
            candidates = [group for group in candidates if loads[group] == min_load]
            group = min(candidates, key=priority.__getitem__)
            buckets[group].append(level)
            per_level[group][level_index] += 1
            loads[group] += 1

    if loads != target_sizes:
        raise RuntimeError(
            f"boundary level allocator produced group sizes {loads}, expected {target_sizes}"
        )
    for group, bucket in enumerate(buckets):
        group_rng = random.Random(
            int(seed) + int(batch_number) * 104729 + int(group_count) * 1009 + group * 65537
        )
        group_rng.shuffle(bucket)

    schedule: list[float] = []
    for attempt in range(batch_size):
        group = attempt % group_count
        local_index = attempt // group_count
        schedule.append(buckets[group][local_index])
    return schedule


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
    *,
    group_count: int | None = None,
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
    levels = (
        _shuffled_levels(config.batch_size, config.seed, batch_number)
        if group_count is None
        else _group_balanced_levels(
            config.batch_size,
            config.seed,
            batch_number,
            group_count,
        )
    )
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


def _validate_gate_height_config(config: GateHeightCurriculumConfig) -> None:
    if config.gate_index < 0:
        raise ValueError("gate-height gate_index must be >= 0")
    if config.batch_size < 6:
        raise ValueError("gate-height batch_size must be >= 6")
    values = (config.start_center_z_mm, config.target_center_z_mm, config.step_mm)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gate-height values must be finite")
    if config.step_mm <= 0.0:
        raise ValueError("gate-height step_mm must be > 0")
    if not 0.0 < config.current_success_rate <= 1.0:
        raise ValueError("gate-height current_success_rate must be in (0,1]")
    if not 0.0 < config.retention_success_rate <= 1.0:
        raise ValueError("gate-height retention_success_rate must be in (0,1]")


def ensure_gate_height_state(
    state: MutableMapping[str, Any], config: GateHeightCurriculumConfig
) -> MutableMapping[str, Any]:
    """Return persistent state for the simple current+review height curriculum."""

    _validate_gate_height_config(config)
    payload = state.get("gate_height_curriculum")
    if payload is None:
        payload = {
            "schema_version": 4,
            "gate_index": int(config.gate_index),
            "start_center_z_mm": float(config.start_center_z_mm),
            "target_center_z_mm": float(config.target_center_z_mm),
            "step_mm": float(config.step_mm),
            "batch_number": 0,
            "completed_attempt_indices": [],
            "role_attempts": {"current": 0, "review": 0},
            "role_successes": {"current": 0, "review": 0},
            "frontier_center_z_mm": float(config.start_center_z_mm),
            "mastered_centers_z_mm": [],
            "last_role_success_rates": {},
            "last_adjustment": "initialized",
            "advances": 0,
            "curriculum_complete": False,
        }
        state["gate_height_curriculum"] = payload
    elif not isinstance(payload, dict):
        raise ValueError("gate-height curriculum state must be a JSON object")
    elif int(payload.get("schema_version", 0)) != 4:
        raise ValueError(
            "incompatible gate-height curriculum schema; start a clean fork from the source checkpoint"
        )
    else:
        if int(payload.get("gate_index", -1)) != int(config.gate_index):
            raise ValueError("cannot change gate-height gate_index in an existing curriculum")
        if not math.isclose(
            float(payload.get("target_center_z_mm", config.target_center_z_mm)),
            float(config.target_center_z_mm),
            abs_tol=1e-12,
        ):
            raise ValueError("cannot change gate-height target in an existing curriculum")
        payload.setdefault("completed_attempt_indices", [])
        payload.setdefault("mastered_centers_z_mm", [])
        payload.setdefault("last_role_success_rates", {})
        payload.setdefault("last_adjustment", "initialized")
        payload.setdefault("advances", 0)
        payload.setdefault("curriculum_complete", False)
        for key in ("role_attempts", "role_successes"):
            mapping = payload.setdefault(key, {})
            mapping.setdefault("current", 0)
            mapping.setdefault("review", 0)
    state["gate_height_frontier_center_z_mm"] = float(payload["frontier_center_z_mm"])
    state["gate_height_curriculum_complete"] = bool(payload["curriculum_complete"])
    return payload


def next_gate_height_attempt_for_group(
    state: MutableMapping[str, Any],
    config: GateHeightCurriculumConfig,
    *,
    group_index: int,
    group_count: int,
    issued_attempts: set[int] | frozenset[int] = frozenset(),
) -> int | None:
    if group_count < 1 or group_index < 0 or group_index >= group_count:
        raise ValueError("invalid gate-height group assignment")
    payload = ensure_gate_height_state(state, config)
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


def gate_height_role(
    attempt_index: int, *, group_count: int, recovery_mode: bool = False, acquisition_only: bool = False
) -> str:
    """Return current or review; every episode is an ordinary learning episode.

    In the production 24-attempt/4-slot batch each slot receives four current
    frontier episodes and two review episodes. Group offsets interleave both
    roles so hard/current episodes are not issued as a single block.
    """

    del recovery_mode  # retained only for call-site compatibility
    attempt = int(attempt_index)
    groups = int(group_count)
    if attempt < 0 or groups < 1:
        raise ValueError("invalid gate-height role arguments")
    if acquisition_only:
        return "current"
    group = attempt % groups
    local_index = attempt // groups
    pattern = ("current", "review", "current", "current", "review", "current")
    return pattern[(local_index + group) % len(pattern)]


def _gate_height_step_toward(current: float, target: float, step: float) -> float:
    return move_toward(float(current), float(target), float(step))


def _gate_height_role_ordinal(
    attempt_index: int, *, group_count: int, role: str
) -> int:
    return sum(
        gate_height_role(i, group_count=group_count) == role
        for i in range(int(attempt_index))
    )


def gate_height_condition_for_attempt(
    state: MutableMapping[str, Any],
    config: GateHeightCurriculumConfig,
    attempt_index: int,
    *,
    group_count: int,
) -> tuple[float, str]:
    payload = ensure_gate_height_state(state, config)
    attempt = int(attempt_index)
    if attempt < 0 or attempt >= config.batch_size:
        raise ValueError("gate-height attempt_index outside current batch")
    role = gate_height_role(
        attempt, group_count=group_count, acquisition_only=bool(config.acquisition_only)
    )
    frontier = float(payload["frontier_center_z_mm"])
    if role == "current":
        center = frontier
    else:
        mastered = [float(value) for value in payload.get("mastered_centers_z_mm", [])]
        pool = mastered if mastered else [float(config.start_center_z_mm)]
        ordinal = _gate_height_role_ordinal(
            attempt,
            group_count=group_count,
            role=role,
        )
        batch = int(payload.get("batch_number", 0))
        center = pool[(batch * 8 + ordinal) % len(pool)]
    state["gate_height_episode_center_z_mm"] = float(center)
    state["gate_height_episode_role"] = role
    return float(center), role


def record_gate_height_result(
    state: MutableMapping[str, Any],
    config: GateHeightCurriculumConfig,
    *,
    success: bool,
    attempt_index: int,
    role: str,
) -> dict[str, Any] | None:
    """Use the same learning episodes for both training and progression checks."""

    payload = ensure_gate_height_state(state, config)
    role = str(role)
    roles = ("current", "review")
    if role not in roles:
        raise ValueError("gate-height role must be current or review")
    completed = {int(value) for value in payload.get("completed_attempt_indices", [])}
    attempt = int(attempt_index)
    if attempt < 0 or attempt >= config.batch_size or attempt in completed:
        raise ValueError(f"invalid or duplicate gate-height attempt {attempt}")
    completed.add(attempt)
    payload["completed_attempt_indices"] = sorted(completed)
    attempts = dict(payload.get("role_attempts", {}))
    successes = dict(payload.get("role_successes", {}))
    attempts[role] = int(attempts.get(role, 0)) + 1
    successes[role] = int(successes.get(role, 0)) + int(bool(success))
    payload["role_attempts"] = attempts
    payload["role_successes"] = successes
    if len(completed) < config.batch_size:
        return None

    rates = {
        key: (
            int(successes.get(key, 0)) / int(attempts.get(key, 0))
            if int(attempts.get(key, 0)) > 0
            else 0.0
        )
        for key in roles
    }
    current_rate = rates["current"]
    review_rate = rates["review"]
    before = float(payload["frontier_center_z_mm"])
    after = before
    adjustment = "hold"
    if current_rate >= config.current_success_rate:
        mastered = [float(value) for value in payload.get("mastered_centers_z_mm", [])]
        if not any(math.isclose(before, value, abs_tol=1e-12) for value in mastered):
            mastered.append(before)
            mastered.sort()
        payload["mastered_centers_z_mm"] = mastered
        if math.isclose(before, config.target_center_z_mm, abs_tol=1e-12):
            payload["curriculum_complete"] = True
            adjustment = "complete"
        else:
            after = _gate_height_step_toward(before, config.target_center_z_mm, config.step_mm)
            payload["frontier_center_z_mm"] = after
            payload["advances"] = int(payload.get("advances", 0)) + 1
            adjustment = "advance"

    payload["last_role_success_rates"] = rates
    payload["last_adjustment"] = adjustment
    completed_batch = {
        "batch_number": int(payload.get("batch_number", 0)),
        "frontier_center_before_z_mm": before,
        "frontier_center_after_z_mm": after,
        "role_success_rates": rates,
        "adjustment": adjustment,
        "mastered_centers_z_mm": list(payload.get("mastered_centers_z_mm", [])),
        "curriculum_complete": bool(payload.get("curriculum_complete", False)),
    }
    payload["batch_number"] = int(payload.get("batch_number", 0)) + 1
    payload["completed_attempt_indices"] = []
    payload["role_attempts"] = {role: 0 for role in roles}
    payload["role_successes"] = {role: 0 for role in roles}
    state["gate_height_frontier_center_z_mm"] = float(payload["frontier_center_z_mm"])
    state["gate_height_curriculum_complete"] = bool(payload["curriculum_complete"])
    return completed_batch


def boundary_frontier_role(attempt_index: int, *, group_count: int) -> str:
    """Return a completion-order-independent focus/evaluation role.

    Boundary attempts are statically assigned to groups by ``attempt % group_count``.
    Alternating each group's local attempt index gives every course seed the same
    number of focus and full-course evaluation episodes in the usual 24/4 batch.
    """

    attempt = int(attempt_index)
    groups = int(group_count)
    if attempt < 0:
        raise ValueError("attempt_index must be >= 0")
    if groups < 1:
        raise ValueError("group_count must be >= 1")
    local_index = attempt // groups
    return "focus" if local_index % 2 == 0 else "evaluate"


def frontier_focus_level(
    state: MutableMapping[str, Any], config: BoundaryBandConfig, *, gate_count: int
) -> float:
    target = boundary_target_gates(state, config, gate_count=gate_count)
    payload = ensure_boundary_state(state, config)
    if int(payload.get("frontier_focus_target_gates", target)) != target:
        payload["frontier_focus_target_gates"] = target
        payload["frontier_focus_level"] = 0.0
    return min(1.0, max(0.0, float(payload.get("frontier_focus_level", 0.0))))


def frontier_focus_condition(
    base: SpawnCondition,
    *,
    target_gate_x_mm: float,
    target_gate_z_mm: float,
    previous_gate_x_mm: float | None,
    previous_gate_z_mm: float | None,
    previous_gate_half_gap_mm: float | None = None,
    focus_level: float,
    near_gate_distance_mm: float = 2.5,
    easy_speed_mm_s: float = 450.0,
    clearance_margin_mm: float = 1.0,
) -> SpawnCondition:
    """Build one local success-experience reset for the current frontier gate.

    Level 0 begins close to the target opening centre, making an initial reward
    experience likely without injecting any action. Level 1 spans the actual
    inter-gate transition: for gate 1 it returns to the normal boundary reset;
    for later gates it starts just after the previous wall at the safe edge of
    the previous opening nearest the target opening. This matches the v6 course
    reachability criterion instead of creating an artificially harder reset.
    Full-course mastery is measured separately on evaluation episodes, never
    from this focused reset.
    """

    level = float(focus_level)
    if not 0.0 <= level <= 1.0 or not math.isfinite(level):
        raise ValueError("focus_level must be finite and in [0,1]")
    if near_gate_distance_mm <= 0.0 or not math.isfinite(near_gate_distance_mm):
        raise ValueError("near_gate_distance_mm must be finite and > 0")
    if clearance_margin_mm < 0.0 or not math.isfinite(clearance_margin_mm):
        raise ValueError("clearance_margin_mm must be finite and >= 0")
    values = (
        base.x_mm,
        base.z_mm,
        base.speed_mm_s,
        target_gate_x_mm,
        target_gate_z_mm,
        easy_speed_mm_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("frontier focus condition values must be finite")

    easy = SpawnCondition(
        float(target_gate_x_mm) - float(near_gate_distance_mm),
        float(target_gate_z_mm),
        max(float(base.speed_mm_s), float(easy_speed_mm_s)),
    )
    if previous_gate_x_mm is None or previous_gate_z_mm is None:
        hard = base
    else:
        if not all(
            math.isfinite(float(value))
            for value in (previous_gate_x_mm, previous_gate_z_mm)
        ):
            raise ValueError("previous gate focus values must be finite")
        previous_start_z = float(previous_gate_z_mm)
        if previous_gate_half_gap_mm is not None:
            half_gap = float(previous_gate_half_gap_mm)
            if not math.isfinite(half_gap) or half_gap <= clearance_margin_mm:
                raise ValueError(
                    "previous gate half-gap must exceed the clearance margin"
                )
            safe_half_gap = half_gap - float(clearance_margin_mm)
            delta = float(target_gate_z_mm) - float(previous_gate_z_mm)
            if delta > 0.0:
                previous_start_z += safe_half_gap
            elif delta < 0.0:
                previous_start_z -= safe_half_gap
        hard = SpawnCondition(
            float(previous_gate_x_mm) + float(near_gate_distance_mm),
            previous_start_z,
            float(base.speed_mm_s),
        )
    return SpawnCondition(
        easy.x_mm + level * (hard.x_mm - easy.x_mm),
        easy.z_mm + level * (hard.z_mm - easy.z_mm),
        easy.speed_mm_s + level * (hard.speed_mm_s - easy.speed_mm_s),
    )


def boundary_target_gates(
    state: MutableMapping[str, Any],
    config: BoundaryBandConfig,
    *,
    gate_count: int,
) -> int:
    """Return the next gate frontier target for an arbitrary-length course.

    The frontier is monotonic: after at least ``k`` gates are passed reliably in
    one complete boundary batch, ``k`` becomes mastered and the next target is
    ``k + 1``.  Once the full course is mastered the target remains the full
    gate count so spawn difficulty can continue to harden independently.
    """

    if gate_count < 1:
        raise ValueError("gate_count must be >= 1")
    payload = ensure_boundary_state(state, config)
    existing = payload.get("frontier_gate_count")
    if existing is not None and int(existing) != int(gate_count):
        if int(payload.get("attempts_in_batch", 0)) != 0:
            raise ValueError("cannot change gate_count during an active boundary batch")
        payload["frontier_mastered_gates"] = 0
        payload["frontier_target_gates"] = 1
        payload["frontier_focus_target_gates"] = 1
        payload["frontier_focus_level"] = 0.0
        payload["gate_pass_counts_in_batch"] = {}
        payload["group_gate_pass_counts"] = {}
    payload["frontier_gate_count"] = int(gate_count)
    mastered = max(0, min(int(payload.get("frontier_mastered_gates", 0)), gate_count))
    target = gate_count if mastered >= gate_count else mastered + 1
    payload["frontier_mastered_gates"] = mastered
    payload["frontier_target_gates"] = target
    state["gate_frontier_mastered_gates"] = mastered
    state["gate_frontier_target_gates"] = target
    return target


def record_boundary_result(
    state: MutableMapping[str, Any],
    config: BoundaryBandConfig,
    *,
    success: bool | None = None,
    passed_gates: int | None = None,
    gate_count: int | None = None,
    attempt_index: int | None = None,
    group: int | str | None = None,
    frontier_role: str = "evaluate",
) -> dict[str, Any] | None:
    payload = ensure_boundary_state(state, config)
    frontier_mode = passed_gates is not None or gate_count is not None
    role = str(frontier_role)
    if role not in {"focus", "evaluate"}:
        raise ValueError("frontier_role must be focus or evaluate")
    target_gates = None
    if frontier_mode:
        if passed_gates is None or gate_count is None:
            raise ValueError("passed_gates and gate_count must be provided together")
        passed_gates = int(passed_gates)
        gate_count = int(gate_count)
        if gate_count < 1 or passed_gates < 0 or passed_gates > gate_count:
            raise ValueError("invalid passed_gates/gate_count")
        target_gates = boundary_target_gates(state, config, gate_count=gate_count)
        success = passed_gates >= (1 if role == "focus" else target_gates)
    elif success is None:
        raise ValueError("success is required when gate frontier data is not provided")
    success = bool(success)

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

    if frontier_mode:
        first_gate_success = (
            role == "evaluate"
            and passed_gates is not None
            and passed_gates >= 1
        )
    else:
        first_gate_success = success
    if first_gate_success:
        state["successful_first_gates"] = int(state.get("successful_first_gates", 0)) + 1
    state["consecutive_failures"] = 0

    payload["completed_attempt_indices"] = sorted(completed_indices)
    payload["attempts_in_batch"] = len(completed_indices)
    if success:
        payload["successes_in_batch"] = int(payload["successes_in_batch"]) + 1

    count_for_evaluation = not frontier_mode or role == "evaluate"
    if frontier_mode:
        if role == "focus":
            payload["frontier_focus_attempts_in_batch"] = int(
                payload.get("frontier_focus_attempts_in_batch", 0)
            ) + 1
            if success:
                payload["frontier_focus_successes_in_batch"] = int(
                    payload.get("frontier_focus_successes_in_batch", 0)
                ) + 1
        else:
            payload["frontier_evaluation_attempts_in_batch"] = int(
                payload.get("frontier_evaluation_attempts_in_batch", 0)
            ) + 1
            if success:
                payload["frontier_evaluation_successes_in_batch"] = int(
                    payload.get("frontier_evaluation_successes_in_batch", 0)
                ) + 1

    if group is not None and count_for_evaluation:
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

    if (
        frontier_mode
        and role == "evaluate"
        and gate_count is not None
        and passed_gates is not None
    ):
        pass_counts = dict(payload.get("gate_pass_counts_in_batch", {}))
        for gate_index in range(1, gate_count + 1):
            key = str(gate_index)
            pass_counts[key] = int(pass_counts.get(key, 0)) + int(passed_gates >= gate_index)
        payload["gate_pass_counts_in_batch"] = pass_counts
        if group is not None:
            gkey = str(group)
            grouped = dict(payload.get("group_gate_pass_counts", {}))
            group_counts = dict(grouped.get(gkey, {}))
            for gate_index in range(1, gate_count + 1):
                key = str(gate_index)
                group_counts[key] = int(group_counts.get(key, 0)) + int(passed_gates >= gate_index)
            grouped[gkey] = group_counts
            payload["group_gate_pass_counts"] = grouped

    completed: dict[str, Any] | None = None
    if int(payload["attempts_in_batch"]) >= config.batch_size:
        total_attempts = int(payload["attempts_in_batch"])
        if frontier_mode:
            attempts = int(payload.get("frontier_evaluation_attempts_in_batch", 0))
            successes = int(payload.get("frontier_evaluation_successes_in_batch", 0))
            if attempts < 1:
                raise RuntimeError("frontier boundary batch completed without evaluation episodes")
        else:
            attempts = total_attempts
            successes = int(payload["successes_in_batch"])
        raw_rate = successes / attempts
        focus_attempts = int(payload.get("frontier_focus_attempts_in_batch", 0)) if frontier_mode else 0
        focus_successes = int(payload.get("frontier_focus_successes_in_batch", 0)) if frontier_mode else 0
        focus_rate = (focus_successes / focus_attempts) if focus_attempts else None
        focus_level_before = float(payload.get("frontier_focus_level", 0.0))
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

        raw_gate_rates: dict[str, float] = {}
        gate_rates: dict[str, float] = {}
        frontier_before = int(payload.get("frontier_mastered_gates", 0))
        frontier_after = frontier_before
        evaluated_target = target_gates
        if frontier_mode and gate_count is not None:
            pass_counts = dict(payload.get("gate_pass_counts_in_batch", {}))
            grouped_counts = dict(payload.get("group_gate_pass_counts", {}))
            for gate_index in range(1, gate_count + 1):
                key = str(gate_index)
                raw_gate_rates[key] = int(pass_counts.get(key, 0)) / attempts
                if group_attempts:
                    per_group = [
                        int(dict(grouped_counts.get(group_key, {})).get(key, 0)) / count
                        for group_key, count in sorted(group_attempts.items())
                    ]
                    gate_rates[key] = sum(per_group) / len(per_group)
                else:
                    gate_rates[key] = raw_gate_rates[key]
            qualified = [
                gate_index
                for gate_index in range(1, gate_count + 1)
                if gate_rates[str(gate_index)] >= config.harden_success_rate
            ]
            if qualified:
                frontier_after = max(frontier_before, max(qualified))
            frontier_after = min(frontier_after, gate_count)
            payload["frontier_mastered_gates"] = frontier_after
            payload["frontier_target_gates"] = (
                gate_count if frontier_after >= gate_count else frontier_after + 1
            )
            payload["last_batch_gate_pass_rates"] = gate_rates
            payload["last_batch_raw_gate_pass_rates"] = raw_gate_rates
            if frontier_after > frontier_before:
                payload["frontier_advances"] = int(payload.get("frontier_advances", 0)) + 1
            state["gate_frontier_mastered_gates"] = frontier_after
            state["gate_frontier_target_gates"] = int(payload["frontier_target_gates"])

            next_target = int(payload["frontier_target_gates"])
            if frontier_after > frontier_before:
                payload["frontier_focus_target_gates"] = next_target
                payload["frontier_focus_level"] = 0.0
            elif focus_rate is not None:
                level = focus_level_before
                if focus_rate >= config.harden_success_rate:
                    level = min(1.0, level + 0.25)
                elif focus_rate < config.ease_success_rate:
                    level = max(0.0, level - 0.125)
                payload["frontier_focus_target_gates"] = next_target
                payload["frontier_focus_level"] = level
            payload["last_frontier_focus_success_rate"] = focus_rate
            payload["last_frontier_focus_level"] = float(
                payload.get("frontier_focus_level", focus_level_before)
            )

        hard = _condition_from_dict(payload["hard"])
        easy = _condition_from_dict(payload["easy"])
        recovery_hard = _condition_from_dict(payload["recovery_hard"])
        recovery_easy = _condition_from_dict(payload["recovery_easy"])

        adjustment = "hold"
        if frontier_mode and frontier_after > frontier_before:
            adjustment = "frontier_advanced"
        elif rate >= config.harden_success_rate:
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
        payload["gate_pass_counts_in_batch"] = {}
        payload["group_gate_pass_counts"] = {}
        payload["frontier_focus_attempts_in_batch"] = 0
        payload["frontier_focus_successes_in_batch"] = 0
        payload["frontier_evaluation_attempts_in_batch"] = 0
        payload["frontier_evaluation_successes_in_batch"] = 0
        state["boundary_batch_number"] = int(payload["batch_number"])
        state["boundary_ease_level"] = None
        completed = {
            "successes": successes,
            "attempts": attempts,
            "total_attempts": total_attempts,
            "success_rate": rate,
            "raw_success_rate": raw_rate,
            "focus_attempts": focus_attempts,
            "focus_successes": focus_successes,
            "focus_success_rate": focus_rate,
            "focus_level_before": focus_level_before if frontier_mode else None,
            "focus_level_after": float(payload.get("frontier_focus_level", 0.0)) if frontier_mode else None,
            "group_success_rates": group_rates,
            "aggregation": aggregation,
            "adjustment": adjustment,
            "hard": _condition_dict(hard),
            "easy": _condition_dict(easy),
            "evaluated_target_gates": evaluated_target,
            "frontier_mastered_gates": frontier_after if frontier_mode else None,
            "frontier_target_gates": int(payload.get("frontier_target_gates", 1)) if frontier_mode else None,
            "gate_pass_rates": gate_rates,
            "raw_gate_pass_rates": raw_gate_rates,
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
