"""Gate-height acquisition/review curriculum policy."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, MutableMapping

from virtual_fly.training.curriculum_core import move_toward

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


def _validate_gate_height_config(config: GateHeightCurriculumConfig) -> None:
    if config.gate_index < 0:
        raise ValueError("gate-height gate_index must be >= 0")
    minimum_batch_size = 1 if config.acquisition_only else 6
    if config.batch_size < minimum_batch_size:
        raise ValueError(
            f"gate-height batch_size must be >= {minimum_batch_size}"
        )
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
