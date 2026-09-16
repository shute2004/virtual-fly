from __future__ import annotations

from typing import Any, Mapping


VALID_LAUNCH_MODES = frozenset({"async", "wave"})


def validate_resume_launch_mode(
    population_state: Mapping[str, Any],
    requested_launch_mode: str,
) -> str:
    if requested_launch_mode not in VALID_LAUNCH_MODES:
        raise ValueError(f"unknown launch mode: {requested_launch_mode}")
    stored_launch_mode = str(population_state.get("launch_mode", "async"))
    if stored_launch_mode not in VALID_LAUNCH_MODES:
        raise ValueError(f"stored experiment launch mode is invalid: {stored_launch_mode}")
    if stored_launch_mode != requested_launch_mode:
        raise ValueError(
            "cannot change launch mode while resuming the same experiment: "
            f"stored={stored_launch_mode} requested={requested_launch_mode}; "
            "fork the experiment first"
        )
    return stored_launch_mode


def launch_round_for_index(launched_index: int, population: int) -> int:
    """Return the population launch round for one zero-based launch index."""

    if launched_index < 0:
        raise ValueError("launched_index must be >= 0")
    if population < 1:
        raise ValueError("population must be >= 1")
    return launched_index // population


def checkpoint_can_flush(
    *,
    launch_mode: str,
    checkpoint_pending: bool,
    active_slots: int,
) -> bool:
    """Return whether a pending periodic checkpoint may be persisted now.

    Async mode may checkpoint while other slots are still running; orphaned
    active trajectories are reconciled on resume. Wave mode instead waits until
    the whole launch round has terminated so a persisted state never represents
    only a prefix of one synchronized round.
    """

    if launch_mode not in VALID_LAUNCH_MODES:
        raise ValueError(f"unknown launch mode: {launch_mode}")
    if active_slots < 0:
        raise ValueError("active_slots must be >= 0")
    if not checkpoint_pending:
        return False
    return launch_mode == "async" or active_slots == 0
