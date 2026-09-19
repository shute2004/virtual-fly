"""Episode-local physical slot state for process/packed population training.

This module is intentionally free of curriculum scheduling and shared-weight commit
logic. It only maps one scheduled episode onto one physical worker and serializes
its result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from virtual_fly.embodiment.config import FlyppyBodyConfig
from virtual_fly.runtime.body_worker import FlyppyBodyProcess
from virtual_fly.training.curriculum import SpawnCondition

@dataclass
class ProcessSlotState:
    slot: int
    active: bool = False
    episode: int = -1
    source_weight_version: int = 0
    spawn_x_mm: float = 0.0
    spawn_z_mm: float = 0.0
    initial_speed_mm_s: float = 0.0
    initial_vz_mm_s: float = 0.0
    control_steps: int = 0
    passed_gates: int = 0
    collision: bool = False
    collision_reason: str | None = None
    finished: bool = False
    max_x_mm: float = float("-inf")
    min_z_mm: float = float("inf")
    max_z_mm: float = float("-inf")
    final_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reward_events: int = 0
    aversive_events: int = 0
    last_retinal: dict[str, Any] | None = None
    last_motor: dict[str, Any] | None = None
    last_body_spikes: dict[int, bool] | None = None
    boundary_ease_level: float | None = None
    boundary_attempt_index: int | None = None
    boundary_batch_number: int | None = None
    frontier_role: str = "evaluate"
    frontier_target_gate: int | None = None
    course_start_gate_index: int | None = None
    gate_height_role: str | None = None
    gate_height_center_z_mm: float | None = None
    gate_height_attempt_index: int | None = None
    gate_height_batch_number: int | None = None
    gate_height_learning: bool | None = None
    launch_round: int = 0


def worker_config(args) -> dict[str, object]:
    return FlyppyBodyConfig.from_namespace(args).to_worker_mapping()


def reset_slot(
    slot: ProcessSlotState,
    worker: FlyppyBodyProcess,
    *,
    episode: int,
    source_weight_version: int,
    condition: SpawnCondition,
    initial_vz_mm_s: float = 0.0,
    boundary_ease_level: float | None = None,
    boundary_attempt_index: int | None = None,
    boundary_batch_number: int | None = None,
    frontier_role: str = "evaluate",
    frontier_target_gate: int | None = None,
    course_start_gate_index: int | None = None,
    gate_height_role: str | None = None,
    gate_height_center_z_mm: float | None = None,
    gate_height_attempt_index: int | None = None,
    gate_height_batch_number: int | None = None,
    gate_height_learning: bool | None = None,
    gate_center_overrides: dict[int, float] | None = None,
    launch_round: int = 0,
) -> None:
    slot.active = True
    slot.episode = int(episode)
    slot.source_weight_version = int(source_weight_version)
    slot.spawn_x_mm = float(condition.x_mm)
    slot.spawn_z_mm = float(condition.z_mm)
    slot.initial_speed_mm_s = float(condition.speed_mm_s)
    slot.initial_vz_mm_s = float(initial_vz_mm_s)
    slot.control_steps = 0
    slot.passed_gates = 0
    slot.collision = False
    slot.collision_reason = None
    slot.finished = False
    slot.max_x_mm = float("-inf")
    slot.min_z_mm = float("inf")
    slot.max_z_mm = float("-inf")
    slot.reward_events = 0
    slot.aversive_events = 0
    slot.boundary_ease_level = boundary_ease_level
    slot.boundary_attempt_index = boundary_attempt_index
    slot.boundary_batch_number = boundary_batch_number
    slot.frontier_role = str(frontier_role)
    slot.frontier_target_gate = frontier_target_gate
    slot.course_start_gate_index = (
        None if course_start_gate_index is None else int(course_start_gate_index)
    )
    slot.gate_height_role = None if gate_height_role is None else str(gate_height_role)
    slot.gate_height_center_z_mm = None if gate_height_center_z_mm is None else float(gate_height_center_z_mm)
    slot.gate_height_attempt_index = None if gate_height_attempt_index is None else int(gate_height_attempt_index)
    slot.gate_height_batch_number = None if gate_height_batch_number is None else int(gate_height_batch_number)
    slot.gate_height_learning = None if gate_height_learning is None else bool(gate_height_learning)
    slot.launch_round = int(launch_round)
    slot.last_retinal = None
    slot.last_motor = None
    slot.last_body_spikes = {}
    initial = worker.reset(
        episode=slot.episode,
        source_weight_version=slot.source_weight_version,
        spawn_x_mm=slot.spawn_x_mm,
        spawn_z_mm=slot.spawn_z_mm,
        initial_speed_mm_s=slot.initial_speed_mm_s,
        initial_vz_mm_s=slot.initial_vz_mm_s,
        course_start_gate_index=slot.course_start_gate_index,
        gate_center_overrides=gate_center_overrides,
    )
    slot.final_velocity = tuple(float(value) for value in initial["velocity"])


def result_for_slot(
    slot: ProcessSlotState,
    commit: dict[str, object],
    *,
    curriculum_mode: str = "adaptive",
    environment_version: str = "v3",
) -> dict[str, object]:
    return {
        "episode": slot.episode,
        "slot": slot.slot,
        "source_weight_version": slot.source_weight_version,
        "commit_from_version": int(commit["commit_from_version"]),
        "commit_weight_version": int(commit["commit_weight_version"]),
        "version_staleness": int(commit["staleness"]),
        "control_steps": slot.control_steps,
        "passed_gates": slot.passed_gates,
        "collision": slot.collision,
        "collision_reason": slot.collision_reason,
        "finished": slot.finished,
        "max_x_mm": slot.max_x_mm,
        "min_z_mm": slot.min_z_mm,
        "max_z_mm": slot.max_z_mm,
        "final_vx_mm_s": float(slot.final_velocity[0]),
        "spawn_x_mm": slot.spawn_x_mm,
        "spawn_z_mm": slot.spawn_z_mm,
        "initial_speed_mm_s": slot.initial_speed_mm_s,
        "initial_vz_mm_s": slot.initial_vz_mm_s,
        "environment_version": environment_version,
        "motor_boundary": "whole-body",
        "curriculum_mode": curriculum_mode,
        "boundary_ease_level": slot.boundary_ease_level,
        "boundary_attempt_index": slot.boundary_attempt_index,
        "boundary_batch_number": slot.boundary_batch_number,
        "frontier_role": slot.frontier_role,
        "frontier_target_gate": slot.frontier_target_gate,
        "course_start_gate_index": slot.course_start_gate_index,
        "gate_height_role": slot.gate_height_role,
        "gate_height_center_z_mm": slot.gate_height_center_z_mm,
        "gate_height_attempt_index": slot.gate_height_attempt_index,
        "gate_height_batch_number": slot.gate_height_batch_number,
        "gate_height_learning": slot.gate_height_learning,
        "launch_round": slot.launch_round,
        "reward_events": slot.reward_events,
        "aversive_events": slot.aversive_events,
    }
