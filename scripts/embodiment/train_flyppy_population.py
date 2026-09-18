#!/usr/bin/env python3
"""Asynchronous shared-weight Flyppy v3 population trainer.

Each Fly slot owns its episode-local CNS/body/plasticity state. All slots feed
one learned global weight history. At episode completion the slot's exact
additive+clamp plasticity transaction is rebased onto the newest global weights;
weights are never averaged and a stale slot never overwrites the global state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

from virtual_fly.physics import FLYBODY_V3, FLYPPY_GEOMETRY_V3, FLYPPY_GEOMETRY_V4
from virtual_fly.training.checkpointing import (
    persist_shared_checkpoint as persist_population_checkpoint,
    population_state_snapshot,
    prepare_population_resume,
    recover_population_storage,
)
from virtual_fly.training.population_schedule import checkpoint_can_flush, launch_round_for_index
from virtual_fly.training.flyppy_config import (
    adaptive_config,
    fixed_spawn_condition,
    gate_collision_aversive_current,
    load_viewer_body_ids,
    parse_args,
    population_boundary_config,
    run_reproducibility_metadata,
    save_json_atomic,
    target_condition,
    validate,
    write_run_provenance,
)
from virtual_fly.training.curriculum import (
    AdaptiveCurriculumConfig,
    BoundaryBandConfig,
    SpawnCondition,
    boundary_condition_for_attempt,
    boundary_frontier_role,
    boundary_target_gates,
    frontier_focus_condition,
    frontier_focus_level,
    current_adaptive_condition,
    next_boundary_attempt_for_group,
    ensure_boundary_state,
    load_state as load_curriculum_state,
    record_adaptive_result,
    record_boundary_result,
)

from virtual_fly.runtime.telemetry import LiveTelemetryPublisher
from virtual_fly.embodiment.course import FlyppyCourse
from virtual_fly.embodiment.config import FlyppyBodyConfig
from virtual_fly.embodiment.factory import build_flyppy_stack, reset_flyppy_stack
from virtual_fly.embodiment.haltere import HaltereCampaniformSensor
from virtual_fly.embodiment.retina import MaleCNSRetina
from virtual_fly.embodiment.world import FlyppyWorld
from virtual_fly.embodiment.periphery import WholeBodyPeriphery
from virtual_fly.runtime.neural_bridge import PopulationNeuralBridgeClient
from virtual_fly.reproducibility import (
    HALTERE_FULL_KIND,
    HALTERE_TIMING_KIND,
    build_run_provenance,
    compatibility_warnings,
    validate_derived_artifact,
    validate_haltere_map,
    validate_production_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]








@dataclass
class SlotRuntime:
    slot: int
    course: FlyppyCourse
    world: FlyppyWorld
    body: Any
    periphery: WholeBodyPeriphery
    vision: MaleCNSRetina
    haltere_sensor: HaltereCampaniformSensor
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
    last_retinal: Any = None
    last_haltere: Any = None
    last_peripheral: Any = None
    last_body_spikes: dict[int, bool] | None = None
    reward_events: int = 0
    aversive_events: int = 0
    boundary_ease_level: float | None = None
    boundary_attempt_index: int | None = None
    boundary_batch_number: int | None = None
    frontier_role: str = "evaluate"
    frontier_target_gate: int | None = None
    course_start_gate_index: int | None = None
    launch_round: int = 0




















def make_slot(args: argparse.Namespace, slot_id: int) -> SlotRuntime:
    stack = build_flyppy_stack(FlyppyBodyConfig.from_namespace(args), slot_id)
    return SlotRuntime(
        slot=slot_id,
        course=stack.course,
        world=stack.world,
        body=stack.body,
        periphery=stack.periphery,
        vision=stack.vision,
        haltere_sensor=stack.haltere_sensor,
    )


def begin_episode(
    slot: SlotRuntime,
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
    gate_center_overrides: dict[int, float] | None = None,
    launch_round: int = 0,
) -> None:
    slot.active = True
    slot.episode = episode
    slot.source_weight_version = source_weight_version
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
    slot.launch_round = int(launch_round)
    slot.last_retinal = None
    slot.last_haltere = None
    slot.last_peripheral = None
    slot.last_body_spikes = {}

    slot.final_velocity = reset_flyppy_stack(
        slot,
        condition,
        initial_vz_mm_s=slot.initial_vz_mm_s,
        course_start_gate_index=slot.course_start_gate_index,
        gate_center_overrides=gate_center_overrides,
    )


def result_for_slot(
    slot: SlotRuntime,
    commit: dict,
    *,
    curriculum_mode: str = "adaptive",
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
        "environment_version": slot.course.environment_version,
        "motor_boundary": "whole-body",
        "curriculum_mode": curriculum_mode,
        "boundary_ease_level": slot.boundary_ease_level,
        "boundary_attempt_index": slot.boundary_attempt_index,
        "boundary_batch_number": slot.boundary_batch_number,
        "frontier_role": slot.frontier_role,
        "frontier_target_gate": slot.frontier_target_gate,
        "course_start_gate_index": slot.course_start_gate_index,
        "launch_round": slot.launch_round,
        "reward_events": slot.reward_events,
        "aversive_events": slot.aversive_events,
    }


def main() -> int:
    args = parse_args()
    probe_course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count, environment_version=args.environment_version)
    first_gate = probe_course.gates[0]
    validate(args, first_gate)

    output = args.output_dir
    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint"
    state_path = output / "curriculum-state.json"
    trajectory_path = output / "trajectory.jsonl"
    commit_log_path = output / "commit-log.jsonl"
    summary_path = output / "summary.json"
    storage_recovery = recover_population_storage(output)
    checkpoint_dir_recovery = storage_recovery.checkpoint_directory
    checkpoint_exists = storage_recovery.checkpoint_exists
    if checkpoint_dir_recovery.action not in {"active_checkpoint", "no_checkpoint"}:
        print(f"checkpoint_directory_recovery action={checkpoint_dir_recovery.action}")
    staged_recovery = storage_recovery.staged_state
    if staged_recovery is not None and staged_recovery.action not in {
        "consistent",
        "legacy_checkpoint",
        "none",
    }:
        print(
            "checkpoint_state_recovery action={} checkpoint_v={} active_v={} pending_v={}".format(
                staged_recovery.action,
                staged_recovery.checkpoint_global_weight_version,
                staged_recovery.active_global_weight_version,
                staged_recovery.pending_global_weight_version,
            )
        )

    start_condition = SpawnCondition(
        args.curriculum_start_x_mm,
        float(first_gate.center_z_mm),
        args.curriculum_start_speed_mm_s,
    )
    state = load_curriculum_state(
        state_path,
        start=start_condition,
        target=target_condition(args),
        checkpoint_exists=checkpoint_exists,
    )
    state["curriculum_mode"] = args.curriculum_mode
    state["environment_version"] = args.environment_version
    state["flight_body_version"] = getattr(args, "flight_body_version", "v3")
    state["motor_boundary"] = "whole-body"
    adaptive = adaptive_config(args, first_gate)
    boundary = (
        population_boundary_config(args, first_gate, state)
        if args.curriculum_mode == "boundary-band"
        else None
    )
    boundary_issued_attempts: set[int] = set()
    if boundary is not None:
        ensure_boundary_state(state, boundary)
        state["consecutive_failures"] = 0

    try:
        resume_plan = prepare_population_resume(
            output,
            requested_launch_mode=args.launch_mode,
            checkpoint_exists=checkpoint_exists,
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    initial_global_version = resume_plan.initial_global_weight_version
    start_episode = resume_plan.start_episode
    reproducibility = run_reproducibility_metadata(
        args, body_runtime="in-process", vision_mode_override="raster", vision_rays_override=0
    )
    provenance_path = write_run_provenance(
        output,
        start_episode=start_episode,
        initial_global_weight_version=initial_global_version,
        payload=reproducibility,
    )
    for warning in reproducibility["conditions"].get("compatibility_warnings", []):
        print(f"compatibility_warning={warning}")
    reconciliation = resume_plan.reconciliation
    if reconciliation is not None and reconciliation.changed:
        print(
            "resume_reconciled stable_v={} removed_commits={} removed_trajectory_lines={} "
            "removed_episodes={} next_episode={}".format(
                reconciliation.stable_global_weight_version,
                reconciliation.removed_commits,
                reconciliation.removed_trajectory_lines,
                list(reconciliation.removed_episode_ids),
                reconciliation.next_episode,
            )
        )
    trajectory_mode = resume_plan.trajectory_mode
    commit_mode = resume_plan.commit_mode
    viewer_ids = load_viewer_body_ids(args.viewer_graph) if args.telemetry else ()
    publisher = LiveTelemetryPublisher(output, enabled=args.telemetry)
    slots = [make_slot(args, index) for index in range(args.population)]
    control_dt_s = slots[0].body.timestep * args.physics_steps

    print(
        "population_runtime=enabled population={} shared_weight=true weight_averaging=false "
        "environment={} motor_boundary=whole-body telemetry={} launch_mode={}".format(
            args.population,
            args.environment_version,
            args.telemetry,
            args.launch_mode,
        )
    )

    launched = 0
    completed = 0
    results: list[dict[str, object]] = []
    commit_records: list[dict[str, object]] = []
    saved_state: dict[str, object] = {}
    started = time.perf_counter()
    aggregate_control_steps = 0
    checkpoint_pending = False

    with trajectory_path.open(trajectory_mode, encoding="utf-8") as trajectory, commit_log_path.open(
        commit_mode, encoding="utf-8"
    ) as commit_log:
        with PopulationNeuralBridgeClient(
            snapshot=args.snapshot,
            groups=args.groups,
            slots=args.population,
        ) as brain:
            brain.ping()
            backend_name = str(brain.ready.get("backend", "gpu-population"))
            if checkpoint_exists:
                loaded = brain.load_checkpoint(
                    checkpoint,
                    global_weight_version=initial_global_version,
                )
                print(
                    f"resumed_checkpoint={loaded.get('path')} neural_step={loaded.get('step')} "
                    f"global_weight_version={brain.global_weight_version}"
                )

            def persist_shared_checkpoint() -> dict[str, Any]:
                return persist_population_checkpoint(
                    output_dir=output,
                    checkpoint_dir=checkpoint,
                    curriculum_state=state,
                    population_state=population_state_snapshot(
                        population=args.population,
                        global_weight_version=brain.global_weight_version,
                        curriculum_mode=args.curriculum_mode,
                        launch_mode=args.launch_mode,
                    ),
                    save_checkpoint=brain.save_checkpoint,
                )

            def launch_slot(slot: SlotRuntime) -> bool:
                nonlocal launched
                if launched >= args.episodes:
                    return False
                ease_level: float | None = None
                attempt_index: int | None = None
                batch_number: int | None = None
                frontier_role = "evaluate"
                frontier_target_gate: int | None = None
                course_start_gate_index: int | None = None
                if boundary is None:
                    condition = current_adaptive_condition(state)
                else:
                    attempt_index = next_boundary_attempt_for_group(
                        state,
                        boundary,
                        group_index=slot.slot,
                        group_count=args.population,
                        issued_attempts=boundary_issued_attempts,
                    )
                    if attempt_index is None:
                        return False
                    batch_number = int(dict(state["boundary_band"])["batch_number"])
                    condition, ease_level = boundary_condition_for_attempt(
                        state,
                        boundary,
                        attempt_index,
                        group_count=args.population,
                    )
                    frontier_target_gate = boundary_target_gates(
                        state,
                        boundary,
                        gate_count=args.gate_count,
                    )
                    frontier_role = boundary_frontier_role(
                        attempt_index,
                        group_count=args.population,
                    )
                    if frontier_role == "focus":
                        focus_level = frontier_focus_level(
                            state,
                            boundary,
                            gate_count=args.gate_count,
                        )
                        target_local_index = frontier_target_gate - 1
                        target_gate = slot.course.gates[target_local_index]
                        previous_gate = (
                            slot.course.gates[target_local_index - 1]
                            if target_local_index > 0
                            else None
                        )
                        condition = frontier_focus_condition(
                            condition,
                            target_gate_x_mm=float(target_gate.x_mm),
                            target_gate_z_mm=float(target_gate.center_z_mm),
                            previous_gate_x_mm=(
                                None if previous_gate is None else float(previous_gate.x_mm)
                            ),
                            previous_gate_z_mm=(
                                None
                                if previous_gate is None
                                else float(previous_gate.center_z_mm)
                            ),
                            previous_gate_half_gap_mm=(
                                None
                                if previous_gate is None
                                else float(previous_gate.half_gap_mm)
                            ),
                            focus_level=focus_level,
                        )
                        course_start_gate_index = (
                            slot.course.source_gate_offset + target_local_index
                        )
                    boundary_issued_attempts.add(attempt_index)
                fixed_spawn = fixed_spawn_condition(args)
                if fixed_spawn is not None:
                    condition = fixed_spawn
                episode = start_episode + launched
                launch_round = launch_round_for_index(launched, args.population)
                begin_episode(
                    slot,
                    episode=episode,
                    source_weight_version=brain.global_weight_version,
                    condition=condition,
                    initial_vz_mm_s=(args.fixed_spawn_vz_mm_s if fixed_spawn is not None else 0.0),
                    boundary_ease_level=ease_level,
                    boundary_attempt_index=attempt_index,
                    boundary_batch_number=batch_number,
                    frontier_role=frontier_role,
                    frontier_target_gate=frontier_target_gate,
                    course_start_gate_index=course_start_gate_index,
                    launch_round=launch_round,
                )
                launched += 1
                return True

            for slot in slots:
                if launched >= args.episodes:
                    break
                launch_slot(slot)

            while completed < args.episodes:
                active_slots = [slot for slot in slots if slot.active]
                if not active_slots:
                    raise RuntimeError("population trainer has no active slots before target completion")

                batch_requests = []
                for slot in active_slots:
                    retinal = slot.vision.encode(slot.body.sim, slot.body.fly)
                    haltere = slot.haltere_sensor.encode(slot.body, dt_s=control_dt_s)
                    slot.last_retinal = retinal
                    slot.last_haltere = haltere
                    sensory_currents = (*retinal.body_currents, *haltere.body_currents)
                    neural_sample = (
                        args.telemetry
                        and slot.slot == args.telemetry_slot
                        and slot.control_steps % args.telemetry_stride == 0
                    )
                    if neural_sample and viewer_ids:
                        read_body = tuple(dict.fromkeys((*slot.periphery.body_ids, *viewer_ids)))
                    else:
                        read_body = slot.periphery.body_ids
                    batch_requests.append(
                        {
                            "slot": slot.slot,
                            "stimulate_body": sensory_currents,
                            "read_body": read_body,
                        }
                    )

                batch_spikes = brain.step_batch(batch_requests, plasticity=True)
                terminal_slots: list[SlotRuntime] = []

                for slot in active_slots:
                    body_spikes = batch_spikes.get(slot.slot, {})
                    slot.last_body_spikes = body_spikes
                    peripheral = slot.periphery.step(body_spikes, dt_s=control_dt_s)
                    slot.last_peripheral = peripheral
                    physical_collision, _ = slot.world.step_muscles_until_boundary_contact(
                        slot.body,
                        peripheral,
                        physics_steps=args.physics_steps,
                    )

                    position = slot.body.thorax_position_mm()
                    velocity = slot.body.root_linear_velocity_mm_s()
                    slot.final_velocity = tuple(float(value) for value in velocity)
                    x_mm = float(position[0])
                    z_mm = float(position[2])
                    slot.max_x_mm = max(slot.max_x_mm, x_mm)
                    slot.min_z_mm = min(slot.min_z_mm, z_mm)
                    slot.max_z_mm = max(slot.max_z_mm, z_mm)
                    control_step = slot.control_steps
                    slot.control_steps += 1
                    aggregate_control_steps += 1

                    gate_observation = slot.course.observe(x_mm, z_mm)
                    body_min_x_mm = None
                    if slot.course.needs_full_body_x_sample(x_mm):
                        body_min_x_mm, _ = slot.world.full_body_x_bounds_mm(slot.body.sim)
                    event = slot.course.update(
                        x_mm,
                        z_mm,
                        physical_collision_reason=physical_collision,
                        analytic_body_collision=False,
                        body_min_x_mm=body_min_x_mm,
                    )
                    reward = False
                    aversive = False
                    aversive_current_applied = None
                    gate_miss_distance_mm = None
                    if event.passed_gate:
                        slot.passed_gates += 1
                        slot.reward_events += 1
                        brain.step_slot(
                            slot.slot,
                            stimulate={"reward_dan": args.reward_current},
                            plasticity=True,
                            steps=args.reinforcement_steps,
                        )
                        reward = True
                    if event.collision:
                        slot.collision = True
                        slot.collision_reason = event.collision_reason
                        slot.aversive_events += 1
                        aversive_current_applied, gate_miss_distance_mm = gate_collision_aversive_current(
                            args,
                            gate_observation,
                            event.collision_reason,
                        )
                        brain.step_slot(
                            slot.slot,
                            stimulate={"aversive_dan": aversive_current_applied},
                            plasticity=True,
                            steps=args.reinforcement_steps,
                        )
                        aversive = True
                    if event.finished:
                        slot.finished = True

                    next_gate = getattr(slot.course, "absolute_next_gate_index", slot.course.next_gate_index)
                    if control_step % args.trajectory_stride == 0 or reward or aversive or event.finished:
                        trajectory.write(
                            json.dumps(
                                {
                                    "episode": slot.episode,
                                    "slot": slot.slot,
                                    "source_weight_version": slot.source_weight_version,
                                    "control_step": control_step,
                                    "x_mm": x_mm,
                                    "y_mm": float(position[1]),
                                    "z_mm": z_mm,
                                    "vx_mm_s": float(velocity[0]),
                                    "vy_mm_s": float(velocity[1]),
                                    "vz_mm_s": float(velocity[2]),
                                    "next_gate": next_gate,
                                    "motor_periphery": peripheral.compact_diagnostics(),
                                    "retinal_input": {
                                        "active_columns": slot.last_retinal.active_columns,
                                        "active_photoreceptors": slot.last_retinal.active_photoreceptors,
                                        "mean_current": slot.last_retinal.mean_current,
                                        "max_current": slot.last_retinal.max_current,
                                    },
                                    "reward_stimulated": reward,
                                    "aversive_stimulated": aversive,
                                    "aversive_current": aversive_current_applied,
                                    "gate_miss_distance_mm": gate_miss_distance_mm,
                                    "passed_gate": event.passed_gate,
                                    "collision": event.collision,
                                    "collision_reason": event.collision_reason,
                                    "finished": event.finished,
                                    "environment_version": args.environment_version,
                                    "motor_boundary": "whole-body",
                                    "curriculum_mode": args.curriculum_mode,
                                    "launch_mode": args.launch_mode,
                                    "boundary_ease_level": slot.boundary_ease_level,
                                    "boundary_attempt_index": slot.boundary_attempt_index,
                                    "boundary_batch_number": slot.boundary_batch_number,
                                    "frontier_role": slot.frontier_role,
                                    "frontier_target_gate": slot.frontier_target_gate,
                                    "course_start_gate_index": slot.course_start_gate_index,
                                    "launch_round": slot.launch_round,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )

                    if args.telemetry and slot.slot == args.telemetry_slot:
                        neural_sample = control_step % args.telemetry_stride == 0
                        body_sample = control_step % args.body_telemetry_stride == 0
                        if neural_sample or reward or aversive or event.finished:
                            active_viewer = [
                                body_id for body_id in viewer_ids if body_spikes.get(body_id, False)
                            ]
                            publisher.publish_neural(
                                episode=slot.episode,
                                control_step=control_step,
                                neural_step=brain.last_step,
                                depolarizing_body_ids=active_viewer,
                                hyperpolarizing_body_ids=(),
                                reward=reward,
                                aversive=aversive,
                            )
                        if body_sample or reward or aversive or event.finished:
                            publisher.publish_body(
                                episode=slot.episode,
                                control_step=control_step,
                                sim=slot.body.sim,
                                next_gate=next_gate,
                                passed_gate=event.passed_gate,
                                collision=event.collision,
                                collision_reason=event.collision_reason,
                                reward=reward,
                                aversive=aversive,
                                motor=peripheral.compact_diagnostics(),
                                retinal={
                                    "active_columns": slot.last_retinal.active_columns,
                                    "active_photoreceptors": slot.last_retinal.active_photoreceptors,
                                    "mean_current": slot.last_retinal.mean_current,
                                    "max_current": slot.last_retinal.max_current,
                                },
                            )

                    if slot.collision or slot.finished or slot.control_steps >= args.max_control_steps:
                        terminal_slots.append(slot)

                trajectory.flush()

                for slot in sorted(terminal_slots, key=lambda item: item.slot):
                    commit = brain.commit_slot(
                        slot.slot,
                        source_weight_version=slot.source_weight_version,
                    )
                    completed += 1
                    state["curriculum_episodes"] = int(state["curriculum_episodes"]) + 1
                    batch_completed = None
                    if boundary is None:
                        record_adaptive_result(state, adaptive, success=slot.passed_gates > 0)
                    else:
                        if slot.boundary_attempt_index is None:
                            raise RuntimeError(
                                "boundary-band slot completed without an attempt index"
                            )
                        target_gates = boundary_target_gates(
                            state,
                            boundary,
                            gate_count=args.gate_count,
                        )
                        batch_completed = record_boundary_result(
                            state,
                            boundary,
                            passed_gates=slot.passed_gates,
                            gate_count=args.gate_count,
                            attempt_index=slot.boundary_attempt_index,
                            group=slot.slot,
                            frontier_role=slot.frontier_role,
                        )
                        if batch_completed is not None:
                            boundary_issued_attempts.clear()
                            print(
                                "boundary_batch_complete target_gates={} mastered={} next_target={} "
                                "success={}/{} rate={:.3f} raw_rate={:.3f} aggregation={} "
                                "adjustment={} gate_rates={} hard={} easy={}".format(
                                    batch_completed["evaluated_target_gates"],
                                    batch_completed["frontier_mastered_gates"],
                                    batch_completed["frontier_target_gates"],
                                    batch_completed["successes"],
                                    batch_completed["attempts"],
                                    batch_completed["success_rate"],
                                    batch_completed["raw_success_rate"],
                                    batch_completed["aggregation"],
                                    batch_completed["adjustment"],
                                    batch_completed["gate_pass_rates"],
                                    batch_completed["hard"],
                                    batch_completed["easy"],
                                )
                            )
                    result = result_for_slot(
                        slot,
                        commit,
                        curriculum_mode=args.curriculum_mode,
                    )
                    results.append(result)
                    commit_record = {
                        "commit_seq": int(commit["commit_weight_version"]),
                        "slot": slot.slot,
                        "episode": slot.episode,
                        "source_weight_version": int(commit["source_weight_version"]),
                        "commit_from_version": int(commit["commit_from_version"]),
                        "commit_weight_version": int(commit["commit_weight_version"]),
                        "staleness": int(commit["staleness"]),
                        "reward_events": slot.reward_events,
                        "aversive_events": slot.aversive_events,
                        "passed_gates": slot.passed_gates,
                        "collision": slot.collision,
                        "collision_reason": slot.collision_reason,
                        "control_steps": slot.control_steps,
                        "spawn_x_mm": slot.spawn_x_mm,
                        "spawn_z_mm": slot.spawn_z_mm,
                        "initial_speed_mm_s": slot.initial_speed_mm_s,
                        "curriculum_mode": args.curriculum_mode,
                        "launch_mode": args.launch_mode,
                        "boundary_ease_level": slot.boundary_ease_level,
                        "boundary_attempt_index": slot.boundary_attempt_index,
                        "boundary_batch_number": slot.boundary_batch_number,
                        "frontier_role": slot.frontier_role,
                        "frontier_target_gate": slot.frontier_target_gate,
                        "course_start_gate_index": slot.course_start_gate_index,
                        "launch_round": slot.launch_round,
                    }
                    commit_log.write(json.dumps(commit_record, separators=(",", ":")) + "\n")
                    commit_log.flush()
                    commit_records.append(commit_record)
                    print(
                        "population_commit episode={} slot={} source_v={} from_v={} -> v{} "
                        "staleness={} steps={} passed={} collision={}".format(
                            slot.episode,
                            slot.slot,
                            slot.source_weight_version,
                            commit["commit_from_version"],
                            commit["commit_weight_version"],
                            commit["staleness"],
                            slot.control_steps,
                            slot.passed_gates,
                            slot.collision,
                        )
                    )

                    if completed % args.checkpoint_every == 0:
                        checkpoint_pending = True

                    if (
                        args.launch_mode == "async"
                        and boundary is None
                        and launched < args.episodes
                    ):
                        launch_slot(slot)
                    else:
                        slot.active = False

                if checkpoint_can_flush(
                    launch_mode=args.launch_mode,
                    checkpoint_pending=checkpoint_pending,
                    active_slots=sum(slot.active for slot in slots),
                ):
                    saved_state = persist_shared_checkpoint()
                    checkpoint_pending = False

                if args.launch_mode == "wave" and launched < args.episodes:
                    if not any(slot.active for slot in slots):
                        for idle_slot in slots:
                            if launched >= args.episodes:
                                break
                            launch_slot(idle_slot)
                elif boundary is not None and launched < args.episodes:
                    for idle_slot in slots:
                        if launched >= args.episodes:
                            break
                        if idle_slot.active:
                            continue
                        launch_slot(idle_slot)

            saved_state = persist_shared_checkpoint()
            final_global_version = brain.global_weight_version
            backend_name = str(brain.ready.get("backend", "gpu-population"))

    elapsed = time.perf_counter() - started
    staleness_values = [int(item["staleness"]) for item in commit_records]
    summary = {
        "schema_version": 12,
        "experiment": f"flyppy_{args.environment_version}_shared_weight_population",
        "backend": backend_name,
        "persistent_runtime": True,
        "population": args.population,
        "shared_weight": True,
        "weight_averaging": False,
        "commit_semantics": "episode-local additive+clamp transaction rebased onto latest global weight",
        "launch_mode": args.launch_mode,
        "environment_version": args.environment_version,
        "flight_body_version": getattr(args, "flight_body_version", "v3"),
        "haltere_sensory_feedback": {
            "enabled": float(args.haltere_current_gain) > 0.0,
            "current_gain": float(args.haltere_current_gain),
            "map": str(args.haltere_sensory_map),
            "map_kind": str(args.haltere_sensory_kind),
            "transduction": str(args.haltere_transduction),
        },
        "motor_boundary": "whole-body",
        "physical_spec": {
            "morphology": FLYBODY_V3.morphology,
            "body_length_mm": FLYBODY_V3.body_length_mm,
            "wing_length_mm": FLYBODY_V3.wing_length_mm,
            "total_mass_mg": FLYBODY_V3.total_mass_mg,
            "neural_sex": FLYBODY_V3.neural_sex,
            "morphology_sex": FLYBODY_V3.morphology_sex,
        },
        "environment_spec": {
            "version": args.environment_version,
            "corridor_low_z_mm": probe_course.floor_z_mm,
            "corridor_high_z_mm": probe_course.ceiling_z_mm,
            "lateral_half_width_mm": probe_course.lateral_half_width_mm,
            "first_gate_x_mm": first_gate.x_mm,
            "gate_gap_height_mm": 2.0 * first_gate.half_gap_mm,
            "gate_half_thickness_mm": first_gate.half_thickness_mm,
        },
        "curriculum_mode": args.curriculum_mode,
        "episodes_this_run": args.episodes,
        "episode_start": start_episode,
        "episode_end": start_episode + args.episodes - 1,
        "control_dt_seconds": control_dt_s,
        "aggregate_control_steps": aggregate_control_steps,
        "aggregate_control_steps_per_second": aggregate_control_steps / elapsed,
        "simulated_seconds_this_run": aggregate_control_steps * control_dt_s,
        "total_passed_gates_this_run": sum(int(item["passed_gates"]) for item in results),
        "total_collisions_this_run": sum(bool(item["collision"]) for item in results),
        "curriculum": state,
        "episode_results": sorted(results, key=lambda item: int(item["episode"])),
        "population_weight_checkpoint": str(checkpoint),
        "checkpoint_semantics": "global-weights-only-v1",
        "checkpoint_resume_behavior": "population fast neural/plasticity state is reset; only global weights are restored",
        "checkpoint_neural_step": saved_state.get("step"),
        "checkpoint_neural_step_semantics": "aggregate-slot-neural-step-count-v1",
        "reproducibility": reproducibility,
        "provenance_file": str(provenance_path),
        "global_weight_version_start": initial_global_version,
        "global_weight_version_end": final_global_version,
        "mean_version_staleness": (
            sum(staleness_values) / len(staleness_values) if staleness_values else 0.0
        ),
        "max_version_staleness": max(staleness_values) if staleness_values else 0,
        "commit_log": str(commit_log_path),
        "telemetry_dir": str(output / "live"),
        "viewer_graph": str(args.viewer_graph),
        "elapsed_seconds": elapsed,
        "teaching_signal": "gate pass -> PAM reward DAN current; gate collision -> PPL current graded by vertical miss distance; floor/ceiling -> full PPL current; focused frontier episodes create local success experience but never count as mastery; only full-course evaluation episodes advance the arbitrary gate-count frontier",
    }
    save_json_atomic(summary_path, summary)
    publisher.publish_status(
        running=False,
        backend=backend_name,
        episode=results[-1]["episode"] if results else None,
        control_step=results[-1]["control_steps"] if results else None,
        curriculum=state,
    )
    print(f"summary={summary_path}")
    print(f"trajectory={trajectory_path}")
    print(f"commit_log={commit_log_path}")
    print(f"checkpoint={checkpoint}")
    print(f"elapsed={elapsed:.3f}s")
    print(f"aggregate_control_steps_per_second={aggregate_control_steps / elapsed:.3f}")
    print("flyppy_async_shared_weight_population=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
