#!/usr/bin/env python3
"""Production-equivalent Flyppy trainer with process-isolated physical bodies.

The parent process owns the shared GPU MaleCNS runtime and asynchronous shared
weight history. Each spawned child process owns one complete physical fly:
FlyBody/MuJoCo, compound-eye rendering, retinal transduction, whole-body
periphery, and Flyppy course state.

The neural learning semantics intentionally match train_flyppy_population.py:
there is no weight averaging. Episode-local additive+clamp plasticity
transactions are rebased onto the latest global weights when a slot commits.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from typing import Any

import train_flyppy_population as reference
from flyppy_body_worker import FlyppyBodyProcess, spawn_body_processes
from flyppy_course import FlyppyCourse
from live_telemetry import LiveTelemetryPublisher
from population_neural_bridge_client import PopulationNeuralBridgeClient
from virtual_fly.physics import FLYBODY_V3
from virtual_fly.training.checkpointing import (
    persist_shared_checkpoint as persist_population_checkpoint,
    population_state_snapshot,
    prepare_population_resume,
    recover_population_storage,
)
from virtual_fly.training.population_schedule import checkpoint_can_flush, launch_round_for_index
from virtual_fly.training.curriculum import (
    SpawnCondition,
    boundary_condition_for_attempt,
    current_adaptive_condition,
    next_boundary_attempt_for_group,
    ensure_boundary_state,
    load_state as load_curriculum_state,
    record_adaptive_result,
    record_boundary_result,
)


@dataclass
class ProcessSlotState:
    slot: int
    active: bool = False
    episode: int = -1
    source_weight_version: int = 0
    spawn_x_mm: float = 0.0
    spawn_z_mm: float = 0.0
    initial_speed_mm_s: float = 0.0
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
    launch_round: int = 0


def worker_config(args) -> dict[str, object]:
    return {
        "seed": int(args.seed),
        "gate_count": int(args.gate_count),
        "wing_motor_map": str(args.wing_motor_map),
        "body_motor_map": str(args.body_motor_map),
        "retinotopic_map": str(args.retinotopic_map),
        "photoreceptor_current_gain": float(args.photoreceptor_current_gain),
    }


def reset_slot(
    slot: ProcessSlotState,
    worker: FlyppyBodyProcess,
    *,
    episode: int,
    source_weight_version: int,
    condition: SpawnCondition,
    boundary_ease_level: float | None = None,
    boundary_attempt_index: int | None = None,
    boundary_batch_number: int | None = None,
    launch_round: int = 0,
) -> None:
    slot.active = True
    slot.episode = int(episode)
    slot.source_weight_version = int(source_weight_version)
    slot.spawn_x_mm = float(condition.x_mm)
    slot.spawn_z_mm = float(condition.z_mm)
    slot.initial_speed_mm_s = float(condition.speed_mm_s)
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
    )
    slot.final_velocity = tuple(float(value) for value in initial["velocity"])


def result_for_slot(
    slot: ProcessSlotState,
    commit: dict[str, object],
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
        "environment_version": "v3",
        "motor_boundary": "whole-body",
        "curriculum_mode": curriculum_mode,
        "boundary_ease_level": slot.boundary_ease_level,
        "boundary_attempt_index": slot.boundary_attempt_index,
        "boundary_batch_number": slot.boundary_batch_number,
        "launch_round": slot.launch_round,
        "reward_events": slot.reward_events,
        "aversive_events": slot.aversive_events,
    }


def main() -> int:
    args = reference.parse_args()
    probe_course = FlyppyCourse(
        seed=args.seed,
        gate_count=args.gate_count,
        environment_version="v3",
    )
    first_gate = probe_course.gates[0]
    reference.validate(args, first_gate)

    timeout_s = float(os.environ.get("VF_FLYPPY_BODY_WORKER_TIMEOUT_S", "120"))
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise SystemExit("VF_FLYPPY_BODY_WORKER_TIMEOUT_S must be finite and positive")

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
        target=reference.target_condition(args),
        checkpoint_exists=checkpoint_exists,
    )
    state["curriculum_mode"] = args.curriculum_mode
    state["environment_version"] = "v3"
    state["motor_boundary"] = "whole-body"
    adaptive = reference.adaptive_config(args, first_gate)
    boundary = (
        reference.population_boundary_config(args, first_gate, state)
        if args.curriculum_mode == "boundary-band"
        else None
    )
    boundary_issued_attempts: set[int] = set()
    if boundary is not None:
        ensure_boundary_state(state, boundary)
        # This field is meaningful only to the historical adaptive policy. Keep
        # it neutral once an experiment switches to asynchronous batch updates.
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
    # Loading the small immutable viewer-ID list once has no per-step cost. The
    # expensive viewer-specific neural reads and body snapshots remain gated by
    # telemetry_active below.
    viewer_ids = reference.load_viewer_body_ids(args.viewer_graph)
    publisher = LiveTelemetryPublisher(output, enabled=bool(args.telemetry))

    workers = spawn_body_processes(
        population=args.population,
        physics_steps=args.physics_steps,
        timeout_s=timeout_s,
        config=worker_config(args),
    )
    by_slot = {worker.slot_index: worker for worker in workers}
    slots = [ProcessSlotState(slot=index) for index in range(args.population)]

    control_dts = {round(worker.control_dt_s, 15) for worker in workers}
    if len(control_dts) != 1:
        for worker in workers:
            worker.close()
        raise RuntimeError(f"body workers disagree on control dt: {control_dts}")
    control_dt_s = float(workers[0].control_dt_s)

    telemetry_mode = "always" if args.telemetry else "viewer-demand"
    print(
        "population_runtime=enabled population={} shared_weight=true weight_averaging=false "
        "environment=v3 motor_boundary=whole-body body_runtime=process telemetry={} launch_mode={}".format(
            args.population,
            telemetry_mode,
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
    backend_name = "gpu-population"
    checkpoint_pending = False

    try:
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
                            body_runtime="process-isolated",
                        ),
                        save_checkpoint=brain.save_checkpoint,
                    )

                def launch_slot(slot: ProcessSlotState) -> bool:
                    """Launch one episode without letting async completion order pick difficulty."""

                    nonlocal launched
                    if launched >= args.episodes:
                        return False
                    ease_level: float | None = None
                    attempt_index: int | None = None
                    batch_number: int | None = None
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
                        boundary_issued_attempts.add(attempt_index)
                    episode = start_episode + launched
                    launch_round = launch_round_for_index(launched, args.population)
                    reset_slot(
                        slot,
                        by_slot[slot.slot],
                        episode=episode,
                        source_weight_version=brain.global_weight_version,
                        condition=condition,
                        boundary_ease_level=ease_level,
                        boundary_attempt_index=attempt_index,
                        boundary_batch_number=batch_number,
                        launch_round=launch_round,
                    )
                    launched += 1
                    return True

                for slot in slots:
                    if launched >= args.episodes:
                        break
                    launch_slot(slot)

                publisher.publish_status(
                    running=True,
                    backend=backend_name,
                    episode=slots[0].episode if slots else None,
                    control_step=0,
                    curriculum=state,
                )

                while completed < args.episodes:
                    active_slots = [slot for slot in slots if slot.active]
                    if not active_slots:
                        raise RuntimeError(
                            "process population trainer has no active slots before target completion"
                        )

                    # Viewer demand is a separate runtime condition from the
                    # --telemetry CLI override. With neither active, no viewer
                    # neural reads, body snapshots, or telemetry writes occur.
                    telemetry_active = bool(args.telemetry) or publisher.requested()

                    # Observe every active physical fly concurrently.
                    for slot in active_slots:
                        by_slot[slot.slot].request("observe")
                    observations = {
                        slot.slot: by_slot[slot.slot].receive("observe")
                        for slot in active_slots
                    }

                    batch_requests = []
                    for slot in active_slots:
                        retinal = observations[slot.slot]
                        slot.last_retinal = retinal
                        neural_sample = (
                            telemetry_active
                            and slot.slot == args.telemetry_slot
                            and slot.control_steps % args.telemetry_stride == 0
                        )
                        body_ids = by_slot[slot.slot].body_ids
                        if neural_sample and viewer_ids:
                            read_body = tuple(dict.fromkeys((*body_ids, *viewer_ids)))
                        else:
                            read_body = body_ids
                        batch_requests.append(
                            {
                                "slot": slot.slot,
                                "stimulate_body": retinal["body_currents"],
                                "read_body": read_body,
                            }
                        )

                    batch_spikes = brain.step_batch(batch_requests, plasticity=True)

                    # Apply each slot's individual MN output concurrently.
                    for slot in active_slots:
                        body_spikes = batch_spikes.get(slot.slot, {})
                        slot.last_body_spikes = body_spikes
                        active_ids = tuple(
                            int(body_id)
                            for body_id, fired in body_spikes.items()
                            if bool(fired)
                        )
                        by_slot[slot.slot].request("act", active_ids)
                    acts = {
                        slot.slot: by_slot[slot.slot].receive("act")
                        for slot in active_slots
                    }

                    terminal_slots: list[ProcessSlotState] = []
                    for slot in active_slots:
                        act = acts[slot.slot]
                        slot.last_motor = dict(act["motor"])
                        position = tuple(float(value) for value in act["position"])
                        velocity = tuple(float(value) for value in act["velocity"])
                        slot.final_velocity = velocity
                        x_mm = position[0]
                        z_mm = position[2]
                        slot.max_x_mm = max(slot.max_x_mm, x_mm)
                        slot.min_z_mm = min(slot.min_z_mm, z_mm)
                        slot.max_z_mm = max(slot.max_z_mm, z_mm)
                        control_step = slot.control_steps
                        slot.control_steps += 1
                        aggregate_control_steps += 1

                        reward = False
                        aversive = False
                        if bool(act["passed_gate"]):
                            slot.passed_gates += 1
                            slot.reward_events += 1
                            brain.step_slot(
                                slot.slot,
                                stimulate={"reward_dan": args.reward_current},
                                plasticity=True,
                                steps=args.reinforcement_steps,
                            )
                            reward = True
                        if bool(act["collision"]):
                            slot.collision = True
                            slot.collision_reason = act["collision_reason"]
                            slot.aversive_events += 1
                            brain.step_slot(
                                slot.slot,
                                stimulate={"aversive_dan": args.aversive_current},
                                plasticity=True,
                                steps=args.reinforcement_steps,
                            )
                            aversive = True
                        if bool(act["finished"]):
                            slot.finished = True

                        next_gate = int(act["next_gate"])
                        retinal_diag = {
                            "active_columns": int(slot.last_retinal["active_columns"]),
                            "active_photoreceptors": int(
                                slot.last_retinal["active_photoreceptors"]
                            ),
                            "mean_current": float(slot.last_retinal["mean_current"]),
                            "max_current": float(slot.last_retinal["max_current"]),
                        }
                        if (
                            control_step % args.trajectory_stride == 0
                            or reward
                            or aversive
                            or slot.finished
                        ):
                            trajectory.write(
                                json.dumps(
                                    {
                                        "episode": slot.episode,
                                        "slot": slot.slot,
                                        "source_weight_version": slot.source_weight_version,
                                        "control_step": control_step,
                                        "x_mm": x_mm,
                                        "y_mm": position[1],
                                        "z_mm": z_mm,
                                        "vx_mm_s": velocity[0],
                                        "vy_mm_s": velocity[1],
                                        "vz_mm_s": velocity[2],
                                        "next_gate": next_gate,
                                        "motor_periphery": slot.last_motor,
                                        "retinal_input": retinal_diag,
                                        "reward_stimulated": reward,
                                        "aversive_stimulated": aversive,
                                        "passed_gate": bool(act["passed_gate"]),
                                        "collision": bool(act["collision"]),
                                        "collision_reason": act["collision_reason"],
                                        "finished": bool(act["finished"]),
                                        "environment_version": "v3",
                                        "motor_boundary": "whole-body",
                                        "curriculum_mode": args.curriculum_mode,
                                        "launch_mode": args.launch_mode,
                                        "boundary_ease_level": slot.boundary_ease_level,
                                        "boundary_attempt_index": slot.boundary_attempt_index,
                                        "boundary_batch_number": slot.boundary_batch_number,
                                        "launch_round": slot.launch_round,
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )

                        if telemetry_active and slot.slot == args.telemetry_slot:
                            neural_sample = control_step % args.telemetry_stride == 0
                            body_sample = control_step % args.body_telemetry_stride == 0
                            body_spikes = slot.last_body_spikes or {}
                            if neural_sample or reward or aversive or slot.finished:
                                active_viewer = [
                                    body_id
                                    for body_id in viewer_ids
                                    if body_spikes.get(body_id, False)
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
                            if body_sample or reward or aversive or slot.finished:
                                snapshot = by_slot[slot.slot].snapshot()
                                publisher.publish_body_state(
                                    episode=slot.episode,
                                    control_step=control_step,
                                    sim_time_s=float(snapshot["sim_time_s"]),
                                    qpos=snapshot["qpos"],
                                    qvel=snapshot["qvel"],
                                    next_gate=next_gate,
                                    passed_gate=bool(act["passed_gate"]),
                                    collision=bool(act["collision"]),
                                    collision_reason=act["collision_reason"],
                                    reward=reward,
                                    aversive=aversive,
                                    motor=slot.last_motor or {},
                                    retinal=retinal_diag,
                                )

                        if (
                            slot.collision
                            or slot.finished
                            or slot.control_steps >= args.max_control_steps
                        ):
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
                            record_adaptive_result(
                                state,
                                adaptive,
                                success=slot.passed_gates > 0,
                            )
                        else:
                            if slot.boundary_attempt_index is None:
                                raise RuntimeError(
                                    "boundary-band slot completed without an attempt index"
                                )
                            batch_completed = record_boundary_result(
                                state,
                                boundary,
                                success=slot.passed_gates > 0,
                                attempt_index=slot.boundary_attempt_index,
                                group=slot.slot,
                            )
                            if batch_completed is not None:
                                # Every condition in the old band has now produced
                                # an outcome.  The next band can reuse attempt IDs.
                                boundary_issued_attempts.clear()
                                print(
                                    "boundary_batch_complete success={}/{} rate={:.3f} raw_rate={:.3f} "
                                    "aggregation={} adjustment={} hard={} easy={}".format(
                                        batch_completed["successes"],
                                        batch_completed["attempts"],
                                        batch_completed["success_rate"],
                                        batch_completed["raw_success_rate"],
                                        batch_completed["aggregation"],
                                        batch_completed["adjustment"],
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
                            "launch_round": slot.launch_round,
                        }
                        commit_log.write(
                            json.dumps(commit_record, separators=(",", ":")) + "\n"
                        )
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
                        # Every slot in a wave starts from the same global weight
                        # version.  Do not let fast-finishing slots begin another
                        # trajectory while slower slots are still experiencing the
                        # previous version.  Commits may still arrive in completion
                        # order, but the next wave starts only after all prior
                        # trajectories have terminated and committed.
                        if not any(slot.active for slot in slots):
                            for idle_slot in slots:
                                if launched >= args.episodes:
                                    break
                                launch_slot(idle_slot)
                    elif boundary is not None and launched < args.episodes:
                        # Async boundary-band mode keeps the fixed per-slot attempt
                        # assignment but immediately refills an idle slot while
                        # conditions for that slot remain in the current batch.
                        for idle_slot in slots:
                            if launched >= args.episodes:
                                break
                            if idle_slot.active:
                                continue
                            launch_slot(idle_slot)

                saved_state = persist_shared_checkpoint()
                final_global_version = brain.global_weight_version
                backend_name = str(brain.ready.get("backend", "gpu-population"))
    finally:
        for worker in workers:
            try:
                worker.close()
            except Exception:
                pass

    elapsed = time.perf_counter() - started
    staleness_values = [int(item["staleness"]) for item in commit_records]
    summary = {
        "schema_version": 13,
        "experiment": "flyppy_v3_shared_weight_population",
        "backend": backend_name,
        "persistent_runtime": True,
        "population": args.population,
        "shared_weight": True,
        "weight_averaging": False,
        "commit_semantics": "episode-local additive+clamp transaction rebased onto latest global weight",
        "body_runtime": "process-isolated",
        "launch_mode": args.launch_mode,
        "environment_version": "v3",
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
            "version": "v3",
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
        "full_cns_checkpoint": str(checkpoint),
        "checkpoint_neural_step": saved_state.get("step"),
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
        "teaching_signal": "gate pass -> PAM reward DAN current; collision -> PPL aversive DAN current; curriculum changes episode-reset initial conditions only",
    }
    reference.save_json_atomic(summary_path, summary)
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
    print("flyppy_async_shared_weight_process_population=PASS")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
