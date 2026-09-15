#!/usr/bin/env python3
"""Asynchronous shared-weight Flyppy v3 population trainer.

Each Fly slot owns its episode-local CNS/body/plasticity state. All slots feed
one learned global weight history. At episode completion the slot's exact
additive+clamp plasticity transaction is rebased onto the newest global weights;
weights are never averaged and a stale slot never overwrites the global state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

from virtual_fly.physics import FLYBODY_V3, FLYPPY_GEOMETRY_V3
from virtual_fly.training.curriculum import (
    AdaptiveCurriculumConfig,
    BoundaryBandConfig,
    SpawnCondition,
    boundary_condition_for_attempt,
    current_adaptive_condition,
    ensure_boundary_state,
    load_state as load_curriculum_state,
    record_adaptive_result,
    record_boundary_result,
)

from flybody_v3_adapter import FlyBodyV3NeuromuscularAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from live_telemetry import LiveTelemetryPublisher
from malecns_retina import MaleCNSRetina
from population_neural_bridge_client import PopulationNeuralBridgeClient
from whole_body_periphery import WholeBodyPeriphery


@dataclass
class SlotRuntime:
    slot: int
    course: FlyppyCourse
    world: FlyppyWorld
    body: Any
    periphery: WholeBodyPeriphery
    vision: MaleCNSRetina
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
    last_retinal: Any = None
    last_peripheral: Any = None
    last_body_spikes: dict[int, bool] | None = None
    reward_events: int = 0
    aversive_events: int = 0
    boundary_ease_level: float | None = None
    boundary_attempt_index: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=24, help="total episodes launched across all slots")
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument("--groups", type=Path, default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"))
    parser.add_argument("--retinotopic-map", type=Path, default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"))
    parser.add_argument("--wing-motor-map", type=Path, default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"))
    parser.add_argument("--body-motor-map", type=Path, default=Path("artifacts/malecns-v1.0/body-motor-neurons-v0.json"))
    parser.add_argument("--viewer-graph", type=Path, default=Path("artifacts/embodiment/neural-viewer-graph-v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/flyppy-v3"))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-control-steps", type=int, default=1800)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--photoreceptor-current-gain", type=float, default=2.0)
    parser.add_argument("--reward-current", type=float, default=2.0)
    parser.add_argument("--aversive-current", type=float, default=2.0)
    parser.add_argument("--reinforcement-steps", type=int, default=4)
    parser.add_argument("--telemetry", action="store_true", help="publish detached viewer telemetry for one slot")
    parser.add_argument("--telemetry-slot", type=int, default=0)
    parser.add_argument("--telemetry-stride", type=int, default=10)
    parser.add_argument("--body-telemetry-stride", type=int, default=1)

    parser.add_argument(
        "--curriculum-mode",
        choices=("adaptive", "boundary-band"),
        default="adaptive",
        help="adaptive preserves the historical per-episode policy; boundary-band updates only after a fixed batch",
    )
    parser.add_argument("--boundary-batch-size", type=int, default=24)
    parser.add_argument("--boundary-harden-success-rate", type=float, default=0.80)
    parser.add_argument("--boundary-ease-success-rate", type=float, default=0.40)

    parser.add_argument("--curriculum-start-x-mm", type=float, default=8.91)
    parser.add_argument("--curriculum-target-x-mm", type=float, default=0.0)
    parser.add_argument("--curriculum-target-z-mm", type=float, default=8.91)
    parser.add_argument("--curriculum-start-speed-mm-s", type=float, default=400.0)
    parser.add_argument("--curriculum-target-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--curriculum-x-step-mm", type=float, default=0.297)
    parser.add_argument("--curriculum-failure-x-step-mm", type=float, default=0.1485)
    parser.add_argument("--curriculum-z-step-mm", type=float, default=0.1485)
    parser.add_argument("--curriculum-speed-step-mm-s", type=float, default=12.5)
    parser.add_argument("--curriculum-max-easy-speed-mm-s", type=float, default=450.0)
    return parser.parse_args()


def save_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def infer_next_episode(trajectory_path: Path) -> int:
    if not trajectory_path.exists():
        return 0
    highest = -1
    with trajectory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                highest = max(highest, int(json.loads(line).get("episode", -1)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return highest + 1


def load_viewer_body_ids(path: Path) -> tuple[int, ...]:
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    return tuple(sorted({int(node["body_id"]) for node in payload.get("nodes", [])}))


def target_condition(args: argparse.Namespace) -> SpawnCondition:
    return SpawnCondition(
        args.curriculum_target_x_mm,
        args.curriculum_target_z_mm,
        args.curriculum_target_speed_mm_s,
    )


def adaptive_config(args: argparse.Namespace, first_gate) -> AdaptiveCurriculumConfig:
    return AdaptiveCurriculumConfig(
        target=target_condition(args),
        x_step_mm=args.curriculum_x_step_mm,
        z_step_mm=args.curriculum_z_step_mm,
        speed_step_mm_s=args.curriculum_speed_step_mm_s,
        max_easy_speed_mm_s=args.curriculum_max_easy_speed_mm_s,
        failure_x_step_mm=args.curriculum_failure_x_step_mm,
        first_gate_x_mm=float(first_gate.x_mm),
        first_gate_z_mm=float(first_gate.center_z_mm),
    )


def validate(args: argparse.Namespace, first_gate) -> None:
    if args.episodes < 1:
        raise SystemExit("episodes must be >= 1")
    if args.population < 1:
        raise SystemExit("population must be >= 1")
    if args.population > 32:
        raise SystemExit("population > 32 exceeds the GPU runtime active-mask limit")
    if not 0 <= args.telemetry_slot < args.population:
        raise SystemExit("telemetry-slot must identify an existing population slot")
    if args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("control/physics steps must be >= 1")
    if min(args.trajectory_stride, args.telemetry_stride, args.body_telemetry_stride) < 1:
        raise SystemExit("trajectory/telemetry strides must be >= 1")
    if args.checkpoint_every < 1:
        raise SystemExit("checkpoint-every must be >= 1")
    if args.boundary_batch_size < 5:
        raise SystemExit("boundary-batch-size must be >= 5")
    if not (
        0.0
        <= args.boundary_ease_success_rate
        < args.boundary_harden_success_rate
        <= 1.0
    ):
        raise SystemExit("boundary success-rate thresholds are invalid")
    if args.reinforcement_steps < 1 or args.reinforcement_steps % 2 != 0:
        raise SystemExit(
            "population prototype currently requires an even reinforcement-steps value; production v3 uses 4"
        )
    if args.curriculum_start_x_mm >= first_gate.x_mm:
        raise SystemExit("curriculum-start-x-mm must be before the first gate")
    required = [
        args.snapshot / "manifest.json",
        args.groups,
        args.retinotopic_map,
        args.wing_motor_map,
        args.body_motor_map,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing population training inputs:\n  " + "\n  ".join(missing))
    positive = [
        args.photoreceptor_current_gain,
        args.reward_current,
        args.aversive_current,
        args.curriculum_x_step_mm,
        args.curriculum_failure_x_step_mm,
        args.curriculum_z_step_mm,
        args.curriculum_start_speed_mm_s,
        args.curriculum_target_speed_mm_s,
        args.curriculum_speed_step_mm_s,
        args.curriculum_max_easy_speed_mm_s,
    ]
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
        raise SystemExit("positive training parameters must be finite and > 0")


def population_boundary_config(
    args: argparse.Namespace,
    first_gate,
    state: dict[str, Any],
) -> BoundaryBandConfig:
    """Build a local v3 boundary band around the resumed adaptive condition.

    Existing ``boundary_band`` state, when present, owns the persisted endpoints.
    These derived endpoints are therefore only the migration/start defaults when
    a continuation experiment first switches from adaptive to boundary-band.
    """

    adaptive = adaptive_config(args, first_gate)
    hard_state = dict(state)
    easy_state = dict(state)
    record_adaptive_result(hard_state, adaptive, success=True)
    record_adaptive_result(easy_state, adaptive, success=False)
    hard = current_adaptive_condition(hard_state)
    easy = current_adaptive_condition(easy_state)

    def span(a: float, b: float, fallback: float) -> float:
        width = abs(float(b) - float(a))
        return width if width > 1e-12 else float(fallback)

    harden_step = SpawnCondition(
        span(hard.x_mm, easy.x_mm, args.curriculum_x_step_mm),
        span(hard.z_mm, easy.z_mm, args.curriculum_z_step_mm),
        span(hard.speed_mm_s, easy.speed_mm_s, args.curriculum_speed_step_mm_s),
    )
    ease_step = SpawnCondition(
        harden_step.x_mm / 2.0,
        harden_step.z_mm / 2.0,
        harden_step.speed_mm_s / 2.0,
    )
    return BoundaryBandConfig(
        hard=hard,
        easy=easy,
        target=target_condition(args),
        batch_size=args.boundary_batch_size,
        harden_success_rate=args.boundary_harden_success_rate,
        ease_success_rate=args.boundary_ease_success_rate,
        harden_step=harden_step,
        ease_step=ease_step,
        seed=args.seed,
    )


def make_slot(args: argparse.Namespace, slot_id: int) -> SlotRuntime:
    course = FlyppyCourse(
        seed=args.seed + slot_id,
        gate_count=args.gate_count,
        environment_version="v3",
    )
    world = FlyppyWorld(course)
    body = FlyBodyV3NeuromuscularAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, FLYPPY_GEOMETRY_V3.corridor_high_z_mm / 2.0),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=True,
        enable_observer_camera=False,
    )
    return SlotRuntime(
        slot=slot_id,
        course=course,
        world=world,
        body=body,
        periphery=WholeBodyPeriphery(args.wing_motor_map, args.body_motor_map),
        vision=MaleCNSRetina(
            args.retinotopic_map,
            current_gain=args.photoreceptor_current_gain,
        ),
    )


def begin_episode(
    slot: SlotRuntime,
    *,
    episode: int,
    source_weight_version: int,
    condition: SpawnCondition,
    boundary_ease_level: float | None = None,
    boundary_attempt_index: int | None = None,
) -> None:
    slot.active = True
    slot.episode = episode
    slot.source_weight_version = source_weight_version
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
    slot.last_retinal = None
    slot.last_peripheral = None
    slot.last_body_spikes = {}

    slot.course.reset()
    slot.body.reset()
    slot.body.set_root_position_mm((slot.spawn_x_mm, 0.0, slot.spawn_z_mm))
    slot.body.set_root_linear_velocity_mm_s((slot.initial_speed_mm_s, 0.0, 0.0))
    slot.periphery.reset()
    slot.vision.reset_adaptation()
    velocity = slot.body.root_linear_velocity_mm_s()
    slot.final_velocity = tuple(float(value) for value in velocity)


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
        "environment_version": "v3",
        "motor_boundary": "whole-body",
        "curriculum_mode": curriculum_mode,
        "boundary_ease_level": slot.boundary_ease_level,
        "boundary_attempt_index": slot.boundary_attempt_index,
        "reward_events": slot.reward_events,
        "aversive_events": slot.aversive_events,
    }


def main() -> int:
    args = parse_args()
    probe_course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count, environment_version="v3")
    first_gate = probe_course.gates[0]
    validate(args, first_gate)

    output = args.output_dir
    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint"
    state_path = output / "curriculum-state.json"
    population_state_path = output / "population-state.json"
    trajectory_path = output / "trajectory.jsonl"
    commit_log_path = output / "commit-log.jsonl"
    summary_path = output / "summary.json"
    checkpoint_exists = (checkpoint / "manifest.json").exists()

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
    state["environment_version"] = "v3"
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

    population_state: dict[str, object] = {}
    if checkpoint_exists and population_state_path.exists():
        population_state = json.loads(population_state_path.read_text(encoding="utf-8"))
    initial_global_version = int(population_state.get("global_weight_version", 0))

    start_episode = infer_next_episode(trajectory_path) if checkpoint_exists else 0
    trajectory_mode = "a" if checkpoint_exists else "w"
    commit_mode = "a" if checkpoint_exists and commit_log_path.exists() else "w"
    viewer_ids = load_viewer_body_ids(args.viewer_graph) if args.telemetry else ()
    publisher = LiveTelemetryPublisher(output, enabled=args.telemetry)
    slots = [make_slot(args, index) for index in range(args.population)]
    control_dt_s = slots[0].body.timestep * args.physics_steps

    print(
        "population_runtime=enabled population={} shared_weight=true weight_averaging=false "
        "environment=v3 motor_boundary=whole-body telemetry={}".format(
            args.population,
            args.telemetry,
        )
    )

    launched = 0
    completed = 0
    results: list[dict[str, object]] = []
    commit_records: list[dict[str, object]] = []
    saved_state: dict[str, object] = {}
    started = time.perf_counter()
    aggregate_control_steps = 0

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

            def launch_slot(slot: SlotRuntime) -> bool:
                nonlocal launched
                if launched >= args.episodes:
                    return False
                ease_level: float | None = None
                attempt_index: int | None = None
                if boundary is None:
                    condition = current_adaptive_condition(state)
                else:
                    payload = ensure_boundary_state(state, boundary)
                    completed_attempts = {
                        int(value)
                        for value in payload.get("completed_attempt_indices", [])
                    }
                    attempt_index = next(
                        (
                            attempt
                            for attempt in range(boundary.batch_size)
                            if attempt not in completed_attempts
                            and attempt not in boundary_issued_attempts
                        ),
                        None,
                    )
                    if attempt_index is None:
                        return False
                    condition, ease_level = boundary_condition_for_attempt(
                        state,
                        boundary,
                        attempt_index,
                    )
                    boundary_issued_attempts.add(attempt_index)
                episode = start_episode + launched
                begin_episode(
                    slot,
                    episode=episode,
                    source_weight_version=brain.global_weight_version,
                    condition=condition,
                    boundary_ease_level=ease_level,
                    boundary_attempt_index=attempt_index,
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
                    slot.last_retinal = retinal
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
                            "stimulate_body": retinal.body_currents,
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
                    slot.body.step_muscles(peripheral, physics_steps=args.physics_steps)

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

                    physical_collision = slot.world.physical_collision_reason(slot.body.sim)
                    event = slot.course.update(
                        x_mm,
                        z_mm,
                        physical_collision_reason=physical_collision,
                        analytic_body_collision=False,
                    )
                    reward = False
                    aversive = False
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
                        brain.step_slot(
                            slot.slot,
                            stimulate={"aversive_dan": args.aversive_current},
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
                                    "passed_gate": event.passed_gate,
                                    "collision": event.collision,
                                    "collision_reason": event.collision_reason,
                                    "finished": event.finished,
                                    "environment_version": "v3",
                                    "motor_boundary": "whole-body",
                                    "curriculum_mode": args.curriculum_mode,
                                    "boundary_ease_level": slot.boundary_ease_level,
                                    "boundary_attempt_index": slot.boundary_attempt_index,
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
                        batch_completed = record_boundary_result(
                            state,
                            boundary,
                            success=slot.passed_gates > 0,
                            attempt_index=slot.boundary_attempt_index,
                            group=slot.slot,
                        )
                        if batch_completed is not None:
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
                        "boundary_ease_level": slot.boundary_ease_level,
                        "boundary_attempt_index": slot.boundary_attempt_index,
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

                    checkpoint_due = completed % args.checkpoint_every == 0
                    if checkpoint_due:
                        saved_state = brain.save_checkpoint(checkpoint)
                        save_json_atomic(state_path, state)
                        save_json_atomic(
                            population_state_path,
                            {
                                "schema_version": 1,
                                "population": args.population,
                                "global_weight_version": brain.global_weight_version,
                                "commit_semantics": "episode-local additive+clamp transaction rebased onto latest global weight",
                                "weight_averaging": False,
                                "curriculum_mode": args.curriculum_mode,
                            },
                        )

                    if boundary is None and launched < args.episodes:
                        launch_slot(slot)
                    else:
                        slot.active = False

                if boundary is not None and launched < args.episodes:
                    for idle_slot in slots:
                        if launched >= args.episodes:
                            break
                        if idle_slot.active:
                            continue
                        launch_slot(idle_slot)

            saved_state = brain.save_checkpoint(checkpoint)
            save_json_atomic(state_path, state)
            save_json_atomic(
                population_state_path,
                {
                    "schema_version": 1,
                    "population": args.population,
                    "global_weight_version": brain.global_weight_version,
                    "commit_semantics": "episode-local additive+clamp transaction rebased onto latest global weight",
                    "weight_averaging": False,
                    "curriculum_mode": args.curriculum_mode,
                },
            )
            final_global_version = brain.global_weight_version
            backend_name = str(brain.ready.get("backend", "gpu-population"))

    elapsed = time.perf_counter() - started
    staleness_values = [int(item["staleness"]) for item in commit_records]
    summary = {
        "schema_version": 11,
        "experiment": "flyppy_v3_async_shared_weight_population",
        "backend": backend_name,
        "persistent_runtime": True,
        "population": args.population,
        "shared_weight": True,
        "weight_averaging": False,
        "commit_semantics": "episode-local additive+clamp transaction rebased onto latest global weight",
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
