#!/usr/bin/env python3
"""Persistent high-throughput Flyppy curriculum trainer.

One Python process, one MuJoCo simulation, and one MaleCNS runtime remain alive
for the whole run. Curriculum changes only episode-reset initial conditions.
Live visualization is completely external and consumes atomically published
telemetry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

from virtual_fly.physics import CANONICAL_FLY, FLYPPY_GEOMETRY
from virtual_fly.training.curriculum import (
    AdaptiveCurriculumConfig,
    BoundaryBandConfig,
    SpawnCondition,
    current_adaptive_condition,
    current_boundary_condition,
    load_state as load_curriculum_state,
    record_adaptive_result,
    record_boundary_result,
)

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flybody_neuromuscular_adapter import FlyBodyNeuromuscularAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from live_telemetry import LiveTelemetryPublisher
from malecns_retina import MaleCNSRetina
from neural_bridge_client import NeuralBridgeClient
from whole_body_periphery import WholeBodyPeriphery
from wing_muscle_periphery import WingMusclePeriphery


DEFAULT_OUTPUT = Path("artifacts/experiments/flyppy-v1")
DEFAULT_VIEWER_GRAPH = Path("artifacts/embodiment/neural-viewer-graph-v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument("--groups", type=Path, default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"))
    parser.add_argument("--retinotopic-map", type=Path, default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"))
    parser.add_argument("--wing-motor-map", type=Path, default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"))
    parser.add_argument("--body-motor-map", type=Path, default=Path("artifacts/malecns-v1.0/body-motor-neurons-v0.json"))
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--viewer-graph", type=Path, default=DEFAULT_VIEWER_GRAPH)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-control-steps", type=int, default=1800)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument(
        "--environment-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 preserves historical geometry; v2 uses the explicit canonical physical scale",
    )
    parser.add_argument(
        "--motor-boundary",
        choices=("wing-only", "whole-body"),
        default="wing-only",
        help="whole-body adds only anatomically grounded leg/hDVM motor outputs",
    )
    parser.add_argument(
        "--telemetry-stride",
        type=int,
        default=5,
        help="MaleCNS/activity telemetry cadence in control steps",
    )
    parser.add_argument(
        "--body-telemetry-stride",
        type=int,
        default=1,
        help="FlyBody pose telemetry cadence; keep at 1 for smooth detached rendering",
    )
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--photoreceptor-current-gain", type=float, default=2.0)
    parser.add_argument("--reward-current", type=float, default=2.0)
    parser.add_argument("--aversive-current", type=float, default=2.0)
    parser.add_argument("--reinforcement-steps", type=int, default=4)

    parser.add_argument(
        "--curriculum-mode",
        choices=("adaptive", "boundary-band"),
        default="adaptive",
        help="episode-reset curriculum policy; never changes the CNS learning rule",
    )
    parser.add_argument("--curriculum-start-x-mm", type=float, default=5.5)
    parser.add_argument("--curriculum-target-x-mm", type=float, default=0.0)
    parser.add_argument("--curriculum-x-step-mm", type=float, default=0.75)
    parser.add_argument("--curriculum-target-z-mm", type=float, default=5.0)
    parser.add_argument("--curriculum-z-step-mm", type=float, default=0.25)
    parser.add_argument("--curriculum-start-speed-mm-s", type=float, default=400.0)
    parser.add_argument("--curriculum-target-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--curriculum-speed-step-mm-s", type=float, default=25.0)
    parser.add_argument("--curriculum-max-easy-speed-mm-s", type=float, default=500.0)
    parser.add_argument("--curriculum-failure-x-step-mm", type=float, default=0.5)

    # Historical v1 gate-2 boundary band. v2 starts a new experiment and does
    # not reuse these values unless explicitly requested.
    parser.add_argument("--boundary-hard-x-mm", type=float, default=10.500)
    parser.add_argument("--boundary-hard-z-mm", type=float, default=5.290)
    parser.add_argument("--boundary-hard-speed-mm-s", type=float, default=350.0)
    parser.add_argument("--boundary-easy-x-mm", type=float, default=10.625)
    parser.add_argument("--boundary-easy-z-mm", type=float, default=5.3525)
    parser.add_argument("--boundary-easy-speed-mm-s", type=float, default=356.25)
    parser.add_argument("--boundary-batch-size", type=int, default=24)
    parser.add_argument("--boundary-harden-success-rate", type=float, default=0.80)
    parser.add_argument("--boundary-ease-success-rate", type=float, default=0.40)
    parser.add_argument("--boundary-harden-x-step-mm", type=float, default=0.125)
    parser.add_argument("--boundary-harden-z-step-mm", type=float, default=0.0625)
    parser.add_argument("--boundary-harden-speed-step-mm-s", type=float, default=6.25)
    parser.add_argument("--boundary-ease-x-step-mm", type=float, default=0.0625)
    parser.add_argument("--boundary-ease-z-step-mm", type=float, default=0.03125)
    parser.add_argument("--boundary-ease-speed-step-mm-s", type=float, default=3.125)

    parser.add_argument("--render", action="store_true", help="deprecated; use scripts/dev/view_flyppy.sh")
    parser.add_argument("--record-video", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--synapse-trace", action="store_true", help=argparse.SUPPRESS)
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


def deliver_reinforcement(brain: NeuralBridgeClient, group: str, current: float, steps: int) -> None:
    brain.step(stimulate={group: current}, read=(), plasticity=True, steps=steps)


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


def boundary_config(args: argparse.Namespace) -> BoundaryBandConfig:
    return BoundaryBandConfig(
        hard=SpawnCondition(
            args.boundary_hard_x_mm,
            args.boundary_hard_z_mm,
            args.boundary_hard_speed_mm_s,
        ),
        easy=SpawnCondition(
            args.boundary_easy_x_mm,
            args.boundary_easy_z_mm,
            args.boundary_easy_speed_mm_s,
        ),
        target=target_condition(args),
        batch_size=args.boundary_batch_size,
        harden_success_rate=args.boundary_harden_success_rate,
        ease_success_rate=args.boundary_ease_success_rate,
        harden_step=SpawnCondition(
            args.boundary_harden_x_step_mm,
            args.boundary_harden_z_step_mm,
            args.boundary_harden_speed_step_mm_s,
        ),
        ease_step=SpawnCondition(
            args.boundary_ease_x_step_mm,
            args.boundary_ease_z_step_mm,
            args.boundary_ease_speed_step_mm_s,
        ),
        seed=args.seed,
    )


def validate(args: argparse.Namespace, first_gate) -> None:
    if args.episodes < 1:
        raise SystemExit("episodes must be >= 1")
    if args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("control/physics steps must be >= 1")
    if args.trajectory_stride < 1 or args.telemetry_stride < 1 or args.body_telemetry_stride < 1:
        raise SystemExit("trajectory/telemetry strides must be >= 1")
    if args.checkpoint_every < 1:
        raise SystemExit("checkpoint-every must be >= 1")
    if args.motor_boundary == "whole-body" and not args.body_motor_map.exists():
        raise SystemExit(f"whole-body motor map not found: {args.body_motor_map}")
    if args.curriculum_mode == "adaptive" and args.curriculum_start_x_mm >= first_gate.x_mm:
        raise SystemExit("curriculum-start-x-mm must be before the first gate")
    if args.curriculum_mode == "boundary-band":
        if args.boundary_hard_x_mm >= first_gate.x_mm or args.boundary_easy_x_mm >= first_gate.x_mm:
            raise SystemExit("boundary-band spawn x must be before the target gate")
        try:
            probe_state: dict[str, object] = {}
            current_boundary_condition(probe_state, boundary_config(args))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    positive = (
        args.photoreceptor_current_gain,
        args.reward_current,
        args.aversive_current,
        args.reinforcement_steps,
        args.curriculum_x_step_mm,
        args.curriculum_z_step_mm,
        args.curriculum_start_speed_mm_s,
        args.curriculum_target_speed_mm_s,
        args.curriculum_speed_step_mm_s,
        args.curriculum_max_easy_speed_mm_s,
        args.curriculum_failure_x_step_mm,
    )
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
        raise SystemExit("positive training parameters must be finite and > 0")


def main() -> int:
    args = parse_args()
    course = FlyppyCourse(
        seed=args.seed,
        gate_count=args.gate_count,
        environment_version=args.environment_version,
    )
    first_gate = course.gates[0]
    validate(args, first_gate)

    if args.render:
        print("note: --render is detached; run scripts/dev/view_flyppy.sh in another terminal")

    output = args.output_dir
    checkpoint = output / "checkpoint"
    state_path = output / "curriculum-state.json"
    trajectory_path = output / "trajectory.jsonl"
    summary_path = output / "summary.json"

    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

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
    state["environment_version"] = args.environment_version
    state["motor_boundary"] = args.motor_boundary
    adaptive = adaptive_config(args, first_gate)
    boundary = boundary_config(args)

    start_episode = infer_next_episode(trajectory_path) if checkpoint_exists else 0
    trajectory_mode = "a" if checkpoint_exists else "w"
    viewer_ids = load_viewer_body_ids(args.viewer_graph)
    publisher = LiveTelemetryPublisher(output)

    world = FlyppyWorld(course)
    if args.motor_boundary == "whole-body":
        body = FlyBodyNeuromuscularAdapter(
            tethered=False,
            world=world,
            spawn_position_mm=(0.0, 0.0, FLYPPY_GEOMETRY.corridor_high_z_mm / 2.0),
            initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
            enable_vision=True,
            enable_observer_camera=False,
        )
        periphery = WholeBodyPeriphery(args.wing_motor_map, args.body_motor_map)
    else:
        body = FlyBodyMuscleAdapter(
            tethered=False,
            world=world,
            spawn_position_mm=(0.0, 0.0, 5.0),
            initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
            enable_vision=True,
            enable_observer_camera=False,
        )
        periphery = WingMusclePeriphery(args.wing_motor_map)
    vision = MaleCNSRetina(args.retinotopic_map, current_gain=args.photoreceptor_current_gain)
    control_dt_s = body.timestep * args.physics_steps

    print(
        f"persistent_runtime=enabled episodes_per_process={args.episodes} "
        f"telemetry_nodes={len(viewer_ids)} neural_telemetry_stride={args.telemetry_stride} "
        f"body_telemetry_stride={args.body_telemetry_stride}"
    )
    print(
        "environment={} motor_boundary={} body_length_mm={:.3f} mass_mg={:.3f} "
        "first_gate_x={:.3f} gate_gap_mm={:.3f} corridor_height_mm={:.3f}".format(
            args.environment_version,
            args.motor_boundary,
            CANONICAL_FLY.body_length_mm,
            CANONICAL_FLY.total_mass_mg,
            first_gate.x_mm,
            2.0 * first_gate.half_gap_mm,
            course.ceiling_z_mm - course.floor_z_mm,
        )
    )
    if body.mass_normalization is not None:
        print(f"mass_normalization={body.mass_normalization.as_dict()}")
    print(
        "curriculum mode={} gate_center_z={:.3f} target=(x={:.3f},z={:.3f},vx={:.1f})".format(
            args.curriculum_mode,
            first_gate.center_z_mm,
            args.curriculum_target_x_mm,
            args.curriculum_target_z_mm,
            args.curriculum_target_speed_mm_s,
        )
    )

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    saved_state: dict[str, object] = {}
    backend_name = args.backend

    with trajectory_path.open(trajectory_mode, encoding="utf-8") as trajectory:
        with NeuralBridgeClient(snapshot=args.snapshot, groups=args.groups, backend=args.backend) as brain:
            brain.ping()
            backend_name = str(brain.ready.get("backend", args.backend))
            print(
                "neural_backend={} neurons={} edges={}".format(
                    backend_name, brain.ready.get("neurons"), brain.ready.get("edges")
                )
            )
            if checkpoint_exists:
                loaded = brain.load_checkpoint(checkpoint)
                print(f"resumed_checkpoint={loaded.get('path')} neural_step={loaded.get('step')}")

            for local_episode in range(args.episodes):
                episode = start_episode + local_episode
                boundary_level: float | None = None
                if args.curriculum_mode == "boundary-band":
                    condition, boundary_level = current_boundary_condition(state, boundary)
                else:
                    condition = current_adaptive_condition(state)
                spawn_x = float(condition.x_mm)
                spawn_z = float(condition.z_mm)
                speed = float(condition.speed_mm_s)

                course.reset()
                body.reset()
                body.set_root_position_mm((spawn_x, 0.0, spawn_z))
                body.set_root_linear_velocity_mm_s((speed, 0.0, 0.0))
                periphery.reset()
                vision.reset_adaptation()
                brain.reset_dynamics()

                live_curriculum = {
                    "mode": args.curriculum_mode,
                    "environment_version": args.environment_version,
                    "motor_boundary": args.motor_boundary,
                    "spawn_x_mm": spawn_x,
                    "spawn_z_mm": spawn_z,
                    "initial_speed_mm_s": speed,
                }
                if boundary_level is not None:
                    live_curriculum["boundary_ease_level"] = boundary_level
                    live_curriculum["boundary_batch_number"] = int(
                        state.get("boundary_batch_number", 0)
                    )
                publisher.publish_status(
                    running=True,
                    backend=backend_name,
                    episode=episode,
                    control_step=0,
                    curriculum=live_curriculum,
                )
                level_text = "" if boundary_level is None else f" boundary_ease={boundary_level:.2f}"
                print(
                    "curriculum_episode={} mode={} spawn=(x={:.3f},z={:.3f}) "
                    "initial_vx={:.1f}{}".format(
                        int(state["curriculum_episodes"]),
                        args.curriculum_mode,
                        spawn_x,
                        spawn_z,
                        speed,
                        level_text,
                    )
                )

                passed = 0
                collision = False
                collision_reason = None
                finished = False
                max_x = float("-inf")
                min_z = float("inf")
                max_z = float("-inf")
                final_velocity = body.root_linear_velocity_mm_s()
                step_count = 0

                for control_step in range(args.max_control_steps):
                    neural_telemetry_sample = control_step % args.telemetry_stride == 0
                    body_telemetry_sample = control_step % args.body_telemetry_stride == 0
                    if neural_telemetry_sample and viewer_ids:
                        read_body = tuple(dict.fromkeys((*periphery.body_ids, *viewer_ids)))
                    else:
                        read_body = periphery.body_ids

                    retinal = vision.encode(body.sim, body.fly)
                    _, body_spikes = brain.step_with_body_readout(
                        stimulate_body=retinal.body_currents,
                        read=(),
                        read_body=read_body,
                        plasticity=True,
                    )
                    peripheral = periphery.step(body_spikes, dt_s=control_dt_s)
                    body.step_muscles(peripheral, physics_steps=args.physics_steps)

                    position = body.thorax_position_mm()
                    final_velocity = body.root_linear_velocity_mm_s()
                    x_mm = float(position[0])
                    z_mm = float(position[2])
                    max_x = max(max_x, x_mm)
                    min_z = min(min_z, z_mm)
                    max_z = max(max_z, z_mm)
                    step_count = control_step + 1

                    physical_collision = (
                        world.physical_collision_reason(body.sim)
                        if args.environment_version == "v2"
                        else None
                    )
                    event = course.update(
                        x_mm,
                        z_mm,
                        physical_collision_reason=physical_collision,
                        analytic_body_collision=args.environment_version == "v1",
                    )
                    reward = False
                    aversive = False
                    if event.passed_gate:
                        passed += 1
                        deliver_reinforcement(brain, "reward_dan", args.reward_current, args.reinforcement_steps)
                        reward = True
                    if event.collision:
                        collision = True
                        collision_reason = event.collision_reason
                        deliver_reinforcement(brain, "aversive_dan", args.aversive_current, args.reinforcement_steps)
                        aversive = True
                    if event.finished:
                        finished = True

                    next_gate = getattr(course, "absolute_next_gate_index", course.next_gate_index)

                    if control_step % args.trajectory_stride == 0 or reward or aversive or finished:
                        trajectory.write(
                            json.dumps(
                                {
                                    "episode": episode,
                                    "control_step": control_step,
                                    "x_mm": x_mm,
                                    "y_mm": float(position[1]),
                                    "z_mm": z_mm,
                                    "vx_mm_s": float(final_velocity[0]),
                                    "vy_mm_s": float(final_velocity[1]),
                                    "vz_mm_s": float(final_velocity[2]),
                                    "next_gate": next_gate,
                                    "motor_periphery": peripheral.compact_diagnostics(),
                                    "retinal_input": {
                                        "active_columns": retinal.active_columns,
                                        "active_photoreceptors": retinal.active_photoreceptors,
                                        "mean_current": retinal.mean_current,
                                        "max_current": retinal.max_current,
                                    },
                                    "reward_stimulated": reward,
                                    "aversive_stimulated": aversive,
                                    "passed_gate": event.passed_gate,
                                    "collision": event.collision,
                                    "collision_reason": event.collision_reason,
                                    "finished": event.finished,
                                    "environment_version": args.environment_version,
                                    "motor_boundary": args.motor_boundary,
                                    "curriculum_mode": args.curriculum_mode,
                                    "boundary_ease_level": boundary_level,
                                },
                                separators=(",", ":"),
                            ) + "\n"
                        )

                    if neural_telemetry_sample or reward or aversive or finished:
                        active_viewer = [
                            body_id
                            for body_id in viewer_ids
                            if body_spikes.get(body_id, False)
                        ]
                        publisher.publish_neural(
                            episode=episode,
                            control_step=control_step,
                            neural_step=brain.last_step,
                            depolarizing_body_ids=active_viewer,
                            hyperpolarizing_body_ids=(),
                            reward=reward,
                            aversive=aversive,
                        )

                    if body_telemetry_sample or reward or aversive or finished:
                        publisher.publish_body(
                            episode=episode,
                            control_step=control_step,
                            sim=body.sim,
                            next_gate=next_gate,
                            passed_gate=event.passed_gate,
                            collision=event.collision,
                            collision_reason=event.collision_reason,
                            reward=reward,
                            aversive=aversive,
                            motor=peripheral.compact_diagnostics(),
                            retinal={
                                "active_columns": retinal.active_columns,
                                "active_photoreceptors": retinal.active_photoreceptors,
                                "mean_current": retinal.mean_current,
                                "max_current": retinal.max_current,
                            },
                        )

                    if collision or finished:
                        break

                trajectory.flush()
                state["curriculum_episodes"] = int(state["curriculum_episodes"]) + 1
                batch_result = None
                if args.curriculum_mode == "boundary-band":
                    batch_result = record_boundary_result(state, boundary, success=passed > 0)
                    current_boundary_condition(state, boundary)
                else:
                    record_adaptive_result(state, adaptive, success=passed > 0)

                result = {
                    "episode": episode,
                    "control_steps": step_count,
                    "passed_gates": passed,
                    "collision": collision,
                    "collision_reason": collision_reason,
                    "finished": finished,
                    "max_x_mm": max_x,
                    "min_z_mm": min_z,
                    "max_z_mm": max_z,
                    "final_vx_mm_s": float(final_velocity[0]),
                    "spawn_x_mm": spawn_x,
                    "spawn_z_mm": spawn_z,
                    "initial_speed_mm_s": speed,
                    "environment_version": args.environment_version,
                    "motor_boundary": args.motor_boundary,
                    "curriculum_mode": args.curriculum_mode,
                    "boundary_ease_level": boundary_level,
                }
                results.append(result)
                print(
                    "episode={} steps={} passed={} collision={} finished={} max_x={:.3f} final_vx={:.3f}".format(
                        episode,
                        step_count,
                        passed,
                        collision,
                        finished,
                        max_x,
                        float(final_velocity[0]),
                    )
                )
                if batch_result is not None:
                    print(
                        "boundary_batch successes={}/{} rate={:.3f} adjustment={} hard={} easy={}".format(
                            batch_result["successes"],
                            batch_result["attempts"],
                            batch_result["success_rate"],
                            batch_result["adjustment"],
                            batch_result["hard"],
                            batch_result["easy"],
                        )
                    )
                print(
                    "curriculum_result passed={} next_spawn=(x={:.3f},z={:.3f}) next_vx={:.1f} complete={}".format(
                        passed,
                        float(state["spawn_x_mm"]),
                        float(state["spawn_z_mm"]),
                        float(state["initial_speed_mm_s"]),
                        state["curriculum_complete"],
                    )
                )

                checkpoint_due = (
                    (local_episode + 1) % args.checkpoint_every == 0
                    or local_episode + 1 == args.episodes
                )
                if checkpoint_due:
                    print(f"saving full CNS checkpoint after episode {episode} ...")
                    saved_state = brain.save_checkpoint(checkpoint)
                    save_json_atomic(state_path, state)
                    print(
                        f"checkpoint={saved_state.get('path')} neural_step={saved_state.get('step')}"
                    )

                publisher.publish_status(
                    running=True,
                    backend=backend_name,
                    episode=episode,
                    control_step=step_count,
                    curriculum=state,
                )

    elapsed = time.perf_counter() - started
    publisher.publish_status(
        running=False,
        backend=backend_name,
        episode=results[-1]["episode"] if results else None,
        control_step=results[-1]["control_steps"] if results else None,
        curriculum=state,
    )
    physical_spec = {
        "morphology": CANONICAL_FLY.morphology,
        "body_length_mm": CANONICAL_FLY.body_length_mm,
        "wing_length_mm": CANONICAL_FLY.wing_length_mm,
        "total_mass_mg": CANONICAL_FLY.total_mass_mg,
        "neural_sex": CANONICAL_FLY.neural_sex,
        "morphology_sex": CANONICAL_FLY.morphology_sex,
        "mass_normalization": (
            body.mass_normalization.as_dict() if body.mass_normalization is not None else None
        ),
    }
    environment_spec = {
        "version": args.environment_version,
        "corridor_low_z_mm": course.floor_z_mm,
        "corridor_high_z_mm": course.ceiling_z_mm,
        "lateral_half_width_mm": course.lateral_half_width_mm,
        "first_gate_x_mm": first_gate.x_mm,
        "gate_gap_height_mm": 2.0 * first_gate.half_gap_mm,
        "gate_half_thickness_mm": first_gate.half_thickness_mm,
    }
    summary = {
        "schema_version": 9,
        "experiment": (
            "flyppy_v2_whole_body_physical_curriculum"
            if args.environment_version == "v2" and args.motor_boundary == "whole-body"
            else "flyppy_persistent_curriculum_individual_wing_mn"
        ),
        "backend": backend_name,
        "persistent_runtime": True,
        "environment_version": args.environment_version,
        "motor_boundary": args.motor_boundary,
        "physical_spec": physical_spec,
        "environment_spec": environment_spec,
        "curriculum_mode": args.curriculum_mode,
        "episodes_this_run": args.episodes,
        "episode_start": start_episode,
        "episode_end": start_episode + args.episodes - 1,
        "control_dt_seconds": control_dt_s,
        "total_passed_gates_this_run": sum(int(item["passed_gates"]) for item in results),
        "total_collisions_this_run": sum(bool(item["collision"]) for item in results),
        "curriculum": state,
        "episode_results": results,
        "full_cns_checkpoint": str(checkpoint),
        "checkpoint_neural_step": saved_state.get("step"),
        "telemetry_dir": str(output / "live"),
        "viewer_graph": str(args.viewer_graph),
        "elapsed_seconds": elapsed,
        "teaching_signal": "gate pass -> PAM reward DAN current; collision -> PPL aversive DAN current; curriculum changes episode-reset initial conditions only",
    }
    save_json_atomic(summary_path, summary)
    print(f"summary={summary_path}")
    print(f"trajectory={trajectory_path}")
    print(f"checkpoint={checkpoint}")
    print(f"elapsed={elapsed:.3f}s")
    print("flyppy_persistent_curriculum=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
