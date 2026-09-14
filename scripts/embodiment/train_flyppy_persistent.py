#!/usr/bin/env python3
"""Fast persistent Flyppy curriculum learner.

FlyBody, the Rust neural bridge and the GPU runtime are constructed once and
kept alive across episodes. Only trial-local state is reset; learned synaptic
weights remain resident. Rendering is deliberately absent. Lightweight live
telemetry is published for independent observers and never advances neural time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from live_telemetry import LiveTelemetryPublisher
from malecns_retina import MaleCNSRetina
from neural_bridge_client import NeuralBridgeClient
from train_flyppy_curriculum import load_state, move_toward, new_state, save_state
from wing_muscle_periphery import WingMusclePeriphery


DEFAULT_OUTPUT = Path("artifacts/experiments/flyppy-v1")
DEFAULT_VIEWER_GRAPH = Path("artifacts/embodiment/neural-viewer-graph-v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument("--groups", type=Path, default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"))
    parser.add_argument("--retinotopic-map", type=Path, default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"))
    parser.add_argument("--wing-motor-map", type=Path, default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"))
    parser.add_argument("--viewer-graph", type=Path, default=DEFAULT_VIEWER_GRAPH)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-control-steps", type=int, default=1800)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument("--telemetry-stride", type=int, default=10)
    parser.add_argument("--no-live-telemetry", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--photoreceptor-current-gain", type=float, default=2.0)
    parser.add_argument("--reward-current", type=float, default=2.0)
    parser.add_argument("--aversive-current", type=float, default=2.0)
    parser.add_argument("--reinforcement-steps", type=int, default=4)
    parser.add_argument("--fresh", action="store_true")

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

    # Compatibility only. Visualization is now a separate observer process.
    parser.add_argument("--render", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--record-video", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def deliver_reinforcement(brain: NeuralBridgeClient, group: str, current: float, steps: int) -> None:
    brain.step(stimulate={group: current}, read=(), plasticity=True, steps=steps)


def viewer_body_ids(path: Path) -> tuple[int, ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(int(node["body_id"]) for node in payload.get("nodes", []))


def unique_body_ids(*groups: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for group in groups:
        for body_id in group:
            value = int(body_id)
            if value not in seen:
                seen.add(value)
                result.append(value)
    return tuple(result)


def validate(args: argparse.Namespace) -> None:
    if args.episodes < 1 or args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("episodes/control/physics steps must be >= 1")
    if args.trajectory_stride < 1 or args.telemetry_stride < 1 or args.checkpoint_every < 1:
        raise SystemExit("trajectory/telemetry/checkpoint stride must be >= 1")
    for path in (args.snapshot / "manifest.json", args.groups, args.retinotopic_map, args.wing_motor_map):
        if not path.exists():
            raise SystemExit(f"required artifact not found: {path}")
    for value in (
        args.photoreceptor_current_gain,
        args.reward_current,
        args.aversive_current,
        args.curriculum_x_step_mm,
        args.curriculum_z_step_mm,
        args.curriculum_start_speed_mm_s,
        args.curriculum_target_speed_mm_s,
        args.curriculum_speed_step_mm_s,
        args.curriculum_max_easy_speed_mm_s,
        args.curriculum_failure_x_step_mm,
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit("curriculum/current parameters must be finite and positive")
    if args.render or args.record_video is not None:
        print("note=inline-rendering-is-detached use scripts/dev/view_learning.sh in another terminal")


def main() -> int:
    args = parse_args()
    validate(args)

    course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count)
    first_gate = course.gates[0]
    if args.curriculum_start_x_mm >= first_gate.x_mm:
        raise SystemExit("curriculum-start-x-mm must be before the first gate")

    output_dir = args.output_dir
    checkpoint = output_dir / "checkpoint"
    state_path = output_dir / "curriculum-state.json"
    trajectory_path = output_dir / "trajectory.jsonl"
    summary_path = output_dir / "summary.json"
    if args.fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_x = float(args.curriculum_start_x_mm)
    default_z = float(first_gate.center_z_mm)
    default_speed = float(args.curriculum_start_speed_mm_s)
    state = load_state(
        state_path,
        default_x=default_x,
        default_z=default_z,
        default_speed=default_speed,
    )
    resuming = checkpoint.joinpath("manifest.json").exists() and not args.fresh
    if not resuming and not args.fresh:
        state = new_state(default_x=default_x, default_z=default_z, default_speed=default_speed)

    world = FlyppyWorld(course)
    body = FlyBodyMuscleAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, args.curriculum_target_z_mm),
        initial_linear_velocity_mm_s=(args.curriculum_target_speed_mm_s, 0.0, 0.0),
        enable_vision=True,
        enable_observer_camera=False,
    )
    periphery = WingMusclePeriphery(args.wing_motor_map)
    vision = MaleCNSRetina(args.retinotopic_map, current_gain=args.photoreceptor_current_gain)
    control_dt_s = body.timestep * args.physics_steps

    telemetry = LiveTelemetryPublisher(output_dir, enabled=not args.no_live_telemetry)
    graph_body_ids = viewer_body_ids(args.viewer_graph) if telemetry.enabled else ()
    telemetry_read_ids = unique_body_ids(tuple(periphery.body_ids), graph_body_ids)

    print(
        "persistent_runtime=FlyBody+MaleCNS+GPU kept alive across episodes "
        f"telemetry_nodes={len(graph_body_ids)} control_dt_s={control_dt_s:.6g}"
    )
    print(
        "curriculum first_gate_x={:.3f} gate_center_z={:.3f} target=(x={:.3f},z={:.3f},vx={:.1f})".format(
            float(first_gate.x_mm),
            float(first_gate.center_z_mm),
            args.curriculum_target_x_mm,
            args.curriculum_target_z_mm,
            args.curriculum_target_speed_mm_s,
        )
    )

    trajectory_mode = "a" if resuming else "w"
    run_results: list[dict[str, object]] = []
    run_started = time.perf_counter()
    backend_name = "unknown"

    with trajectory_path.open(trajectory_mode, encoding="utf-8", buffering=1) as trajectory_file:
        with NeuralBridgeClient(snapshot=args.snapshot, groups=args.groups, backend=args.backend) as brain:
            brain.ping()
            backend_name = str(brain.ready.get("backend", "unknown"))
            print(
                "neural_backend={} neurons={} edges={}".format(
                    backend_name,
                    brain.ready.get("neurons"),
                    brain.ready.get("edges"),
                )
            )
            if resuming:
                loaded = brain.load_checkpoint(checkpoint)
                print(f"resumed_checkpoint={checkpoint} neural_step={loaded.get('step')}")

            telemetry.publish_status(
                running=True,
                backend=backend_name,
                episode=None,
                control_step=None,
                curriculum=state,
            )

            for local_episode in range(args.episodes):
                episode = int(state.get("curriculum_episodes", 0))
                spawn_x = float(state["spawn_x_mm"])
                spawn_z = float(state["spawn_z_mm"])
                speed = float(state["initial_speed_mm_s"])
                print(
                    "curriculum_episode={} spawn=(x={:.3f},z={:.3f}) initial_vx={:.1f}".format(
                        episode, spawn_x, spawn_z, speed
                    )
                )

                reset_started = time.perf_counter()
                course.reset()
                body.reset()
                body.set_root_position_mm((spawn_x, 0.0, spawn_z))
                body.set_root_linear_velocity_mm_s((speed, 0.0, 0.0))
                periphery.reset()
                vision.reset_adaptation()
                brain.reset_dynamics()
                reset_seconds = time.perf_counter() - reset_started

                episode_started = time.perf_counter()
                episode_passed = 0
                collision = False
                finished = False
                max_x = float("-inf")
                final_velocity = body.root_linear_velocity_mm_s()
                step_count = 0
                last_viewer_spikes: dict[int, bool] = {}

                telemetry.publish_status(
                    running=True,
                    backend=backend_name,
                    episode=episode,
                    control_step=0,
                    curriculum=state,
                )

                for control_step in range(args.max_control_steps):
                    telemetry_due = telemetry.enabled and control_step % args.telemetry_stride == 0
                    read_ids = telemetry_read_ids if telemetry_due else tuple(periphery.body_ids)
                    retinal = vision.encode(body.sim, body.fly)
                    _, body_spikes = brain.step_with_body_readout(
                        stimulate_body=retinal.body_currents,
                        read=(),
                        read_body=read_ids,
                        plasticity=True,
                    )
                    if telemetry_due:
                        last_viewer_spikes = dict(body_spikes)
                    motor_spikes = {
                        body_id: bool(body_spikes.get(body_id, False))
                        for body_id in periphery.body_ids
                    }
                    peripheral_state = periphery.step(motor_spikes, dt_s=control_dt_s)
                    body.step_muscles(peripheral_state, physics_steps=args.physics_steps)

                    position = body.thorax_position_mm()
                    final_velocity = body.root_linear_velocity_mm_s()
                    x_mm = float(position[0])
                    z_mm = float(position[2])
                    max_x = max(max_x, x_mm)
                    step_count = control_step + 1
                    event = course.update(x_mm, z_mm)
                    reward = False
                    aversive = False
                    if event.passed_gate:
                        episode_passed += 1
                        deliver_reinforcement(
                            brain,
                            "reward_dan",
                            args.reward_current,
                            args.reinforcement_steps,
                        )
                        reward = True
                    if event.collision:
                        collision = True
                        deliver_reinforcement(
                            brain,
                            "aversive_dan",
                            args.aversive_current,
                            args.reinforcement_steps,
                        )
                        aversive = True
                    if event.finished:
                        finished = True

                    event_frame = reward or aversive or collision or finished
                    if telemetry.enabled and (telemetry_due or event_frame):
                        # Observer-only data must never advance the CNS. Event
                        # frames between periodic neural samples reuse the latest
                        # sampled activity and mark the DAN event separately.
                        viewer_spikes = body_spikes if telemetry_due else last_viewer_spikes
                        depolarizing = [
                            body_id
                            for body_id in graph_body_ids
                            if viewer_spikes.get(body_id, False)
                        ]
                        telemetry.publish_body(
                            episode=episode,
                            control_step=control_step,
                            sim=body.sim,
                            next_gate=course.next_gate_index,
                            passed_gate=bool(event.passed_gate),
                            collision=collision,
                            collision_reason=event.collision_reason,
                            reward=reward,
                            aversive=aversive,
                            motor=peripheral_state.compact_diagnostics(),
                            retinal={
                                "active_columns": retinal.active_columns,
                                "active_photoreceptors": retinal.active_photoreceptors,
                                "mean_current": retinal.mean_current,
                                "max_current": retinal.max_current,
                            },
                        )
                        telemetry.publish_neural(
                            episode=episode,
                            control_step=control_step,
                            neural_step=brain.last_step,
                            depolarizing_body_ids=depolarizing,
                            hyperpolarizing_body_ids=(),
                            reward=reward,
                            aversive=aversive,
                        )

                    if control_step % args.trajectory_stride == 0 or event_frame:
                        trajectory_file.write(
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
                                    "next_gate": course.next_gate_index,
                                    "motor_periphery": peripheral_state.compact_diagnostics(),
                                    "retinal_input": {
                                        "active_columns": retinal.active_columns,
                                        "active_photoreceptors": retinal.active_photoreceptors,
                                        "mean_current": retinal.mean_current,
                                        "max_current": retinal.max_current,
                                    },
                                    "reward_stimulated": reward,
                                    "aversive_stimulated": aversive,
                                    "passed_gate": bool(event.passed_gate),
                                    "collision": collision,
                                    "collision_reason": event.collision_reason,
                                    "finished": finished,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    if collision or finished:
                        break

                loop_seconds = time.perf_counter() - episode_started

                state["curriculum_episodes"] = episode + 1
                if episode_passed > 0:
                    state["successful_first_gates"] = int(
                        state.get("successful_first_gates", 0)
                    ) + 1
                    state["consecutive_failures"] = 0
                    state["spawn_x_mm"] = move_toward(
                        spawn_x,
                        args.curriculum_target_x_mm,
                        args.curriculum_x_step_mm,
                    )
                    state["spawn_z_mm"] = move_toward(
                        spawn_z,
                        args.curriculum_target_z_mm,
                        args.curriculum_z_step_mm,
                    )
                    state["initial_speed_mm_s"] = move_toward(
                        speed,
                        args.curriculum_target_speed_mm_s,
                        args.curriculum_speed_step_mm_s,
                    )
                else:
                    state["consecutive_failures"] = int(
                        state.get("consecutive_failures", 0)
                    ) + 1
                    easiest_x = float(first_gate.x_mm) - 1.0
                    state["spawn_x_mm"] = min(
                        easiest_x,
                        spawn_x + args.curriculum_failure_x_step_mm,
                    )
                    state["initial_speed_mm_s"] = min(
                        args.curriculum_max_easy_speed_mm_s,
                        speed + args.curriculum_speed_step_mm_s,
                    )

                state["target_x_mm"] = float(args.curriculum_target_x_mm)
                state["target_z_mm"] = float(args.curriculum_target_z_mm)
                state["target_speed_mm_s"] = float(args.curriculum_target_speed_mm_s)
                state["curriculum_complete"] = bool(
                    math.isclose(
                        float(state["spawn_x_mm"]),
                        args.curriculum_target_x_mm,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        float(state["spawn_z_mm"]),
                        args.curriculum_target_z_mm,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        float(state["initial_speed_mm_s"]),
                        args.curriculum_target_speed_mm_s,
                        abs_tol=1e-9,
                    )
                )

                checkpoint_seconds = 0.0
                checkpoint_saved = False
                if (
                    (local_episode + 1) % args.checkpoint_every == 0
                    or local_episode + 1 == args.episodes
                ):
                    checkpoint_started = time.perf_counter()
                    brain.save_checkpoint(checkpoint)
                    save_state(state_path, state)
                    checkpoint_seconds = time.perf_counter() - checkpoint_started
                    checkpoint_saved = True

                result = {
                    "episode": episode,
                    "control_steps": step_count,
                    "passed_gates": episode_passed,
                    "collision": collision,
                    "finished": finished,
                    "max_x_mm": max_x,
                    "final_vx_mm_s": float(final_velocity[0]),
                    "reset_seconds": reset_seconds,
                    "loop_seconds": loop_seconds,
                    "checkpoint_seconds": checkpoint_seconds,
                    "checkpoint_saved": checkpoint_saved,
                }
                run_results.append(result)
                print(
                    "episode={} steps={} passed={} collision={} max_x={:.3f} final_vx={:.3f} "
                    "timing(reset={:.3f}s loop={:.3f}s checkpoint={:.3f}s)".format(
                        episode,
                        step_count,
                        episode_passed,
                        collision,
                        max_x,
                        float(final_velocity[0]),
                        reset_seconds,
                        loop_seconds,
                        checkpoint_seconds,
                    )
                )
                print(
                    "curriculum_result next_spawn=(x={:.3f},z={:.3f}) next_vx={:.1f} "
                    "failures={} complete={}".format(
                        float(state["spawn_x_mm"]),
                        float(state["spawn_z_mm"]),
                        float(state["initial_speed_mm_s"]),
                        int(state["consecutive_failures"]),
                        state["curriculum_complete"],
                    )
                )

            telemetry.publish_status(
                running=False,
                backend=backend_name,
                episode=int(state.get("curriculum_episodes", 0)) - 1,
                control_step=run_results[-1]["control_steps"] if run_results else None,
                curriculum=state,
            )

    elapsed = time.perf_counter() - run_started
    summary = {
        "schema_version": 8,
        "experiment": "flyppy_persistent_curriculum_v2",
        "backend": backend_name,
        "episodes_this_run": len(run_results),
        "elapsed_seconds": elapsed,
        "persistent_runtime": True,
        "checkpoint_every": args.checkpoint_every,
        "live_telemetry": telemetry.enabled,
        "telemetry_affects_neural_steps": False,
        "viewer_graph": str(args.viewer_graph) if telemetry.enabled else None,
        "curriculum_state": state,
        "episode_results": run_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    print(f"trajectory={trajectory_path}")
    print(f"checkpoint={checkpoint}")
    print(f"curriculum_state={state_path}")
    print(f"elapsed={elapsed:.3f}s")
    print("flyppy_persistent_curriculum=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
