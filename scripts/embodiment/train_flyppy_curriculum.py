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

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from live_telemetry import LiveTelemetryPublisher
from malecns_retina import MaleCNSRetina
from neural_bridge_client import NeuralBridgeClient
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
    parser.add_argument("--render", action="store_true", help="deprecated; use scripts/dev/view_flyppy.sh")
    parser.add_argument("--record-video", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--synapse-trace", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def move_toward(value: float, target: float, step: float) -> float:
    if value < target:
        return min(target, value + step)
    if value > target:
        return max(target, value - step)
    return target


def initial_state(args: argparse.Namespace, first_gate) -> dict[str, object]:
    return {
        "schema_version": 3,
        "spawn_x_mm": float(args.curriculum_start_x_mm),
        "spawn_z_mm": float(first_gate.center_z_mm),
        "initial_speed_mm_s": float(args.curriculum_start_speed_mm_s),
        "successful_first_gates": 0,
        "consecutive_failures": 0,
        "curriculum_episodes": 0,
        "target_x_mm": float(args.curriculum_target_x_mm),
        "target_z_mm": float(args.curriculum_target_z_mm),
        "target_speed_mm_s": float(args.curriculum_target_speed_mm_s),
        "curriculum_complete": False,
    }


def load_curriculum_state(path: Path, args: argparse.Namespace, first_gate, *, checkpoint_exists: bool) -> dict[str, object]:
    if not checkpoint_exists or not path.exists():
        return initial_state(args, first_gate)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return initial_state(args, first_gate)
    if int(state.get("schema_version", 0)) not in (2, 3):
        return initial_state(args, first_gate)
    state["schema_version"] = 3
    state.setdefault("spawn_x_mm", float(args.curriculum_start_x_mm))
    state.setdefault("spawn_z_mm", float(first_gate.center_z_mm))
    state.setdefault("initial_speed_mm_s", float(args.curriculum_start_speed_mm_s))
    state.setdefault("successful_first_gates", 0)
    state.setdefault("consecutive_failures", 0)
    state.setdefault("curriculum_episodes", 0)
    return state


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


def validate(args: argparse.Namespace, first_gate) -> None:
    if args.episodes < 1:
        raise SystemExit("episodes must be >= 1")
    if args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("control/physics steps must be >= 1")
    if args.trajectory_stride < 1 or args.telemetry_stride < 1 or args.body_telemetry_stride < 1:
        raise SystemExit("trajectory/telemetry strides must be >= 1")
    if args.checkpoint_every < 1:
        raise SystemExit("checkpoint-every must be >= 1")
    if args.curriculum_start_x_mm >= first_gate.x_mm:
        raise SystemExit("curriculum-start-x-mm must be before the first gate")
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
    course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count)
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
    state = load_curriculum_state(state_path, args, first_gate, checkpoint_exists=checkpoint_exists)
    start_episode = infer_next_episode(trajectory_path) if checkpoint_exists else 0
    trajectory_mode = "a" if checkpoint_exists else "w"
    viewer_ids = load_viewer_body_ids(args.viewer_graph)
    publisher = LiveTelemetryPublisher(output)

    world = FlyppyWorld(course)
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
        "curriculum first_gate_x={:.3f} gate_center_z={:.3f} target=(x={:.3f},z={:.3f},vx={:.1f})".format(
            first_gate.x_mm,
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
                spawn_x = float(state["spawn_x_mm"])
                spawn_z = float(state["spawn_z_mm"])
                speed = float(state["initial_speed_mm_s"])

                course.reset()
                body.reset()
                body.set_root_position_mm((spawn_x, 0.0, spawn_z))
                body.set_root_linear_velocity_mm_s((speed, 0.0, 0.0))
                periphery.reset()
                vision.reset_adaptation()
                brain.reset_dynamics()

                publisher.publish_status(
                    running=True,
                    backend=backend_name,
                    episode=episode,
                    control_step=0,
                    curriculum={"spawn_x_mm": spawn_x, "spawn_z_mm": spawn_z, "initial_speed_mm_s": speed},
                )
                print(
                    "curriculum_episode={} spawn=(x={:.3f},z={:.3f}) initial_vx={:.1f}".format(
                        int(state["curriculum_episodes"]), spawn_x, spawn_z, speed
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

                    event = course.update(x_mm, z_mm)
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
                                },
                                separators=(",", ":"),
                            ) + "\n"
                        )

                    if neural_telemetry_sample or reward or aversive or finished:
                        active_viewer = [body_id for body_id in viewer_ids if body_spikes.get(body_id, False)]
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
                if passed > 0:
                    state["successful_first_gates"] = int(state.get("successful_first_gates", 0)) + 1
                    state["consecutive_failures"] = 0
                    state["spawn_x_mm"] = move_toward(spawn_x, args.curriculum_target_x_mm, args.curriculum_x_step_mm)
                    state["spawn_z_mm"] = move_toward(spawn_z, args.curriculum_target_z_mm, args.curriculum_z_step_mm)
                    state["initial_speed_mm_s"] = move_toward(speed, args.curriculum_target_speed_mm_s, args.curriculum_speed_step_mm_s)
                else:
                    state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
                    state["spawn_x_mm"] = min(float(first_gate.x_mm) - 1.0, spawn_x + args.curriculum_failure_x_step_mm)
                    state["spawn_z_mm"] = move_toward(spawn_z, float(first_gate.center_z_mm), args.curriculum_z_step_mm)
                    state["initial_speed_mm_s"] = min(args.curriculum_max_easy_speed_mm_s, speed + args.curriculum_speed_step_mm_s)

                state["target_x_mm"] = float(args.curriculum_target_x_mm)
                state["target_z_mm"] = float(args.curriculum_target_z_mm)
                state["target_speed_mm_s"] = float(args.curriculum_target_speed_mm_s)
                state["curriculum_complete"] = bool(
                    math.isclose(float(state["spawn_x_mm"]), args.curriculum_target_x_mm, abs_tol=1e-9)
                    and math.isclose(float(state["spawn_z_mm"]), args.curriculum_target_z_mm, abs_tol=1e-9)
                    and math.isclose(float(state["initial_speed_mm_s"]), args.curriculum_target_speed_mm_s, abs_tol=1e-9)
                )

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
                }
                results.append(result)
                print(
                    "episode={} steps={} passed={} collision={} finished={} max_x={:.3f} final_vx={:.3f}".format(
                        episode, step_count, passed, collision, finished, max_x, float(final_velocity[0])
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

                checkpoint_due = (local_episode + 1) % args.checkpoint_every == 0 or local_episode + 1 == args.episodes
                if checkpoint_due:
                    print(f"saving full CNS checkpoint after episode {episode} ...")
                    saved_state = brain.save_checkpoint(checkpoint)
                    save_json_atomic(state_path, state)
                    print(f"checkpoint={saved_state.get('path')} neural_step={saved_state.get('step')}")

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
    summary = {
        "schema_version": 7,
        "experiment": "flyppy_persistent_curriculum_individual_wing_mn",
        "backend": backend_name,
        "persistent_runtime": True,
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
