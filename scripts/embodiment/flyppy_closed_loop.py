#!/usr/bin/env python3
"""First embodied MaleCNS <-> FlyBody <-> Flyppy learning loop.

Information flow is deliberately constrained:

    physical Flyppy world
      -> FlyBody eye cameras
      -> local MaleCNS optic-column light sampling
      -> current into actual R1-R6 photoreceptor body IDs
      -> persistent MaleCNS runtime with local plasticity
      -> bilateral DNg02 population readout
      -> FlyBody wing-amplitude adapter
      -> physical world

The sensory boundary does not calculate motion, edges, gate position, or any other
visual feature. Spatial and temporal visual processing is left to the MaleCNS
network. Gate coordinates are never injected into the CNS. Passing a gate
stimulates the configured reward DAN population; collision stimulates the
configured aversive DAN population.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from flybody_adapter import FlyBodyWingAdapter, WingDrive
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from malecns_retina import MaleCNSRetina
from neural_bridge_client import NeuralBridgeClient
from synapse_monitor import SynapseMonitor, SynapseMonitorConfig
from training_visualizer import TrainingVisualizer


MOTOR_GROUPS = ("flight_thrust_left", "flight_thrust_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/malecns-v1.0")
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"),
    )
    parser.add_argument(
        "--retinotopic-map",
        type=Path,
        default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"),
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="number of episodes to run in this invocation",
    )
    parser.add_argument("--max-control-steps", type=int, default=1800)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument(
        "--initial-forward-speed-mm-s",
        type=float,
        default=300.0,
        help=(
            "one-shot +x velocity applied only at episode reset; 300 mm/s is the "
            "midpoint of the original FlyBody vision-flight 20-40 cm/s range"
        ),
    )
    parser.add_argument(
        "--photoreceptor-current-gain",
        type=float,
        default=2.0,
        help=(
            "calibrated local irradiance-to-current scale for R1-R6; this changes "
            "transduction strength only and does not compute visual features"
        ),
    )
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-current", type=float, default=2.0)
    parser.add_argument("--aversive-current", type=float, default=2.0)
    parser.add_argument("--reinforcement-steps", type=int, default=4)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v0"),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="restore full CNS dynamic/plasticity state before running new episodes",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="full CNS checkpoint directory (default: OUTPUT_DIR/checkpoint)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=8,
        metavar="EPISODES",
        help="save full CNS state every N episodes; final episode is always saved",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="show a live 3D tracking-camera view; off by default",
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        default=None,
        metavar="DIR",
        help="save one 3D MP4 per episode into DIR; off by default",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=0.2,
        help="3D playback speed relative to simulated time (default: 0.2x)",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--synapse-trace",
        action="store_true",
        help=(
            "at low frequency, dump current CNS weights and keep only aggregate "
            "plasticity statistics plus the largest changed synapses; off by default"
        ),
    )
    parser.add_argument("--synapse-top-n", type=int, default=64)
    parser.add_argument(
        "--synapse-trace-every",
        type=int,
        default=1,
        metavar="EPISODES",
        help="capture a synapse-change snapshot every N episodes",
    )
    parser.add_argument("--synapse-min-delta", type=float, default=1e-7)
    return parser.parse_args()


def deliver_reinforcement(
    brain: NeuralBridgeClient,
    group: str,
    current: float,
    steps: int,
) -> None:
    if current <= 0.0 or steps < 1:
        raise ValueError("reinforcement parameters must be positive")
    brain.step(
        stimulate={group: current},
        read=(),
        plasticity=True,
        steps=steps,
    )


def infer_next_episode(trajectory_path: Path) -> int:
    if not trajectory_path.exists():
        return 0
    highest = -1
    with trajectory_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            highest = max(highest, int(record.get("episode", -1)))
    return highest + 1


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("episodes must be >= 1")
    if args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("control/physics steps must be >= 1")
    if not np.isfinite(args.initial_forward_speed_mm_s) or args.initial_forward_speed_mm_s <= 0:
        raise SystemExit("initial-forward-speed-mm-s must be finite and > 0")
    if not np.isfinite(args.photoreceptor_current_gain) or args.photoreceptor_current_gain <= 0:
        raise SystemExit("photoreceptor-current-gain must be finite and > 0")
    if args.gate_count < 1 or args.trajectory_stride < 1:
        raise SystemExit("gate-count and trajectory-stride must be >= 1")
    if args.checkpoint_every < 1:
        raise SystemExit("checkpoint-every must be >= 1")
    if args.playback_speed <= 0.0 or args.video_fps <= 0:
        raise SystemExit("playback-speed and video-fps must be positive")
    if args.synapse_top_n < 1 or args.synapse_trace_every < 1:
        raise SystemExit("synapse-top-n and synapse-trace-every must be >= 1")
    if args.synapse_min_delta < 0.0:
        raise SystemExit("synapse-min-delta must be >= 0")
    if not args.retinotopic_map.exists():
        raise SystemExit(f"retinotopic vision map not found: {args.retinotopic_map}")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.exists():
        raise SystemExit(f"resume checkpoint not found: {args.resume_checkpoint}")

    visualization_enabled = args.render or args.record_video is not None
    course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count)
    world = FlyppyWorld(course)
    body = FlyBodyWingAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(args.initial_forward_speed_mm_s, 0.0, 0.0),
        enable_vision=True,
        enable_observer_camera=visualization_enabled,
    )
    vision = MaleCNSRetina(
        args.retinotopic_map,
        current_gain=args.photoreceptor_current_gain,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "trajectory.jsonl"
    summary_path = args.output_dir / "summary.json"
    synapse_trace_path = args.output_dir / "synapse-snapshots.jsonl"
    checkpoint_dir = args.checkpoint_dir or (args.output_dir / "checkpoint")
    resuming = args.resume_checkpoint is not None
    start_episode = infer_next_episode(trajectory_path) if resuming else 0

    if not resuming:
        trajectory_path.unlink(missing_ok=True)
        synapse_trace_path.unlink(missing_ok=True)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

    if args.synapse_trace:
        synapse_monitor: SynapseMonitor | None = SynapseMonitor(
            snapshot=args.snapshot,
            output=synapse_trace_path,
            config=SynapseMonitorConfig(
                top_n=args.synapse_top_n,
                min_abs_delta=args.synapse_min_delta,
            ),
        )
    else:
        synapse_monitor = None

    episode_results: list[dict[str, object]] = []
    total_passed = 0
    total_collisions = 0
    state_checkpoint_info: dict[str, object] = {}
    resume_info: dict[str, object] | None = None
    started = time.perf_counter()

    observer_camera = body.observer_camera_name or "training_view"
    with TrainingVisualizer(
        sim=body.sim,
        camera_name=observer_camera,
        live=args.render,
        record_dir=args.record_video,
        playback_speed=args.playback_speed,
        output_fps=args.video_fps,
    ) as visualizer:
        trajectory_mode = "a" if resuming else "w"
        with trajectory_path.open(trajectory_mode, encoding="utf-8") as trajectory_file:
            with NeuralBridgeClient(
                snapshot=args.snapshot,
                groups=args.groups,
                backend=args.backend,
            ) as brain:
                brain.ping()
                ready = dict(brain.ready)
                if args.resume_checkpoint is not None:
                    resume_info = brain.load_checkpoint(args.resume_checkpoint)
                    print(
                        "resumed_checkpoint={} neural_step={}".format(
                            resume_info.get("path"), resume_info.get("step")
                        )
                    )

                for local_episode in range(args.episodes):
                    episode = start_episode + local_episode
                    course.reset()
                    body.reset()
                    if visualizer.enabled:
                        visualizer.begin_episode(episode)
                    episode_passed = 0
                    collision = False
                    finished = False
                    max_x = float("-inf")
                    min_z = float("inf")
                    max_z = float("-inf")
                    final_velocity = body.root_linear_velocity_mm_s()
                    step_count = 0

                    for control_step in range(args.max_control_steps):
                        retinal = vision.encode(body.sim, body.fly)
                        readout = brain.step(
                            stimulate_body=retinal.body_currents,
                            read=MOTOR_GROUPS,
                            plasticity=True,
                        )
                        left = float(readout["flight_thrust_left"]["spike_fraction"])
                        right = float(readout["flight_thrust_right"]["spike_fraction"])

                        body.step(
                            WingDrive(left=left, right=right),
                            physics_steps=args.physics_steps,
                        )
                        if visualizer.enabled:
                            visualizer.sync()

                        position = body.thorax_position_mm()
                        final_velocity = body.root_linear_velocity_mm_s()
                        x_mm = float(position[0])
                        z_mm = float(position[2])
                        max_x = max(max_x, x_mm)
                        min_z = min(min_z, z_mm)
                        max_z = max(max_z, z_mm)
                        step_count = control_step + 1

                        event = course.update(x_mm, z_mm)
                        reward_stimulated = False
                        aversive_stimulated = False
                        if event.passed_gate:
                            episode_passed += 1
                            total_passed += 1
                            deliver_reinforcement(
                                brain,
                                "reward_dan",
                                args.reward_current,
                                args.reinforcement_steps,
                            )
                            reward_stimulated = True
                        if event.collision:
                            collision = True
                            total_collisions += 1
                            deliver_reinforcement(
                                brain,
                                "aversive_dan",
                                args.aversive_current,
                                args.reinforcement_steps,
                            )
                            aversive_stimulated = True
                        if event.finished:
                            finished = True

                        if (
                            control_step % args.trajectory_stride == 0
                            or event.passed_gate
                            or event.collision
                            or event.finished
                        ):
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
                                        "motor_left": left,
                                        "motor_right": right,
                                        "retinal_input": {
                                            "active_columns": retinal.active_columns,
                                            "active_photoreceptors": retinal.active_photoreceptors,
                                            "mean_current": retinal.mean_current,
                                            "max_current": retinal.max_current,
                                        },
                                        "reward_stimulated": reward_stimulated,
                                        "aversive_stimulated": aversive_stimulated,
                                        "passed_gate": event.passed_gate,
                                        "collision": event.collision,
                                        "finished": event.finished,
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )

                        if collision or finished:
                            break

                    video_path = None
                    if visualizer.enabled:
                        saved = visualizer.end_episode(save=True)
                        video_path = str(saved) if saved is not None else None

                    synapse_snapshot = None
                    if synapse_monitor is not None and (
                        (local_episode + 1) % args.synapse_trace_every == 0
                        or local_episode + 1 == args.episodes
                    ):
                        print(f"capturing synapse snapshot after episode {episode} ...")
                        synapse_snapshot = synapse_monitor.capture(
                            brain,
                            episode=episode,
                            control_step=step_count,
                        )
                        print(
                            "synapse_changed={} max_abs_delta={:.6f}".format(
                                synapse_snapshot["changed_synapses"],
                                synapse_snapshot["max_abs_delta"],
                            )
                        )

                    checkpoint_saved_this_episode = False
                    if (
                        (local_episode + 1) % args.checkpoint_every == 0
                        or local_episode + 1 == args.episodes
                    ):
                        print(f"saving full CNS checkpoint after episode {episode} ...")
                        state_checkpoint_info = brain.save_checkpoint(checkpoint_dir)
                        checkpoint_saved_this_episode = True
                        print(
                            "checkpoint={} neural_step={}".format(
                                state_checkpoint_info.get("path"),
                                state_checkpoint_info.get("step"),
                            )
                        )

                    result = {
                        "episode": episode,
                        "control_steps": step_count,
                        "passed_gates": episode_passed,
                        "collision": collision,
                        "finished": finished,
                        "max_x_mm": max_x,
                        "min_z_mm": min_z,
                        "max_z_mm": max_z,
                        "final_vx_mm_s": float(final_velocity[0]),
                        "final_vy_mm_s": float(final_velocity[1]),
                        "final_vz_mm_s": float(final_velocity[2]),
                        "video": video_path,
                        "synapse_snapshot": synapse_snapshot is not None,
                        "checkpoint_saved": checkpoint_saved_this_episode,
                    }
                    episode_results.append(result)
                    print(
                        "episode={} steps={} passed={} collision={} finished={} max_x={:.3f} final_vx={:.3f}".format(
                            episode,
                            step_count,
                            episode_passed,
                            collision,
                            finished,
                            max_x,
                            final_velocity[0],
                        )
                    )
                    trajectory_file.flush()

    elapsed = time.perf_counter() - started
    passed_by_episode = [int(item["passed_gates"]) for item in episode_results]
    first_half = passed_by_episode[: max(1, len(passed_by_episode) // 2)]
    second_half = passed_by_episode[len(passed_by_episode) // 2 :]
    checkpoint_weight_file = checkpoint_dir / "weights.f32le"
    summary = {
        "schema_version": 4,
        "experiment": "flyppy_closed_loop_v0",
        "backend": ready.get("backend"),
        "neurons": ready.get("neurons"),
        "edges": ready.get("edges"),
        "episodes_this_run": args.episodes,
        "episode_start": start_episode,
        "episode_end": start_episode + args.episodes - 1,
        "gate_count": args.gate_count,
        "physics_steps_per_control": args.physics_steps,
        "initial_forward_speed_mm_s": args.initial_forward_speed_mm_s,
        "initial_forward_speed_scope": (
            "one-shot episode initial condition only; no external forward-velocity "
            "controller or per-step translational forcing is applied"
        ),
        "total_passed_gates_this_run": total_passed,
        "total_collisions_this_run": total_collisions,
        "mean_passed_first_half": float(np.mean(first_half)) if first_half else 0.0,
        "mean_passed_second_half": float(np.mean(second_half)) if second_half else 0.0,
        "elapsed_seconds": elapsed,
        "resumed_from": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "resume_neural_step": resume_info.get("step") if resume_info else None,
        "full_cns_checkpoint": str(checkpoint_dir),
        "checkpoint_neural_step": state_checkpoint_info.get("step"),
        "checkpoint_scope": (
            "CNS membrane potentials, spikes, refractory counters, activity traces, "
            "neuromodulation state, synaptic weights, and eligibility traces. Topology, "
            "neurotransmitter annotations, numerical parameters, and modulator roles are "
            "reconstructed from the same snapshot/configuration. Body/environment state "
            "is intentionally reset at the episode boundary."
        ),
        "learned_weights_file": str(checkpoint_weight_file),
        "visualization": {
            "live_body_3d": bool(args.render),
            "record_dir": str(args.record_video) if args.record_video else None,
            "playback_speed": args.playback_speed,
            "fps": args.video_fps,
            "trajectory_trace": str(trajectory_path),
            "synapse_trace": str(synapse_trace_path) if args.synapse_trace else None,
            "synapse_top_n": args.synapse_top_n if args.synapse_trace else None,
        },
        "episode_results": episode_results,
        "sensory_interface": (
            "FlyBody raw eye cameras -> MaleCNS assignedOlHex retinotopic local samples -> "
            "per-body-ID R1-R6 current; no external motion/edge/obstacle feature extraction"
        ),
        "retinotopic_map": str(args.retinotopic_map),
        "photoreceptor_current_gain": args.photoreceptor_current_gain,
        "motor_interface": "MaleCNS DNg02 populations -> FlyBody wing amplitude",
        "body_physics": (
            "FlyGym 2.1 FlyBody with source FlyBody flight wing gains, damping, "
            "stiffness, 50-us timestep, restored per-wing MuJoCo ellipsoid-fluid "
            "geometries, and unit-corrected air density/viscosity"
        ),
        "teaching_signal": (
            "gate pass -> PAM08 candidate stimulation; collision -> PPL1 candidate stimulation"
        ),
        "important_limit": (
            "Optic-column/body-ID retinotopy comes from MaleCNS data, while the geometric "
            "projection of those columns onto FlyBody's eye camera and the local "
            "irradiance-to-current scale remain calibrated sensory-transduction seams. "
            "The CNS itself performs downstream visual processing."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    print(f"trajectory={trajectory_path}")
    print(f"checkpoint={checkpoint_dir}")
    if args.synapse_trace:
        print(f"synapse_trace={synapse_trace_path}")
    print(f"elapsed={elapsed:.3f}s")
    print("flyppy_closed_loop=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
