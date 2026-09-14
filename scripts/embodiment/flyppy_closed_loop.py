#!/usr/bin/env python3
"""Embodied MaleCNS <-> peripheral muscle <-> FlyBody <-> Flyppy learning loop.

Target information flow:

    physical Flyppy world
      -> FlyBody compound-eye readout
      -> current into actual MaleCNS R1-R6 body IDs
      -> persistent MaleCNS runtime with local plasticity
      -> exact spikes from individual released wing motor-neuron body IDs
      -> independent motor-unit / neuromuscular states
      -> anatomical wing-muscle identities
      -> virtual-muscle physical torque at the FlyBody wing hinge
      -> physical world

There is no population action decoder, policy matrix, target action, Q-value,
backpropagation or signed scalar reward in this path. Teaching events inject
current into released dopaminergic neurons. To make initial learning practical,
small one-shot forward-progress milestones before the first gate also activate the
reward-associated DAN group; gate passage remains the stronger success event.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from malecns_retina import MaleCNSRetina
from neural_bridge_client import NeuralBridgeClient
from synapse_monitor import SynapseMonitor, SynapseMonitorConfig
from training_visualizer import TrainingVisualizer
from wing_muscle_periphery import WingMusclePeriphery


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
    parser.add_argument(
        "--wing-motor-map",
        type=Path,
        default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"),
        help="individual released wing-MN -> muscle inventory",
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-control-steps", type=int, default=1800)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument(
        "--initial-forward-speed-mm-s",
        type=float,
        default=300.0,
        help="one-shot +x velocity applied only at episode reset",
    )
    parser.add_argument(
        "--photoreceptor-current-gain",
        type=float,
        default=2.0,
        help="local irradiance/contrast-to-current scale for R1-R6",
    )
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-current", type=float, default=2.0)
    parser.add_argument("--aversive-current", type=float, default=2.0)
    parser.add_argument("--reinforcement-steps", type=int, default=4)
    parser.add_argument(
        "--progress-reward-current",
        type=float,
        default=0.5,
        help="weaker PAM01 current for pre-gate forward-progress milestones",
    )
    parser.add_argument(
        "--progress-reward-steps",
        type=int,
        default=2,
        help="neural steps for one progress-milestone PAM01 pulse",
    )
    parser.add_argument(
        "--progress-reward-start-mm",
        type=float,
        default=6.5,
        help="first pre-gate x milestone that produces a weak success event",
    )
    parser.add_argument(
        "--progress-reward-spacing-mm",
        type=float,
        default=0.25,
        help="spacing between one-shot pre-gate progress milestones",
    )
    parser.add_argument(
        "--no-progress-reward",
        dest="progress_reward",
        action="store_false",
        help="disable pre-gate progress shaping and use gate/collision events only",
    )
    parser.set_defaults(progress_reward=True)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v1"),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="restore learned CNS state before running new episodes",
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
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", type=Path, default=None, metavar="DIR")
    parser.add_argument("--playback-speed", type=float, default=0.2)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--synapse-trace", action="store_true")
    parser.add_argument("--synapse-top-n", type=int, default=64)
    parser.add_argument("--synapse-trace-every", type=int, default=1, metavar="EPISODES")
    parser.add_argument("--synapse-min-delta", type=float, default=1e-7)
    return parser.parse_args()


def deliver_reinforcement(
    brain: NeuralBridgeClient,
    group: str,
    current: float,
    steps: int,
) -> None:
    """Inject current into an explicit released DAN group, never a reward scalar."""

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
    if args.progress_reward:
        if (
            not np.isfinite(args.progress_reward_current)
            or args.progress_reward_current <= 0.0
            or args.progress_reward_steps < 1
            or not np.isfinite(args.progress_reward_start_mm)
            or args.progress_reward_start_mm <= 0.0
            or not np.isfinite(args.progress_reward_spacing_mm)
            or args.progress_reward_spacing_mm <= 0.0
        ):
            raise SystemExit("progress-reward parameters must be finite and positive")
    if not args.retinotopic_map.exists():
        raise SystemExit(f"retinotopic vision map not found: {args.retinotopic_map}")
    if not args.wing_motor_map.exists():
        raise SystemExit(f"wing motor map not found: {args.wing_motor_map}")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.exists():
        raise SystemExit(f"resume checkpoint not found: {args.resume_checkpoint}")

    visualization_enabled = args.render or args.record_video is not None
    course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count)
    first_gate_x_mm = float(course.gates[0].x_mm)
    if args.progress_reward and args.progress_reward_start_mm >= first_gate_x_mm:
        raise SystemExit(
            "progress-reward-start-mm must be before the first gate plane "
            f"({first_gate_x_mm:.3f} mm)"
        )
    world = FlyppyWorld(course)
    body = FlyBodyMuscleAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(args.initial_forward_speed_mm_s, 0.0, 0.0),
        enable_vision=True,
        enable_observer_camera=visualization_enabled,
    )
    periphery = WingMusclePeriphery(args.wing_motor_map)
    vision = MaleCNSRetina(
        args.retinotopic_map,
        current_gain=args.photoreceptor_current_gain,
    )
    control_dt_s = body.timestep * args.physics_steps

    print(
        "motor_boundary=individual-wing-MN selected={} excluded={} control_dt_s={:.6g}".format(
            periphery.selected_count,
            periphery.excluded_count,
            control_dt_s,
        )
    )
    if args.progress_reward:
        print(
            "learning_shaping=pre-gate-progress start_x={:.3f} spacing={:.3f} "
            "current={:.3f} steps={}".format(
                args.progress_reward_start_mm,
                args.progress_reward_spacing_mm,
                args.progress_reward_current,
                args.progress_reward_steps,
            )
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
    total_progress_rewards = 0
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
                    periphery.reset()
                    vision.reset_adaptation()
                    # Learning lives in mutable synaptic weights. Crash transients,
                    # membrane state, traces, modulation and eligibility are trial-
                    # local and must not leak through the artificial episode reset.
                    brain.reset_dynamics()
                    if visualizer.enabled:
                        visualizer.begin_episode(episode)

                    episode_passed = 0
                    episode_progress_rewards = 0
                    collision = False
                    finished = False
                    max_x = float("-inf")
                    min_z = float("inf")
                    max_z = float("-inf")
                    final_velocity = body.root_linear_velocity_mm_s()
                    step_count = 0
                    last_peripheral = None
                    next_progress_x = float(args.progress_reward_start_mm)

                    for control_step in range(args.max_control_steps):
                        retinal = vision.encode(body.sim, body.fly)
                        _, motor_spikes = brain.step_with_body_readout(
                            stimulate_body=retinal.body_currents,
                            read=(),
                            read_body=periphery.body_ids,
                            plasticity=True,
                        )
                        peripheral_state = periphery.step(
                            motor_spikes,
                            dt_s=control_dt_s,
                        )
                        last_peripheral = peripheral_state
                        body.step_muscles(
                            peripheral_state,
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
                        progress_reward_stimulated = False
                        progress_milestone_x_mm: float | None = None
                        aversive_stimulated = False

                        # Weak shaping before the first gate. Each threshold can
                        # fire only once per episode, and the signal is still a
                        # physical current pulse to the released reward DANs.
                        if (
                            args.progress_reward
                            and not event.collision
                            and course.next_gate_index == 0
                            and next_progress_x < first_gate_x_mm
                            and x_mm >= next_progress_x
                        ):
                            progress_milestone_x_mm = next_progress_x
                            deliver_reinforcement(
                                brain,
                                "reward_dan",
                                args.progress_reward_current,
                                args.progress_reward_steps,
                            )
                            progress_reward_stimulated = True
                            episode_progress_rewards += 1
                            total_progress_rewards += 1
                            while next_progress_x <= x_mm:
                                next_progress_x += args.progress_reward_spacing_mm

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
                            or progress_reward_stimulated
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
                                        "motor_periphery": peripheral_state.compact_diagnostics(),
                                        "wing_torque": dict(body.last_wing_torque),
                                        "retinal_input": {
                                            "active_columns": retinal.active_columns,
                                            "active_photoreceptors": retinal.active_photoreceptors,
                                            "mean_current": retinal.mean_current,
                                            "max_current": retinal.max_current,
                                        },
                                        "progress_reward_stimulated": progress_reward_stimulated,
                                        "progress_milestone_x_mm": progress_milestone_x_mm,
                                        "reward_stimulated": reward_stimulated,
                                        "aversive_stimulated": aversive_stimulated,
                                        "passed_gate": event.passed_gate,
                                        "collision": event.collision,
                                        "collision_reason": event.collision_reason,
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
                        "progress_rewards": episode_progress_rewards,
                        "collision": collision,
                        "finished": finished,
                        "max_x_mm": max_x,
                        "min_z_mm": min_z,
                        "max_z_mm": max_z,
                        "final_vx_mm_s": float(final_velocity[0]),
                        "final_vy_mm_s": float(final_velocity[1]),
                        "final_vz_mm_s": float(final_velocity[2]),
                        "final_motor_periphery": (
                            last_peripheral.compact_diagnostics()
                            if last_peripheral is not None
                            else None
                        ),
                        "video": video_path,
                        "synapse_snapshot": synapse_snapshot is not None,
                        "checkpoint_saved": checkpoint_saved_this_episode,
                    }
                    episode_results.append(result)
                    print(
                        "episode={} steps={} passed={} progress_rewards={} collision={} "
                        "finished={} max_x={:.3f} final_vx={:.3f}".format(
                            episode,
                            step_count,
                            episode_passed,
                            episode_progress_rewards,
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
        "schema_version": 6,
        "experiment": "flyppy_closed_loop_v1_individual_wing_mn_progress_shaping",
        "backend": ready.get("backend"),
        "neurons": ready.get("neurons"),
        "edges": ready.get("edges"),
        "episodes_this_run": args.episodes,
        "episode_start": start_episode,
        "episode_end": start_episode + args.episodes - 1,
        "gate_count": args.gate_count,
        "physics_steps_per_control": args.physics_steps,
        "control_dt_seconds": control_dt_s,
        "initial_forward_speed_mm_s": args.initial_forward_speed_mm_s,
        "initial_forward_speed_scope": (
            "one-shot episode initial condition only; no external forward-velocity "
            "controller or per-step translational forcing is applied"
        ),
        "total_passed_gates_this_run": total_passed,
        "total_progress_rewards_this_run": total_progress_rewards,
        "total_collisions_this_run": total_collisions,
        "mean_passed_first_half": float(np.mean(first_half)) if first_half else 0.0,
        "mean_passed_second_half": float(np.mean(second_half)) if second_half else 0.0,
        "elapsed_seconds": elapsed,
        "resumed_from": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "resume_neural_step": resume_info.get("step") if resume_info else None,
        "full_cns_checkpoint": str(checkpoint_dir),
        "checkpoint_neural_step": state_checkpoint_info.get("step"),
        "checkpoint_scope": (
            "Mutable synaptic weights persist across episodes. Membrane potentials, "
            "activity events, refractory counters, activity traces, local dopamine "
            "modulation and eligibility traces reset at each artificial episode boundary."
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
            "FlyBody compound-eye ommatidia -> locally adapted MaleCNS retinotopic samples -> "
            "per-body-ID R1-R6 current; no external motion/edge/obstacle feature extraction"
        ),
        "retinotopic_map": str(args.retinotopic_map),
        "photoreceptor_current_gain": args.photoreceptor_current_gain,
        "motor_interface": {
            "wing_motor_map": str(args.wing_motor_map),
            "selected_individual_motor_units": periphery.selected_count,
            "excluded_uncertain_motor_units": periphery.excluded_count,
            "path": (
                "exact individual MaleCNS wing-MN spikes -> independent motor-unit state -> "
                "anatomical muscle activation -> calibrated virtual-muscle joint torque -> FlyBody"
            ),
            "population_action_decoder": False,
            "dng02_population_readout_used_for_action": False,
        },
        "progress_shaping": {
            "enabled": bool(args.progress_reward),
            "start_x_mm": args.progress_reward_start_mm if args.progress_reward else None,
            "spacing_mm": args.progress_reward_spacing_mm if args.progress_reward else None,
            "current": args.progress_reward_current if args.progress_reward else None,
            "steps": args.progress_reward_steps if args.progress_reward else None,
            "scope": "one-shot x-threshold success events before the first gate only",
        },
        "body_physics": (
            "FlyGym 2.1 FlyBody with source FlyBody flight wing damping, stiffness, "
            "50-us timestep, restored per-wing MuJoCo ellipsoid-fluid geometries, "
            "unit-corrected air density/viscosity, and qfrc_applied virtual-muscle torque"
        ),
        "teaching_signal": (
            "pre-gate x milestones -> weak current into released PAM01 DANs; gate pass -> "
            "stronger PAM01 current; collision -> current into released PPL101 DANs. "
            "No signed scalar reward/punishment enters neural dynamics or plasticity."
        ),
        "important_limit": (
            "Optic-column/body-ID retinotopy comes from MaleCNS data. FlyBody lacks "
            "anatomical wing muscles, so muscle-to-hinge mechanics are a calibrated "
            "virtual-muscle approximation. The first torque model activates only "
            "qualitatively constrained b1/b2/b3/i1 steering effects; other identified "
            "muscles retain independent activation state but do not receive guessed "
            "moment arms. The geometric eye projection and irradiance-to-current scale "
            "also remain calibrated sensory-transduction seams."
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
