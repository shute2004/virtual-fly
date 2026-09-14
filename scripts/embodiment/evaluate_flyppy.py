#!/usr/bin/env python3
"""Evaluate Flyppy behavior with frozen plasticity and no reinforcement.

Evaluation deliberately uses the same current sensorimotor path as training:

FlyBody eye -> MaleCNS R1-R6 body currents -> whole CNS -> individual released
wing motor-neuron spikes -> WingMusclePeriphery -> FlyBodyMuscleAdapter.

No DNg02 population-average decoder is used here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from malecns_retina import MaleCNSRetina
from neural_bridge_client import NeuralBridgeClient
from wing_muscle_periphery import WingMusclePeriphery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
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
    )
    parser.add_argument("--photoreceptor-current-gain", type=float, default=2.0)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--learned-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-control-steps", type=int, default=1800)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--spawn-x-mm", type=float, default=0.0)
    parser.add_argument("--spawn-z-mm", type=float, default=5.0)
    parser.add_argument("--initial-forward-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--course-start-gate", type=int, default=0)
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v1/evaluation.json"),
    )
    return parser.parse_args()


def make_body(args: argparse.Namespace, course: FlyppyCourse) -> FlyBodyMuscleAdapter:
    world = FlyppyWorld(course)
    return FlyBodyMuscleAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(args.spawn_x_mm, 0.0, args.spawn_z_mm),
        initial_linear_velocity_mm_s=(args.initial_forward_speed_mm_s, 0.0, 0.0),
        enable_vision=True,
        enable_observer_camera=False,
    )


def evaluate_state(
    *,
    label: str,
    checkpoint: Path | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    os.environ["VF_COURSE_START_GATE"] = str(args.course_start_gate)
    course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count)
    body = make_body(args, course)
    periphery = WingMusclePeriphery(args.wing_motor_map)
    vision = MaleCNSRetina(
        args.retinotopic_map,
        current_gain=args.photoreceptor_current_gain,
    )
    control_dt_s = body.timestep * args.physics_steps
    episodes: list[dict[str, object]] = []

    with NeuralBridgeClient(
        snapshot=args.snapshot,
        groups=args.groups,
        backend=args.backend,
    ) as brain:
        brain.ping()
        if checkpoint is not None:
            brain.load_checkpoint(checkpoint)

        for episode in range(args.episodes):
            course.reset()
            body.reset()
            body.set_root_position_mm((args.spawn_x_mm, 0.0, args.spawn_z_mm))
            body.set_root_linear_velocity_mm_s(
                (args.initial_forward_speed_mm_s, 0.0, 0.0)
            )
            periphery.reset()
            vision.reset_adaptation()
            brain.reset_dynamics()
            if args.settle_steps:
                brain.step(stimulate={}, read=(), plasticity=False, steps=args.settle_steps)

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
                retinal = vision.encode(body.sim, body.fly)
                _, body_spikes = brain.step_with_body_readout(
                    stimulate_body=retinal.body_currents,
                    read=(),
                    read_body=periphery.body_ids,
                    plasticity=False,
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
                if event.passed_gate:
                    passed += 1
                if event.collision:
                    collision = True
                    collision_reason = event.collision_reason
                if event.finished:
                    finished = True
                if collision or finished:
                    break

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
                "final_vy_mm_s": float(final_velocity[1]),
                "final_vz_mm_s": float(final_velocity[2]),
            }
            episodes.append(result)
            print(
                "evaluation={} episode={} passed={} collision={} finished={} max_x={:.3f}".format(
                    label,
                    episode,
                    passed,
                    collision,
                    finished,
                    max_x,
                )
            )

    passed_values = [int(item["passed_gates"]) for item in episodes]
    max_x_values = [float(item["max_x_mm"]) for item in episodes]
    return {
        "label": label,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "episodes": episodes,
        "mean_passed_gates": float(statistics.fmean(passed_values)),
        "mean_max_x_mm": float(statistics.fmean(max_x_values)),
        "finished_fraction": sum(bool(item["finished"]) for item in episodes) / len(episodes),
        "collision_fraction": sum(bool(item["collision"]) for item in episodes) / len(episodes),
    }


def main() -> int:
    args = parse_args()
    if args.episodes < 1 or args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("episode/control/physics counts must be positive")
    if args.settle_steps < 0 or args.gate_count < 1 or args.course_start_gate < 0:
        raise SystemExit("settle-steps/course-start-gate/gate-count are invalid")
    numeric_positive = (
        args.initial_forward_speed_mm_s,
        args.photoreceptor_current_gain,
    )
    if any(not np.isfinite(value) or value <= 0 for value in numeric_positive):
        raise SystemExit("speed and photoreceptor gain must be finite and > 0")
    if not np.isfinite(args.spawn_x_mm) or not np.isfinite(args.spawn_z_mm):
        raise SystemExit("spawn coordinates must be finite")
    for path, label in (
        (args.retinotopic_map, "retinotopic vision map"),
        (args.wing_motor_map, "wing motor map"),
        (args.learned_checkpoint, "learned checkpoint"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")
    if args.baseline_checkpoint is not None and not args.baseline_checkpoint.exists():
        raise SystemExit(f"baseline checkpoint not found: {args.baseline_checkpoint}")

    started = time.perf_counter()
    baseline = evaluate_state(label="baseline", checkpoint=args.baseline_checkpoint, args=args)
    learned = evaluate_state(label="learned", checkpoint=args.learned_checkpoint, args=args)
    delta_passed = float(learned["mean_passed_gates"]) - float(baseline["mean_passed_gates"])
    delta_max_x = float(learned["mean_max_x_mm"]) - float(baseline["mean_max_x_mm"])
    output = {
        "schema_version": 3,
        "experiment": "flyppy_frozen_state_individual_wing_mn_evaluation",
        "plasticity_during_evaluation": False,
        "reinforcement_during_evaluation": False,
        "sensorimotor_path": "FlyBody eye -> MaleCNS R1-R6 -> individual released wing MN -> WingMusclePeriphery -> FlyBodyMuscleAdapter",
        "retinotopic_map": str(args.retinotopic_map),
        "wing_motor_map": str(args.wing_motor_map),
        "photoreceptor_current_gain": args.photoreceptor_current_gain,
        "settle_steps": args.settle_steps,
        "course_seed": args.seed,
        "course_start_gate": args.course_start_gate,
        "gate_count": args.gate_count,
        "spawn_x_mm": args.spawn_x_mm,
        "spawn_z_mm": args.spawn_z_mm,
        "initial_forward_speed_mm_s": args.initial_forward_speed_mm_s,
        "episodes_per_state": args.episodes,
        "baseline": baseline,
        "learned": learned,
        "delta_mean_passed_gates": delta_passed,
        "delta_mean_max_x_mm": delta_max_x,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"delta_mean_passed_gates={delta_passed:+.6f}")
    print(f"delta_mean_max_x_mm={delta_max_x:+.6f}")
    print(f"evaluation={args.output}")
    print("flyppy_evaluation=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
