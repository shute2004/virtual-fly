#!/usr/bin/env python3
"""Evaluate Flyppy behavior without plasticity or reinforcement.

This intentionally reuses the same embodied sensory/motor loop as training while
turning learning off. It can compare an untrained MaleCNS (or an explicit baseline
checkpoint) against a learned checkpoint on the exact same deterministic course.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from flybody_adapter import FlyBodyWingAdapter, WingDrive
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from neural_bridge_client import NeuralBridgeClient
from visual_motion_encoder import VerticalMotionEncoder


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
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--learned-checkpoint",
        type=Path,
        required=True,
        help="full CNS checkpoint produced by Flyppy training",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="optional baseline CNS checkpoint; omit to compare against fresh MaleCNS state",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-control-steps", type=int, default=1800)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--initial-forward-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v0/evaluation.json"),
    )
    return parser.parse_args()


def positive_stimuli(
    stimuli: dict[str, float], epsilon: float = 1e-6
) -> dict[str, float]:
    return {name: value for name, value in stimuli.items() if value > epsilon}


def evaluate_episode(
    *,
    snapshot: Path,
    groups: Path,
    backend: str,
    checkpoint: Path | None,
    seed: int,
    gate_count: int,
    max_control_steps: int,
    physics_steps: int,
    initial_forward_speed_mm_s: float,
) -> dict[str, object]:
    course = FlyppyCourse(seed=seed, gate_count=gate_count)
    world = FlyppyWorld(course)
    body = FlyBodyWingAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(initial_forward_speed_mm_s, 0.0, 0.0),
        enable_vision=True,
    )
    vision = VerticalMotionEncoder()
    vision.encode(body.ommatidia_readouts())

    passed = 0
    collision = False
    finished = False
    max_x = float("-inf")
    min_z = float("inf")
    max_z = float("-inf")
    final_velocity = body.root_linear_velocity_mm_s()
    step_count = 0

    with NeuralBridgeClient(
        snapshot=snapshot,
        groups=groups,
        backend=backend,
    ) as brain:
        brain.ping()
        if checkpoint is not None:
            brain.load_checkpoint(checkpoint)

        for control_step in range(max_control_steps):
            sensory = positive_stimuli(
                vision.encode(body.ommatidia_readouts()).as_stimuli()
            )
            readout = brain.step(
                stimulate=sensory,
                read=MOTOR_GROUPS,
                plasticity=False,
            )
            left = float(readout["flight_thrust_left"]["spike_fraction"])
            right = float(readout["flight_thrust_right"]["spike_fraction"])

            body.step(
                WingDrive(left=left, right=right),
                physics_steps=physics_steps,
            )
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
            if event.finished:
                finished = True
            if collision or finished:
                break

    return {
        "control_steps": step_count,
        "passed_gates": passed,
        "collision": collision,
        "finished": finished,
        "max_x_mm": max_x,
        "min_z_mm": min_z,
        "max_z_mm": max_z,
        "final_vx_mm_s": float(final_velocity[0]),
        "final_vy_mm_s": float(final_velocity[1]),
        "final_vz_mm_s": float(final_velocity[2]),
    }


def evaluate_state(
    *,
    label: str,
    checkpoint: Path | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    episodes = []
    for episode in range(args.episodes):
        result = evaluate_episode(
            snapshot=args.snapshot,
            groups=args.groups,
            backend=args.backend,
            checkpoint=checkpoint,
            seed=args.seed,
            gate_count=args.gate_count,
            max_control_steps=args.max_control_steps,
            physics_steps=args.physics_steps,
            initial_forward_speed_mm_s=args.initial_forward_speed_mm_s,
        )
        result["episode"] = episode
        episodes.append(result)
        print(
            "evaluation={} episode={} passed={} collision={} finished={} max_x={:.3f}".format(
                label,
                episode,
                result["passed_gates"],
                result["collision"],
                result["finished"],
                result["max_x_mm"],
            )
        )

    passed = [int(item["passed_gates"]) for item in episodes]
    max_x = [float(item["max_x_mm"]) for item in episodes]
    return {
        "label": label,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "episodes": episodes,
        "mean_passed_gates": float(statistics.fmean(passed)),
        "mean_max_x_mm": float(statistics.fmean(max_x)),
        "finished_fraction": float(
            sum(bool(item["finished"]) for item in episodes) / len(episodes)
        ),
        "collision_fraction": float(
            sum(bool(item["collision"]) for item in episodes) / len(episodes)
        ),
    }


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("episodes must be >= 1")
    if args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("control/physics steps must be >= 1")
    if args.gate_count < 1:
        raise SystemExit("gate-count must be >= 1")
    if (
        not np.isfinite(args.initial_forward_speed_mm_s)
        or args.initial_forward_speed_mm_s <= 0
    ):
        raise SystemExit("initial-forward-speed-mm-s must be finite and > 0")
    if not args.learned_checkpoint.exists():
        raise SystemExit(f"learned checkpoint not found: {args.learned_checkpoint}")
    if args.baseline_checkpoint is not None and not args.baseline_checkpoint.exists():
        raise SystemExit(f"baseline checkpoint not found: {args.baseline_checkpoint}")

    started = time.perf_counter()
    baseline = evaluate_state(
        label="baseline",
        checkpoint=args.baseline_checkpoint,
        args=args,
    )
    learned = evaluate_state(
        label="learned",
        checkpoint=args.learned_checkpoint,
        args=args,
    )

    delta_passed = float(learned["mean_passed_gates"]) - float(
        baseline["mean_passed_gates"]
    )
    delta_max_x = float(learned["mean_max_x_mm"]) - float(
        baseline["mean_max_x_mm"]
    )
    output = {
        "schema_version": 1,
        "experiment": "flyppy_frozen_state_evaluation_v0",
        "plasticity_during_evaluation": False,
        "reinforcement_during_evaluation": False,
        "course_seed": args.seed,
        "gate_count": args.gate_count,
        "episodes_per_state": args.episodes,
        "baseline": baseline,
        "learned": learned,
        "delta_mean_passed_gates": delta_passed,
        "delta_mean_max_x_mm": delta_max_x,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "Positive deltas are evidence of a behavioral change associated with the learned "
            "CNS state. They are not by themselves sufficient to establish robust learning; "
            "control experiments and additional course seeds remain necessary."
        ),
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
