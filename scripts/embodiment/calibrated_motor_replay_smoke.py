#!/usr/bin/env python3
"""Fail closed unless the current muscle calibration stabilizes a recorded motor trace.

This is a regression preflight for an already-recorded trajectory. It does not
advance the CNS and does not alter the motor states. The exact peripheral DLM,
DVM, and direct-steering activations from one stride-1 episode are replayed
through the *current* FlyBodyMuscleAdapter defaults. A pass therefore means that
changes to the calibrated muscle->hinge seam removed the previously reproduced
physical collision without hiding or modifying the CNS motor history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from wing_muscle_periphery import PeripheralSnapshot


ACTIVE_STEERING = frozenset({"b1", "b2", "b3", "i1"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v1/trajectory.jsonl"),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--initial-forward-speed-mm-s", type=float, default=300.0)
    return parser.parse_args()


def load_episode(path: Path, episode: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if int(record.get("episode", -1)) == episode:
                records.append(record)
    records.sort(key=lambda item: int(item["control_step"]))
    if not records:
        raise RuntimeError(f"trajectory contains no records for episode {episode}")
    actual = [int(item["control_step"]) for item in records]
    if actual != list(range(len(records))):
        raise RuntimeError(
            "calibrated replay requires the prior episode to use trajectory stride 1"
        )
    return records


def state_from_record(record: dict[str, object]) -> PeripheralSnapshot:
    motor = dict(record.get("motor_periphery", {}) or {})
    steering_raw = dict(motor.get("steering", {}) or {})
    steering = {
        str(key): float(value)
        for key, value in steering_raw.items()
        if str(key).split(":", 1)[-1] in ACTIVE_STEERING
    }
    return PeripheralSnapshot(
        activation_by_body={},
        muscle_activation=steering,
        dlm_activation={
            "left": float(motor.get("dlm_left", 0.0)),
            "right": float(motor.get("dlm_right", 0.0)),
        },
        dvm_activation={
            "left": float(motor.get("dvm_left", 0.0)),
            "right": float(motor.get("dvm_right", 0.0)),
        },
        active_spikes=int(motor.get("spikes", 0)),
        selected_motor_units=1,
    )


def main() -> int:
    args = parse_args()
    records = load_episode(args.trajectory, args.episode)
    course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count)
    world = FlyppyWorld(course)
    body = FlyBodyMuscleAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(args.initial_forward_speed_mm_s, 0.0, 0.0),
        enable_vision=False,
    )

    min_z = float("inf")
    max_z = float("-inf")
    max_x = float("-inf")
    collision_reason: str | None = None
    passed = 0
    steps = 0
    for record in records:
        body.step_muscles(state_from_record(record), physics_steps=args.physics_steps)
        position = body.thorax_position_mm()
        x_mm = float(position[0])
        z_mm = float(position[2])
        min_z = min(min_z, z_mm)
        max_z = max(max_z, z_mm)
        max_x = max(max_x, x_mm)
        steps += 1
        event = course.update(x_mm, z_mm)
        if event.passed_gate:
            passed += 1
        if event.collision:
            collision_reason = event.collision_reason
            break
        if event.finished:
            break

    final_position = np.asarray(body.thorax_position_mm(), dtype=np.float64)
    final_velocity = np.asarray(body.root_linear_velocity_mm_s(), dtype=np.float64)
    full_horizon = steps == len(records)
    stable = collision_reason is None and full_horizon and float(final_velocity[0]) > 0.0

    print(
        "calibrated_adapter power_gain={:.3f} steering_gain={:.3f} "
        "swap_dlm_dvm_phase={}".format(
            body.power_gain,
            body.steering_gain,
            body.swap_dlm_dvm_phase,
        )
    )
    print(
        "replay records={} steps={} passed={} collision={} reason={} "
        "max_x={:.3f} min_z={:.3f} max_z={:.3f} final_x={:.3f} final_vx={:.3f}".format(
            len(records),
            steps,
            passed,
            collision_reason is not None,
            collision_reason,
            max_x,
            min_z,
            max_z,
            float(final_position[0]),
            float(final_velocity[0]),
        )
    )
    if not stable:
        raise RuntimeError(
            "current calibrated muscle adapter did not stabilize the previously recorded "
            "CNS motor trajectory; abort neural training rather than confounding body "
            "calibration with learning"
        )
    print("calibrated_motor_replay=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
