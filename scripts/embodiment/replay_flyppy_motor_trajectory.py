#!/usr/bin/env python3
"""Replay recorded Flyppy peripheral motor states without the CNS.

This diagnostic isolates the physical consequence of the motor pattern already
recorded in ``trajectory.jsonl``. It never reads visual input, reward, synaptic
weights, or gate geometry to choose a motor command. The same recorded state is
replayed under three mechanical ablations:

1. ``recorded``: power and steering exactly as stored in the trajectory;
2. ``power_only``: same DLM/DVM history with direct steering activation removed;
3. ``symmetric_power_only``: steering removed and left/right DLM/DVM activation
   replaced by their instantaneous bilateral mean.

Because the target Flyppy run was recorded with trajectory stride 1, each record
corresponds to the peripheral state that drove one physical control step. This
allows us to distinguish a body-mechanics problem from an unstable CNS motor
pattern without advancing or modifying the CNS.
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flyppy-motor-replay.json"),
    )
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
    expected = list(range(len(records)))
    actual = [int(item["control_step"]) for item in records]
    if actual != expected:
        raise RuntimeError(
            "motor replay requires trajectory stride 1 and contiguous control steps; "
            f"first mismatch around {actual[:12]}"
        )
    return records


def peripheral_from_record(record: dict[str, object], mode: str) -> PeripheralSnapshot:
    motor = dict(record.get("motor_periphery", {}) or {})
    dlm_left = float(motor.get("dlm_left", 0.0))
    dlm_right = float(motor.get("dlm_right", 0.0))
    dvm_left = float(motor.get("dvm_left", 0.0))
    dvm_right = float(motor.get("dvm_right", 0.0))

    steering_raw = dict(motor.get("steering", {}) or {})
    steering: dict[str, float] = {}
    if mode == "recorded":
        for key, value in steering_raw.items():
            muscle = str(key).split(":", 1)[-1]
            if muscle in ACTIVE_STEERING:
                steering[str(key)] = float(value)
    elif mode == "power_only":
        pass
    elif mode == "symmetric_power_only":
        dlm_mean = 0.5 * (dlm_left + dlm_right)
        dvm_mean = 0.5 * (dvm_left + dvm_right)
        dlm_left = dlm_right = dlm_mean
        dvm_left = dvm_right = dvm_mean
    else:
        raise ValueError(f"unknown replay mode {mode!r}")

    return PeripheralSnapshot(
        activation_by_body={},
        muscle_activation=steering,
        dlm_activation={"left": dlm_left, "right": dlm_right},
        dvm_activation={"left": dvm_left, "right": dvm_right},
        active_spikes=int(motor.get("spikes", 0)),
        # step_muscles only requires a nonzero count; individual state has already
        # been resolved and recorded by the original run.
        selected_motor_units=1,
    )


def summarize_recorded_pattern(records: list[dict[str, object]]) -> dict[str, object]:
    dlm_l: list[float] = []
    dlm_r: list[float] = []
    dvm_l: list[float] = []
    dvm_r: list[float] = []
    steering_scale_l: list[float] = []
    steering_scale_r: list[float] = []
    spike_counts: list[int] = []

    effects = {"b1": 0.18, "b2": 0.24, "b3": -0.18, "i1": -0.12}
    for record in records:
        motor = dict(record.get("motor_periphery", {}) or {})
        dlm_l.append(float(motor.get("dlm_left", 0.0)))
        dlm_r.append(float(motor.get("dlm_right", 0.0)))
        dvm_l.append(float(motor.get("dvm_left", 0.0)))
        dvm_r.append(float(motor.get("dvm_right", 0.0)))
        spike_counts.append(int(motor.get("spikes", 0)))
        steering = dict(motor.get("steering", {}) or {})
        for side, output in (("left", steering_scale_l), ("right", steering_scale_r)):
            scale = 0.0
            for muscle, effect in effects.items():
                scale += effect * float(steering.get(f"{side}:{muscle}", 0.0))
            output.append(scale)

    def stats(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(array)),
            "mean": float(np.mean(array)),
            "max": float(np.max(array)),
        }

    left_power = 0.5 * (np.asarray(dlm_l) + np.asarray(dvm_l))
    right_power = 0.5 * (np.asarray(dlm_r) + np.asarray(dvm_r))
    return {
        "dlm_left": stats(dlm_l),
        "dlm_right": stats(dlm_r),
        "dvm_left": stats(dvm_l),
        "dvm_right": stats(dvm_r),
        "power_left_mean_of_dlm_dvm": stats(left_power.tolist()),
        "power_right_mean_of_dlm_dvm": stats(right_power.tolist()),
        "max_abs_power_side_difference": float(np.max(np.abs(left_power - right_power))),
        "steering_scale_left": stats(steering_scale_l),
        "steering_scale_right": stats(steering_scale_r),
        "max_abs_steering_side_difference": float(
            np.max(np.abs(np.asarray(steering_scale_l) - np.asarray(steering_scale_r)))
        ),
        "motor_spikes_total": int(sum(spike_counts)),
        "motor_spikes_max_per_step": int(max(spike_counts)),
    }


def replay(
    records: list[dict[str, object]],
    *,
    mode: str,
    seed: int,
    gate_count: int,
    physics_steps: int,
    initial_forward_speed_mm_s: float,
) -> dict[str, object]:
    course = FlyppyCourse(seed=seed, gate_count=gate_count)
    world = FlyppyWorld(course)
    body = FlyBodyMuscleAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(initial_forward_speed_mm_s, 0.0, 0.0),
        enable_vision=False,
    )

    min_z = float("inf")
    max_z = float("-inf")
    max_x = float("-inf")
    collision_reason: str | None = None
    passed = 0
    steps = 0
    for record in records:
        state = peripheral_from_record(record, mode)
        body.step_muscles(state, physics_steps=physics_steps)
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

    position = body.thorax_position_mm()
    velocity = body.root_linear_velocity_mm_s()
    return {
        "mode": mode,
        "steps": steps,
        "passed_gates": passed,
        "collision": collision_reason is not None,
        "collision_reason": collision_reason,
        "max_x_mm": max_x,
        "min_z_mm": min_z,
        "max_z_mm": max_z,
        "final_position_mm": np.asarray(position, dtype=np.float64).tolist(),
        "final_velocity_mm_s": np.asarray(velocity, dtype=np.float64).tolist(),
    }


def main() -> int:
    args = parse_args()
    if args.physics_steps < 1:
        raise SystemExit("physics-steps must be >= 1")
    if args.initial_forward_speed_mm_s <= 0.0:
        raise SystemExit("initial-forward-speed-mm-s must be > 0")

    records = load_episode(args.trajectory, args.episode)
    pattern = summarize_recorded_pattern(records)
    replays = [
        replay(
            records,
            mode=mode,
            seed=args.seed,
            gate_count=args.gate_count,
            physics_steps=args.physics_steps,
            initial_forward_speed_mm_s=args.initial_forward_speed_mm_s,
        )
        for mode in ("recorded", "power_only", "symmetric_power_only")
    ]

    result = {
        "trajectory": str(args.trajectory),
        "episode": args.episode,
        "records": len(records),
        "recorded_motor_pattern": pattern,
        "replays": replays,
        "interpretation": (
            "No CNS state is advanced. recorded reproduces the stored peripheral command; "
            "power_only removes direct steering; symmetric_power_only additionally removes "
            "instantaneous left/right power asymmetry. Differences therefore localize which "
            "part of the recorded motor pattern destabilizes the physical flight model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "power dlm_left={:.3f}..{:.3f} dlm_right={:.3f}..{:.3f} "
        "dvm_left={:.3f}..{:.3f} dvm_right={:.3f}..{:.3f}".format(
            pattern["dlm_left"]["min"], pattern["dlm_left"]["max"],
            pattern["dlm_right"]["min"], pattern["dlm_right"]["max"],
            pattern["dvm_left"]["min"], pattern["dvm_left"]["max"],
            pattern["dvm_right"]["min"], pattern["dvm_right"]["max"],
        )
    )
    print(
        "motor_spikes total={} max_per_step={} max_power_side_diff={:.6f} "
        "max_steering_side_diff={:.6f}".format(
            pattern["motor_spikes_total"],
            pattern["motor_spikes_max_per_step"],
            pattern["max_abs_power_side_difference"],
            pattern["max_abs_steering_side_difference"],
        )
    )
    for sample in replays:
        print(
            "replay mode={} steps={} passed={} collision={} reason={} max_x={:.3f} "
            "min_z={:.3f} max_z={:.3f} final_vx={:.3f}".format(
                sample["mode"], sample["steps"], sample["passed_gates"],
                sample["collision"], sample["collision_reason"], sample["max_x_mm"],
                sample["min_z_mm"], sample["max_z_mm"], sample["final_velocity_mm_s"][0],
            )
        )
    print(f"result={args.output}")
    print("flyppy_motor_replay=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
