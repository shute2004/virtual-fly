#!/usr/bin/env python3
"""Measure the free-flight envelope of the virtual wing-muscle mechanics.

This diagnostic deliberately bypasses the CNS and supplies only bilateral,
constant DLM/DVM activation to the physical virtual-muscle layer. It is therefore
not a controller and does not use Flyppy geometry or reward. Its purpose is to
answer a narrower calibration question before expensive neural training:

    can the current muscle->hinge torque model physically sustain a plausible
    free-flight operating point long enough to reach the first Flyppy gate?

A legacy FlyBody position-actuated wingbeat is also measured as a reference. If
that reference is viable while every virtual-muscle operating point is not, the
problem is localized to the virtual-muscle mechanics rather than the restored
FlyBody flight physics as a whole.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flybody_adapter import FlyBodyWingAdapter, WingDrive
from flybody_muscle_adapter import FlyBodyMuscleAdapter
from wing_muscle_periphery import PeripheralSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.08)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--initial-forward-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--required-forward-mm", type=float, default=8.0)
    parser.add_argument("--minimum-z-mm", type=float, default=0.65)
    parser.add_argument("--maximum-z-mm", type=float, default=9.35)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=(0.15, 0.25, 0.35, 0.5, 0.7, 1.0),
        help="constant bilateral DLM/DVM activation levels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-muscle-flight-envelope.json"),
    )
    return parser.parse_args()


def bilateral_power_state(level: float) -> PeripheralSnapshot:
    return PeripheralSnapshot(
        activation_by_body={},
        muscle_activation={},
        dlm_activation={"left": level, "right": level},
        dvm_activation={"left": level, "right": level},
        active_spikes=0,
        selected_motor_units=1,
    )


def run_virtual_muscle(
    *,
    seconds: float,
    physics_steps: int,
    level: float,
    initial_forward_speed_mm_s: float,
    aerodynamics: bool,
) -> dict[str, object]:
    body = FlyBodyMuscleAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(initial_forward_speed_mm_s, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=aerodynamics,
    )
    state = bilateral_power_state(level)
    start = body.thorax_position_mm()
    start_velocity = body.root_linear_velocity_mm_s()
    control_dt = body.timestep * physics_steps
    controls = max(1, math.ceil(seconds / control_dt))
    min_z = float(start[2])
    max_z = float(start[2])
    max_abs_torque = 0.0
    saturated_values = 0
    torque_values = 0

    for _ in range(controls):
        body.step_muscles(state, physics_steps=physics_steps)
        position = body.thorax_position_mm()
        min_z = min(min_z, float(position[2]))
        max_z = max(max_z, float(position[2]))
        for value in body.last_wing_torque.values():
            magnitude = abs(float(value))
            max_abs_torque = max(max_abs_torque, magnitude)
            torque_values += 1
            if magnitude >= body.max_abs_torque * (1.0 - 1e-6):
                saturated_values += 1

    end = body.thorax_position_mm()
    end_velocity = body.root_linear_velocity_mm_s()
    delta = end - start
    elapsed = controls * control_dt
    return {
        "mode": "virtual_muscle",
        "level": level,
        "aerodynamics": aerodynamics,
        "elapsed_seconds": elapsed,
        "start_mm": start.tolist(),
        "end_mm": end.tolist(),
        "displacement_mm": delta.tolist(),
        "mean_velocity_mm_s": (delta / elapsed).tolist(),
        "start_root_velocity_mm_s": start_velocity.tolist(),
        "end_root_velocity_mm_s": end_velocity.tolist(),
        "min_z_mm": min_z,
        "max_z_mm": max_z,
        "max_abs_torque": max_abs_torque,
        "torque_saturation_fraction": (
            saturated_values / torque_values if torque_values else 0.0
        ),
    }


def run_position_reference(
    *,
    seconds: float,
    physics_steps: int,
    initial_forward_speed_mm_s: float,
) -> dict[str, object]:
    body = FlyBodyWingAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(initial_forward_speed_mm_s, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=True,
    )
    start = body.thorax_position_mm()
    start_velocity = body.root_linear_velocity_mm_s()
    control_dt = body.timestep * physics_steps
    controls = max(1, math.ceil(seconds / control_dt))
    min_z = float(start[2])
    max_z = float(start[2])
    for _ in range(controls):
        # Zero DNg02 drive still gives the calibrated nominal FlyBody wingbeat;
        # this is only a physical reference, never used in the target learning path.
        body.step(WingDrive(0.0, 0.0), physics_steps=physics_steps)
        position = body.thorax_position_mm()
        min_z = min(min_z, float(position[2]))
        max_z = max(max_z, float(position[2]))
    end = body.thorax_position_mm()
    end_velocity = body.root_linear_velocity_mm_s()
    delta = end - start
    elapsed = controls * control_dt
    return {
        "mode": "legacy_position_reference",
        "elapsed_seconds": elapsed,
        "start_mm": start.tolist(),
        "end_mm": end.tolist(),
        "displacement_mm": delta.tolist(),
        "mean_velocity_mm_s": (delta / elapsed).tolist(),
        "start_root_velocity_mm_s": start_velocity.tolist(),
        "end_root_velocity_mm_s": end_velocity.tolist(),
        "min_z_mm": min_z,
        "max_z_mm": max_z,
    }


def viable(
    sample: dict[str, object],
    *,
    required_forward_mm: float,
    minimum_z_mm: float,
    maximum_z_mm: float,
) -> bool:
    return (
        float(sample["displacement_mm"][0]) >= required_forward_mm
        and float(sample["min_z_mm"]) >= minimum_z_mm
        and float(sample["max_z_mm"]) <= maximum_z_mm
        and float(sample["end_root_velocity_mm_s"][0]) > 0.0
    )


def main() -> int:
    args = parse_args()
    if args.seconds <= 0.0 or args.physics_steps < 1:
        raise SystemExit("seconds must be > 0 and physics-steps must be >= 1")
    if args.initial_forward_speed_mm_s <= 0.0 or args.required_forward_mm <= 0.0:
        raise SystemExit("forward speed/distance must be positive")
    if not args.minimum_z_mm < args.maximum_z_mm:
        raise SystemExit("minimum-z-mm must be below maximum-z-mm")
    if not args.levels or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in args.levels):
        raise SystemExit("levels must contain finite values in [0, 1]")

    samples: list[dict[str, object]] = []
    for level in args.levels:
        for aerodynamics in (True, False):
            sample = run_virtual_muscle(
                seconds=args.seconds,
                physics_steps=args.physics_steps,
                level=float(level),
                initial_forward_speed_mm_s=args.initial_forward_speed_mm_s,
                aerodynamics=aerodynamics,
            )
            sample["viable"] = viable(
                sample,
                required_forward_mm=args.required_forward_mm,
                minimum_z_mm=args.minimum_z_mm,
                maximum_z_mm=args.maximum_z_mm,
            )
            samples.append(sample)

    reference = run_position_reference(
        seconds=args.seconds,
        physics_steps=args.physics_steps,
        initial_forward_speed_mm_s=args.initial_forward_speed_mm_s,
    )
    reference["viable"] = viable(
        reference,
        required_forward_mm=args.required_forward_mm,
        minimum_z_mm=args.minimum_z_mm,
        maximum_z_mm=args.maximum_z_mm,
    )

    aero_samples = [sample for sample in samples if bool(sample["aerodynamics"])]
    best = max(
        aero_samples,
        key=lambda sample: (
            bool(sample["viable"]),
            float(sample["displacement_mm"][0]),
            -abs(float(sample["end_mm"][2]) - 5.0),
        ),
    )
    viable_levels = [float(s["level"]) for s in aero_samples if bool(s["viable"])]

    for sample in samples + [reference]:
        arrays = (
            np.asarray(sample["displacement_mm"], dtype=np.float64),
            np.asarray(sample["end_root_velocity_mm_s"], dtype=np.float64),
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise RuntimeError("non-finite free-flight diagnostic state")

    result = {
        "seconds_requested": args.seconds,
        "physics_steps_per_control": args.physics_steps,
        "initial_forward_speed_mm_s": args.initial_forward_speed_mm_s,
        "required_forward_mm": args.required_forward_mm,
        "minimum_z_mm": args.minimum_z_mm,
        "maximum_z_mm": args.maximum_z_mm,
        "virtual_muscle_samples": samples,
        "legacy_position_reference": reference,
        "viable_virtual_muscle_levels": viable_levels,
        "best_virtual_muscle_sample": best,
        "interpretation": (
            "Body-mechanics calibration only. Constant bilateral power-muscle activation "
            "contains no Flyppy observation, reward, action decoder, or CNS output."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "legacy_reference viable={} dx={:.3f} min_z={:.3f} max_z={:.3f} final_vx={:.3f}".format(
            reference["viable"],
            float(reference["displacement_mm"][0]),
            float(reference["min_z_mm"]),
            float(reference["max_z_mm"]),
            float(reference["end_root_velocity_mm_s"][0]),
        )
    )
    for sample in aero_samples:
        print(
            "muscle level={:.2f} viable={} dx={:.3f} min_z={:.3f} max_z={:.3f} "
            "final_vx={:.3f} max_torque={:.3f} saturation={:.4f}".format(
                float(sample["level"]),
                sample["viable"],
                float(sample["displacement_mm"][0]),
                float(sample["min_z_mm"]),
                float(sample["max_z_mm"]),
                float(sample["end_root_velocity_mm_s"][0]),
                float(sample["max_abs_torque"]),
                float(sample["torque_saturation_fraction"]),
            )
        )
    print(f"viable_virtual_muscle_levels={viable_levels}")
    print(f"result={args.output}")
    print("flybody_muscle_flight_envelope=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
