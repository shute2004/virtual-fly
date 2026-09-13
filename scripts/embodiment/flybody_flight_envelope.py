#!/usr/bin/env python3
"""Measure the FlyBody operating envelope required by the Flyppy course.

The source FlyBody vision-flight task starts the animal with forward velocity. We do
the same here, but do not continuously force translation. The diagnostic then asks
whether wing/body physics can keep the fly airborne while it traverses the first
Flyppy gate distance.

For each bilateral DNg02 operating point, a matched no-wing-fluid control is also
run from the same initial state. This separates "the initial velocity moved the
body" from aerodynamic effects introduced by the restored wing fluid geometry.

This is a body calibration diagnostic, not a neural-learning test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flybody_adapter import FlyBodyWingAdapter, WingDrive
from flybody_flight_physics import FLIGHT_AIR_DENSITY, FLIGHT_AIR_VISCOSITY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.12)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument(
        "--initial-forward-speed-mm-s",
        type=float,
        default=300.0,
        help="one-shot +x initial speed; 300 mm/s is the midpoint of FlyBody's 20-40 cm/s vision-flight range",
    )
    parser.add_argument(
        "--required-forward-mm",
        type=float,
        default=8.0,
        help="minimum +x displacement; default matches the first Flyppy gate",
    )
    parser.add_argument(
        "--minimum-z-mm",
        type=float,
        default=0.65,
        help="minimum thorax height allowed during the diagnostic",
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-flight-envelope.json"),
    )
    return parser.parse_args()


def run_sample(
    seconds: float,
    physics_steps: int,
    drive: WingDrive,
    *,
    initial_forward_speed_mm_s: float,
    aerodynamics: bool,
) -> dict[str, object]:
    body = FlyBodyWingAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(initial_forward_speed_mm_s, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=aerodynamics,
    )

    # Keep the matched control in the same unit system. The adapter already applies
    # these values, but assigning the compiled options here makes the experimental
    # invariant explicit and protects the diagnostic from future adapter changes.
    if not aerodynamics:
        body.sim.mj_model.opt.density = FLIGHT_AIR_DENSITY
        body.sim.mj_model.opt.viscosity = FLIGHT_AIR_VISCOSITY

    start = body.thorax_position_mm()
    start_velocity = body.root_linear_velocity_mm_s()
    control_dt = body.timestep * physics_steps
    controls = max(1, math.ceil(seconds / control_dt))
    min_z = float(start[2])
    max_z = float(start[2])
    for _ in range(controls):
        body.step(drive, physics_steps=physics_steps)
        position = body.thorax_position_mm()
        min_z = min(min_z, float(position[2]))
        max_z = max(max_z, float(position[2]))
    end = body.thorax_position_mm()
    end_velocity = body.root_linear_velocity_mm_s()
    delta = end - start
    velocity_delta = end_velocity - start_velocity
    elapsed = controls * control_dt
    return {
        "drive": {"left": drive.left, "right": drive.right},
        "aerodynamics": aerodynamics,
        "elapsed_seconds": elapsed,
        "start_mm": start.tolist(),
        "end_mm": end.tolist(),
        "displacement_mm": delta.tolist(),
        "mean_velocity_mm_s": (delta / elapsed).tolist(),
        "start_root_velocity_mm_s": start_velocity.tolist(),
        "end_root_velocity_mm_s": end_velocity.tolist(),
        "delta_root_velocity_mm_s": velocity_delta.tolist(),
        "min_z_mm": min_z,
        "max_z_mm": max_z,
    }


def sample_is_viable(
    sample: dict[str, object],
    *,
    required_forward_mm: float,
    minimum_z_mm: float,
) -> bool:
    return (
        float(sample["displacement_mm"][0]) >= required_forward_mm
        and float(sample["min_z_mm"]) >= minimum_z_mm
        and float(sample["end_root_velocity_mm_s"][0]) > 0.0
    )


def main() -> int:
    args = parse_args()
    if args.seconds <= 0 or args.physics_steps < 1:
        raise SystemExit("seconds must be positive and physics-steps must be >= 1")
    if not math.isfinite(args.initial_forward_speed_mm_s) or args.initial_forward_speed_mm_s <= 0:
        raise SystemExit("initial-forward-speed-mm-s must be finite and > 0")
    if args.required_forward_mm <= 0 or args.minimum_z_mm < 0:
        raise SystemExit("required-forward-mm must be > 0 and minimum-z-mm must be >= 0")
    if not args.levels or any(not math.isfinite(v) or v < 0 or v > 1 for v in args.levels):
        raise SystemExit("levels must contain finite values in [0, 1]")

    paired_samples: list[dict[str, object]] = []
    all_samples: list[dict[str, object]] = []
    for level in args.levels:
        drive = WingDrive(left=float(level), right=float(level))
        with_aero = run_sample(
            args.seconds,
            args.physics_steps,
            drive,
            initial_forward_speed_mm_s=args.initial_forward_speed_mm_s,
            aerodynamics=True,
        )
        no_wing_fluid = run_sample(
            args.seconds,
            args.physics_steps,
            drive,
            initial_forward_speed_mm_s=args.initial_forward_speed_mm_s,
            aerodynamics=False,
        )
        with_aero_viable = sample_is_viable(
            with_aero,
            required_forward_mm=args.required_forward_mm,
            minimum_z_mm=args.minimum_z_mm,
        )
        control_viable = sample_is_viable(
            no_wing_fluid,
            required_forward_mm=args.required_forward_mm,
            minimum_z_mm=args.minimum_z_mm,
        )
        all_samples.extend((with_aero, no_wing_fluid))
        paired_samples.append(
            {
                "drive": with_aero["drive"],
                "with_aerodynamics": with_aero,
                "without_wing_fluid": no_wing_fluid,
                "with_aerodynamics_viable": with_aero_viable,
                "without_wing_fluid_viable": control_viable,
                "aerodynamics_required_for_viability": (
                    with_aero_viable and not control_viable
                ),
                "aero_delta_displacement_mm": (
                    np.asarray(with_aero["displacement_mm"], dtype=np.float64)
                    - np.asarray(no_wing_fluid["displacement_mm"], dtype=np.float64)
                ).tolist(),
                "aero_delta_final_velocity_mm_s": (
                    np.asarray(with_aero["end_root_velocity_mm_s"], dtype=np.float64)
                    - np.asarray(no_wing_fluid["end_root_velocity_mm_s"], dtype=np.float64)
                ).tolist(),
            }
        )

    # Asymmetric samples are useful for exposing whether steering destroys altitude
    # or forward motion, but they are not candidates for the bilateral flight gate.
    for drive in (WingDrive(1.0, 0.0), WingDrive(0.0, 1.0)):
        all_samples.append(
            run_sample(
                args.seconds,
                args.physics_steps,
                drive,
                initial_forward_speed_mm_s=args.initial_forward_speed_mm_s,
                aerodynamics=True,
            )
        )

    for sample in all_samples:
        displacement = np.asarray(sample["displacement_mm"], dtype=np.float64)
        velocity = np.asarray(sample["end_root_velocity_mm_s"], dtype=np.float64)
        if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(velocity)):
            raise RuntimeError("flight envelope contains non-finite body motion")

    bilateral_aero = [pair["with_aerodynamics"] for pair in paired_samples]
    viable_pairs = [
        pair
        for pair in paired_samples
        if bool(pair["aerodynamics_required_for_viability"])
    ]
    viable = [pair["with_aerodynamics"] for pair in viable_pairs]
    best = max(
        bilateral_aero,
        key=lambda sample: (
            float(sample["min_z_mm"]),
            float(sample["displacement_mm"][0]),
        ),
    )
    flight_viable = bool(viable)

    result = {
        "seconds_requested": args.seconds,
        "physics_steps_per_control": args.physics_steps,
        "initial_forward_speed_mm_s": args.initial_forward_speed_mm_s,
        "required_forward_mm": args.required_forward_mm,
        "minimum_z_mm": args.minimum_z_mm,
        "paired_bilateral_samples": paired_samples,
        "asymmetric_aero_samples": all_samples[-2:],
        "best_bilateral_drive": best["drive"],
        "best_forward_displacement_mm": float(best["displacement_mm"][0]),
        "best_min_z_mm": float(best["min_z_mm"]),
        "best_final_forward_velocity_mm_s": float(best["end_root_velocity_mm_s"][0]),
        "flight_viable": flight_viable,
        "viable_bilateral_drives": [sample["drive"] for sample in viable],
        "interpretation": (
            "Body-interface calibration only. A passing operating point must traverse "
            "the first Flyppy gate distance, remain above the configured minimum "
            "height for the full diagnostic, still have positive forward velocity, "
            "and fail the same viability test when the two wing-fluid geoms are removed. "
            "This prevents the one-shot initial speed by itself from satisfying the "
            "flight gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"best_forward_displacement_mm={result['best_forward_displacement_mm']:.6f}")
    print(f"best_min_z_mm={result['best_min_z_mm']:.6f}")
    print(
        "best_final_forward_velocity_mm_s="
        f"{result['best_final_forward_velocity_mm_s']:.6f}"
    )
    print(f"best_bilateral_drive={best['drive']}")
    print(f"result={args.output}")
    if not flight_viable:
        raise RuntimeError(
            "no tested bilateral DNg02 operating point requires restored wing aerodynamics "
            "to reach the first Flyppy gate while remaining airborne; calibrate flight pose, "
            "wing kinematics, or body physics before starting neural learning"
        )
    print("flybody_flight_envelope=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
