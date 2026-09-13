#!/usr/bin/env python3
"""Measure whether the current FlyBody wing adapter can physically advance +x.

Flyppy gates are arranged along +x. A controller cannot learn to pass them if the
body/wing interface has no operating point that produces forward progress. This
script measures that prerequisite directly instead of assuming that a changed
trajectory implies useful flight.

This is a body calibration diagnostic, not a neural-learning test. Each sample is
run from a fresh identical free-body initial state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flybody_adapter import FlyBodyWingAdapter, WingDrive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.12)
    parser.add_argument("--physics-steps", type=int, default=10)
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
) -> dict[str, object]:
    body = FlyBodyWingAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 5.0),
        enable_vision=False,
    )
    start = body.thorax_position_mm()
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
    delta = end - start
    elapsed = controls * control_dt
    return {
        "drive": {"left": drive.left, "right": drive.right},
        "elapsed_seconds": elapsed,
        "start_mm": start.tolist(),
        "end_mm": end.tolist(),
        "displacement_mm": delta.tolist(),
        "mean_velocity_mm_s": (delta / elapsed).tolist(),
        "min_z_mm": min_z,
        "max_z_mm": max_z,
    }


def main() -> int:
    args = parse_args()
    if args.seconds <= 0 or args.physics_steps < 1:
        raise SystemExit("seconds must be positive and physics-steps must be >= 1")
    if not args.levels or any(not math.isfinite(v) or v < 0 or v > 1 for v in args.levels):
        raise SystemExit("levels must contain finite values in [0, 1]")

    samples: list[dict[str, object]] = []
    for level in args.levels:
        samples.append(
            run_sample(
                args.seconds,
                args.physics_steps,
                WingDrive(left=float(level), right=float(level)),
            )
        )

    # Add asymmetric samples to expose whether steering control destroys or
    # preserves forward motion. These are diagnostic only.
    samples.append(run_sample(args.seconds, args.physics_steps, WingDrive(1.0, 0.0)))
    samples.append(run_sample(args.seconds, args.physics_steps, WingDrive(0.0, 1.0)))

    for sample in samples:
        displacement = np.asarray(sample["displacement_mm"], dtype=np.float64)
        velocity = np.asarray(sample["mean_velocity_mm_s"], dtype=np.float64)
        if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(velocity)):
            raise RuntimeError("flight envelope contains non-finite body motion")

    bilateral = [
        sample
        for sample in samples
        if float(sample["drive"]["left"]) == float(sample["drive"]["right"])
    ]
    best = max(bilateral, key=lambda sample: float(sample["displacement_mm"][0]))
    best_dx = float(best["displacement_mm"][0])
    best_vx = float(best["mean_velocity_mm_s"][0])
    forward_capable = best_dx > 0.0

    result = {
        "seconds_requested": args.seconds,
        "physics_steps_per_control": args.physics_steps,
        "samples": samples,
        "best_bilateral_drive": best["drive"],
        "best_forward_displacement_mm": best_dx,
        "best_forward_velocity_mm_s": best_vx,
        "forward_capable": forward_capable,
        "interpretation": (
            "Body-interface calibration only. forward_capable means at least one "
            "tested bilateral DNg02 operating point moved the free FlyBody in the "
            "+x direction used by Flyppy. It does not establish stable long-duration "
            "flight or learning."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"best_forward_displacement_mm={best_dx:.9f}")
    print(f"best_forward_velocity_mm_s={best_vx:.6f}")
    print(f"best_bilateral_drive={best['drive']}")
    print(f"result={args.output}")
    if not forward_capable:
        raise RuntimeError(
            "current wing adapter has no tested +x forward-flight operating point; "
            "Flyppy gates cannot be reached without recalibrating body initial conditions "
            "or wing/body flight dynamics"
        )
    print("flybody_flight_envelope=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
