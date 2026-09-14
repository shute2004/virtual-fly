#!/usr/bin/env python3
"""Verify that DNg02 drive changes the free FlyBody trajectory.

The existing wing smoke only proves that actuator joints move. This test uses two
fresh free-body simulations with identical initial states and asks whether a
bilateral DNg02 drive changes the physical trajectory relative to nominal wing
beating. It is a body-interface test, not a learning experiment.
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
    parser.add_argument("--seconds", type=float, default=0.08)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-freeflight-smoke.json"),
    )
    return parser.parse_args()


def run(seconds: float, physics_steps: int, drive: WingDrive) -> tuple[np.ndarray, np.ndarray]:
    body = FlyBodyWingAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 5.0),
        enable_vision=False,
    )
    start = body.thorax_position_mm()
    control_dt = body.timestep * physics_steps
    control_steps = max(1, math.ceil(seconds / control_dt))
    for _ in range(control_steps):
        body.step(drive, physics_steps=physics_steps)
    return start, body.thorax_position_mm()


def main() -> int:
    args = parse_args()
    if args.seconds <= 0 or args.physics_steps < 1:
        raise SystemExit("seconds must be positive and physics-steps must be >= 1")

    baseline_start, baseline_end = run(
        args.seconds,
        args.physics_steps,
        WingDrive(left=0.0, right=0.0),
    )
    driven_start, driven_end = run(
        args.seconds,
        args.physics_steps,
        WingDrive(left=1.0, right=1.0),
    )

    baseline_delta = baseline_end - baseline_start
    driven_delta = driven_end - driven_start
    trajectory_difference = driven_delta - baseline_delta

    arrays = (baseline_start, baseline_end, driven_start, driven_end)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError("free-flight simulation produced non-finite body position")

    difference_norm = float(np.linalg.norm(trajectory_difference))
    if difference_norm <= 1e-5:
        raise RuntimeError(
            "DNg02 amplitude drive did not measurably change the free-body trajectory"
        )

    result = {
        "seconds": args.seconds,
        "physics_steps_per_control": args.physics_steps,
        "baseline_start_mm": baseline_start.tolist(),
        "baseline_end_mm": baseline_end.tolist(),
        "baseline_displacement_mm": baseline_delta.tolist(),
        "driven_start_mm": driven_start.tolist(),
        "driven_end_mm": driven_end.tolist(),
        "driven_displacement_mm": driven_delta.tolist(),
        "trajectory_difference_mm": trajectory_difference.tolist(),
        "trajectory_difference_norm_mm": difference_norm,
        "interpretation": (
            "Interface smoke only: changing bilateral DNg02 drive changes the "
            "free FlyBody physical trajectory. No behavioral-learning claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"trajectory_difference_norm_mm={difference_norm:.9f}")
    print(f"result={args.output}")
    print("flybody_freeflight_smoke=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
