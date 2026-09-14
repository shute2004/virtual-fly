#!/usr/bin/env python3
"""Verify that FlyBody compound eyes see the Flyppy obstacle world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybody_runtime import FlyBodyRuntime
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flyppy-vision-smoke.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    course = FlyppyCourse(seed=args.seed, gate_count=3)
    world = FlyppyWorld(course)
    body = FlyBodyRuntime(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, 5.0),
        enable_vision=True,
    )

    readouts = body.ommatidia_readouts()
    if readouts.ndim != 3 or readouts.shape[0] != 2 or readouts.shape[2] != 2:
        raise RuntimeError(f"unexpected ommatidia shape: {readouts.shape}")
    if not np.all(np.isfinite(readouts)):
        raise RuntimeError("compound-eye output contains non-finite values")

    active = readouts.sum(axis=2)
    left = active[0]
    right = active[1]
    if float(np.std(active)) <= 1e-6:
        raise RuntimeError(
            "compound-eye output is effectively uniform; Flyppy geometry was not visually resolved"
        )

    result = {
        "shape": list(readouts.shape),
        "ommatidia_per_eye": int(readouts.shape[1]),
        "left_mean": float(np.mean(left)),
        "right_mean": float(np.mean(right)),
        "global_std": float(np.std(active)),
        "left_min_max": [float(np.min(left)), float(np.max(left))],
        "right_min_max": [float(np.min(right)), float(np.max(right))],
        "first_gate": {
            "x_mm": course.gates[0].x_mm,
            "center_z_mm": course.gates[0].center_z_mm,
            "low_z_mm": course.gates[0].low_z_mm,
            "high_z_mm": course.gates[0].high_z_mm,
        },
        "interpretation": (
            "FlyGym/FlyBody compound-eye ommatidia are rendering the physical Flyppy world. "
            "This does not define the MaleCNS retinotopic neuron mapping."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"ommatidia_per_eye={result['ommatidia_per_eye']}")
    print(f"global_std={result['global_std']:.6f}")
    print(f"result={args.output}")
    print("flyppy_vision_smoke=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
