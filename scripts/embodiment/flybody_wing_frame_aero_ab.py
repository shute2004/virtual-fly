#!/usr/bin/env python3
"""A/B the FlyGym wing-body pose correction against the original FlyBody frame.

FlyGym 2.1's experimental ``FlyBody`` rotates each wing body by +/-90 degrees in
``_correct_wing_default_pose`` so position actuators have a convenient resting
pose.  The original TuragaLab FlyBody flight model does not perform that
post-conversion body-frame correction; its measured wing kinematics, joint axes,
and fluid ellipsoids are defined in the original wing-body frame.

This calibration-only diagnostic leaves the normal virtual-fly runtime untouched.
It temporarily substitutes a FlyBody subclass whose
``_correct_wing_default_pose`` is a no-op, then reuses the exact-kinematic
wing-fluid support test.  Therefore the comparison isolates the wing-body-frame
conversion while bypassing CNS, muscles, actuator tracking, and wing inertia
tracking.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import flybody_runtime
from flygym.compose.fly import FlyBody as FlyGymFlyBody
from flybody_kinematic_aero_support import run_case
from flybody_measured_wingbeat import DEFAULT_PATTERN, MeasuredWingbeatCycle


class SourceWingFrameFlyBody(FlyGymFlyBody):
    """FlyGym FlyBody without its position-actuator wing-body frame correction."""

    def _correct_wing_default_pose(self) -> None:  # type: ignore[override]
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--phase-samples", type=int, default=1000)
    parser.add_argument(
        "--root-pitches-deg",
        type=float,
        nargs="+",
        default=(0.0, 47.5),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-wing-frame-aero-ab.json"),
    )
    return parser.parse_args()


def run_with_fly_class(
    fly_class,
    cycle: MeasuredWingbeatCycle,
    *,
    root_pitch_deg: float,
    phase_samples: int,
) -> dict[str, object]:
    original = flybody_runtime.FlyBody
    flybody_runtime.FlyBody = fly_class
    try:
        return run_case(
            cycle,
            root_pitch_deg=float(root_pitch_deg),
            leg_pose="source_retracted",
            phase_samples=phase_samples,
        )
    finally:
        flybody_runtime.FlyBody = original


def main() -> int:
    args = parse_args()
    if not args.pattern.exists():
        raise SystemExit(
            f"measured wing pattern missing: {args.pattern}; run "
            "uv run python scripts/dev/prefetch_flybody_flight_data.py"
        )
    if args.phase_samples < 100:
        raise SystemExit("phase-samples must be >= 100")
    if not args.root_pitches_deg or any(
        not math.isfinite(value) for value in args.root_pitches_deg
    ):
        raise SystemExit("root-pitches-deg must contain finite values")

    cycle = MeasuredWingbeatCycle(args.pattern)
    rows: list[dict[str, object]] = []
    modes = (
        ("flygym_corrected", FlyGymFlyBody),
        ("source_original", SourceWingFrameFlyBody),
    )
    for mode, fly_class in modes:
        for root_pitch in args.root_pitches_deg:
            row = run_with_fly_class(
                fly_class,
                cycle,
                root_pitch_deg=float(root_pitch),
                phase_samples=args.phase_samples,
            )
            row = dict(row)
            row["wing_frame_mode"] = mode
            rows.append(row)
            print(
                "wing_frame_aero mode={} root_pitch={:.1f} support_ratio={:.3f} "
                "support_accel={:+.1f}".format(
                    mode,
                    float(root_pitch),
                    float(row["support_ratio_to_weight"]),
                    float(row["mean_support_accel_mm_s2"]),
                )
            )

    corrected = max(
        (row for row in rows if row["wing_frame_mode"] == "flygym_corrected"),
        key=lambda row: float(row["support_ratio_to_weight"]),
    )
    source = max(
        (row for row in rows if row["wing_frame_mode"] == "source_original"),
        key=lambda row: float(row["support_ratio_to_weight"]),
    )
    corrected_ratio = float(corrected["support_ratio_to_weight"])
    source_ratio = float(source["support_ratio_to_weight"])
    improvement = source_ratio / corrected_ratio if corrected_ratio != 0.0 else float("inf")

    if 0.70 <= source_ratio <= 1.30:
        diagnosis = "FLYGYM_WING_FRAME_CORRECTION_IS_PRIMARY_AERO_PORT_MISMATCH"
    elif source_ratio >= corrected_ratio * 1.5:
        diagnosis = "FLYGYM_WING_FRAME_CORRECTION_STRONGLY_REDUCES_LIFT_BUT_IS_NOT_ONLY_MISMATCH"
    else:
        diagnosis = "WING_FRAME_CORRECTION_NOT_PRIMARY_AERO_MISMATCH"

    result = {
        "schema_version": 1,
        "pattern": str(args.pattern),
        "phase_samples": int(args.phase_samples),
        "cases": rows,
        "best_flygym_corrected_support_ratio": corrected_ratio,
        "best_source_original_support_ratio": source_ratio,
        "source_to_corrected_support_ratio": improvement,
        "diagnosis": diagnosis,
        "interpretation": (
            "The source_original mode disables only FlyGym 2.1's +/-90 degree "
            "_correct_wing_default_pose transformation. Exact measured wing qpos/qvel "
            "are imposed in both modes, so actuator and neural layers are excluded."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
