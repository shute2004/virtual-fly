#!/usr/bin/env python3
"""Diagnose symmetric open-loop Flyppy-v3 vertical flight capability.

This diagnostic uses the source flight wing frame, -47.5 degree root pose,
published 0.983 mg FlyBody mass, and official measured baseline wing cycle through
``FlyBodyV3MuscleAdapter``. No CNS, reward signal, policy, or Flyppy gate
observation is involved.

Stable free flight under a fixed symmetric muscle state is useful information, but
it is not a source FlyBody requirement: upstream flight applies closed-loop policy
action on top of the wing-beat generator.  By default this script still exits
non-zero when the sweep cannot sustain/climb so it remains useful as a strict
standalone diagnostic.  ``--nonblocking`` records and reports the same result
without treating open-loop instability as a neural-training veto.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from flybody_v3_adapter import FlyBodyV3MuscleAdapter
from flybody_vertical_flight_capability import power_state, run_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.10)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--spawn-z-mm", type=float, default=8.91)
    parser.add_argument(
        "--forward-speeds-mm-s", type=float, nargs="+", default=(0.0, 300.0)
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00),
    )
    parser.add_argument("--sustain-tolerance-mm", type=float, default=0.25)
    parser.add_argument("--climb-threshold-mm", type=float, default=0.10)
    parser.add_argument(
        "--nonblocking",
        action="store_true",
        help=(
            "record/report open-loop instability but return success; use when this "
            "diagnostic is embedded in a source-equivalent closed-loop preflight"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-v3-vertical-flight-capability.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seconds <= 0.0 or args.physics_steps < 1:
        raise SystemExit("seconds must be > 0 and physics-steps must be >= 1")
    if args.spawn_z_mm <= 0.0:
        raise SystemExit("spawn-z-mm must be positive")
    if args.sustain_tolerance_mm < 0.0 or args.climb_threshold_mm <= 0.0:
        raise SystemExit("vertical thresholds are invalid")
    if not args.levels or any(
        not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in args.levels
    ):
        raise SystemExit("levels must contain finite values in [0,1]")
    if any(not math.isfinite(v) or v < 0.0 for v in args.forward_speeds_mm_s):
        raise SystemExit("forward speeds must be finite and non-negative")

    body = FlyBodyV3MuscleAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, args.spawn_z_mm),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=True,
    )

    samples: list[dict[str, object]] = []
    for speed in args.forward_speeds_mm_s:
        for dlm in args.levels:
            for dvm in args.levels:
                sample = run_case(
                    body,
                    dlm=float(dlm),
                    dvm=float(dvm),
                    forward_speed_mm_s=float(speed),
                    seconds=args.seconds,
                    physics_steps=args.physics_steps,
                )
                sample["sustains_altitude"] = (
                    float(sample["delta_z_mm"]) >= -args.sustain_tolerance_mm
                )
                sample["climb_capable"] = (
                    float(sample["delta_z_mm"]) >= args.climb_threshold_mm
                )
                samples.append(sample)

    sustaining = [sample for sample in samples if bool(sample["sustains_altitude"])]
    climbing = [sample for sample in samples if bool(sample["climb_capable"])]
    best = max(samples, key=lambda sample: float(sample["delta_z_mm"]))
    worst = min(samples, key=lambda sample: float(sample["delta_z_mm"]))
    passed = bool(sustaining and climbing)

    result = {
        "schema_version": 2,
        "flight_physics_version": "v3-source-equivalent",
        "source_root_pitch_deg": -47.5,
        "published_mass_mg": 0.983,
        "baseline_wing_pattern": "official FlyBody wing_pattern_fmech.npy",
        "diagnostic_role": "symmetric-open-loop-stability",
        "training_veto": False if args.nonblocking else None,
        "seconds_requested": args.seconds,
        "physics_steps_per_control": args.physics_steps,
        "spawn_z_mm": args.spawn_z_mm,
        "forward_speeds_mm_s": list(args.forward_speeds_mm_s),
        "levels": list(args.levels),
        "sustain_tolerance_mm": args.sustain_tolerance_mm,
        "climb_threshold_mm": args.climb_threshold_mm,
        "sample_count": len(samples),
        "sustaining_count": len(sustaining),
        "climbing_count": len(climbing),
        "passed": passed,
        "best_case": best,
        "worst_case": worst,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "v3_vertical_flight samples={} sustaining={} climbing={} passed={}".format(
            len(samples), len(sustaining), len(climbing), passed
        )
    )
    print(
        "best dlm={:.2f} dvm={:.2f} vx={:.1f} dz={:+.3f} max_gain={:+.3f} "
        "end_vz={:+.3f}".format(
            float(best["dlm"]),
            float(best["dvm"]),
            float(best["forward_speed_mm_s"]),
            float(best["delta_z_mm"]),
            float(best["max_altitude_gain_mm"]),
            float(best["end_vz_mm_s"]),
        )
    )
    print(f"result={args.output}")

    if passed:
        print("flybody_v3_open_loop_stability=PASS")
        return 0
    if args.nonblocking:
        print("flybody_v3_open_loop_stability=FAIL_NONBLOCKING")
        print(
            "note=source FlyBody does not require fixed symmetric baseline-wingbeat "
            "free-flight stability; closed-loop MaleCNS control may provide stabilization"
        )
        return 0
    raise RuntimeError(
        "Flyppy v3 fixed symmetric open-loop sweep cannot both sustain and gain "
        "altitude; this strict diagnostic does not by itself veto closed-loop neural training"
    )


if __name__ == "__main__":
    raise SystemExit(main())
