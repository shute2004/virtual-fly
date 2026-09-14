#!/usr/bin/env python3
"""Measure whether the physical FlyBody seam can actually hold or gain altitude.

This is a CNS-independent body-mechanics diagnostic.  It supplies only symmetric,
constant bilateral DLM/DVM activation to the existing virtual-muscle seam and
measures vertical motion over many wingbeats.  No Flyppy observation, reward,
action decoder, learned policy, or prior neural trajectory is used.

The historical flight-envelope smoke primarily proved forward travel before
collision.  A body can satisfy that test while continuously losing altitude.
This diagnostic answers the stricter question required by the Flyppy task:

    does the fixed body + wing aerodynamics admit at least one operating point
    that sustains altitude, and at least one that ends above its starting height?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from wing_muscle_periphery import PeripheralSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.10)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--spawn-z-mm", type=float, default=8.25)
    parser.add_argument(
        "--forward-speeds-mm-s",
        type=float,
        nargs="+",
        default=(0.0, 300.0),
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00),
        help="DLM and DVM activation grid values in [0,1]",
    )
    parser.add_argument(
        "--sustain-tolerance-mm",
        type=float,
        default=0.25,
        help="maximum allowed end-height loss for altitude-sustaining classification",
    )
    parser.add_argument(
        "--climb-threshold-mm",
        type=float,
        default=0.10,
        help="minimum positive end-height gain for climb-capable classification",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-vertical-flight-capability.json"),
    )
    return parser.parse_args()


def power_state(dlm: float, dvm: float) -> PeripheralSnapshot:
    return PeripheralSnapshot(
        activation_by_body={},
        muscle_activation={},
        dlm_activation={"left": dlm, "right": dlm},
        dvm_activation={"left": dvm, "right": dvm},
        active_spikes=0,
        selected_motor_units=1,
    )


def run_case(
    body: FlyBodyMuscleAdapter,
    *,
    dlm: float,
    dvm: float,
    forward_speed_mm_s: float,
    seconds: float,
    physics_steps: int,
) -> dict[str, object]:
    body.reset()
    body.set_root_linear_velocity_mm_s((forward_speed_mm_s, 0.0, 0.0))
    state = power_state(dlm, dvm)
    start = body.thorax_position_mm()
    start_z = float(start[2])
    control_dt_s = body.timestep * physics_steps
    controls = max(1, math.ceil(seconds / control_dt_s))

    min_z = start_z
    max_z = start_z
    min_vz = 0.0
    max_vz = 0.0
    positive_vz_steps = 0

    for _ in range(controls):
        body.step_muscles(state, physics_steps=physics_steps)
        z = float(body.thorax_position_mm()[2])
        vz = float(body.root_linear_velocity_mm_s()[2])
        min_z = min(min_z, z)
        max_z = max(max_z, z)
        min_vz = min(min_vz, vz)
        max_vz = max(max_vz, vz)
        positive_vz_steps += int(vz > 0.0)

    end = body.thorax_position_mm()
    end_velocity = body.root_linear_velocity_mm_s()
    elapsed = controls * control_dt_s
    end_z = float(end[2])
    return {
        "dlm": dlm,
        "dvm": dvm,
        "forward_speed_mm_s": forward_speed_mm_s,
        "elapsed_seconds": elapsed,
        "start_z_mm": start_z,
        "end_z_mm": end_z,
        "delta_z_mm": end_z - start_z,
        "min_z_mm": min_z,
        "max_z_mm": max_z,
        "max_altitude_gain_mm": max_z - start_z,
        "max_altitude_loss_mm": start_z - min_z,
        "end_vz_mm_s": float(end_velocity[2]),
        "min_vz_mm_s": min_vz,
        "max_vz_mm_s": max_vz,
        "positive_vz_fraction": positive_vz_steps / controls,
        "end_x_mm": float(end[0]),
    }


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

    body = FlyBodyMuscleAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, args.spawn_z_mm),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=True,
        normalize_canonical_mass=True,
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

    result = {
        "schema_version": 1,
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
        "best_case": best,
        "worst_case": worst,
        "samples": samples,
        "interpretation": (
            "CNS-independent physical capability test. PASS requires that the fixed "
            "virtual-muscle and aerodynamic seam contains both an altitude-sustaining "
            "operating point and a positive end-height-gain operating point."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "vertical_flight samples={} sustaining={} climbing={}".format(
            len(samples), len(sustaining), len(climbing)
        )
    )
    print(
        "best dlm={:.2f} dvm={:.2f} vx={:.1f} dz={:+.3f} max_gain={:+.3f} "
        "end_vz={:+.3f} positive_vz_fraction={:.3f}".format(
            float(best["dlm"]),
            float(best["dvm"]),
            float(best["forward_speed_mm_s"]),
            float(best["delta_z_mm"]),
            float(best["max_altitude_gain_mm"]),
            float(best["end_vz_mm_s"]),
            float(best["positive_vz_fraction"]),
        )
    )
    print(f"result={args.output}")

    if not sustaining:
        raise RuntimeError(
            "virtual-muscle seam has no altitude-sustaining operating point; "
            "repair body flight physics before further neural training"
        )
    if not climbing:
        raise RuntimeError(
            "virtual-muscle seam has no positive altitude-gain operating point; "
            "repair body flight physics before further neural training"
        )

    print("flybody_vertical_flight_capability=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
