#!/usr/bin/env python3
"""Diagnose why the current FlyBody seam cannot sustain altitude.

This is calibration-only and CNS-independent. It compares three mass regimes and
two wing-control seams:

* current v2 mass: 1.09 mg;
* published FlyBody mass: 0.983 mg;
* native compiled FlyGym/FlyBody mass before any virtual-fly normalization;

and

* the current symmetric DLM/DVM virtual-muscle seam;
* the source-style simple WingBeatPatternGenerator fallback driven through the
  position-reference adapter.

The purpose is causal localization, not task control. No Flyppy observation,
reward, learned policy, CNS activity, or previous trajectory is used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flybody_adapter import FlyBodyWingAdapter, WingDrive
from flybody_biophysics import normalize_fly_mass
from flybody_muscle_adapter import FlyBodyMuscleAdapter
from wing_muscle_periphery import PeripheralSnapshot

CURRENT_V2_MASS_MG = 1.09
PUBLISHED_FLYBODY_MASS_MG = 0.983


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
        default=(0.25, 0.40, 0.60, 0.80, 1.00),
    )
    parser.add_argument("--sustain-tolerance-mm", type=float, default=0.25)
    parser.add_argument("--climb-threshold-mm", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-vertical-flight-diagnosis.json"),
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


def compiled_mass_mg(body) -> float:
    # World/static bodies have zero mass. This diagnostic contains one dynamic fly.
    mass_g = float(np.sum(np.asarray(body.sim.mj_model.body_mass)[1:]))
    if not math.isfinite(mass_g) or mass_g <= 0.0:
        raise RuntimeError(f"invalid compiled body mass: {mass_g}")
    return mass_g * 1000.0


def apply_mass_case(body, target_mass_mg: float | None) -> dict[str, float | str]:
    native_mass_mg = compiled_mass_mg(body)
    if target_mass_mg is None:
        return {
            "mode": "native",
            "native_mass_mg": native_mass_mg,
            "actual_mass_mg": native_mass_mg,
            "scale": 1.0,
        }
    normalized = normalize_fly_mass(
        body.sim,
        body.fly,
        target_mass_g=float(target_mass_mg) / 1000.0,
    )
    return {
        "mode": "normalized",
        "native_mass_mg": native_mass_mg,
        "actual_mass_mg": normalized.target_mass_g * 1000.0,
        "scale": normalized.scale,
    }


def vertical_metrics(body, *, seconds: float, physics_steps: int, step_once) -> dict[str, float]:
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
        step_once()
        z = float(body.thorax_position_mm()[2])
        vz = float(body.root_linear_velocity_mm_s()[2])
        min_z = min(min_z, z)
        max_z = max(max_z, z)
        min_vz = min(min_vz, vz)
        max_vz = max(max_vz, vz)
        positive_vz_steps += int(vz > 0.0)

    end = body.thorax_position_mm()
    end_velocity = body.root_linear_velocity_mm_s()
    end_z = float(end[2])
    return {
        "elapsed_seconds": controls * control_dt_s,
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


def classify(sample: dict[str, object], sustain_tol: float, climb_threshold: float) -> None:
    dz = float(sample["delta_z_mm"])
    sample["sustains_altitude"] = dz >= -sustain_tol
    sample["climb_capable"] = dz >= climb_threshold


def run_virtual_grid(
    *,
    target_mass_mg: float | None,
    label: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    body = FlyBodyMuscleAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, args.spawn_z_mm),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=True,
        normalize_canonical_mass=False,
    )
    mass = apply_mass_case(body, target_mass_mg)
    samples: list[dict[str, object]] = []
    for speed in args.forward_speeds_mm_s:
        for dlm in args.levels:
            for dvm in args.levels:
                body.reset()
                body.set_root_linear_velocity_mm_s((float(speed), 0.0, 0.0))
                state = power_state(float(dlm), float(dvm))
                sample: dict[str, object] = {
                    "seam": "virtual_muscle",
                    "mass_case": label,
                    "actual_mass_mg": mass["actual_mass_mg"],
                    "forward_speed_mm_s": float(speed),
                    "dlm": float(dlm),
                    "dvm": float(dvm),
                }
                sample.update(
                    vertical_metrics(
                        body,
                        seconds=args.seconds,
                        physics_steps=args.physics_steps,
                        step_once=lambda: body.step_muscles(state, physics_steps=args.physics_steps),
                    )
                )
                classify(sample, args.sustain_tolerance_mm, args.climb_threshold_mm)
                samples.append(sample)
    best = max(samples, key=lambda row: float(row["delta_z_mm"]))
    return {
        "mass_case": label,
        "mass": mass,
        "sample_count": len(samples),
        "sustaining_count": sum(bool(row["sustains_altitude"]) for row in samples),
        "climbing_count": sum(bool(row["climb_capable"]) for row in samples),
        "best_case": best,
        "samples": samples,
    }


def run_position_reference(
    *,
    target_mass_mg: float | None,
    label: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    body = FlyBodyWingAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, args.spawn_z_mm),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=True,
        normalize_canonical_mass=False,
    )
    mass = apply_mass_case(body, target_mass_mg)
    samples: list[dict[str, object]] = []
    for speed in args.forward_speeds_mm_s:
        body.reset()
        body.set_root_linear_velocity_mm_s((float(speed), 0.0, 0.0))
        sample: dict[str, object] = {
            "seam": "simple_wbpg_position_reference",
            "mass_case": label,
            "actual_mass_mg": mass["actual_mass_mg"],
            "forward_speed_mm_s": float(speed),
        }
        sample.update(
            vertical_metrics(
                body,
                seconds=args.seconds,
                physics_steps=args.physics_steps,
                step_once=lambda: body.step(WingDrive(0.0, 0.0), physics_steps=args.physics_steps),
            )
        )
        classify(sample, args.sustain_tolerance_mm, args.climb_threshold_mm)
        samples.append(sample)
    best = max(samples, key=lambda row: float(row["delta_z_mm"]))
    return {
        "mass_case": label,
        "mass": mass,
        "sample_count": len(samples),
        "sustaining_count": sum(bool(row["sustains_altitude"]) for row in samples),
        "climbing_count": sum(bool(row["climb_capable"]) for row in samples),
        "best_case": best,
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    if args.seconds <= 0.0 or args.physics_steps < 1:
        raise SystemExit("seconds must be > 0 and physics-steps must be >= 1")
    if args.spawn_z_mm <= 0.0:
        raise SystemExit("spawn-z-mm must be positive")
    if not args.levels or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in args.levels):
        raise SystemExit("levels must contain finite values in [0,1]")

    mass_cases = (
        ("current_v2_1.09mg", CURRENT_V2_MASS_MG),
        ("flybody_published_0.983mg", PUBLISHED_FLYBODY_MASS_MG),
        ("native_compiled", None),
    )
    virtual_results = []
    reference_results = []
    for label, target in mass_cases:
        virtual_results.append(run_virtual_grid(target_mass_mg=target, label=label, args=args))
        reference_results.append(run_position_reference(target_mass_mg=target, label=label, args=args))

    native_mass_mg = float(virtual_results[-1]["mass"]["actual_mass_mg"])
    published_ref = next(row for row in reference_results if row["mass_case"] == "flybody_published_0.983mg")
    published_virtual = next(row for row in virtual_results if row["mass_case"] == "flybody_published_0.983mg")

    if int(published_virtual["sustaining_count"]) > 0:
        diagnosis = "CURRENT_MASS_WAS_A_MAJOR_CAUSE"
    elif int(published_ref["sustaining_count"]) > 0:
        diagnosis = "VIRTUAL_MUSCLE_MAPPING_IS_PRIMARY_CAUSE"
    else:
        diagnosis = "SIMPLE_WBPG_OR_FLIGHT_POSTURE_STILL_INSUFFICIENT"

    result = {
        "schema_version": 1,
        "seconds_requested": args.seconds,
        "physics_steps_per_control": args.physics_steps,
        "spawn_z_mm": args.spawn_z_mm,
        "forward_speeds_mm_s": list(args.forward_speeds_mm_s),
        "levels": list(args.levels),
        "sustain_tolerance_mm": args.sustain_tolerance_mm,
        "climb_threshold_mm": args.climb_threshold_mm,
        "current_v2_mass_mg": CURRENT_V2_MASS_MG,
        "published_flybody_mass_mg": PUBLISHED_FLYBODY_MASS_MG,
        "native_compiled_mass_mg": native_mass_mg,
        "virtual_muscle": virtual_results,
        "simple_wbpg_position_reference": reference_results,
        "diagnosis": diagnosis,
        "notes": [
            "Published FlyBody flight used a measured 0.983 mg female model.",
            "The simple analytic WBPG pattern is the FlyBody source fallback for testing/prototyping, not the realistic recorded baseline used for final flight results.",
            "Original FlyBody flight tasks retracted and froze leg DoFs; this diagnosis intentionally leaves the current virtual-fly body construction unchanged so mass and wing seam can be isolated first.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"native_compiled_mass_mg={native_mass_mg:.6f}")
    for row in reference_results:
        best = row["best_case"]
        print(
            "reference mass_case={} mass_mg={:.6f} sustaining={} climbing={} best_dz={:+.3f} best_vx={:.1f}".format(
                row["mass_case"],
                float(row["mass"]["actual_mass_mg"]),
                int(row["sustaining_count"]),
                int(row["climbing_count"]),
                float(best["delta_z_mm"]),
                float(best["forward_speed_mm_s"]),
            )
        )
    for row in virtual_results:
        best = row["best_case"]
        print(
            "virtual mass_case={} mass_mg={:.6f} sustaining={} climbing={} best_dz={:+.3f} dlm={:.2f} dvm={:.2f} vx={:.1f}".format(
                row["mass_case"],
                float(row["mass"]["actual_mass_mg"]),
                int(row["sustaining_count"]),
                int(row["climbing_count"]),
                float(best["delta_z_mm"]),
                float(best["dlm"]),
                float(best["dvm"]),
                float(best["forward_speed_mm_s"]),
            )
        )
    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
