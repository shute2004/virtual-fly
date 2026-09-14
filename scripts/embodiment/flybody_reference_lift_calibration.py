#!/usr/bin/env python3
"""Reproduce the FlyBody reference lift-calibration seam as closely as possible.

This diagnostic is deliberately below the CNS and below the virtual-muscle layer.
It drives the six wing joints with the official measured baseline wing-beat pattern
and asks two independent questions:

1. Do the converted wing actuators track the target angles with source-like error?
2. With the body held in the source flight attitude, does the aerodynamic model
   generate mean upward support comparable to body weight?

The published FlyBody methods tuned wing gain/damping until wing tracking error
was below roughly 5% of wing-angle amplitude, then tuned ``fluidcoef`` so mean
lift approximately balanced the 0.983 mg model weight over several wing beats.
The fixed baseline WPG alone is *not* expected to produce stable free hover, so
this script does not use free-flight height as the primary port-equivalence test.

Implementation note
-------------------
The current FlyGym 2.1 FlyBody keeps all biological leg joints.  The original
FlyBody flight task retracted the legs and removed their DoFs.  For this
calibration-only test we emulate the corresponding static geometry by putting all
66 leg DoFs at their spring-reference angles and resetting them every physics
step.  This is sufficient to isolate wing tracking and vertical aerodynamic
support without pretending that the current training model has frozen legs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from flybody_adapter import FlyBodyWingAdapter, WingDrive
from flybody_biophysics import normalize_fly_mass
from flybody_measured_wingbeat import DEFAULT_PATTERN, MeasuredWingbeatCycle, install_measured_wingbeat


PUBLISHED_FLYBODY_MASS_MG = 0.983
SOURCE_GRAVITY_MM_S2 = 9810.0
LEG_PREFIXES = ("lf_", "lm_", "lh_", "rf_", "rm_", "rh_")
AXES = ("yaw", "roll", "pitch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--warmup-wingbeats", type=float, default=2.0)
    parser.add_argument("--measure-wingbeats", type=float, default=6.0)
    parser.add_argument(
        "--leg-poses",
        nargs="+",
        choices=("neutral", "source_retracted"),
        default=("neutral", "source_retracted"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-reference-lift-calibration.json"),
    )
    return parser.parse_args()


def is_leg_dof(dof) -> bool:
    return str(dof.child.name).startswith(LEG_PREFIXES)


def leg_springref_state(body) -> list[tuple[int, int, float]]:
    state: list[tuple[int, int, float]] = []
    for dof in body._all_dofs:
        if not is_leg_dof(dof):
            continue
        qpos_address, qvel_address = body._joint_state_addresses(dof)
        spring = float(body.sim.mj_model.qpos_spring[qpos_address])
        state.append((qpos_address, qvel_address, spring))
    if len(state) != 66:
        raise RuntimeError(f"expected 66 FlyBody leg DoFs, found {len(state)}")
    return state


def enforce_leg_pose(body, leg_pose: str, spring_state: list[tuple[int, int, float]]) -> None:
    if leg_pose == "neutral":
        return
    if leg_pose != "source_retracted":
        raise KeyError(leg_pose)
    for qpos_address, qvel_address, spring in spring_state:
        body.sim.mj_data.qpos[qpos_address] = spring
        body.sim.mj_data.qvel[qvel_address] = 0.0


def root_reference(body) -> tuple[np.ndarray, int, int]:
    qpos_address = body._root_freejoint_qpos_address()
    qvel_address = body._root_freejoint_dof_address()
    return (
        np.asarray(body.sim.mj_data.qpos[qpos_address : qpos_address + 7], dtype=np.float64).copy(),
        qpos_address,
        qvel_address,
    )


def reset_root_kinematics(
    body,
    root_qpos: np.ndarray,
    qpos_address: int,
    qvel_address: int,
) -> None:
    body.sim.mj_data.qpos[qpos_address : qpos_address + 7] = root_qpos
    body.sim.mj_data.qvel[qvel_address : qvel_address + 6] = 0.0


def make_body(pattern: Path, *, aerodynamics: bool) -> tuple[FlyBodyWingAdapter, MeasuredWingbeatCycle]:
    body = FlyBodyWingAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 8.25),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=aerodynamics,
        normalize_canonical_mass=False,
    )
    normalize_fly_mass(
        body.sim,
        body.fly,
        target_mass_g=PUBLISHED_FLYBODY_MASS_MG / 1000.0,
    )
    cycle = install_measured_wingbeat(body, pattern)
    return body, cycle


def run_clamped_case(
    pattern: Path,
    *,
    aerodynamics: bool,
    leg_pose: str,
    warmup_wingbeats: float,
    measure_wingbeats: float,
) -> dict[str, object]:
    body, cycle = make_body(pattern, aerodynamics=aerodynamics)
    spring_state = leg_springref_state(body)
    enforce_leg_pose(body, leg_pose, spring_state)
    mj.mj_forward(body.sim.mj_model, body.sim.mj_data)
    root_qpos, root_qpos_address, root_qvel_address = root_reference(body)

    wing_period_s = 1.0 / body.wingbeat_hz
    total_s = (warmup_wingbeats + measure_wingbeats) * wing_period_s
    warmup_s = warmup_wingbeats * wing_period_s
    steps = max(1, math.ceil(total_s / body.timestep))

    errors: dict[str, list[float]] = {axis: [] for axis in AXES}
    net_vertical_accels: list[float] = []
    root_speed_before_step = 0.0

    for _ in range(steps):
        # The source calibration places the body in a fixed flight attitude.  Reset
        # root position/orientation/velocity while allowing wing dynamics to evolve.
        reset_root_kinematics(
            body, root_qpos, root_qpos_address, root_qvel_address
        )
        enforce_leg_pose(body, leg_pose, spring_state)
        mj.mj_forward(body.sim.mj_model, body.sim.mj_data)

        phase = (2.0 * math.pi * body.wingbeat_hz * body._time) % (2.0 * math.pi)
        target = cycle.angles(phase, 1.0)
        root_speed_before_step = float(
            body.sim.mj_data.qvel[root_qvel_address + 2]
        )
        body.step(WingDrive(0.0, 0.0), physics_steps=1)
        root_speed_after_step = float(
            body.sim.mj_data.qvel[root_qvel_address + 2]
        )

        if body._time < warmup_s:
            continue
        net_vertical_accels.append(
            (root_speed_after_step - root_speed_before_step) / body.timestep
        )
        actual = body.wing_joint_angles_rad()
        for axis in AXES:
            errors[axis].append(abs(actual[f"left_{axis}"] - target[axis]))
            errors[axis].append(abs(actual[f"right_{axis}"] - target[axis]))

    if not net_vertical_accels:
        raise RuntimeError("no samples collected in reference lift calibration")

    amplitudes = np.ptp(cycle.pattern, axis=0)
    tracking: dict[str, dict[str, float]] = {}
    for i, axis in enumerate(AXES):
        values = np.asarray(errors[axis], dtype=np.float64)
        amp = float(amplitudes[i])
        tracking[axis] = {
            "amplitude_rad": amp,
            "mean_abs_error_rad": float(np.mean(values)),
            "normalized_mean_abs_error": (
                float(np.mean(values)) / amp if amp > 0.0 else float("inf")
            ),
            "max_abs_error_rad": float(np.max(values)),
        }

    return {
        "aerodynamics": aerodynamics,
        "leg_pose": leg_pose,
        "body_mass_mg": PUBLISHED_FLYBODY_MASS_MG,
        "body_pitch_deg": body.flight_body_pitch_deg,
        "wingbeat_hz": body.wingbeat_hz,
        "physics_timestep_s": body.timestep,
        "pattern_samples": cycle.samples,
        "warmup_wingbeats": warmup_wingbeats,
        "measure_wingbeats": measure_wingbeats,
        "mean_net_vertical_acceleration_mm_s2": float(np.mean(net_vertical_accels)),
        "std_net_vertical_acceleration_mm_s2": float(np.std(net_vertical_accels)),
        "tracking": tracking,
    }


def main() -> int:
    args = parse_args()
    if not args.pattern.exists():
        raise SystemExit(
            f"measured wing pattern missing: {args.pattern}; run "
            "uv run python scripts/dev/prefetch_flybody_flight_data.py"
        )
    if args.warmup_wingbeats < 0.0 or args.measure_wingbeats <= 0.0:
        raise SystemExit("wingbeat counts are invalid")

    cases: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for leg_pose in args.leg_poses:
        off = run_clamped_case(
            args.pattern,
            aerodynamics=False,
            leg_pose=leg_pose,
            warmup_wingbeats=args.warmup_wingbeats,
            measure_wingbeats=args.measure_wingbeats,
        )
        on = run_clamped_case(
            args.pattern,
            aerodynamics=True,
            leg_pose=leg_pose,
            warmup_wingbeats=args.warmup_wingbeats,
            measure_wingbeats=args.measure_wingbeats,
        )
        cases.extend((off, on))

        off_accel = float(off["mean_net_vertical_acceleration_mm_s2"])
        on_accel = float(on["mean_net_vertical_acceleration_mm_s2"])
        aerodynamic_support = on_accel - off_accel
        support_ratio = aerodynamic_support / SOURCE_GRAVITY_MM_S2
        max_tracking_error = max(
            float(on["tracking"][axis]["normalized_mean_abs_error"])
            for axis in AXES
        )
        comparisons.append(
            {
                "leg_pose": leg_pose,
                "no_aero_mean_net_accel_mm_s2": off_accel,
                "aero_mean_net_accel_mm_s2": on_accel,
                "aerodynamic_support_accel_mm_s2": aerodynamic_support,
                "support_ratio_to_weight": support_ratio,
                "max_normalized_tracking_mae": max_tracking_error,
                "source_tracking_target_lt": 0.05,
            }
        )

    # Diagnostic classification, intentionally broad.  Exact acceptance thresholds
    # will be locked only after comparing this port against the upstream model.
    best = max(comparisons, key=lambda row: float(row["support_ratio_to_weight"]))
    if float(best["max_normalized_tracking_mae"]) >= 0.05:
        diagnosis = "WING_ACTUATOR_OR_INERTIA_PORT_MISMATCH"
    elif float(best["support_ratio_to_weight"]) < 0.70:
        diagnosis = "AERODYNAMIC_PORT_UNDERPRODUCES_REFERENCE_LIFT"
    elif float(best["support_ratio_to_weight"]) > 1.30:
        diagnosis = "AERODYNAMIC_PORT_OVERPRODUCES_REFERENCE_LIFT"
    else:
        diagnosis = "REFERENCE_LIFT_PORT_APPROXIMATELY_CONSISTENT"

    result = {
        "schema_version": 1,
        "pattern": str(args.pattern),
        "published_flybody_mass_mg": PUBLISHED_FLYBODY_MASS_MG,
        "source_gravity_mm_s2": SOURCE_GRAVITY_MM_S2,
        "cases": cases,
        "comparisons": comparisons,
        "diagnosis": diagnosis,
        "interpretation": (
            "Calibration-only test of measured wing-angle tracking and mean vertical "
            "aerodynamic support. Stable free hover is deliberately not required because "
            "the FlyBody paper states that the fixed baseline WPG alone lacks the feedback "
            "needed for stable hover."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for row in comparisons:
        print(
            "reference_lift leg_pose={} no_aero_accel={:+.1f} aero_accel={:+.1f} "
            "support={:+.1f} support_ratio={:.3f} max_tracking_nmae={:.4f}".format(
                row["leg_pose"],
                float(row["no_aero_mean_net_accel_mm_s2"]),
                float(row["aero_mean_net_accel_mm_s2"]),
                float(row["aerodynamic_support_accel_mm_s2"]),
                float(row["support_ratio_to_weight"]),
                float(row["max_normalized_tracking_mae"]),
            )
        )
    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
