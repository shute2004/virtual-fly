#!/usr/bin/env python3
"""Diagnose lift loss between Flyppy-v3 source kinematics and virtual muscles.

The Flyppy-v3 body already restores the source wing frame, source -47.5 degree
root pose, published mass, flight fluid geometry, and the official measured
wingbeat.  This diagnostic keeps that same physical body and compares four
seams while the root is re-anchored so free-flight attitude/fall cannot hide the
cause:

1. exact_kinematic:
   impose the measured qpos/qvel exactly and measure total MuJoCo fluid support.
2. source_force:
   reproduce upstream FlyBody's held force command at the 0.2 ms control cadence.
3. virtual_muscle:
   run the production virtual-muscle seam at bilateral DLM=DVM=1.
4. virtual_muscle_no_position:
   same as (3), but zero the compiled POSITION actuator gain/bias to prove that
   the legacy six wing actuators are or are not mechanically interfering.

No CNS, reward signal, policy, Flyppy gate observation, or learned state is used.
This script is diagnostic-only and does not calibrate gains.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from flybody_measured_wingbeat import (
    AXES,
    DEFAULT_PATTERN,
    SOURCE_CONTROL_TIMESTEP_S,
    MeasuredWingbeatCycle,
)
from flybody_reference_lift_calibration import (
    apply_held_source_force,
    disable_position_wing_actuators,
    set_wing_state_from_vector,
    source_controller,
    source_force_command,
)
from flybody_v3_adapter import FlyBodyV3MuscleAdapter
from flybody_vertical_flight_capability import power_state


SOURCE_GRAVITY_MM_S2 = 9810.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--phase-samples", type=int, default=1000)
    parser.add_argument("--warmup-wingbeats", type=float, default=2.0)
    parser.add_argument("--measure-wingbeats", type=float, default=6.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-v3-muscle-flight-diagnosis.json"),
    )
    return parser.parse_args()


def make_body() -> FlyBodyV3MuscleAdapter:
    return FlyBodyV3MuscleAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 8.91),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=True,
    )


def fly_mass_g(body: FlyBodyV3MuscleAdapter) -> float:
    ids: list[int] = []
    for segment in body.fly.get_bodysegs_order():
        name = body.fly.bodyseg_to_mjcfbody[segment].name
        body_id = mj.mj_name2id(body.sim.mj_model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id <= 0:
            raise RuntimeError(f"compiled FlyBody body not found: {name}")
        ids.append(body_id)
    return float(np.sum(np.asarray(body.sim.mj_model.body_mass)[sorted(set(ids))]))


def root_reference(
    body: FlyBodyV3MuscleAdapter,
) -> tuple[np.ndarray, int, int]:
    qpos_address = body._root_freejoint_qpos_address()
    qvel_address = body._root_freejoint_dof_address()
    qpos = np.asarray(
        body.sim.mj_data.qpos[qpos_address : qpos_address + 7],
        dtype=np.float64,
    ).copy()
    return qpos, qpos_address, qvel_address


def anchor_root(
    body: FlyBodyV3MuscleAdapter,
    root_qpos: np.ndarray,
    root_qpos_address: int,
    root_qvel_address: int,
) -> None:
    body.sim.mj_data.qpos[root_qpos_address : root_qpos_address + 7] = root_qpos
    body.sim.mj_data.qvel[root_qvel_address : root_qvel_address + 6] = 0.0
    mj.mj_forward(body.sim.mj_model, body.sim.mj_data)


def set_exact_wings(
    body: FlyBodyV3MuscleAdapter,
    cycle: MeasuredWingbeatCycle,
    phase: float,
) -> dict[str, float]:
    angles = cycle.angles(phase, 1.0)
    velocities = cycle.velocities(phase, 1.0, body.wingbeat_hz)
    for side in ("left", "right"):
        for axis in AXES:
            actuator_index = body._wing_indices[(side, axis)]
            dof = body._actuated_dofs[actuator_index]
            qpos_address, qvel_address = body._wing_state_addresses(dof)
            body.sim.mj_data.qpos[qpos_address] = angles[axis]
            body.sim.mj_data.qvel[qvel_address] = velocities[axis]
    return angles


def fluid_vector(body: FlyBodyV3MuscleAdapter, root_qvel_address: int) -> np.ndarray:
    return np.asarray(
        body.sim.mj_data.qfrc_fluid[root_qvel_address : root_qvel_address + 3],
        dtype=np.float64,
    ).copy()


def summarize_fluid(
    vectors: list[np.ndarray],
    *,
    mass_g: float,
) -> dict[str, object]:
    if not vectors:
        raise RuntimeError("no fluid-force samples were collected")
    array = np.asarray(vectors, dtype=np.float64)
    mean_xyz = np.mean(array, axis=0)
    vertical_ratio = float(mean_xyz[2] / mass_g / SOURCE_GRAVITY_MM_S2)
    magnitude_ratio = float(
        np.mean(np.linalg.norm(array, axis=1)) / mass_g / SOURCE_GRAVITY_MM_S2
    )
    return {
        "mean_force_xyz_g_mm_s2": [float(value) for value in mean_xyz],
        "vertical_support_ratio_to_weight": vertical_ratio,
        "mean_force_magnitude_ratio_to_weight": magnitude_ratio,
    }


def tracking_summary(
    cycle: MeasuredWingbeatCycle,
    errors: dict[str, list[float]],
) -> dict[str, object]:
    amplitudes = np.ptp(cycle.pattern, axis=0)
    by_axis: dict[str, dict[str, float]] = {}
    for index, axis in enumerate(AXES):
        values = np.asarray(errors[axis], dtype=np.float64)
        if values.size == 0:
            raise RuntimeError(f"no tracking samples for {axis}")
        amplitude = float(amplitudes[index])
        mae = float(np.mean(values))
        by_axis[axis] = {
            "amplitude_rad": amplitude,
            "mean_abs_error_rad": mae,
            "normalized_mean_abs_error": (
                mae / amplitude if amplitude > 0.0 else float("inf")
            ),
            "max_abs_error_rad": float(np.max(values)),
        }
    return {
        "by_axis": by_axis,
        "max_normalized_mean_abs_error": max(
            float(row["normalized_mean_abs_error"]) for row in by_axis.values()
        ),
    }


def collect_tracking(
    body: FlyBodyV3MuscleAdapter,
    desired: dict[str, float],
    errors: dict[str, list[float]],
) -> None:
    actual = body.wing_joint_angles_rad()
    for axis in AXES:
        errors[axis].append(abs(actual[f"left_{axis}"] - desired[axis]))
        errors[axis].append(abs(actual[f"right_{axis}"] - desired[axis]))


def run_exact(
    cycle: MeasuredWingbeatCycle,
    *,
    phase_samples: int,
) -> dict[str, object]:
    body = make_body()
    mass_g = fly_mass_g(body)
    root_qpos, root_qpos_address, root_qvel_address = root_reference(body)
    vectors: list[np.ndarray] = []

    for index in range(phase_samples):
        phase = 2.0 * math.pi * index / phase_samples
        body.sim.mj_data.qpos[root_qpos_address : root_qpos_address + 7] = root_qpos
        body.sim.mj_data.qvel[root_qvel_address : root_qvel_address + 6] = 0.0
        set_exact_wings(body, cycle, phase)
        body.sim.mj_data.qfrc_applied[:] = 0.0
        mj.mj_forward(body.sim.mj_model, body.sim.mj_data)
        vectors.append(fluid_vector(body, root_qvel_address))

    result = summarize_fluid(vectors, mass_g=mass_g)
    result.update(
        {
            "mode": "exact_kinematic",
            "samples": phase_samples,
            "body_mass_mg": mass_g * 1000.0,
        }
    )
    return result


def run_source_force(
    cycle: MeasuredWingbeatCycle,
    *,
    warmup_wingbeats: float,
    measure_wingbeats: float,
) -> dict[str, object]:
    body = make_body()
    mass_g = fly_mass_g(body)
    controller = source_controller(body, cycle)
    initial_qpos, initial_qvel = controller.reset(initial_phase=0.0)
    set_wing_state_from_vector(body, initial_qpos, initial_qvel)
    disable_position_wing_actuators(body)
    body._time = 0.0

    root_qpos, root_qpos_address, root_qvel_address = root_reference(body)
    control_stride = int(round(SOURCE_CONTROL_TIMESTEP_S / body.timestep))
    wing_period_s = 1.0 / body.wingbeat_hz
    warmup_s = warmup_wingbeats * wing_period_s
    total_s = (warmup_wingbeats + measure_wingbeats) * wing_period_s
    steps = max(1, math.ceil(total_s / body.timestep))

    held_target = controller.current()
    held_force = source_force_command(body, held_target)
    vectors: list[np.ndarray] = []
    errors: dict[str, list[float]] = {axis: [] for axis in AXES}
    peak_force = 0.0

    for physics_index in range(steps):
        anchor_root(body, root_qpos, root_qpos_address, root_qvel_address)
        if physics_index % control_stride == 0:
            held_target = controller.step()
            held_force = source_force_command(body, held_target)
        peak_force = max(peak_force, apply_held_source_force(body, held_force))
        anchor_root(body, root_qpos, root_qpos_address, root_qvel_address)
        if body._time < warmup_s:
            continue
        vectors.append(fluid_vector(body, root_qvel_address))
        desired = {axis: float(held_target[i]) for i, axis in enumerate(AXES)}
        collect_tracking(body, desired, errors)

    result = summarize_fluid(vectors, mass_g=mass_g)
    result.update(
        {
            "mode": "source_force",
            "body_mass_mg": mass_g * 1000.0,
            "peak_abs_drive_force": peak_force,
            "tracking": tracking_summary(cycle, errors),
        }
    )
    return result


def run_virtual_muscle(
    cycle: MeasuredWingbeatCycle,
    *,
    disable_position: bool,
    warmup_wingbeats: float,
    measure_wingbeats: float,
) -> dict[str, object]:
    body = make_body()
    mass_g = fly_mass_g(body)
    if disable_position:
        disable_position_wing_actuators(body)

    state = power_state(1.0, 1.0)
    root_qpos, root_qpos_address, root_qvel_address = root_reference(body)
    wing_period_s = 1.0 / body.wingbeat_hz
    warmup_s = warmup_wingbeats * wing_period_s
    total_s = (warmup_wingbeats + measure_wingbeats) * wing_period_s
    steps = max(1, math.ceil(total_s / body.timestep))

    vectors: list[np.ndarray] = []
    errors: dict[str, list[float]] = {axis: [] for axis in AXES}
    peak_torque = 0.0

    for _ in range(steps):
        anchor_root(body, root_qpos, root_qpos_address, root_qvel_address)
        body.step_muscles(state, physics_steps=1)
        for value in body.last_wing_torque.values():
            peak_torque = max(peak_torque, abs(float(value)))
        anchor_root(body, root_qpos, root_qpos_address, root_qvel_address)
        if body._time < warmup_s:
            continue
        vectors.append(fluid_vector(body, root_qvel_address))
        phase = (2.0 * math.pi * body.wingbeat_hz * body._time) % (2.0 * math.pi)
        desired = cycle.angles(phase, 1.0)
        collect_tracking(body, desired, errors)

    result = summarize_fluid(vectors, mass_g=mass_g)
    result.update(
        {
            "mode": (
                "virtual_muscle_no_position" if disable_position else "virtual_muscle"
            ),
            "body_mass_mg": mass_g * 1000.0,
            "peak_abs_virtual_muscle_torque": peak_torque,
            "tracking": tracking_summary(cycle, errors),
        }
    )
    return result


def main() -> int:
    args = parse_args()
    if not args.pattern.exists():
        raise SystemExit(
            f"measured wing pattern missing: {args.pattern}; run "
            "uv run python scripts/dev/prefetch_flybody_flight_data.py"
        )
    if args.phase_samples < 100:
        raise SystemExit("phase-samples must be >= 100")
    if args.warmup_wingbeats < 0.0 or args.measure_wingbeats <= 0.0:
        raise SystemExit("wingbeat counts are invalid")

    cycle = MeasuredWingbeatCycle(args.pattern)
    exact = run_exact(cycle, phase_samples=args.phase_samples)
    source_force = run_source_force(
        cycle,
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )
    virtual = run_virtual_muscle(
        cycle,
        disable_position=False,
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )
    virtual_no_position = run_virtual_muscle(
        cycle,
        disable_position=True,
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )
    rows = [exact, source_force, virtual, virtual_no_position]

    exact_ratio = float(exact["vertical_support_ratio_to_weight"])
    source_ratio = float(source_force["vertical_support_ratio_to_weight"])
    virtual_ratio = float(virtual["vertical_support_ratio_to_weight"])
    no_position_ratio = float(
        virtual_no_position["vertical_support_ratio_to_weight"]
    )
    source_tracking = float(
        source_force["tracking"]["max_normalized_mean_abs_error"]
    )
    virtual_tracking = float(
        virtual["tracking"]["max_normalized_mean_abs_error"]
    )

    if not 0.85 <= exact_ratio <= 1.20:
        diagnosis = "V3_EXACT_KINEMATICS_NOT_SOURCE_EQUIVALENT"
    elif source_ratio < 0.75 or source_tracking > 0.10:
        diagnosis = "V3_SOURCE_FORCE_DYNAMIC_TRACKING_MISMATCH"
    elif no_position_ratio > virtual_ratio + 0.10:
        diagnosis = "LEGACY_POSITION_ACTUATORS_INTERFERE_WITH_VIRTUAL_MUSCLES"
    elif virtual_ratio < 0.75 or virtual_tracking > 0.10:
        diagnosis = "VIRTUAL_MUSCLE_SEAM_LOSES_SOURCE_LIFT"
    else:
        diagnosis = "CLAMPED_V3_MUSCLE_SUPPORT_OK_FREE_FLIGHT_STABILITY_REMAINS"

    result = {
        "schema_version": 1,
        "flight_physics_version": "v3-source-equivalent",
        "source_root_pitch_deg": -47.5,
        "pattern": str(args.pattern),
        "warmup_wingbeats": args.warmup_wingbeats,
        "measure_wingbeats": args.measure_wingbeats,
        "cases": rows,
        "diagnosis": diagnosis,
        "interpretation": (
            "All dynamic cases use the same Flyppy-v3 source-equivalent body. "
            "The root is re-anchored only to remove free-flight attitude/fall as a "
            "confounder. No gain is calibrated by this diagnostic."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        line = (
            "v3_muscle_diagnosis mode={} total_vertical={:+.3f} "
            "total_magnitude={:.3f}"
        ).format(
            row["mode"],
            float(row["vertical_support_ratio_to_weight"]),
            float(row["mean_force_magnitude_ratio_to_weight"]),
        )
        tracking = row.get("tracking")
        if tracking is not None:
            line += " track_nmae_max={:.4f}".format(
                float(tracking["max_normalized_mean_abs_error"])
            )
        if "peak_abs_drive_force" in row:
            line += " peak_force={:.1f}".format(float(row["peak_abs_drive_force"]))
        if "peak_abs_virtual_muscle_torque" in row:
            line += " peak_torque={:.1f}".format(
                float(row["peak_abs_virtual_muscle_torque"])
            )
        print(line)

    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
