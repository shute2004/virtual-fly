#!/usr/bin/env python3
"""Reproduce the FlyBody reference lift-calibration seam as closely as possible.

This diagnostic is deliberately below the CNS and below the virtual-muscle layer.
It drives the six wing joints with the official measured baseline wing-beat pattern
and compares three control seams:

* ``position_physics_rate``: historical virtual-fly POSITION shortcut updated every
  50 us physics step.
* ``position_source_rate``: the same POSITION shortcut, but targets are generated
  and held at FlyBody's 0.2 ms flight-control cadence.
* ``source_force_rate``: POSITION actuators are disabled and the upstream FlyBody
  force command is reproduced directly.  Every 0.2 ms the controller computes
  ``gain * clip(target_angle - current_angle, -1, 1)`` once, then holds that force
  unchanged across the four 50 us physics steps in the control interval.

The source-style cases use the upstream WPG resampling cadence and therefore isolate
whether the remaining mismatch lives in FlyGym actuator translation or deeper in
wing inertia / geometry / fluid mechanics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from flygym.compose import ActuatorType

from flybody_adapter import FlyBodyWingAdapter, WingDrive
from flybody_biophysics import normalize_fly_mass
from flybody_flight_physics import FLIGHT_WING_POSITION_KP
from flybody_measured_wingbeat import (
    AXES,
    DEFAULT_PATTERN,
    SOURCE_CONTROL_TIMESTEP_S,
    MeasuredWingbeatCycle,
    SourceStyleWingbeatController,
    install_measured_wingbeat,
)


PUBLISHED_FLYBODY_MASS_MG = 0.983
SOURCE_GRAVITY_MM_S2 = 9810.0
SOURCE_ROOT_PITCH_DEG = 0.0
LEGACY_VIRTUAL_FLY_ROOT_PITCH_DEG = 47.5
SOURCE_WING_ERROR_CLIP_RAD = 1.0
LEG_PREFIXES = ("lf_", "lm_", "lh_", "rf_", "rm_", "rh_")
DRIVE_MODES = (
    "position_physics_rate",
    "position_source_rate",
    "source_force_rate",
)


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
        "--root-pitches-deg",
        type=float,
        nargs="+",
        default=(SOURCE_ROOT_PITCH_DEG, LEGACY_VIRTUAL_FLY_ROOT_PITCH_DEG),
    )
    parser.add_argument(
        "--drive-modes",
        nargs="+",
        choices=DRIVE_MODES,
        default=DRIVE_MODES,
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


def make_body(
    pattern: Path,
    *,
    aerodynamics: bool,
    root_pitch_deg: float,
) -> tuple[FlyBodyWingAdapter, MeasuredWingbeatCycle]:
    body = FlyBodyWingAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 8.25),
        flight_body_pitch_deg=float(root_pitch_deg),
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


def source_controller(body, cycle: MeasuredWingbeatCycle) -> SourceStyleWingbeatController:
    controller = SourceStyleWingbeatController(
        cycle,
        wingbeat_hz=body.wingbeat_hz,
        control_timestep_s=SOURCE_CONTROL_TIMESTEP_S,
    )
    ratio = SOURCE_CONTROL_TIMESTEP_S / body.timestep
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"source control timestep {SOURCE_CONTROL_TIMESTEP_S} is not an integer "
            f"multiple of physics timestep {body.timestep}"
        )
    return controller


def set_wing_state_from_vector(
    body,
    qpos_vector: np.ndarray,
    qvel_vector: np.ndarray,
) -> None:
    for side in ("left", "right"):
        for axis_index, axis in enumerate(AXES):
            actuator_index = body._wing_indices[(side, axis)]
            dof = body._actuated_dofs[actuator_index]
            qpos_address, qvel_address = body._wing_state_addresses(dof)
            body.sim.mj_data.qpos[qpos_address] = float(qpos_vector[axis_index])
            body.sim.mj_data.qvel[qvel_address] = float(qvel_vector[axis_index])


def position_target_vector(body, target_angles: np.ndarray) -> np.ndarray:
    target = body._neutral_target.copy()
    for side in ("left", "right"):
        for axis_index, axis in enumerate(AXES):
            target[body._wing_indices[(side, axis)]] = float(target_angles[axis_index])
    return target


def disable_position_wing_actuators(body) -> None:
    mapping = body.fly.jointdof_to_mjcfactuator_by_type[ActuatorType.POSITION]
    for dof in body._actuated_dofs:
        actuator = mapping[dof]
        actuator_id = mj.mj_name2id(
            body.sim.mj_model,
            mj.mjtObj.mjOBJ_ACTUATOR,
            actuator.name,
        )
        if actuator_id < 0:
            raise RuntimeError(f"compiled wing actuator not found: {actuator.name}")
        body.sim.mj_model.actuator_gainprm[actuator_id, :] = 0.0
        body.sim.mj_model.actuator_biasprm[actuator_id, :] = 0.0


def source_force_command(body, target_angles: np.ndarray) -> dict[int, float]:
    """Compute the upstream force command once at a control-step boundary."""

    command: dict[int, float] = {}
    for side in ("left", "right"):
        for axis_index, axis in enumerate(AXES):
            actuator_index = body._wing_indices[(side, axis)]
            dof = body._actuated_dofs[actuator_index]
            qpos_address, qvel_address = body._wing_state_addresses(dof)
            error = float(target_angles[axis_index]) - float(body.sim.mj_data.qpos[qpos_address])
            clipped_error = min(
                SOURCE_WING_ERROR_CLIP_RAD,
                max(-SOURCE_WING_ERROR_CLIP_RAD, error),
            )
            command[qvel_address] = FLIGHT_WING_POSITION_KP * clipped_error
    return command


def apply_held_source_force(body, command: dict[int, float]) -> float:
    """Apply one already-computed source force command for one physics step."""

    body.sim.mj_data.qfrc_applied[:] = 0.0
    peak = 0.0
    for qvel_address, torque in command.items():
        body.sim.mj_data.qfrc_applied[qvel_address] += torque
        peak = max(peak, abs(torque))
    body.sim.step()
    body.sim.mj_data.qfrc_applied[:] = 0.0
    body._time += body.timestep
    return peak


def current_angles(body) -> dict[str, float]:
    return body.wing_joint_angles_rad()


def tracking_target_dict(vector: np.ndarray) -> dict[str, float]:
    return {axis: float(vector[i]) for i, axis in enumerate(AXES)}


def run_clamped_case(
    pattern: Path,
    *,
    aerodynamics: bool,
    leg_pose: str,
    root_pitch_deg: float,
    drive_mode: str,
    warmup_wingbeats: float,
    measure_wingbeats: float,
) -> dict[str, object]:
    body, cycle = make_body(
        pattern,
        aerodynamics=aerodynamics,
        root_pitch_deg=root_pitch_deg,
    )
    spring_state = leg_springref_state(body)

    controller: SourceStyleWingbeatController | None = None
    control_stride = 1
    held_target: np.ndarray | None = None
    held_source_force: dict[int, float] = {}
    if drive_mode in {"position_source_rate", "source_force_rate"}:
        controller = source_controller(body, cycle)
        initial_qpos, initial_qvel = controller.reset(initial_phase=0.0)
        set_wing_state_from_vector(body, initial_qpos, initial_qvel)
        held_target = controller.current()
        control_stride = int(round(SOURCE_CONTROL_TIMESTEP_S / body.timestep))
        body._time = 0.0
        if drive_mode == "position_source_rate":
            body.sim.set_actuator_inputs(
                body.fly.name,
                ActuatorType.POSITION,
                position_target_vector(body, held_target),
            )
        else:
            disable_position_wing_actuators(body)

    enforce_leg_pose(body, leg_pose, spring_state)
    mj.mj_forward(body.sim.mj_model, body.sim.mj_data)
    root_qpos, root_qpos_address, root_qvel_address = root_reference(body)

    wing_period_s = 1.0 / body.wingbeat_hz
    total_s = (warmup_wingbeats + measure_wingbeats) * wing_period_s
    warmup_s = warmup_wingbeats * wing_period_s
    steps = max(1, math.ceil(total_s / body.timestep))

    errors: dict[str, list[float]] = {axis: [] for axis in AXES}
    net_vertical_accels: list[float] = []
    peak_abs_drive_force = 0.0

    for physics_index in range(steps):
        reset_root_kinematics(body, root_qpos, root_qpos_address, root_qvel_address)
        enforce_leg_pose(body, leg_pose, spring_state)
        mj.mj_forward(body.sim.mj_model, body.sim.mj_data)

        root_speed_before_step = float(body.sim.mj_data.qvel[root_qvel_address + 2])

        if drive_mode == "position_physics_rate":
            body.step(WingDrive(0.0, 0.0), physics_steps=1)
            phase_after = (2.0 * math.pi * body.wingbeat_hz * body._time) % (2.0 * math.pi)
            target_dict = cycle.angles(phase_after, 1.0)
            actuator_force = np.asarray(body.sim.mj_data.actuator_force, dtype=np.float64)
            if actuator_force.size:
                peak_abs_drive_force = max(
                    peak_abs_drive_force,
                    float(np.max(np.abs(actuator_force))),
                )
        else:
            assert controller is not None and held_target is not None
            if physics_index % control_stride == 0:
                held_target = controller.step()
                if drive_mode == "position_source_rate":
                    body.sim.set_actuator_inputs(
                        body.fly.name,
                        ActuatorType.POSITION,
                        position_target_vector(body, held_target),
                    )
                else:
                    held_source_force = source_force_command(body, held_target)

            if drive_mode == "position_source_rate":
                body.sim.step()
                body._time += body.timestep
                actuator_force = np.asarray(body.sim.mj_data.actuator_force, dtype=np.float64)
                if actuator_force.size:
                    peak_abs_drive_force = max(
                        peak_abs_drive_force,
                        float(np.max(np.abs(actuator_force))),
                    )
            elif drive_mode == "source_force_rate":
                peak_abs_drive_force = max(
                    peak_abs_drive_force,
                    apply_held_source_force(body, held_source_force),
                )
            else:
                raise KeyError(drive_mode)
            target_dict = tracking_target_dict(held_target)

        root_speed_after_step = float(body.sim.mj_data.qvel[root_qvel_address + 2])
        if body._time < warmup_s:
            continue

        net_vertical_accels.append(
            (root_speed_after_step - root_speed_before_step) / body.timestep
        )
        actual = current_angles(body)
        for axis in AXES:
            errors[axis].append(abs(actual[f"left_{axis}"] - target_dict[axis]))
            errors[axis].append(abs(actual[f"right_{axis}"] - target_dict[axis]))

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
        "root_pitch_deg": float(root_pitch_deg),
        "drive_mode": drive_mode,
        "body_mass_mg": PUBLISHED_FLYBODY_MASS_MG,
        "wingbeat_hz": body.wingbeat_hz,
        "physics_timestep_s": body.timestep,
        "source_control_timestep_s": SOURCE_CONTROL_TIMESTEP_S,
        "pattern_samples": cycle.samples,
        "warmup_wingbeats": warmup_wingbeats,
        "measure_wingbeats": measure_wingbeats,
        "mean_net_vertical_acceleration_mm_s2": float(np.mean(net_vertical_accels)),
        "std_net_vertical_acceleration_mm_s2": float(np.std(net_vertical_accels)),
        "peak_abs_drive_force": peak_abs_drive_force,
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
    if not args.root_pitches_deg or any(not math.isfinite(v) for v in args.root_pitches_deg):
        raise SystemExit("root-pitches-deg must contain finite values")

    cases: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for root_pitch_deg in args.root_pitches_deg:
        for leg_pose in args.leg_poses:
            for drive_mode in args.drive_modes:
                off = run_clamped_case(
                    args.pattern,
                    aerodynamics=False,
                    leg_pose=leg_pose,
                    root_pitch_deg=float(root_pitch_deg),
                    drive_mode=drive_mode,
                    warmup_wingbeats=args.warmup_wingbeats,
                    measure_wingbeats=args.measure_wingbeats,
                )
                on = run_clamped_case(
                    args.pattern,
                    aerodynamics=True,
                    leg_pose=leg_pose,
                    root_pitch_deg=float(root_pitch_deg),
                    drive_mode=drive_mode,
                    warmup_wingbeats=args.warmup_wingbeats,
                    measure_wingbeats=args.measure_wingbeats,
                )
                cases.extend((off, on))

                off_accel = float(off["mean_net_vertical_acceleration_mm_s2"])
                on_accel = float(on["mean_net_vertical_acceleration_mm_s2"])
                aerodynamic_support = on_accel - off_accel
                support_ratio = aerodynamic_support / SOURCE_GRAVITY_MM_S2
                tracking_by_axis = {
                    axis: float(on["tracking"][axis]["normalized_mean_abs_error"])
                    for axis in AXES
                }
                comparisons.append(
                    {
                        "root_pitch_deg": float(root_pitch_deg),
                        "leg_pose": leg_pose,
                        "drive_mode": drive_mode,
                        "no_aero_mean_net_accel_mm_s2": off_accel,
                        "aero_mean_net_accel_mm_s2": on_accel,
                        "aerodynamic_support_accel_mm_s2": aerodynamic_support,
                        "support_ratio_to_weight": support_ratio,
                        "tracking_nmae_by_axis": tracking_by_axis,
                        "max_normalized_tracking_mae": max(tracking_by_axis.values()),
                        "peak_abs_drive_force": float(on["peak_abs_drive_force"]),
                        "source_tracking_target_lt": 0.05,
                    }
                )

    source_cases = [
        row
        for row in comparisons
        if math.isclose(float(row["root_pitch_deg"]), SOURCE_ROOT_PITCH_DEG, abs_tol=1e-9)
        and row["leg_pose"] == "source_retracted"
        and row["drive_mode"] == "source_force_rate"
    ]
    if not source_cases:
        source_cases = comparisons
    best = max(source_cases, key=lambda row: float(row["support_ratio_to_weight"]))
    if float(best["max_normalized_tracking_mae"]) >= 0.05:
        diagnosis = "WING_INERTIA_JOINT_OR_FRAME_PORT_MISMATCH"
    elif float(best["support_ratio_to_weight"]) < 0.70:
        diagnosis = "AERODYNAMIC_PORT_UNDERPRODUCES_REFERENCE_LIFT"
    elif float(best["support_ratio_to_weight"]) > 1.30:
        diagnosis = "AERODYNAMIC_PORT_OVERPRODUCES_REFERENCE_LIFT"
    else:
        diagnosis = "REFERENCE_LIFT_PORT_APPROXIMATELY_CONSISTENT"

    result = {
        "schema_version": 4,
        "pattern": str(args.pattern),
        "published_flybody_mass_mg": PUBLISHED_FLYBODY_MASS_MG,
        "source_gravity_mm_s2": SOURCE_GRAVITY_MM_S2,
        "source_equivalent_root_pitch_deg": SOURCE_ROOT_PITCH_DEG,
        "legacy_virtual_fly_root_pitch_deg": LEGACY_VIRTUAL_FLY_ROOT_PITCH_DEG,
        "source_control_timestep_s": SOURCE_CONTROL_TIMESTEP_S,
        "source_wing_error_clip_rad": SOURCE_WING_ERROR_CLIP_RAD,
        "cases": cases,
        "comparisons": comparisons,
        "diagnosis": diagnosis,
        "interpretation": (
            "Calibration-only comparison of the historical virtual-fly POSITION seam, "
            "a source-rate POSITION seam, and the upstream source-style held-force seam. "
            "Stable free hover is deliberately not required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for row in comparisons:
        track = row["tracking_nmae_by_axis"]
        print(
            "reference_lift root_pitch={:.1f} leg_pose={} drive={} support_ratio={:.3f} "
            "track_nmae=yaw:{:.4f},roll:{:.4f},pitch:{:.4f} max={:.4f} peak_force={:.1f}".format(
                float(row["root_pitch_deg"]),
                row["leg_pose"],
                row["drive_mode"],
                float(row["support_ratio_to_weight"]),
                float(track["yaw"]),
                float(track["roll"]),
                float(track["pitch"]),
                float(row["max_normalized_tracking_mae"]),
                float(row["peak_abs_drive_force"]),
            )
        )
    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
