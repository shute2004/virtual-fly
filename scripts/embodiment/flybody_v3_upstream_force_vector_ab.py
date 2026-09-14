#!/usr/bin/env python3
"""Compare Flyppy-v3 fluid-force vectors against upstream FlyBody exactly.

This diagnostic addresses a remaining source-equivalence gap after vertical support
was restored.  Under the official measured wing kinematics and the source -47.5 deg
root pose, upstream FlyBody produces about one body-weight of total fluid force.
Flyppy-v3 now reproduces the vertical component, but its total force magnitude can
still differ.  This script compares three force components in the same normalized
(body-weight) coordinates:

* baseline_inertia_fluid: ordinary MuJoCo fluid force with the explicit wing
  ellipsoid geoms disabled;
* incremental_wing_ellipsoid: the additional force caused by enabling the source
  wing ellipsoid geoms;
* active_total_fluid: the sum actually present in the flight model.

Both models receive the same measured wing qpos/qvel at the same phases.  CNS,
virtual muscles, actuators, reward, task observations, and free-flight attitude are
all bypassed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from flybody_measured_wingbeat import AXES, DEFAULT_PATTERN, MeasuredWingbeatCycle
from flybody_upstream_source_aero_reference import SOURCE_GRAVITY_CM_S2, UPSTREAM_XML
from flybody_upstream_source_orientation_sweep import run_pitch as run_upstream_pitch
from flybody_v3_adapter import FlyBodyV3MuscleAdapter


SOURCE_ROOT_PITCH_DEG = -47.5
GRAVITY_MM_S2 = SOURCE_GRAVITY_CM_S2 * 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--phase-samples", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-v3-upstream-force-vector-ab.json"),
    )
    return parser.parse_args()


def make_v3(*, aerodynamics: bool) -> FlyBodyV3MuscleAdapter:
    return FlyBodyV3MuscleAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 8.91),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=aerodynamics,
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


def set_v3_exact_state(
    body: FlyBodyV3MuscleAdapter,
    cycle: MeasuredWingbeatCycle,
    phase: float,
    root_qpos: np.ndarray,
    root_qpos_address: int,
    root_qvel_address: int,
) -> None:
    body.sim.mj_data.qpos[root_qpos_address : root_qpos_address + 7] = root_qpos
    body.sim.mj_data.qvel[root_qvel_address : root_qvel_address + 6] = 0.0

    angles = cycle.angles(phase, 1.0)
    velocities = cycle.velocities(phase, 1.0, body.wingbeat_hz)
    for side in ("left", "right"):
        for axis in AXES:
            actuator_index = body._wing_indices[(side, axis)]
            dof = body._actuated_dofs[actuator_index]
            qpos_address, qvel_address = body._wing_state_addresses(dof)
            body.sim.mj_data.qpos[qpos_address] = angles[axis]
            body.sim.mj_data.qvel[qvel_address] = velocities[axis]

    body.sim.mj_data.ctrl[:] = 0.0
    body.sim.mj_data.qfrc_applied[:] = 0.0
    mj.mj_forward(body.sim.mj_model, body.sim.mj_data)


def normalized_summary(force: np.ndarray, weight: float) -> dict[str, object]:
    mean = np.mean(force, axis=0)
    vector_bw = mean / weight
    magnitude = float(np.linalg.norm(vector_bw))
    direction = vector_bw / max(magnitude, 1e-30)
    return {
        "mean_force_xyz_to_weight": [float(v) for v in vector_bw],
        "force_magnitude_to_weight": magnitude,
        "vertical_support_ratio_to_weight": float(vector_bw[2]),
        "mean_force_direction": [float(v) for v in direction],
    }


def run_v3(cycle: MeasuredWingbeatCycle, phase_samples: int) -> dict[str, object]:
    off = make_v3(aerodynamics=False)
    on = make_v3(aerodynamics=True)
    mass_off = fly_mass_g(off)
    mass_on = fly_mass_g(on)
    if not math.isclose(mass_off, mass_on, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError(f"v3 aero A/B mass mismatch: off={mass_off} on={mass_on}")

    off_root_qpos_address = off._root_freejoint_qpos_address()
    off_root_qvel_address = off._root_freejoint_dof_address()
    on_root_qpos_address = on._root_freejoint_qpos_address()
    on_root_qvel_address = on._root_freejoint_dof_address()
    off_root_qpos = np.asarray(
        off.sim.mj_data.qpos[off_root_qpos_address : off_root_qpos_address + 7],
        dtype=np.float64,
    ).copy()
    on_root_qpos = np.asarray(
        on.sim.mj_data.qpos[on_root_qpos_address : on_root_qpos_address + 7],
        dtype=np.float64,
    ).copy()

    off_force: list[np.ndarray] = []
    on_force: list[np.ndarray] = []
    for i in range(phase_samples):
        phase = 2.0 * math.pi * i / phase_samples
        set_v3_exact_state(
            off,
            cycle,
            phase,
            off_root_qpos,
            off_root_qpos_address,
            off_root_qvel_address,
        )
        set_v3_exact_state(
            on,
            cycle,
            phase,
            on_root_qpos,
            on_root_qpos_address,
            on_root_qvel_address,
        )
        off_force.append(
            np.asarray(
                off.sim.mj_data.qfrc_fluid[
                    off_root_qvel_address : off_root_qvel_address + 3
                ],
                dtype=np.float64,
            ).copy()
        )
        on_force.append(
            np.asarray(
                on.sim.mj_data.qfrc_fluid[
                    on_root_qvel_address : on_root_qvel_address + 3
                ],
                dtype=np.float64,
            ).copy()
        )

    off_array = np.asarray(off_force, dtype=np.float64)
    on_array = np.asarray(on_force, dtype=np.float64)
    delta = on_array - off_array
    weight = mass_on * GRAVITY_MM_S2
    return {
        "mass_mg": mass_on * 1000.0,
        "baseline_inertia_fluid": normalized_summary(off_array, weight),
        "incremental_wing_ellipsoid": normalized_summary(delta, weight),
        "active_total_fluid": normalized_summary(on_array, weight),
    }


def upstream_normalized(row: dict[str, object]) -> dict[str, object]:
    mass_g = float(row["mass_mg"]) / 1000.0
    weight = mass_g * SOURCE_GRAVITY_CM_S2
    result: dict[str, object] = {"mass_mg": float(row["mass_mg"])}
    for key in (
        "baseline_inertia_fluid",
        "incremental_wing_ellipsoid",
        "active_total_fluid",
    ):
        source = row[key]
        vector = np.asarray(source["mean_force_g_cm_s2"], dtype=np.float64) / weight
        result[key] = {
            "mean_force_xyz_to_weight": [float(v) for v in vector],
            "force_magnitude_to_weight": float(np.linalg.norm(vector)),
            "vertical_support_ratio_to_weight": float(vector[2]),
            "mean_force_direction": [
                float(v) for v in np.asarray(source["mean_force_direction"], dtype=np.float64)
            ],
        }
    return result


def vector_error(source: dict[str, object], v3: dict[str, object]) -> dict[str, object]:
    source_vec = np.asarray(source["mean_force_xyz_to_weight"], dtype=np.float64)
    v3_vec = np.asarray(v3["mean_force_xyz_to_weight"], dtype=np.float64)
    delta = v3_vec - source_vec
    return {
        "delta_xyz_to_weight": [float(v) for v in delta],
        "l2_error_to_weight": float(np.linalg.norm(delta)),
        "source_xyz_to_weight": [float(v) for v in source_vec],
        "v3_xyz_to_weight": [float(v) for v in v3_vec],
    }


def main() -> int:
    args = parse_args()
    if not args.pattern.exists():
        raise SystemExit(f"measured wing pattern missing: {args.pattern}")
    if not UPSTREAM_XML.exists():
        raise SystemExit(
            "upstream FlyBody source missing; run "
            "uv run python scripts/dev/prefetch_upstream_flybody_source.py"
        )
    if args.phase_samples < 100:
        raise SystemExit("phase-samples must be >= 100")

    cycle = MeasuredWingbeatCycle(args.pattern)
    upstream_raw = run_upstream_pitch(cycle, SOURCE_ROOT_PITCH_DEG, args.phase_samples)
    upstream = upstream_normalized(upstream_raw)
    v3 = run_v3(cycle, args.phase_samples)

    errors = {
        key: vector_error(upstream[key], v3[key])
        for key in (
            "baseline_inertia_fluid",
            "incremental_wing_ellipsoid",
            "active_total_fluid",
        )
    }
    baseline_error = float(errors["baseline_inertia_fluid"]["l2_error_to_weight"])
    wing_error = float(errors["incremental_wing_ellipsoid"]["l2_error_to_weight"])
    total_error = float(errors["active_total_fluid"]["l2_error_to_weight"])

    if total_error <= 0.10:
        diagnosis = "V3_TOTAL_FLUID_VECTOR_APPROXIMATELY_SOURCE_EQUIVALENT"
    elif baseline_error > max(0.10, wing_error * 1.5):
        diagnosis = "V3_BASELINE_BODY_FLUID_VECTOR_MISMATCH"
    elif wing_error > max(0.10, baseline_error * 1.5):
        diagnosis = "V3_WING_ELLIPSOID_VECTOR_MISMATCH"
    else:
        diagnosis = "V3_BASELINE_AND_WING_FLUID_VECTORS_BOTH_DIFFER"

    result = {
        "schema_version": 1,
        "root_pitch_deg": SOURCE_ROOT_PITCH_DEG,
        "phase_samples": args.phase_samples,
        "pattern": str(args.pattern),
        "upstream": upstream,
        "v3": v3,
        "errors": errors,
        "diagnosis": diagnosis,
        "interpretation": (
            "Normalized 3D root translational qfrc_fluid vectors under identical exact "
            "measured wing kinematics. Separating baseline and explicit wing-ellipsoid "
            "terms identifies which physical port produces any residual horizontal-force gap."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for key in (
        "baseline_inertia_fluid",
        "incremental_wing_ellipsoid",
        "active_total_fluid",
    ):
        row = errors[key]
        source_vec = row["source_xyz_to_weight"]
        v3_vec = row["v3_xyz_to_weight"]
        print(
            "v3_upstream_force component={} source=[{:+.3f},{:+.3f},{:+.3f}] "
            "v3=[{:+.3f},{:+.3f},{:+.3f}] error={:.3f}".format(
                key,
                *[float(v) for v in source_vec],
                *[float(v) for v in v3_vec],
                float(row["l2_error_to_weight"]),
            )
        )
    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
