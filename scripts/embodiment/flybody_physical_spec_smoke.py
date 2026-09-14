#!/usr/bin/env python3
"""Validate the version-2 physical fly and environment scale in MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from virtual_fly.physics import CANONICAL_FLY, FLYPPY_GEOMETRY

from flybody_biophysics import HALTERE_HALF_RANGE_RAD, HEAD_YAW_HALF_RANGE_RAD
from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-physical-spec-v2.json"),
    )
    return parser.parse_args()


def fly_body_ids(body: FlyBodyMuscleAdapter) -> set[int]:
    result: set[int] = set()
    for segment in body.fly.get_bodysegs_order():
        name = body.fly.bodyseg_to_mjcfbody[segment].name
        body_id = mj.mj_name2id(body.sim.mj_model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"compiled body not found: {name}")
        result.add(int(body_id))
    return result


def conservative_geom_envelope(body: FlyBodyMuscleAdapter) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space lower/upper bounds using each geom bounding sphere."""

    model = body.sim.mj_model
    data = body.sim.mj_data
    body_ids = fly_body_ids(body)
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    rbound = np.asarray(model.geom_rbound, dtype=np.float64)
    for geom_id, owner in enumerate(np.asarray(model.geom_bodyid, dtype=np.int32)):
        if int(owner) not in body_ids:
            continue
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        radius = float(rbound[geom_id])
        if not math.isfinite(radius) or radius < 0.0:
            raise RuntimeError(f"invalid geom bounding radius at {geom_id}: {radius}")
        lower = np.minimum(lower, center - radius)
        upper = np.maximum(upper, center + radius)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise RuntimeError("failed to measure FlyBody geometry envelope")
    return lower, upper


def main() -> int:
    args = parse_args()
    if args.samples < 12:
        raise SystemExit("samples must be >= 12")

    course = FlyppyCourse(seed=0, gate_count=2, environment_version="v2")
    world = FlyppyWorld(course)
    body = FlyBodyMuscleAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, FLYPPY_GEOMETRY.corridor_high_z_mm / 2.0),
        enable_vision=False,
    )

    if body.mass_normalization is None:
        raise RuntimeError("canonical FlyBody mass normalization was disabled")
    normalized_mass_mg = body.mass_normalization.target_mass_g * 1000.0
    if not math.isclose(normalized_mass_mg, CANONICAL_FLY.total_mass_mg, rel_tol=2e-6):
        raise RuntimeError(
            f"canonical mass mismatch: expected {CANONICAL_FLY.total_mass_mg}, got {normalized_mass_mg}"
        )

    ranges = body.compiled_biological_joint_ranges
    for key in ("c_thorax-l_haltere-pitch", "c_thorax-r_haltere-pitch"):
        low, high = ranges[key]
        if not (
            math.isclose(low, -HALTERE_HALF_RANGE_RAD, abs_tol=1e-6)
            and math.isclose(high, HALTERE_HALF_RANGE_RAD, abs_tol=1e-6)
        ):
            raise RuntimeError(f"haltere range override missing for {key}: {(low, high)}")
    head_low, head_high = ranges["c_thorax-c_head-yaw"]
    if not (
        math.isclose(head_low, -HEAD_YAW_HALF_RANGE_RAD, abs_tol=1e-6)
        and math.isclose(head_high, HEAD_YAW_HALF_RANGE_RAD, abs_tol=1e-6)
    ):
        raise RuntimeError(f"head yaw range override missing: {(head_low, head_high)}")

    global_lower = np.full(3, np.inf, dtype=np.float64)
    global_upper = np.full(3, -np.inf, dtype=np.float64)
    for sample in range(args.samples):
        phase = 2.0 * math.pi * sample / args.samples
        wing = body._wing_pattern(phase, 1.0)
        for side in ("left", "right"):
            for axis in ("yaw", "roll", "pitch"):
                actuator_index = body._wing_indices[(side, axis)]
                dof = body._actuated_dofs[actuator_index]
                qpos_address, _ = body._wing_state_addresses(dof)
                body.sim.mj_data.qpos[qpos_address] = wing[axis]
        # Include a naturalistic ~180 degree haltere stroke in the envelope.
        haltere_angle = math.radians(90.0) * math.sin(phase + math.pi)
        for prefix in ("l", "r"):
            qpos_address, _ = body.joint_state_addresses(
                f"c_thorax-{prefix}_haltere-pitch"
            )
            body.sim.mj_data.qpos[qpos_address] = haltere_angle
        mj.mj_forward(body.sim.mj_model, body.sim.mj_data)
        lower, upper = conservative_geom_envelope(body)
        global_lower = np.minimum(global_lower, lower)
        global_upper = np.maximum(global_upper, upper)

    extent = global_upper - global_lower
    gate_gap = float(2.0 * course.gates[0].half_gap_mm)
    vertical_clearance = gate_gap - float(extent[2])
    lateral_opening = 2.0 * course.lateral_half_width_mm
    lateral_clearance = lateral_opening - float(extent[1])
    if vertical_clearance <= 0.0:
        raise RuntimeError(
            "v2 gate is smaller than conservative swept FlyBody vertical envelope: "
            f"gap={gate_gap:.6f} envelope={extent[2]:.6f}"
        )
    if lateral_clearance <= 0.0:
        raise RuntimeError(
            "v2 lateral opening is smaller than conservative swept FlyBody envelope: "
            f"opening={lateral_opening:.6f} envelope={extent[1]:.6f}"
        )

    result = {
        "schema_version": 1,
        "canonical_fly": {
            "morphology": CANONICAL_FLY.morphology,
            "body_length_mm": CANONICAL_FLY.body_length_mm,
            "wing_length_mm": CANONICAL_FLY.wing_length_mm,
            "total_mass_mg": CANONICAL_FLY.total_mass_mg,
            "morphology_sex": CANONICAL_FLY.morphology_sex,
            "neural_sex": CANONICAL_FLY.neural_sex,
        },
        "mass_normalization": body.mass_normalization.as_dict(),
        "joint_range_overrides": body.joint_range_overrides,
        "environment": {
            "version": "v2",
            "corridor_height_mm": course.ceiling_z_mm - course.floor_z_mm,
            "gate_gap_height_mm": gate_gap,
            "first_gate_x_mm": course.gates[0].x_mm,
            "gate_spacing_mm": course.gates[1].x_mm - course.gates[0].x_mm,
            "lateral_opening_mm": lateral_opening,
            "gate_full_thickness_mm": 2.0 * course.gates[0].half_thickness_mm,
        },
        "swept_envelope": {
            "method": "conservative MuJoCo geom bounding spheres over one wing/haltere cycle; neutral non-flight joints",
            "samples": args.samples,
            "extent_x_mm": float(extent[0]),
            "extent_y_mm": float(extent[1]),
            "extent_z_mm": float(extent[2]),
            "vertical_gate_clearance_mm": vertical_clearance,
            "lateral_gate_clearance_mm": lateral_clearance,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"canonical_body_length_mm={CANONICAL_FLY.body_length_mm:.6f}")
    print(f"canonical_mass_mg={CANONICAL_FLY.total_mass_mg:.6f}")
    print(f"swept_envelope_xyz_mm={extent.tolist()}")
    print(f"gate_gap_height_mm={gate_gap:.6f}")
    print(f"vertical_gate_clearance_mm={vertical_clearance:.6f}")
    print(f"result={args.output}")
    print("flybody_physical_spec_v2=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
