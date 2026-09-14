#!/usr/bin/env python3
"""Validate Flyppy-v3 physical scale, joint ranges, and gate clearance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from virtual_fly.physics import FLYBODY_V3, FLYPPY_GEOMETRY_V3

from flybody_biophysics import HALTERE_HALF_RANGE_RAD, HEAD_YAW_HALF_RANGE_RAD
from flybody_physical_spec_smoke import conservative_geom_envelope
from flybody_v3_adapter import FlyBodyV3MuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-physical-spec-v3.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 12:
        raise SystemExit("samples must be >= 12")

    course = FlyppyCourse(seed=0, gate_count=2, environment_version="v3")
    world = FlyppyWorld(course)
    body = FlyBodyV3MuscleAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, FLYPPY_GEOMETRY_V3.corridor_high_z_mm / 2.0),
        enable_vision=False,
    )

    if body.mass_normalization is None:
        raise RuntimeError("Flyppy v3 FlyBody mass normalization was disabled")
    normalized_mass_mg = body.mass_normalization.target_mass_g * 1000.0
    if not math.isclose(normalized_mass_mg, FLYBODY_V3.total_mass_mg, rel_tol=2e-6):
        raise RuntimeError(
            f"v3 mass mismatch: expected {FLYBODY_V3.total_mass_mg}, got {normalized_mass_mg}"
        )
    if not bool(getattr(body, "source_equivalent_wing_frame", False)):
        raise RuntimeError("v3 source-equivalent wing frame is not active")
    if not math.isclose(float(body.flight_body_pitch_deg), -47.5, abs_tol=1e-12):
        raise RuntimeError(f"v3 source root pitch mismatch: {body.flight_body_pitch_deg}")

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
            "v3 gate is smaller than conservative swept FlyBody vertical envelope: "
            f"gap={gate_gap:.6f} envelope={extent[2]:.6f}"
        )
    if lateral_clearance <= 0.0:
        raise RuntimeError(
            "v3 lateral opening is smaller than conservative swept FlyBody envelope: "
            f"opening={lateral_opening:.6f} envelope={extent[1]:.6f}"
        )

    result = {
        "schema_version": 1,
        "flight_physics_version": "v3-source-equivalent",
        "physical_fly": {
            "morphology": FLYBODY_V3.morphology,
            "body_length_mm": FLYBODY_V3.body_length_mm,
            "wing_length_mm": FLYBODY_V3.wing_length_mm,
            "total_mass_mg": FLYBODY_V3.total_mass_mg,
            "morphology_sex": FLYBODY_V3.morphology_sex,
            "neural_sex": FLYBODY_V3.neural_sex,
            "root_pitch_deg": -47.5,
            "source_equivalent_wing_frame": True,
        },
        "mass_normalization": body.mass_normalization.as_dict(),
        "joint_range_overrides": body.joint_range_overrides,
        "environment": {
            "version": "v3",
            "corridor_height_mm": course.ceiling_z_mm - course.floor_z_mm,
            "gate_gap_height_mm": gate_gap,
            "first_gate_x_mm": course.gates[0].x_mm,
            "gate_spacing_mm": course.gates[1].x_mm - course.gates[0].x_mm,
            "lateral_opening_mm": lateral_opening,
            "gate_full_thickness_mm": 2.0 * course.gates[0].half_thickness_mm,
        },
        "swept_envelope": {
            "method": "conservative MuJoCo geom bounding spheres over official measured wing cycle plus 180-degree haltere stroke; neutral non-flight joints",
            "samples": args.samples,
            "extent_x_mm": float(extent[0]),
            "extent_y_mm": float(extent[1]),
            "extent_z_mm": float(extent[2]),
            "vertical_gate_clearance_mm": vertical_clearance,
            "lateral_gate_clearance_mm": lateral_clearance,
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"v3_body_length_mm={FLYBODY_V3.body_length_mm:.6f}")
    print(f"v3_mass_mg={FLYBODY_V3.total_mass_mg:.6f}")
    print(f"v3_swept_envelope_xyz_mm={extent.tolist()}")
    print(f"v3_gate_gap_height_mm={gate_gap:.6f}")
    print(f"v3_vertical_gate_clearance_mm={vertical_clearance:.6f}")
    print(f"v3_lateral_gate_clearance_mm={lateral_clearance:.6f}")
    print(f"result={args.output}")
    print("flybody_physical_spec_v3=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
