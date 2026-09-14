#!/usr/bin/env python3
"""Sweep upstream FlyBody root pitch under exact measured wing kinematics.

Calibration-only diagnostic.  This compiles TuragaLab/flybody's original
``fruitfly.xml`` directly with the currently installed MuJoCo, enables only the
source wing fluid geoms, imposes the official measured wing qpos/qvel exactly,
and evaluates the resulting fluid generalized force for several root pitches.

The purpose is to distinguish a genuinely weak aerodynamic force from a force
whose world-axis projection is wrong because the source flight pose was not
reproduced.  FlyGym and virtual-fly body conversion are not involved.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from flybody_measured_wingbeat import DEFAULT_PATTERN, MeasuredWingbeatCycle
from flybody_upstream_source_aero_reference import (
    ACTIVE_XML,
    SOURCE_GRAVITY_CM_S2,
    SOURCE_WINGBEAT_HZ,
    UPSTREAM_XML,
    compile_model,
    freejoint_addresses,
    joint_addresses,
    total_fly_mass_g,
    write_active_xml,
)


def quat_y_deg(deg: float) -> np.ndarray:
    half = math.radians(deg) * 0.5
    return np.asarray((math.cos(half), 0.0, math.sin(half), 0.0), dtype=np.float64)


def set_state(
    model: mj.MjModel,
    data: mj.MjData,
    cycle: MeasuredWingbeatCycle,
    phase: float,
    root_pitch_deg: float,
    root_qpos: int,
    root_dof: int,
    wing_addresses: dict[tuple[str, str], tuple[int, int]],
) -> None:
    data.qpos[root_qpos : root_qpos + 3] = 0.0
    data.qpos[root_qpos + 3 : root_qpos + 7] = quat_y_deg(root_pitch_deg)
    data.qvel[root_dof : root_dof + 6] = 0.0

    angles = cycle.angles(phase, 1.0)
    velocities = cycle.velocities(phase, 1.0, SOURCE_WINGBEAT_HZ)
    for side in ("left", "right"):
        for axis in ("yaw", "roll", "pitch"):
            qa, da = wing_addresses[(side, axis)]
            data.qpos[qa] = angles[axis]
            data.qvel[da] = velocities[axis]

    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mj.mj_forward(model, data)


def run_pitch(
    cycle: MeasuredWingbeatCycle,
    root_pitch_deg: float,
    phase_samples: int,
) -> dict[str, object]:
    write_active_xml(UPSTREAM_XML, ACTIVE_XML)
    off_model, off_data = compile_model(UPSTREAM_XML)
    on_model, on_data = compile_model(ACTIVE_XML)
    off_qpos, off_dof = freejoint_addresses(off_model)
    on_qpos, on_dof = freejoint_addresses(on_model)
    off_wings = joint_addresses(off_model)
    on_wings = joint_addresses(on_model)

    mass = total_fly_mass_g(on_model)
    if not math.isclose(mass, total_fly_mass_g(off_model), rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError("aero A/B mass mismatch")

    passive_delta: list[np.ndarray] = []
    fluid_delta: list[np.ndarray] = []
    for i in range(phase_samples):
        phase = 2.0 * math.pi * i / phase_samples
        set_state(
            off_model,
            off_data,
            cycle,
            phase,
            root_pitch_deg,
            off_qpos,
            off_dof,
            off_wings,
        )
        set_state(
            on_model,
            on_data,
            cycle,
            phase,
            root_pitch_deg,
            on_qpos,
            on_dof,
            on_wings,
        )
        passive_delta.append(
            np.asarray(on_data.qfrc_passive[on_dof : on_dof + 3], dtype=np.float64)
            - np.asarray(off_data.qfrc_passive[off_dof : off_dof + 3], dtype=np.float64)
        )
        # qfrc_fluid is non-zero in the source/off model as well because the world
        # already has non-zero density/viscosity and MuJoCo can apply the ordinary
        # inertia-based fluid model.  The isolated contribution of the explicitly
        # enabled wing ellipsoid geoms is therefore aero-on minus aero-off here too.
        fluid_delta.append(
            np.asarray(on_data.qfrc_fluid[on_dof : on_dof + 3], dtype=np.float64)
            - np.asarray(off_data.qfrc_fluid[off_dof : off_dof + 3], dtype=np.float64)
        )

    passive = np.asarray(passive_delta, dtype=np.float64)
    fluid = np.asarray(fluid_delta, dtype=np.float64)
    mean_passive = np.mean(passive, axis=0)
    mean_fluid = np.mean(fluid, axis=0)
    if not np.allclose(mean_passive, mean_fluid, rtol=1e-8, atol=1e-10):
        raise RuntimeError(
            "aero-on/off qfrc_passive delta disagrees with aero-on/off qfrc_fluid delta: "
            f"passive={mean_passive.tolist()} fluid_delta={mean_fluid.tolist()}"
        )

    weight = mass * SOURCE_GRAVITY_CM_S2
    magnitude_ratio = float(np.linalg.norm(mean_fluid) / weight)
    vertical_ratio = float(mean_fluid[2] / weight)
    direction = mean_fluid / max(np.linalg.norm(mean_fluid), 1e-30)
    return {
        "root_pitch_deg": float(root_pitch_deg),
        "mass_mg": mass * 1000.0,
        "mean_fluid_force_g_cm_s2": mean_fluid.tolist(),
        "mean_fluid_force_direction": direction.tolist(),
        "force_magnitude_to_weight": magnitude_ratio,
        "vertical_support_ratio_to_weight": vertical_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--phase-samples", type=int, default=1000)
    parser.add_argument(
        "--root-pitches-deg",
        type=float,
        nargs="+",
        default=(-47.5, 0.0, 47.5),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-upstream-source-orientation-sweep.json"),
    )
    args = parser.parse_args()

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
    rows = [
        run_pitch(cycle, float(pitch), int(args.phase_samples))
        for pitch in args.root_pitches_deg
    ]
    for row in rows:
        force = row["mean_fluid_force_g_cm_s2"]
        print(
            "upstream_orientation root_pitch={:+.1f} vertical_ratio={:+.3f} "
            "magnitude_ratio={:.3f} force_xyz=[{:+.4f},{:+.4f},{:+.4f}]".format(
                float(row["root_pitch_deg"]),
                float(row["vertical_support_ratio_to_weight"]),
                float(row["force_magnitude_to_weight"]),
                float(force[0]),
                float(force[1]),
                float(force[2]),
            )
        )

    best_vertical = max(rows, key=lambda x: float(x["vertical_support_ratio_to_weight"]))
    best_magnitude = max(rows, key=lambda x: float(x["force_magnitude_to_weight"]))
    if float(best_magnitude["force_magnitude_to_weight"]) >= 0.70:
        diagnosis = "AERO_FORCE_MAGNITUDE_NEAR_WEIGHT_BUT_ORIENTATION_MATTERS"
    else:
        diagnosis = "AERO_FORCE_MAGNITUDE_ITSELF_UNDER_WEIGHT"

    result = {
        "schema_version": 2,
        "mujoco_version": getattr(mj, "__version__", "unknown"),
        "pattern": str(args.pattern),
        "phase_samples": int(args.phase_samples),
        "cases": rows,
        "best_vertical_case": best_vertical,
        "best_magnitude_case": best_magnitude,
        "diagnosis": diagnosis,
        "force_definition": (
            "aero-on minus aero-off qfrc_fluid; the off/source model may already "
            "carry ordinary inertia-based fluid force because density/viscosity are non-zero"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
