#!/usr/bin/env python3
"""Sweep upstream FlyBody root pitch under exact measured wing kinematics.

Calibration-only diagnostic. This compiles TuragaLab/flybody's original
``fruitfly.xml`` directly with the currently installed MuJoCo, enables the source
wing ellipsoid-fluid geoms, imposes the official measured wing qpos/qvel exactly,
and evaluates fluid generalized force for several root pitches.

Two quantities are intentionally reported separately:

* total active fluid force: the actual ``qfrc_fluid`` present in the source flight
  model with wing ellipsoid fluid enabled;
* incremental ellipsoid contribution: aero-on minus aero-off ``qfrc_fluid``.

The distinction matters because the source XML already has non-zero air density and
viscosity, so the aero-off model can still carry MuJoCo's ordinary inertia-based
fluid force.
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


def _force_summary(force: np.ndarray, weight: float) -> dict[str, object]:
    mean = np.mean(force, axis=0)
    magnitude = float(np.linalg.norm(mean))
    direction = mean / max(magnitude, 1e-30)
    return {
        "mean_force_g_cm_s2": mean.tolist(),
        "mean_force_direction": direction.tolist(),
        "force_magnitude_to_weight": magnitude / weight,
        "vertical_support_ratio_to_weight": float(mean[2] / weight),
    }


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
    off_fluid: list[np.ndarray] = []
    on_fluid: list[np.ndarray] = []
    for i in range(phase_samples):
        phase = 2.0 * math.pi * i / phase_samples
        set_state(
            off_model, off_data, cycle, phase, root_pitch_deg,
            off_qpos, off_dof, off_wings,
        )
        set_state(
            on_model, on_data, cycle, phase, root_pitch_deg,
            on_qpos, on_dof, on_wings,
        )

        passive_delta.append(
            np.asarray(on_data.qfrc_passive[on_dof : on_dof + 3], dtype=np.float64)
            - np.asarray(off_data.qfrc_passive[off_dof : off_dof + 3], dtype=np.float64)
        )
        off_fluid.append(
            np.asarray(off_data.qfrc_fluid[off_dof : off_dof + 3], dtype=np.float64).copy()
        )
        on_fluid.append(
            np.asarray(on_data.qfrc_fluid[on_dof : on_dof + 3], dtype=np.float64).copy()
        )

    passive = np.asarray(passive_delta, dtype=np.float64)
    off_force = np.asarray(off_fluid, dtype=np.float64)
    on_force = np.asarray(on_fluid, dtype=np.float64)
    fluid_delta = on_force - off_force

    mean_passive = np.mean(passive, axis=0)
    mean_delta = np.mean(fluid_delta, axis=0)
    if not np.allclose(mean_passive, mean_delta, rtol=1e-8, atol=1e-10):
        raise RuntimeError(
            "aero-on/off qfrc_passive delta disagrees with aero-on/off qfrc_fluid delta: "
            f"passive={mean_passive.tolist()} fluid_delta={mean_delta.tolist()}"
        )

    weight = mass * SOURCE_GRAVITY_CM_S2
    return {
        "root_pitch_deg": float(root_pitch_deg),
        "mass_mg": mass * 1000.0,
        "baseline_inertia_fluid": _force_summary(off_force, weight),
        "active_total_fluid": _force_summary(on_force, weight),
        "incremental_wing_ellipsoid": _force_summary(fluid_delta, weight),
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
        total = row["active_total_fluid"]
        incremental = row["incremental_wing_ellipsoid"]
        baseline = row["baseline_inertia_fluid"]
        force = total["mean_force_g_cm_s2"]
        print(
            "upstream_orientation root_pitch={:+.1f} "
            "total_vertical={:+.3f} total_magnitude={:.3f} "
            "incremental_vertical={:+.3f} baseline_vertical={:+.3f} "
            "total_force_xyz=[{:+.4f},{:+.4f},{:+.4f}]".format(
                float(row["root_pitch_deg"]),
                float(total["vertical_support_ratio_to_weight"]),
                float(total["force_magnitude_to_weight"]),
                float(incremental["vertical_support_ratio_to_weight"]),
                float(baseline["vertical_support_ratio_to_weight"]),
                float(force[0]), float(force[1]), float(force[2]),
            )
        )

    best_vertical = max(
        rows,
        key=lambda x: float(x["active_total_fluid"]["vertical_support_ratio_to_weight"]),
    )
    best_magnitude = max(
        rows,
        key=lambda x: float(x["active_total_fluid"]["force_magnitude_to_weight"]),
    )
    best_vertical_ratio = float(
        best_vertical["active_total_fluid"]["vertical_support_ratio_to_weight"]
    )
    best_magnitude_ratio = float(
        best_magnitude["active_total_fluid"]["force_magnitude_to_weight"]
    )

    if 0.70 <= best_vertical_ratio <= 1.30:
        diagnosis = "UPSTREAM_TOTAL_FLUID_CAN_APPROXIMATELY_SUPPORT_WEIGHT"
    elif best_magnitude_ratio >= 0.70:
        diagnosis = "UPSTREAM_TOTAL_FLUID_MAGNITUDE_NEAR_WEIGHT_BUT_ORIENTATION_MATTERS"
    else:
        diagnosis = "UPSTREAM_TOTAL_FLUID_MAGNITUDE_ITSELF_UNDER_WEIGHT"

    result = {
        "schema_version": 3,
        "mujoco_version": getattr(mj, "__version__", "unknown"),
        "pattern": str(args.pattern),
        "phase_samples": int(args.phase_samples),
        "cases": rows,
        "best_vertical_case": best_vertical,
        "best_magnitude_case": best_magnitude,
        "diagnosis": diagnosis,
        "force_definition": {
            "active_total_fluid": "qfrc_fluid in the source model with wing ellipsoid fluid enabled",
            "baseline_inertia_fluid": "qfrc_fluid in the source model before enabling wing ellipsoid fluid",
            "incremental_wing_ellipsoid": "active_total_fluid minus baseline_inertia_fluid",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
