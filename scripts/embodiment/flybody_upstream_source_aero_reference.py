#!/usr/bin/env python3
"""Compare exact-kinematic aerodynamic support against the upstream FlyBody XML.

This is the strongest current port-equivalence diagnostic: both the upstream source
XML and virtual-fly are evaluated by the *same installed MuJoCo library* and with the
same measured ``wing_pattern_fmech.npy`` qpos/qvel.  The upstream case bypasses
FlyGym completely.

If the upstream XML supports weight while virtual-fly does not, the mismatch lies in
the FlyGym/virtual-fly conversion.  If both underproduce similarly, the remaining
mismatch is in the diagnostic/calibration reproduction or MuJoCo-version semantics,
not merely in FlyGym's model conversion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from flybody_measured_wingbeat import AXES, DEFAULT_PATTERN, MeasuredWingbeatCycle


UPSTREAM_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
UPSTREAM_XML = (
    Path("artifacts/upstream")
    / f"flybody-{UPSTREAM_COMMIT}"
    / "flybody/fruitfly/assets/fruitfly.xml"
)
ACTIVE_XML = UPSTREAM_XML.with_name("fruitfly_exact_aero_active.xml")
SOURCE_FLUID_COEFS = (1.0, 0.5, 1.5, 1.7, 1.0)
SOURCE_GRAVITY_CM_S2 = 981.0
SOURCE_WINGBEAT_HZ = 218.0
WING_JOINTS = {
    "left": {
        "yaw": "wing_yaw_left",
        "roll": "wing_roll_left",
        "pitch": "wing_pitch_left",
    },
    "right": {
        "yaw": "wing_yaw_right",
        "roll": "wing_roll_right",
        "pitch": "wing_pitch_right",
    },
}
WING_FLUID_GEOMS = ("wing_left_fluid", "wing_right_fluid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--source-xml", type=Path, default=UPSTREAM_XML)
    parser.add_argument("--phase-samples", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-upstream-source-aero-reference.json"),
    )
    return parser.parse_args()


def write_active_xml(source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    found: set[str] = set()
    coefs = " ".join(str(v) for v in SOURCE_FLUID_COEFS)
    for geom in root.iter("geom"):
        name = geom.attrib.get("name")
        if name not in WING_FLUID_GEOMS:
            continue
        geom.set("fluidshape", "ellipsoid")
        geom.set("fluidcoef", coefs)
        found.add(name)
    missing = set(WING_FLUID_GEOMS) - found
    if missing:
        raise RuntimeError(f"upstream wing fluid geoms not found: {sorted(missing)}")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def compile_model(path: Path) -> tuple[mj.MjModel, mj.MjData]:
    model = mj.MjModel.from_xml_path(str(path))
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    return model, data


def freejoint_addresses(model: mj.MjModel) -> tuple[int, int]:
    free_ids = np.flatnonzero(
        np.asarray(model.jnt_type) == int(mj.mjtJoint.mjJNT_FREE)
    )
    if len(free_ids) != 1:
        raise RuntimeError(f"expected exactly one upstream free joint, found {len(free_ids)}")
    jid = int(free_ids[0])
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def joint_addresses(model: mj.MjModel) -> dict[tuple[str, str], tuple[int, int]]:
    result: dict[tuple[str, str], tuple[int, int]] = {}
    for side, by_axis in WING_JOINTS.items():
        for axis, name in by_axis.items():
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"upstream wing joint not found: {name}")
            result[(side, axis)] = (
                int(model.jnt_qposadr[jid]),
                int(model.jnt_dofadr[jid]),
            )
    return result


def set_exact_state(
    model: mj.MjModel,
    data: mj.MjData,
    cycle: MeasuredWingbeatCycle,
    phase: float,
    root_qpos_address: int,
    root_dof_address: int,
    wing_addresses: dict[tuple[str, str], tuple[int, int]],
) -> None:
    # Keep the source root fixed at identity pose/zero velocity.  Exact wing qpos/qvel
    # are imposed directly; no actuator or joint-tracking dynamics are involved.
    data.qpos[root_qpos_address : root_qpos_address + 7] = (
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    data.qvel[root_dof_address : root_dof_address + 6] = 0.0

    angles = cycle.angles(phase, 1.0)
    velocities = cycle.velocities(phase, 1.0, SOURCE_WINGBEAT_HZ)
    for side in ("left", "right"):
        for axis in AXES:
            qpos_address, qvel_address = wing_addresses[(side, axis)]
            data.qpos[qpos_address] = angles[axis]
            data.qvel[qvel_address] = velocities[axis]

    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mj.mj_forward(model, data)


def total_fly_mass_g(model: mj.MjModel) -> float:
    # World body is index 0 and carries no fly mass.
    mass = float(np.sum(np.asarray(model.body_mass, dtype=np.float64)[1:]))
    if not math.isfinite(mass) or mass <= 0.0:
        raise RuntimeError(f"invalid upstream FlyBody mass: {mass}")
    return mass


def run_case(
    cycle: MeasuredWingbeatCycle,
    source_xml: Path,
    phase_samples: int,
) -> dict[str, object]:
    write_active_xml(source_xml, ACTIVE_XML)
    off_model, off_data = compile_model(source_xml)
    on_model, on_data = compile_model(ACTIVE_XML)

    off_root_qpos, off_root_dof = freejoint_addresses(off_model)
    on_root_qpos, on_root_dof = freejoint_addresses(on_model)
    off_wings = joint_addresses(off_model)
    on_wings = joint_addresses(on_model)

    mass_off = total_fly_mass_g(off_model)
    mass_on = total_fly_mass_g(on_model)
    if not math.isclose(mass_off, mass_on, rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError(f"upstream aero A/B mass mismatch: off={mass_off} on={mass_on}")

    support_forces: list[float] = []
    for index in range(phase_samples):
        phase = 2.0 * math.pi * index / phase_samples
        set_exact_state(
            off_model,
            off_data,
            cycle,
            phase,
            off_root_qpos,
            off_root_dof,
            off_wings,
        )
        set_exact_state(
            on_model,
            on_data,
            cycle,
            phase,
            on_root_qpos,
            on_root_dof,
            on_wings,
        )
        support_forces.append(
            float(on_data.qfrc_passive[on_root_dof + 2])
            - float(off_data.qfrc_passive[off_root_dof + 2])
        )

    force = np.asarray(support_forces, dtype=np.float64)
    mean_support_accel = float(np.mean(force) / mass_on)
    ratio = mean_support_accel / SOURCE_GRAVITY_CM_S2
    fluid_rows = []
    geom_fluid = np.asarray(on_model.geom_fluid, dtype=np.float64).reshape(on_model.ngeom, -1)
    for name in WING_FLUID_GEOMS:
        gid = mj.mj_name2id(on_model, mj.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise RuntimeError(f"compiled upstream fluid geom missing: {name}")
        fluid_rows.append(
            {
                "name": name,
                "size_cm": np.asarray(on_model.geom_size[gid], dtype=np.float64).tolist(),
                "fluid_prefix": geom_fluid[gid, :6].tolist(),
            }
        )

    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "source_xml": str(source_xml),
        "mujoco_version": getattr(mj, "__version__", "unknown"),
        "phase_samples": int(phase_samples),
        "body_mass_mg": mass_on * 1000.0,
        "gravity_cm_s2": SOURCE_GRAVITY_CM_S2,
        "air_density_g_cm3": float(on_model.opt.density),
        "air_viscosity_g_cm_s": float(on_model.opt.viscosity),
        "mean_added_vertical_force_g_cm_s2": float(np.mean(force)),
        "std_added_vertical_force_g_cm_s2": float(np.std(force)),
        "mean_support_accel_cm_s2": mean_support_accel,
        "support_ratio_to_weight": ratio,
        "fluid_geoms": fluid_rows,
    }


def main() -> int:
    args = parse_args()
    if not args.pattern.exists():
        raise SystemExit(
            f"measured wing pattern missing: {args.pattern}; run the flight-data prefetch first"
        )
    if not args.source_xml.exists():
        raise SystemExit(
            f"upstream FlyBody source missing: {args.source_xml}; run "
            "uv run python scripts/dev/prefetch_upstream_flybody_source.py"
        )
    if args.phase_samples < 100:
        raise SystemExit("phase-samples must be >= 100")

    cycle = MeasuredWingbeatCycle(args.pattern)
    row = run_case(cycle, args.source_xml, args.phase_samples)
    ratio = float(row["support_ratio_to_weight"])
    if 0.70 <= ratio <= 1.30:
        diagnosis = "UPSTREAM_XML_SUPPORTS_WEIGHT_UNDER_CURRENT_MUJOCO"
    elif ratio < 0.70:
        diagnosis = "UPSTREAM_XML_ALSO_UNDERPRODUCES_WITH_CURRENT_EXACT_KINEMATICS"
    else:
        diagnosis = "UPSTREAM_XML_OVERPRODUCES_WITH_CURRENT_EXACT_KINEMATICS"

    result = {
        "schema_version": 1,
        "case": row,
        "diagnosis": diagnosis,
        "interpretation": (
            "Upstream source XML is compiled directly by the same installed MuJoCo; "
            "FlyGym is not involved. Exact official wing qpos/qvel are imposed and "
            "aero-on minus aero-off root vertical passive force is compared to weight."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "upstream_source_aero support_ratio={:.3f} support_accel={:+.3f}cm/s2 "
        "mass_mg={:.6f} density={:.9g} viscosity={:.9g}".format(
            ratio,
            float(row["mean_support_accel_cm_s2"]),
            float(row["body_mass_mg"]),
            float(row["air_density_g_cm3"]),
            float(row["air_viscosity_g_cm_s"]),
        )
    )
    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
