#!/usr/bin/env python3
"""Verify that restored FlyBody flight physics survive MjSpec compilation."""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np

from flybody_adapter import FlyBodyWingAdapter, WingDrive
from flybody_flight_physics import (
    FLIGHT_AIR_DENSITY,
    FLIGHT_AIR_VISCOSITY,
    FLIGHT_FLUID_COEFS,
    FLIGHT_PHYSICS_TIMESTEP_S,
    FLIGHT_WING_DAMPING,
    FLIGHT_WING_INERTIAL_MASS,
    FLIGHT_WING_POSITION_KP,
    FLIGHT_WING_STIFFNESS,
    WING_FLIGHT_GEOMS,
)


def names(model: mj.MjModel, objtype: mj.mjtObj, count: int) -> list[str]:
    return [mj.mj_id2name(model, objtype, i) or "" for i in range(count)]


def find_suffix(all_names: list[str], suffix: str) -> int:
    matches = [i for i, name in enumerate(all_names) if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one compiled name ending {suffix!r}, got {matches}")
    return matches[0]


def require_spec_geom(body: FlyBodyWingAdapter, name: str):
    """Return a source MjSpec geom whose compile-time-only fields can be inspected."""

    geom = body.fly.mjcf_root.geom(name)
    if geom is None:
        raise RuntimeError(f"FlyBody source MjSpec geom not found: {name}")
    return geom


def assert_compiled_flight_physics(body: FlyBodyWingAdapter) -> None:
    model = body.sim.mj_model

    if not math.isclose(
        model.opt.timestep, FLIGHT_PHYSICS_TIMESTEP_S, rel_tol=0, abs_tol=1e-12
    ):
        raise RuntimeError(f"flight timestep mismatch: {model.opt.timestep}")
    if not math.isclose(
        model.opt.density, FLIGHT_AIR_DENSITY, rel_tol=1e-12, abs_tol=0
    ):
        raise RuntimeError(f"air density mismatch: {model.opt.density}")
    if not math.isclose(
        model.opt.viscosity, FLIGHT_AIR_VISCOSITY, rel_tol=1e-12, abs_tol=0
    ):
        raise RuntimeError(f"air viscosity mismatch: {model.opt.viscosity}")

    geom_names = names(model, mj.mjtObj.mjOBJ_GEOM, model.ngeom)
    geom_fluid = np.asarray(model.geom_fluid, dtype=np.float64).reshape(model.ngeom, -1)
    if geom_fluid.shape[1] < 6:
        raise RuntimeError(
            f"MuJoCo geom_fluid row is unexpectedly short: {geom_fluid.shape[1]}"
        )
    # MuJoCo 3.x currently stores mjNFLUID=12 values per geom. The first six are
    # the ellipsoid-model activation flag followed by the five MJCF fluidcoef
    # values. The remaining compiled values are internal derived parameters and
    # are deliberately not asserted here.
    expected_fluid_prefix = np.asarray((1.0, *FLIGHT_FLUID_COEFS), dtype=np.float64)
    for spec in WING_FLIGHT_GEOMS:
        fluid_id = find_suffix(geom_names, spec.fluid_name)
        fluid_prefix = geom_fluid[fluid_id, :6]
        if not np.allclose(fluid_prefix, expected_fluid_prefix, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"{spec.fluid_name} fluid coefficients mismatch: {fluid_prefix.tolist()}"
            )
        if not np.allclose(model.geom_size[fluid_id], spec.size_mm, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"{spec.fluid_name} size mismatch: {model.geom_size[fluid_id].tolist()}"
            )

        inertial_id = find_suffix(geom_names, spec.inertial_name)
        if not np.allclose(
            model.geom_size[inertial_id], spec.size_mm, rtol=0, atol=1e-12
        ):
            raise RuntimeError(
                f"{spec.inertial_name} size mismatch: "
                f"{model.geom_size[inertial_id].tolist()}"
            )

        # Geom mass is a compile-time MjSpec property. MuJoCo folds geom mass and
        # inertia into body inertial properties and deliberately does not retain a
        # geom_mass array in mjModel, so inspect the source MjSpec rather than a
        # non-existent runtime field.
        fluid_spec = require_spec_geom(body, spec.fluid_name)
        if not math.isclose(float(fluid_spec.mass), 0.0, rel_tol=0, abs_tol=1e-15):
            raise RuntimeError(
                f"{spec.fluid_name} must be massless, got {fluid_spec.mass}"
            )

        inertial_spec = require_spec_geom(body, spec.inertial_name)
        if not math.isclose(
            float(inertial_spec.mass),
            FLIGHT_WING_INERTIAL_MASS,
            rel_tol=1e-12,
            abs_tol=0,
        ):
            raise RuntimeError(
                f"{spec.inertial_name} mass mismatch: {inertial_spec.mass}"
            )

        membrane_spec = require_spec_geom(body, spec.membrane_name)
        if not math.isclose(
            float(membrane_spec.mass), 0.0, rel_tol=0, abs_tol=1e-15
        ):
            raise RuntimeError(
                f"{spec.membrane_name} still carries transferred inertial mass: "
                f"{membrane_spec.mass}"
            )

    joint_names = names(model, mj.mjtObj.mjOBJ_JOINT, model.njnt)
    wing_joint_ids = [
        i
        for i, name in enumerate(joint_names)
        if ("-l_wing-" in name or "-r_wing-" in name)
    ]
    if len(wing_joint_ids) != 6:
        raise RuntimeError(f"expected 6 compiled wing joints, got {len(wing_joint_ids)}")
    for joint_id in wing_joint_ids:
        if not math.isclose(
            float(model.jnt_stiffness[joint_id]),
            FLIGHT_WING_STIFFNESS,
            rel_tol=1e-12,
            abs_tol=0,
        ):
            raise RuntimeError(
                f"wing stiffness mismatch for {joint_names[joint_id]}: "
                f"{model.jnt_stiffness[joint_id]}"
            )
        dof_id = int(model.jnt_dofadr[joint_id])
        if not math.isclose(
            float(model.dof_damping[dof_id]),
            FLIGHT_WING_DAMPING,
            rel_tol=1e-12,
            abs_tol=0,
        ):
            raise RuntimeError(
                f"wing damping mismatch for {joint_names[joint_id]}: "
                f"{model.dof_damping[dof_id]}"
            )

    actuator_names = names(model, mj.mjtObj.mjOBJ_ACTUATOR, model.nu)
    position_actuator_ids = [
        i for i, name in enumerate(actuator_names) if name.endswith("-position")
    ]
    wing_actuator_ids = [
        i
        for i in position_actuator_ids
        if ("-l_wing-" in actuator_names[i] or "-r_wing-" in actuator_names[i])
    ]
    if len(position_actuator_ids) != 6 or len(wing_actuator_ids) != 6:
        unexpected = [actuator_names[i] for i in position_actuator_ids if i not in wing_actuator_ids]
        raise RuntimeError(
            "flight adapter must expose exactly six wing POSITION actuators; "
            f"position={len(position_actuator_ids)} wing={len(wing_actuator_ids)} "
            f"unexpected={unexpected}"
        )
    for actuator_id in wing_actuator_ids:
        kp = float(model.actuator_gainprm[actuator_id, 0])
        if not math.isclose(kp, FLIGHT_WING_POSITION_KP, rel_tol=1e-12, abs_tol=0):
            raise RuntimeError(
                f"wing actuator gain mismatch for {actuator_names[actuator_id]}: {kp}"
            )


def assert_reset_initial_condition() -> None:
    initial_velocity = np.asarray((123.0, -4.0, 5.0), dtype=np.float64)
    body = FlyBodyWingAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=tuple(initial_velocity),
        enable_wing_aerodynamics=True,
    )
    if not np.allclose(
        body.root_linear_velocity_mm_s(), initial_velocity, rtol=0, atol=1e-12
    ):
        raise RuntimeError("free FlyBody did not receive requested initial velocity")

    initial_wings = body.wing_joint_angles_rad()
    expected_initial_wings = {
        "left_yaw": -0.8,
        "left_roll": -0.1,
        "left_pitch": 0.8,
        "right_yaw": -0.8,
        "right_roll": -0.1,
        "right_pitch": 0.8,
    }
    for name, expected in expected_initial_wings.items():
        actual = initial_wings[name]
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
            raise RuntimeError(f"initial wing phase mismatch for {name}: {actual}")

    body.step(WingDrive(left=0.0, right=0.0), physics_steps=10)
    if np.allclose(body.root_linear_velocity_mm_s(), initial_velocity, rtol=0, atol=1e-12):
        raise RuntimeError(
            "root velocity was unchanged after physics steps; reset test cannot distinguish "
            "a real reset from a stale state"
        )

    body.reset()
    if not np.allclose(
        body.root_linear_velocity_mm_s(), initial_velocity, rtol=0, atol=1e-12
    ):
        raise RuntimeError("FlyBody reset did not restore requested initial velocity")
    reset_wings = body.wing_joint_angles_rad()
    for name, expected in expected_initial_wings.items():
        actual = reset_wings[name]
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
            raise RuntimeError(f"FlyBody reset did not restore wing phase for {name}: {actual}")

    # The source prototype defines one base cycle and repeats it. In particular,
    # its 1.5*phase roll term must repeat after 2*pi rather than alternating sign
    # between successive cycles.
    base = body._wing_pattern(0.4, 1.0)
    repeated = body._wing_pattern(0.4 + 2.0 * math.pi, 1.0)
    for axis in ("yaw", "roll", "pitch"):
        if not math.isclose(base[axis], repeated[axis], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"wing pattern is not cycle-periodic on axis {axis}")


def main() -> int:
    tethered = FlyBodyWingAdapter(tethered=True, enable_wing_aerodynamics=True)
    assert_compiled_flight_physics(tethered)
    assert_reset_initial_condition()

    model = tethered.sim.mj_model
    print(f"flight_timestep_s={model.opt.timestep:.8f}")
    print(f"air_density={model.opt.density:.9g}")
    print(f"air_viscosity={model.opt.viscosity:.9g}")
    print(f"wing_fluid_geoms={len(WING_FLIGHT_GEOMS)}")
    print(f"wing_inertial_geoms={len(WING_FLIGHT_GEOMS)}")
    print(f"wing_position_kp={FLIGHT_WING_POSITION_KP:.6f}")
    print("wing_position_actuators=6")
    print("flight_reset_initial_conditions=PASS")
    print("flybody_flight_physics=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
