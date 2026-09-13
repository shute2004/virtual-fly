#!/usr/bin/env python3
"""Flight-physics compatibility layer for FlyGym 2.1's experimental FlyBody.

FlyGym 2.1 imports the FlyBody articulated body, joints, meshes, and actuator
configuration, but its FlyBody conversion intentionally omits the source model's
``*_fluid`` wing geoms. That is appropriate for walking experiments, but it removes
the per-wing MuJoCo fluid interaction used by the original FlyBody flight tasks.

This module restores only those missing flight-physics pieces and applies the
flight-specific joint/actuator calibration published with the source FlyBody task.
It does not implement a controller or choose behavior for the fly.

Unit conversion
---------------
The source FlyBody MJCF is authored in centimetres while FlyGym's parsed FlyBody is
in millimetres. FlyGym's parser already scales ordinary body lengths and torque-like
joint/actuator parameters. The omitted flight-only quantities therefore need the
same conversion here:

* position / geom size: x10
* torque-like stiffness, damping, actuator gain: x100
* fluid density (mass / length^3): x1e-3
* dynamic viscosity (mass / length / time): x1e-1

The source constants come from TuragaLab/flybody's public ``fruitfly.xml`` and
``flybody/tasks/constants.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from flygym.compose import ActuatorType
from flygym.compose.fly import FlyBody
from flygym.compose.world.base_world import BaseWorld
from flygym.utils.mjcf import GEOM_TYPES


SOURCE_LENGTH_TO_MM = 10.0
SOURCE_TORQUE_TO_FLYGYM = SOURCE_LENGTH_TO_MM**2
SOURCE_DENSITY_TO_FLYGYM = SOURCE_LENGTH_TO_MM**-3
SOURCE_VISCOSITY_TO_FLYGYM = SOURCE_LENGTH_TO_MM**-1

# Original FlyBody flight-task constants (source units: cm, source torque units).
SOURCE_FLIGHT_PHYSICS_TIMESTEP_S = 5.0e-5
SOURCE_WING_GAIN = 18.0
SOURCE_WING_STIFFNESS = 0.01
SOURCE_WING_DAMPING = 0.007769230
SOURCE_FLUID_COEFS = (1.0, 0.5, 1.5, 1.7, 1.0)
SOURCE_AIR_DENSITY = 0.00128
SOURCE_AIR_VISCOSITY = 0.000185

FLIGHT_PHYSICS_TIMESTEP_S = SOURCE_FLIGHT_PHYSICS_TIMESTEP_S
FLIGHT_WING_POSITION_KP = SOURCE_WING_GAIN * SOURCE_TORQUE_TO_FLYGYM
FLIGHT_WING_STIFFNESS = SOURCE_WING_STIFFNESS * SOURCE_TORQUE_TO_FLYGYM
FLIGHT_WING_DAMPING = SOURCE_WING_DAMPING * SOURCE_TORQUE_TO_FLYGYM
FLIGHT_FLUID_COEFS = SOURCE_FLUID_COEFS
FLIGHT_AIR_DENSITY = SOURCE_AIR_DENSITY * SOURCE_DENSITY_TO_FLYGYM
FLIGHT_AIR_VISCOSITY = SOURCE_AIR_VISCOSITY * SOURCE_VISCOSITY_TO_FLYGYM


@dataclass(frozen=True)
class WingFluidGeom:
    body_segment: str
    geom_name: str
    size_mm: tuple[float, float, float]
    pos_mm: tuple[float, float, float]
    quat: tuple[float, float, float, float]


# Geometry copied from the source FlyBody MJCF and converted cm -> mm.
WING_FLUID_GEOMS = (
    WingFluidGeom(
        body_segment="l_wing",
        geom_name="wing_left_fluid",
        size_mm=(0.005, 0.551, 1.14),
        pos_mm=(0.263, -1.48, -0.289),
        quat=(-0.685, -0.634, 0.265, -0.243),
    ),
    WingFluidGeom(
        body_segment="r_wing",
        geom_name="wing_right_fluid",
        size_mm=(0.005, 0.551, 1.14),
        pos_mm=(-0.263, 1.48, 0.289),
        quat=(0.243, 0.265, 0.634, -0.685),
    ),
)


def partition_wing_dofs(jointdofs: Iterable) -> tuple[list, list]:
    """Return ``(non_wing, wing)`` DOFs while preserving input order."""

    non_wing = []
    wing = []
    for dof in jointdofs:
        (wing if dof.child.is_wing() else non_wing).append(dof)
    return non_wing, wing


def apply_flight_wing_joint_parameters(fly: FlyBody) -> None:
    """Apply the original FlyBody flight-task stiffness/damping to wing joints."""

    wing_count = 0
    for dof, joint in fly.jointdof_to_mjcfjoint.items():
        if not dof.child.is_wing():
            continue
        joint.stiffness = FLIGHT_WING_STIFFNESS
        joint.damping = FLIGHT_WING_DAMPING
        wing_count += 1
    if wing_count != 6:
        raise RuntimeError(f"expected 6 FlyBody wing DOFs, found {wing_count}")


def add_flight_wing_aerodynamics(fly: FlyBody) -> None:
    """Restore the two per-wing MuJoCo ellipsoid-fluid geoms omitted by FlyGym."""

    body_by_name = {segment.name: body for segment, body in fly.bodyseg_to_mjcfbody.items()}
    for spec in WING_FLUID_GEOMS:
        try:
            wing_body = body_by_name[spec.body_segment]
        except KeyError as exc:
            raise RuntimeError(
                f"FlyBody segment {spec.body_segment!r} required for flight is missing"
            ) from exc

        geom = wing_body.add_geom(
            type=GEOM_TYPES["ellipsoid"],
            name=spec.geom_name,
            size=spec.size_mm,
            pos=spec.pos_mm,
            quat=spec.quat,
            mass=0.0,
            contype=0,
            conaffinity=0,
            group=3,
            rgba=(0.0, 0.0, 0.0, 0.0),
        )
        # MjSpec exposes MJCF's fluidshape="ellipsoid" as fluid_ellipsoid and
        # fluidcoef as fluid_coefs.
        geom.fluid_ellipsoid = 1.0
        geom.fluid_coefs = list(FLIGHT_FLUID_COEFS)


def apply_flight_air_parameters(world: BaseWorld) -> None:
    """Set air density/viscosity in FlyGym's millimetre unit system."""

    world.mjcf_root.option.density = FLIGHT_AIR_DENSITY
    world.mjcf_root.option.viscosity = FLIGHT_AIR_VISCOSITY


def add_flight_position_actuators(fly: FlyBody, jointdofs: Iterable) -> None:
    """Add position actuators, using the source flight gain only for wing DOFs."""

    non_wing, wing = partition_wing_dofs(jointdofs)
    if non_wing:
        fly.add_actuators(non_wing, ActuatorType.POSITION)
    if len(wing) != 6:
        raise RuntimeError(f"expected 6 actuated FlyBody wing DOFs, found {len(wing)}")
    fly.add_actuators(
        wing,
        ActuatorType.POSITION,
        kp=FLIGHT_WING_POSITION_KP,
    )
