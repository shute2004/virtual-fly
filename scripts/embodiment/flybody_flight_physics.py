#!/usr/bin/env python3
"""Flight-physics compatibility layer for FlyGym 2.1's experimental FlyBody.

FlyGym 2.1 imports the FlyBody articulated body, joints, meshes, and actuator
configuration, but its conversion intentionally omits the source model's
``*_fluid`` and ``*_inertial`` wing geoms. The inertial mass is transferred to the
wing membrane mesh, which preserves mass but not necessarily the source inertia
tensor. That is suitable for non-flight use but is a mismatch for the original
FlyBody flight task.

This module restores those flight-only pieces and applies the flight-specific
joint/actuator calibration published with the source FlyBody task. It does not
implement a controller or choose behavior for the fly.

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
* mass: unchanged

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
SOURCE_BODY_PITCH_DEG = 47.5
SOURCE_WING_GAIN = 18.0
SOURCE_WING_STIFFNESS = 0.01
SOURCE_WING_DAMPING = 0.007769230
SOURCE_WING_INERTIAL_MASS = 8.0e-6
SOURCE_FLUID_COEFS = (1.0, 0.5, 1.5, 1.7, 1.0)
SOURCE_AIR_DENSITY = 0.00128
SOURCE_AIR_VISCOSITY = 0.000185

FLIGHT_PHYSICS_TIMESTEP_S = SOURCE_FLIGHT_PHYSICS_TIMESTEP_S
FLIGHT_BODY_PITCH_DEG = SOURCE_BODY_PITCH_DEG
FLIGHT_WING_POSITION_KP = SOURCE_WING_GAIN * SOURCE_TORQUE_TO_FLYGYM
FLIGHT_WING_STIFFNESS = SOURCE_WING_STIFFNESS * SOURCE_TORQUE_TO_FLYGYM
FLIGHT_WING_DAMPING = SOURCE_WING_DAMPING * SOURCE_TORQUE_TO_FLYGYM
FLIGHT_WING_INERTIAL_MASS = SOURCE_WING_INERTIAL_MASS
FLIGHT_FLUID_COEFS = SOURCE_FLUID_COEFS
FLIGHT_AIR_DENSITY = SOURCE_AIR_DENSITY * SOURCE_DENSITY_TO_FLYGYM
FLIGHT_AIR_VISCOSITY = SOURCE_AIR_VISCOSITY * SOURCE_VISCOSITY_TO_FLYGYM


@dataclass(frozen=True)
class WingFlightGeom:
    body_segment: str
    source_side: str
    size_mm: tuple[float, float, float]
    pos_mm: tuple[float, float, float]
    quat: tuple[float, float, float, float]

    @property
    def fluid_name(self) -> str:
        return f"wing_{self.source_side}_fluid"

    @property
    def inertial_name(self) -> str:
        return f"wing_{self.source_side}_inertial"

    @property
    def membrane_name(self) -> str:
        return f"{self.body_segment}_membrane"


# Geometry copied from the source FlyBody MJCF and converted cm -> mm.
WING_FLIGHT_GEOMS = (
    WingFlightGeom(
        body_segment="l_wing",
        source_side="left",
        size_mm=(0.005, 0.551, 1.14),
        pos_mm=(0.263, -1.48, -0.289),
        quat=(-0.685, -0.634, 0.265, -0.243),
    ),
    WingFlightGeom(
        body_segment="r_wing",
        source_side="right",
        size_mm=(0.005, 0.551, 1.14),
        pos_mm=(-0.263, 1.48, 0.289),
        quat=(0.243, 0.265, 0.634, -0.685),
    ),
)

# Backward-compatible alias used by the flight-physics smoke test.
WING_FLUID_GEOMS = WING_FLIGHT_GEOMS


def partition_wing_dofs(jointdofs: Iterable) -> tuple[list, list]:
    """Return ``(non_wing, wing)`` DOFs while preserving input order."""

    non_wing = []
    wing = []
    for dof in jointdofs:
        (wing if dof.child.is_wing() else non_wing).append(dof)
    return non_wing, wing


def restore_flight_wing_inertia(fly: FlyBody) -> None:
    """Restore source wing inertial boxes instead of membrane-mesh mass transfer."""

    body_by_name = {segment.name: body for segment, body in fly.bodyseg_to_mjcfbody.items()}
    geoms_by_name = {
        segment.name: list(geoms) for segment, geoms in fly.bodyseg_to_mjcfgeom.items()
    }

    for spec in WING_FLIGHT_GEOMS:
        try:
            wing_body = body_by_name[spec.body_segment]
            wing_geoms = geoms_by_name[spec.body_segment]
        except KeyError as exc:
            raise RuntimeError(
                f"FlyBody segment {spec.body_segment!r} required for flight is missing"
            ) from exc

        membranes = [geom for geom in wing_geoms if geom.name == spec.membrane_name]
        if len(membranes) != 1:
            raise RuntimeError(
                f"expected one {spec.membrane_name!r} geom, found {len(membranes)}"
            )
        # FlyGym transfers the source wing-inertial mass onto the membrane mesh.
        # Remove that transferred mass before re-introducing the source inertial box.
        membranes[0].mass = 0.0

        wing_body.add_geom(
            type=GEOM_TYPES["box"],
            name=spec.inertial_name,
            size=spec.size_mm,
            pos=spec.pos_mm,
            quat=spec.quat,
            mass=FLIGHT_WING_INERTIAL_MASS,
            contype=0,
            conaffinity=0,
            group=3,
            rgba=(0.0, 0.0, 0.0, 0.0),
        )


def apply_flight_wing_joint_parameters(fly: FlyBody) -> None:
    """Apply source flight wing joint mechanics and restore wing inertia."""

    wing_count = 0
    for dof, joint in fly.jointdof_to_mjcfjoint.items():
        if not dof.child.is_wing():
            continue
        joint.stiffness = FLIGHT_WING_STIFFNESS
        joint.damping = FLIGHT_WING_DAMPING
        wing_count += 1
    if wing_count != 6:
        raise RuntimeError(f"expected 6 FlyBody wing DOFs, found {wing_count}")

    # Apply this regardless of whether explicit fluid geoms are enabled. The
    # no-fluid flight-envelope control must differ only by wing aerodynamics, not
    # by wing mass/inertia distribution.
    restore_flight_wing_inertia(fly)


def add_flight_wing_aerodynamics(fly: FlyBody) -> None:
    """Restore the two per-wing MuJoCo ellipsoid-fluid geoms omitted by FlyGym."""

    body_by_name = {segment.name: body for segment, body in fly.bodyseg_to_mjcfbody.items()}
    for spec in WING_FLIGHT_GEOMS:
        try:
            wing_body = body_by_name[spec.body_segment]
        except KeyError as exc:
            raise RuntimeError(
                f"FlyBody segment {spec.body_segment!r} required for flight is missing"
            ) from exc

        # MjSpec exposes MJCF's fluidshape="ellipsoid" as fluid_ellipsoid and
        # fluidcoef as fluid_coefs. Passing them through add_geom avoids relying on
        # mutability details of the generated fixed-size array bindings.
        wing_body.add_geom(
            type=GEOM_TYPES["ellipsoid"],
            name=spec.fluid_name,
            size=spec.size_mm,
            pos=spec.pos_mm,
            quat=spec.quat,
            mass=0.0,
            contype=0,
            conaffinity=0,
            group=3,
            rgba=(0.0, 0.0, 0.0, 0.0),
            fluid_ellipsoid=1.0,
            fluid_coefs=list(FLIGHT_FLUID_COEFS),
        )


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
