#!/usr/bin/env python3
"""MuJoCo world for the first Flyppy embodiment experiment."""

from __future__ import annotations

from flygym.compose import ContactParams, FlatGroundWorld
from flygym.flybody.anatomy_flybody import FlyBodyContactBodiesPreset
from flygym.utils.mjcf import GEOM_TYPES

from flyppy_course import FlyppyCourse


class FlyppyWorld(FlatGroundWorld):
    """Flat world with visible, collidable upper/lower Flyppy gate blocks."""

    def __init__(
        self,
        course: FlyppyCourse,
        *,
        lateral_half_width_mm: float = 8.0,
        wall_thickness_mm: float = 0.25,
    ) -> None:
        super().__init__(name="flyppy_world", half_size=200)
        if lateral_half_width_mm <= 0 or wall_thickness_mm <= 0:
            raise ValueError("gate dimensions must be positive")
        self.course = course
        self.lateral_half_width_mm = float(lateral_half_width_mm)
        self.wall_thickness_mm = float(wall_thickness_mm)
        self.obstacle_geoms = []

        worldbody = self.mjcf_root.worldbody
        for index, gate in enumerate(course.gates):
            lower_height = gate.low_z_mm - course.floor_z_mm
            upper_height = course.ceiling_z_mm - gate.high_z_mm
            if lower_height <= 0 or upper_height <= 0:
                raise ValueError(f"gate {index} does not fit inside corridor")

            lower_center_z = course.floor_z_mm + lower_height / 2.0
            upper_center_z = gate.high_z_mm + upper_height / 2.0
            rgba = [0.18, 0.18, 0.18, 1.0]
            self.obstacle_geoms.append(
                worldbody.add_geom(
                    type=GEOM_TYPES["box"],
                    name=f"flyppy_gate_{index}_lower",
                    size=[wall_thickness_mm, lateral_half_width_mm, lower_height / 2.0],
                    pos=[gate.x_mm, 0.0, lower_center_z],
                    rgba=rgba,
                    contype=0,
                    conaffinity=0,
                )
            )
            self.obstacle_geoms.append(
                worldbody.add_geom(
                    type=GEOM_TYPES["box"],
                    name=f"flyppy_gate_{index}_upper",
                    size=[wall_thickness_mm, lateral_half_width_mm, upper_height / 2.0],
                    pos=[gate.x_mm, 0.0, upper_center_z],
                    rgba=rgba,
                    contype=0,
                    conaffinity=0,
                )
            )

        # A physical ceiling prevents the free body from escaping above the
        # analytic corridor. The floor already comes from FlatGroundWorld.
        self.obstacle_geoms.append(
            worldbody.add_geom(
                type=GEOM_TYPES["box"],
                name="flyppy_ceiling",
                size=[100.0, lateral_half_width_mm, wall_thickness_mm],
                pos=[50.0, 0.0, course.ceiling_z_mm + wall_thickness_mm],
                rgba=[0.65, 0.65, 0.65, 1.0],
                contype=0,
                conaffinity=0,
            )
        )

    def add_obstacle_contacts(self, fly) -> int:
        """Pair gate/ceiling geoms with FlyBody collision geoms explicitly."""
        contact = ContactParams()
        segments = (
            FlyBodyContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD.to_body_segments_list()
        )
        fly_geoms = [geom for segment in segments for geom in fly.bodyseg_to_mjcfgeom[segment]]
        count = 0
        for obstacle in self.obstacle_geoms:
            for fly_geom in fly_geoms:
                self.mjcf_root.add_pair(
                    geomname1=fly_geom.name,
                    geomname2=obstacle.name,
                    name=f"flyppy_contact_{count}",
                    friction=contact.get_friction_tuple(),
                    solref=contact.get_solref_tuple(),
                    solimp=contact.get_solimp_tuple(),
                    margin=contact.margin,
                )
                count += 1
        return count
