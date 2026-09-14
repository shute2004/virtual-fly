#!/usr/bin/env python3
"""MuJoCo world for the Flyppy embodiment experiments."""

from __future__ import annotations

import mujoco as mj

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
        lateral_half_width_mm: float | None = None,
        wall_thickness_mm: float | None = None,
    ) -> None:
        super().__init__(name="flyppy_world", half_size=200)
        lateral = (
            course.lateral_half_width_mm
            if lateral_half_width_mm is None
            else float(lateral_half_width_mm)
        )
        # Historical v1 visual wall half-thickness was 0.25 mm.  v2/v3 use the
        # explicit versioned task specification shared with course geometry.
        wall_half = (
            course.gate_half_thickness_mm
            if wall_thickness_mm is None and course.environment_version in {"v2", "v3"}
            else 0.25 if wall_thickness_mm is None else float(wall_thickness_mm)
        )
        if lateral <= 0 or wall_half <= 0:
            raise ValueError("gate dimensions must be positive")
        self.course = course
        self.lateral_half_width_mm = float(lateral)
        self.wall_half_thickness_mm = float(wall_half)
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
                    size=[self.wall_half_thickness_mm, self.lateral_half_width_mm, lower_height / 2.0],
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
                    size=[self.wall_half_thickness_mm, self.lateral_half_width_mm, upper_height / 2.0],
                    pos=[gate.x_mm, 0.0, upper_center_z],
                    rgba=rgba,
                    contype=0,
                    conaffinity=0,
                )
            )

        self.obstacle_geoms.append(
            worldbody.add_geom(
                type=GEOM_TYPES["box"],
                name="flyppy_ceiling",
                size=[100.0, self.lateral_half_width_mm, self.wall_half_thickness_mm],
                pos=[50.0, 0.0, course.ceiling_z_mm + self.wall_half_thickness_mm],
                rgba=[0.65, 0.65, 0.65, 1.0],
                contype=0,
                conaffinity=0,
            )
        )

    def add_obstacle_contacts(self, fly) -> int:
        """Pair every FlyBody collision geom with gate/ceiling geoms.

        v1's historical world used legs/thorax/abdomen/head only.  Full-body
        obstacle contact is intentional for v2/v3 so a wing, haltere, antenna or
        other articulated segment cannot visually pass through a gate without a
        physical collision event.
        """

        contact = ContactParams()
        segments = FlyBodyContactBodiesPreset.ALL.to_body_segments_list()
        fly_geoms = [
            geom
            for segment in segments
            for geom in fly.bodyseg_to_mjcfgeom[segment]
        ]
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

    @staticmethod
    def physical_collision_reason(sim) -> str | None:
        """Classify an actual MuJoCo contact involving a Flyppy boundary.

        Only contacts at or inside zero distance count as collision.  MuJoCo can
        retain positive-distance contacts within a configured margin; treating
        those as impact would artificially shrink the aperture.
        """

        model = sim.mj_model
        data = sim.mj_data
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if float(contact.dist) > 0.0:
                continue
            name1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
            name2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
            names = (name1, name2)
            if "ground_plane" in names:
                return "floor"
            if "flyppy_ceiling" in names:
                return "ceiling"
            if any(name.startswith("flyppy_gate_") for name in names):
                return "gate"
        return None
