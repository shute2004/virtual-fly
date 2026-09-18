#!/usr/bin/env python3
"""MuJoCo world for the Flyppy embodiment experiments."""

from __future__ import annotations

import mujoco as mj
import numpy as np

from flygym.compose import ContactParams, FlatGroundWorld
from flygym.flybody.anatomy_flybody import FlyBodyContactBodiesPreset
from flygym.utils.mjcf import GEOM_TYPES

from virtual_fly.embodiment.course import FlyppyCourse


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
        # Historical v1 visual wall half-thickness was 0.25 mm.  v2+ use the
        # explicit versioned task specification shared with course geometry.
        wall_half = (
            course.gate_half_thickness_mm
            if wall_thickness_mm is None and course.environment_version in {"v2", "v3", "v4", "v5", "v6", "v7"}
            else 0.25 if wall_thickness_mm is None else float(wall_thickness_mm)
        )
        if lateral <= 0 or wall_half <= 0:
            raise ValueError("gate dimensions must be positive")
        self.course = course
        self.lateral_half_width_mm = float(lateral)
        self.wall_half_thickness_mm = float(wall_half)
        self.obstacle_geoms = []
        self._fly_collision_geom_names: tuple[str, ...] = ()
        self._fly_mesh_cache_model_id: int | None = None
        self._fly_mesh_cache: tuple[tuple[int, np.ndarray], ...] = ()

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

        if course.environment_version == "v7":
            # v7 makes ``lateral_half_width_mm`` an actual physical corridor
            # boundary. Earlier versions intentionally retain their historical
            # open-y geometry for experiment reproducibility.
            corridor_height = course.ceiling_z_mm - course.floor_z_mm
            corridor_center_z = course.floor_z_mm + corridor_height / 2.0
            for side, sign in (("left", -1.0), ("right", 1.0)):
                self.obstacle_geoms.append(
                    worldbody.add_geom(
                        type=GEOM_TYPES["box"],
                        name=f"flyppy_side_{side}",
                        size=[100.0, self.wall_half_thickness_mm, corridor_height / 2.0],
                        pos=[
                            50.0,
                            sign * (self.lateral_half_width_mm + self.wall_half_thickness_mm),
                            corridor_center_z,
                        ],
                        rgba=[0.65, 0.65, 0.65, 0.08],
                        contype=0,
                        conaffinity=0,
                    )
                )

    def set_gate_center_z_mm(self, sim, gate_index: int, center_z_mm: float) -> None:
        """Move one compiled MuJoCo gate and the matching course opening together."""

        gate = self.course.set_gate_center_z_mm(gate_index, center_z_mm)
        lower_height = gate.low_z_mm - self.course.floor_z_mm
        upper_height = self.course.ceiling_z_mm - gate.high_z_mm
        if lower_height <= 0.0 or upper_height <= 0.0:
            raise ValueError("moved gate no longer fits inside corridor")

        model = sim.mj_model
        for suffix, height, center_z in (
            (
                "lower",
                lower_height,
                self.course.floor_z_mm + lower_height / 2.0,
            ),
            (
                "upper",
                upper_height,
                gate.high_z_mm + upper_height / 2.0,
            ),
        ):
            name = f"flyppy_gate_{int(gate_index)}_{suffix}"
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise RuntimeError(f"missing movable Flyppy gate geom: {name}")
            model.geom_pos[geom_id] = (float(gate.x_mm), 0.0, float(center_z))
            model.geom_size[geom_id] = (
                self.wall_half_thickness_mm,
                self.lateral_half_width_mm,
                float(height) / 2.0,
            )
        mj.mj_forward(model, sim.mj_data)

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
        self._fly_collision_geom_names = tuple(geom.name for geom in fly_geoms)
        self._fly_mesh_cache_model_id = None
        self._fly_mesh_cache = ()
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

    def full_body_x_bounds_mm(self, sim) -> tuple[float, float]:
        """Return exact world-x bounds of all FlyBody collision meshes.

        FlyBody's obstacle-contact set currently consists of mesh geoms. Their
        compiled mesh vertices are transformed by each geom's live MuJoCo pose,
        so articulated wings, legs, antennae and the abdomen all contribute to
        the clearance bound rather than a thorax-radius approximation.
        """

        model = sim.mj_model
        data = sim.mj_data
        model_id = id(model)
        if self._fly_mesh_cache_model_id != model_id:
            entries: list[tuple[int, np.ndarray]] = []
            for name in self._fly_collision_geom_names:
                geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
                if geom_id < 0:
                    raise RuntimeError(f"missing FlyBody collision geom: {name}")
                if int(model.geom_type[geom_id]) != int(mj.mjtGeom.mjGEOM_MESH):
                    raise RuntimeError(
                        f"FlyBody clearance requires mesh geom, got type={int(model.geom_type[geom_id])} name={name}"
                    )
                mesh_id = int(model.geom_dataid[geom_id])
                vertex_adr = int(model.mesh_vertadr[mesh_id])
                vertex_num = int(model.mesh_vertnum[mesh_id])
                vertices = np.asarray(
                    model.mesh_vert[vertex_adr : vertex_adr + vertex_num],
                    dtype=np.float64,
                ).copy()
                entries.append((geom_id, vertices))
            if not entries:
                raise RuntimeError("FlyBody clearance has no collision meshes")
            self._fly_mesh_cache = tuple(entries)
            self._fly_mesh_cache_model_id = model_id

        minimum = float("inf")
        maximum = float("-inf")
        for geom_id, vertices in self._fly_mesh_cache:
            # geom_xmat is row-major local->world rotation. The first row is
            # enough because only world x support is needed.
            world_x_axis_local = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)[0]
            center_x = float(data.geom_xpos[geom_id][0])
            x_values = vertices @ world_x_axis_local + center_x
            minimum = min(minimum, float(np.min(x_values)))
            maximum = max(maximum, float(np.max(x_values)))
        return minimum, maximum

    def step_muscles_until_boundary_contact(
        self,
        body,
        state,
        *,
        physics_steps: int,
        trace: list[dict[str, object]] | None = None,
    ) -> tuple[str | None, int]:
        """Advance physics one substep at a time and stop on the first boundary contact.

        One neural/muscle command may be held across multiple MuJoCo substeps. If
        contacts are checked only after the final substep, a transient wall/floor/
        ceiling contact can be missed and later misclassified as a clean gate pass.
        """

        if physics_steps < 1:
            raise ValueError("physics_steps must be >= 1")
        for completed in range(1, physics_steps + 1):
            body.step_muscles(state, physics_steps=1)
            if trace is not None:
                sim = body.sim
                root_dof = int(body._root_freejoint_dof_address())
                diagnostics = {
                    "root_fluid_force": [
                        float(value)
                        for value in sim.mj_data.qfrc_fluid[root_dof : root_dof + 3]
                    ],
                    "wing_angles": {
                        key: float(value)
                        for key, value in body.wing_joint_angles_rad().items()
                    },
                    "wing_torque": {
                        key: float(value)
                        for key, value in getattr(body, "last_wing_torque", {}).items()
                    },
                    "stroke_scale": {
                        key: float(value)
                        for key, value in getattr(
                            body, "last_stroke_amplitude_scale", {}
                        ).items()
                    },
                    "vertical_steering_activation": {
                        key: float(value)
                        for key, value in getattr(
                            body, "last_vertical_steering_activation", {}
                        ).items()
                    },
                }
                trace.append(
                    {
                        "sim_time_s": float(sim.mj_data.time),
                        "qpos": [float(value) for value in sim.mj_data.qpos],
                        "qvel": [float(value) for value in sim.mj_data.qvel],
                        "diagnostics": diagnostics,
                    }
                )
            reason = self.physical_collision_reason(body.sim)
            if reason is not None:
                return reason, completed
        return None, physics_steps

    @staticmethod
    def physical_collision_detail(sim) -> tuple[str | None, str | None]:
        """Classify a boundary contact and return the contacted boundary geom name."""

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
                return "floor", "ground_plane"
            if "flyppy_ceiling" in names:
                return "ceiling", "flyppy_ceiling"
            for name in names:
                if name.startswith("flyppy_side_"):
                    return "side", name
            for name in names:
                if name.startswith("flyppy_gate_"):
                    return "gate", name
        return None, None

    @staticmethod
    def physical_collision_reason(sim) -> str | None:
        """Classify an actual MuJoCo contact involving a Flyppy boundary.

        Only contacts at or inside zero distance count as collision.  MuJoCo can
        retain positive-distance contacts within a configured margin; treating
        those as impact would artificially shrink the aperture.
        """

        return FlyppyWorld.physical_collision_detail(sim)[0]
