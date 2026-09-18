#!/usr/bin/env python3
"""Shared FlyBody/MuJoCo runtime used by current and legacy motor interfaces.

This module owns body construction, flight-physics compatibility, root state,
joint access, vision access, canonical body mass and researched mechanical joint
limits.  It contains no CNS readout or action-selection policy.
"""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np

from flygym import Simulation
from flygym.compose import (
    ActuatorType,
    FlatGroundWorld,
    KinematicPosePreset,
    TetheredWorld,
)
from flygym.compose.fly import FlyBody
from flygym.compose.world.base_world import BaseWorld
from flygym.flybody.anatomy_flybody import (
    FlyBodyAxisOrder,
    FlyBodyContactBodiesPreset,
    FlyBodyJointPreset,
    FlyBodySkeleton,
)
from flygym.utils.math import Rotation3D

from virtual_fly.embodiment.body.biophysics import (
    apply_researched_joint_ranges,
    compiled_joint_ranges,
    normalize_fly_mass,
)
from virtual_fly.embodiment.body.flight_physics import (
    FLIGHT_BODY_PITCH_DEG,
    FLIGHT_PHYSICS_TIMESTEP_S,
    add_flight_position_actuators,
    add_flight_wing_aerodynamics,
    apply_flight_air_parameters,
    apply_flight_wing_joint_parameters,
)


class FlyBodyRuntime:
    """Physical FlyBody runtime without a neural motor decoder."""

    def __init__(
        self,
        *,
        tethered: bool = True,
        spawn_position_mm: tuple[float, float, float] = (0.0, 0.0, 4.0),
        flight_body_pitch_deg: float = FLIGHT_BODY_PITCH_DEG,
        initial_linear_velocity_mm_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_wing_phase: float = 0.0,
        wingbeat_hz: float = 218.0,
        enable_vision: bool = False,
        enable_observer_camera: bool = False,
        enable_wing_aerodynamics: bool = True,
        normalize_canonical_mass: bool = True,
        world: BaseWorld | None = None,
    ) -> None:
        if wingbeat_hz <= 0:
            raise ValueError("wingbeat_hz must be positive")
        if not math.isfinite(flight_body_pitch_deg):
            raise ValueError("flight_body_pitch_deg must be finite")
        if not math.isfinite(initial_wing_phase):
            raise ValueError("initial_wing_phase must be finite")
        if tethered and world is not None:
            raise ValueError("custom world is only supported for free-body simulations")

        spawn = np.asarray(spawn_position_mm, dtype=np.float64)
        if spawn.shape != (3,) or not np.all(np.isfinite(spawn)):
            raise ValueError("spawn_position_mm must contain 3 finite values")
        initial_velocity = np.asarray(initial_linear_velocity_mm_s, dtype=np.float64)
        if initial_velocity.shape != (3,) or not np.all(np.isfinite(initial_velocity)):
            raise ValueError("initial_linear_velocity_mm_s must contain 3 finite values")
        if tethered and not np.allclose(initial_velocity, 0.0):
            raise ValueError("initial linear velocity is only valid for free-body simulations")

        self.tethered = bool(tethered)
        self.flight_body_pitch_deg = float(flight_body_pitch_deg)
        self.initial_linear_velocity_mm_s = initial_velocity.copy()
        self.initial_wing_phase = float(initial_wing_phase) % (2.0 * math.pi)
        self.wingbeat_hz = float(wingbeat_hz)
        self.vision_enabled = bool(enable_vision)
        self.wing_aerodynamics_enabled = bool(enable_wing_aerodynamics)
        self.observer_camera_name: str | None = None
        observer_camera_key = "training_view"

        self.fly = FlyBody(name="virtual_fly")
        skeleton = FlyBodySkeleton(
            joint_preset=FlyBodyJointPreset.ALL_BIOLOGICAL,
            axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
        )
        self.fly.add_joints(skeleton, KinematicPosePreset.FLYBODY_NEUTRAL)
        self.joint_range_overrides = apply_researched_joint_ranges(self.fly)
        apply_flight_wing_joint_parameters(self.fly)

        # The current wing actuator path still needs only the six wing POSITION
        # actuators.  Grounded leg/haltere motor output is applied as physical
        # generalized force, so it does not need a synthetic position command.
        add_flight_position_actuators(self.fly, list(skeleton.iter_jointdofs()))
        self.fly.add_tendons()
        self.fly.add_tendon_actuators()

        # FlyGym intentionally keeps FlyBody visual materials opt-in.  Training
        # builds enable compound-eye vision, so leave those models untouched to
        # avoid changing sensory input.  Detached observers and non-sensory
        # diagnostics can safely render the official FlyBody Drosophila visuals.
        if not self.vision_enabled:
            self.fly.colorize()

        if self.wing_aerodynamics_enabled:
            add_flight_wing_aerodynamics(self.fly)
        if self.vision_enabled:
            self.fly.add_vision(draw_sensor_markers=False)
        if enable_observer_camera:
            self.fly.add_tracking_camera(
                name=observer_camera_key,
                mode="track",
                pos_offset=(-3.5, -14.0, 7.0),
                rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 0.45, 0.89)),
                fovy=45.0,
            )

        if self.tethered:
            active_world = TetheredWorld()
            active_world.add_fly(
                self.fly,
                (0.0, 0.0, 0.0),
                Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
            )
        else:
            active_world = world if world is not None else FlatGroundWorld()
            pitch = math.radians(self.flight_body_pitch_deg)
            spawn_rotation = Rotation3D(
                "quat",
                (math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0),
            )
            active_world.add_fly(
                self.fly,
                tuple(float(value) for value in spawn),
                spawn_rotation,
                bodysegs_with_ground_contact=FlyBodyContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
                add_ground_contact_sensors=True,
            )
            add_obstacle_contacts = getattr(active_world, "add_obstacle_contacts", None)
            if add_obstacle_contacts is not None:
                add_obstacle_contacts(self.fly)

        apply_flight_air_parameters(active_world)

        if enable_observer_camera:
            self.observer_camera_name = self.fly.cameraname_to_mjcfcamera[
                observer_camera_key
            ].name

        self.world = active_world
        self.sim = Simulation(active_world, timestep=FLIGHT_PHYSICS_TIMESTEP_S)
        self.sim.reset()
        self.timestep = float(self.sim.mj_model.opt.timestep)
        self.mass_normalization = (
            normalize_fly_mass(self.sim, self.fly) if normalize_canonical_mass else None
        )

        self._all_dofs = list(self.fly.get_jointdofs_order())
        self._dof_by_key = {self._dof_key(dof): dof for dof in self._all_dofs}
        self._actuated_dofs = list(
            self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        )
        all_index = {self._dof_key(dof): i for i, dof in enumerate(self._all_dofs)}
        joint_angles = np.asarray(self.sim.get_joint_angles(self.fly.name), dtype=np.float64)
        self._neutral_target = np.asarray(
            [joint_angles[all_index[self._dof_key(dof)]] for dof in self._actuated_dofs],
            dtype=np.float64,
        )

        self._wing_indices: dict[tuple[str, str], int] = {}
        for actuator_index, dof in enumerate(self._actuated_dofs):
            if not dof.child.is_wing():
                continue
            side = "left" if dof.child.name.startswith("l_") else "right"
            self._wing_indices[(side, dof.axis.value)] = actuator_index

        expected = {
            (side, axis)
            for side in ("left", "right")
            for axis in ("yaw", "roll", "pitch")
        }
        if set(self._wing_indices) != expected or len(self._actuated_dofs) != 6:
            raise RuntimeError(
                "flight runtime requires exactly six wing POSITION actuators; "
                f"actuated={len(self._actuated_dofs)} wings={sorted(self._wing_indices)}"
            )

        body_order = self.fly.get_bodysegs_order()
        self._body_segment_index = {
            segment.name: i for i, segment in enumerate(body_order)
        }
        self._thorax_index = self._body_segment_index["c_thorax"]
        self._time = 0.0
        self._initialize_episode_state()
        self.compiled_biological_joint_ranges = compiled_joint_ranges(self.sim, self.fly)

    @staticmethod
    def _dof_key(dof) -> tuple[str, str, str]:
        return dof.parent.name, dof.child.name, dof.axis.value

    @staticmethod
    def _dof_name(dof) -> str:
        return f"{dof.parent.name}-{dof.child.name}-{dof.axis.value}"

    def _root_freejoint_id(self) -> int:
        joint_types = np.asarray(self.sim.mj_model.jnt_type)
        free_ids = np.flatnonzero(joint_types == int(mj.mjtJoint.mjJNT_FREE))
        if len(free_ids) != 1:
            raise RuntimeError(f"expected one root freejoint, found {len(free_ids)}")
        return int(free_ids[0])

    def _root_freejoint_dof_address(self) -> int:
        return int(self.sim.mj_model.jnt_dofadr[self._root_freejoint_id()])

    def _root_freejoint_qpos_address(self) -> int:
        return int(self.sim.mj_model.jnt_qposadr[self._root_freejoint_id()])

    def _joint_state_addresses(self, dof) -> tuple[int, int]:
        joint_name = self.fly.jointdof_to_mjcfjoint[dof].name
        joint_id = mj.mj_name2id(
            self.sim.mj_model, mj.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise RuntimeError(f"compiled FlyBody joint not found: {joint_name}")
        return (
            int(self.sim.mj_model.jnt_qposadr[joint_id]),
            int(self.sim.mj_model.jnt_dofadr[joint_id]),
        )

    def _wing_state_addresses(self, dof) -> tuple[int, int]:
        return self._joint_state_addresses(dof)

    def joint_dof(self, name: str):
        parts = name.split("-")
        if len(parts) < 3:
            raise KeyError(name)
        key = ("-".join(parts[:-2]), parts[-2], parts[-1])
        # FlyBody segment names themselves contain no hyphen today, but use a
        # direct name scan so future naming does not silently select a wrong DOF.
        for dof in self._all_dofs:
            if self._dof_name(dof) == name:
                return dof
        raise KeyError(f"FlyBody biological DOF not found: {name}")

    def joint_state_addresses(self, name: str) -> tuple[int, int]:
        return self._joint_state_addresses(self.joint_dof(name))

    def joint_range_rad(self, name: str) -> tuple[float, float]:
        dof = self.joint_dof(name)
        joint_name = self.fly.jointdof_to_mjcfjoint[dof].name
        joint_id = mj.mj_name2id(self.sim.mj_model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or not bool(self.sim.mj_model.jnt_limited[joint_id]):
            raise RuntimeError(f"FlyBody joint is not a limited biological hinge: {name}")
        low, high = self.sim.mj_model.jnt_range[joint_id]
        return float(low), float(high)

    def body_segment_position_mm(self, name: str) -> np.ndarray:
        try:
            index = self._body_segment_index[name]
        except KeyError as exc:
            raise KeyError(f"FlyBody segment not found: {name}") from exc
        return np.asarray(
            self.sim.get_body_positions(self.fly.name)[index], dtype=np.float64
        ).copy()

    def set_root_position_mm(
        self, position: np.ndarray | tuple[float, float, float]
    ) -> None:
        if self.tethered:
            raise RuntimeError("tethered FlyBody has no free root position")
        vector = np.asarray(position, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("root position must contain 3 finite values")
        qpos_address = self._root_freejoint_qpos_address()
        self.sim.mj_data.qpos[qpos_address : qpos_address + 3] = vector
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

    def set_root_linear_velocity_mm_s(
        self, velocity: np.ndarray | tuple[float, float, float]
    ) -> None:
        if self.tethered:
            raise RuntimeError("tethered FlyBody has no free root velocity")
        vector = np.asarray(velocity, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("root linear velocity must contain 3 finite values")
        dof_address = self._root_freejoint_dof_address()
        self.sim.mj_data.qvel[dof_address : dof_address + 3] = vector
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

    def root_linear_velocity_mm_s(self) -> np.ndarray:
        if self.tethered:
            raise RuntimeError("tethered FlyBody has no free root velocity")
        dof_address = self._root_freejoint_dof_address()
        return np.asarray(
            self.sim.mj_data.qvel[dof_address : dof_address + 3], dtype=np.float64
        ).copy()

    @staticmethod
    def _wing_pattern(phase: float, amplitude_scale: float) -> dict[str, float]:
        phase = phase % (2.0 * math.pi)
        return {
            "yaw": 0.3 + amplitude_scale * 1.1 * math.sin(phase - math.pi / 2.0),
            "roll": -0.1 + amplitude_scale * 0.25 * math.sin(1.5 * phase),
            "pitch": 0.8 + amplitude_scale * 1.35 * math.sin(phase),
        }

    def _wing_pattern_velocity(
        self, phase: float, amplitude_scale: float
    ) -> dict[str, float]:
        phase = phase % (2.0 * math.pi)
        omega = 2.0 * math.pi * self.wingbeat_hz
        return {
            "yaw": amplitude_scale * 1.1 * math.cos(phase - math.pi / 2.0) * omega,
            "roll": amplitude_scale * 0.25 * 1.5 * math.cos(1.5 * phase) * omega,
            "pitch": amplitude_scale * 1.35 * math.cos(phase) * omega,
        }

    def _initialize_episode_state(self) -> None:
        """Place wings on the analytic cycle and apply the one-shot root velocity."""

        phase = self.initial_wing_phase
        target = self._neutral_target.copy()
        pattern = self._wing_pattern(phase, 1.0)
        velocity = self._wing_pattern_velocity(phase, 1.0)
        for side in ("left", "right"):
            for axis in ("yaw", "roll", "pitch"):
                actuator_index = self._wing_indices[(side, axis)]
                dof = self._actuated_dofs[actuator_index]
                qpos_address, qvel_address = self._wing_state_addresses(dof)
                self.sim.mj_data.qpos[qpos_address] = pattern[axis]
                self.sim.mj_data.qvel[qvel_address] = velocity[axis]
                target[actuator_index] = pattern[axis]

        if not self.tethered:
            root_dof = self._root_freejoint_dof_address()
            self.sim.mj_data.qvel[root_dof : root_dof + 3] = self.initial_linear_velocity_mm_s

        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, target)
        self._time = phase / (2.0 * math.pi * self.wingbeat_hz)
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

    def reset(self) -> None:
        self.sim.reset()
        self._initialize_episode_state()

    def thorax_position_mm(self) -> np.ndarray:
        return np.asarray(
            self.sim.get_body_positions(self.fly.name)[self._thorax_index],
            dtype=np.float64,
        ).copy()

    def ommatidia_readouts(self) -> np.ndarray:
        if not self.vision_enabled:
            raise RuntimeError("vision was not enabled for this FlyBody instance")
        return np.asarray(
            self.sim.get_ommatidia_readouts(self.fly.name), dtype=np.float32
        )

    def wing_joint_angles_rad(self) -> dict[str, float]:
        all_angles = np.asarray(self.sim.get_joint_angles(self.fly.name), dtype=np.float64)
        all_index = {self._dof_key(dof): i for i, dof in enumerate(self._all_dofs)}
        result: dict[str, float] = {}
        for side, axis in sorted(self._wing_indices):
            actuator_idx = self._wing_indices[(side, axis)]
            dof = self._actuated_dofs[actuator_idx]
            result[f"{side}_{axis}"] = float(
                all_angles[all_index[self._dof_key(dof)]]
            )
        return result