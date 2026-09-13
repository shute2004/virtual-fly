#!/usr/bin/env python3
"""Biologically motivated, deliberately thin CNS -> FlyBody motor adapter.

The adapter does not decide whether the fly should climb, descend, or avoid an
obstacle. It only maps bilateral DNg02 population activity to modulation of a
nominal wing-beat pattern, consistent with experiments showing that DNg02
population activity regulates wing stroke amplitude and thrust.

The nominal analytic wing beat is a prototype pattern adapted from the public
FlyBody test pattern. Its exact kinematics are a calibration parameter, not a
claim about measured D. melanogaster wing motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    FlyBodyActuatedDOFPreset,
    FlyBodyAxisOrder,
    FlyBodyContactBodiesPreset,
    FlyBodyJointPreset,
    FlyBodySkeleton,
)
from flygym.utils.math import Rotation3D


@dataclass(frozen=True)
class WingDrive:
    left: float
    right: float


class FlyBodyWingAdapter:
    def __init__(
        self,
        *,
        tethered: bool = True,
        spawn_position_mm: tuple[float, float, float] = (0.0, 0.0, 4.0),
        wingbeat_hz: float = 200.0,
        dng02_gain: float = 0.35,
        min_scale: float = 0.65,
        max_scale: float = 1.45,
        enable_vision: bool = False,
        enable_observer_camera: bool = False,
        world: BaseWorld | None = None,
    ) -> None:
        if wingbeat_hz <= 0:
            raise ValueError("wingbeat_hz must be positive")
        if not 0 < min_scale <= max_scale:
            raise ValueError("invalid wing amplitude scale limits")
        if tethered and world is not None:
            raise ValueError("custom world is only supported for free-body simulations")

        self.wingbeat_hz = float(wingbeat_hz)
        self.dng02_gain = float(dng02_gain)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.vision_enabled = bool(enable_vision)
        self.observer_camera_name = "training_view" if enable_observer_camera else None

        self.fly = FlyBody(name="virtual_fly")
        skeleton = FlyBodySkeleton(
            joint_preset=FlyBodyJointPreset.ALL_BIOLOGICAL,
            axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
        )
        self.fly.add_joints(skeleton, KinematicPosePreset.FLYBODY_NEUTRAL)
        actuated_dofs = skeleton.get_actuated_dofs_from_preset(
            FlyBodyActuatedDOFPreset.ALL
        )
        self.fly.add_actuators(actuated_dofs, ActuatorType.POSITION)
        self.fly.add_tendons()
        self.fly.add_tendon_actuators()
        if self.vision_enabled:
            self.fly.add_vision(draw_sensor_markers=False)
        if enable_observer_camera:
            self.fly.add_tracking_camera(
                name=self.observer_camera_name,
                mode="track",
                pos_offset=(-3.5, -14.0, 7.0),
                rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 0.45, 0.89)),
                fovy=45.0,
            )

        if tethered:
            active_world = TetheredWorld()
            active_world.add_fly(
                self.fly,
                (0.0, 0.0, 0.0),
                Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
            )
        else:
            active_world = world if world is not None else FlatGroundWorld()
            active_world.add_fly(
                self.fly,
                spawn_position_mm,
                Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
                bodysegs_with_ground_contact=FlyBodyContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
                add_ground_contact_sensors=True,
            )
            add_obstacle_contacts = getattr(active_world, "add_obstacle_contacts", None)
            if add_obstacle_contacts is not None:
                add_obstacle_contacts(self.fly)

        self.world = active_world
        self.sim = Simulation(active_world)
        self.sim.reset()
        self.timestep = float(self.sim.mj_model.opt.timestep)
        self._time = 0.0

        self._all_dofs = list(self.fly.get_jointdofs_order())
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
        if set(self._wing_indices) != expected:
            raise RuntimeError(
                f"unexpected FlyBody wing actuator set: {sorted(self._wing_indices)}"
            )

        body_order = self.fly.get_bodysegs_order()
        self._thorax_index = next(
            i for i, segment in enumerate(body_order) if segment.name == "c_thorax"
        )

    @staticmethod
    def _dof_key(dof) -> tuple[str, str, str]:
        return dof.parent.name, dof.child.name, dof.axis.value

    def reset(self) -> None:
        self.sim.reset()
        self._time = 0.0

    def _scale_from_activity(self, spike_fraction: float) -> float:
        if not math.isfinite(spike_fraction):
            raise ValueError("DNg02 spike fraction must be finite")
        activity = min(1.0, max(0.0, spike_fraction))
        return min(self.max_scale, max(self.min_scale, 1.0 + self.dng02_gain * activity))

    @staticmethod
    def _wing_pattern(phase: float, amplitude_scale: float) -> dict[str, float]:
        return {
            "yaw": 0.3 + amplitude_scale * 1.1 * math.sin(phase - math.pi / 2.0),
            "roll": -0.1 + amplitude_scale * 0.25 * math.sin(1.5 * phase),
            "pitch": 0.8 + amplitude_scale * 1.35 * math.sin(phase),
        }

    def step(self, drive: WingDrive, *, physics_steps: int = 1) -> None:
        if physics_steps < 1:
            raise ValueError("physics_steps must be >= 1")
        left_scale = self._scale_from_activity(drive.left)
        right_scale = self._scale_from_activity(drive.right)

        for _ in range(physics_steps):
            phase = 2.0 * math.pi * self.wingbeat_hz * self._time
            target = self._neutral_target.copy()
            for side, scale in (("left", left_scale), ("right", right_scale)):
                pattern = self._wing_pattern(phase, scale)
                for axis, angle in pattern.items():
                    target[self._wing_indices[(side, axis)]] = angle
            self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, target)
            self.sim.step()
            self._time += self.timestep

    def thorax_position_mm(self) -> np.ndarray:
        return np.asarray(
            self.sim.get_body_positions(self.fly.name)[self._thorax_index],
            dtype=np.float64,
        ).copy()

    def ommatidia_readouts(self) -> np.ndarray:
        if not self.vision_enabled:
            raise RuntimeError("vision was not enabled for this FlyBody instance")
        return np.asarray(self.sim.get_ommatidia_readouts(self.fly.name), dtype=np.float32)

    def wing_joint_angles_rad(self) -> dict[str, float]:
        all_angles = np.asarray(self.sim.get_joint_angles(self.fly.name), dtype=np.float64)
        all_index = {self._dof_key(dof): i for i, dof in enumerate(self._all_dofs)}
        result: dict[str, float] = {}
        for side, axis in sorted(self._wing_indices):
            actuator_idx = self._wing_indices[(side, axis)]
            dof = self._actuated_dofs[actuator_idx]
            result[f"{side}_{axis}"] = float(all_angles[all_index[self._dof_key(dof)]])
        return result
