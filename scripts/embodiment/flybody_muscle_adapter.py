#!/usr/bin/env python3
"""FlyBody mechanics driven by individual-MN-derived peripheral muscle state.

Released wing motor-neuron spikes feed ``WingMusclePeriphery`` and the resulting
motor-unit/muscle state produces physical torque on FlyBody wing DOFs.  The base
class also exposes a per-physics-step hook for anatomically grounded non-wing
muscles; the wing-only v1 path leaves that hook empty.
"""

from __future__ import annotations

import math

import numpy as np

from flygym.compose import ActuatorType

from flybody_flight_physics import FLIGHT_WING_POSITION_KP
from flybody_runtime import FlyBodyRuntime
from wing_muscle_periphery import PeripheralSnapshot


STEERING_STROKE_EFFECT = {
    "b1": +0.18,
    "b2": +0.24,
    "b3": -0.18,
    "i1": -0.12,
}

BOOTSTRAP_POWER_GAIN = 1.0
BOOTSTRAP_STEERING_GAIN = 1.0
BOOTSTRAP_SWAP_DLM_DVM_PHASE = False


class FlyBodyMuscleAdapter(FlyBodyRuntime):
    """Apply peripheral muscle state as physical torque on FlyBody wing DOFs."""

    def __init__(
        self,
        *args,
        virtual_power_kp: float = FLIGHT_WING_POSITION_KP,
        virtual_power_kd: float = 0.08,
        virtual_control_timestep_s: float | None = None,
        power_gain: float = BOOTSTRAP_POWER_GAIN,
        steering_gain: float = BOOTSTRAP_STEERING_GAIN,
        swap_dlm_dvm_phase: bool = BOOTSTRAP_SWAP_DLM_DVM_PHASE,
        max_abs_torque: float = 2500.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if virtual_power_kp <= 0.0 or virtual_power_kd < 0.0:
            raise ValueError("virtual muscle gains are invalid")
        if power_gain < 0.0 or steering_gain < 0.0 or max_abs_torque <= 0.0:
            raise ValueError("power_gain/steering_gain/max_abs_torque are invalid")

        control_timestep = (
            self.timestep
            if virtual_control_timestep_s is None
            else float(virtual_control_timestep_s)
        )
        if control_timestep <= 0.0:
            raise ValueError("virtual_control_timestep_s must be positive")
        stride = control_timestep / self.timestep
        if not math.isclose(stride, round(stride), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "virtual_control_timestep_s must be an integer multiple of the "
                f"physics timestep ({self.timestep})"
            )

        self.virtual_power_kp = float(virtual_power_kp)
        self.virtual_power_kd = float(virtual_power_kd)
        self.virtual_control_timestep_s = control_timestep
        self._virtual_control_stride = int(round(stride))
        self._virtual_control_phase = 0
        self.power_gain = float(power_gain)
        self.steering_gain = float(steering_gain)
        self.swap_dlm_dvm_phase = bool(swap_dlm_dvm_phase)
        self.max_abs_torque = float(max_abs_torque)
        self.last_wing_torque: dict[str, float] = self._zero_wing_torque()
        self._held_wing_torque_by_dof: dict[int, float] = {}

    @staticmethod
    def _zero_wing_torque() -> dict[str, float]:
        return {
            f"{side}:{axis}": 0.0
            for side in ("left", "right")
            for axis in ("yaw", "roll", "pitch")
        }

    @staticmethod
    def _wing_snapshot(state) -> PeripheralSnapshot:
        if isinstance(state, PeripheralSnapshot):
            return state
        wing = getattr(state, "wing", None)
        if isinstance(wing, PeripheralSnapshot):
            return wing
        raise TypeError("peripheral state does not expose a WingMusclePeriphery snapshot")

    def _neutralize_position_actuators(self) -> None:
        target = self._neutral_target.copy()
        for side in ("left", "right"):
            for axis in ("yaw", "roll", "pitch"):
                actuator_index = self._wing_indices[(side, axis)]
                dof = self._actuated_dofs[actuator_index]
                qpos_address, _ = self._wing_state_addresses(dof)
                target[actuator_index] = float(self.sim.mj_data.qpos[qpos_address])
        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, target)

    def _steering_stroke_offset(
        self, state: PeripheralSnapshot, side: str, phase: float
    ) -> tuple[float, float]:
        scale = 0.0
        for muscle, effect in STEERING_STROKE_EFFECT.items():
            scale += effect * state.muscle(side, muscle)
        scale *= self.steering_gain

        stroke = 1.1 * math.sin(phase - math.pi / 2.0)
        omega = 2.0 * math.pi * self.wingbeat_hz
        stroke_velocity = 1.1 * math.cos(phase - math.pi / 2.0) * omega
        return scale * stroke, scale * stroke_velocity

    def _power_activation(
        self, state: PeripheralSnapshot, side: str, phase: float
    ) -> float:
        positive_half_cycle = math.sin(phase) >= 0.0
        if self.swap_dlm_dvm_phase:
            activation = (
                state.power_dvm(side) if positive_half_cycle else state.power_dlm(side)
            )
        else:
            activation = (
                state.power_dlm(side) if positive_half_cycle else state.power_dvm(side)
            )
        return self.power_gain * activation

    def _compute_wing_torque(
        self, state: PeripheralSnapshot, phase: float
    ) -> tuple[dict[str, float], dict[int, float]]:
        base_target = self._wing_pattern(phase, 1.0)
        base_velocity = self._wing_pattern_velocity(phase, 1.0)
        labelled: dict[str, float] = {}
        by_dof: dict[int, float] = {}

        for side in ("left", "right"):
            power = self._power_activation(state, side, phase)
            steering_yaw, steering_yaw_velocity = self._steering_stroke_offset(
                state, side, phase
            )
            for axis in ("yaw", "roll", "pitch"):
                actuator_index = self._wing_indices[(side, axis)]
                dof = self._actuated_dofs[actuator_index]
                qpos_address, qvel_address = self._wing_state_addresses(dof)
                angle = float(self.sim.mj_data.qpos[qpos_address])
                velocity = float(self.sim.mj_data.qvel[qvel_address])

                target_angle = float(base_target[axis])
                target_velocity = float(base_velocity[axis])
                steering_angle = 0.0
                steering_velocity = 0.0
                if axis == "yaw":
                    steering_angle = steering_yaw
                    steering_velocity = steering_yaw_velocity

                power_torque = power * (
                    self.virtual_power_kp * (target_angle - angle)
                    + self.virtual_power_kd * (target_velocity - velocity)
                )
                steering_torque = self.virtual_power_kp * steering_angle
                steering_torque += self.virtual_power_kd * steering_velocity
                torque = power_torque + steering_torque
                torque = min(self.max_abs_torque, max(-self.max_abs_torque, torque))
                labelled[f"{side}:{axis}"] = torque
                by_dof[qvel_address] = by_dof.get(qvel_address, 0.0) + torque

        return labelled, by_dof

    def _apply_additional_muscle_torque(self, state, phase: float) -> dict[str, float]:
        """Hook for grounded non-wing muscles; v1 wing-only path intentionally empty."""
        return {}

    def step_muscles(self, state, *, physics_steps: int = 1) -> None:
        if physics_steps < 1:
            raise ValueError("physics_steps must be >= 1")
        if int(getattr(state, "selected_motor_units", 0)) < 1:
            raise ValueError("peripheral state contains no motor units")
        wing_state = self._wing_snapshot(state)

        for _ in range(physics_steps):
            phase = (2.0 * math.pi * self.wingbeat_hz * self._time) % (
                2.0 * math.pi
            )
            self._neutralize_position_actuators()
            self.sim.mj_data.qfrc_applied[:] = 0.0

            # The physical integrator may run faster than the muscle command seam.
            # Recompute the wing command only at the configured control cadence and
            # hold it between updates.  ``None`` retains the historical one-update-
            # per-physics-step behavior used by v1/v2; Flyppy v3 explicitly selects
            # the upstream FlyBody 0.2 ms flight-control cadence.
            if self._virtual_control_phase == 0:
                (
                    self.last_wing_torque,
                    self._held_wing_torque_by_dof,
                ) = self._compute_wing_torque(wing_state, phase)

            for qvel_address, torque in self._held_wing_torque_by_dof.items():
                self.sim.mj_data.qfrc_applied[qvel_address] += torque

            self._apply_additional_muscle_torque(state, phase)
            if not np.all(np.isfinite(self.sim.mj_data.qfrc_applied)):
                raise RuntimeError("non-finite virtual-muscle torque")
            self.sim.step()
            self.sim.mj_data.qfrc_applied[:] = 0.0
            self._time += self.timestep
            self._virtual_control_phase = (
                self._virtual_control_phase + 1
            ) % self._virtual_control_stride

    def reset(self) -> None:
        super().reset()
        self.last_wing_torque = self._zero_wing_torque()
        self._held_wing_torque_by_dof = {}
        self._virtual_control_phase = 0
