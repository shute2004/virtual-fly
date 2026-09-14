#!/usr/bin/env python3
"""FlyBody mechanics driven by individual-MN-derived peripheral muscle state.

This is a virtual-muscle compatibility layer for FlyBody's six idealized wing
joint DOFs. FlyBody does not currently expose the anatomical Drosophila wing
muscles, so the peripheral muscle activations are converted to joint torque here
rather than being converted to an action or position command.

The power-muscle path uses an autonomous thoracic wingbeat phase because
Drosophila DLM/DVM indirect flight muscles are asynchronous: low-frequency motor
input maintains calcium while stretch activation and thoracic resonance produce
high-frequency wingbeats. The oscillator therefore belongs to the peripheral
mechanical approximation, not to the CNS action-selection path.

Only steering effects with a reasonably constrained qualitative sign are active
in this first boundary: b1/b2 increase stroke-amplitude drive, b3 opposes the
basalar pair, and i1 reduces stroke-amplitude drive. Other identified muscles
still maintain independent neuromuscular states but exert no guessed torque yet.
All numerical gains and the mapping between DLM/DVM activation and the virtual
hinge phase are explicitly bootstrap body-interface assumptions rather than
biological measurements.

Crucially, these defaults are not selected from prior Flyppy trajectories. An
untrained CNS is allowed to produce unstable flight and collide; the body layer
must not absorb neural mistakes by retuning itself to previous task outcomes.
"""

from __future__ import annotations

import math

import numpy as np

from flygym.compose import ActuatorType

from flybody_adapter import FlyBodyWingAdapter
from flybody_flight_physics import FLIGHT_WING_POSITION_KP
from wing_muscle_periphery import PeripheralSnapshot


# Calibrated virtual moment effects, expressed as a fraction of the nominal yaw
# stroke waveform. Sign constraints follow known basalar antagonism and the
# reported stroke-amplitude effect of i1. These are not measured moment arms.
STEERING_STROKE_EFFECT = {
    "b1": +0.18,
    "b2": +0.24,
    "b3": -0.18,
    "i1": -0.12,
}

# Fixed bootstrap seam. Keep this independent of CNS/Flyppy outcomes. The
# body-only free-flight envelope preflight verifies that these values admit
# physically viable operating points before neural learning starts.
BOOTSTRAP_POWER_GAIN = 1.0
BOOTSTRAP_STEERING_GAIN = 1.0
BOOTSTRAP_SWAP_DLM_DVM_PHASE = False


class FlyBodyMuscleAdapter(FlyBodyWingAdapter):
    """Apply peripheral muscle state as physical torque on FlyBody wing DOFs."""

    def __init__(
        self,
        *args,
        virtual_power_kp: float = FLIGHT_WING_POSITION_KP,
        virtual_power_kd: float = 0.08,
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
        self.virtual_power_kp = float(virtual_power_kp)
        self.virtual_power_kd = float(virtual_power_kd)
        self.power_gain = float(power_gain)
        self.steering_gain = float(steering_gain)
        self.swap_dlm_dvm_phase = bool(swap_dlm_dvm_phase)
        self.max_abs_torque = float(max_abs_torque)
        self.last_wing_torque: dict[str, float] = {
            f"{side}:{axis}": 0.0
            for side in ("left", "right")
            for axis in ("yaw", "roll", "pitch")
        }

    def _neutralize_position_actuators(self) -> None:
        """Set each idealized position actuator to its current joint position.

        This removes the legacy DNg02/analytic position command while preserving
        the compiled FlyBody actuator layout. Wing motion for this step then comes
        from qfrc_applied virtual-muscle torque plus passive joint/aero mechanics.
        """

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
        """Return yaw angle/velocity offsets from explicitly mapped steering units."""

        scale = 0.0
        for muscle, effect in STEERING_STROKE_EFFECT.items():
            scale += effect * state.muscle(side, muscle)
        scale *= self.steering_gain

        # Same mechanical stroke coordinate as FlyBody's published prototype
        # wingbeat, but used only as a virtual moment-arm waveform.
        stroke = 1.1 * math.sin(phase - math.pi / 2.0)
        omega = 2.0 * math.pi * self.wingbeat_hz
        stroke_velocity = 1.1 * math.cos(phase - math.pi / 2.0) * omega
        return scale * stroke, scale * stroke_velocity

    def _power_activation(
        self, state: PeripheralSnapshot, side: str, phase: float
    ) -> float:
        # DLM and DVM are antagonistic asynchronous power-muscle groups. The
        # mapping to this abstract hinge oscillator is a bootstrap coordinate
        # convention, not a measured neural phase. It is intentionally fixed
        # during learning and is never chosen from prior task performance.
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

    def step_muscles(
        self, state: PeripheralSnapshot, *, physics_steps: int = 1
    ) -> None:
        if physics_steps < 1:
            raise ValueError("physics_steps must be >= 1")
        if state.selected_motor_units < 1:
            raise ValueError("peripheral state contains no motor units")

        for _ in range(physics_steps):
            phase = (2.0 * math.pi * self.wingbeat_hz * self._time) % (
                2.0 * math.pi
            )
            base_target = self._wing_pattern(phase, 1.0)
            base_velocity = self._wing_pattern_velocity(phase, 1.0)
            self._neutralize_position_actuators()
            self.sim.mj_data.qfrc_applied[:] = 0.0

            step_torque: dict[str, float] = {}
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
                    self.sim.mj_data.qfrc_applied[qvel_address] += torque
                    step_torque[f"{side}:{axis}"] = torque

            if not np.all(np.isfinite(self.sim.mj_data.qfrc_applied)):
                raise RuntimeError("non-finite virtual-muscle torque")
            self.sim.step()
            self.sim.mj_data.qfrc_applied[:] = 0.0
            self._time += self.timestep
            self.last_wing_torque = step_torque

    def reset(self) -> None:
        super().reset()
        self.last_wing_torque = {
            f"{side}:{axis}": 0.0
            for side in ("left", "right")
            for axis in ("yaw", "roll", "pitch")
        }
