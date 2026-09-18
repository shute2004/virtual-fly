#!/usr/bin/env python3
"""Separate measured b2/hg3 wing-hinge modes on top of FlyBody v4.

v5 compressed b2 and hg3 recruitment into one shared vertical-force mode.  The
published RoboFly measurements show that the two muscles have opposite pitch-
torque derivatives, so that compression discards potentially important attitude
information.  v6 keeps their measured CNN-derived wing-kinematic modes separate:

    MaleCNS b2 activation  -> b2 measured yaw/pitch mode
    MaleCNS hg3 activation -> hg3 measured yaw/pitch mode

The modes are summed at the wing joints and then clipped only by FlyBody's source
joint ranges.  No course state, target state, body attitude, or reward enters
this mapping.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from flybody_muscle_adapter import STEERING_STROKE_EFFECT
from flybody_v4_adapter import FlyBodyV4MuscleAdapter, FlyBodyV4NeuromuscularAdapter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEPARATE_MODES = ROOT / "artifacts/embodiment/wing-steering-b2-hg3-modes-v1.json"


class _PeriodicMode:
    def __init__(self, yaw: list[float], pitch: list[float]) -> None:
        if len(yaw) < 8 or len(yaw) != len(pitch):
            raise ValueError("periodic steering mode requires equal yaw/pitch arrays")
        self.yaw = tuple(float(v) for v in yaw)
        self.pitch = tuple(float(v) for v in pitch)
        self.count = len(self.yaw)
        dphase = 2.0 * math.pi / self.count
        self.yaw_dphase = tuple(
            (self.yaw[(i + 1) % self.count] - self.yaw[(i - 1) % self.count]) / (2.0 * dphase)
            for i in range(self.count)
        )
        self.pitch_dphase = tuple(
            (self.pitch[(i + 1) % self.count] - self.pitch[(i - 1) % self.count]) / (2.0 * dphase)
            for i in range(self.count)
        )

    def _interp(self, values: tuple[float, ...], phase: float) -> float:
        u = (phase % (2.0 * math.pi)) * self.count / (2.0 * math.pi)
        i0 = int(math.floor(u)) % self.count
        frac = u - math.floor(u)
        return (1.0 - frac) * values[i0] + frac * values[(i0 + 1) % self.count]

    def sample(self, phase: float) -> tuple[float, float, float, float]:
        return (
            self._interp(self.yaw, phase),
            self._interp(self.pitch, phase),
            self._interp(self.yaw_dphase, phase),
            self._interp(self.pitch_dphase, phase),
        )


class _SeparateMeasuredSteeringMixin:
    def __init__(
        self,
        *args,
        separate_steering_modes: Path = DEFAULT_SEPARATE_MODES,
        measured_steering_gain: float = 1.0,
        **kwargs,
    ) -> None:
        if not math.isfinite(measured_steering_gain) or not 0.0 <= measured_steering_gain <= 1.0:
            raise ValueError("measured_steering_gain must be finite and in [0,1]")
        super().__init__(*args, **kwargs)
        payload = json.loads(Path(separate_steering_modes).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("separate steering mode schema 1 is required")
        modes = payload["modes"]
        self._measured_modes = {
            name: _PeriodicMode(
                modes[name]["yaw_delta_rad"],
                modes[name]["pitch_delta_rad"],
            )
            for name in ("b2", "hg3")
        }
        self.measured_steering_gain = float(measured_steering_gain)
        self.last_measured_steering_activation = {
            "left": {"b2": 0.0, "hg3": 0.0},
            "right": {"b2": 0.0, "hg3": 0.0},
        }
        self.flight_physics_version = "v6-aifm-plus-separate-b2-hg3"

    def _remaining_legacy_steering_offset(self, wing, side: str, phase: float) -> tuple[float, float]:
        # b2 is represented by its measured multi-axis mode below. b1/b3/i1 remain
        # on the historical seam until their own measured modes are validated.
        scale = 0.0
        for muscle in ("b1", "b3", "i1"):
            scale += STEERING_STROKE_EFFECT[muscle] * wing.muscle(side, muscle)
        scale *= self.steering_gain
        stroke = 1.1 * math.sin(phase - math.pi / 2.0)
        omega = 2.0 * math.pi * self.wingbeat_hz
        stroke_velocity = 1.1 * math.cos(phase - math.pi / 2.0) * omega
        return scale * stroke, scale * stroke_velocity

    def _joint_range(self, actuator_index: int) -> tuple[float, float]:
        joint_id = int(self.sim.mj_model.actuator_trnid[actuator_index, 0])
        low, high = self.sim.mj_model.jnt_range[joint_id]
        return float(low), float(high)

    def _compute_wing_torque(self, state, phase: float):
        wing = self._wing_snapshot(state)
        base_target = self._wing_pattern(phase, 1.0)
        base_velocity = self._wing_pattern_velocity(phase, 1.0)
        labelled: dict[str, float] = {}
        by_dof: dict[int, float] = {}
        omega = 2.0 * math.pi * self.wingbeat_hz

        for side in ("left", "right"):
            power = self._power_activation(wing, side, phase)
            stroke_scale = self._stroke_amplitude_scale(wing, side)
            activations = {
                name: self.measured_steering_gain * min(1.0, max(0.0, wing.muscle(side, name)))
                for name in ("b2", "hg3")
            }
            self.last_measured_steering_activation[side] = dict(activations)
            measured_angle = {"yaw": 0.0, "roll": 0.0, "pitch": 0.0}
            measured_velocity = {"yaw": 0.0, "roll": 0.0, "pitch": 0.0}
            for name, activation in activations.items():
                dyaw, dpitch, dyaw_dphase, dpitch_dphase = self._measured_modes[name].sample(phase)
                measured_angle["yaw"] += activation * dyaw
                measured_angle["pitch"] += activation * dpitch
                measured_velocity["yaw"] += activation * dyaw_dphase * omega
                measured_velocity["pitch"] += activation * dpitch_dphase * omega

            legacy_yaw, legacy_yaw_velocity = self._remaining_legacy_steering_offset(
                wing, side, phase
            )
            for axis in ("yaw", "roll", "pitch"):
                actuator_index = self._wing_indices[(side, axis)]
                dof = self._actuated_dofs[actuator_index]
                qpos_address, qvel_address = self._wing_state_addresses(dof)
                angle = float(self.sim.mj_data.qpos[qpos_address])
                velocity = float(self.sim.mj_data.qvel[qvel_address])

                target_angle = float(base_target[axis])
                target_velocity = float(base_velocity[axis])
                if axis == "yaw":
                    target_angle = self._measured_stroke_mean + stroke_scale * (
                        target_angle - self._measured_stroke_mean
                    )
                    target_velocity *= stroke_scale
                target_angle += measured_angle[axis]
                target_velocity += measured_velocity[axis]

                low, high = self._joint_range(actuator_index)
                clipped = min(high, max(low, target_angle))
                if clipped != target_angle:
                    target_angle = clipped
                    target_velocity = 0.0

                steering_angle = legacy_yaw if axis == "yaw" else 0.0
                steering_velocity = legacy_yaw_velocity if axis == "yaw" else 0.0
                power_torque = power * (
                    self.virtual_power_kp * (target_angle - angle)
                    + self.virtual_power_kd * (target_velocity - velocity)
                )
                steering_torque = self.virtual_power_kp * steering_angle
                steering_torque += self.virtual_power_kd * steering_velocity
                torque = power_torque + steering_torque
                if math.isfinite(self.max_abs_torque):
                    torque = min(self.max_abs_torque, max(-self.max_abs_torque, torque))
                labelled[f"{side}:{axis}"] = torque
                by_dof[qvel_address] = by_dof.get(qvel_address, 0.0) + torque
        return labelled, by_dof

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "last_measured_steering_activation"):
            self.last_measured_steering_activation = {
                "left": {"b2": 0.0, "hg3": 0.0},
                "right": {"b2": 0.0, "hg3": 0.0},
            }


class FlyBodyV6MuscleAdapter(_SeparateMeasuredSteeringMixin, FlyBodyV4MuscleAdapter):
    """A-IFM power plus separate published b2 and hg3 wing-hinge modes."""


class FlyBodyV6NeuromuscularAdapter(
    _SeparateMeasuredSteeringMixin,
    FlyBodyV4NeuromuscularAdapter,
):
    """Whole-body v6 seam with the existing MaleCNS motor boundary."""
