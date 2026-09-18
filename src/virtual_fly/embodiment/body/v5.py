#!/usr/bin/env python3
"""Measured steering-mode extension of the A-IFM FlyBody v4 seam.

v5 keeps v4's power-muscle -> above-hover stroke modulation and adds one
published, data-derived wing-hinge mode for vertical-force control.  The mode is
constructed offline from Melis, Siwanowicz & Dickinson (Nature 2024): the Fig. 4
b2 and hg3 virtual-muscle experiments are weighted by their measured RoboFly
vertical-force sensitivities and reduced to the wing coordinates that currently
exist in FlyBody (stroke/yaw and wing rotation/pitch).

No force or task signal is injected here.  Identified MaleCNS b2/hg3 motor output
selects a periodic wing-kinematic perturbation; FlyBody's own aerodynamics then
generates the resulting force.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from virtual_fly.embodiment.body.muscle import STEERING_STROKE_EFFECT
from virtual_fly.embodiment.body.v4 import FlyBodyV4MuscleAdapter, FlyBodyV4NeuromuscularAdapter


from virtual_fly.paths import REPO_ROOT

ROOT = REPO_ROOT
DEFAULT_VERTICAL_STEERING_MODE = (
    ROOT / "artifacts/embodiment/wing-steering-vertical-mode-v1.json"
)


class _PeriodicVerticalSteeringMode:
    def __init__(self, path: Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("wing steering vertical mode schema 1 is required")
        yaw = tuple(float(v) for v in payload.get("yaw_delta_rad", ()))
        pitch = tuple(float(v) for v in payload.get("pitch_delta_rad", ()))
        if len(yaw) < 8 or len(yaw) != len(pitch):
            raise ValueError("wing steering mode must contain equal periodic yaw/pitch arrays")
        if any(not math.isfinite(v) for v in (*yaw, *pitch)):
            raise ValueError("wing steering mode contains non-finite values")
        self.yaw = yaw
        self.pitch = pitch
        self.count = len(yaw)
        dphase = 2.0 * math.pi / self.count
        self.yaw_dphase = tuple(
            (yaw[(i + 1) % self.count] - yaw[(i - 1) % self.count]) / (2.0 * dphase)
            for i in range(self.count)
        )
        self.pitch_dphase = tuple(
            (pitch[(i + 1) % self.count] - pitch[(i - 1) % self.count]) / (2.0 * dphase)
            for i in range(self.count)
        )
        coeffs = payload.get("robo_fly_vertical_coefficients", {})
        self.b2_weight = float(coeffs.get("b2", 0.81548))
        self.hg3_weight = float(coeffs.get("hg3", 1.23604))
        if self.b2_weight <= 0.0 or self.hg3_weight <= 0.0:
            raise ValueError("vertical steering mode weights must be positive")

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


class _MeasuredVerticalSteeringMixin:
    def __init__(
        self,
        *args,
        vertical_steering_mode: Path = DEFAULT_VERTICAL_STEERING_MODE,
        vertical_steering_gain: float = 1.0,
        **kwargs,
    ) -> None:
        if not math.isfinite(vertical_steering_gain) or not 0.0 <= vertical_steering_gain <= 1.0:
            raise ValueError("vertical_steering_gain must be finite and in [0,1]")
        super().__init__(*args, **kwargs)
        self._vertical_steering = _PeriodicVerticalSteeringMode(vertical_steering_mode)
        self.vertical_steering_gain = float(vertical_steering_gain)
        self.last_vertical_steering_activation = {"left": 0.0, "right": 0.0}
        self.last_vertical_steering_offset = {
            "left": {"yaw": 0.0, "pitch": 0.0},
            "right": {"yaw": 0.0, "pitch": 0.0},
        }
        self.flight_physics_version = "v5-aifm-plus-measured-vertical-steering"

    def _vertical_mode_activation(self, wing, side: str) -> float:
        # The v5 body interprets the existing 0..1 peripheral steering state as
        # recruitment above the tonic activity used to define the published
        # baseline: 0 leaves the measured FlyBody cycle unchanged, 1 reaches the
        # published maximum-direction virtual experiment.  This is an explicit
        # calibration assumption; no Flyppy state or reward enters this mapping.
        b2 = min(1.0, max(0.0, wing.muscle(side, "b2")))
        hg3 = min(1.0, max(0.0, wing.muscle(side, "hg3")))
        total = self._vertical_steering.b2_weight + self._vertical_steering.hg3_weight
        value = (
            self._vertical_steering.b2_weight * b2
            + self._vertical_steering.hg3_weight * hg3
        ) / total
        value *= self.vertical_steering_gain
        value = min(1.0, max(0.0, value))
        self.last_vertical_steering_activation[side] = value
        return value

    def _remaining_legacy_steering_offset(self, wing, side: str, phase: float) -> tuple[float, float]:
        # b2 is intentionally removed here because v5 represents it through the
        # measured b2/hg3 hinge mode.  b1/b3/i1 remain on the historical v3 seam
        # until their measured multi-axis modes are added separately.
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
            mode_activation = self._vertical_mode_activation(wing, side)
            dyaw, dpitch, dyaw_dphase, dpitch_dphase = self._vertical_steering.sample(phase)
            measured_angle = {
                "yaw": mode_activation * dyaw,
                "roll": 0.0,
                "pitch": mode_activation * dpitch,
            }
            measured_velocity = {
                "yaw": mode_activation * dyaw_dphase * omega,
                "roll": 0.0,
                "pitch": mode_activation * dpitch_dphase * omega,
            }
            self.last_vertical_steering_offset[side] = {
                "yaw": measured_angle["yaw"],
                "pitch": measured_angle["pitch"],
            }
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
        if hasattr(self, "last_vertical_steering_activation"):
            self.last_vertical_steering_activation = {"left": 0.0, "right": 0.0}
            self.last_vertical_steering_offset = {
                "left": {"yaw": 0.0, "pitch": 0.0},
                "right": {"yaw": 0.0, "pitch": 0.0},
            }


class FlyBodyV5MuscleAdapter(_MeasuredVerticalSteeringMixin, FlyBodyV4MuscleAdapter):
    """A-IFM power modulation plus measured b2/hg3 vertical steering mode."""


class FlyBodyV5NeuromuscularAdapter(
    _MeasuredVerticalSteeringMixin,
    FlyBodyV4NeuromuscularAdapter,
):
    """Whole-body v5 seam with the same MaleCNS motor boundary as v3/v4."""
