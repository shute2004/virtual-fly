#!/usr/bin/env python3
"""A-IFM power-modulated FlyBody flight seam.

v3 is preserved as the source-equivalent hovering baseline.  v4 adds one
biologically grounded capability that v3 lacks: tonic power-muscle drive can
change wing kinematics above the measured hovering cycle.

The current first step is deliberately narrow:

* the released individual DLM/DVM MNs keep their existing peripheral activation;
* the legacy saturating aggregate still supplies tracking-torque authority so
  v3's validated hovering support is not silently recalibrated;
* mean per-MN DLM/DVM activation is treated as a coarse A-IFM Ca/power proxy;
* power above the established-flight bootstrap level expands only the measured
  stroke (yaw) excursion, up to +7.5%; roll/pitch and 218 Hz timing are unchanged;
* the v3-only 2500 generalized-force clamp is removed because source FlyBody's
  wing actuator is not force-limited.

This does not add a task policy or external action decoder.  It changes only the
physical transduction from identified flight-motor output to wing mechanics.
"""
from __future__ import annotations

import math
from pathlib import Path

from virtual_fly.embodiment.body.measured_wingbeat import DEFAULT_PATTERN, MeasuredWingbeatCycle
from virtual_fly.embodiment.body.v3 import FlyBodyV3MuscleAdapter, FlyBodyV3NeuromuscularAdapter


AIFM_HOVER_PROXY = 0.35
MAX_STROKE_AMPLITUDE_SCALE = 1.075


class _AIFMPowerModulationMixin:
    def __init__(
        self,
        *args,
        measured_wing_pattern: Path = DEFAULT_PATTERN,
        aifm_hover_proxy: float = AIFM_HOVER_PROXY,
        max_stroke_amplitude_scale: float = MAX_STROKE_AMPLITUDE_SCALE,
        **kwargs,
    ) -> None:
        if not 0.0 < aifm_hover_proxy < 1.0:
            raise ValueError("aifm_hover_proxy must be in (0,1)")
        if not 1.0 <= max_stroke_amplitude_scale <= 1.10:
            raise ValueError("max_stroke_amplitude_scale must be in [1.0,1.10]")

        # The upstream FlyBody flight actuator is not force-limited.  Keep v3's
        # historical clamp untouched and remove it only for this new seam.
        kwargs.setdefault("max_abs_torque", math.inf)
        super().__init__(
            *args,
            measured_wing_pattern=measured_wing_pattern,
            **kwargs,
        )
        cycle = MeasuredWingbeatCycle(Path(measured_wing_pattern))
        self.aifm_hover_proxy = float(aifm_hover_proxy)
        self.max_stroke_amplitude_scale = float(max_stroke_amplitude_scale)
        self._measured_stroke_mean = float(cycle.mean[0])
        self.last_aifm_power_proxy = {"left": 0.0, "right": 0.0}
        self.last_stroke_amplitude_scale = {"left": 1.0, "right": 1.0}
        self.flight_physics_version = "v4-aifm-power-modulated"

    def _aifm_power_proxy(self, state, side: str) -> float:
        wing = self._wing_snapshot(state)
        proxy = 0.5 * (wing.mean_power_dlm(side) + wing.mean_power_dvm(side))
        return min(1.0, max(0.0, float(proxy)))

    def _stroke_amplitude_scale(self, state, side: str) -> float:
        proxy = self._aifm_power_proxy(state, side)
        self.last_aifm_power_proxy[side] = proxy
        if proxy <= self.aifm_hover_proxy:
            scale = 1.0
        else:
            fraction = (proxy - self.aifm_hover_proxy) / (1.0 - self.aifm_hover_proxy)
            scale = 1.0 + (self.max_stroke_amplitude_scale - 1.0) * min(1.0, fraction)
        self.last_stroke_amplitude_scale[side] = scale
        return scale

    def _compute_wing_torque(self, state, phase: float):
        # Keep v3's measured hover trajectory for roll/pitch and modify only the
        # primary stroke axis.  Tracking authority remains the legacy DLM/DVM
        # aggregate; the per-MN mean drives the new kinematic modulation.
        wing = self._wing_snapshot(state)
        base_target = self._wing_pattern(phase, 1.0)
        base_velocity = self._wing_pattern_velocity(phase, 1.0)
        labelled: dict[str, float] = {}
        by_dof: dict[int, float] = {}

        for side in ("left", "right"):
            power = self._power_activation(wing, side, phase)
            stroke_scale = self._stroke_amplitude_scale(wing, side)
            steering_yaw, steering_yaw_velocity = self._steering_stroke_offset(
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

                steering_angle = steering_yaw if axis == "yaw" else 0.0
                steering_velocity = steering_yaw_velocity if axis == "yaw" else 0.0
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
        if hasattr(self, "last_aifm_power_proxy"):
            self.last_aifm_power_proxy = {"left": 0.0, "right": 0.0}
            self.last_stroke_amplitude_scale = {"left": 1.0, "right": 1.0}


class FlyBodyV4MuscleAdapter(_AIFMPowerModulationMixin, FlyBodyV3MuscleAdapter):
    """v3 physical body plus A-IFM-driven above-hover stroke modulation."""


class FlyBodyV4NeuromuscularAdapter(
    _AIFMPowerModulationMixin,
    FlyBodyV3NeuromuscularAdapter,
):
    """Whole-body v4 seam with the same MaleCNS motor boundary as v3."""
