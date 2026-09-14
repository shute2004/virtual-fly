#!/usr/bin/env python3
"""Legacy bilateral DNg02 -> wing-amplitude adapter.

This module is retained only for old smoke tests and historical experiments.
The current Flyppy learning path does **not** use DNg02 population averages; it
uses individual released wing motor-neuron spikes -> ``WingMusclePeriphery`` ->
``FlyBodyMuscleAdapter``.

Shared FlyBody/MuJoCo construction lives in ``flybody_runtime.py`` so current
code no longer inherits from this legacy motor decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from flygym.compose import ActuatorType

from flybody_runtime import FlyBodyRuntime


@dataclass(frozen=True)
class WingDrive:
    """Normalized left/right DNg02 population activity for legacy experiments."""

    left: float
    right: float


class FlyBodyWingAdapter(FlyBodyRuntime):
    """Legacy adapter mapping bilateral DNg02 activity to wing amplitude."""

    def __init__(
        self,
        *,
        dng02_mean_gain: float = 0.30,
        dng02_steering_gain: float = 0.50,
        min_scale: float = 0.65,
        max_scale: float = 1.45,
        **kwargs,
    ) -> None:
        if dng02_mean_gain < 0 or dng02_steering_gain < 0:
            raise ValueError("DNg02 gains must be non-negative")
        if not 0 < min_scale <= max_scale:
            raise ValueError("invalid wing amplitude scale limits")
        self.dng02_mean_gain = float(dng02_mean_gain)
        self.dng02_steering_gain = float(dng02_steering_gain)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        super().__init__(**kwargs)

    @staticmethod
    def _bounded_activity(value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("DNg02 spike fraction must be finite")
        return min(1.0, max(0.0, value))

    def _wing_scales(self, drive: WingDrive) -> tuple[float, float]:
        left_activity = self._bounded_activity(drive.left)
        right_activity = self._bounded_activity(drive.right)
        mean_activity = 0.5 * (left_activity + right_activity)

        left_differential = right_activity - left_activity
        right_differential = left_activity - right_activity
        left_scale = (
            1.0
            + self.dng02_mean_gain * mean_activity
            + self.dng02_steering_gain * left_differential
        )
        right_scale = (
            1.0
            + self.dng02_mean_gain * mean_activity
            + self.dng02_steering_gain * right_differential
        )
        return (
            min(self.max_scale, max(self.min_scale, left_scale)),
            min(self.max_scale, max(self.min_scale, right_scale)),
        )

    def step(self, drive: WingDrive, *, physics_steps: int = 1) -> None:
        if physics_steps < 1:
            raise ValueError("physics_steps must be >= 1")
        left_scale, right_scale = self._wing_scales(drive)

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
