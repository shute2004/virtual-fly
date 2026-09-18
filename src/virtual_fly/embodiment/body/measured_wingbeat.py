#!/usr/bin/env python3
"""Install and reproduce FlyBody's measured baseline wing-beat cycle.

The published FlyBody controller uses a baseline wing-beat pattern derived from
hovering Drosophila kinematics.  ``MeasuredWingbeatCycle`` exposes a continuous
phase interpolation used by virtual-fly diagnostics and legacy adapters.
``SourceStyleWingbeatController`` reproduces the upstream FlyBody WPG's base-frequency
resampling and 0.2 ms control cadence closely enough to compare the port against
the original control seam.

Neither helper adds a task policy, reward shaping, or neural output decoder.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import MethodType

import numpy as np


DEFAULT_PATTERN = Path("artifacts/flybody-data/wing_pattern_fmech.npy")
AXES = ("yaw", "roll", "pitch")
SOURCE_CONTROL_TIMESTEP_S = 2.0e-4


class MeasuredWingbeatCycle:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        pattern = np.load(self.path, allow_pickle=False)
        if pattern.ndim != 2 or pattern.shape[1] != 3 or pattern.shape[0] < 20:
            raise RuntimeError(
                f"unexpected measured wingbeat shape {pattern.shape}; expected (N,3)"
            )
        pattern = np.asarray(pattern, dtype=np.float64)
        if not np.all(np.isfinite(pattern)):
            raise RuntimeError("measured wingbeat contains non-finite values")
        self.pattern = pattern
        self.samples = int(pattern.shape[0])
        self.mean = np.mean(pattern, axis=0)

    def _sample_vector(self, phase: float) -> np.ndarray:
        unit = (phase % (2.0 * math.pi)) / (2.0 * math.pi)
        position = unit * self.samples
        i0 = int(math.floor(position)) % self.samples
        frac = position - math.floor(position)
        i1 = (i0 + 1) % self.samples
        return (1.0 - frac) * self.pattern[i0] + frac * self.pattern[i1]

    def _derivative_vector(self, phase: float, wingbeat_hz: float) -> np.ndarray:
        unit = (phase % (2.0 * math.pi)) / (2.0 * math.pi)
        position = unit * self.samples
        i0 = int(math.floor(position)) % self.samples
        im1 = (i0 - 1) % self.samples
        ip1 = (i0 + 1) % self.samples
        return 0.5 * (self.pattern[ip1] - self.pattern[im1]) * self.samples * wingbeat_hz

    def angles(self, phase: float, amplitude_scale: float) -> dict[str, float]:
        value = self._sample_vector(phase)
        scaled = self.mean + float(amplitude_scale) * (value - self.mean)
        return {axis: float(scaled[i]) for i, axis in enumerate(AXES)}

    def velocities(
        self, phase: float, amplitude_scale: float, wingbeat_hz: float
    ) -> dict[str, float]:
        value = float(amplitude_scale) * self._derivative_vector(phase, wingbeat_hz)
        return {axis: float(value[i]) for i, axis in enumerate(AXES)}


class SourceStyleWingbeatController:
    """Base-frequency FlyBody WPG trajectory at the upstream control cadence.

    This intentionally mirrors ``TuragaLab/flybody``'s
    ``WingBeatPatternGenerator`` construction for a fixed base frequency.  In
    particular, targets are resampled to the 0.2 ms flight control timestep and
    then held across the four 50 us physics steps in each control interval.
    """

    def __init__(
        self,
        cycle: MeasuredWingbeatCycle,
        *,
        wingbeat_hz: float = 218.0,
        control_timestep_s: float = SOURCE_CONTROL_TIMESTEP_S,
        min_repeats: int = 10,
        max_repeats: int = 20,
    ) -> None:
        if wingbeat_hz <= 0.0 or control_timestep_s <= 0.0:
            raise ValueError("wingbeat frequency and control timestep must be positive")
        if min_repeats < 1 or max_repeats < min_repeats:
            raise ValueError("invalid repeat range")

        self.cycle = cycle
        self.wingbeat_hz = float(wingbeat_hz)
        self.control_timestep_s = float(control_timestep_s)

        beat_time = 1.0 / self.wingbeat_hz
        repeats = np.arange(min_repeats, max_repeats + 1)
        relative_error = ((repeats * beat_time) % self.control_timestep_s) / self.control_timestep_s
        argmin_over = int(np.argmin(relative_error))
        argmin_under = int(np.argmin(np.abs(1.0 - relative_error)))
        if relative_error[argmin_over] < abs(1.0 - relative_error[argmin_under]):
            argmin = argmin_over
            shift = self.control_timestep_s
        else:
            argmin = argmin_under
            shift = 0.0

        # Preserve the upstream implementation exactly.  It uses the selected
        # array index + 1 here rather than repeats[argmin].
        n_repeats = argmin + 1
        repeated = np.tile(cycle.pattern, (n_repeats, 1))
        dt_data = beat_time / cycle.samples
        duration = repeated.shape[0] * dt_data
        t_axis_data = np.linspace(0.0, duration, repeated.shape[0])
        t_axis_ctrl = np.arange(0.0, duration - shift, self.control_timestep_s)
        if t_axis_ctrl.size < 2:
            raise RuntimeError("source-style WPG resampling produced too few control samples")

        trajectory = np.empty((t_axis_ctrl.shape[0], 3), dtype=np.float64)
        for i in range(3):
            trajectory[:, i] = np.interp(t_axis_ctrl, t_axis_data, repeated[:, i])

        self.trajectory = trajectory
        self.samples = int(trajectory.shape[0])
        self._step = 0

    def reset(self, initial_phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        phase = float(initial_phase) % 1.0
        # For the fixed-frequency diagnostic a linear phase proxy is sufficient
        # to select the nearest location in the repeated source trajectory.
        step = int(round(phase * self.samples)) % self.samples
        self._step = step
        qpos = self.trajectory[self._step].copy()
        qvel = (
            self.trajectory[(self._step + 1) % self.samples] - qpos
        ) / self.control_timestep_s
        return qpos, qvel

    def step(self) -> np.ndarray:
        self._step = (self._step + 1) % self.samples
        return self.trajectory[self._step].copy()

    def current(self) -> np.ndarray:
        return self.trajectory[self._step].copy()


def install_measured_wingbeat(body, path: Path = DEFAULT_PATTERN) -> MeasuredWingbeatCycle:
    """Replace one adapter instance's base wing cycle, then reset onto that cycle."""

    cycle = MeasuredWingbeatCycle(path)

    def wing_pattern(_self, phase: float, amplitude_scale: float) -> dict[str, float]:
        return cycle.angles(phase, amplitude_scale)

    def wing_pattern_velocity(
        _self, phase: float, amplitude_scale: float
    ) -> dict[str, float]:
        return cycle.velocities(phase, amplitude_scale, body.wingbeat_hz)

    body._wing_pattern = MethodType(wing_pattern, body)
    body._wing_pattern_velocity = MethodType(wing_pattern_velocity, body)
    body.measured_wingbeat_path = str(cycle.path)
    body.reset()
    return cycle
