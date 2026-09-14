#!/usr/bin/env python3
"""Install FlyBody's measured baseline wing-beat cycle onto an existing adapter.

The published FlyBody controller uses a baseline wing-beat pattern derived from
hovering Drosophila kinematics.  This helper loads the official
``wing_pattern_fmech.npy`` file and replaces only the adapter's base cyclic wing
kinematics.  It does not add an action decoder, task policy, reward shaping, or
neural output transformation.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import MethodType

import numpy as np


DEFAULT_PATTERN = Path("artifacts/flybody-data/wing_pattern_fmech.npy")
AXES = ("yaw", "roll", "pitch")


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
        # Central difference in sample index, then samples/second.
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
