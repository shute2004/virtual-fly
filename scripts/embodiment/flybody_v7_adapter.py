#!/usr/bin/env python3
"""Task-independent neutral-trim extension of FlyBody v4.

v7 keeps v4's A-IFM power modulation unchanged and blends the published raw
measured hover cycle toward a symmetric neutral-flight trim.  The trim was
calibrated only against FlyBody's own mean fluid wrench at the source -47.5 deg
flight pose: preserve the baseline translational wrench while minimizing mean
root pitch fluid torque and kinematic departure.

No Flyppy geometry, reward, CNS activity, obstacle state, or learned controller
enters this calibration.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import MethodType

from flybody_measured_wingbeat import DEFAULT_PATTERN, MeasuredWingbeatCycle
from flybody_v4_adapter import FlyBodyV4MuscleAdapter, FlyBodyV4NeuromuscularAdapter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEUTRAL_PATTERN = ROOT / "artifacts/embodiment/wing-pattern-neutral-trim-v1.npy"


class _NeutralTrimMixin:
    def __init__(
        self,
        *args,
        measured_wing_pattern: Path = DEFAULT_NEUTRAL_PATTERN,
        neutral_trim_strength: float = 1.0,
        **kwargs,
    ) -> None:
        strength = float(neutral_trim_strength)
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("neutral_trim_strength must be finite and in [0,1]")

        # Build v4 on the untouched published pattern first.  The neutral trim is
        # then blended at the body seam, so strength=0 is exactly the v4 baseline.
        super().__init__(*args, measured_wing_pattern=DEFAULT_PATTERN, **kwargs)
        raw = MeasuredWingbeatCycle(DEFAULT_PATTERN)
        trimmed = MeasuredWingbeatCycle(Path(measured_wing_pattern))
        if raw.pattern.shape != trimmed.pattern.shape:
            raise ValueError("neutral trim must have the same shape as the source pattern")

        def wing_pattern(_self, phase: float, amplitude_scale: float) -> dict[str, float]:
            a = raw.angles(phase, amplitude_scale)
            b = trimmed.angles(phase, amplitude_scale)
            return {axis: (1.0 - strength) * a[axis] + strength * b[axis] for axis in a}

        def wing_pattern_velocity(_self, phase: float, amplitude_scale: float) -> dict[str, float]:
            a = raw.velocities(phase, amplitude_scale, self.wingbeat_hz)
            b = trimmed.velocities(phase, amplitude_scale, self.wingbeat_hz)
            return {axis: (1.0 - strength) * a[axis] + strength * b[axis] for axis in a}

        self._wing_pattern = MethodType(wing_pattern, self)
        self._wing_pattern_velocity = MethodType(wing_pattern_velocity, self)
        blended_mean = (1.0 - strength) * raw.mean + strength * trimmed.mean
        self._measured_stroke_mean = float(blended_mean[0])
        self.neutral_trim_strength = strength
        self.neutral_trim_pattern = str(Path(measured_wing_pattern))
        self.measured_wingbeat_path = str(DEFAULT_PATTERN)
        self.flight_physics_version = "v7-aifm-neutral-trim"
        self.reset()


class FlyBodyV7MuscleAdapter(_NeutralTrimMixin, FlyBodyV4MuscleAdapter):
    """A-IFM body with the task-independent neutral-flight wing trim."""


class FlyBodyV7NeuromuscularAdapter(_NeutralTrimMixin, FlyBodyV4NeuromuscularAdapter):
    """Whole-body v7 seam with the same MaleCNS motor boundary as v4."""
