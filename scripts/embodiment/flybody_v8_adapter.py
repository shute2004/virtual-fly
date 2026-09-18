#!/usr/bin/env python3
"""Measured steering + neutral-trim FlyBody seam.

v8 combines two previously isolated, task-independent body improvements:

* v6: separate Melis/Siwanowicz/Dickinson 2024 b2 and hg3 measured yaw/pitch
  wing-hinge modes on top of the v4 A-IFM power modulation;
* v7: the symmetric neutral-flight measured-wingbeat trim.

No body attitude, Flyppy geometry, reward, target, or external controller enters
this seam.  MaleCNS motor output still drives the same peripheral muscles; v8
only restores the measured multi-axis mechanical consequences of b2/hg3 while
retaining v7's neutral baseline cycle.
"""
from __future__ import annotations

from flybody_v6_adapter import FlyBodyV6MuscleAdapter, FlyBodyV6NeuromuscularAdapter
from flybody_v7_adapter import _NeutralTrimMixin


class FlyBodyV8MuscleAdapter(_NeutralTrimMixin, FlyBodyV6MuscleAdapter):
    """v6 measured b2/hg3 steering with the v7 neutral wingbeat baseline."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flight_physics_version = "v8-v6-steering-plus-neutral-trim"


class FlyBodyV8NeuromuscularAdapter(_NeutralTrimMixin, FlyBodyV6NeuromuscularAdapter):
    """Whole-body v8 seam; decision making remains entirely inside MaleCNS."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flight_physics_version = "v8-v6-steering-plus-neutral-trim"
