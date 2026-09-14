#!/usr/bin/env python3
"""Source-equivalent FlyBody flight adapters for Flyppy v3.

Flyppy v2 is intentionally preserved as a historical baseline.  v3 restores the
source FlyBody flight coordinate convention before any biological joints are added,
uses the source flight root pose, normalizes to the published FlyBody mass, and
uses the official measured hovering wing-beat cycle as the baseline wing pattern.

The MaleCNS -> individual motor neuron -> peripheral muscle path remains unchanged.
This module changes only the physical body seam that those muscles actuate.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math

import flybody_runtime
from flygym.compose.fly import FlyBody as FlyGymFlyBody

from virtual_fly.physics import FLYBODY_REFERENCE
from flybody_biophysics import normalize_fly_mass
from flybody_measured_wingbeat import (
    DEFAULT_PATTERN,
    SOURCE_CONTROL_TIMESTEP_S,
    install_measured_wingbeat,
)
from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flybody_neuromuscular_adapter import FlyBodyNeuromuscularAdapter


SOURCE_FLIGHT_ROOT_PITCH_DEG = -47.5


class SourceFlightFrameFlyBody(FlyGymFlyBody):
    """FlyGym FlyBody without its position-actuator wing-frame rest-pose correction.

    FlyGym 2.1 rotates the left/right wing bodies by +/-90 degrees in
    ``_correct_wing_default_pose``.  That is useful for its generic position-
    actuator convention but is not part of the source FlyBody flight coordinate
    system used by the published measured wing kinematics and fluid geometry.
    """

    def _correct_wing_default_pose(self) -> None:  # type: ignore[override]
        return


@contextmanager
def _source_flight_frame_build():
    """Temporarily make FlyBodyRuntime construct the source flight-frame body."""

    original = flybody_runtime.FlyBody
    flybody_runtime.FlyBody = SourceFlightFrameFlyBody
    try:
        yield
    finally:
        flybody_runtime.FlyBody = original


def _prepare_v3_kwargs(kwargs: dict) -> dict:
    prepared = dict(kwargs)
    supplied_pitch = prepared.get("flight_body_pitch_deg")
    if supplied_pitch is not None and not math.isclose(
        float(supplied_pitch), SOURCE_FLIGHT_ROOT_PITCH_DEG, abs_tol=1e-12
    ):
        raise ValueError(
            "Flyppy v3 requires the source-equivalent root pitch "
            f"{SOURCE_FLIGHT_ROOT_PITCH_DEG} deg"
        )
    prepared["flight_body_pitch_deg"] = SOURCE_FLIGHT_ROOT_PITCH_DEG

    # FlyBodyRuntime's historical default is the v2 provisional 1.09 mg body.
    # Skip that normalization and apply the published FlyBody 0.983 mg target below.
    prepared["normalize_canonical_mass"] = False

    # The source FlyBody flight task integrates physics at 50 us but updates and
    # holds its wing command every 0.2 ms.  Controller A/B on the source-equivalent
    # body showed that recomputing the virtual-muscle command every 50 us reduces
    # vertical support from ~1.05 BW to ~0.86 BW.  v1/v2 retain their historical
    # defaults; v3 alone restores the source command cadence.  The upstream flight
    # actuator is proportional, so v3 also removes the provisional derivative term.
    prepared.setdefault("virtual_power_kd", 0.0)
    prepared.setdefault("virtual_control_timestep_s", SOURCE_CONTROL_TIMESTEP_S)
    return prepared


def _finish_v3_body(body, pattern_path: Path) -> None:
    body.mass_normalization = normalize_fly_mass(
        body.sim,
        body.fly,
        target_mass_g=FLYBODY_REFERENCE.total_mass_g,
    )
    if not pattern_path.exists():
        raise RuntimeError(
            f"Flyppy v3 measured wing pattern missing: {pattern_path}; run "
            "uv run python scripts/dev/prefetch_flybody_flight_data.py"
        )
    install_measured_wingbeat(body, pattern_path)
    body.flight_physics_version = "v3-source-equivalent"
    body.source_equivalent_wing_frame = True
    body.reference_body_mass_mg = FLYBODY_REFERENCE.total_mass_mg


class FlyBodyV3MuscleAdapter(FlyBodyMuscleAdapter):
    """Wing-muscle adapter using the source-equivalent v3 physical flight seam."""

    def __init__(self, *args, measured_wing_pattern: Path = DEFAULT_PATTERN, **kwargs) -> None:
        prepared = _prepare_v3_kwargs(kwargs)
        with _source_flight_frame_build():
            super().__init__(*args, **prepared)
        _finish_v3_body(self, Path(measured_wing_pattern))


class FlyBodyV3NeuromuscularAdapter(FlyBodyNeuromuscularAdapter):
    """Whole-body MaleCNS motor adapter using source-equivalent v3 flight physics."""

    def __init__(self, *args, measured_wing_pattern: Path = DEFAULT_PATTERN, **kwargs) -> None:
        prepared = _prepare_v3_kwargs(kwargs)
        with _source_flight_frame_build():
            super().__init__(*args, **prepared)
        _finish_v3_body(self, Path(measured_wing_pattern))
