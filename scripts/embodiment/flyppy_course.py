#!/usr/bin/env python3
"""Deterministic geometric Flyppy course.

Version 1 preserves the historical 10 mm corridor / 4 mm gate experiment.
Version 2 derives every dimension from the canonical physical fly specification
and delegates collision detection to the full MuJoCo body rather than a thorax
center plus fixed radius approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random

from virtual_fly.physics import FLYPPY_GEOMETRY


@dataclass(frozen=True)
class Gate:
    x_mm: float
    center_z_mm: float
    half_gap_mm: float
    half_thickness_mm: float = 0.30

    @property
    def low_z_mm(self) -> float:
        return self.center_z_mm - self.half_gap_mm

    @property
    def high_z_mm(self) -> float:
        return self.center_z_mm + self.half_gap_mm


@dataclass(frozen=True)
class CourseObservation:
    next_gate_dx_mm: float
    next_gate_center_dz_mm: float
    gap_low_dz_mm: float
    gap_high_dz_mm: float
    floor_dz_mm: float
    ceiling_dz_mm: float


@dataclass(frozen=True)
class CourseEvent:
    passed_gate: bool = False
    collision: bool = False
    finished: bool = False
    gate_index: int | None = None
    collision_reason: str | None = None


class FlyppyCourse:
    """Deterministic 2D gate layout in the x-z plane, in millimetres."""

    def __init__(
        self,
        *,
        seed: int = 0,
        gate_count: int = 12,
        environment_version: str = "v1",
        first_gate_x_mm: float | None = None,
        gate_spacing_mm: float | None = None,
        corridor_low_z_mm: float | None = None,
        corridor_high_z_mm: float | None = None,
        gap_half_height_mm: float | None = None,
        center_margin_mm: float | None = None,
        gate_half_thickness_mm: float | None = None,
        lateral_half_width_mm: float | None = None,
    ) -> None:
        if environment_version not in {"v1", "v2"}:
            raise ValueError("environment_version must be v1 or v2")
        if gate_count < 1:
            raise ValueError("gate_count must be >= 1")

        if environment_version == "v2":
            first_gate_x_mm = (
                FLYPPY_GEOMETRY.first_gate_x_mm
                if first_gate_x_mm is None
                else first_gate_x_mm
            )
            gate_spacing_mm = (
                FLYPPY_GEOMETRY.gate_spacing_mm
                if gate_spacing_mm is None
                else gate_spacing_mm
            )
            corridor_low_z_mm = (
                FLYPPY_GEOMETRY.corridor_low_z_mm
                if corridor_low_z_mm is None
                else corridor_low_z_mm
            )
            corridor_high_z_mm = (
                FLYPPY_GEOMETRY.corridor_high_z_mm
                if corridor_high_z_mm is None
                else corridor_high_z_mm
            )
            gap_half_height_mm = (
                FLYPPY_GEOMETRY.gate_gap_half_height_mm
                if gap_half_height_mm is None
                else gap_half_height_mm
            )
            center_margin_mm = (
                FLYPPY_GEOMETRY.center_margin_mm
                if center_margin_mm is None
                else center_margin_mm
            )
            gate_half_thickness_mm = (
                FLYPPY_GEOMETRY.gate_half_thickness_mm
                if gate_half_thickness_mm is None
                else gate_half_thickness_mm
            )
            lateral_half_width_mm = (
                FLYPPY_GEOMETRY.lateral_half_width_mm
                if lateral_half_width_mm is None
                else lateral_half_width_mm
            )
        else:
            first_gate_x_mm = 8.0 if first_gate_x_mm is None else first_gate_x_mm
            gate_spacing_mm = 7.0 if gate_spacing_mm is None else gate_spacing_mm
            corridor_low_z_mm = 0.0 if corridor_low_z_mm is None else corridor_low_z_mm
            corridor_high_z_mm = 10.0 if corridor_high_z_mm is None else corridor_high_z_mm
            gap_half_height_mm = 2.0 if gap_half_height_mm is None else gap_half_height_mm
            center_margin_mm = 0.5 if center_margin_mm is None else center_margin_mm
            gate_half_thickness_mm = (
                0.30 if gate_half_thickness_mm is None else gate_half_thickness_mm
            )
            lateral_half_width_mm = (
                8.0 if lateral_half_width_mm is None else lateral_half_width_mm
            )

        values = (
            first_gate_x_mm,
            gate_spacing_mm,
            corridor_low_z_mm,
            corridor_high_z_mm,
            gap_half_height_mm,
            center_margin_mm,
            gate_half_thickness_mm,
            lateral_half_width_mm,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("course geometry must be finite")
        if float(corridor_high_z_mm) <= float(corridor_low_z_mm):
            raise ValueError("corridor bounds are invalid")
        if float(gap_half_height_mm) <= 0.0:
            raise ValueError("gap_half_height_mm must be > 0")
        if float(gate_spacing_mm) <= 0.0 or float(gate_half_thickness_mm) <= 0.0:
            raise ValueError("gate spacing/thickness must be positive")
        if float(lateral_half_width_mm) <= 0.0:
            raise ValueError("lateral_half_width_mm must be positive")

        self.environment_version = environment_version
        self.floor_z_mm = float(corridor_low_z_mm)
        self.ceiling_z_mm = float(corridor_high_z_mm)
        self.lateral_half_width_mm = float(lateral_half_width_mm)
        self.gate_half_thickness_mm = float(gate_half_thickness_mm)
        self._next_gate = 0
        self._last_x_mm = -math.inf

        low_center = self.floor_z_mm + float(gap_half_height_mm) + float(center_margin_mm)
        high_center = self.ceiling_z_mm - float(gap_half_height_mm) - float(center_margin_mm)
        if high_center <= low_center:
            raise ValueError("corridor is too small for requested gate gap")

        rng = random.Random(seed)
        all_gates = tuple(
            Gate(
                x_mm=float(first_gate_x_mm) + i * float(gate_spacing_mm),
                center_z_mm=rng.uniform(low_center, high_center),
                half_gap_mm=float(gap_half_height_mm),
                half_thickness_mm=self.gate_half_thickness_mm,
            )
            for i in range(gate_count)
        )

        start_text = os.environ.get("VF_COURSE_START_GATE", "0").strip() or "0"
        try:
            start_gate = int(start_text)
        except ValueError as exc:
            raise ValueError("VF_COURSE_START_GATE must be an integer") from exc
        if start_gate < 0 or start_gate >= len(all_gates):
            raise ValueError(
                f"VF_COURSE_START_GATE must be in [0, {len(all_gates) - 1}], got {start_gate}"
            )
        self.source_gate_offset = start_gate
        self.gates = all_gates[start_gate:]

    @property
    def next_gate_index(self) -> int:
        return self.absolute_next_gate_index

    @property
    def absolute_next_gate_index(self) -> int:
        return self.source_gate_offset + self._next_gate

    @property
    def finished(self) -> bool:
        return self._next_gate >= len(self.gates)

    def reset(self) -> None:
        self._next_gate = 0
        self._last_x_mm = -math.inf

    def observe(self, x_mm: float, z_mm: float) -> CourseObservation:
        gate = self.gates[-1] if self.finished else self.gates[self._next_gate]
        return CourseObservation(
            next_gate_dx_mm=gate.x_mm - x_mm,
            next_gate_center_dz_mm=gate.center_z_mm - z_mm,
            gap_low_dz_mm=gate.low_z_mm - z_mm,
            gap_high_dz_mm=gate.high_z_mm - z_mm,
            floor_dz_mm=self.floor_z_mm - z_mm,
            ceiling_dz_mm=self.ceiling_z_mm - z_mm,
        )

    def update(
        self,
        x_mm: float,
        z_mm: float,
        *,
        body_radius_mm: float = 0.65,
        physical_collision_reason: str | None = None,
        analytic_body_collision: bool = True,
    ) -> CourseEvent:
        """Advance gate state after one physics/control step.

        v1 uses the historical thorax-center radius approximation. v2 passes a
        MuJoCo-derived ``physical_collision_reason`` and sets
        ``analytic_body_collision=False`` so the complete articulated body,
        including wings and halteres, decides collisions.
        """

        if physical_collision_reason not in {None, "floor", "ceiling", "gate"}:
            raise ValueError(f"unknown physical collision reason: {physical_collision_reason}")
        if physical_collision_reason is not None:
            self._last_x_mm = x_mm
            return CourseEvent(
                collision=True,
                gate_index=self.absolute_next_gate_index if not self.finished else None,
                collision_reason=physical_collision_reason,
            )

        if analytic_body_collision:
            if body_radius_mm <= 0:
                raise ValueError("body_radius_mm must be > 0")
            if z_mm - body_radius_mm <= self.floor_z_mm:
                self._last_x_mm = x_mm
                return CourseEvent(
                    collision=True,
                    gate_index=self.absolute_next_gate_index if not self.finished else None,
                    collision_reason="floor",
                )
            if z_mm + body_radius_mm >= self.ceiling_z_mm:
                self._last_x_mm = x_mm
                return CourseEvent(
                    collision=True,
                    gate_index=self.absolute_next_gate_index if not self.finished else None,
                    collision_reason="ceiling",
                )

        if self.finished:
            self._last_x_mm = x_mm
            return CourseEvent(finished=True)

        gate = self.gates[self._next_gate]
        if analytic_body_collision:
            horizontal_overlap = (
                abs(x_mm - gate.x_mm) <= gate.half_thickness_mm + body_radius_mm
            )
            vertical_clear = (
                z_mm - body_radius_mm > gate.low_z_mm
                and z_mm + body_radius_mm < gate.high_z_mm
            )
            if horizontal_overlap and not vertical_clear:
                self._last_x_mm = x_mm
                return CourseEvent(
                    collision=True,
                    gate_index=self.absolute_next_gate_index,
                    collision_reason="gate",
                )
        else:
            # Full-body MuJoCo contact already established clearance; only the
            # thorax plane crossing is needed to count a pass.
            vertical_clear = gate.low_z_mm < z_mm < gate.high_z_mm

        crossed_plane = self._last_x_mm < gate.x_mm <= x_mm
        if crossed_plane:
            if not vertical_clear:
                self._last_x_mm = x_mm
                return CourseEvent(
                    collision=True,
                    gate_index=self.absolute_next_gate_index,
                    collision_reason="gate",
                )
            passed = self.absolute_next_gate_index
            self._next_gate += 1
            self._last_x_mm = x_mm
            return CourseEvent(
                passed_gate=True,
                finished=self.finished,
                gate_index=passed,
            )

        self._last_x_mm = x_mm
        return CourseEvent()


def _self_test() -> None:
    previous = os.environ.pop("VF_COURSE_START_GATE", None)
    try:
        course = FlyppyCourse(seed=7, gate_count=2)
        gate = course.gates[0]
        z = gate.center_z_mm
        assert not course.update(gate.x_mm - 1.0, z).collision
        event = course.update(gate.x_mm + 0.1, z)
        assert event.passed_gate and event.gate_index == 0

        v2 = FlyppyCourse(seed=7, gate_count=2, environment_version="v2")
        assert math.isclose(v2.ceiling_z_mm, FLYPPY_GEOMETRY.corridor_high_z_mm)
        assert math.isclose(v2.gates[0].x_mm, FLYPPY_GEOMETRY.first_gate_x_mm)
        assert math.isclose(
            2.0 * v2.gates[0].half_gap_mm, FLYPPY_GEOMETRY.gate_gap_height_mm
        )
        v2_gate = v2.gates[0]
        assert not v2.update(
            v2_gate.x_mm - 1.0,
            v2_gate.center_z_mm,
            analytic_body_collision=False,
        ).collision
        event = v2.update(
            v2_gate.x_mm + 0.1,
            v2_gate.center_z_mm,
            analytic_body_collision=False,
        )
        assert event.passed_gate

        bad = FlyppyCourse(seed=7, gate_count=1)
        gate = bad.gates[0]
        assert not bad.update(gate.x_mm - 1.0, gate.center_z_mm).collision
        event = bad.update(gate.x_mm, bad.floor_z_mm + 0.1)
        assert event.collision and event.collision_reason == "floor"

        os.environ["VF_COURSE_START_GATE"] = "1"
        staged = FlyppyCourse(seed=7, gate_count=3)
        assert staged.source_gate_offset == 1
        assert staged.next_gate_index == 1
        assert len(staged.gates) == 2
        target = staged.gates[0]
        assert math.isclose(target.x_mm, 15.0)
        assert not staged.update(target.x_mm - 1.0, target.center_z_mm).collision
        event = staged.update(target.x_mm + 0.1, target.center_z_mm)
        assert event.passed_gate and event.gate_index == 1
        assert staged.next_gate_index == 2
        print("flyppy_course=PASS")
    finally:
        if previous is None:
            os.environ.pop("VF_COURSE_START_GATE", None)
        else:
            os.environ["VF_COURSE_START_GATE"] = previous


if __name__ == "__main__":
    _self_test()
