#!/usr/bin/env python3
"""Deterministic geometric course used by the first Flyppy closed loop.

The course does not decide how the fly moves. It only turns a body position into
observable geometry and biologically-deliverable outcome events (gate passed or
collision). Neural stimulation is owned by the experiment layer, not this file.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


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


class FlyppyCourse:
    """Small deterministic 2D flight course in the x-z plane.

    Units are millimetres, matching FlyGym/FlyBody. The y coordinate is left to
    the body simulator; the first course intentionally tests altitude control
    before adding lateral steering.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        gate_count: int = 12,
        first_gate_x_mm: float = 8.0,
        gate_spacing_mm: float = 7.0,
        corridor_low_z_mm: float = 0.0,
        corridor_high_z_mm: float = 10.0,
        gap_half_height_mm: float = 2.0,
        center_margin_mm: float = 0.5,
    ) -> None:
        if gate_count < 1:
            raise ValueError("gate_count must be >= 1")
        if corridor_high_z_mm <= corridor_low_z_mm:
            raise ValueError("corridor bounds are invalid")
        if gap_half_height_mm <= 0:
            raise ValueError("gap_half_height_mm must be > 0")

        self.floor_z_mm = float(corridor_low_z_mm)
        self.ceiling_z_mm = float(corridor_high_z_mm)
        self._next_gate = 0
        self._last_x_mm = -math.inf

        low_center = self.floor_z_mm + gap_half_height_mm + center_margin_mm
        high_center = self.ceiling_z_mm - gap_half_height_mm - center_margin_mm
        if high_center <= low_center:
            raise ValueError("corridor is too small for requested gate gap")

        rng = random.Random(seed)
        self.gates = tuple(
            Gate(
                x_mm=first_gate_x_mm + i * gate_spacing_mm,
                center_z_mm=rng.uniform(low_center, high_center),
                half_gap_mm=gap_half_height_mm,
            )
            for i in range(gate_count)
        )

    @property
    def next_gate_index(self) -> int:
        return self._next_gate

    @property
    def finished(self) -> bool:
        return self._next_gate >= len(self.gates)

    def reset(self) -> None:
        self._next_gate = 0
        self._last_x_mm = -math.inf

    def observe(self, x_mm: float, z_mm: float) -> CourseObservation:
        if self.finished:
            gate = self.gates[-1]
        else:
            gate = self.gates[self._next_gate]
        return CourseObservation(
            next_gate_dx_mm=gate.x_mm - x_mm,
            next_gate_center_dz_mm=gate.center_z_mm - z_mm,
            gap_low_dz_mm=gate.low_z_mm - z_mm,
            gap_high_dz_mm=gate.high_z_mm - z_mm,
            floor_dz_mm=self.floor_z_mm - z_mm,
            ceiling_dz_mm=self.ceiling_z_mm - z_mm,
        )

    def update(self, x_mm: float, z_mm: float, *, body_radius_mm: float = 0.65) -> CourseEvent:
        if body_radius_mm <= 0:
            raise ValueError("body_radius_mm must be > 0")

        if z_mm - body_radius_mm <= self.floor_z_mm or z_mm + body_radius_mm >= self.ceiling_z_mm:
            self._last_x_mm = x_mm
            return CourseEvent(collision=True, gate_index=self._next_gate if not self.finished else None)

        if self.finished:
            self._last_x_mm = x_mm
            return CourseEvent(finished=True)

        gate = self.gates[self._next_gate]
        horizontal_overlap = abs(x_mm - gate.x_mm) <= gate.half_thickness_mm + body_radius_mm
        vertical_clear = (
            z_mm - body_radius_mm > gate.low_z_mm
            and z_mm + body_radius_mm < gate.high_z_mm
        )
        if horizontal_overlap and not vertical_clear:
            self._last_x_mm = x_mm
            return CourseEvent(collision=True, gate_index=self._next_gate)

        crossed_plane = self._last_x_mm < gate.x_mm <= x_mm
        if crossed_plane:
            if not vertical_clear:
                self._last_x_mm = x_mm
                return CourseEvent(collision=True, gate_index=self._next_gate)
            passed = self._next_gate
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
    course = FlyppyCourse(seed=7, gate_count=2)
    gate = course.gates[0]
    z = gate.center_z_mm
    assert not course.update(gate.x_mm - 1.0, z).collision
    event = course.update(gate.x_mm + 0.1, z)
    assert event.passed_gate and event.gate_index == 0

    course.reset()
    assert course.next_gate_index == 0 and not course.finished

    bad = FlyppyCourse(seed=7, gate_count=1)
    gate = bad.gates[0]
    assert not bad.update(gate.x_mm - 1.0, gate.center_z_mm).collision
    event = bad.update(gate.x_mm, bad.floor_z_mm + 0.1)
    assert event.collision
    print("flyppy_course=PASS")


if __name__ == "__main__":
    _self_test()
