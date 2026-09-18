#!/usr/bin/env python3
"""Present packed Flyppy body processes as one logical handle per slot.

This adapter lets the existing process-body trainer reuse its proven learning,
curriculum, checkpoint, commit, and telemetry loop unchanged while several
logical fly slots share one MuJoCo owner process.
"""

from __future__ import annotations

from typing import Any

from virtual_fly.runtime.packed_body_worker import (
    PackedFlyppyBodyProcess,
    default_process_count,
    spawn_packed_body_processes,
)


class PackedFlyppySlotHandle:
    def __init__(self, worker: PackedFlyppyBodyProcess, slot_index: int) -> None:
        self._worker = worker
        self.slot_index = int(slot_index)
        self.body_ids = tuple(int(value) for value in worker.body_ids[self.slot_index])
        self.control_dt_s = float(worker.control_dt_s)

    def _wrap_payload(self, command: str, payload):
        if command == "observe":
            return (self.slot_index,)
        if command == "act":
            return {self.slot_index: tuple(int(value) for value in payload)}
        if command == "snapshot":
            return (self.slot_index,)
        if command == "reset":
            return {self.slot_index: dict(payload)}
        return payload

    def _unwrap(self, command: str, payload):
        if command in {"observe", "act", "snapshot", "reset"}:
            return payload[self.slot_index]
        return payload

    def request(self, command: str, payload=None) -> None:
        self._worker.request(command, self._wrap_payload(command, payload))

    def receive(self, phase: str):
        return self._unwrap(phase, self._worker.receive(phase))

    def call(self, command: str, payload=None):
        self.request(command, payload)
        return self.receive(command)

    def reset(
        self,
        *,
        episode: int,
        source_weight_version: int,
        spawn_x_mm: float,
        spawn_z_mm: float,
        initial_speed_mm_s: float,
        initial_vz_mm_s: float = 0.0,
        course_start_gate_index: int | None = None,
        gate_center_overrides: dict[int, float] | None = None,
    ):
        return self.call(
            "reset",
            {
                "episode": int(episode),
                "source_weight_version": int(source_weight_version),
                "spawn_x_mm": float(spawn_x_mm),
                "spawn_z_mm": float(spawn_z_mm),
                "initial_speed_mm_s": float(initial_speed_mm_s),
                "initial_vz_mm_s": float(initial_vz_mm_s),
                "course_start_gate_index": (
                    None
                    if course_start_gate_index is None
                    else int(course_start_gate_index)
                ),
                "gate_center_overrides": {
                    int(key): float(value)
                    for key, value in (gate_center_overrides or {}).items()
                },
            },
        )

    def snapshot(self):
        return self.call("snapshot")

    def close(self) -> None:
        # Idempotent at the process level. Multiple slot handles can refer to the
        # same packed worker; the first close shuts it down and later closes are
        # harmless.
        self._worker.close()


def spawn_packed_slot_handles(
    *,
    population: int,
    process_count: int | None,
    physics_steps: int,
    timeout_s: float,
    config: dict[str, Any],
) -> tuple[list[PackedFlyppySlotHandle], int]:
    resolved = default_process_count(population) if process_count is None else min(
        int(population), int(process_count)
    )
    workers, slot_to_worker = spawn_packed_body_processes(
        population=population,
        process_count=resolved,
        physics_steps=physics_steps,
        timeout_s=timeout_s,
        config=config,
    )
    handles = [
        PackedFlyppySlotHandle(slot_to_worker[slot_index], slot_index)
        for slot_index in range(population)
    ]
    return handles, resolved
