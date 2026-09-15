#!/usr/bin/env python3
"""Packed-body production candidate for Flyppy population training.

The proven process-body trainer owns all learning/curriculum/checkpoint logic.
This launcher replaces only its body-process factory so several logical fly slots
share one MuJoCo owner process.  The default packing rule is the measured M1
optimum family: min(population, 4) body processes.

Set VF_FLYPPY_BODY_PROCESSES to a positive integer to override the default.
"""

from __future__ import annotations

import os

import train_flyppy_population_process as trainer
from flyppy_packed_slot_adapter import spawn_packed_slot_handles


def _spawn_packed(
    *,
    population: int,
    physics_steps: int,
    timeout_s: float,
    config: dict[str, object],
):
    raw = os.environ.get("VF_FLYPPY_BODY_PROCESSES", "").strip()
    process_count = None
    if raw and raw.lower() != "auto":
        try:
            process_count = int(raw)
        except ValueError as exc:
            raise SystemExit("VF_FLYPPY_BODY_PROCESSES must be 'auto' or a positive integer") from exc
        if process_count < 1:
            raise SystemExit("VF_FLYPPY_BODY_PROCESSES must be positive")

    handles, resolved = spawn_packed_slot_handles(
        population=population,
        process_count=process_count,
        physics_steps=physics_steps,
        timeout_s=timeout_s,
        config=config,
    )
    print(
        "body_process_packing=packed population={} body_processes={} "
        "flies_per_process_mean={:.3f}".format(
            population,
            resolved,
            population / resolved,
        )
    )
    return handles


def main() -> int:
    trainer.spawn_body_processes = _spawn_packed
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
