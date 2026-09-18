#!/usr/bin/env python3
"""Packed-body production trainer for Flyppy population learning.

The proven process-body trainer owns all learning/curriculum/checkpoint logic.
This launcher replaces only its body-process factory so several logical fly slots
share one MuJoCo owner process. The default packing rule is the measured M1
optimum family: min(population, 4) body processes.

Viewer demand is handled explicitly by the process trainer's telemetry publisher;
this wrapper does not change the type or semantics of argparse fields.

Set VF_FLYPPY_BODY_PROCESSES to a positive integer to override the default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import train_flyppy_population_process as trainer
from flyppy_packed_slot_adapter import spawn_packed_slot_handles


_RESOLVED_BODY_PROCESSES: int | None = None
_RESOLVED_VISION_MODE: str | None = None
_RESOLVED_VISION_RAYS: int | None = None


def _spawn_packed(
    *,
    population: int,
    physics_steps: int,
    timeout_s: float,
    config: dict[str, object],
):
    global _RESOLVED_BODY_PROCESSES, _RESOLVED_VISION_MODE, _RESOLVED_VISION_RAYS

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
    _RESOLVED_BODY_PROCESSES = int(resolved)
    if handles:
        worker = handles[0]._worker
        _RESOLVED_VISION_MODE = str(getattr(worker, "vision_mode", "raster"))
        _RESOLVED_VISION_RAYS = int(
            getattr(worker, "vision_rays_per_ommatidium", 7)
        )
    print(
        "body_process_packing=packed population={} body_processes={} "
        "flies_per_process_mean={:.3f} vision_mode={} rays_per_ommatidium={}".format(
            population,
            resolved,
            population / resolved,
            _RESOLVED_VISION_MODE,
            _RESOLVED_VISION_RAYS,
        )
    )
    return handles


def _argv_value(name: str, default: str) -> str:
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return str(args[index + 1])
        prefix = f"{name}="
        if arg.startswith(prefix):
            return str(arg.split("=", 1)[1])
    return default


def _output_dir_from_argv() -> Path:
    return Path(_argv_value("--output-dir", "artifacts/experiments/flyppy-v3"))


def _patch_runtime_metadata(output_dir: Path) -> None:
    if _RESOLVED_BODY_PROCESSES is None:
        return

    metadata = {
        "body_runtime": "packed-process",
        "body_processes": int(_RESOLVED_BODY_PROCESSES),
        "body_process_rule": "min(population, 4) unless VF_FLYPPY_BODY_PROCESSES overrides it",
        "vision_runtime": _RESOLVED_VISION_MODE or "raster",
        "vision_rays_per_ommatidium": (
            int(_RESOLVED_VISION_RAYS) if _RESOLVED_VISION_RAYS is not None else None
        ),
        "vision_framebuffer": (_RESOLVED_VISION_MODE or "raster") == "raster",
        "viewer_telemetry": "on-demand",
        "environment_version": _argv_value("--environment-version", "v3"),
        "flight_body_version": _argv_value("--flight-body-version", "v3"),
    }
    for name in ("summary.json", "population-state.json"):
        path = output_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(metadata)
        temp = path.with_name(f".{path.name}.packed.tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)


def main() -> int:
    trainer.spawn_body_processes = _spawn_packed
    result = trainer.main()
    if result == 0:
        _patch_runtime_metadata(_output_dir_from_argv())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
