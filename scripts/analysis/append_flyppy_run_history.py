#!/usr/bin/env python3
"""Append one Flyppy run summary to a compact Git-friendly history CSV.

The trainer's ignored ``summary.json`` already records wall-clock elapsed time
and episode results.  This script preserves one row per completed run under
``reports/flyppy/history.csv`` so learning speed and vertical flight behavior can
be compared across runs without committing checkpoints or trajectories.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path("artifacts/experiments/flyppy-v2/summary.json")
DEFAULT_OUTPUT = Path("reports/flyppy/history.csv")

FIELDS = [
    "recorded_at_utc",
    "run_key",
    "experiment",
    "environment_version",
    "motor_boundary",
    "backend",
    "population",
    "body_runtime",
    "body_processes",
    "vision_runtime",
    "vision_rays_per_ommatidium",
    "vision_framebuffer",
    "episode_start",
    "episode_end",
    "episode_count",
    "elapsed_seconds",
    "seconds_per_episode",
    "total_control_steps",
    "control_steps_per_second",
    "control_dt_seconds",
    "simulated_seconds",
    "simulation_realtime_factor",
    "success_episodes",
    "success_rate",
    "total_passed_gates",
    "max_passed_gates",
    "collisions",
    "finished_episodes",
    "max_x_mm",
    "episodes_ever_above_spawn",
    "max_altitude_gain_mm",
    "worst_altitude_loss_mm",
    "mean_max_altitude_gain_mm",
    "checkpoint_neural_step",
    "curriculum_mode",
    "curriculum_episodes_total",
    "spawn_x_mm_after_run",
    "spawn_z_mm_after_run",
    "initial_speed_mm_s_after_run",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def build_row(payload: dict[str, Any]) -> dict[str, Any]:
    episodes = list(payload.get("episode_results", []))
    if not episodes:
        raise SystemExit("summary contains no episode_results")

    episode_start = int(payload.get("episode_start", episodes[0].get("episode", 0)))
    episode_end = int(payload.get("episode_end", episodes[-1].get("episode", episode_start)))
    episode_count = len(episodes)
    elapsed = finite_float(payload.get("elapsed_seconds"))
    total_steps = sum(int(row.get("control_steps", 0)) for row in episodes)
    control_dt = finite_float(payload.get("control_dt_seconds"))
    simulated_seconds = total_steps * control_dt
    successes = sum(int(row.get("passed_gates", 0)) > 0 for row in episodes)
    total_passed = sum(int(row.get("passed_gates", 0)) for row in episodes)
    max_passed = max(int(row.get("passed_gates", 0)) for row in episodes)
    collisions = sum(bool(row.get("collision", False)) for row in episodes)
    finishes = sum(bool(row.get("finished", False)) for row in episodes)
    max_x = max(finite_float(row.get("max_x_mm"), float("-inf")) for row in episodes)

    altitude_gains = [
        finite_float(row.get("max_z_mm")) - finite_float(row.get("spawn_z_mm"))
        for row in episodes
    ]
    altitude_losses = [
        finite_float(row.get("spawn_z_mm")) - finite_float(row.get("min_z_mm"))
        for row in episodes
    ]
    episodes_above_spawn = sum(gain > 1e-6 for gain in altitude_gains)
    max_altitude_gain = max(altitude_gains)
    worst_altitude_loss = max(altitude_losses)
    mean_max_altitude_gain = sum(altitude_gains) / episode_count

    curriculum = dict(payload.get("curriculum", {}))
    experiment = str(payload.get("experiment", "-"))
    environment_version = str(payload.get("environment_version", "-"))
    motor_boundary = str(payload.get("motor_boundary", "-"))
    checkpoint_step = payload.get("checkpoint_neural_step", "")
    population = payload.get("population", "")
    body_runtime = payload.get("body_runtime", "")
    body_processes = payload.get("body_processes", "")
    vision_runtime = payload.get("vision_runtime", "")
    vision_rays = payload.get("vision_rays_per_ommatidium", "")
    vision_framebuffer = payload.get("vision_framebuffer", "")

    run_key = "|".join(
        [
            experiment,
            environment_version,
            motor_boundary,
            str(episode_start),
            str(episode_end),
            str(checkpoint_step),
        ]
    )

    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_key": run_key,
        "experiment": experiment,
        "environment_version": environment_version,
        "motor_boundary": motor_boundary,
        "backend": payload.get("backend", "-"),
        "population": population,
        "body_runtime": body_runtime,
        "body_processes": body_processes,
        "vision_runtime": vision_runtime,
        "vision_rays_per_ommatidium": vision_rays,
        "vision_framebuffer": (
            str(bool(vision_framebuffer)).lower()
            if isinstance(vision_framebuffer, bool)
            else vision_framebuffer
        ),
        "episode_start": episode_start,
        "episode_end": episode_end,
        "episode_count": episode_count,
        "elapsed_seconds": f"{elapsed:.6f}",
        "seconds_per_episode": f"{elapsed / episode_count:.6f}" if elapsed > 0 else "",
        "total_control_steps": total_steps,
        "control_steps_per_second": f"{total_steps / elapsed:.6f}" if elapsed > 0 else "",
        "control_dt_seconds": f"{control_dt:.9f}" if control_dt > 0 else "",
        "simulated_seconds": f"{simulated_seconds:.6f}" if control_dt > 0 else "",
        "simulation_realtime_factor": f"{simulated_seconds / elapsed:.6f}" if elapsed > 0 and control_dt > 0 else "",
        "success_episodes": successes,
        "success_rate": f"{successes / episode_count:.6f}",
        "total_passed_gates": total_passed,
        "max_passed_gates": max_passed,
        "collisions": collisions,
        "finished_episodes": finishes,
        "max_x_mm": f"{max_x:.6f}",
        "episodes_ever_above_spawn": episodes_above_spawn,
        "max_altitude_gain_mm": f"{max_altitude_gain:.6f}",
        "worst_altitude_loss_mm": f"{worst_altitude_loss:.6f}",
        "mean_max_altitude_gain_mm": f"{mean_max_altitude_gain:.6f}",
        "checkpoint_neural_step": checkpoint_step,
        "curriculum_mode": payload.get("curriculum_mode", curriculum.get("curriculum_mode", "-")),
        "curriculum_episodes_total": curriculum.get("curriculum_episodes", ""),
        "spawn_x_mm_after_run": curriculum.get("spawn_x_mm", ""),
        "spawn_z_mm_after_run": curriculum.get("spawn_z_mm", ""),
        "initial_speed_mm_s_after_run": curriculum.get("initial_speed_mm_s", ""),
    }


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        raise SystemExit(f"summary not found: {args.summary}")
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    row = build_row(payload)

    rows = read_existing(args.output)
    rows = [existing for existing in rows if existing.get("run_key") != row["run_key"]]
    rows.append({key: str(row.get(key, "")) for key in FIELDS})

    # Rewriting with the current field list intentionally migrates older rows:
    # newly introduced runtime fields stay blank for historical runs whose
    # summaries did not record them.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: existing.get(key, "") for key in FIELDS} for existing in rows)

    print(f"flyppy_run_history={args.output}")
    print(f"run_key={row['run_key']}")
    print(f"vision_runtime={row['vision_runtime']}")
    print(f"vision_rays_per_ommatidium={row['vision_rays_per_ommatidium']}")
    print(f"elapsed_seconds={row['elapsed_seconds']}")
    print(f"control_steps_per_second={row['control_steps_per_second']}")
    print(f"success_rate={row['success_rate']}")
    print(f"episodes_ever_above_spawn={row['episodes_ever_above_spawn']}")
    print(f"max_altitude_gain_mm={row['max_altitude_gain_mm']}")
    print(f"worst_altitude_loss_mm={row['worst_altitude_loss_mm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
