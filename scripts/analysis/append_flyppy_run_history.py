#!/usr/bin/env python3
"""Append one completed Flyppy run to the stable history CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from virtual_fly.reporting.history import (
    FIELDS,
    build_row,
    merge_history_rows,
    read_existing,
    write_history,
)

DEFAULT_SUMMARY = Path("artifacts/experiments/flyppy-v2/summary.json")
DEFAULT_OUTPUT = Path("reports/flyppy/history.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        raise SystemExit(f"summary not found: {args.summary}")
    row = build_row(json.loads(args.summary.read_text(encoding="utf-8")))
    write_history(args.output, merge_history_rows(read_existing(args.output), row))
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
