#!/usr/bin/env python3
"""Summarize steering-muscle activation saturation from an offline playback.

Read-only diagnostic: this does not modify checkpoints or simulation state.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("playback", type=Path)
    args = p.parse_args()
    payload = json.loads(args.playback.read_text(encoding="utf-8"))
    windows = (
        ("pre_gate1", 1, 72),
        ("gate1_to_gate2", 73, 122),
        ("post_gate2", 123, 10**9),
    )
    for label, lo, hi in windows:
        values: dict[str, list[float]] = defaultdict(list)
        for frame in payload.get("frames", []):
            step = int(frame.get("control_step", -1))
            if not lo <= step <= hi:
                continue
            steering = (frame.get("motor") or {}).get("steering") or {}
            for key, value in steering.items():
                values[str(key)].append(float(value))
        print(label)
        for key in sorted(values):
            rows = values[key]
            print(
                f"  {key:16s} mean={sum(rows)/len(rows):.4f} "
                f"min={min(rows):.4f} max={max(rows):.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
