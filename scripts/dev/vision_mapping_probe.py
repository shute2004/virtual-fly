#!/usr/bin/env python3
"""Inspect released MaleCNS annotations for optic-lobe mapping fields.

This does not invent a retinotopic mapping. It reports what spatial/column-like
metadata are actually available locally so the next implementation can choose a
mapping backed by released data rather than neuron ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/malecns-v1.0")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/vision-mapping-probe.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = pd.read_feather(args.snapshot / "annotations.feather")
    columns = list(annotations.columns)

    spatial_keywords = (
        "optic",
        "column",
        "eye",
        "soma",
        "position",
        "location",
        "roi",
        "coord",
        "x",
        "y",
        "z",
    )
    candidate_columns = [
        name for name in columns if any(key in name.lower() for key in spatial_keywords)
    ]

    type_column = None
    for candidate in ("type", "flywireType", "instance"):
        if candidate in annotations.columns:
            type_column = candidate
            break

    l1_count = 0
    if type_column is not None:
        l1_count = int(
            annotations[type_column]
            .fillna("")
            .astype(str)
            .str.fullmatch(r"L1(?:_[LR])?", case=False)
            .sum()
        )

    report = {
        "annotation_columns": columns,
        "spatial_candidate_columns": candidate_columns,
        "l1_count_from_primary_type_column": l1_count,
        "primary_type_column": type_column,
        "decision_rule": (
            "Do not assign FlyGym ommatidia to MaleCNS neurons by row/index order. "
            "Use released spatial/eyemap data or a documented cross-dataset mapping."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"annotation_columns={len(columns)}")
    print(f"spatial_candidate_columns={candidate_columns}")
    print(f"l1_count={l1_count}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
