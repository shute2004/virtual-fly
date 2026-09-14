#!/usr/bin/env python3
"""Search the preprocessed MaleCNS annotations without inventing neuron roles.

Examples:
  uv run python scripts/data/query_annotations.py --pattern 'PAM|PPL1'
  uv run python scripts/data/query_annotations.py --pattern 'PAM|PPL1' --dopamine-only

The command only searches released annotations / neurotransmitter predictions.
Assigning reward or aversive roles remains an explicit experiment decision.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DOPAMINE_CODE = 4
DEFAULT_COLUMNS = [
    "bodyId",
    "type",
    "flywireType",
    "instance",
    "class",
    "subclass",
    "superclass",
    "side",
    "fruDsx",
    "status",
    "statusLabel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("artifacts/malecns-v1.0"),
        help="preprocessed snapshot directory",
    )
    parser.add_argument("--pattern", required=True, help="case-insensitive regular expression")
    parser.add_argument("--dopamine-only", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations_path = args.snapshot / "annotations.feather"
    body_ids_path = args.snapshot / "body_ids.u64le"
    nt_path = args.snapshot / "neurotransmitters.u8"

    annotations = pd.read_feather(annotations_path)
    body_ids = np.fromfile(body_ids_path, dtype="<u8")
    nt = np.fromfile(nt_path, dtype=np.uint8)
    if len(body_ids) != len(nt):
        raise RuntimeError("body_ids and neurotransmitters lengths differ")

    nt_by_body = pd.Series(nt, index=body_ids, name="nt_code")
    annotations = annotations.copy()
    annotations["nt_code"] = annotations["bodyId"].map(nt_by_body).fillna(0).astype(np.uint8)

    searchable = [column for column in DEFAULT_COLUMNS if column in annotations.columns and column != "bodyId"]
    regex = re.compile(args.pattern, re.IGNORECASE)
    mask = pd.Series(False, index=annotations.index)
    for column in searchable:
        mask |= annotations[column].fillna("").astype(str).str.contains(regex, regex=True)
    if args.dopamine_only:
        mask &= annotations["nt_code"].eq(DOPAMINE_CODE)

    result = annotations[mask]
    columns = [column for column in DEFAULT_COLUMNS if column in result.columns] + ["nt_code"]
    result = result[columns].head(args.limit)

    print(f"matches={int(mask.sum())} shown={len(result)}")
    if len(result):
        print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
