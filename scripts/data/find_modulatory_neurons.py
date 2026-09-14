#!/usr/bin/env python3
"""Discover annotated MaleCNS neuromodulatory neuron candidates.

This utility does not assign reward/punishment semantics. It only joins the
prepared snapshot's body IDs, released annotations and consensus transmitter
codes, then emits dopamine-neuron candidates whose annotation text matches a
query such as PAM or PPL1.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DOPAMINE_CODE = 4
SEARCH_COLUMNS = (
    "type",
    "flywireType",
    "instance",
    "class",
    "subclass",
    "superclass",
    "nerve",
    "receptorType",
    "fruDsx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument("--query", action="append", default=["PAM", "PPL1"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/malecns-v1.0/modulatory-candidates.json"))
    return parser.parse_args()


def load_u64(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u8")


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot
    body_ids = load_u64(snapshot / "body_ids.u64le")
    nt = np.fromfile(snapshot / "neurotransmitters.u8", dtype=np.uint8)
    annotations = pd.read_feather(snapshot / "annotations.feather")

    if len(body_ids) != len(nt) or len(body_ids) != len(annotations):
        raise RuntimeError("snapshot metadata arrays are not aligned")
    if "bodyId" not in annotations.columns:
        raise RuntimeError("annotations.feather has no bodyId column")

    annotation_ids = annotations["bodyId"].astype(np.uint64).to_numpy()
    if not np.array_equal(annotation_ids, body_ids):
        raise RuntimeError("annotations.feather bodyId order does not match snapshot body_ids")

    columns = [name for name in SEARCH_COLUMNS if name in annotations.columns]
    text = annotations[columns].fillna("").astype(str).agg(" | ".join, axis=1)
    dopamine = nt == DOPAMINE_CODE

    groups: dict[str, list[dict[str, object]]] = {}
    for query in args.query:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matched = dopamine & text.str.contains(pattern, regex=True, na=False).to_numpy()
        indices = np.flatnonzero(matched)
        rows: list[dict[str, object]] = []
        for index in indices:
            record: dict[str, object] = {
                "index": int(index),
                "body_id": int(body_ids[index]),
            }
            for column in columns:
                value = annotations.iloc[index][column]
                if pd.notna(value) and str(value):
                    record[column] = str(value)
            rows.append(record)
        groups[query] = rows
        print(f"{query}: {len(rows)} dopamine candidate(s)")

    result = {
        "dataset": "male-cns:v1.0",
        "semantics": "annotation discovery only; valence must be assigned by experiment design",
        "queries": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
