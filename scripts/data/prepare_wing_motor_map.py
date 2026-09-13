#!/usr/bin/env python3
"""Inventory released MaleCNS wing motor neurons without decoding an action.

This script is deliberately descriptive. It does not average motor activity,
construct a policy, or convert a population vector into a wing command. It only
extracts individual released motor-neuron body IDs at the CNS/body boundary so a
later peripheral muscle model can connect each neuron to its biological target.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/malecns-v1.0")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"),
    )
    return parser.parse_args()


def text_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[name].fillna("").astype(str).str.strip()


def cell_text(row: pd.Series, name: str) -> str:
    if name not in row or pd.isna(row[name]):
        return ""
    return str(row[name]).strip()


def resolved_side(row: pd.Series) -> str:
    for column in ("side", "somaSide", "rootSide"):
        value = cell_text(row, column).upper()
        if value in ("L", "R"):
            return value
    return ""


def main() -> int:
    args = parse_args()
    annotation_path = args.snapshot / "annotations.feather"
    body_path = args.snapshot / "body_ids.u64le"
    if not annotation_path.exists() or not body_path.exists():
        raise SystemExit(
            f"MaleCNS snapshot is incomplete at {args.snapshot}; run bootstrap first"
        )

    annotations = pd.read_feather(annotation_path)
    bodies = np.fromfile(body_path, dtype="<u8")
    if len(annotations) != len(bodies):
        raise RuntimeError("annotations/body_ids length mismatch")
    annotation_ids = annotations["bodyId"].astype(np.uint64).to_numpy()
    if not np.array_equal(annotation_ids, bodies):
        raise RuntimeError("annotations are not aligned with body_ids.u64le")

    superclass = text_column(annotations, "superclass").str.lower()
    subclass = text_column(annotations, "subclass").str.lower()
    mask = superclass.eq("vnc_motor") & subclass.eq("wm")
    rows = annotations.loc[mask].copy()
    if rows.empty:
        raise RuntimeError(
            "no MaleCNS neurons matched superclass=vnc_motor, subclass=wm; "
            "refusing to guess a motor population"
        )

    neurons: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        record: dict[str, object] = {
            "body_id": int(row["bodyId"]),
            "type": cell_text(row, "type"),
            "side": resolved_side(row),
        }
        for column in (
            "flywireType",
            "instance",
            "class",
            "subclass",
            "superclass",
            "nerve",
            "entryNerve",
            "exitNerve",
            "receptorType",
        ):
            value = cell_text(row, column)
            if value:
                record[column] = value
        neurons.append(record)

    neurons.sort(key=lambda item: (str(item["type"]), str(item["side"]), int(item["body_id"])))
    side_counts = Counter(str(item["side"]) or "unknown" for item in neurons)
    type_to_ids: dict[str, list[int]] = {}
    for item in neurons:
        type_name = str(item["type"]) or "<untyped>"
        type_to_ids.setdefault(type_name, []).append(int(item["body_id"]))

    if side_counts.get("L", 0) == 0 or side_counts.get("R", 0) == 0:
        raise RuntimeError(
            f"wing motor-neuron inventory did not resolve both sides: {dict(side_counts)}"
        )

    payload = {
        "schema_version": 1,
        "dataset": "male-cns:v1.0",
        "purpose": (
            "descriptive inventory of individual released wing-motor neurons; "
            "not an action decoder"
        ),
        "selection": {
            "superclass": "vnc_motor",
            "subclass": "wm",
            "note": (
                "All matching released neurons are preserved. No activity averaging, "
                "population readout, or muscle command is computed here."
            ),
        },
        "counts": {
            "neurons": len(neurons),
            "left": side_counts.get("L", 0),
            "right": side_counts.get("R", 0),
            "unknown_side": side_counts.get("unknown", 0),
            "types": len(type_to_ids),
        },
        "types": [
            {"type": type_name, "body_ids": ids}
            for type_name, ids in sorted(type_to_ids.items())
        ],
        "neurons": neurons,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wing_motor_neurons={len(neurons)}")
    print(f"left={side_counts.get('L', 0)}")
    print(f"right={side_counts.get('R', 0)}")
    print(f"unknown_side={side_counts.get('unknown', 0)}")
    print(f"types={len(type_to_ids)}")
    for type_name, ids in sorted(type_to_ids.items()):
        joined = ",".join(str(value) for value in ids)
        print(f"type={type_name} body_ids={joined}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
