#!/usr/bin/env python3
"""Inventory released MaleCNS wing motor neurons without decoding an action.

This script is deliberately descriptive. It does not average motor activity,
construct a policy, or convert a population vector into a wing command. It only
extracts individual released motor-neuron body IDs at the CNS/body boundary and
attaches literature-supported muscle identities where they are known.

The resulting map is a peripheral wiring inventory, not a controller. Unknown or
putative muscle identities remain explicitly unresolved rather than being guessed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Muscle identities are taken from the MANC wing-MN identification literature.
# These labels describe anatomical targets only; they do not define an action,
# torque, gain, sign, or any other motor command.
MUSCLE_TARGETS: dict[str, dict[str, str]] = {
    # Power muscles. MaleCNS type labels group some individually innervated
    # fibers, so the type-level annotation cannot safely assign one body ID to
    # one c/d/e/f or 1a/1b/... fiber without additional evidence.
    "DLMn a, b": {
        "muscle_class": "power",
        "target_muscle": "DLM a,b",
        "mapping_status": "identified",
        "note": "one DLMn a/b motor neuron per side innervates the two dorsal DLM fibers a and b",
    },
    "DLMn c-f": {
        "muscle_class": "power",
        "target_muscle": "DLM c-f",
        "mapping_status": "identified_group",
        "note": "type groups the four DLM motor neurons for fibers c-f; individual fiber identity is not inferred here",
    },
    "DVMn 1a-c": {
        "muscle_class": "power",
        "target_muscle": "DVM1 a-c",
        "mapping_status": "identified_group",
        "note": "type groups the three DVM1 motor neurons; individual a/b/c identity is not inferred here",
    },
    "DVMn 2a, b": {
        "muscle_class": "power",
        "target_muscle": "DVM2 a,b",
        "mapping_status": "identified_group",
        "note": "type groups the two DVM2 motor neurons; individual a/b identity is not inferred here",
    },
    "DVMn 3a, b": {
        "muscle_class": "power",
        "target_muscle": "DVM3 a,b",
        "mapping_status": "identified_group",
        "note": "type groups the two DVM3 motor neurons; individual a/b identity is not inferred here",
    },
    # Direct steering muscles.
    "b1 MN": {"muscle_class": "direct_steering", "target_muscle": "b1", "mapping_status": "identified"},
    "b2 MN": {"muscle_class": "direct_steering", "target_muscle": "b2", "mapping_status": "identified"},
    "b3 MN": {"muscle_class": "direct_steering", "target_muscle": "b3", "mapping_status": "identified"},
    "i1 MN": {"muscle_class": "direct_steering", "target_muscle": "i1", "mapping_status": "identified"},
    "i2 MN": {"muscle_class": "direct_steering", "target_muscle": "i2", "mapping_status": "identified"},
    "iii1 MN": {"muscle_class": "direct_steering", "target_muscle": "iii1", "mapping_status": "identified"},
    "iii3 MN": {"muscle_class": "direct_steering", "target_muscle": "iii3", "mapping_status": "identified"},
    "MNwm35": {
        "muscle_class": "direct_steering",
        "target_muscle": "iii4",
        "mapping_status": "putative",
        "note": "MANC identifies MNwm35 as the putative iii4 motor neuron",
    },
    "hg1 MN": {"muscle_class": "direct_steering", "target_muscle": "hg1", "mapping_status": "identified"},
    "hg2 MN": {"muscle_class": "direct_steering", "target_muscle": "hg2", "mapping_status": "identified"},
    "hg3 MN": {"muscle_class": "direct_steering", "target_muscle": "hg3", "mapping_status": "identified"},
    "hg4 MN": {"muscle_class": "direct_steering", "target_muscle": "hg4", "mapping_status": "identified"},
    # Indirect control / thoracic tension muscles.
    "tp1 MN": {"muscle_class": "indirect_control", "target_muscle": "tp1", "mapping_status": "identified"},
    "tp2 MN": {"muscle_class": "indirect_control", "target_muscle": "tp2", "mapping_status": "identified"},
    "tpn MN": {
        "muscle_class": "indirect_control",
        "target_muscle": "tp1/tp2",
        "mapping_status": "identified_variable",
        "note": "tpn can innervate both tergopleural fibers; do not collapse it onto one fiber",
    },
    "ps1 MN": {"muscle_class": "indirect_control", "target_muscle": "ps1", "mapping_status": "identified"},
    "ps2 MN": {"muscle_class": "indirect_control", "target_muscle": "ps2", "mapping_status": "identified"},
    "TTMn": {"muscle_class": "indirect_control", "target_muscle": "TTM", "mapping_status": "identified"},
    "STTMm": {
        "muscle_class": "indirect_control",
        "target_muscle": "satellite TTM",
        "mapping_status": "identified",
    },
    # MANC explicitly leaves this muscle target unresolved. A pleurosternal
    # target has been hypothesized, but virtual-fly must not promote that
    # hypothesis to an identified connection.
    "MNwm36": {
        "muscle_class": "unknown",
        "target_muscle": "",
        "mapping_status": "unknown",
        "note": "muscle target unresolved in MANC; possible pleurosternal target remains a hypothesis",
    },
}

REFERENCE_URLS = [
    "https://elifesciences.org/articles/96084",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10312520/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10312524/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12637562/",
]


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
        type_name = cell_text(row, "type")
        record: dict[str, object] = {
            "body_id": int(row["bodyId"]),
            "type": type_name,
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

        target = MUSCLE_TARGETS.get(type_name)
        if target is None:
            record.update(
                {
                    "muscle_class": "unknown",
                    "target_muscle": "",
                    "mapping_status": "unknown",
                    "mapping_note": "no literature mapping attached to this released type",
                }
            )
        else:
            record.update(
                {
                    "muscle_class": target["muscle_class"],
                    "target_muscle": target["target_muscle"],
                    "mapping_status": target["mapping_status"],
                }
            )
            if note := target.get("note"):
                record["mapping_note"] = note
        neurons.append(record)

    neurons.sort(
        key=lambda item: (str(item["type"]), str(item["side"]), int(item["body_id"]))
    )
    side_counts = Counter(str(item["side"]) or "unknown" for item in neurons)
    status_counts = Counter(str(item["mapping_status"]) for item in neurons)
    type_to_ids: dict[str, list[int]] = {}
    for item in neurons:
        type_name = str(item["type"]) or "<untyped>"
        type_to_ids.setdefault(type_name, []).append(int(item["body_id"]))

    if side_counts.get("L", 0) == 0 or side_counts.get("R", 0) == 0:
        raise RuntimeError(
            f"wing motor-neuron inventory did not resolve both sides: {dict(side_counts)}"
        )

    payload = {
        "schema_version": 2,
        "dataset": "male-cns:v1.0",
        "purpose": (
            "descriptive inventory of individual released wing-motor neurons and "
            "literature-supported peripheral muscle identities; not an action decoder"
        ),
        "selection": {
            "superclass": "vnc_motor",
            "subclass": "wm",
            "note": (
                "All matching released neurons are preserved. No activity averaging, "
                "population readout, or muscle command is computed here."
            ),
        },
        "provenance": {
            "neuron_identity": "observed: MaleCNS v1.0 annotations/body IDs",
            "muscle_target": "literature: MANC/light-level wing motor-neuron identification",
            "important_boundary": (
                "A muscle identity does not define a torque. Muscle activation dynamics, "
                "attachment/moment arms, and force generation require separate physiological "
                "models and must not be replaced by an external action decoder."
            ),
            "references": REFERENCE_URLS,
        },
        "counts": {
            "neurons": len(neurons),
            "left": side_counts.get("L", 0),
            "right": side_counts.get("R", 0),
            "unknown_side": side_counts.get("unknown", 0),
            "types": len(type_to_ids),
            "identified": status_counts.get("identified", 0),
            "identified_group": status_counts.get("identified_group", 0),
            "identified_variable": status_counts.get("identified_variable", 0),
            "putative": status_counts.get("putative", 0),
            "unknown_mapping": status_counts.get("unknown", 0),
        },
        "types": [
            {
                "type": type_name,
                "body_ids": ids,
                "muscle_target": MUSCLE_TARGETS.get(type_name),
            }
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
    print(f"identified={status_counts.get('identified', 0)}")
    print(f"identified_group={status_counts.get('identified_group', 0)}")
    print(f"identified_variable={status_counts.get('identified_variable', 0)}")
    print(f"putative={status_counts.get('putative', 0)}")
    print(f"unknown_mapping={status_counts.get('unknown', 0)}")
    for type_name, ids in sorted(type_to_ids.items()):
        joined = ",".join(str(value) for value in ids)
        target = MUSCLE_TARGETS.get(type_name)
        if target is None:
            target_text = "UNRESOLVED"
        elif target["target_muscle"]:
            target_text = f"{target['target_muscle']} ({target['mapping_status']})"
        else:
            target_text = target["mapping_status"]
        print(f"type={type_name} body_ids={joined} target={target_text}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
