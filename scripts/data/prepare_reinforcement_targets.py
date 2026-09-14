#!/usr/bin/env python3
"""Inventory released MaleCNS dopamine neurons implicated in reinforcement.

This script does not assign a numerical reward sign and does not change synaptic
weights. It only resolves literature-named dopaminergic cell types to released
MaleCNS body IDs so experiments can inject current into explicit neurons rather
than passing a scalar reward into the plasticity rule.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


DOPAMINE_CODE = 4

# Literature classifications used only to identify candidate electrical-
# stimulation targets. They are not numerical valence labels inside the neural
# runtime. Different DANs remain different because they project to different
# anatomical compartments in the released connectome.
REWARD_CANDIDATE_TYPES = (
    "PAM01",
    "PAM02",
    "PAM04",
    "PAM05",
    "PAM06",
    "PAM07",
    "PAM08",
    "PAM09",
    "PAM10",
    "PAM11",
    "PAM15",
)
PUNISHMENT_CANDIDATE_TYPES = (
    "PAM12",
    "PAM13",
    "PAM14",
    "PPL101",
    "PPL103",
    "PPL106",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/malecns-v1.0")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns-v1.0/reinforcement-dans-v0.json"),
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


def record(row: pd.Series, classification: str) -> dict[str, object]:
    item: dict[str, object] = {
        "body_id": int(row["bodyId"]),
        "type": cell_text(row, "type"),
        "side": resolved_side(row),
        "literature_candidate_class": classification,
    }
    for column in (
        "flywireType",
        "instance",
        "class",
        "subclass",
        "superclass",
    ):
        value = cell_text(row, column)
        if value:
            item[column] = value
    return item


def main() -> int:
    args = parse_args()
    annotation_path = args.snapshot / "annotations.feather"
    body_path = args.snapshot / "body_ids.u64le"
    nt_path = args.snapshot / "neurotransmitters.u8"
    if not annotation_path.exists() or not body_path.exists() or not nt_path.exists():
        raise SystemExit(
            f"MaleCNS snapshot is incomplete at {args.snapshot}; run bootstrap first"
        )

    annotations = pd.read_feather(annotation_path)
    bodies = np.fromfile(body_path, dtype="<u8")
    nt = np.fromfile(nt_path, dtype=np.uint8)
    if len(annotations) != len(bodies) or len(nt) != len(bodies):
        raise RuntimeError("snapshot metadata arrays are not aligned")
    annotation_ids = annotations["bodyId"].astype(np.uint64).to_numpy()
    if not np.array_equal(annotation_ids, bodies):
        raise RuntimeError("annotations are not aligned with body_ids.u64le")

    types = text_column(annotations, "type")
    dopamine = nt == DOPAMINE_CODE
    reward_set = set(REWARD_CANDIDATE_TYPES)
    punishment_set = set(PUNISHMENT_CANDIDATE_TYPES)

    candidates: list[dict[str, object]] = []
    for index in np.flatnonzero(dopamine):
        row = annotations.iloc[int(index)]
        type_name = str(types.iloc[int(index)])
        if type_name in reward_set:
            candidates.append(record(row, "reward"))
        elif type_name in punishment_set:
            candidates.append(record(row, "punishment"))

    if not any(item["literature_candidate_class"] == "reward" for item in candidates):
        raise RuntimeError(
            "no literature-listed reward DAN types were found in the released snapshot; "
            "refusing to invent a replacement"
        )
    if not any(item["literature_candidate_class"] == "punishment" for item in candidates):
        raise RuntimeError(
            "no literature-listed punishment DAN types were found in the released snapshot; "
            "refusing to invent a replacement"
        )

    candidates.sort(
        key=lambda item: (
            str(item["literature_candidate_class"]),
            str(item["type"]),
            str(item["side"]),
            int(item["body_id"]),
        )
    )
    class_counts = Counter(str(item["literature_candidate_class"]) for item in candidates)
    type_to_ids: dict[tuple[str, str], list[int]] = {}
    for item in candidates:
        key = (str(item["literature_candidate_class"]), str(item["type"]))
        type_to_ids.setdefault(key, []).append(int(item["body_id"]))

    # Also expose every released dopaminergic PAM/PPL1 type for audit, including
    # types that are intentionally not placed in the candidate lists above.
    family_mask = dopamine & types.str.match(r"^(?:PAM|PPL1)", case=False, na=False).to_numpy()
    released_family_counts = Counter(types.iloc[np.flatnonzero(family_mask)].tolist())

    payload = {
        "schema_version": 1,
        "dataset": "male-cns:v1.0",
        "purpose": (
            "resolve explicit dopaminergic electrical-stimulation candidates; "
            "no scalar reward or signed modulator value is produced"
        ),
        "literature_candidate_types": {
            "reward": list(REWARD_CANDIDATE_TYPES),
            "punishment": list(PUNISHMENT_CANDIDATE_TYPES),
        },
        "counts": {
            "reward_candidates": class_counts.get("reward", 0),
            "punishment_candidates": class_counts.get("punishment", 0),
        },
        "resolved_candidates": candidates,
        "released_pam_ppl1_type_counts": dict(sorted(released_family_counts.items())),
        "important_boundary": (
            "The experiment may inject current into selected body IDs. The neural runtime "
            "must not receive reward=+1/punishment=-1 or any equivalent external valence sign."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"reward_candidates={class_counts.get('reward', 0)}")
    print(f"punishment_candidates={class_counts.get('punishment', 0)}")
    for (classification, type_name), ids in sorted(type_to_ids.items()):
        joined = ",".join(str(value) for value in ids)
        print(f"{classification}_type={type_name} body_ids={joined}")
    print("released_dopamine_pam_ppl1_types:")
    for type_name, count in sorted(released_family_counts.items()):
        print(f"  {type_name}={count}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
