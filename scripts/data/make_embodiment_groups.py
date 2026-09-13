#!/usr/bin/env python3
"""Build stable MaleCNS groups used at the first brain/body boundary.

The selected cell types come from released MaleCNS annotations. Functional
interpretation is kept explicit: DNg02 is used as a flight-amplitude descending
readout; PAM08/PPL1 remain experimental neuromodulatory groups; T4/T5 c/d
populations are used as the first vertical-motion sensory interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOPAMINE_CODE = 4
VERTICAL_MOTION_TYPES = ("T4c", "T4d", "T5c", "T5d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/malecns-v1.0")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"),
    )
    return parser.parse_args()


def text_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[name].fillna("").astype(str)


def row_records(frame: pd.DataFrame, mask: np.ndarray) -> list[dict[str, object]]:
    rows = frame.loc[mask]
    result: list[dict[str, object]] = []
    for index, row in rows.iterrows():
        item: dict[str, object] = {
            "index": int(index),
            "body_id": int(row["bodyId"]),
        }
        for column in ("type", "flywireType", "instance", "side", "somaSide"):
            if column in row and pd.notna(row[column]) and str(row[column]):
                item[column] = str(row[column])
        result.append(item)
    return result


def body_ids(records: list[dict[str, object]]) -> list[int]:
    return [int(record["body_id"]) for record in records]


def add_group(
    groups: dict[str, dict[str, object]],
    resolved: dict[str, list[dict[str, object]]],
    name: str,
    records: list[dict[str, object]],
    *,
    modulator_role: int = 0,
) -> None:
    if not records:
        raise RuntimeError(f"group {name} resolved to zero neurons")
    groups[name] = {
        "body_ids": body_ids(records),
        "modulator_role": modulator_role,
    }
    resolved[name] = records


def main() -> int:
    args = parse_args()
    annotations = pd.read_feather(args.snapshot / "annotations.feather")
    body_ids_snapshot = np.fromfile(args.snapshot / "body_ids.u64le", dtype="<u8")
    nt = np.fromfile(args.snapshot / "neurotransmitters.u8", dtype=np.uint8)

    if len(annotations) != len(body_ids_snapshot) or len(nt) != len(body_ids_snapshot):
        raise RuntimeError("snapshot metadata arrays are not aligned")
    annotation_ids = annotations["bodyId"].astype(np.uint64).to_numpy()
    if not np.array_equal(annotation_ids, body_ids_snapshot):
        raise RuntimeError("annotations order does not match body_ids.u64le")

    type_text = text_column(annotations, "type")
    flywire_type = text_column(annotations, "flywireType")
    instance = text_column(annotations, "instance")
    searchable = type_text + " | " + flywire_type + " | " + instance

    side = text_column(annotations, "side").str.upper()
    soma_side = text_column(annotations, "somaSide").str.upper()
    left = (side.eq("L") | soma_side.eq("L")).to_numpy(copy=True)
    right = (side.eq("R") | soma_side.eq("R")).to_numpy(copy=True)

    groups: dict[str, dict[str, object]] = {}
    resolved: dict[str, list[dict[str, object]]] = {}

    dng02 = searchable.str.contains(
        r"\bDNg02(?:_|\b)", case=False, regex=True
    ).to_numpy(copy=True)
    add_group(
        groups,
        resolved,
        "flight_thrust_left",
        row_records(annotations, dng02 & left),
    )
    add_group(
        groups,
        resolved,
        "flight_thrust_right",
        row_records(annotations, dng02 & right),
    )

    dopamine = nt == DOPAMINE_CODE
    pam08_mask = dopamine & searchable.str.contains(
        r"\bPAM08(?:_|\b)", case=False, regex=True
    ).to_numpy(copy=True)
    ppl1_mask = dopamine & searchable.str.contains(
        r"\bPPL1", case=False, regex=True
    ).to_numpy(copy=True)
    add_group(
        groups,
        resolved,
        "reward_dan",
        row_records(annotations, pam08_mask),
        modulator_role=1,
    )
    add_group(
        groups,
        resolved,
        "aversive_dan",
        row_records(annotations, ppl1_mask),
        modulator_role=-1,
    )

    # T4/T5 c and d are the vertical-motion channels. We keep ON (T4) and OFF
    # (T5) pathways separate and preserve the two optic-lobe hemispheres.
    for visual_type in VERTICAL_MOTION_TYPES:
        type_mask = searchable.str.contains(
            rf"\b{visual_type}(?:_|\b)", case=False, regex=True
        ).to_numpy(copy=True)
        for side_name, side_mask in (("left", left), ("right", right)):
            add_group(
                groups,
                resolved,
                f"vision_{visual_type.lower()}_{side_name}",
                row_records(annotations, type_mask & side_mask),
            )

    bridge = {
        "schema_version": 1,
        "groups": groups,
        "provenance": {
            "flight_thrust": (
                "DNg02 population; used as bilateral flight-amplitude descending readout"
            ),
            "reward_dan": (
                "PAM08 annotation candidates; experimental valence assignment"
            ),
            "aversive_dan": (
                "PPL1 annotation candidates; experimental valence assignment"
            ),
            "vertical_motion": (
                "T4c/T5c are upward-motion channels and T4d/T5d are "
                "downward-motion channels; population-level input is provisional "
                "until an individual MaleCNS retinotopic mapping is available"
            ),
        },
        "resolved": resolved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bridge, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name in sorted(groups):
        print(f"{name}={len(resolved[name])}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
