#!/usr/bin/env python3
"""Build stable MaleCNS groups used at the first brain/body boundary.

The selected cell types come from released MaleCNS annotations. Functional
interpretation is kept explicit: DNg02 is used as a flight-amplitude descending
readout; PAM08/PPL1 remain experimental neuromodulatory groups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOPAMINE_CODE = 4


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

    dng02 = searchable.str.contains(r"\bDNg02(?:_|\b)", case=False, regex=True).to_numpy()
    side = text_column(annotations, "side").str.upper()
    soma_side = text_column(annotations, "somaSide").str.upper()
    left = (side.eq("L") | soma_side.eq("L")).to_numpy()
    right = (side.eq("R") | soma_side.eq("R")).to_numpy()

    dng02_left_records = row_records(annotations, dng02 & left)
    dng02_right_records = row_records(annotations, dng02 & right)
    if not dng02_left_records or not dng02_right_records:
        raise RuntimeError(
            "could not resolve bilateral DNg02 groups from released MaleCNS annotations"
        )

    dopamine = nt == DOPAMINE_CODE
    pam08_mask = dopamine & searchable.str.contains(
        r"\bPAM08(?:_|\b)", case=False, regex=True
    ).to_numpy()
    ppl1_mask = dopamine & searchable.str.contains(
        r"\bPPL1", case=False, regex=True
    ).to_numpy()
    pam08_records = row_records(annotations, pam08_mask)
    ppl1_records = row_records(annotations, ppl1_mask)
    if not pam08_records or not ppl1_records:
        raise RuntimeError("could not resolve PAM08/PPL1 groups")

    bridge = {
        "schema_version": 1,
        "groups": {
            "flight_thrust_left": {
                "body_ids": body_ids(dng02_left_records),
                "modulator_role": 0,
            },
            "flight_thrust_right": {
                "body_ids": body_ids(dng02_right_records),
                "modulator_role": 0,
            },
            "reward_dan": {
                "body_ids": body_ids(pam08_records),
                "modulator_role": 1,
            },
            "aversive_dan": {
                "body_ids": body_ids(ppl1_records),
                "modulator_role": -1,
            },
        },
        "provenance": {
            "flight_thrust": "DNg02 population; used as bilateral flight-amplitude descending readout",
            "reward_dan": "PAM08 annotation candidates; experimental valence assignment",
            "aversive_dan": "PPL1 annotation candidates; experimental valence assignment",
        },
        "resolved": {
            "flight_thrust_left": dng02_left_records,
            "flight_thrust_right": dng02_right_records,
            "reward_dan": pam08_records,
            "aversive_dan": ppl1_records,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bridge, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"flight_thrust_left={len(dng02_left_records)}")
    print(f"flight_thrust_right={len(dng02_right_records)}")
    print(f"reward_dan={len(pam08_records)}")
    print(f"aversive_dan={len(ppl1_records)}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
