#!/usr/bin/env python3
"""Build stable MaleCNS groups used at experimental boundaries.

The selected cell types come from released MaleCNS annotations. Group names are
only experiment-side handles for applying current to explicit neurons or for
legacy diagnostics. They do not assign positive/negative numerical valence to
neurons and they are not an action decoder.

The target motor boundary uses individual released wing motor-neuron body IDs.
DNg02 and T4/T5 groups remain only for historical/smoke-test compatibility; the
Flyppy learning path neither averages DNg02 activity nor injects T4/T5 features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOPAMINE_CODE = 4
VERTICAL_MOTION_TYPES = ("T4c", "T4d", "T5c", "T5d")

# Reinforcement stimulation is targeted at named MB-compartment DAN types rather
# than at an abstract signed reward variable. PAM01 corresponds to PAM-gamma5,
# a reward-associated compartment. Aversive reinforcement uses the released
# PPL1 punishment ensemble spanning gamma1pedc, gamma2alpha'1, and the additional
# released PPL106 pair. The runtime receives only injected current at these
# released body IDs and derives dopamine from anatomy.
REWARD_DAN_TYPE = "PAM01"
AVERSIVE_DAN_TYPES = ("PPL101", "PPL103", "PPL106")


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
) -> None:
    if not records:
        raise RuntimeError(f"group {name} resolved to zero neurons")
    groups[name] = {"body_ids": body_ids(records)}
    resolved[name] = records


def exact_dopamine_type_mask(
    annotations: pd.DataFrame,
    nt: np.ndarray,
    type_name: str,
) -> np.ndarray:
    annotated_type = text_column(annotations, "type")
    return (nt == DOPAMINE_CODE) & annotated_type.eq(type_name).to_numpy(copy=True)


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

    # Legacy diagnostic only. The target Flyppy motor path never reads these
    # spike fractions; it requests each selected wing-MN body ID independently.
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

    reward_mask = exact_dopamine_type_mask(annotations, nt, REWARD_DAN_TYPE)
    aversive_mask = np.zeros(len(annotations), dtype=bool)
    for type_name in AVERSIVE_DAN_TYPES:
        aversive_mask |= exact_dopamine_type_mask(annotations, nt, type_name)
    add_group(
        groups,
        resolved,
        "reward_dan",
        row_records(annotations, reward_mask),
    )
    add_group(
        groups,
        resolved,
        "aversive_dan",
        row_records(annotations, aversive_mask),
    )

    # Legacy diagnostic groups only. The target sensory boundary stimulates
    # individual R1-R6 body IDs and leaves motion selectivity to the CNS.
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
        "schema_version": 2,
        "groups": groups,
        "provenance": {
            "flight_thrust": (
                "DNg02 population; legacy diagnostic only. Target motor output reads "
                "individual released wing motor-neuron body IDs and never feeds this "
                "population average into behavior."
            ),
            "reward_dan": (
                f"{REWARD_DAN_TYPE} dopaminergic type (PAM-gamma5 reward-associated MB "
                "compartment). The group identifies neurons to receive experimental current; "
                "no +1 reward or signed valence enters the neural runtime."
            ),
            "aversive_dan": (
                "PPL1 punishment DAN ensemble (PPL101/PPL103/PPL106). The group identifies "
                "released neurons to receive experimental current; no -1 punishment or signed "
                "valence enters the neural runtime."
            ),
            "reinforcement_references": [
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC10945696/",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC7443709/",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC7618526/",
            ],
            "vertical_motion": (
                "legacy T4/T5 diagnostic groups; not used by current R1-R6 Flyppy sensory input"
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
    print(f"reward_dan_type={REWARD_DAN_TYPE}")
    print(f"aversive_dan_types={','.join(AVERSIVE_DAN_TYPES)}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
