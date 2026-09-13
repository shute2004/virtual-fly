#!/usr/bin/env python3
"""Build a first real MaleCNS associative-conditioning experiment config.

The selected cue neurons, DANs and readout neurons all come from released
MaleCNS annotations. This script only identifies anatomical groups; the
learning rule remains inside the neural simulator.
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
    "hemibrainType",
    "mancType",
    "instance",
    "class",
    "subclass",
    "superclass",
    "synonyms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns-v1.0/conditioning-v0.json"),
    )
    parser.add_argument("--reward-cue", default=r"(?:^|[^A-Za-z0-9])DA1_lPN(?:[^A-Za-z0-9]|$)|DA1 lPN")
    parser.add_argument("--aversive-cue", default=r"(?:^|[^A-Za-z0-9])DL3_lPN(?:[^A-Za-z0-9]|$)|DL3 lPN")
    parser.add_argument("--reward-dan", default=r"(?:^|[^A-Za-z0-9])PAM08(?:[^A-Za-z0-9]|$)")
    parser.add_argument("--aversive-dan", default=r"PPL1")
    parser.add_argument("--readout", default=r"MBON")
    parser.add_argument("--max-cue-neurons", type=int, default=64)
    parser.add_argument("--max-dan-neurons", type=int, default=64)
    parser.add_argument("--max-readout-neurons", type=int, default=256)
    return parser.parse_args()


def combined_text(frame: pd.DataFrame) -> pd.Series:
    columns = [name for name in SEARCH_COLUMNS if name in frame.columns]
    if not columns:
        raise RuntimeError("no searchable annotation columns found")
    return frame[columns].fillna("").astype(str).agg(" | ".join, axis=1)


def select(
    frame: pd.DataFrame,
    text: pd.Series,
    pattern: str,
    limit: int,
    *,
    dopamine_only: bool,
) -> list[dict[str, object]]:
    regex = re.compile(pattern, re.IGNORECASE)
    mask = text.str.contains(regex, regex=True, na=False).to_numpy(dtype=bool, copy=True)
    if dopamine_only:
        mask = mask & (frame["nt_code"].to_numpy() == DOPAMINE_CODE)
    indices = np.flatnonzero(mask)[:limit]
    rows: list[dict[str, object]] = []
    for index in indices:
        row = frame.iloc[int(index)]
        record: dict[str, object] = {
            "index": int(index),
            "body_id": int(row["bodyId"]),
        }
        for column in SEARCH_COLUMNS:
            if column in frame.columns:
                value = row[column]
                if pd.notna(value) and str(value):
                    record[column] = str(value)
        rows.append(record)
    return rows


def require(name: str, rows: list[dict[str, object]], pattern: str) -> None:
    if rows:
        print(f"{name}: {len(rows)} neuron(s)")
        return
    raise RuntimeError(f"no {name} neurons matched pattern: {pattern!r}")


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot
    annotations = pd.read_feather(snapshot / "annotations.feather")
    body_ids = np.fromfile(snapshot / "body_ids.u64le", dtype="<u8")
    transmitters = np.fromfile(snapshot / "neurotransmitters.u8", dtype=np.uint8)

    if len(annotations) != len(body_ids) or len(body_ids) != len(transmitters):
        raise RuntimeError("snapshot metadata arrays are not aligned")
    if not np.array_equal(annotations["bodyId"].astype(np.uint64).to_numpy(), body_ids):
        raise RuntimeError("annotations and body_ids are not in the same order")

    annotations = annotations.copy()
    annotations["nt_code"] = transmitters
    text = combined_text(annotations)

    reward_cue = select(
        annotations, text, args.reward_cue, args.max_cue_neurons, dopamine_only=False
    )
    aversive_cue = select(
        annotations, text, args.aversive_cue, args.max_cue_neurons, dopamine_only=False
    )
    reward_dan = select(
        annotations, text, args.reward_dan, args.max_dan_neurons, dopamine_only=True
    )
    aversive_dan = select(
        annotations, text, args.aversive_dan, args.max_dan_neurons, dopamine_only=True
    )
    readout = select(
        annotations, text, args.readout, args.max_readout_neurons, dopamine_only=False
    )

    require("reward cue", reward_cue, args.reward_cue)
    require("aversive cue", aversive_cue, args.aversive_cue)
    require("reward DAN", reward_dan, args.reward_dan)
    require("aversive DAN", aversive_dan, args.aversive_dan)
    require("MBON readout", readout, args.readout)

    config = {
        "schema_version": 1,
        "dataset": "male-cns:v1.0",
        "experiment": "olfactory_associative_conditioning_v0",
        "notes": [
            "DA1_lPN and DL3_lPN are released annotated olfactory projection-neuron populations.",
            "PAM08 is used as an appetitive-reinforcement candidate population.",
            "PPL1 is used as an aversive-reinforcement candidate population.",
            "This v0 experiment tests whether the simulator produces experience-dependent MaleCNS state changes; it is not yet Flyppy body learning.",
        ],
        "groups": {
            "reward_cue": reward_cue,
            "aversive_cue": aversive_cue,
            "reward_dan": reward_dan,
            "aversive_dan": aversive_dan,
            "readout": readout,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
