#!/usr/bin/env python3
"""Prepare the MaleCNS haltere campaniform-afferent boundary.

Selection is annotation-driven and deliberately narrow:

* entryNerve == DMetaN (dorsal metathoracic / haltere nerve)
* class == mechanosensory_proprioceptive
* subclass == campaniform sensilla
* rootSide must resolve to L/R

The resulting artifact contains released MaleCNS body IDs only.  It does not
assign task semantics or downstream steering weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--annotations",
        type=Path,
        default=Path("artifacts/malecns-v1.0/annotations.feather"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns-v1.0/haltere-campaniform-sensory-v1.json"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_feather(args.annotations)
    required = {"bodyId", "entryNerve", "class", "subclass", "rootSide"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"annotations missing columns: {sorted(missing)}")

    selected = df[
        (df["entryNerve"].fillna("") == "DMetaN")
        & (df["class"].fillna("") == "mechanosensory_proprioceptive")
        & (df["subclass"].fillna("") == "campaniform sensilla")
    ].copy()
    if selected.empty:
        raise RuntimeError("selected zero haltere campaniform afferents")
    if not selected["rootSide"].isin(("L", "R")).all():
        bad = selected.loc[~selected["rootSide"].isin(("L", "R")), ["bodyId", "rootSide"]]
        raise RuntimeError(f"unresolved haltere afferent sides:\n{bad}")

    neurons = []
    for row in selected.sort_values(["rootSide", "bodyId"]).to_dict("records"):
        neurons.append(
            {
                "body_id": int(row["bodyId"]),
                "side": str(row["rootSide"]),
                "type": None if pd.isna(row.get("type")) else str(row.get("type")),
                "instance": None if pd.isna(row.get("instance")) else str(row.get("instance")),
                "status_label": None
                if pd.isna(row.get("statusLabel"))
                else str(row.get("statusLabel")),
            }
        )

    by_side = {
        "left": [r["body_id"] for r in neurons if r["side"] == "L"],
        "right": [r["body_id"] for r in neurons if r["side"] == "R"],
    }
    payload = {
        "schema_version": 1,
        "kind": "male-cns-haltere-campaniform-afferents",
        "selection": {
            "entryNerve": "DMetaN",
            "class": "mechanosensory_proprioceptive",
            "subclass": "campaniform sensilla",
            "side_field": "rootSide",
        },
        "count": len(neurons),
        "counts_by_side": {k: len(v) for k, v in by_side.items()},
        "body_ids_by_side": by_side,
        "neurons": neurons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"haltere_sensory_map=PASS count={len(neurons)} "
        f"left={len(by_side['left'])} right={len(by_side['right'])} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
