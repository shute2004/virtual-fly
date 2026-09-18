#!/usr/bin/env python3
"""Prepare an inferred MaleCNS haltere timing-afferent boundary.

This boundary is derived only from released MaleCNS anatomy/connectivity.  It is
not a Flyppy/task-specific feature selector.

Start from the released DMetaN campaniform-afferent boundary, then retain each
afferent that makes at least one released chemical synapse directly onto either
of these established wingbeat-timing targets:

* b1 motor neurons (left/right)
* w-cHIN interneurons (all released MaleCNS instances)

The resulting group is labelled ``inferred`` because MaleCNS annotations do not
currently identify the peripheral dF1/dF2/etc. campaniform field for each axon,
and released chemical connectivity does not include the known electrical part
of the haltere -> b1 pathway.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / "artifacts/malecns-v1.0"
DEFAULT_BASE_MAP = DEFAULT_SNAPSHOT / "haltere-campaniform-sensory-v1.json"
DEFAULT_OUTPUT = DEFAULT_SNAPSHOT / "haltere-timing-afferents-v1.json"
B1_BODY_IDS = (801310, 804301)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument("--base-map", type=Path, default=DEFAULT_BASE_MAP)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot.resolve()
    base = json.loads(args.base_map.resolve().read_text(encoding="utf-8"))

    bodies = np.fromfile(snapshot / "body_ids.u64le", dtype="<u8")
    row_offsets = np.fromfile(snapshot / "row_offsets.u32le", dtype="<u4")
    pre_indices = np.fromfile(snapshot / "pre_indices.u32le", dtype="<u4")
    counts = np.fromfile(snapshot / "synapse_counts.u32le", dtype="<u4")
    annotations = pd.read_feather(snapshot / "annotations.feather")
    by_body = annotations.drop_duplicates("bodyId").set_index("bodyId")
    index = {int(body_id): i for i, body_id in enumerate(bodies)}

    wchin = annotations[
        annotations["type"].fillna("").astype(str).str.strip().eq("w-cHIN")
    ]
    wchin_ids = tuple(sorted(int(v) for v in wchin["bodyId"].tolist()))
    if not wchin_ids:
        raise RuntimeError("released MaleCNS annotations contain no w-cHIN")

    targets = tuple(B1_BODY_IDS) + wchin_ids
    score: dict[int, int] = {}
    target_hits: dict[int, dict[int, int]] = {}
    for target in targets:
        try:
            j = index[target]
        except KeyError as exc:
            raise RuntimeError(f"timing target {target} is absent from snapshot") from exc
        start, end = int(row_offsets[j]), int(row_offsets[j + 1])
        for pre_index, weight in zip(pre_indices[start:end], counts[start:end]):
            body_id = int(bodies[int(pre_index)])
            weight_i = int(weight)
            score[body_id] = score.get(body_id, 0) + weight_i
            target_hits.setdefault(body_id, {})[target] = weight_i

    base_by_side = {
        "left": tuple(int(v) for v in base["body_ids_by_side"]["left"]),
        "right": tuple(int(v) for v in base["body_ids_by_side"]["right"]),
    }
    selected_by_side: dict[str, list[int]] = {}
    neurons: list[dict[str, object]] = []
    for side, body_ids in base_by_side.items():
        selected = [body_id for body_id in body_ids if score.get(body_id, 0) > 0]
        selected_by_side[side] = selected
        for body_id in selected:
            row = by_body.loc[body_id]
            neurons.append(
                {
                    "body_id": body_id,
                    "side": "L" if side == "left" else "R",
                    "type": None if pd.isna(row.get("type")) else str(row.get("type")),
                    "instance": None if pd.isna(row.get("instance")) else str(row.get("instance")),
                    "status_label": None
                    if pd.isna(row.get("statusLabel"))
                    else str(row.get("statusLabel")),
                    "released_timing_target_synapses": int(score[body_id]),
                    "target_synapses": {
                        str(target): int(weight)
                        for target, weight in sorted(target_hits.get(body_id, {}).items())
                    },
                }
            )

    payload = {
        "schema_version": 1,
        "kind": "male-cns-haltere-timing-afferents-inferred-v1",
        "evidence_status": "inferred_from_released_connectivity",
        "base_boundary": str(args.base_map),
        "selection": {
            "source": "released MaleCNS chemical connectivity",
            "rule": "DMetaN campaniform afferent with >=1 released synapse onto b1 MN or w-cHIN",
            "b1_body_ids": list(B1_BODY_IDS),
            "w_chin_body_ids": list(wchin_ids),
            "peripheral_field_identity": "unknown; dF1/dF2/etc. not assigned by this artifact",
            "electrical_synapses": "not represented in released weight table",
        },
        "count": sum(len(v) for v in selected_by_side.values()),
        "counts_by_side": {side: len(v) for side, v in selected_by_side.items()},
        "body_ids_by_side": selected_by_side,
        "neurons": sorted(neurons, key=lambda row: (str(row["side"]), int(row["body_id"]))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "haltere_timing_map=PASS "
        f"count={payload['count']} left={len(selected_by_side['left'])} "
        f"right={len(selected_by_side['right'])} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
