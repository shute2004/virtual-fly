#!/usr/bin/env python3
"""Build a compact static MaleCNS graph for the external live viewer.

The full released graph has ~25.6M edges and is unnecessarily large for an
interactive observer. This file keeps all embodiment-boundary neurons plus the
strongest released connections until bounded node/edge budgets are reached.
Positions are deterministic schematic 3D coordinates; they are not anatomical
MaleCNS morphology coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"),
    )
    parser.add_argument(
        "--motor-map",
        type=Path,
        default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/neural-viewer-graph-v1.json"),
    )
    parser.add_argument("--max-nodes", type=int, default=6000)
    parser.add_argument("--max-edges", type=int, default=12000)
    return parser.parse_args()


def stable_unit(body_id: int, salt: str) -> float:
    digest = hashlib.blake2b(f"{body_id}:{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def schematic_position(body_id: int, side: str, superclass: str) -> list[float]:
    side_norm = side.strip().upper()
    if side_norm == "L":
        x_base = -4.0
    elif side_norm == "R":
        x_base = 4.0
    else:
        x_base = 0.0
    band = (
        int.from_bytes(hashlib.blake2b(superclass.encode(), digest_size=2).digest(), "little")
        % 9
    ) - 4
    return [
        x_base + (stable_unit(body_id, "x") - 0.5) * 4.0,
        band * 1.25 + (stable_unit(body_id, "y") - 0.5) * 1.1,
        (stable_unit(body_id, "z") - 0.5) * 9.0,
    ]


def main() -> int:
    args = parse_args()
    if args.max_nodes < 256 or args.max_edges < 256:
        raise SystemExit("viewer graph budgets are too small")

    manifest = json.loads((args.snapshot / "manifest.json").read_text(encoding="utf-8"))
    body_ids = np.fromfile(args.snapshot / manifest["body_ids_file"], dtype="<u8")
    row_offsets = np.fromfile(args.snapshot / manifest["row_offsets_file"], dtype="<u4")
    pre_indices = np.fromfile(args.snapshot / manifest["pre_indices_file"], dtype="<u4")
    counts = np.fromfile(args.snapshot / manifest["synapse_counts_file"], dtype="<u4")
    nt = np.fromfile(args.snapshot / manifest["neurotransmitters_file"], dtype=np.uint8)

    required: set[int] = set()
    groups = json.loads(args.groups.read_text(encoding="utf-8"))
    for group in groups.get("groups", {}).values():
        required.update(int(value) for value in group.get("body_ids", []))
    motor = json.loads(args.motor_map.read_text(encoding="utf-8"))
    required.update(int(row["body_id"]) for row in motor.get("neurons", []))

    candidate_count = min(len(counts), max(args.max_edges * 8, 50_000))
    if candidate_count == len(counts):
        candidates = np.arange(len(counts), dtype=np.int64)
    else:
        candidates = np.argpartition(counts, len(counts) - candidate_count)[-candidate_count:]
    candidates = candidates[np.argsort(counts[candidates], kind="stable")[::-1]]
    posts = np.searchsorted(row_offsets, candidates, side="right") - 1

    available = {int(value) for value in body_ids.tolist()}
    selected: set[int] = {value for value in required if value in available}
    selected_edges: list[tuple[int, int, int]] = []
    for edge_index, post_index in zip(candidates.tolist(), posts.tolist(), strict=True):
        pre_index = int(pre_indices[edge_index])
        pre = int(body_ids[pre_index])
        post = int(body_ids[int(post_index)])
        new_nodes = int(pre not in selected) + int(post not in selected)
        if len(selected) + new_nodes > args.max_nodes:
            continue
        selected.add(pre)
        selected.add(post)
        selected_edges.append((pre, post, int(counts[edge_index])))
        if len(selected_edges) >= args.max_edges:
            break

    selected.update(value for value in required if value in available)

    annotations = pd.read_feather(args.snapshot / manifest["annotations_file"])
    annotations = annotations.drop_duplicates("bodyId").set_index("bodyId")
    index_by_body = {int(body): i for i, body in enumerate(body_ids.tolist())}
    nodes = []
    for body_id in sorted(selected):
        row = annotations.loc[body_id] if body_id in annotations.index else None
        side = "" if row is None else str(row.get("side", "") or "")
        superclass = "" if row is None else str(row.get("superclass", "") or "")
        neuron_type = "" if row is None else str(row.get("type", "") or "")
        dense = index_by_body.get(body_id)
        nodes.append(
            {
                "body_id": body_id,
                "side": side,
                "superclass": superclass,
                "type": neuron_type,
                "nt": int(nt[dense]) if dense is not None else 0,
                "position": schematic_position(body_id, side, superclass),
            }
        )

    selected_lookup = set(selected)
    edges = [
        {"pre": pre, "post": post, "synapse_count": count}
        for pre, post, count in selected_edges
        if pre in selected_lookup and post in selected_lookup
    ]
    payload = {
        "schema_version": 1,
        "layout": "deterministic-schematic-not-anatomical",
        "source_dataset": manifest.get("dataset"),
        "nodes": nodes,
        "edges": edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(f"viewer_nodes={len(nodes)} viewer_edges={len(edges)}")
    print(f"viewer_graph={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
