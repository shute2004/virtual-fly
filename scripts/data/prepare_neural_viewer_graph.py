#!/usr/bin/env python3
"""Build a compact static MaleCNS graph for the detached live viewer.

The full released graph has ~25.6M edges. The observer therefore keeps the
reinforcement and wing-motor boundary plus a bounded set of strongly connected
CNS neurons. This graph is viewer-only and is never fed back into learning.

Coordinates are deliberately schematic. They preserve released body IDs,
connections, side labels and broad annotation categories, but they are not
morphology/skeleton coordinates. The layout is shaped as bilateral brain lobes,
optic lobes and a ventral nerve cord so live activity is easier to interpret
than the previous rectangular hash bands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


VIEWER_REQUIRED_GROUPS = ("reward_dan", "aversive_dan")


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
    parser.add_argument("--max-nodes", type=int, default=2500)
    parser.add_argument("--max-edges", type=int, default=6000)
    return parser.parse_args()


def stable_unit(body_id: int, salt: str) -> float:
    digest = hashlib.blake2b(f"{body_id}:{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def classify_region(superclass: str, neuron_type: str, nerve: str) -> str:
    text = f"{superclass} {neuron_type} {nerve}".lower()
    if any(token in text for token in ("optic", "visual", "photoreceptor")):
        return "optic_lobe"
    if any(
        token in text
        for token in (
            "descending",
            "ascending",
            "motor",
            "sensory",
            "nerve",
            "vnc",
            "ventral",
        )
    ):
        return "ventral_nerve_cord"
    return "central_brain"


def ellipsoid_point(
    body_id: int,
    *,
    salt: str,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> list[float]:
    # Deterministic approximately uniform volume sample inside an ellipsoid.
    azimuth = 2.0 * math.pi * stable_unit(body_id, f"{salt}:azimuth")
    cos_polar = 2.0 * stable_unit(body_id, f"{salt}:polar") - 1.0
    sin_polar = math.sqrt(max(0.0, 1.0 - cos_polar * cos_polar))
    radius = stable_unit(body_id, f"{salt}:radius") ** (1.0 / 3.0)
    unit = (
        radius * sin_polar * math.cos(azimuth),
        radius * cos_polar,
        radius * sin_polar * math.sin(azimuth),
    )
    return [
        center[0] + radii[0] * unit[0],
        center[1] + radii[1] * unit[1],
        center[2] + radii[2] * unit[2],
    ]


def schematic_position(
    body_id: int,
    side: str,
    superclass: str,
    neuron_type: str,
    nerve: str,
) -> tuple[list[float], str]:
    region = classify_region(superclass, neuron_type, nerve)
    side_norm = side.strip().upper()
    if side_norm == "L":
        sign = -1.0
    elif side_norm == "R":
        sign = 1.0
    else:
        sign = -1.0 if stable_unit(body_id, "side") < 0.5 else 1.0

    if region == "optic_lobe":
        center = (sign * 4.5, 1.2, 0.0)
        radii = (1.65, 2.45, 2.05)
    elif region == "ventral_nerve_cord":
        center = (sign * 0.75, -5.2, 0.0)
        radii = (1.15, 4.2, 1.35)
    else:
        center = (sign * 1.55, 1.35, 0.0)
        radii = (2.7, 3.0, 2.55)

    return ellipsoid_point(
        body_id,
        salt=region,
        center=center,
        radii=radii,
    ), region


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
    groups = json.loads(args.groups.read_text(encoding="utf-8")).get("groups", {})
    for name in VIEWER_REQUIRED_GROUPS:
        required.update(int(value) for value in groups.get(name, {}).get("body_ids", []))
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

    annotations = pd.read_feather(args.snapshot / manifest["annotations_file"])
    annotations = annotations.drop_duplicates("bodyId").set_index("bodyId")
    index_by_body = {int(body): i for i, body in enumerate(body_ids.tolist())}
    nodes = []
    for body_id in sorted(selected):
        row = annotations.loc[body_id] if body_id in annotations.index else None
        side = "" if row is None else clean_text(row.get("side", ""))
        superclass = "" if row is None else clean_text(row.get("superclass", ""))
        neuron_type = "" if row is None else clean_text(row.get("type", ""))
        nerve = ""
        if row is not None:
            nerve = " ".join(
                clean_text(row.get(name, ""))
                for name in ("nerve", "entryNerve", "exitNerve")
            ).strip()
        position, region = schematic_position(
            body_id,
            side,
            superclass,
            neuron_type,
            nerve,
        )
        dense = index_by_body.get(body_id)
        nodes.append(
            {
                "body_id": body_id,
                "side": side,
                "superclass": superclass,
                "type": neuron_type,
                "region": region,
                "nt": int(nt[dense]) if dense is not None else 0,
                "position": position,
            }
        )

    selected_lookup = set(selected)
    edges = [
        {"pre": pre, "post": post, "synapse_count": count}
        for pre, post, count in selected_edges
        if pre in selected_lookup and post in selected_lookup
    ]
    payload = {
        "schema_version": 3,
        "layout": "schematic-bilateral-brain-optic-lobes-vnc-not-anatomical",
        "source_dataset": manifest.get("dataset"),
        "viewer_only": True,
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
