#!/usr/bin/env python3
"""Build a compact anatomically positioned MaleCNS graph for the live viewer.

The released MaleCNS graph has ~25.6M edges, so the live observer displays a
bounded set of strong connections plus the reinforcement and wing-motor boundary.
Unlike the earlier schematic layout, displayed neurons use released MaleCNS soma
coordinates.  A deterministic sample of all released soma positions is included
as a viewer-only anatomical background point cloud.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--anatomy-annotations",
        type=Path,
        default=None,
        help="official MaleCNS annotations containing somaLocation; defaults to snapshot annotations",
    )
    parser.add_argument("--max-nodes", type=int, default=2500)
    parser.add_argument("--max-edges", type=int, default=6000)
    parser.add_argument("--max-anatomy-points", type=int, default=20000)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
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


def location_vector(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def row_location(row: pd.Series | None) -> np.ndarray | None:
    if row is None:
        return None
    for name in ("somaLocation", "tosomaLocation"):
        if name in row.index:
            vector = location_vector(row.get(name))
            if vector is not None:
                return vector
    return None


def anatomical_transform(
    vector: np.ndarray,
    *,
    center: np.ndarray,
    scale: float,
) -> list[float]:
    """Map MaleCNS EM voxels into viewer axes without changing relative scale.

    MaleCNS x is left/right.  Its long z axis runs brain->VNC, so viewer y uses
    -z to keep the brain above the VNC.  Source y becomes viewer depth.
    """

    delta = (vector - center) * scale
    return [
        round(float(-delta[0]), 5),
        round(float(-delta[2]), 5),
        round(float(delta[1]), 5),
    ]


def main() -> int:
    args = parse_args()
    if args.max_nodes < 256 or args.max_edges < 256:
        raise SystemExit("viewer graph budgets are too small")
    if args.max_anatomy_points < 1000:
        raise SystemExit("max-anatomy-points must be >= 1000")

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

    annotation_path = args.anatomy_annotations or (args.snapshot / manifest["annotations_file"])
    annotations = pd.read_feather(annotation_path)
    if "somaLocation" not in annotations.columns and args.anatomy_annotations is None:
        raw_candidate = (
            args.snapshot.parent
            / "raw"
            / "male-cns-v1.0"
            / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        )
        if raw_candidate.exists():
            annotation_path = raw_candidate
            annotations = pd.read_feather(annotation_path)
    annotations = annotations.drop_duplicates("bodyId").set_index("bodyId")
    if "somaLocation" not in annotations.columns:
        raise RuntimeError(
            "MaleCNS annotations lack somaLocation; provide the official v1.0 annotation feather with --anatomy-annotations"
        )

    aligned = annotations.reindex(body_ids)
    all_locations: list[np.ndarray] = []
    all_location_body_ids: list[int] = []
    for body_id, (_, row) in zip(body_ids.tolist(), aligned.iterrows(), strict=True):
        location = row_location(row)
        if location is not None:
            all_locations.append(location)
            all_location_body_ids.append(int(body_id))
    if not all_locations:
        raise RuntimeError("MaleCNS snapshot contains no usable soma coordinates")

    location_array = np.stack(all_locations)
    minimum = location_array.min(axis=0)
    maximum = location_array.max(axis=0)
    center = (minimum + maximum) / 2.0
    longest_extent = float(np.max(maximum - minimum))
    if longest_extent <= 0.0:
        raise RuntimeError("degenerate MaleCNS soma coordinate extent")
    scale = 14.0 / longest_extent

    location_by_body = {
        body_id: location
        for body_id, location in zip(all_location_body_ids, all_locations, strict=True)
    }
    index_by_body = {int(body): i for i, body in enumerate(body_ids.tolist())}

    nodes = []
    for body_id in sorted(selected):
        location = location_by_body.get(body_id)
        if location is None:
            continue
        row = annotations.loc[body_id] if body_id in annotations.index else None
        side = "" if row is None else clean_text(row.get("somaSide", row.get("side", "")))
        superclass = "" if row is None else clean_text(row.get("superclass", ""))
        neuron_type = "" if row is None else clean_text(row.get("type", ""))
        nerve = ""
        if row is not None:
            nerve = " ".join(
                clean_text(row.get(name, ""))
                for name in ("nerve", "entryNerve", "exitNerve")
            ).strip()
        region = classify_region(superclass, neuron_type, nerve)
        dense = index_by_body.get(body_id)
        nodes.append(
            {
                "body_id": body_id,
                "side": side,
                "superclass": superclass,
                "type": neuron_type,
                "region": region,
                "nt": int(nt[dense]) if dense is not None else 0,
                "position": anatomical_transform(location, center=center, scale=scale),
            }
        )

    displayed = {int(node["body_id"]) for node in nodes}
    missing_required = sorted(required - displayed)
    if missing_required:
        raise RuntimeError(
            "required viewer neurons lack released soma/tosoma coordinates: "
            + ",".join(str(value) for value in missing_required[:16])
        )

    edges = [
        {"pre": pre, "post": post, "synapse_count": count}
        for pre, post, count in selected_edges
        if pre in displayed and post in displayed
    ]

    sample_count = min(args.max_anatomy_points, len(all_locations))
    rng = np.random.default_rng(0)
    sample_indices = np.sort(
        rng.choice(len(all_locations), size=sample_count, replace=False)
        if sample_count < len(all_locations)
        else np.arange(len(all_locations), dtype=np.int64)
    )
    anatomy_points = [
        anatomical_transform(location_array[int(index)], center=center, scale=scale)
        for index in sample_indices.tolist()
    ]

    payload = {
        "schema_version": 4,
        "layout": "male-cns-v1.0-released-soma-coordinates",
        "source_dataset": manifest.get("dataset"),
        "coordinate_source": "official somaLocation/tosomaLocation; MaleCNS EM 8 nm voxel space",
        "viewer_axes": "x=-source_x, y=-source_z, z=source_y; one isotropic display scale",
        "viewer_only": True,
        "anatomy_points": anatomy_points,
        "nodes": nodes,
        "edges": edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(f"anatomy_points={len(anatomy_points)}")
    print(f"viewer_nodes={len(nodes)} viewer_edges={len(edges)}")
    print(f"viewer_graph={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
