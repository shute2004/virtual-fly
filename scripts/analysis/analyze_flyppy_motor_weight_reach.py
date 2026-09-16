#!/usr/bin/env python3
"""Analyze where learned Flyppy weight changes sit relative to actuated motor output.

This is an offline checkpoint analysis.  It does not advance MaleCNS state or
modify the checkpoint.  Distances are measured on released directed MaleCNS
connectivity by traversing incoming edges backwards from the motor neurons:

    distance 0: actuated motor neuron itself
    distance 1: neurons with an edge directly into an actuated motor neuron
    distance 2: neurons that can reach motor in two released edges
    ...

A changed edge is assigned to the distance of its postsynaptic neuron.  Thus a
changed edge whose post is distance 0 directly changes input to an actuated MN;
a changed edge whose post is distance 1 can affect motor through one additional
released edge.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_CHECKPOINT = Path("artifacts/experiments/flyppy-v3/checkpoint")
DEFAULT_WING_MAP = Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json")
DEFAULT_BODY_MAP = Path("artifacts/malecns-v1.0/body-motor-neurons-v0.json")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
DEFAULT_PLASTIC_GRAPH = Path("artifacts/malecns-v1.0/plastic-fast-graph-v1.json")
DEFAULT_JSON = Path("reports/flyppy/diagnostics/motor_weight_reach.json")
DEFAULT_REPORT = Path("reports/flyppy/diagnostics/motor_weight_reach.md")
WING_STATUSES = frozenset({"identified", "identified_group"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wing-map", type=Path, default=DEFAULT_WING_MAP)
    parser.add_argument("--body-map", type=Path, default=DEFAULT_BODY_MAP)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--plastic-graph", type=Path, default=DEFAULT_PLASTIC_GRAPH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--weight-epsilon", type=float, default=1e-7)
    parser.add_argument("--chunk-edges", type=int, default=1_000_000)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def selected_motor_ids(wing_map: Path, body_map: Path) -> tuple[set[int], set[int]]:
    wing_payload = load_json(wing_map)
    body_payload = load_json(body_map)
    wing = {
        int(row["body_id"])
        for row in wing_payload.get("neurons", [])
        if str(row.get("mapping_status", "")) in WING_STATUSES
    }
    somatic = {
        int(row["body_id"])
        for row in body_payload.get("neurons", [])
        if str(row.get("mechanical_status", "")) == "grounded"
    }
    if not wing or not somatic:
        raise RuntimeError(
            f"actuated motor selection is empty: wing={len(wing)} somatic={len(somatic)}"
        )
    return wing, somatic


def resolve_indices(body_ids: np.ndarray, selected: Iterable[int]) -> np.ndarray:
    wanted = {int(value) for value in selected}
    mapping = {
        int(body_id): index
        for index, body_id in enumerate(body_ids)
        if int(body_id) in wanted
    }
    missing = sorted(wanted - mapping.keys())
    if missing:
        preview = ",".join(str(value) for value in missing[:8])
        raise RuntimeError(f"{len(missing)} selected motor body IDs absent from snapshot: {preview}")
    return np.asarray(sorted(mapping.values()), dtype=np.int64)


def upstream_distances(
    row_offsets: np.ndarray,
    pre_indices: np.ndarray,
    motor_indices: np.ndarray,
    max_hops: int,
) -> tuple[np.ndarray, list[int]]:
    n = len(row_offsets) - 1
    distance = np.full(n, -1, dtype=np.int16)
    distance[motor_indices] = 0
    frontier = np.asarray(motor_indices, dtype=np.int64)
    layer_sizes = [int(len(frontier))]

    for hop in range(1, max_hops + 1):
        candidate = np.zeros(n, dtype=np.bool_)
        for post in frontier:
            start = int(row_offsets[int(post)])
            end = int(row_offsets[int(post) + 1])
            if end > start:
                candidate[np.asarray(pre_indices[start:end], dtype=np.int64)] = True
        candidate[distance >= 0] = False
        frontier = np.flatnonzero(candidate).astype(np.int64, copy=False)
        distance[frontier] = hop
        layer_sizes.append(int(len(frontier)))
        if len(frontier) == 0:
            layer_sizes.extend([0] * (max_hops - hop))
            break
    return distance, layer_sizes


def empty_edge_stats() -> dict[str, float | int]:
    return {
        "edges": 0,
        "changed_edges": 0,
        "positive_edges": 0,
        "negative_edges": 0,
        "sum_abs_delta": 0.0,
        "sum_signed_delta": 0.0,
        "max_abs_delta": 0.0,
    }


def update_stats(
    stats: dict[str, float | int],
    weights: np.ndarray,
    synapse_counts: np.ndarray,
    *,
    synapse_scale: float,
    epsilon: float,
) -> None:
    if len(weights) == 0:
        return
    initial = np.asarray(synapse_counts, dtype=np.float32) * np.float32(synapse_scale)
    delta = np.asarray(weights, dtype=np.float32) - initial
    abs_delta = np.abs(delta)
    changed = abs_delta > epsilon
    stats["edges"] = int(stats["edges"]) + len(delta)
    if not np.any(changed):
        return
    selected = delta[changed]
    selected_abs = abs_delta[changed]
    stats["changed_edges"] = int(stats["changed_edges"]) + int(np.count_nonzero(changed))
    stats["positive_edges"] = int(stats["positive_edges"]) + int(np.count_nonzero(selected > 0.0))
    stats["negative_edges"] = int(stats["negative_edges"]) + int(np.count_nonzero(selected < 0.0))
    stats["sum_abs_delta"] = float(stats["sum_abs_delta"]) + float(
        np.sum(selected_abs, dtype=np.float64)
    )
    stats["sum_signed_delta"] = float(stats["sum_signed_delta"]) + float(
        np.sum(selected, dtype=np.float64)
    )
    stats["max_abs_delta"] = max(float(stats["max_abs_delta"]), float(np.max(selected_abs)))


def finalize_stats(stats: dict[str, float | int]) -> dict[str, float | int]:
    edges = int(stats["edges"])
    changed = int(stats["changed_edges"])
    result = dict(stats)
    result["changed_fraction"] = changed / edges if edges else 0.0
    result["mean_abs_delta_changed"] = (
        float(stats["sum_abs_delta"]) / changed if changed else 0.0
    )
    result["mean_signed_delta_changed"] = (
        float(stats["sum_signed_delta"]) / changed if changed else 0.0
    )
    return result


def stats_for_posts(
    posts: np.ndarray,
    row_offsets: np.ndarray,
    weights: np.ndarray,
    synapse_counts: np.ndarray,
    *,
    synapse_scale: float,
    epsilon: float,
) -> dict[str, float | int]:
    stats = empty_edge_stats()
    for post in posts:
        start = int(row_offsets[int(post)])
        end = int(row_offsets[int(post) + 1])
        update_stats(
            stats,
            weights[start:end],
            synapse_counts[start:end],
            synapse_scale=synapse_scale,
            epsilon=epsilon,
        )
    return finalize_stats(stats)


def global_stats(
    weights: np.ndarray,
    synapse_counts: np.ndarray,
    *,
    synapse_scale: float,
    epsilon: float,
    chunk_edges: int,
) -> dict[str, float | int]:
    stats = empty_edge_stats()
    for start in range(0, len(weights), chunk_edges):
        end = min(len(weights), start + chunk_edges)
        update_stats(
            stats,
            weights[start:end],
            synapse_counts[start:end],
            synapse_scale=synapse_scale,
            epsilon=epsilon,
        )
    return finalize_stats(stats)


def direct_motor_summary(
    name: str,
    indices: np.ndarray,
    row_offsets: np.ndarray,
    plastic_row_offsets: np.ndarray,
    weights: np.ndarray,
    synapse_counts: np.ndarray,
    *,
    synapse_scale: float,
    epsilon: float,
) -> dict[str, Any]:
    plastic_incoming = np.asarray(
        plastic_row_offsets[indices + 1] - plastic_row_offsets[indices], dtype=np.uint64
    )
    return {
        "name": name,
        "motor_neurons": int(len(indices)),
        "motor_neurons_with_plastic_incoming": int(np.count_nonzero(plastic_incoming)),
        "plastic_incoming_edges": int(np.sum(plastic_incoming, dtype=np.uint64)),
        "incoming_weight_change": stats_for_posts(
            indices,
            row_offsets,
            weights,
            synapse_counts,
            synapse_scale=synapse_scale,
            epsilon=epsilon,
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Flyppy motor-path weight reach",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- checkpoint neural step: {payload['checkpoint_neural_step']}",
        f"- weight epsilon: {payload['weight_epsilon']:.1e}",
        f"- synapse scale: {payload['synapse_scale']}",
        f"- max upstream hops: {payload['max_hops']}",
        "",
        "## Direct actuated motor input",
        "",
        "| target | neurons | neurons with plastic incoming | plastic incoming edges | incoming edges | changed | changed % | mean | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["direct_motor"]:
        stats = row["incoming_weight_change"]
        lines.append(
            "| {name} | {motor_neurons} | {motor_neurons_with_plastic_incoming} | "
            "{plastic_incoming_edges} | {edges} | {changed_edges} | {fraction:.3%} | "
            "{mean:.6g} | {max_delta:.6g} |".format(
                **row,
                edges=int(stats["edges"]),
                changed_edges=int(stats["changed_edges"]),
                fraction=float(stats["changed_fraction"]),
                mean=float(stats["mean_abs_delta_changed"]),
                max_delta=float(stats["max_abs_delta"]),
            )
        )

    lines.extend(
        [
            "",
            "## Released-connectome upstream distance",
            "",
            "距離0はactuated motor neuron自身。各行のweight統計は、その距離にあるpostsynaptic neuronへ入るedgeを集計する。",
            "",
            "| post distance to motor | neurons | incoming edges | changed | changed % | positive | negative | mean abs delta | max abs delta |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["distance_layers"]:
        stats = row["incoming_weight_change"]
        lines.append(
            "| {distance} | {neurons} | {edges} | {changed} | {fraction:.3%} | "
            "{positive} | {negative} | {mean:.6g} | {max_delta:.6g} |".format(
                distance=row["distance"],
                neurons=row["neurons"],
                edges=int(stats["edges"]),
                changed=int(stats["changed_edges"]),
                fraction=float(stats["changed_fraction"]),
                positive=int(stats["positive_edges"]),
                negative=int(stats["negative_edges"]),
                mean=float(stats["mean_abs_delta_changed"]),
                max_delta=float(stats["max_abs_delta"]),
            )
        )

    global_stats_payload = payload["global_weight_change"]
    lines.extend(
        [
            "",
            "## Global reference",
            "",
            f"- all edges: {global_stats_payload['edges']}",
            f"- changed edges: {global_stats_payload['changed_edges']} ({global_stats_payload['changed_fraction']:.3%})",
            f"- positive / negative changed edges: {global_stats_payload['positive_edges']} / {global_stats_payload['negative_edges']}",
            f"- mean abs delta among changed: {global_stats_payload['mean_abs_delta_changed']:.6g}",
            f"- max abs delta: {global_stats_payload['max_abs_delta']:.6g}",
            "",
            "## Interpretation boundary",
            "",
            "この解析はreleased MaleCNS connectivity上の経路距離とcheckpoint weight差だけを測る。",
            "距離が近いことは、そのedgeが実際のepisodeでmotor activityを因果的に変えたことを意味しない。",
            "逆にdirect motor incomingが変化していなくても、上流回路の可塑性がmotor出力を変えることは可能である。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.max_hops < 0 or args.max_hops > 12:
        raise SystemExit("--max-hops must be in 0..12")
    if not math.isfinite(args.weight_epsilon) or args.weight_epsilon < 0.0:
        raise SystemExit("--weight-epsilon must be finite and >= 0")
    if args.chunk_edges < 1:
        raise SystemExit("--chunk-edges must be positive")

    snapshot = absolute(args.snapshot)
    checkpoint = absolute(args.checkpoint)
    wing_map = absolute(args.wing_map)
    body_map = absolute(args.body_map)
    calibration_path = absolute(args.calibration)
    plastic_graph_path = absolute(args.plastic_graph)
    output_json = absolute(args.output_json)
    report = absolute(args.report)

    required = [
        snapshot / "manifest.json",
        checkpoint / "manifest.json",
        wing_map,
        body_map,
        calibration_path,
        plastic_graph_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing motor reach inputs:\n  " + "\n  ".join(missing))

    manifest = load_json(snapshot / "manifest.json")
    checkpoint_manifest = load_json(checkpoint / "manifest.json")
    calibration = load_json(calibration_path)
    plastic_manifest = load_json(plastic_graph_path)
    n = int(manifest["neuron_count"])
    e = int(manifest["edge_count"])
    if int(checkpoint_manifest["neuron_count"]) != n or int(checkpoint_manifest["edge_count"]) != e:
        raise RuntimeError("checkpoint does not match MaleCNS snapshot dimensions")
    synapse_scale = float(calibration["synapse_scale"])
    if not math.isfinite(synapse_scale) or synapse_scale <= 0.0:
        raise RuntimeError("invalid calibrated synapse scale")

    body_ids = np.memmap(snapshot / str(manifest["body_ids_file"]), dtype="<u8", mode="r")
    row_offsets = np.memmap(snapshot / str(manifest["row_offsets_file"]), dtype="<u4", mode="r")
    pre_indices = np.memmap(snapshot / str(manifest["pre_indices_file"]), dtype="<u4", mode="r")
    synapse_counts = np.memmap(
        snapshot / str(manifest["synapse_counts_file"]), dtype="<u4", mode="r"
    )
    weights = np.memmap(
        checkpoint / str(checkpoint_manifest["weights_file"]), dtype="<f4", mode="r"
    )
    plastic_file = snapshot / str(plastic_manifest["files"]["plastic_row_offsets_u32le"])
    plastic_row_offsets = np.memmap(plastic_file, dtype="<u4", mode="r")

    if len(body_ids) != n or len(row_offsets) != n + 1 or len(pre_indices) != e:
        raise RuntimeError("snapshot array dimensions are inconsistent")
    if len(synapse_counts) != e or len(weights) != e or len(plastic_row_offsets) != n + 1:
        raise RuntimeError("checkpoint/plastic graph array dimensions are inconsistent")

    wing_ids, somatic_ids = selected_motor_ids(wing_map, body_map)
    wing_indices = resolve_indices(body_ids, wing_ids)
    somatic_indices = resolve_indices(body_ids, somatic_ids)
    all_indices = np.asarray(sorted(set(wing_indices.tolist()) | set(somatic_indices.tolist())), dtype=np.int64)

    distance, layer_sizes = upstream_distances(
        row_offsets,
        pre_indices,
        all_indices,
        args.max_hops,
    )
    distance_layers = []
    for hop in range(args.max_hops + 1):
        posts = np.flatnonzero(distance == hop).astype(np.int64, copy=False)
        distance_layers.append(
            {
                "distance": hop,
                "neurons": int(len(posts)),
                "incoming_weight_change": stats_for_posts(
                    posts,
                    row_offsets,
                    weights,
                    synapse_counts,
                    synapse_scale=synapse_scale,
                    epsilon=args.weight_epsilon,
                ),
            }
        )

    direct_motor = [
        direct_motor_summary(
            "wing",
            wing_indices,
            row_offsets,
            plastic_row_offsets,
            weights,
            synapse_counts,
            synapse_scale=synapse_scale,
            epsilon=args.weight_epsilon,
        ),
        direct_motor_summary(
            "somatic",
            somatic_indices,
            row_offsets,
            plastic_row_offsets,
            weights,
            synapse_counts,
            synapse_scale=synapse_scale,
            epsilon=args.weight_epsilon,
        ),
        direct_motor_summary(
            "all_actuated",
            all_indices,
            row_offsets,
            plastic_row_offsets,
            weights,
            synapse_counts,
            synapse_scale=synapse_scale,
            epsilon=args.weight_epsilon,
        ),
    ]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "snapshot": portable(snapshot),
        "checkpoint": portable(checkpoint),
        "checkpoint_neural_step": int(checkpoint_manifest.get("step", -1)),
        "synapse_scale": synapse_scale,
        "weight_epsilon": float(args.weight_epsilon),
        "max_hops": int(args.max_hops),
        "actuated_motor_counts": {
            "wing": len(wing_ids),
            "somatic": len(somatic_ids),
            "all": len(all_indices),
        },
        "upstream_layer_sizes": layer_sizes,
        "direct_motor": direct_motor,
        "distance_layers": distance_layers,
        "global_weight_change": global_stats(
            weights,
            synapse_counts,
            synapse_scale=synapse_scale,
            epsilon=args.weight_epsilon,
            chunk_edges=args.chunk_edges,
        ),
    }
    save_json(output_json, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(payload), encoding="utf-8")

    print(f"checkpoint_neural_step={payload['checkpoint_neural_step']}")
    for row in direct_motor:
        stats = row["incoming_weight_change"]
        print(
            "direct_motor={} neurons={} plastic_posts={} plastic_edges={} changed={}/{} ({:.3%})".format(
                row["name"],
                row["motor_neurons"],
                row["motor_neurons_with_plastic_incoming"],
                row["plastic_incoming_edges"],
                stats["changed_edges"],
                stats["edges"],
                stats["changed_fraction"],
            )
        )
    for row in distance_layers:
        stats = row["incoming_weight_change"]
        print(
            "distance={} neurons={} changed={}/{} ({:.3%})".format(
                row["distance"],
                row["neurons"],
                stats["changed_edges"],
                stats["edges"],
                stats["changed_fraction"],
            )
        )
    print(f"json={output_json}")
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
