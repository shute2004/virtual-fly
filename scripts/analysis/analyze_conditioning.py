#!/usr/bin/env python3
"""Analyze where the first MaleCNS conditioning run changed synapses.

This is deliberately separate from the trainer: the trainer only evolves neural
state; analysis inspects the resulting state afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Must match NeuralParams::default().synapse_scale for conditioning-v0.
INITIAL_SYNAPSE_SCALE = 0.02
CHANGE_EPSILON = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    return parser.parse_args()


def indices(config: dict, name: str) -> np.ndarray:
    rows = config["groups"][name]
    values = np.asarray([int(row["index"]) for row in rows], dtype=np.uint32)
    if values.size == 0:
        raise RuntimeError(f"empty group: {name}")
    return np.unique(values)


def summarize(deltas: np.ndarray) -> dict[str, float | int]:
    if deltas.size == 0:
        return {
            "edge_count": 0,
            "changed_edges": 0,
            "mean_delta": 0.0,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }
    abs_delta = np.abs(deltas)
    return {
        "edge_count": int(deltas.size),
        "changed_edges": int(np.count_nonzero(abs_delta > CHANGE_EPSILON)),
        "mean_delta": float(np.mean(deltas, dtype=np.float64)),
        "mean_abs_delta": float(np.mean(abs_delta, dtype=np.float64)),
        "max_abs_delta": float(np.max(abs_delta)),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    body_ids = np.fromfile(args.snapshot / "body_ids.u64le", dtype="<u8")
    row_offsets = np.fromfile(args.snapshot / "row_offsets.u32le", dtype="<u4")
    pre_indices = np.fromfile(args.snapshot / "pre_indices.u32le", dtype="<u4")
    counts = np.fromfile(args.snapshot / "synapse_counts.u32le", dtype="<u4")
    learned = np.fromfile(args.experiment / "learned_weights.f32le", dtype="<f4")

    if len(pre_indices) != len(counts) or len(counts) != len(learned):
        raise RuntimeError("edge arrays and learned weights are not aligned")
    if len(row_offsets) != len(body_ids) + 1:
        raise RuntimeError("CSR row offsets do not match body IDs")

    initial = counts.astype(np.float32) * np.float32(INITIAL_SYNAPSE_SCALE)
    delta = learned - initial

    reward_cue = indices(config, "reward_cue")
    aversive_cue = indices(config, "aversive_cue")
    readout = indices(config, "readout")

    readout_edge_chunks: list[np.ndarray] = []
    for post in readout:
        start = int(row_offsets[int(post)])
        end = int(row_offsets[int(post) + 1])
        if end > start:
            readout_edge_chunks.append(np.arange(start, end, dtype=np.uint32))
    readout_edges = (
        np.concatenate(readout_edge_chunks)
        if readout_edge_chunks
        else np.empty(0, dtype=np.uint32)
    )

    reward_mask = np.isin(pre_indices[readout_edges], reward_cue, assume_unique=False)
    aversive_mask = np.isin(pre_indices[readout_edges], aversive_cue, assume_unique=False)
    reward_to_readout_edges = readout_edges[reward_mask]
    aversive_to_readout_edges = readout_edges[aversive_mask]

    abs_delta = np.abs(delta)
    changed = np.flatnonzero(abs_delta > CHANGE_EPSILON)
    top_count = min(20, changed.size)
    if top_count:
        candidate_abs = abs_delta[changed]
        positions = np.argpartition(candidate_abs, -top_count)[-top_count:]
        top_edges = changed[positions]
        top_edges = top_edges[np.argsort(abs_delta[top_edges])[::-1]]
    else:
        top_edges = np.empty(0, dtype=np.int64)

    edge_posts = np.empty(len(pre_indices), dtype=np.uint32)
    for post in range(len(body_ids)):
        start = int(row_offsets[post])
        end = int(row_offsets[post + 1])
        edge_posts[start:end] = post

    top_changes = []
    for edge in top_edges:
        pre = int(pre_indices[edge])
        post = int(edge_posts[edge])
        top_changes.append(
            {
                "edge": int(edge),
                "pre_index": pre,
                "post_index": post,
                "pre_body_id": int(body_ids[pre]),
                "post_body_id": int(body_ids[post]),
                "initial_weight": float(initial[edge]),
                "learned_weight": float(learned[edge]),
                "delta": float(delta[edge]),
            }
        )

    analysis = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "experiment": config["experiment"],
        "initial_synapse_scale": INITIAL_SYNAPSE_SCALE,
        "all_synapses": summarize(delta),
        "mbon_incoming": summarize(delta[readout_edges]),
        "reward_cue_to_mbon": summarize(delta[reward_to_readout_edges]),
        "aversive_cue_to_mbon": summarize(delta[aversive_to_readout_edges]),
        "top_absolute_changes": top_changes,
    }

    output = args.experiment / "analysis.json"
    output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        "all_changed={changed} / {total}".format(
            changed=analysis["all_synapses"]["changed_edges"],
            total=analysis["all_synapses"]["edge_count"],
        )
    )
    print(
        "mbon_incoming_changed={changed} / {total} mean_abs_delta={mean:.9f}".format(
            changed=analysis["mbon_incoming"]["changed_edges"],
            total=analysis["mbon_incoming"]["edge_count"],
            mean=analysis["mbon_incoming"]["mean_abs_delta"],
        )
    )
    print(
        "reward_cue_to_mbon_changed={changed} / {total}".format(
            changed=analysis["reward_cue_to_mbon"]["changed_edges"],
            total=analysis["reward_cue_to_mbon"]["edge_count"],
        )
    )
    print(
        "aversive_cue_to_mbon_changed={changed} / {total}".format(
            changed=analysis["aversive_cue_to_mbon"]["changed_edges"],
            total=analysis["aversive_cue_to_mbon"]["edge_count"],
        )
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
