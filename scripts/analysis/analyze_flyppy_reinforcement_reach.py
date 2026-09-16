#!/usr/bin/env python3
"""Map Flyppy reinforcement DAN reach onto learned weights and motor-upstream layers.

The current bootstrap plasticity rule makes fast incoming edges mutable when the
postsynaptic neuron can receive released dopamine.  Flyppy does not stimulate all
DANs: gate passage injects current into released PAM01 neurons and collision into
released PPL101 neurons.  This analysis measures the posts reached directly by
those task DAN groups and where those posts sit relative to actuated motor output.

Final checkpoint weight deltas are observational: a changed edge in a PAM01- or
PPL101-reached post is not attributed causally to a particular reinforcement
event by this offline analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_flyppy_motor_weight_reach import (
    resolve_indices,
    selected_motor_ids,
    stats_for_posts,
    upstream_distances,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_CHECKPOINT = Path("artifacts/experiments/flyppy-v3/checkpoint")
DEFAULT_GROUPS = Path("artifacts/malecns-v1.0/embodiment-groups-v0.json")
DEFAULT_WING_MAP = Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json")
DEFAULT_BODY_MAP = Path("artifacts/malecns-v1.0/body-motor-neurons-v0.json")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
DEFAULT_PLASTIC_GRAPH = Path("artifacts/malecns-v1.0/plastic-fast-graph-v1.json")
DEFAULT_COMMIT_LOG = Path("artifacts/experiments/flyppy-v3/commit-log.jsonl")
DEFAULT_JSON = Path("reports/flyppy/diagnostics/reinforcement_reach.json")
DEFAULT_REPORT = Path("reports/flyppy/diagnostics/reinforcement_reach.md")
DOPAMINE_CODE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--wing-map", type=Path, default=DEFAULT_WING_MAP)
    parser.add_argument("--body-map", type=Path, default=DEFAULT_BODY_MAP)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--plastic-graph", type=Path, default=DEFAULT_PLASTIC_GRAPH)
    parser.add_argument("--commit-log", type=Path, default=DEFAULT_COMMIT_LOG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-motor-hops", type=int, default=4)
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def direct_posts_from_sources(
    pre_indices: np.ndarray,
    row_offsets: np.ndarray,
    synapse_counts: np.ndarray,
    source_indices: np.ndarray,
    *,
    neuron_count: int,
    chunk_edges: int,
) -> tuple[np.ndarray, int, int]:
    post_mask = np.zeros(neuron_count, dtype=np.bool_)
    edge_count = 0
    contact_count = 0
    sources = np.asarray(source_indices, dtype=np.uint32)
    for start in range(0, len(pre_indices), chunk_edges):
        end = min(len(pre_indices), start + chunk_edges)
        local = np.asarray(pre_indices[start:end], dtype=np.uint32)
        matched = np.isin(local, sources, assume_unique=False)
        count = int(np.count_nonzero(matched))
        if count == 0:
            continue
        edge_count += count
        contact_count += int(
            np.sum(
                np.asarray(synapse_counts[start:end], dtype=np.uint64)[matched],
                dtype=np.uint64,
            )
        )
        source_edges = np.flatnonzero(matched).astype(np.int64, copy=False) + start
        posts = np.searchsorted(row_offsets, source_edges, side="right") - 1
        post_mask[np.asarray(posts, dtype=np.int64)] = True
    return (
        np.flatnonzero(post_mask).astype(np.int64, copy=False),
        edge_count,
        contact_count,
    )


def motor_overlap(posts: np.ndarray, distance: np.ndarray, max_hops: int) -> dict[str, int]:
    values = np.asarray(distance[posts], dtype=np.int16)
    result = {str(hop): int(np.count_nonzero(values == hop)) for hop in range(max_hops + 1)}
    result[f"outside_{max_hops}_hops"] = int(np.count_nonzero(values < 0))
    return result


def post_summary(
    name: str,
    posts: np.ndarray,
    *,
    row_offsets: np.ndarray,
    plastic_row_offsets: np.ndarray,
    weights: np.ndarray,
    synapse_counts: np.ndarray,
    synapse_scale: float,
    epsilon: float,
    motor_distance: np.ndarray,
    max_motor_hops: int,
) -> dict[str, Any]:
    plastic_counts = np.asarray(
        plastic_row_offsets[posts + 1] - plastic_row_offsets[posts], dtype=np.uint64
    )
    distance_stats: dict[str, dict[str, float | int]] = {}
    for hop in range(max_motor_hops + 1):
        layer_posts = posts[np.asarray(motor_distance[posts] == hop)]
        distance_stats[str(hop)] = stats_for_posts(
            layer_posts,
            row_offsets,
            weights,
            synapse_counts,
            synapse_scale=synapse_scale,
            epsilon=epsilon,
        )
    outside_posts = posts[np.asarray(motor_distance[posts] < 0)]
    distance_stats[f"outside_{max_motor_hops}_hops"] = stats_for_posts(
        outside_posts,
        row_offsets,
        weights,
        synapse_counts,
        synapse_scale=synapse_scale,
        epsilon=epsilon,
    )
    return {
        "name": name,
        "posts": int(len(posts)),
        "posts_with_plastic_incoming": int(np.count_nonzero(plastic_counts)),
        "plastic_incoming_edges": int(np.sum(plastic_counts, dtype=np.uint64)),
        "incoming_weight_change": stats_for_posts(
            posts,
            row_offsets,
            weights,
            synapse_counts,
            synapse_scale=synapse_scale,
            epsilon=epsilon,
        ),
        "motor_distance": motor_overlap(posts, motor_distance, max_motor_hops),
        "motor_distance_weight_change": distance_stats,
    }


def render_report(payload: dict[str, Any]) -> str:
    max_hops = int(payload["max_motor_hops"])
    distance_columns = " | ".join(f"motor d{hop}" for hop in range(max_hops + 1))
    distance_rule = "|".join(["---:"] * (max_hops + 1))
    lines = [
        "# Flyppy reinforcement DAN reach",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- checkpoint neural step: {payload['checkpoint_neural_step']}",
        f"- reward DAN: PAM01 / {payload['source_groups']['reward_dan']['neurons']} neurons",
        f"- aversive DAN: PPL101 / {payload['source_groups']['aversive_dan']['neurons']} neurons",
        f"- all released dopamine neurons: {payload['source_groups']['all_dopamine']['neurons']}",
        f"- weight epsilon: {payload['weight_epsilon']:.1e}",
        "",
        "## DAN source reach",
        "",
        "| source | neurons | released outgoing edges | released contacts | directly reached posts |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("reward_dan", "aversive_dan", "all_dopamine"):
        item = payload["source_groups"][name]
        lines.append(
            f"| {name} | {item['neurons']} | {item['released_outgoing_edges']} | {item['released_outgoing_contacts']} | {item['directly_reached_posts']} |"
        )

    overlap = payload["direct_post_overlap"]
    lines.extend(
        [
            "",
            "## Reward / aversive direct-post overlap",
            "",
            f"- shared posts: {overlap['shared_posts']}",
            f"- reward-only posts: {overlap['reward_only_posts']}",
            f"- aversive-only posts: {overlap['aversive_only_posts']}",
            f"- reward側 overlap: {float(overlap['reward_overlap_fraction']):.3%}",
            f"- aversive側 overlap: {float(overlap['aversive_overlap_fraction']):.3%}",
            f"- Jaccard: {float(overlap['jaccard']):.3%}",
            "",
            "| motor distance | reward posts | aversive posts | shared | reward overlap | aversive overlap | Jaccard |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for hop, item in sorted(payload["motor_distance_overlap"].items(), key=lambda pair: int(pair[0])):
        lines.append(
            "| {hop} | {reward} | {aversive} | {shared} | {reward_overlap:.1%} | {aversive_overlap:.1%} | {jaccard:.1%} |".format(
                hop=hop,
                reward=item["reward_posts"],
                aversive=item["aversive_posts"],
                shared=item["shared_posts"],
                reward_overlap=float(item["reward_overlap_fraction"]),
                aversive_overlap=float(item["aversive_overlap_fraction"]),
                jaccard=float(item["jaccard"]),
            )
        )

    exposure = payload["commit_log_exposure"]
    reward_source = payload["source_groups"]["reward_dan"]
    aversive_source = payload["source_groups"]["aversive_dan"]
    ratio = exposure["reward_over_aversive_event_weighted_contact_ratio"]
    ratio_text = "n/a" if ratio is None else f"{float(ratio):.3f}x"
    lines.extend(
        [
            "",
            "## Async commit log exposure proxy",
            "",
            f"- coverage: episode {exposure['episode_start']}..{exposure['episode_end']} / {exposure['records']} commits",
            f"- reward events: {exposure['reward_events']}",
            f"- aversive events: {exposure['aversive_events']}",
            "",
            "| source | events | event × neurons | event × outgoing edges | event × released contacts |",
            "|---|---:|---:|---:|---:|",
            f"| reward_dan | {reward_source['observed_events']} | {reward_source['event_weighted_neurons']} | {reward_source['event_weighted_edges']} | {reward_source['event_weighted_contacts']} |",
            f"| aversive_dan | {aversive_source['observed_events']} | {aversive_source['event_weighted_neurons']} | {aversive_source['event_weighted_edges']} | {aversive_source['event_weighted_contacts']} |",
            "",
            f"event × released contacts の reward / aversive 比: **{ratio_text}**",
            "",
            "この値は実DAN spike数そのものではなく、観測event数とreleased connectivityを掛けた構造的exposure proxyである。",
            "commit logはasync shared-weight化後のみを含み、それ以前のv3 serial episodeは含まない。",
        ]
    )

    lines.extend(
        [
            "",
            "## Reached-post set と学習weight",
            "",
            f"| post set | posts | plastic posts | plastic incoming edges | changed incoming | changed % | positive | negative | {distance_columns} | outside d{max_hops} |",
            f"|---|---:|---:|---:|---:|---:|---:|---:|{distance_rule}|---:|",
        ]
    )
    for row in payload["post_sets"]:
        stats = row["incoming_weight_change"]
        distance_values = " | ".join(str(row["motor_distance"][str(hop)]) for hop in range(max_hops + 1))
        lines.append(
            "| {name} | {posts} | {plastic_posts} | {plastic_edges} | {changed}/{edges} | {fraction:.3%} | "
            "{positive} | {negative} | {distance_values} | {outside} |".format(
                name=row["name"],
                posts=row["posts"],
                plastic_posts=row["posts_with_plastic_incoming"],
                plastic_edges=row["plastic_incoming_edges"],
                changed=int(stats["changed_edges"]),
                edges=int(stats["edges"]),
                fraction=float(stats["changed_fraction"]),
                positive=int(stats["positive_edges"]),
                negative=int(stats["negative_edges"]),
                distance_values=distance_values,
                outside=row["motor_distance"][f"outside_{max_hops}_hops"],
            )
        )

    lines.extend(
        [
            "",
            "## Motor近傍のreinforcement到達post",
            "",
            "| post set | motor distance | posts | changed incoming | changed % | positive | negative | mean signed delta | mean abs delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    primary = {row["name"]: row for row in payload["post_sets"]}
    for name in ("reward_reached", "aversive_reached", "task_reinforcement_union"):
        row = primary[name]
        for hop in range(1, min(max_hops, 3) + 1):
            stats = row["motor_distance_weight_change"][str(hop)]
            lines.append(
                "| {name} | {hop} | {posts} | {changed}/{edges} | {fraction:.3%} | {positive} | {negative} | {signed:.6g} | {absolute:.6g} |".format(
                    name=name,
                    hop=hop,
                    posts=row["motor_distance"][str(hop)],
                    changed=int(stats["changed_edges"]),
                    edges=int(stats["edges"]),
                    fraction=float(stats["changed_fraction"]),
                    positive=int(stats["positive_edges"]),
                    negative=int(stats["negative_edges"]),
                    signed=float(stats["mean_signed_delta_changed"]),
                    absolute=float(stats["mean_abs_delta_changed"]),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "`reward_reached` はPAM01から、`aversive_reached` はPPL101からreleased edgeを直接受けるpost集合である。",
            "direct-post overlapは構造的な共通到達先を示すだけで、同じepisode履歴でPAM/PPLが同じweight deltaを作ることまでは意味しない。",
            "`other_dopamine_only` はtaskで明示刺激する2群からは直接入力を受けないが、他のreleased dopamine neuronから入力を受けるpost集合である。",
            "weight変化はcheckpointの最終差分であり、その変化をPAM/PPLイベント単独へ因果帰属するものではない。",
            "motor distanceはreleased connectivity上でactuated motor neuronを0としてincoming方向へ遡った最短距離である。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.max_motor_hops < 0 or args.max_motor_hops > 12:
        raise SystemExit("--max-motor-hops must be in 0..12")
    if args.chunk_edges < 1:
        raise SystemExit("--chunk-edges must be positive")

    snapshot = absolute(args.snapshot)
    checkpoint = absolute(args.checkpoint)
    groups_path = absolute(args.groups)
    wing_map = absolute(args.wing_map)
    body_map = absolute(args.body_map)
    calibration_path = absolute(args.calibration)
    plastic_graph_path = absolute(args.plastic_graph)
    commit_log_path = absolute(args.commit_log)
    output_json = absolute(args.output_json)
    report = absolute(args.report)

    manifest = load_json(snapshot / "manifest.json")
    checkpoint_manifest = load_json(checkpoint / "manifest.json")
    groups_payload = load_json(groups_path)
    calibration = load_json(calibration_path)
    plastic_manifest = load_json(plastic_graph_path)
    n = int(manifest["neuron_count"])
    e = int(manifest["edge_count"])
    synapse_scale = float(calibration["synapse_scale"])

    body_ids = np.memmap(snapshot / str(manifest["body_ids_file"]), dtype="<u8", mode="r")
    row_offsets = np.memmap(snapshot / str(manifest["row_offsets_file"]), dtype="<u4", mode="r")
    pre_indices = np.memmap(snapshot / str(manifest["pre_indices_file"]), dtype="<u4", mode="r")
    synapse_counts = np.memmap(snapshot / str(manifest["synapse_counts_file"]), dtype="<u4", mode="r")
    neurotransmitters = np.memmap(
        snapshot / str(manifest["neurotransmitters_file"]), dtype=np.uint8, mode="r"
    )
    weights = np.memmap(
        checkpoint / str(checkpoint_manifest["weights_file"]), dtype="<f4", mode="r"
    )
    plastic_row_offsets = np.memmap(
        snapshot / str(plastic_manifest["files"]["plastic_row_offsets_u32le"]),
        dtype="<u4",
        mode="r",
    )
    if not (
        len(body_ids) == n
        and len(neurotransmitters) == n
        and len(row_offsets) == n + 1
        and len(pre_indices) == e
        and len(synapse_counts) == e
        and len(weights) == e
        and len(plastic_row_offsets) == n + 1
    ):
        raise RuntimeError("reinforcement reach input dimensions are inconsistent")

    groups = groups_payload.get("groups", {})
    reward_ids = set(int(value) for value in groups["reward_dan"]["body_ids"])
    aversive_ids = set(int(value) for value in groups["aversive_dan"]["body_ids"])
    reward_indices = resolve_indices(body_ids, reward_ids)
    aversive_indices = resolve_indices(body_ids, aversive_ids)
    dopamine_indices = np.flatnonzero(np.asarray(neurotransmitters) == DOPAMINE_CODE).astype(
        np.int64, copy=False
    )

    reward_posts, reward_edges, reward_contacts = direct_posts_from_sources(
        pre_indices,
        row_offsets,
        synapse_counts,
        reward_indices,
        neuron_count=n,
        chunk_edges=args.chunk_edges,
    )
    aversive_posts, aversive_edges, aversive_contacts = direct_posts_from_sources(
        pre_indices,
        row_offsets,
        synapse_counts,
        aversive_indices,
        neuron_count=n,
        chunk_edges=args.chunk_edges,
    )
    dopamine_posts, dopamine_edges, dopamine_contacts = direct_posts_from_sources(
        pre_indices,
        row_offsets,
        synapse_counts,
        dopamine_indices,
        neuron_count=n,
        chunk_edges=args.chunk_edges,
    )

    reward_mask = np.zeros(n, dtype=np.bool_)
    aversive_mask = np.zeros(n, dtype=np.bool_)
    dopamine_mask = np.zeros(n, dtype=np.bool_)
    reward_mask[reward_posts] = True
    aversive_mask[aversive_posts] = True
    dopamine_mask[dopamine_posts] = True

    both = np.flatnonzero(reward_mask & aversive_mask).astype(np.int64, copy=False)
    reward_only = np.flatnonzero(reward_mask & ~aversive_mask).astype(np.int64, copy=False)
    aversive_only = np.flatnonzero(aversive_mask & ~reward_mask).astype(np.int64, copy=False)
    task_union = np.flatnonzero(reward_mask | aversive_mask).astype(np.int64, copy=False)
    direct_post_overlap = {
        "shared_posts": int(len(both)),
        "reward_only_posts": int(len(reward_only)),
        "aversive_only_posts": int(len(aversive_only)),
        "union_posts": int(len(task_union)),
        "reward_overlap_fraction": (
            float(len(both)) / float(len(reward_posts)) if len(reward_posts) else 0.0
        ),
        "aversive_overlap_fraction": (
            float(len(both)) / float(len(aversive_posts)) if len(aversive_posts) else 0.0
        ),
        "jaccard": float(len(both)) / float(len(task_union)) if len(task_union) else 0.0,
    }
    other_dopamine_only = np.flatnonzero(dopamine_mask & ~(reward_mask | aversive_mask)).astype(
        np.int64, copy=False
    )

    wing_ids, somatic_ids = selected_motor_ids(wing_map, body_map)
    motor_indices = np.asarray(
        sorted(
            set(resolve_indices(body_ids, wing_ids).tolist())
            | set(resolve_indices(body_ids, somatic_ids).tolist())
        ),
        dtype=np.int64,
    )
    motor_distance, _ = upstream_distances(
        row_offsets,
        pre_indices,
        motor_indices,
        args.max_motor_hops,
    )
    motor_distance_overlap: dict[str, dict[str, int | float]] = {}
    for hop in range(args.max_motor_hops + 1):
        reward_at_hop = reward_mask & (motor_distance == hop)
        aversive_at_hop = aversive_mask & (motor_distance == hop)
        shared_at_hop = reward_at_hop & aversive_at_hop
        union_at_hop = reward_at_hop | aversive_at_hop
        reward_count = int(np.count_nonzero(reward_at_hop))
        aversive_count = int(np.count_nonzero(aversive_at_hop))
        shared_count = int(np.count_nonzero(shared_at_hop))
        union_count = int(np.count_nonzero(union_at_hop))
        motor_distance_overlap[str(hop)] = {
            "reward_posts": reward_count,
            "aversive_posts": aversive_count,
            "shared_posts": shared_count,
            "union_posts": union_count,
            "reward_overlap_fraction": shared_count / reward_count if reward_count else 0.0,
            "aversive_overlap_fraction": shared_count / aversive_count if aversive_count else 0.0,
            "jaccard": shared_count / union_count if union_count else 0.0,
        }

    named_sets = [
        ("reward_reached", reward_posts),
        ("aversive_reached", aversive_posts),
        ("reward_only", reward_only),
        ("aversive_only", aversive_only),
        ("reward_and_aversive", both),
        ("task_reinforcement_union", task_union),
        ("other_dopamine_only", other_dopamine_only),
        ("all_dopamine_reached", dopamine_posts),
    ]
    post_sets = [
        post_summary(
            name,
            posts,
            row_offsets=row_offsets,
            plastic_row_offsets=plastic_row_offsets,
            weights=weights,
            synapse_counts=synapse_counts,
            synapse_scale=synapse_scale,
            epsilon=args.weight_epsilon,
            motor_distance=motor_distance,
            max_motor_hops=args.max_motor_hops,
        )
        for name, posts in named_sets
    ]

    commit_records = load_jsonl(commit_log_path) if commit_log_path.exists() else []
    reward_events = sum(int(row.get("reward_events", 0)) for row in commit_records)
    aversive_events = sum(int(row.get("aversive_events", 0)) for row in commit_records)
    commit_first_episode = (
        min(int(row["episode"]) for row in commit_records) if commit_records else None
    )
    commit_last_episode = (
        max(int(row["episode"]) for row in commit_records) if commit_records else None
    )

    reward_contact_events = reward_events * reward_contacts
    aversive_contact_events = aversive_events * aversive_contacts
    contact_event_ratio = (
        reward_contact_events / aversive_contact_events
        if aversive_contact_events > 0
        else None
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "snapshot": portable(snapshot),
        "checkpoint": portable(checkpoint),
        "checkpoint_neural_step": int(checkpoint_manifest.get("step", -1)),
        "weight_epsilon": float(args.weight_epsilon),
        "max_motor_hops": int(args.max_motor_hops),
        "source_groups": {
            "reward_dan": {
                "cell_type": "PAM01",
                "neurons": int(len(reward_indices)),
                "released_outgoing_edges": reward_edges,
                "released_outgoing_contacts": reward_contacts,
                "directly_reached_posts": int(len(reward_posts)),
                "observed_events": reward_events,
                "event_weighted_neurons": reward_events * int(len(reward_indices)),
                "event_weighted_edges": reward_events * reward_edges,
                "event_weighted_contacts": reward_contact_events,
            },
            "aversive_dan": {
                "cell_type": "PPL101",
                "neurons": int(len(aversive_indices)),
                "released_outgoing_edges": aversive_edges,
                "released_outgoing_contacts": aversive_contacts,
                "directly_reached_posts": int(len(aversive_posts)),
                "observed_events": aversive_events,
                "event_weighted_neurons": aversive_events * int(len(aversive_indices)),
                "event_weighted_edges": aversive_events * aversive_edges,
                "event_weighted_contacts": aversive_contact_events,
            },
            "all_dopamine": {
                "neurons": int(len(dopamine_indices)),
                "released_outgoing_edges": dopamine_edges,
                "released_outgoing_contacts": dopamine_contacts,
                "directly_reached_posts": int(len(dopamine_posts)),
            },
        },
        "commit_log_exposure": {
            "path": portable(commit_log_path),
            "records": len(commit_records),
            "episode_start": commit_first_episode,
            "episode_end": commit_last_episode,
            "reward_events": reward_events,
            "aversive_events": aversive_events,
            "reward_over_aversive_event_weighted_contact_ratio": contact_event_ratio,
            "scope_note": "covers async shared-weight commit log only; earlier serial v3 episodes are not present",
        },
        "actuated_motor_neurons": int(len(motor_indices)),
        "direct_post_overlap": direct_post_overlap,
        "motor_distance_overlap": motor_distance_overlap,
        "post_sets": post_sets,
    }
    save_json(output_json, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(payload), encoding="utf-8")

    for name in ("reward_dan", "aversive_dan", "all_dopamine"):
        item = payload["source_groups"][name]
        print(
            f"{name}: neurons={item['neurons']} outgoing_edges={item['released_outgoing_edges']} posts={item['directly_reached_posts']}"
        )
    for row in post_sets:
        stats = row["incoming_weight_change"]
        print(
            "{}: posts={} plastic_edges={} changed={}/{} ({:.3%}) motor_d1={} motor_d2={}".format(
                row["name"],
                row["posts"],
                row["plastic_incoming_edges"],
                stats["changed_edges"],
                stats["edges"],
                stats["changed_fraction"],
                row["motor_distance"].get("1", 0),
                row["motor_distance"].get("2", 0),
            )
        )
    print(f"json={output_json}")
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
