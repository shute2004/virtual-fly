#!/usr/bin/env python3
"""Compile and measure the structurally plastic subset of the released MaleCNS graph.

This is a read-only analysis with respect to the source snapshot. It derives the
subset that can causally change weight under the *current* bootstrap plasticity
rule:

    E_plastic = { fast edge (pre -> post) | post has released dopamine input }

The definition follows the current CPU/GPU model exactly: postsynaptic
modulation can only be produced by released dopaminergic presynaptic neurons,
and only fast-transmitter edges have mutable fast weights.

The script writes compact derived graph arrays under the snapshot directory and
an auditable Markdown report under reports/flyppy/. It does not modify training
checkpoints, curriculum state, or learned weights.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_GRAPH = Path("artifacts/malecns-v1.0/plastic-fast-graph-v1.json")
DEFAULT_REPORT = Path("reports/flyppy/plastic_fast_graph.md")

NT_ACETYLCHOLINE = 1
NT_GABA = 2
NT_GLUTAMATE = 3
NT_DOPAMINE = 4
NT_HISTAMINE = 7
FAST_CODES = frozenset(
    (NT_ACETYLCHOLINE, NT_GABA, NT_GLUTAMATE, NT_HISTAMINE)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_manifest(snapshot: Path) -> dict[str, Any]:
    path = snapshot / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object in {path}")
    return payload


def mib(value: int) -> float:
    return value / (1024.0**2)


def gib(value: int) -> float:
    return value / (1024.0**3)


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def main() -> int:
    args = parse_args()
    snapshot = absolute(args.snapshot)
    graph_path = absolute(args.graph)
    report_path = absolute(args.report)
    manifest = load_manifest(snapshot)

    n = int(manifest["neuron_count"])
    e = int(manifest["edge_count"])
    row_offsets = np.fromfile(
        snapshot / str(manifest["row_offsets_file"]), dtype="<u4"
    )
    pre_indices = np.memmap(
        snapshot / str(manifest["pre_indices_file"]), dtype="<u4", mode="r"
    )
    synapse_counts = np.memmap(
        snapshot / str(manifest["synapse_counts_file"]), dtype="<u4", mode="r"
    )
    neurotransmitters = np.fromfile(
        snapshot / str(manifest["neurotransmitters_file"]), dtype=np.uint8
    )
    role_file = manifest.get("modulator_roles_file")
    if role_file:
        modulator_roles = np.fromfile(snapshot / str(role_file), dtype=np.uint8)
    else:
        modulator_roles = (neurotransmitters == NT_DOPAMINE).astype(np.uint8)

    if row_offsets.size != n + 1:
        raise RuntimeError("row_offsets length does not match manifest neuron_count")
    if pre_indices.size != e or synapse_counts.size != e:
        raise RuntimeError("edge arrays do not match manifest edge_count")
    if neurotransmitters.size != n:
        raise RuntimeError("neurotransmitters length does not match neuron_count")
    if modulator_roles.size != n:
        raise RuntimeError("modulator_roles length does not match neuron_count")
    if int(row_offsets[-1]) != e:
        raise RuntimeError("final row offset does not match edge_count")

    dopamine_post = np.zeros(n, dtype=np.bool_)
    dopamine_in_degree = np.zeros(n, dtype=np.uint32)
    fast_in_degree = np.zeros(n, dtype=np.uint32)
    plastic_in_degree = np.zeros(n, dtype=np.uint32)
    plastic_row_offsets = np.zeros(n + 1, dtype="<u4")

    fast_edges = 0
    dopamine_edges = 0
    fast_contacts = 0
    dopamine_contacts = 0

    # Pass 1: derive S_D and per-post structural counts. The source snapshot is
    # incoming CSR, so this preserves its exact row ordering.
    for post in range(n):
        begin = int(row_offsets[post])
        end = int(row_offsets[post + 1])
        if begin == end:
            plastic_row_offsets[post + 1] = plastic_row_offsets[post]
            continue
        pres = np.asarray(pre_indices[begin:end], dtype=np.uint32)
        nts = neurotransmitters[pres]
        fast_mask = np.isin(nts, tuple(FAST_CODES))
        dopamine_mask = modulator_roles[pres] != 0

        fast_count = int(np.count_nonzero(fast_mask))
        dopamine_count = int(np.count_nonzero(dopamine_mask))
        fast_in_degree[post] = fast_count
        dopamine_in_degree[post] = dopamine_count
        fast_edges += fast_count
        dopamine_edges += dopamine_count

        counts = np.asarray(synapse_counts[begin:end], dtype=np.uint32)
        if fast_count:
            fast_contacts += int(counts[fast_mask].sum(dtype=np.uint64))
        if dopamine_count:
            dopamine_contacts += int(counts[dopamine_mask].sum(dtype=np.uint64))
            dopamine_post[post] = True
            plastic_in_degree[post] = fast_count

        next_offset = int(plastic_row_offsets[post]) + int(plastic_in_degree[post])
        if next_offset > np.iinfo(np.uint32).max:
            raise RuntimeError("plastic edge count exceeds u32 graph index range")
        plastic_row_offsets[post + 1] = next_offset

    p = int(plastic_row_offsets[-1])
    s_d = int(np.count_nonzero(dopamine_post))

    # Pass 2: materialize compact PlasticFastGraph in original incoming order.
    # Each plastic index maps back to its immutable source edge and post neuron.
    plastic_edge_indices = np.empty(p, dtype="<u4")
    plastic_post_indices = np.empty(p, dtype="<u4")
    cursor = 0
    plastic_contacts = 0
    for post in np.flatnonzero(dopamine_post):
        begin = int(row_offsets[post])
        end = int(row_offsets[post + 1])
        pres = np.asarray(pre_indices[begin:end], dtype=np.uint32)
        nts = neurotransmitters[pres]
        fast_mask = np.isin(nts, tuple(FAST_CODES))
        local_offsets = np.flatnonzero(fast_mask).astype(np.uint32, copy=False)
        count = int(local_offsets.size)
        if count == 0:
            continue
        original_edges = local_offsets + np.uint32(begin)
        plastic_edge_indices[cursor : cursor + count] = original_edges
        plastic_post_indices[cursor : cursor + count] = np.uint32(post)
        plastic_contacts += int(
            np.asarray(synapse_counts[original_edges], dtype=np.uint32).sum(
                dtype=np.uint64
            )
        )
        cursor += count
    if cursor != p:
        raise RuntimeError(f"PlasticFastGraph materialization mismatch: {cursor} != {p}")

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    stem = graph_path.stem
    row_file = graph_path.with_name(f"{stem}.row_offsets.u32le")
    edge_file = graph_path.with_name(f"{stem}.edge_indices.u32le")
    post_file = graph_path.with_name(f"{stem}.post_indices.u32le")
    plastic_row_offsets.tofile(row_file)
    plastic_edge_indices.tofile(edge_file)
    plastic_post_indices.tofile(post_file)

    # Current population runtime allocation model.
    neuron_local_bytes = 28 * n
    current_edge_local_bytes = 20 * e  # weight+eligibility + shift+lo+hi
    current_slot_bytes = neuron_local_bytes + current_edge_local_bytes

    # First redesign step: mutable weight+eligibility only on P. A dense
    # transaction over P is shown separately because it is still only an
    # intermediate design, not the target.
    plastic_synapse_bytes = 8 * p
    dense_plastic_transaction_bytes = 12 * p
    plastic_bitmap_bytes = math.ceil(p / 8)
    full_edge_bitmap_bytes = math.ceil(e / 8)
    compact_p_dense_txn_slot_bytes = (
        neuron_local_bytes + plastic_synapse_bytes + dense_plastic_transaction_bytes
    )
    compact_p_sparse_txn_base_bytes = (
        neuron_local_bytes + plastic_synapse_bytes + plastic_bitmap_bytes
    )

    dirty_examples: dict[str, int] = {}
    for fraction in (0.001, 0.01, 0.10, 1.0):
        dirty = math.ceil(p * fraction)
        # u32 plastic/original edge id + shift/lo/hi = 16 bytes/dirty edge.
        dirty_examples[f"dirty_{fraction:g}_slot_bytes"] = (
            compact_p_sparse_txn_base_bytes + 16 * dirty
        )

    active_degrees = plastic_in_degree[dopamine_post].astype(np.uint32, copy=False)
    graph_payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": manifest.get("dataset"),
        "source_neuron_count": n,
        "source_edge_count": e,
        "definition": (
            "fast presynaptic edge whose postsynaptic neuron has at least one "
            "released incoming edge from an annotation-supported dopaminergic modulator"
        ),
        "fast_transmitter_codes": sorted(FAST_CODES),
        "dopamine_transmitter_code": NT_DOPAMINE,
        "modulator_role_source": manifest.get("modulator_role_definition", "legacy transmitter-code fallback"),
        "dopamine_capable_post_count": s_d,
        "plastic_edge_count": p,
        "fast_edge_count": fast_edges,
        "dopamine_edge_count": dopamine_edges,
        "plastic_contact_count": plastic_contacts,
        "fast_contact_count": fast_contacts,
        "dopamine_contact_count": dopamine_contacts,
        "files": {
            "plastic_row_offsets_u32le": row_file.name,
            "plastic_edge_indices_u32le": edge_file.name,
            "plastic_post_indices_u32le": post_file.name,
        },
        "memory_bytes": {
            "current_neuron_local_per_slot": neuron_local_bytes,
            "current_full_edge_local_per_slot": current_edge_local_bytes,
            "current_total_per_slot": current_slot_bytes,
            "plastic_weight_eligibility_per_slot": plastic_synapse_bytes,
            "dense_transaction_over_plastic_edges_per_slot": dense_plastic_transaction_bytes,
            "plastic_dirty_bitmap_per_slot": plastic_bitmap_bytes,
            "full_edge_dirty_bitmap_per_slot": full_edge_bitmap_bytes,
            "compact_P_dense_transaction_total_per_slot": compact_p_dense_txn_slot_bytes,
            "compact_P_sparse_transaction_base_per_slot": compact_p_sparse_txn_base_bytes,
            **dirty_examples,
        },
    }
    graph_path.write_text(
        json.dumps(graph_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MaleCNS PlasticFastGraph structural analysis",
        "",
        f"- generated_at_utc: {graph_payload['generated_at_utc']}",
        f"- dataset: `{manifest.get('dataset')}`",
        "- mutates training/checkpoint state: false",
        "- contract: current bootstrap dopamine/modulation + fast-transmitter plasticity rule",
        "",
        "## Structural result",
        "",
        "| metric | count | fraction |",
        "|---|---:|---:|",
        f"| neurons N | {n:,} | 100% |",
        f"| all edges E | {e:,} | 100% |",
        f"| fast-transmitter edges | {fast_edges:,} | {pct(fast_edges, e):.4f}% of E |",
        f"| dopamine incoming edges | {dopamine_edges:,} | {pct(dopamine_edges, e):.4f}% of E |",
        f"| dopamine-capable posts S_D | {s_d:,} | {pct(s_d, n):.4f}% of N |",
        f"| PlasticFastGraph edges P | **{p:,}** | **{pct(p, e):.4f}% of E** |",
        f"| non-plastic edges E-P | {e - p:,} | {pct(e - p, e):.4f}% of E |",
        "",
        "Under the current rule, only P edges can ever acquire a non-zero weight delta. "
        "Eligibility on E-P cannot affect any future weight because their postsynaptic "
        "modulation is structurally zero.",
        "",
        "## Plastic incoming degree on S_D",
        "",
        f"- mean: {float(active_degrees.mean()) if active_degrees.size else 0.0:.3f}",
        f"- p50: {percentile(active_degrees, 50):.1f}",
        f"- p90: {percentile(active_degrees, 90):.1f}",
        f"- p99: {percentile(active_degrees, 99):.1f}",
        f"- max: {int(active_degrees.max()) if active_degrees.size else 0}",
        "",
        "## Per-slot GPU-state implications",
        "",
        "These are state-size projections only; they do not claim the future runtime "
        "will use every listed intermediate representation.",
        "",
        "| design | bytes / slot | MiB / slot | reduction vs current |",
        "|---|---:|---:|---:|",
        f"| current full-E state | {current_slot_bytes:,} | {mib(current_slot_bytes):.2f} | 1.00x |",
        f"| compact P, but still dense transaction on P | {compact_p_dense_txn_slot_bytes:,} | {mib(compact_p_dense_txn_slot_bytes):.2f} | {current_slot_bytes / max(1, compact_p_dense_txn_slot_bytes):.2f}x |",
        f"| compact P weight+eligibility + P-bit dirty bitmap, before dirty txn payload | {compact_p_sparse_txn_base_bytes:,} | {mib(compact_p_sparse_txn_base_bytes):.2f} | {current_slot_bytes / max(1, compact_p_sparse_txn_base_bytes):.2f}x |",
        "",
        f"A full-E dirty bitmap would be only {full_edge_bitmap_bytes:,} bytes "
        f"({mib(full_edge_bitmap_bytes):.2f} MiB); a P-only bitmap is "
        f"{plastic_bitmap_bytes:,} bytes ({mib(plastic_bitmap_bytes):.2f} MiB).",
        "",
        "If each dirty transaction record uses 16 bytes (u32 edge id + shift/lo/hi):",
        "",
        "| dirty fraction of P | projected MiB / slot |",
        "|---:|---:|",
    ]
    for fraction in (0.001, 0.01, 0.10, 1.0):
        value = dirty_examples[f"dirty_{fraction:g}_slot_bytes"]
        lines.append(f"| {100*fraction:.1f}% | {mib(value):.2f} |")
    lines.extend(
        [
            "",
            "## Generated compact graph",
            "",
            f"- metadata: `{graph_path.relative_to(ROOT) if graph_path.is_relative_to(ROOT) else graph_path}`",
            f"- row offsets: `{row_file.name}` ({plastic_row_offsets.size:,} u32)",
            f"- source edge indices: `{edge_file.name}` ({p:,} u32)",
            f"- post indices: `{post_file.name}` ({p:,} u32)",
            "- ordering: original incoming-CSR edge order is preserved within every post",
            "",
            "## Interpretation boundary",
            "",
            "This graph is only Phase B. It reduces the *possible mutable set* from E to P. "
            "The target runtime still needs dirty/sparse transaction state, outgoing adjacency, "
            "signal propagation frontiers, and eventually a live plasticity frontier so per-step "
            "work scales with current circuit activity rather than with P.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"plastic_fast_graph={graph_path}")
    print(f"plastic_fast_graph_report={report_path}")
    print(f"N={n}")
    print(f"E={e}")
    print(f"S_D={s_d}")
    print(f"P={p}")
    print(f"P_over_E={p / e if e else 0.0:.8f}")
    print(f"current_slot_MiB={mib(current_slot_bytes):.3f}")
    print(
        "compact_P_sparse_txn_base_MiB="
        f"{mib(compact_p_sparse_txn_base_bytes):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
