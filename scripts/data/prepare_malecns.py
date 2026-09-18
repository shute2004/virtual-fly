#!/usr/bin/env python3
"""Prepare MaleCNS v1.0 for the virtual-fly neural runtime.

This script preserves the released connection counts and annotations. It does
not learn weights, infer missing edges, or apply a game-specific policy.
Model-specific fast signs and plasticity are runtime concerns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as feather

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}

NT_CODES = {
    "acetylcholine": 1,
    "gaba": 2,
    "glutamate": 3,
    "dopamine": 4,
    "octopamine": 5,
    "serotonin": 6,
    "histamine": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/raw/male-cns-v1.0"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing official MaleCNS files from the public Google Storage bucket",
    )
    parser.add_argument(
        "--min-synapses",
        type=int,
        default=1,
        help="minimum released connection count per pre->post pair; default 1 preserves all released pairs",
    )
    return parser.parse_args()


def download(url: str, target: Path) -> None:
    if target.exists():
        print(f"using existing {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0"))
        done = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"  {done / (1024**2):8.1f} / {total / (1024**2):.1f} MiB", end="\r")
    if total:
        print()
    partial.replace(target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, download_missing: bool) -> None:
    if path.exists():
        return
    if not download_missing:
        raise SystemExit(f"missing {path}; rerun with --download")
    download(f"{BASE_URL}/{path.name}", path)


def select_annotated_neurons(annotations_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Retain annotated neuronal entries rather than an arbitrary status subset.

    MaleCNS reports 166,691 proofread and annotated neurons, while selecting only
    ``status == Traced`` yields a smaller subset.  For virtual-fly the source
    snapshot should be as close as practical to the released whole-CNS neuron
    inventory, so the import criterion is a non-empty neuronal ``superclass``
    annotation with explicit glia excluded.  The exact resulting count is
    recorded in the manifest rather than asserted against a paper headline.
    """

    annotations = pd.read_feather(annotations_path)
    required = {"bodyId", "status", "superclass"}
    missing = required - set(annotations.columns)
    if missing:
        raise RuntimeError(f"annotation file missing columns: {sorted(missing)}")

    superclass = annotations["superclass"].fillna("").astype(str).str.strip()
    status = annotations["status"].fillna("").astype(str).str.strip().str.lower()
    keep = superclass.ne("") & status.ne("glia")
    if "statusLabel" in annotations.columns:
        status_label = (
            annotations["statusLabel"].fillna("").astype(str).str.strip().str.lower()
        )
        keep &= status_label.ne("glia")

    neurons = annotations[keep].copy().drop_duplicates(subset=["bodyId"])
    bodies = np.sort(neurons["bodyId"].astype(np.uint64).to_numpy())
    if len(bodies) == 0:
        raise RuntimeError("neuron selection produced an empty MaleCNS snapshot")
    print(f"annotated neuronal entries: {len(bodies):,}")
    return bodies, neurons


def membership_indices(values: np.ndarray, sorted_bodies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_bodies, values)
    in_bounds = positions < len(sorted_bodies)
    safe = np.minimum(positions, len(sorted_bodies) - 1)
    present = in_bounds & (sorted_bodies[safe] == values)
    return positions.astype(np.int64, copy=False), present


def build_neurotransmitters(path: Path, bodies: np.ndarray) -> np.ndarray:
    nt = pd.read_feather(path)
    if "body" not in nt.columns or "consensus_nt" not in nt.columns:
        raise RuntimeError("neurotransmitter file missing body/consensus_nt columns")
    nt = nt.dropna(subset=["body"]).drop_duplicates(subset=["body"])
    nt_map = nt.set_index("body")["consensus_nt"]
    labels = nt_map.reindex(bodies).fillna("unknown").astype(str).str.lower()
    return np.fromiter((NT_CODES.get(value, 0) for value in labels), dtype=np.uint8, count=len(bodies))


def build_connectivity(weights_path: Path, bodies: np.ndarray, min_synapses: int):
    if min_synapses < 1:
        raise ValueError("--min-synapses must be >= 1")

    print("reading released connection table (memory-mapped where supported) ...")
    table = feather.read_table(weights_path, memory_map=True)
    expected = {"body_pre", "body_post", "weight"}
    missing = expected - set(table.column_names)
    if missing:
        raise RuntimeError(f"weights file missing columns: {sorted(missing)}")
    if min_synapses > 1:
        table = table.filter(pc.greater_equal(table["weight"], min_synapses))

    pre_body = table["body_pre"].to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    post_body = table["body_post"].to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    counts = table["weight"].to_numpy(zero_copy_only=False)
    del table

    pre_index, pre_ok = membership_indices(pre_body, bodies)
    post_index, post_ok = membership_indices(post_body, bodies)
    keep = pre_ok & post_ok
    pre_index = pre_index[keep]
    post_index = post_index[keep]
    counts = counts[keep]
    del pre_body, post_body, pre_ok, post_ok, keep

    if len(pre_index) > np.iinfo(np.uint32).max:
        raise RuntimeError("edge count exceeds current u32 runtime format")
    if np.max(counts, initial=0) > np.iinfo(np.uint32).max:
        raise RuntimeError("synapse count exceeds u32")

    print(f"retained neuron->neuron pairs: {len(pre_index):,}")
    order = np.lexsort((pre_index, post_index))
    pre_index = pre_index[order].astype(np.uint32, copy=False)
    post_index = post_index[order].astype(np.uint32, copy=False)
    counts = counts[order].astype(np.uint32, copy=False)

    per_post = np.bincount(post_index, minlength=len(bodies)).astype(np.uint64, copy=False)
    row_offsets_u64 = np.empty(len(bodies) + 1, dtype=np.uint64)
    row_offsets_u64[0] = 0
    np.cumsum(per_post, out=row_offsets_u64[1:])
    if row_offsets_u64[-1] > np.iinfo(np.uint32).max:
        raise RuntimeError("CSR offsets exceed current u32 runtime format")
    row_offsets = row_offsets_u64.astype(np.uint32)
    return row_offsets, pre_index, counts


def write_metadata_subset(neurons: pd.DataFrame, bodies: np.ndarray, output: Path) -> None:
    columns = [
        name
        for name in (
            "bodyId",
            "type",
            "flywireType",
            "instance",
            "class",
            "subclass",
            "superclass",
            "side",
            "somaSide",
            "somaLocation",
            "tosomaLocation",
            "rootSide",
            "nerve",
            "entryNerve",
            "exitNerve",
            "receptorType",
            "fruDsx",
            "status",
            "statusLabel",
        )
        if name in neurons.columns
    ]
    aligned = neurons.drop_duplicates("bodyId").set_index("bodyId").reindex(bodies).reset_index()
    feather.write_feather(aligned[columns], output / "annotations.feather")


def main() -> int:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    paths = {name: args.raw_dir / filename for name, filename in FILES.items()}
    for path in paths.values():
        require_file(path, args.download)

    bodies, neurons = select_annotated_neurons(paths["annotations"])
    transmitters = build_neurotransmitters(paths["neurotransmitters"], bodies)
    row_offsets, pre_indices, counts = build_connectivity(
        paths["weights"], bodies, args.min_synapses
    )

    np.asarray(bodies, dtype="<u8").tofile(args.output / "body_ids.u64le")
    np.asarray(row_offsets, dtype="<u4").tofile(args.output / "row_offsets.u32le")
    np.asarray(pre_indices, dtype="<u4").tofile(args.output / "pre_indices.u32le")
    np.asarray(counts, dtype="<u4").tofile(args.output / "synapse_counts.u32le")
    np.asarray(transmitters, dtype=np.uint8).tofile(args.output / "neurotransmitters.u8")
    write_metadata_subset(neurons, bodies, args.output)

    source_hashes = {key: sha256(path) for key, path in paths.items()}
    manifest = {
        "format_version": 1,
        "dataset": "male-cns:v1.0",
        "neuron_count": int(len(bodies)),
        "edge_count": int(len(pre_indices)),
        "synapse_count_sum": int(counts.astype(np.uint64).sum()),
        "body_ids_file": "body_ids.u64le",
        "row_offsets_file": "row_offsets.u32le",
        "pre_indices_file": "pre_indices.u32le",
        "synapse_counts_file": "synapse_counts.u32le",
        "neurotransmitters_file": "neurotransmitters.u8",
        "annotations_file": "annotations.feather",
        "source_sha256": source_hashes,
        "source_base_url": BASE_URL,
        "source_license": "CC-BY",
        "source_edge_filter": {"min_released_synapse_count": args.min_synapses},
        "node_filter": "non-empty superclass; exclude explicit Glia in status/statusLabel",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"wrote snapshot: {args.output}")
    print(f"  neurons: {len(bodies):,}")
    print(f"  edges:   {len(pre_indices):,}")
    print(f"  synapse-count sum: {int(counts.astype(np.uint64).sum()):,}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
