#!/usr/bin/env python3
"""Build a MaleCNS R1-R6 retinotopic stimulation map from released wiring.

The released MaleCNS volume does not contain a complete anatomical eye map for
outer photoreceptors.  In particular, the lamina is incompletely contained and
R1-R6 reconstructions are an undercount of the biological eye.  We therefore do
not fabricate missing receptors or force six cells into every column.

For every released neuron whose official MaleCNS ``type`` is exactly ``R1-R6``:

1. collect all released synaptic contacts onto coordinate-bearing L1/L2/L3
   neurons;
2. sum those contacts by the target's ``assignedOlHex1/assignedOlHex2`` column;
3. assign the photoreceptor to the column with the largest total observed contact
   count;
4. use the R1-R6 neuron's own ``rootSide`` as eye side.

This follows the same information-preserving inference used by an existing
MaleCNS retinal projection implementation.  Exact ties are left unmapped rather
than broken arbitrarily.  The map is a sensory-coordinate inference only; the
released neural graph itself is never filtered or rewritten.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from virtual_fly.reproducibility import git_provenance, sha256_file


ANNOTATION_FILENAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
ANNOTATION_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
    f"flat-connectome/{ANNOTATION_FILENAME}"
)
LAMINA_ANCHOR_TYPES = frozenset(("L1", "L2", "L3"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument(
        "--raw-annotations",
        type=Path,
        default=Path("artifacts/raw/male-cns-v1.0") / ANNOTATION_FILENAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the official MaleCNS annotation feather only if it is missing",
    )
    parser.add_argument(
        "--min-columns-per-side",
        type=int,
        default=100,
        help="fail rather than silently run with a badly resolved eye map",
    )
    return parser.parse_args()


def ensure_annotations(path: Path, download_missing: bool) -> None:
    if path.exists():
        return
    if not download_missing:
        raise SystemExit(
            f"official MaleCNS annotations not found at {path}; rerun with --download"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"downloading {ANNOTATION_URL}")
    with urllib.request.urlopen(ANNOTATION_URL) as response, partial.open("wb") as out:
        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
    partial.replace(path)


def text_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[name].fillna("").astype(str)


def numeric_coordinate(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def root_sides(frame: pd.DataFrame) -> np.ndarray:
    return text_series(frame, "rootSide").str.upper().str.strip().to_numpy(dtype=str)


def build_projection(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    row_offsets: np.ndarray,
    pre_indices: np.ndarray,
    synapse_counts: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    aligned = (
        annotations.drop_duplicates("bodyId")
        .set_index("bodyId")
        .reindex(body_ids)
        .reset_index()
    )
    types = text_series(aligned, "type")
    receptor_mask = types.eq("R1-R6").to_numpy()
    anchor_mask = types.isin(LAMINA_ANCHOR_TYPES).to_numpy()
    h1 = numeric_coordinate(aligned["assignedOlHex1"])
    h2 = numeric_coordinate(aligned["assignedOlHex2"])
    sides = root_sides(aligned)
    coordinate_anchor_mask = anchor_mask & np.isfinite(h1) & np.isfinite(h2)

    # Incoming CSR is arranged by post neuron.  Aggregate every observed
    # R1-R6 -> L1/L2/L3 edge into a per-receptor, per-column contact total.
    receptor_columns: dict[int, dict[tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    receptor_anchor_types: dict[int, dict[tuple[int, int], Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    raw_candidate_edges = 0

    for post_idx in np.flatnonzero(coordinate_anchor_mask):
        key = (int(h1[post_idx]), int(h2[post_idx]))
        anchor_type = str(types.iloc[post_idx])
        begin = int(row_offsets[post_idx])
        end = int(row_offsets[post_idx + 1])
        incoming = pre_indices[begin:end].astype(np.int64, copy=False)
        counts = synapse_counts[begin:end].astype(np.uint32, copy=False)
        keep = receptor_mask[incoming]
        for pre_idx, count in zip(
            incoming[keep].tolist(), counts[keep].tolist(), strict=True
        ):
            receptor_columns[int(pre_idx)][key] += int(count)
            receptor_anchor_types[int(pre_idx)][key][anchor_type] += int(count)
            raw_candidate_edges += 1

    # Group non-ambiguous receptor assignments by eye side + optical column.
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    confidences: list[float] = []
    ambiguous_ties: list[dict[str, object]] = []
    invalid_side: list[int] = []
    multi_column_candidates = 0

    for pre_idx in np.flatnonzero(receptor_mask):
        side = str(sides[pre_idx])
        if side not in ("L", "R"):
            invalid_side.append(int(body_ids[pre_idx]))
            continue
        counts_by_column = receptor_columns.get(int(pre_idx))
        if not counts_by_column:
            continue
        if len(counts_by_column) > 1:
            multi_column_candidates += 1

        max_contacts = max(counts_by_column.values())
        winners = sorted(
            key for key, contacts in counts_by_column.items() if contacts == max_contacts
        )
        total_contacts = int(sum(counts_by_column.values()))
        if len(winners) != 1:
            ambiguous_ties.append(
                {
                    "r1_r6_body_id": int(body_ids[pre_idx]),
                    "root_side": side,
                    "max_column_contacts": int(max_contacts),
                    "total_anchor_contacts": total_contacts,
                    "tied_columns": [list(key) for key in winners],
                }
            )
            continue

        hex1, hex2 = winners[0]
        confidence = float(max_contacts / total_contacts)
        confidences.append(confidence)
        grouped[(side, hex1, hex2)].append(
            {
                "body_id": int(body_ids[pre_idx]),
                "dominant_contacts": int(max_contacts),
                "total_anchor_contacts": total_contacts,
                "projection_confidence": confidence,
                "anchor_contact_breakdown": dict(
                    sorted(receptor_anchor_types[int(pre_idx)][(hex1, hex2)].items())
                ),
            }
        )

    columns: list[dict[str, object]] = []
    for (side, hex1, hex2), receptors in sorted(grouped.items()):
        receptors.sort(key=lambda item: int(item["body_id"]))
        columns.append(
            {
                "side": side,
                "hex1": hex1,
                "hex2": hex2,
                "r1_r6_body_ids": [int(item["body_id"]) for item in receptors],
                "r1_r6_assignments": receptors,
                "assignment": "dominant_observed_contacts_to_L1_L2_L3_column",
            }
        )

    total_receptors = int(receptor_mask.sum())
    mapped_receptors = int(sum(len(item["r1_r6_body_ids"]) for item in columns))
    candidate_receptors = len(receptor_columns)
    confidence_array = np.asarray(confidences, dtype=np.float64)
    diagnostics: dict[str, object] = {
        "annotated_r1_r6_in_snapshot": total_receptors,
        "coordinate_bearing_l1_l2_l3": int(coordinate_anchor_mask.sum()),
        "r1_r6_with_anchor_contacts": candidate_receptors,
        "mapped_r1_r6_neurons": mapped_receptors,
        "unmapped_without_coordinate_anchor_contacts": total_receptors
        - candidate_receptors,
        "unmapped_invalid_root_side": len(invalid_side),
        "unmapped_exact_ties": len(ambiguous_ties),
        "r1_r6_with_multiple_candidate_columns": multi_column_candidates,
        "raw_r1_r6_to_lamina_candidate_edges": raw_candidate_edges,
        "projection_confidence_median": (
            float(np.median(confidence_array)) if len(confidence_array) else 0.0
        ),
        "projection_confidence_below_0_8": int(
            np.sum(confidence_array < 0.8)
        ),
        "invalid_root_side_sample": invalid_side[:16],
        "exact_tie_sample": ambiguous_ties[:16],
    }
    return columns, diagnostics


def validate_columns(columns: list[dict[str, object]], minimum: int) -> None:
    counts = Counter(str(item["side"]) for item in columns)
    if any(counts[side] < minimum for side in ("L", "R")):
        raise RuntimeError(
            f"retinotopic map resolved too few columns: {dict(counts)}; "
            f"minimum per side is {minimum}. Refusing to substitute a guessed mapping."
        )

    coordinate_counts = Counter(
        (str(item["side"]), int(item["hex1"]), int(item["hex2"])) for item in columns
    )
    clashes = [key for key, count in coordinate_counts.items() if count > 1]
    if clashes:
        raise RuntimeError(f"duplicate optic-column coordinates: {clashes[:8]}")

    body_to_columns: dict[int, list[tuple[str, int, int]]] = {}
    for item in columns:
        key = (str(item["side"]), int(item["hex1"]), int(item["hex2"]))
        for body_id in item["r1_r6_body_ids"]:
            body_to_columns.setdefault(int(body_id), []).append(key)
    duplicated = {
        body_id: keys
        for body_id, keys in body_to_columns.items()
        if len(set(keys)) > 1
    }
    if duplicated:
        raise RuntimeError(
            "internal error: one R1-R6 neuron was assigned to multiple optical columns: "
            f"{list(duplicated.items())[:8]}"
        )


def main() -> int:
    args = parse_args()
    ensure_annotations(args.raw_annotations, args.download)

    body_ids = np.fromfile(args.snapshot / "body_ids.u64le", dtype="<u8")
    row_offsets = np.fromfile(args.snapshot / "row_offsets.u32le", dtype="<u4")
    pre_indices = np.fromfile(args.snapshot / "pre_indices.u32le", dtype="<u4")
    synapse_counts = np.fromfile(args.snapshot / "synapse_counts.u32le", dtype="<u4")
    if len(row_offsets) != len(body_ids) + 1:
        raise RuntimeError("snapshot row_offsets length is inconsistent with body IDs")
    if int(row_offsets[-1]) != len(pre_indices):
        raise RuntimeError("snapshot CSR edge count is inconsistent with pre_indices")
    if len(synapse_counts) != len(pre_indices):
        raise RuntimeError("snapshot synapse_counts length is inconsistent with pre_indices")

    annotations = pd.read_feather(args.raw_annotations)
    required = {
        "bodyId",
        "type",
        "rootSide",
        "assignedOlHex1",
        "assignedOlHex2",
    }
    missing = required - set(annotations.columns)
    if missing:
        raise RuntimeError(
            f"official annotation file is missing required retinotopy fields: {sorted(missing)}"
        )

    columns, diagnostics = build_projection(
        annotations,
        body_ids,
        row_offsets,
        pre_indices,
        synapse_counts,
    )
    validate_columns(columns, args.min_columns_per_side)

    column_counts = Counter(str(item["side"]) for item in columns)
    receptor_counts = Counter(len(item["r1_r6_body_ids"]) for item in columns)
    method = "dominant_R1-R6_contacts_to_coordinate_L1_L2_L3_with_R1-R6_rootSide"

    output = {
        "schema_version": 3,
        "dataset": "male-cns:v1.0",
        "artifact_provenance": {
            "generator": "scripts/data/prepare_retinotopic_vision.py",
            "generator_git": git_provenance(Path(__file__).resolve().parents[2]),
            "snapshot_manifest_sha256": sha256_file(args.snapshot / "manifest.json"),
        },
        "sensory_boundary": "released R1-R6 photoreceptors",
        "column_coordinate_system": "MaleCNS assignedOlHex1/assignedOlHex2",
        "assignment_method": method,
        "columns": columns,
        "counts": {
            "left_columns": column_counts["L"],
            "right_columns": column_counts["R"],
            "r1_r6_per_column_histogram": {
                str(key): value for key, value in sorted(receptor_counts.items())
            },
            **diagnostics,
        },
        "provenance": {
            "photoreceptor_identity": "observed: official MaleCNS type == R1-R6",
            "optic_column_coordinates": (
                "observed: official MaleCNS assignedOlHex coordinates on L1/L2/L3"
            ),
            "eye_side": "observed: R1-R6 rootSide",
            "column_assignment": (
                "inferred from observed wiring: sum all released R1-R6 -> L1/L2/L3 "
                "contacts by assignedOlHex column and choose the unique maximum"
            ),
            "tie_policy": "leave exact ties unmapped; never guess",
            "incomplete_volume_policy": (
                "do not synthesize missing R1-R6 cells or force six cells per column"
            ),
            "no_visual_feature_extraction": True,
            "no_spatial_averaging": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"assignment_method={method}")
    print(f"left_columns={column_counts['L']}")
    print(f"right_columns={column_counts['R']}")
    print(f"annotated_r1_r6={diagnostics['annotated_r1_r6_in_snapshot']}")
    print(f"mapped_r1_r6={diagnostics['mapped_r1_r6_neurons']}")
    print(f"r1_r6_per_column={dict(sorted(receptor_counts.items()))}")
    print(f"coordinate_bearing_l1_l2_l3={diagnostics['coordinate_bearing_l1_l2_l3']}")
    print(
        "r1_r6_with_multiple_candidate_columns={}".format(
            diagnostics["r1_r6_with_multiple_candidate_columns"]
        )
    )
    print(
        "projection_confidence_median={:.6f}".format(
            diagnostics["projection_confidence_median"]
        )
    )
    print(
        "projection_confidence_below_0_8={}".format(
            diagnostics["projection_confidence_below_0_8"]
        )
    )
    print(f"unmapped_exact_ties={diagnostics['unmapped_exact_ties']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
