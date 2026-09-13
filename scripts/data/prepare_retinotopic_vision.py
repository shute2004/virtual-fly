#!/usr/bin/env python3
"""Build a local-light -> MaleCNS R1-R6 retinotopic stimulation map.

This script does not invent visual features. It preserves optic-lobe column position
and resolves the actual MaleCNS photoreceptor body IDs that should receive local
light-driven current.

Preferred source:
- official MaleCNS annotation columns ``assignedOlHex1`` / ``assignedOlHex2`` on
  R1-R6 neurons themselves.

Fallback when R1-R6 rows do not carry those coordinates:
- official MaleCNS L1 column coordinates;
- released connectome edges into each L1;
- incoming neurons whose released cell type is exactly R1-R6 / R1-6.

The fallback follows the biological neural-superposition relation (R1-R6
photoreceptors with a common optical axis converge on the same lamina cartridge),
but the body-ID assignment itself is derived from released MaleCNS connectivity,
not from geometric guessing.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATION_FILENAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
ANNOTATION_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
    f"flat-connectome/{ANNOTATION_FILENAME}"
)
R1_R6_PATTERN = re.compile(r"(?:^|[^A-Za-z0-9])R1(?:-|–)R?6(?:$|[^A-Za-z0-9])", re.I)


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
        help="download the official annotation feather only if it is missing",
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


def r1_r6_mask(frame: pd.DataFrame) -> np.ndarray:
    fields = [text_series(frame, name) for name in ("type", "flywireType", "instance")]
    searchable = fields[0] + " | " + fields[1] + " | " + fields[2]
    return searchable.map(lambda value: bool(R1_R6_PATTERN.search(value))).to_numpy()


def l1_mask(frame: pd.DataFrame) -> np.ndarray:
    fields = [text_series(frame, name) for name in ("type", "flywireType", "instance")]
    searchable = fields[0] + " | " + fields[1] + " | " + fields[2]
    return searchable.str.contains(r"(?:^|\W)L1(?:$|\W)", case=False, regex=True).to_numpy()


def side_values(frame: pd.DataFrame) -> np.ndarray:
    side = text_series(frame, "side").str.upper().str.strip()
    soma = text_series(frame, "somaSide").str.upper().str.strip()
    result = np.where(side.isin(["L", "R"]), side, soma)
    return np.asarray(result, dtype=str)


def integer_coordinate(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def body_index(sorted_body_ids: np.ndarray, body_id: int) -> int | None:
    pos = int(np.searchsorted(sorted_body_ids, np.uint64(body_id)))
    if pos >= len(sorted_body_ids) or int(sorted_body_ids[pos]) != int(body_id):
        return None
    return pos


def direct_columns(
    annotations: pd.DataFrame,
    snapshot_ids: set[int],
) -> tuple[list[dict[str, object]], str]:
    required = {"assignedOlHex1", "assignedOlHex2"}
    if not required.issubset(annotations.columns):
        return [], "missing_assignedOlHex_columns"

    mask = r1_r6_mask(annotations)
    rows = annotations.loc[mask].copy()
    if rows.empty:
        return [], "no_R1-R6_annotations"
    rows["_side"] = side_values(rows)
    rows["_h1"] = integer_coordinate(rows["assignedOlHex1"])
    rows["_h2"] = integer_coordinate(rows["assignedOlHex2"])
    rows = rows[
        rows["_side"].isin(["L", "R"])
        & np.isfinite(rows["_h1"])
        & np.isfinite(rows["_h2"])
    ]

    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for _, row in rows.iterrows():
        body_id = int(row["bodyId"])
        if body_id not in snapshot_ids:
            continue
        grouped[(str(row["_side"]), int(row["_h1"]), int(row["_h2"]))].append(body_id)

    columns = [
        {
            "side": side,
            "hex1": h1,
            "hex2": h2,
            "r1_r6_body_ids": sorted(set(body_ids)),
            "assignment": "observed_annotation",
        }
        for (side, h1, h2), body_ids in sorted(grouped.items())
        if body_ids
    ]
    return columns, "observed_R1-R6_assignedOlHex"


def connectivity_columns(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    row_offsets: np.ndarray,
    pre_indices: np.ndarray,
) -> tuple[list[dict[str, object]], str]:
    required = {"assignedOlHex1", "assignedOlHex2"}
    if not required.issubset(annotations.columns):
        raise RuntimeError(
            "official annotations do not contain assignedOlHex1/assignedOlHex2; "
            "cannot construct a retinotopic map without inventing geometry"
        )

    aligned = annotations.drop_duplicates("bodyId").set_index("bodyId").reindex(body_ids)
    r_mask = r1_r6_mask(aligned.reset_index())
    l_mask = l1_mask(aligned.reset_index())
    sides = side_values(aligned.reset_index())
    h1 = integer_coordinate(aligned["assignedOlHex1"])
    h2 = integer_coordinate(aligned["assignedOlHex2"])

    columns: list[dict[str, object]] = []
    for post_idx in np.flatnonzero(l_mask):
        side = sides[post_idx]
        if side not in ("L", "R") or not np.isfinite(h1[post_idx]) or not np.isfinite(h2[post_idx]):
            continue
        begin = int(row_offsets[post_idx])
        end = int(row_offsets[post_idx + 1])
        incoming = pre_indices[begin:end].astype(np.int64, copy=False)
        photoreceptors = incoming[r_mask[incoming]]
        if len(photoreceptors) == 0:
            continue
        columns.append(
            {
                "side": side,
                "hex1": int(h1[post_idx]),
                "hex2": int(h2[post_idx]),
                "l1_body_id": int(body_ids[post_idx]),
                "r1_r6_body_ids": sorted(
                    {int(body_ids[index]) for index in photoreceptors.tolist()}
                ),
                "assignment": "observed_connectivity_to_L1",
            }
        )
    return columns, "R1-R6_incoming_to_observed_L1_columns"


def validate_columns(columns: list[dict[str, object]], minimum: int) -> None:
    counts = Counter(str(item["side"]) for item in columns)
    missing = [side for side in ("L", "R") if counts[side] < minimum]
    if missing:
        raise RuntimeError(
            f"retinotopic map resolved too few columns: {dict(counts)}; "
            f"minimum per side is {minimum}. Refusing to substitute a guessed mapping."
        )
    duplicate = Counter(
        (str(item["side"]), int(item["hex1"]), int(item["hex2"])) for item in columns
    )
    clashes = [key for key, count in duplicate.items() if count > 1]
    if clashes:
        raise RuntimeError(f"duplicate optic-lobe column coordinates: {clashes[:8]}")


def main() -> int:
    args = parse_args()
    ensure_annotations(args.raw_annotations, args.download)

    body_ids = np.fromfile(args.snapshot / "body_ids.u64le", dtype="<u8")
    row_offsets = np.fromfile(args.snapshot / "row_offsets.u32le", dtype="<u4")
    pre_indices = np.fromfile(args.snapshot / "pre_indices.u32le", dtype="<u4")
    if len(row_offsets) != len(body_ids) + 1:
        raise RuntimeError("snapshot row_offsets length is inconsistent with body IDs")

    annotations = pd.read_feather(args.raw_annotations)
    required = {"bodyId", "type", "assignedOlHex1", "assignedOlHex2"}
    missing = required - set(annotations.columns)
    if missing:
        raise RuntimeError(
            f"official annotation file is missing required retinotopy columns: {sorted(missing)}"
        )

    snapshot_set = {int(value) for value in body_ids.tolist()}
    columns, method = direct_columns(annotations, snapshot_set)
    direct_counts = Counter(str(item["side"]) for item in columns)
    if any(direct_counts[side] < args.min_columns_per_side for side in ("L", "R")):
        print(
            "R1-R6 direct column annotations are incomplete; "
            "resolving R1-R6 by released incoming connectivity to L1"
        )
        columns, method = connectivity_columns(
            annotations, body_ids, row_offsets, pre_indices
        )

    validate_columns(columns, args.min_columns_per_side)
    counts = Counter(str(item["side"]) for item in columns)
    r_counts = Counter(len(item["r1_r6_body_ids"]) for item in columns)
    unique_receptors = sorted(
        {
            int(body_id)
            for item in columns
            for body_id in item["r1_r6_body_ids"]
        }
    )

    output = {
        "schema_version": 1,
        "dataset": "male-cns:v1.0",
        "sensory_boundary": "R1-R6 photoreceptors",
        "column_coordinate_system": "MaleCNS assignedOlHex1/assignedOlHex2",
        "assignment_method": method,
        "columns": columns,
        "counts": {
            "left_columns": counts["L"],
            "right_columns": counts["R"],
            "unique_r1_r6_neurons": len(unique_receptors),
            "r1_r6_per_column_histogram": {
                str(key): value for key, value in sorted(r_counts.items())
            },
        },
        "provenance": {
            "optic_column_coordinates": "observed: official MaleCNS annotations",
            "photoreceptor_body_ids": (
                "observed annotation when R1-R6 carries assignedOlHex coordinates; "
                "otherwise inferred only from observed R1-R6 -> L1 released connectivity"
            ),
            "no_visual_feature_extraction": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"assignment_method={method}")
    print(f"left_columns={counts['L']}")
    print(f"right_columns={counts['R']}")
    print(f"unique_r1_r6_neurons={len(unique_receptors)}")
    print(f"r1_r6_per_column={dict(sorted(r_counts.items()))}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
