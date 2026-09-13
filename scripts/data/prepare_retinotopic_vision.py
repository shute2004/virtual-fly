#!/usr/bin/env python3
"""Build the MaleCNS R1-R6 retinotopic stimulation map from observed wiring.

The spatial unit is the lamina cartridge represented by an L1 neuron carrying the
official MaleCNS optic-lobe coordinates ``assignedOlHex1`` / ``assignedOlHex2``.
The photoreceptors assigned to that optical column are the actual annotated R1-R6
neurons with released presynaptic connections into that L1 neuron.

This deliberately follows neural superposition rather than assigning R1-R6 to
columns by row order or by an assumed one-ommatidium grouping. Six R1-R6 cells from
neighboring ommatidia that share an optical axis converge on the same cartridge.
If enough observed columns cannot be resolved, this script fails instead of
inventing a geometric correspondence or averaging visual information.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATION_FILENAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
ANNOTATION_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
    f"flat-connectome/{ANNOTATION_FILENAME}"
)
# MaleCNS/FlyWire annotations encountered in the ecosystem use forms such as
# R1-R6, R1-6, and occasionally underscore-separated variants.
R1_R6_PATTERN = re.compile(
    r"(?:^|[^A-Za-z0-9])R1(?:-|–|_)R?6(?:$|[^A-Za-z0-9])", re.I
)


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


def searchable_types(frame: pd.DataFrame) -> pd.Series:
    fields = [text_series(frame, name) for name in ("type", "flywireType", "instance")]
    return fields[0] + " | " + fields[1] + " | " + fields[2]


def r1_r6_mask(frame: pd.DataFrame) -> np.ndarray:
    return searchable_types(frame).map(
        lambda value: bool(R1_R6_PATTERN.search(value))
    ).to_numpy()


def l1_mask(frame: pd.DataFrame) -> np.ndarray:
    return searchable_types(frame).str.contains(
        r"(?:^|\W)L1(?:$|\W)", case=False, regex=True
    ).to_numpy()


def side_values(frame: pd.DataFrame) -> np.ndarray:
    # Some optic-lobe neurons have no soma-side label. Prefer the direct `side`
    # field, then somaSide, then rootSide; all are observed MaleCNS metadata.
    result = np.full(len(frame), "", dtype=object)
    for name in ("side", "somaSide", "rootSide"):
        values = text_series(frame, name).str.upper().str.strip().to_numpy(dtype=str)
        take = (result == "") & np.isin(values, ["L", "R"])
        result[take] = values[take]
    return np.asarray(result, dtype=str)


def numeric_coordinate(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def connectivity_columns(
    annotations: pd.DataFrame,
    body_ids: np.ndarray,
    row_offsets: np.ndarray,
    pre_indices: np.ndarray,
) -> list[dict[str, object]]:
    aligned = (
        annotations.drop_duplicates("bodyId")
        .set_index("bodyId")
        .reindex(body_ids)
        .reset_index()
    )
    r_mask = r1_r6_mask(aligned)
    l_mask = l1_mask(aligned)
    sides = side_values(aligned)
    h1 = numeric_coordinate(aligned["assignedOlHex1"])
    h2 = numeric_coordinate(aligned["assignedOlHex2"])

    columns: list[dict[str, object]] = []
    for post_idx in np.flatnonzero(l_mask):
        side = sides[post_idx]
        if (
            side not in ("L", "R")
            or not np.isfinite(h1[post_idx])
            or not np.isfinite(h2[post_idx])
        ):
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
                "assignment": "observed_R1-R6_to_L1_connectivity",
            }
        )
    return columns


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
        raise RuntimeError(f"duplicate L1 optic-column coordinates: {clashes[:8]}")

    body_to_columns: dict[int, list[tuple[str, int, int]]] = {}
    for item in columns:
        key = (str(item["side"]), int(item["hex1"]), int(item["hex2"]))
        for body_id in item["r1_r6_body_ids"]:
            body_to_columns.setdefault(int(body_id), []).append(key)
    multi = {
        body_id: keys
        for body_id, keys in body_to_columns.items()
        if len(set(keys)) > 1
    }
    if multi:
        sample = list(multi.items())[:8]
        raise RuntimeError(
            "R1-R6 body IDs resolved to multiple optical columns; refusing an "
            f"ambiguous sensory map: {sample}"
        )


def main() -> int:
    args = parse_args()
    ensure_annotations(args.raw_annotations, args.download)

    body_ids = np.fromfile(args.snapshot / "body_ids.u64le", dtype="<u8")
    row_offsets = np.fromfile(args.snapshot / "row_offsets.u32le", dtype="<u4")
    pre_indices = np.fromfile(args.snapshot / "pre_indices.u32le", dtype="<u4")
    if len(row_offsets) != len(body_ids) + 1:
        raise RuntimeError("snapshot row_offsets length is inconsistent with body IDs")
    if int(row_offsets[-1]) != len(pre_indices):
        raise RuntimeError("snapshot CSR edge count is inconsistent with pre_indices")

    annotations = pd.read_feather(args.raw_annotations)
    required = {"bodyId", "type", "assignedOlHex1", "assignedOlHex2"}
    missing = required - set(annotations.columns)
    if missing:
        raise RuntimeError(
            f"official annotation file is missing required retinotopy fields: {sorted(missing)}"
        )

    columns = connectivity_columns(annotations, body_ids, row_offsets, pre_indices)
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
    method = "observed_R1-R6_to_L1_connectivity_with_observed_L1_hex"

    output = {
        "schema_version": 2,
        "dataset": "male-cns:v1.0",
        "sensory_boundary": "R1-R6 photoreceptors",
        "column_coordinate_system": "L1 MaleCNS assignedOlHex1/assignedOlHex2",
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
            "optic_column_coordinates": (
                "observed: official MaleCNS assignedOlHex coordinates on L1"
            ),
            "photoreceptor_body_ids": (
                "observed/inferred boundary: annotated R1-R6 neurons selected only from "
                "released presynaptic connections into each observed L1 cartridge"
            ),
            "neural_superposition_rule": "literature",
            "no_visual_feature_extraction": True,
            "no_spatial_averaging": True,
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
