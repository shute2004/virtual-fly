#!/usr/bin/env python3
"""Build a fail-closed MaleCNS body-motor map for non-wing effectors.

MANC supplies an exact ``target`` muscle annotation where a motor neuron has been
identified.  This script converts only a small set of literature-supported
muscle functions into physical FlyBody joint effects.  It is a peripheral
anatomical map, not an action decoder: no environment state, reward, target
trajectory or population-average neural activity enters the mapping.

Unknown or mechanically ambiguous targets remain in the output with
``mechanical_status=unresolved`` and are excluded from actuation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


LEG_SUBCLASS_TO_POSITION = {
    "fl": "f",  # front / T1
    "ml": "m",  # middle / T2
    "hl": "h",  # hind / T3
}

REFERENCE_URLS = [
    "https://elifesciences.org/articles/96084",
    "https://elifesciences.org/articles/106446",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10312520/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11338719/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns-v1.0/body-motor-neurons-v0.json"),
    )
    return parser.parse_args()


def text(row: pd.Series, name: str) -> str:
    if name not in row or pd.isna(row[name]):
        return ""
    return str(row[name]).strip()


def norm(value: str) -> str:
    value = value.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def resolved_side(row: pd.Series) -> str:
    for column in ("side", "somaSide", "rootSide"):
        value = text(row, column).upper()
        if value in ("L", "R"):
            return value
    return ""


def leg_effect(target: str) -> tuple[str, str, str] | None:
    """Return (joint_role, action, evidence) for high-confidence leg muscles."""

    t = norm(target)
    # FlyBody's source joint classes call positive pitch ``extend_tibia`` and
    # ``extend_femur``.  These antagonist assignments therefore have an explicit
    # model-coordinate interpretation without guessing a world-space sign.
    if ("tibia extensor" in t or "ti extensor" in t) and "flexor" not in t:
        return "femur_tibia_pitch", "extend", "identified antagonist: tibia extensor"
    if "tibia flexor" in t or "ti flexor" in t:
        return "femur_tibia_pitch", "flex", "identified antagonist: tibia flexor"
    if "trochanter extensor" in t:
        return "coxa_femur_pitch", "extend", "identified antagonist: trochanter extensor"
    if "trochanter flexor" in t or "sternotrochanter" in t:
        return "coxa_femur_pitch", "flex", "identified antagonist: trochanter flexor"

    # These thoracic muscles have explicit forward/backward functions in the
    # MANC circuit paper. Runtime geometry determines the local joint sign, so
    # side-specific coordinate conventions are not guessed here.
    if (
        "sternal anterior rotator" in t
        or "tergopleural promotor" in t
        or "tergoplural promotor" in t
    ):
        return "thorax_coxa_yaw", "protract", "literature: rotates/protracts leg forward"
    if "sternal posterior rotator" in t:
        return "thorax_coxa_yaw", "retract", "literature: opposing posterior rotation during stance"

    # Pleural remotor/abductor has two mechanical components; femur reductor,
    # long-tendon and distal tarsal muscles are not collapsed onto one FlyBody
    # DOF without a moment-arm model.
    return None


def haltere_effect(target: str, type_name: str) -> tuple[str, str, str] | None:
    t = norm(f"{target} {type_name}")
    if "hdvm" in t or "haltere dorsal ventral" in t:
        return "haltere_pitch", "power", "identified asynchronous hDVM power muscle"
    # Haltere steering MN identities are useful and remain in the inventory, but
    # published hDVM+hI1 co-activation does not isolate an hI1-only mechanical
    # transfer function.  Do not manufacture one here.
    return None


def main() -> int:
    args = parse_args()
    annotations = pd.read_feather(args.snapshot / "annotations.feather")
    body_ids = np.fromfile(args.snapshot / "body_ids.u64le", dtype="<u8")
    if len(annotations) != len(body_ids):
        raise RuntimeError("annotations/body_ids length mismatch")
    if not np.array_equal(annotations["bodyId"].astype(np.uint64).to_numpy(), body_ids):
        raise RuntimeError("annotations are not aligned with body_ids.u64le")

    superclass = (
        annotations.get("superclass", pd.Series("", index=annotations.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    rows = annotations.loc[superclass.eq("vnc_motor")].copy()
    if rows.empty:
        raise RuntimeError("no superclass=vnc_motor neurons found")

    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        subclass = text(row, "subclass").lower()
        if subclass == "wm":
            # Wing MNs are already described by wing-motor-neurons-v0.json.
            continue
        target = text(row, "target")
        type_name = text(row, "type")
        side = resolved_side(row)
        effect: tuple[str, str, str] | None = None
        leg_prefix = ""

        if subclass in LEG_SUBCLASS_TO_POSITION and side in ("L", "R"):
            effect = leg_effect(target)
            leg_prefix = ("l" if side == "L" else "r") + LEG_SUBCLASS_TO_POSITION[subclass]
        elif subclass == "hm":
            effect = haltere_effect(target, type_name)

        record: dict[str, object] = {
            "body_id": int(row["bodyId"]),
            "type": type_name,
            "subclass": subclass,
            "side": side,
            "target": target,
            "leg_prefix": leg_prefix,
            "mechanical_status": "grounded" if effect else "unresolved",
        }
        for column in ("flywireType", "instance", "nerve", "entryNerve", "exitNerve"):
            value = text(row, column)
            if value:
                record[column] = value
        if effect:
            joint_role, action, evidence = effect
            record.update(
                {
                    "joint_role": joint_role,
                    "mechanical_action": action,
                    "mechanical_evidence": evidence,
                }
            )
        else:
            record["mechanical_note"] = (
                "preserved in inventory but not actuated: exact peripheral mechanics are not grounded enough"
            )
        records.append(record)

    records.sort(
        key=lambda item: (str(item["subclass"]), str(item["type"]), int(item["body_id"]))
    )
    grounded = [row for row in records if row["mechanical_status"] == "grounded"]
    status = Counter(str(row["mechanical_status"]) for row in records)
    subclass_counts = Counter(str(row["subclass"]) for row in records)
    grounded_subclass = Counter(str(row["subclass"]) for row in grounded)

    if not any(str(row["subclass"]) in LEG_SUBCLASS_TO_POSITION for row in grounded):
        raise RuntimeError(
            "no leg MN target names matched the conservative mechanical map; inspect released target annotations before widening rules"
        )

    payload = {
        "schema_version": 1,
        "dataset": "male-cns:v1.0",
        "purpose": (
            "fail-closed inventory of non-wing MaleCNS motor neurons with a conservative subset mapped to FlyBody mechanics"
        ),
        "provenance": {
            "neuron_and_target_identity": "observed: MaleCNS v1.0 annotations",
            "leg_function": "literature: MANC leg MN target/function studies plus FlyBody joint coordinate definitions",
            "haltere_function": (
                "literature: hDVM is the asynchronous haltere power muscle; steering MN mechanical transfer remains unresolved"
            ),
            "unknown_policy": "unresolved targets are preserved but never converted into a body command",
            "references": REFERENCE_URLS,
        },
        "counts": {
            "inventory": len(records),
            "grounded": len(grounded),
            "unresolved": status.get("unresolved", 0),
            "by_subclass": dict(sorted(subclass_counts.items())),
            "grounded_by_subclass": dict(sorted(grounded_subclass.items())),
        },
        "neurons": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"body_motor_inventory={len(records)}")
    print(f"body_motor_grounded={len(grounded)}")
    print(f"body_motor_unresolved={status.get('unresolved', 0)}")
    print(f"grounded_by_subclass={dict(sorted(grounded_subclass.items()))}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
