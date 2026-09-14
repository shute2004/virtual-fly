#!/usr/bin/env python3
"""Build a fail-closed MaleCNS body-motor map for non-wing effectors.

MaleCNS v1.0 does not require an explicit ``target`` muscle column for identified
leg motor neurons.  The released ``type`` annotation itself carries the curated
muscle-class identity (for example ``Ti flexor MN`` or ``Tr extensor MN``).
This script therefore uses an explicit allow-list of exact released type names,
matching the public MaleCNS locomotor extraction convention, and converts only
those literature-grounded identities into FlyBody joint effects.

Unknown or mechanically ambiguous neurons remain in the output with
``mechanical_status=unresolved`` and are excluded from actuation.  No regex-based
widening, graph-position inference, environment state, reward, target trajectory,
or population-average activity is used to assign an effector.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEG_SUBCLASS_TO_POSITION = {
    "fl": "f",  # front / T1
    "ml": "m",  # middle / T2
    "hl": "h",  # hind / T3
}

# Exact released MaleCNS/MANC type names.  These are intentionally explicit
# rather than fuzzy-matched.  The same leg-MN names are used by the public
# MaleCNS locomotor extraction code and the MANC circuit literature.
LEG_TYPE_EFFECTS: dict[str, tuple[str, str, str]] = {
    "Ti flexor MN": (
        "femur_tibia_pitch",
        "flex",
        "MaleCNS type: tibia flexor motor neuron",
    ),
    "Acc. ti flexor MN": (
        "femur_tibia_pitch",
        "flex",
        "MaleCNS type: accessory tibia flexor motor neuron",
    ),
    "Ti extensor MN": (
        "femur_tibia_pitch",
        "extend",
        "MaleCNS type: tibia extensor motor neuron",
    ),
    "Tr flexor MN": (
        "coxa_femur_pitch",
        "flex",
        "MaleCNS type: trochanter flexor motor neuron",
    ),
    "Acc. tr flexor MN": (
        "coxa_femur_pitch",
        "flex",
        "MaleCNS type: accessory trochanter flexor motor neuron",
    ),
    "Tr extensor MN": (
        "coxa_femur_pitch",
        "extend",
        "MaleCNS type: trochanter extensor motor neuron",
    ),
    "Tergopleural/Pleural promotor MN": (
        "thorax_coxa_yaw",
        "protract",
        "MANC type: tergopleural/pleural promotor rotates the leg forward",
    ),
    "Sternal anterior rotator MN": (
        "thorax_coxa_yaw",
        "protract",
        "MANC type: sternal anterior rotator contributes forward rotation",
    ),
    "Sternal posterior rotator MN": (
        "thorax_coxa_yaw",
        "retract",
        "MANC type: sternal posterior rotator provides opposing posterior rotation",
    ),
}

# hDVM is the identified asynchronous haltere power muscle.  The literature
# commonly labels its motor neuron hDVMn; keep only exact aliases rather than
# interpreting every haltere-MN name as a power muscle.
HALTERE_POWER_TYPES = frozenset({"hDVMn", "hDVM MN"})

REFERENCE_URLS = [
    "https://elifesciences.org/articles/96084",
    "https://elifesciences.org/articles/106446",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10312520/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11338719/",
    "https://github.com/DenisSergeevitch/desktop-fly/blob/master/etl_malecns.py",
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


def resolved_side(row: pd.Series) -> str:
    for column in ("side", "somaSide", "rootSide"):
        value = text(row, column).upper()
        if value in ("L", "R"):
            return value
    return ""


def haltere_effect(row: pd.Series) -> tuple[str, str, str] | None:
    # Prefer the released MaleCNS type, but accept the exact MANC counterpart
    # when the cross-dataset mapping rather than ``type`` carries the identity.
    identities = {text(row, "type"), text(row, "mancType")}
    identities.discard("")
    if identities & HALTERE_POWER_TYPES:
        matched = sorted(identities & HALTERE_POWER_TYPES)[0]
        return (
            "haltere_pitch",
            "power",
            f"identified asynchronous hDVM power motor neuron ({matched})",
        )
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
    leg_types_seen: Counter[str] = Counter()
    matched_leg_types: Counter[str] = Counter()

    for _, row in rows.iterrows():
        subclass = text(row, "subclass").lower()
        if subclass == "wm":
            # Wing MNs are already described by wing-motor-neurons-v0.json.
            continue

        target = text(row, "target")
        type_name = text(row, "type")
        manc_type = text(row, "mancType")
        side = resolved_side(row)
        effect: tuple[str, str, str] | None = None
        leg_prefix = ""
        identity_source = ""

        if subclass in LEG_SUBCLASS_TO_POSITION:
            if type_name:
                leg_types_seen[type_name] += 1
            if side in ("L", "R") and type_name in LEG_TYPE_EFFECTS:
                effect = LEG_TYPE_EFFECTS[type_name]
                matched_leg_types[type_name] += 1
                leg_prefix = ("l" if side == "L" else "r") + LEG_SUBCLASS_TO_POSITION[subclass]
                identity_source = "type"
        elif subclass == "hm":
            effect = haltere_effect(row)
            if effect:
                identity_source = "type_or_mancType"

        record: dict[str, object] = {
            "body_id": int(row["bodyId"]),
            "type": type_name,
            "manc_type": manc_type,
            "subclass": subclass,
            "side": side,
            "target": target,
            "leg_prefix": leg_prefix,
            "mechanical_status": "grounded" if effect else "unresolved",
        }
        for column in (
            "flywireType",
            "instance",
            "nerve",
            "entryNerve",
            "exitNerve",
        ):
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
                    "identity_source": identity_source,
                }
            )
        else:
            record["mechanical_note"] = (
                "preserved in inventory but not actuated: released identity is either "
                "not in the exact grounded allow-list or its peripheral mechanics remain unresolved"
            )
        records.append(record)

    records.sort(
        key=lambda item: (str(item["subclass"]), str(item["type"]), int(item["body_id"]))
    )
    grounded = [row for row in records if row["mechanical_status"] == "grounded"]
    status = Counter(str(row["mechanical_status"]) for row in records)
    subclass_counts = Counter(str(row["subclass"]) for row in records)
    grounded_subclass = Counter(str(row["subclass"]) for row in grounded)

    grounded_leg = [
        row for row in grounded if str(row["subclass"]) in LEG_SUBCLASS_TO_POSITION
    ]
    if not grounded_leg:
        observed = ", ".join(
            f"{name!r}x{count}" for name, count in sorted(leg_types_seen.items())
        )
        raise RuntimeError(
            "no exact released leg MN type matched the conservative mechanical map; "
            f"observed leg types: {observed or '<none>'}"
        )

    payload = {
        "schema_version": 1,
        "dataset": "male-cns:v1.0",
        "purpose": (
            "fail-closed inventory of non-wing MaleCNS motor neurons with a conservative "
            "subset mapped to FlyBody mechanics"
        ),
        "provenance": {
            "neuron_identity": "observed: MaleCNS v1.0 released type/subclass/side annotations",
            "leg_mapping": (
                "exact released MaleCNS type allow-list; names match the public MaleCNS "
                "locomotor extraction and MANC motor-neuron nomenclature"
            ),
            "leg_function": (
                "literature: identified MANC leg MN muscle classes plus FlyBody joint coordinate definitions"
            ),
            "haltere_function": (
                "literature: hDVM is the asynchronous haltere power muscle; steering MN "
                "mechanical transfer remains unresolved"
            ),
            "unknown_policy": "unresolved identities are preserved but never converted into a body command",
            "references": REFERENCE_URLS,
        },
        "counts": {
            "inventory": len(records),
            "grounded": len(grounded),
            "unresolved": status.get("unresolved", 0),
            "by_subclass": dict(sorted(subclass_counts.items())),
            "grounded_by_subclass": dict(sorted(grounded_subclass.items())),
            "matched_leg_types": dict(sorted(matched_leg_types.items())),
        },
        "neurons": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"body_motor_inventory={len(records)}")
    print(f"body_motor_grounded={len(grounded)}")
    print(f"body_motor_unresolved={status.get('unresolved', 0)}")
    print(f"grounded_by_subclass={dict(sorted(grounded_subclass.items()))}")
    print(f"matched_leg_types={dict(sorted(matched_leg_types.items()))}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
