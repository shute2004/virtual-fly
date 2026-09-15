"""Fixed, read-only Flyppy evaluation definitions and summaries.

The evaluation suite is intentionally frozen so checkpoints from different
training times can be compared under the same body/course initial conditions.
This module contains no neural stepping or plasticity logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable, Mapping


SUITE_VERSION = "flyppy-v3-fixed-v1"


@dataclass(frozen=True)
class FixedEvalCondition:
    name: str
    x_mm: float
    z_mm: float
    speed_mm_s: float
    provenance: str


FIXED_EVAL_SUITE_V1: tuple[FixedEvalCondition, ...] = (
    FixedEvalCondition(
        name="part3_frontier",
        x_mm=3.267,
        z_mm=11.467,
        speed_mm_s=450.0,
        provenance="Part3終了時点 episode 160 のadaptive curriculum近傍",
    ),
    FixedEvalCondition(
        name="midpoint",
        x_mm=1.6335,
        z_mm=10.1885,
        speed_mm_s=375.0,
        provenance="part3_frontier と最終targetの各成分中点",
    ),
    FixedEvalCondition(
        name="target",
        x_mm=0.0,
        z_mm=8.91,
        speed_mm_s=300.0,
        provenance="Flyppy v3 curriculum target",
    ),
)


def suite_payload() -> dict[str, object]:
    return {
        "suite_version": SUITE_VERSION,
        "conditions": [asdict(condition) for condition in FIXED_EVAL_SUITE_V1],
    }


def _mean(rows: list[Mapping[str, object]], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def summarize_condition(
    condition: FixedEvalCondition,
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    items = list(rows)
    total = len(items)
    if total == 0:
        raise ValueError(f"condition {condition.name!r} has no evaluation rows")

    first_gate = sum(int(row["passed_gates"]) >= 1 for row in items)
    second_gate = sum(int(row["passed_gates"]) >= 2 for row in items)
    collision = sum(bool(row["collision"]) for row in items)
    finished = sum(bool(row["finished"]) for row in items)
    return {
        "condition": asdict(condition),
        "episodes": total,
        "first_gate_passes": first_gate,
        "first_gate_pass_rate": first_gate / total,
        "second_gate_passes": second_gate,
        "second_gate_pass_rate": second_gate / total,
        "collisions": collision,
        "collision_rate": collision / total,
        "finished": finished,
        "finished_rate": finished / total,
        "mean_passed_gates": _mean(items, "passed_gates"),
        "max_passed_gates": max(int(row["passed_gates"]) for row in items),
        "mean_control_steps": _mean(items, "control_steps"),
        "mean_max_altitude_gain_mm": _mean(items, "max_altitude_gain_mm"),
        "mean_max_altitude_loss_mm": _mean(items, "max_altitude_loss_mm"),
        "mean_final_vx_mm_s": _mean(items, "final_vx_mm_s"),
    }


def summarize_suite(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    items = list(rows)
    by_name: dict[str, list[Mapping[str, object]]] = {
        condition.name: [] for condition in FIXED_EVAL_SUITE_V1
    }
    for row in items:
        name = str(row["condition"])
        if name not in by_name:
            raise ValueError(f"unknown fixed-evaluation condition {name!r}")
        by_name[name].append(row)
    return [
        summarize_condition(condition, by_name[condition.name])
        for condition in FIXED_EVAL_SUITE_V1
    ]


def render_markdown(payload: Mapping[str, object]) -> str:
    conditions = list(payload["condition_summaries"])
    rows = list(payload["episode_results"])
    lines = [
        "# Flyppy v3 fixed learning evaluation",
        "",
        f"- suite: `{payload['suite_version']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- checkpoint neural step: {payload['checkpoint_neural_step']}",
        f"- training episode end: {payload.get('training_episode_end')}",
        f"- global weight version: {payload['global_weight_version']}",
        f"- backend: `{payload['backend']}`",
        f"- population / course seeds: {payload['population']} / {payload['seed_start']}..{payload['seed_end']}",
        f"- vision: `{payload['vision_runtime']}` / {payload['vision_rays_per_ommatidium']} rays/ommatidium",
        f"- plasticity: `{str(payload['plasticity']).lower()}`",
        f"- global weight version unchanged: `{str(payload['global_weight_version_unchanged']).lower()}`",
        f"- transaction dirty edges after evaluation: {payload['transaction_dirty_edges_after']}",
        f"- elapsed: {float(payload['elapsed_seconds']):.3f} s",
        "",
        "## 固定条件別",
        "",
        "| condition | x mm | z mm | vx mm/s | first gate | second gate | collision | mean gates | max gates | mean altitude gain mm | mean altitude loss mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in conditions:
        condition = item["condition"]
        lines.append(
            "| {name} | {x:.4f} | {z:.4f} | {speed:.1f} | {first}/{episodes} ({first_rate:.1%}) | "
            "{second}/{episodes} ({second_rate:.1%}) | {collisions}/{episodes} ({collision_rate:.1%}) | "
            "{mean_gates:.3f} | {max_gates} | {gain:.3f} | {loss:.3f} |".format(
                name=condition["name"],
                x=float(condition["x_mm"]),
                z=float(condition["z_mm"]),
                speed=float(condition["speed_mm_s"]),
                first=int(item["first_gate_passes"]),
                second=int(item["second_gate_passes"]),
                episodes=int(item["episodes"]),
                first_rate=float(item["first_gate_pass_rate"]),
                second_rate=float(item["second_gate_pass_rate"]),
                collisions=int(item["collisions"]),
                collision_rate=float(item["collision_rate"]),
                mean_gates=float(item["mean_passed_gates"]),
                max_gates=int(item["max_passed_gates"]),
                gain=float(item["mean_max_altitude_gain_mm"]),
                loss=float(item["mean_max_altitude_loss_mm"]),
            )
        )

    lines.extend(
        [
            "",
            "## Episode一覧",
            "",
            "| condition | seed | slot | steps | passed | collision | finished | max x mm | min z mm | max z mm | final vx mm/s |",
            "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {condition} | {seed} | {slot} | {steps} | {passed} | {collision} | {finished} | "
            "{max_x:.3f} | {min_z:.3f} | {max_z:.3f} | {final_vx:.3f} |".format(
                condition=row["condition"],
                seed=int(row["course_seed"]),
                slot=int(row["slot"]),
                steps=int(row["control_steps"]),
                passed=int(row["passed_gates"]),
                collision=(row["collision_reason"] if row["collision"] else "-") or "true",
                finished=str(bool(row["finished"])).lower(),
                max_x=float(row["max_x_mm"]),
                min_z=float(row["min_z_mm"]),
                max_z=float(row["max_z_mm"]),
                final_vx=float(row["final_vx_mm_s"]),
            )
        )

    lines.extend(
        [
            "",
            "## 解釈上の契約",
            "",
            "この評価はcheckpointを読み取り専用でロードし、全neural stepを `plasticity=false` で実行する。",
            "gate pass時のPAM刺激とcollision時のPPL刺激は通常taskと同じく与えるが、weight更新は行わない。",
            "各条件・各seedの開始前にslot neural stateをrestartし、body/periphery/retinal adaptationもresetする。",
            "curriculum state、trajectory、checkpointは書き込まず、global weight versionも更新しない。",
            "",
        ]
    )
    return "\n".join(lines)
