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
MOTOR_OUTPUT_METRICS: tuple[str, ...] = (
    "mean_wing_spikes_per_step",
    "mean_somatic_spikes_per_step",
    "mean_active_wing_motor_units",
    "mean_active_somatic_motor_units",
    "mean_power_activation",
    "mean_power_lr_abs_diff",
    "mean_active_steering_channels",
    "mean_abs_leg_drive",
)
MOTOR_OUTPUT_ROW_KEYS = {
    "mean_wing_spikes_per_step": "wing_spikes_per_step",
    "mean_somatic_spikes_per_step": "somatic_spikes_per_step",
}


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


def _mean_optional(rows: list[Mapping[str, object]], key: str) -> float:
    return mean(float(row.get(key, 0.0)) for row in rows) if rows else 0.0


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
        "mean_wing_spikes_per_step": _mean_optional(items, "wing_spikes_per_step"),
        "mean_somatic_spikes_per_step": _mean_optional(items, "somatic_spikes_per_step"),
        "mean_active_wing_motor_units": _mean_optional(items, "mean_active_wing_motor_units"),
        "mean_active_somatic_motor_units": _mean_optional(items, "mean_active_somatic_motor_units"),
        "mean_power_activation": _mean_optional(items, "mean_power_activation"),
        "mean_power_lr_abs_diff": _mean_optional(items, "mean_power_lr_abs_diff"),
        "mean_active_steering_channels": _mean_optional(items, "mean_active_steering_channels"),
        "mean_abs_leg_drive": _mean_optional(items, "mean_abs_leg_drive"),
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


def _motor_value(row: Mapping[str, object], key: str) -> float:
    if key in row:
        return float(row[key])
    row_key = MOTOR_OUTPUT_ROW_KEYS.get(key)
    return float(row.get(row_key, 0.0)) if row_key is not None else 0.0


def _motor_pair(baseline: Mapping[str, object], trained: Mapping[str, object]) -> dict[str, object]:
    metrics: dict[str, dict[str, float | None]] = {}
    for key in MOTOR_OUTPUT_METRICS:
        before = _motor_value(baseline, key)
        after = _motor_value(trained, key)
        metrics[key] = {
            "baseline": before,
            "trained": after,
            "delta": after - before,
            "relative_delta": None if abs(before) <= 1e-12 else (after - before) / abs(before),
        }
    return metrics


def compare_motor_outputs(
    baseline_payload: Mapping[str, object],
    trained_payload: Mapping[str, object],
) -> dict[str, object]:
    """Compare two fixed-suite evaluations under identical frozen conditions."""

    for key in ("suite_version", "population", "seed_start", "seed_end", "vision_runtime", "vision_rays_per_ommatidium"):
        if baseline_payload.get(key) != trained_payload.get(key):
            raise ValueError(
                f"fixed-evaluation comparison requires matching {key}: "
                f"{baseline_payload.get(key)!r} != {trained_payload.get(key)!r}"
            )

    baseline_rows = {
        (str(row["condition"]), int(row["course_seed"])): row
        for row in list(baseline_payload["episode_results"])
    }
    trained_rows = {
        (str(row["condition"]), int(row["course_seed"])): row
        for row in list(trained_payload["episode_results"])
    }
    if baseline_rows.keys() != trained_rows.keys():
        raise ValueError("fixed-evaluation comparison requires identical condition/seed pairs")

    baseline_summaries = {
        str(item["condition"]["name"]): item
        for item in list(baseline_payload["condition_summaries"])
    }
    trained_summaries = {
        str(item["condition"]["name"]): item
        for item in list(trained_payload["condition_summaries"])
    }
    if baseline_summaries.keys() != trained_summaries.keys():
        raise ValueError("fixed-evaluation comparison requires identical condition summaries")

    condition_comparison: list[dict[str, object]] = []
    for condition in FIXED_EVAL_SUITE_V1:
        before = baseline_summaries[condition.name]
        after = trained_summaries[condition.name]
        condition_comparison.append(
            {
                "condition": condition.name,
                "baseline_first_gate_passes": int(before["first_gate_passes"]),
                "trained_first_gate_passes": int(after["first_gate_passes"]),
                "baseline_second_gate_passes": int(before["second_gate_passes"]),
                "trained_second_gate_passes": int(after["second_gate_passes"]),
                "motor": _motor_pair(before, after),
            }
        )

    paired_seed_comparison: list[dict[str, object]] = []
    for condition in FIXED_EVAL_SUITE_V1:
        seeds = sorted(seed for name, seed in baseline_rows if name == condition.name)
        for seed in seeds:
            before = baseline_rows[(condition.name, seed)]
            after = trained_rows[(condition.name, seed)]
            paired_seed_comparison.append(
                {
                    "condition": condition.name,
                    "course_seed": seed,
                    "baseline_passed_gates": int(before["passed_gates"]),
                    "trained_passed_gates": int(after["passed_gates"]),
                    "baseline_control_steps": int(before["control_steps"]),
                    "trained_control_steps": int(after["control_steps"]),
                    "motor": _motor_pair(before, after),
                }
            )

    return {
        "schema_version": 1,
        "suite_version": baseline_payload["suite_version"],
        "baseline_subject": baseline_payload.get("evaluation_subject", "baseline"),
        "baseline_checkpoint": baseline_payload["checkpoint"],
        "baseline_checkpoint_neural_step": baseline_payload["checkpoint_neural_step"],
        "trained_subject": trained_payload.get("evaluation_subject", "trained_checkpoint"),
        "trained_checkpoint": trained_payload["checkpoint"],
        "trained_checkpoint_neural_step": trained_payload["checkpoint_neural_step"],
        "trained_training_episode_end": trained_payload.get("training_episode_end"),
        "population": baseline_payload["population"],
        "seed_start": baseline_payload["seed_start"],
        "seed_end": baseline_payload["seed_end"],
        "vision_runtime": baseline_payload["vision_runtime"],
        "vision_rays_per_ommatidium": baseline_payload["vision_rays_per_ommatidium"],
        "condition_comparison": condition_comparison,
        "paired_seed_comparison": paired_seed_comparison,
    }


def _format_relative(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.1%}"


def render_motor_comparison_markdown(payload: Mapping[str, object]) -> str:
    conditions = list(payload["condition_comparison"])
    pairs = list(payload["paired_seed_comparison"])
    lines = [
        "# Flyppy固定評価: motor出力比較",
        "",
        f"- suite: `{payload['suite_version']}`",
        f"- baseline: `{payload['baseline_subject']}` / step {payload['baseline_checkpoint_neural_step']}",
        f"- trained: `{payload['trained_subject']}` / episode {payload.get('trained_training_episode_end')} / step {payload['trained_checkpoint_neural_step']}",
        f"- population / seeds: {payload['population']} / {payload['seed_start']}..{payload['seed_end']}",
        f"- vision: `{payload['vision_runtime']}` / {payload['vision_rays_per_ommatidium']} rays/ommatidium",
        "",
        "## 条件集計",
        "",
        "| condition | first gate | second gate | wing spikes/step | somatic spikes/step | steering channels | L/R power diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in conditions:
        motor = row["motor"]
        wing = motor["mean_wing_spikes_per_step"]
        somatic = motor["mean_somatic_spikes_per_step"]
        steering = motor["mean_active_steering_channels"]
        lr = motor["mean_power_lr_abs_diff"]
        lines.append(
            "| {condition} | {first0}→{first1} | {second0}→{second1} | "
            "{wing0:.4f}→{wing1:.4f} ({wing_pct}) | {som0:.4f}→{som1:.4f} ({som_pct}) | "
            "{steer0:.3f}→{steer1:.3f} ({steer_pct}) | {lr0:.4f}→{lr1:.4f} ({lr_pct}) |".format(
                condition=row["condition"],
                first0=row["baseline_first_gate_passes"],
                first1=row["trained_first_gate_passes"],
                second0=row["baseline_second_gate_passes"],
                second1=row["trained_second_gate_passes"],
                wing0=float(wing["baseline"]),
                wing1=float(wing["trained"]),
                wing_pct=_format_relative(wing["relative_delta"]),
                som0=float(somatic["baseline"]),
                som1=float(somatic["trained"]),
                som_pct=_format_relative(somatic["relative_delta"]),
                steer0=float(steering["baseline"]),
                steer1=float(steering["trained"]),
                steer_pct=_format_relative(steering["relative_delta"]),
                lr0=float(lr["baseline"]),
                lr1=float(lr["trained"]),
                lr_pct=_format_relative(lr["relative_delta"]),
            )
        )

    lines.extend(
        [
            "",
            "## condition × course seed",
            "",
            "| condition | seed | gates | steps | wing spikes/step | somatic spikes/step | steering channels | L/R power diff |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairs:
        motor = row["motor"]
        wing = motor["mean_wing_spikes_per_step"]
        somatic = motor["mean_somatic_spikes_per_step"]
        steering = motor["mean_active_steering_channels"]
        lr = motor["mean_power_lr_abs_diff"]
        lines.append(
            "| {condition} | {seed} | {g0}→{g1} | {s0}→{s1} | "
            "{w0:.3f}→{w1:.3f} ({wp}) | {m0:.3f}→{m1:.3f} ({mp}) | "
            "{st0:.3f}→{st1:.3f} ({stp}) | {lr0:.4f}→{lr1:.4f} ({lrp}) |".format(
                condition=row["condition"],
                seed=row["course_seed"],
                g0=row["baseline_passed_gates"],
                g1=row["trained_passed_gates"],
                s0=row["baseline_control_steps"],
                s1=row["trained_control_steps"],
                w0=float(wing["baseline"]),
                w1=float(wing["trained"]),
                wp=_format_relative(wing["relative_delta"]),
                m0=float(somatic["baseline"]),
                m1=float(somatic["trained"]),
                mp=_format_relative(somatic["relative_delta"]),
                st0=float(steering["baseline"]),
                st1=float(steering["trained"]),
                stp=_format_relative(steering["relative_delta"]),
                lr0=float(lr["baseline"]),
                lr1=float(lr["trained"]),
                lrp=_format_relative(lr["relative_delta"]),
            )
        )

    lines.extend(
        [
            "",
            "## 読み方",
            "",
            "同一condition・course seedでも、学習済みCNSではmotor出力が変わるため閉ループ軌跡とepisode長も変わり得る。",
            "したがってこの表は『同一の外界時系列へ対するopen-loop応答差』ではなく、固定初期条件から始めた閉ループmotor出力差を示す。",
            "motor出力が変化していれば、上流の可塑変化が固定motor synapseを介してactuated motor neuronへ伝播していることと整合する。",
            "",
        ]
    )
    return "\n".join(lines)


def render_markdown(payload: Mapping[str, object]) -> str:
    conditions = list(payload["condition_summaries"])
    rows = list(payload["episode_results"])
    lines = [
        "# Flyppy v3 fixed learning evaluation",
        "",
        f"- suite: `{payload['suite_version']}`",
        f"- subject: `{payload.get('evaluation_subject', 'trained_checkpoint')}`",
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
            "## Motor output",
            "",
            "| condition | wing spikes/step | somatic spikes/step | active wing units | active somatic units | power activation | L/R power diff | steering channels | abs leg drive |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in conditions:
        lines.append(
            "| {name} | {wing:.4f} | {somatic:.4f} | {active_wing:.3f} | {active_somatic:.3f} | "
            "{power:.4f} | {power_diff:.4f} | {steering:.3f} | {leg:.4f} |".format(
                name=item["condition"]["name"],
                wing=float(item.get("mean_wing_spikes_per_step", 0.0)),
                somatic=float(item.get("mean_somatic_spikes_per_step", 0.0)),
                active_wing=float(item.get("mean_active_wing_motor_units", 0.0)),
                active_somatic=float(item.get("mean_active_somatic_motor_units", 0.0)),
                power=float(item.get("mean_power_activation", 0.0)),
                power_diff=float(item.get("mean_power_lr_abs_diff", 0.0)),
                steering=float(item.get("mean_active_steering_channels", 0.0)),
                leg=float(item.get("mean_abs_leg_drive", 0.0)),
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
