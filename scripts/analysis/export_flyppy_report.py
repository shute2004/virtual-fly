#!/usr/bin/env python3
"""Export a compact, Git-friendly report from the latest Flyppy run.

The large experiment artifacts/checkpoints stay under ignored ``artifacts/``.
This script writes only a small Markdown summary and an episode CSV under
``reports/flyppy/`` so a local run can be committed and reviewed remotely.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path("artifacts/experiments/flyppy-v1/summary.json")
DEFAULT_REPORT_DIR = Path("reports/flyppy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def frontier_rows(episodes: list[dict[str, Any]]) -> list[tuple[float, float, float, int, int]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        if int(row.get("passed_gates", 0)) < 1:
            continue
        key = (float(row["spawn_z_mm"]), float(row["initial_speed_mm_s"]))
        grouped[key].append(row)

    result: list[tuple[float, float, float, int, int]] = []
    for (z_mm, vx_mm_s), rows in sorted(grouped.items()):
        hardest = min(rows, key=lambda item: float(item["spawn_x_mm"]))
        result.append(
            (
                z_mm,
                vx_mm_s,
                float(hardest["spawn_x_mm"]),
                int(hardest["episode"]),
                int(hardest["passed_gates"]),
            )
        )
    return result


def format_condition(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "-"
    try:
        return "x={:.3f}, z={:.3f}, vx={:.3f}".format(
            float(payload["x_mm"]),
            float(payload["z_mm"]),
            float(payload["speed_mm_s"]),
        )
    except (KeyError, TypeError, ValueError):
        return "-"


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        raise SystemExit(f"summary not found: {args.summary}")

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    episodes = list(payload.get("episode_results", []))
    if not episodes:
        raise SystemExit("summary contains no episode_results")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "latest.csv"
    md_path = args.output_dir / "latest.md"

    fields = [
        "episode",
        "curriculum_mode",
        "boundary_ease_level",
        "spawn_x_mm",
        "spawn_z_mm",
        "initial_speed_mm_s",
        "control_steps",
        "passed_gates",
        "collision",
        "collision_reason",
        "finished",
        "max_x_mm",
        "min_z_mm",
        "max_z_mm",
        "final_vx_mm_s",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in episodes:
            writer.writerow({name: row.get(name) for name in fields})

    first_gate_successes = [row for row in episodes if int(row.get("passed_gates", 0)) > 0]
    additional_gate_successes = [row for row in episodes if int(row.get("passed_gates", 0)) > 1]
    finishes = [row for row in episodes if bool(row.get("finished", False))]
    collision_free = [row for row in episodes if not bool(row.get("collision", False))]
    post_first_gate_collisions = [
        row
        for row in first_gate_successes
        if bool(row.get("collision", False))
    ]
    collisions = Counter(
        str(row.get("collision_reason") or "none")
        for row in episodes
        if bool(row.get("collision", False))
    )
    elapsed = float(payload.get("elapsed_seconds", 0.0))
    episode_count = len(episodes)
    curriculum = dict(payload.get("curriculum", {}))
    curriculum_mode = str(payload.get("curriculum_mode") or curriculum.get("curriculum_mode") or "adaptive")
    training_gate_index_raw = curriculum.get("training_gate_index")
    training_gate_index = (
        int(training_gate_index_raw)
        if training_gate_index_raw is not None
        else None
    )
    max_passed = max(int(row.get("passed_gates", 0)) for row in episodes)
    max_x = max(float(row.get("max_x_mm", float("-inf"))) for row in episodes)

    altitude_gains = [
        float(row.get("max_z_mm", row["spawn_z_mm"])) - float(row["spawn_z_mm"])
        for row in episodes
    ]
    altitude_losses = [
        float(row["spawn_z_mm"]) - float(row.get("min_z_mm", row["spawn_z_mm"]))
        for row in episodes
    ]
    episodes_ever_above_spawn = sum(gain > 0.0 for gain in altitude_gains)
    max_altitude_gain = max(altitude_gains)
    worst_altitude_loss = max(altitude_losses)
    mean_max_altitude_gain = sum(altitude_gains) / len(altitude_gains)

    hardest_success = None
    if first_gate_successes:
        hardest_success = min(
            first_gate_successes,
            key=lambda row: float(row["spawn_x_mm"]),
        )

    lines = [
        "# Flyppy latest run",
        "",
        "## 集計",
        "",
        f"- backend: `{payload.get('backend', '-')}`",
        f"- curriculum mode: `{curriculum_mode}`",
        f"- launch mode: `{payload.get('launch_mode', 'async')}`",
    ]
    if curriculum_mode == "adaptive":
        if training_gate_index is None:
            lines.append("- adaptive training criterion: first gate pass (`passed_gates > 0`)")
        else:
            lines.append(f"- adaptive curriculum training gate: {training_gate_index + 1}")
    elif curriculum_mode == "boundary-band":
        lines.append("- boundary-band criterion: batch-level success-rate update")
    elif curriculum_mode == "gate2-height":
        lines.append("- gate2-height criterion: current-frontier success with retention/review bookkeeping")
    lines.extend([
        f"- episode: {payload.get('episode_start', episodes[0].get('episode'))}〜{payload.get('episode_end', episodes[-1].get('episode'))}（{episode_count} episode）",
        f"- elapsed: {elapsed:.3f} s（平均 {elapsed / episode_count:.3f} s/episode）",
        f"- checkpoint neural step: {payload.get('checkpoint_neural_step', '-')}",
        f"- checkpoint neural step semantics: `{payload.get('checkpoint_neural_step_semantics', 'legacy/unspecified')}`",
        f"- first gate通過episode: {len(first_gate_successes)}/{episode_count}（{100.0 * len(first_gate_successes) / episode_count:.1f}%）",
        f"- first gate後にさらに1 gate以上通過: {len(additional_gate_successes)} episode",
        f"- first gate通過後に衝突: {len(post_first_gate_collisions)} episode",
        f"- 無衝突episode: {len(collision_free)} episode",
        f"- stage内course完走: {len(finishes)} episode",
        f"- 1 episode最大通過gate数: {max_passed}",
        f"- run内最大x: {max_x:.3f} mm",
        f"- spawn高度を一度でも上回ったepisode: {episodes_ever_above_spawn}/{episode_count}",
        f"- run内最大高度gain: {max_altitude_gain:+.3f} mm",
        f"- run内最大高度loss: {worst_altitude_loss:.3f} mm",
        f"- episodeごとの最大高度gain平均: {mean_max_altitude_gain:+.3f} mm",
    ])
    if hardest_success is not None:
        lines.append(
            "- 最も小さいspawn xでのfirst gate通過: "
            f"episode {hardest_success['episode']} / x={float(hardest_success['spawn_x_mm']):.3f} / "
            f"z={float(hardest_success['spawn_z_mm']):.3f} / vx={float(hardest_success['initial_speed_mm_s']):.1f} / "
            f"passed={int(hardest_success['passed_gates'])}"
        )

    lines.extend(["", "## 最終カリキュラム状態", ""])
    for key in (
        "training_gate_index",
        "training_phase",
        "curriculum_mode",
        "spawn_x_mm",
        "spawn_z_mm",
        "initial_speed_mm_s",
        "successful_first_gates",
        "consecutive_failures",
        "curriculum_episodes",
        "target_x_mm",
        "target_z_mm",
        "target_speed_mm_s",
        "curriculum_complete",
    ):
        if key in curriculum:
            lines.append(f"- {key}: {fmt(curriculum[key])}")

    boundary = curriculum.get("boundary_band")
    if curriculum_mode == "boundary-band" and isinstance(boundary, dict):
        lines.extend([
            "",
            "## 境界帯カリキュラム",
            "",
            f"- batch_number: {boundary.get('batch_number', '-')}",
            f"- attempts_in_batch: {boundary.get('attempts_in_batch', '-')}",
            f"- successes_in_batch: {boundary.get('successes_in_batch', '-')}",
            f"- last_batch_success_rate: {fmt(boundary.get('last_batch_success_rate'))}",
            f"- last_batch_raw_success_rate: {fmt(boundary.get('last_batch_raw_success_rate'))}",
            f"- last_batch_aggregation: {boundary.get('last_batch_aggregation', '-')}",
            f"- last_batch_group_success_rates: {boundary.get('last_batch_group_success_rates', {})}",
            f"- last_adjustment: {boundary.get('last_adjustment', '-')}",
            f"- harder_shifts: {boundary.get('harder_shifts', '-')}",
            f"- easier_shifts: {boundary.get('easier_shifts', '-')}",
            f"- hard endpoint: {format_condition(boundary.get('hard'))}",
            f"- easy endpoint: {format_condition(boundary.get('easy'))}",
        ])

        by_level: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in episodes:
            level = row.get("boundary_ease_level")
            if level is not None:
                by_level[float(level)].append(row)
        if by_level:
            lines.extend([
                "",
                "| ease level | episode数 | first gate通過 | 通過率 |",
                "|---:|---:|---:|---:|",
            ])
            for level, rows in sorted(by_level.items()):
                level_successes = sum(int(row.get("passed_gates", 0)) > 0 for row in rows)
                lines.append(
                    f"| {level:.2f} | {len(rows)} | {level_successes} | {100.0 * level_successes / len(rows):.1f}% |"
                )

    gate_height = curriculum.get("gate_height_curriculum")
    if curriculum_mode == "gate2-height" and isinstance(gate_height, dict):
        lines.extend([
            "",
            "## Gate2-heightカリキュラム",
            "",
            f"- gate_index: {gate_height.get('gate_index', '-')}",
            f"- frontier_center_z_mm: {fmt(gate_height.get('frontier_center_z_mm'))}",
            f"- target_center_z_mm: {fmt(gate_height.get('target_center_z_mm'))}",
            f"- step_mm: {fmt(gate_height.get('step_mm'))}",
            f"- batch_number: {gate_height.get('batch_number', '-')}",
            f"- role_attempts: {gate_height.get('role_attempts', {})}",
            f"- role_successes: {gate_height.get('role_successes', {})}",
            f"- last_role_success_rates: {gate_height.get('last_role_success_rates', {})}",
            f"- mastered_centers_z_mm: {gate_height.get('mastered_centers_z_mm', [])}",
            f"- last_adjustment: {gate_height.get('last_adjustment', '-')}",
            f"- advances: {gate_height.get('advances', '-')}",
            f"- curriculum_complete: {fmt(gate_height.get('curriculum_complete'))}",
        ])

    lines.extend([
        "",
        "## First-gate成功境界",
        "",
        "同一 `spawn_z / initial_vx` 条件ごとに、first gateを通過した中で最小のspawn xを記録します。",
        "",
        "| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |",
        "|---:|---:|---:|---:|---:|",
    ])
    frontier = frontier_rows(episodes)
    if frontier:
        for z_mm, vx_mm_s, x_mm, episode, passed in frontier:
            lines.append(f"| {z_mm:.3f} | {vx_mm_s:.1f} | {x_mm:.3f} | {episode} | {passed} |")
    else:
        lines.append("| - | - | - | - | - |")

    lines.extend(["", "## Collision", ""])
    if collisions:
        for reason, count in collisions.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    include_level = any(row.get("boundary_ease_level") is not None for row in episodes)
    lines.extend(["", "## Episode一覧", ""])
    if include_level:
        lines.extend([
            "| ep | ease | spawn x | spawn z | vx | steps | passed | collision | min z | max z | max x | final vx |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ])
    else:
        lines.extend([
            "| ep | spawn x | spawn z | vx | steps | passed | collision | min z | max z | max x | final vx |",
            "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ])

    for row in episodes:
        reason = str(row.get("collision_reason") or "-") if row.get("collision") else "-"
        common = {
            "episode": int(row["episode"]),
            "spawn_x": float(row["spawn_x_mm"]),
            "spawn_z": float(row["spawn_z_mm"]),
            "speed": float(row["initial_speed_mm_s"]),
            "steps": int(row["control_steps"]),
            "passed": int(row["passed_gates"]),
            "reason": reason,
            "min_z": float(row["min_z_mm"]),
            "max_z": float(row["max_z_mm"]),
            "max_x": float(row["max_x_mm"]),
            "final_vx": float(row["final_vx_mm_s"]),
        }
        if include_level:
            level = row.get("boundary_ease_level")
            level_text = "-" if level is None else f"{float(level):.2f}"
            lines.append(
                "| {episode} | {level} | {spawn_x:.3f} | {spawn_z:.3f} | {speed:.1f} | {steps} | {passed} | {reason} | {min_z:.3f} | {max_z:.3f} | {max_x:.3f} | {final_vx:.3f} |".format(
                    level=level_text,
                    **common,
                )
            )
        else:
            lines.append(
                "| {episode} | {spawn_x:.3f} | {spawn_z:.3f} | {speed:.1f} | {steps} | {passed} | {reason} | {min_z:.3f} | {max_z:.3f} | {max_x:.3f} | {final_vx:.3f} |".format(
                    **common,
                )
            )

    lines.extend([
        "",
        "## 判定用メモ",
        "",
        f"- curriculum modeは `{curriculum_mode}` です。別modeの判定規則をこのrunへ流用しません。",
        "- `training_gate_index` はadaptive系stateに存在する場合のみ表示します。0始まりです。",
        "- `boundary_ease_level` はboundary-band由来の履歴フィールドで、gate2-heightのcurrent/review判定には使用しません。",
        "- `latest.csv` がepisode単位の機械可読データです。",
        "- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。",
        "- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"flyppy_report_md={md_path}")
    print(f"flyppy_report_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
