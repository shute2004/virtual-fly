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

    successes = [row for row in episodes if int(row.get("passed_gates", 0)) > 0]
    second_gate = [row for row in episodes if int(row.get("passed_gates", 0)) > 1]
    finishes = [row for row in episodes if bool(row.get("finished", False))]
    collisions = Counter(
        str(row.get("collision_reason") or "none")
        for row in episodes
        if bool(row.get("collision", False))
    )
    elapsed = float(payload.get("elapsed_seconds", 0.0))
    episode_count = len(episodes)
    curriculum = dict(payload.get("curriculum", {}))
    training_gate_index = int(curriculum.get("training_gate_index", 0))
    max_passed = max(int(row.get("passed_gates", 0)) for row in episodes)
    max_x = max(float(row.get("max_x_mm", float("-inf"))) for row in episodes)

    hardest_success = None
    if successes:
        hardest_success = min(successes, key=lambda row: float(row["spawn_x_mm"]))

    lines = [
        "# Flyppy latest run",
        "",
        "## 集計",
        "",
        f"- backend: `{payload.get('backend', '-')}`",
        f"- curriculum target gate: {training_gate_index + 1}",
        f"- episode: {payload.get('episode_start', episodes[0].get('episode'))}〜{payload.get('episode_end', episodes[-1].get('episode'))}（{episode_count} episode）",
        f"- elapsed: {elapsed:.3f} s（平均 {elapsed / episode_count:.3f} s/episode）",
        f"- checkpoint neural step: {payload.get('checkpoint_neural_step', '-')}",
        f"- target gate成功episode: {len(successes)}/{episode_count}（{100.0 * len(successes) / episode_count:.1f}%）",
        f"- target gate後にさらに1 gate以上通過: {len(second_gate)} episode",
        f"- stage内course完走: {len(finishes)} episode",
        f"- 1 episode最大通過gate数: {max_passed}",
        f"- run内最大x: {max_x:.3f} mm",
    ]
    if hardest_success is not None:
        lines.append(
            "- 最も小さいspawn xでの成功: "
            f"episode {hardest_success['episode']} / x={float(hardest_success['spawn_x_mm']):.3f} / "
            f"z={float(hardest_success['spawn_z_mm']):.3f} / vx={float(hardest_success['initial_speed_mm_s']):.1f} / "
            f"passed={int(hardest_success['passed_gates'])}"
        )

    lines.extend(["", "## 最終カリキュラム状態", ""])
    for key in (
        "training_gate_index",
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

    lines.extend(["", "## 成功境界", "", "同一 `spawn_z / initial_vx` 条件ごとに、成功した中で最小のspawn xを記録します。", "", "| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |", "|---:|---:|---:|---:|---:|"])
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

    lines.extend([
        "",
        "## Episode一覧",
        "",
        "| ep | spawn x | spawn z | vx | steps | passed | collision | max x | final vx |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    for row in episodes:
        reason = str(row.get("collision_reason") or "-") if row.get("collision") else "-"
        lines.append(
            "| {episode} | {spawn_x:.3f} | {spawn_z:.3f} | {speed:.1f} | {steps} | {passed} | {reason} | {max_x:.3f} | {final_vx:.3f} |".format(
                episode=int(row["episode"]),
                spawn_x=float(row["spawn_x_mm"]),
                spawn_z=float(row["spawn_z_mm"]),
                speed=float(row["initial_speed_mm_s"]),
                steps=int(row["control_steps"]),
                passed=int(row["passed_gates"]),
                reason=reason,
                max_x=float(row["max_x_mm"]),
                final_vx=float(row["final_vx_mm_s"]),
            )
        )

    lines.extend([
        "",
        "## 判定用メモ",
        "",
        "- `training_gate_index` は0始まりです。0=第1gate、1=第2gateです。",
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
