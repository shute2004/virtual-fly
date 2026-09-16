#!/usr/bin/env python3
"""Compare two Flyppy shared-weight scheduler runs from the same parent state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from analyze_flyppy_async_bias import annotate_boundary_batches, load_jsonl, source_version_round_summary


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = Path("reports/flyppy/experiments/scheduler-ab/comparison.json")
DEFAULT_REPORT = Path("reports/flyppy/experiments/scheduler-ab/comparison.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-summary", type=Path, required=True)
    parser.add_argument("--right-summary", type=Path, required=True)
    parser.add_argument("--left-fixed", type=Path)
    parser.add_argument("--right-fixed", type=Path)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row.get(key, 0.0)) for row in rows) if rows else 0.0


def _scheduler_rows(summary_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    commit_log = summary_path.parent / "commit-log.jsonl"
    if not commit_log.exists():
        configured = payload.get("commit_log")
        if configured:
            candidate = absolute(Path(str(configured)))
            if candidate.exists():
                commit_log = candidate
    rows = annotate_boundary_batches(load_jsonl(commit_log))
    start = int(payload["episode_start"])
    end = int(payload["episode_end"])
    return [row for row in rows if start <= int(row["episode"]) <= end]


def summarize_run(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    episodes = [dict(row) for row in payload.get("episode_results", [])]
    if not episodes:
        raise ValueError(f"summary has no episode_results: {summary_path}")
    scheduler_rows = _scheduler_rows(summary_path, payload)
    reward_rows = [row for row in scheduler_rows if int(row.get("reward_events", 0)) > 0]
    no_reward_rows = [row for row in scheduler_rows if int(row.get("reward_events", 0)) == 0]
    first_gate = sum(int(row.get("passed_gates", 0)) >= 1 for row in episodes)
    second_gate = sum(int(row.get("passed_gates", 0)) >= 2 for row in episodes)
    altitude_gains = [
        float(row.get("max_z_mm", 0.0)) - float(row.get("spawn_z_mm", 0.0))
        for row in episodes
    ]
    staleness = [int(row.get("version_staleness", 0)) for row in episodes]
    round_summary = source_version_round_summary(scheduler_rows)
    elapsed = float(payload.get("elapsed_seconds", 0.0))
    aggregate_steps = int(payload.get("aggregate_control_steps", 0))
    control_dt = float(payload.get("control_dt_seconds", 0.0))
    return {
        "summary": portable(summary_path),
        "launch_mode": str(payload.get("launch_mode", "unknown")),
        "episode_start": int(payload["episode_start"]),
        "episode_end": int(payload["episode_end"]),
        "episodes": len(episodes),
        "first_gate_successes": first_gate,
        "first_gate_rate": first_gate / len(episodes),
        "second_gate_successes": second_gate,
        "second_gate_rate": second_gate / len(episodes),
        "total_passed_gates": sum(int(row.get("passed_gates", 0)) for row in episodes),
        "mean_passed_gates": mean(int(row.get("passed_gates", 0)) for row in episodes),
        "collision_rate": sum(bool(row.get("collision", False)) for row in episodes) / len(episodes),
        "mean_control_steps": _mean(episodes, "control_steps"),
        "mean_altitude_gain_mm": mean(altitude_gains),
        "max_altitude_gain_mm": max(altitude_gains),
        "elapsed_seconds": elapsed,
        "aggregate_control_steps_per_second": (
            float(payload.get("aggregate_control_steps_per_second"))
            if payload.get("aggregate_control_steps_per_second") is not None
            else (aggregate_steps / elapsed if elapsed > 0.0 else 0.0)
        ),
        "simulation_realtime_factor": (
            aggregate_steps * control_dt / elapsed
            if elapsed > 0.0 and control_dt > 0.0
            else 0.0
        ),
        "mean_version_staleness": mean(staleness) if staleness else 0.0,
        "max_version_staleness": max(staleness, default=0),
        "reward_mean_staleness": (
            mean(int(row["staleness"]) for row in reward_rows) if reward_rows else 0.0
        ),
        "no_reward_mean_staleness": (
            mean(int(row["staleness"]) for row in no_reward_rows) if no_reward_rows else 0.0
        ),
        "source_version_rounds": round_summary,
    }


def validate_pair(left: dict[str, Any], right: dict[str, Any]) -> None:
    comparable = (
        "population",
        "curriculum_mode",
        "vision_runtime",
        "vision_rays_per_ommatidium",
        "episode_start",
        "episode_end",
    )
    for key in comparable:
        if left.get(key) != right.get(key):
            raise ValueError(
                f"scheduler comparison requires matching {key}: {left.get(key)!r} != {right.get(key)!r}"
            )


def fixed_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json(path)
    conditions = []
    for row in payload.get("condition_summaries", []):
        condition = row.get("condition", {})
        conditions.append(
            {
                "condition": str(condition.get("name", "unknown")),
                "first_gate_passes": int(row.get("first_gate_passes", 0)),
                "second_gate_passes": int(row.get("second_gate_passes", 0)),
                "mean_passed_gates": float(row.get("mean_passed_gates", 0.0)),
                "collision_rate": float(row.get("collision_rate", 0.0)),
            }
        )
    return {
        "path": portable(path),
        "checkpoint_neural_step": int(payload.get("checkpoint_neural_step", -1)),
        "conditions": conditions,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    left = payload["left"]
    right = payload["right"]
    lines = [
        "# Flyppy scheduler A/B comparison",
        "",
        f"- left: `{left['launch_mode']}` / episode {left['episode_start']}..{left['episode_end']}",
        f"- right: `{right['launch_mode']}` / episode {right['episode_start']}..{right['episode_end']}",
        "",
        "| metric | left | right |",
        "|---|---:|---:|",
        f"| first gate | {left['first_gate_successes']}/{left['episodes']} ({left['first_gate_rate']:.1%}) | {right['first_gate_successes']}/{right['episodes']} ({right['first_gate_rate']:.1%}) |",
        f"| second gate | {left['second_gate_successes']}/{left['episodes']} ({left['second_gate_rate']:.1%}) | {right['second_gate_successes']}/{right['episodes']} ({right['second_gate_rate']:.1%}) |",
        f"| total passed gates | {left['total_passed_gates']} | {right['total_passed_gates']} |",
        f"| mean passed gates | {left['mean_passed_gates']:.3f} | {right['mean_passed_gates']:.3f} |",
        f"| collision rate | {left['collision_rate']:.1%} | {right['collision_rate']:.1%} |",
        f"| mean altitude gain mm | {left['mean_altitude_gain_mm']:+.4f} | {right['mean_altitude_gain_mm']:+.4f} |",
        f"| control steps/s | {left['aggregate_control_steps_per_second']:.3f} | {right['aggregate_control_steps_per_second']:.3f} |",
        f"| realtime factor | {left['simulation_realtime_factor']:.5f} | {right['simulation_realtime_factor']:.5f} |",
        f"| mean commit staleness | {left['mean_version_staleness']:.3f} | {right['mean_version_staleness']:.3f} |",
        f"| reward mean staleness | {left['reward_mean_staleness']:.3f} | {right['reward_mean_staleness']:.3f} |",
        f"| no-reward mean staleness | {left['no_reward_mean_staleness']:.3f} | {right['no_reward_mean_staleness']:.3f} |",
    ]
    for side_name, item in (("left", left), ("right", right)):
        rounds = item["source_version_rounds"]
        lines.extend(
            [
                "",
                f"## {side_name}: {item['launch_mode']}",
                "",
                f"- complete attempt rounds: {rounds['complete_rounds']}",
                f"- mean source-version spread: {rounds['mean_source_version_spread']}",
                f"- max source-version spread: {rounds['max_source_version_spread']}",
                f"- zero-spread rounds: {rounds['zero_spread_rounds']}",
            ]
        )

    left_fixed = payload.get("left_fixed")
    right_fixed = payload.get("right_fixed")
    if left_fixed is not None and right_fixed is not None:
        left_by_name = {row["condition"]: row for row in left_fixed["conditions"]}
        right_by_name = {row["condition"]: row for row in right_fixed["conditions"]}
        names = [name for name in left_by_name if name in right_by_name]
        lines.extend(
            [
                "",
                "## Frozen fixed evaluation",
                "",
                "| condition | left first | right first | left second | right second | left mean gates | right mean gates |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in names:
            a = left_by_name[name]
            b = right_by_name[name]
            lines.append(
                f"| {name} | {a['first_gate_passes']} | {b['first_gate_passes']} | {a['second_gate_passes']} | {b['second_gate_passes']} | {a['mean_passed_gates']:.3f} | {b['mean_passed_gates']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    left_path = absolute(args.left_summary)
    right_path = absolute(args.right_summary)
    left_payload = load_json(left_path)
    right_payload = load_json(right_path)
    validate_pair(left_payload, right_payload)
    payload = {
        "schema_version": 1,
        "left": summarize_run(left_path, left_payload),
        "right": summarize_run(right_path, right_payload),
        "left_fixed": fixed_summary(absolute(args.left_fixed) if args.left_fixed else None),
        "right_fixed": fixed_summary(absolute(args.right_fixed) if args.right_fixed else None),
    }
    output_json = absolute(args.output_json)
    report = absolute(args.report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown(payload), encoding="utf-8")
    print(f"left_mode={payload['left']['launch_mode']}")
    print(f"right_mode={payload['right']['launch_mode']}")
    print(f"json={output_json}")
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
