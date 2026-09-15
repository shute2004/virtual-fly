#!/usr/bin/env python3
"""Inspect local Flyppy v3 population-training artifacts without mutating training state.

This diagnostic is intentionally read-only with respect to ``artifacts/``. It
writes one compact Markdown report under ``reports/flyppy/`` so the result can
be committed and reviewed remotely.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = Path("artifacts/experiments/flyppy-v3")
DEFAULT_REPORT = Path("reports/flyppy/population_state_diagnostic.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tail", type=int, default=5)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"__error__": f"expected object, got {type(payload).__name__}"}
    return payload


def inspect_jsonl(path: Path, tail: int) -> tuple[int | None, list[dict[str, Any] | str]]:
    if not path.exists():
        return None, []
    recent: deque[dict[str, Any] | str] = deque(maxlen=max(1, tail))
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                count += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    recent.append(line[:1000])
                    continue
                recent.append(item if isinstance(item, dict) else repr(item))
    except Exception as exc:
        return None, [{"__error__": f"{type(exc).__name__}: {exc}"}]
    return count, list(recent)


def pick(payload: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    if payload is None:
        return {}
    return {key: payload[key] for key in keys if key in payload}


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def append_json_block(lines: list[str], payload: Any) -> None:
    lines.extend(["```json", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])


def main() -> int:
    args = parse_args()
    if args.tail < 1:
        raise SystemExit("--tail must be >= 1")

    experiment = absolute(args.experiment)
    report = absolute(args.report)

    paths = {
        "summary": experiment / "summary.json",
        "curriculum": experiment / "curriculum-state.json",
        "population_state": experiment / "population-state.json",
        "commit_log": experiment / "commit-log.jsonl",
        "trajectory": experiment / "trajectory.jsonl",
        "checkpoint_manifest": experiment / "checkpoint" / "manifest.json",
    }

    summary = read_json(paths["summary"])
    curriculum = read_json(paths["curriculum"])
    population_state = read_json(paths["population_state"])
    checkpoint = read_json(paths["checkpoint_manifest"])
    commit_count, commit_tail = inspect_jsonl(paths["commit_log"], args.tail)
    trajectory_count, trajectory_tail = inspect_jsonl(paths["trajectory"], args.tail)

    latest_commit = next((item for item in reversed(commit_tail) if isinstance(item, dict)), None)
    latest_trajectory = next((item for item in reversed(trajectory_tail) if isinstance(item, dict)), None)

    summary_keys = (
        "schema_version",
        "experiment",
        "backend",
        "population",
        "shared_weight",
        "episode_start",
        "episode_end",
        "episodes_this_run",
        "aggregate_control_steps",
        "aggregate_control_steps_per_second",
        "checkpoint_neural_step",
        "global_weight_version_start",
        "global_weight_version_end",
        "mean_version_staleness",
        "max_version_staleness",
        "elapsed_seconds",
    )
    curriculum_keys = (
        "schema_version",
        "curriculum_mode",
        "curriculum_episodes",
        "successful_first_gates",
        "consecutive_failures",
        "spawn_x_mm",
        "spawn_z_mm",
        "initial_speed_mm_s",
        "curriculum_complete",
    )
    population_keys = (
        "schema_version",
        "population",
        "global_weight_version",
        "commit_semantics",
        "weight_averaging",
    )
    checkpoint_keys = ("schema_version", "dataset", "neuron_count", "edge_count", "step")

    summary_view = pick(summary, summary_keys)
    curriculum_view = pick(curriculum, curriculum_keys)
    population_view = pick(population_state, population_keys)
    checkpoint_view = pick(checkpoint, checkpoint_keys)

    summary_is_population = summary_view.get("experiment") == "flyppy_v3_async_shared_weight_population"
    summary_end = as_int(summary_view.get("episode_end"))
    summary_checkpoint_step = as_int(summary_view.get("checkpoint_neural_step"))
    summary_global_end = as_int(summary_view.get("global_weight_version_end"))
    checkpoint_step = as_int(checkpoint_view.get("step"))
    population_global = as_int(population_view.get("global_weight_version"))
    latest_commit_episode = as_int(latest_commit.get("episode")) if latest_commit else None
    latest_commit_version = as_int(latest_commit.get("commit_weight_version")) if latest_commit else None
    latest_trajectory_episode = as_int(latest_trajectory.get("episode")) if latest_trajectory else None

    evidence: list[str] = []
    if summary is None:
        evidence.append("summary.json is missing")
    elif not summary_is_population:
        evidence.append("summary.json is not a shared-weight population summary")
    if checkpoint_step is not None and summary_checkpoint_step is not None and checkpoint_step > summary_checkpoint_step:
        evidence.append(
            f"checkpoint is ahead of summary: neural step {checkpoint_step} > {summary_checkpoint_step}"
        )
    if latest_commit_episode is not None and summary_end is not None and latest_commit_episode > summary_end:
        evidence.append(
            f"commit log is ahead of summary: episode {latest_commit_episode} > {summary_end}"
        )
    if latest_trajectory_episode is not None and summary_end is not None and latest_trajectory_episode > summary_end:
        evidence.append(
            f"trajectory is ahead of summary: episode {latest_trajectory_episode} > {summary_end}"
        )
    if population_global is not None and summary_global_end is not None and population_global > summary_global_end:
        evidence.append(
            f"population state is ahead of summary: global weight v{population_global} > v{summary_global_end}"
        )
    if latest_commit_version is not None and summary_global_end is not None and latest_commit_version > summary_global_end:
        evidence.append(
            f"commit log is ahead of summary: global weight v{latest_commit_version} > v{summary_global_end}"
        )
    if population_global is not None and not summary_is_population:
        evidence.append(f"population-state.json exists at global weight v{population_global} while summary is non-population")
    if latest_commit is not None and not summary_is_population:
        evidence.append("population commit-log.jsonl contains commit records while summary is non-population")

    file_rows: list[str] = []
    for label, path in paths.items():
        if path.exists():
            file_rows.append(
                f"| {label} | yes | {path.stat().st_size} | {iso_mtime(path)} | `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}` |"
            )
        else:
            file_rows.append(f"| {label} | no | - | - | `{path}` |")

    lines = [
        "# Flyppy population state diagnostic",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- experiment: `{experiment}`",
        "- diagnostic mutates training artifacts: false",
        "",
        "## Files",
        "",
        "| artifact | exists | bytes | mtime UTC | path |",
        "|---|---|---:|---|---|",
        *file_rows,
        "",
        "## Summary",
        "",
    ]
    append_json_block(lines, summary_view if summary is not None else {"missing": True})

    lines.extend(["## Curriculum state", ""])
    append_json_block(lines, curriculum_view if curriculum is not None else {"missing": True})

    lines.extend(["## Population state", ""])
    append_json_block(lines, population_view if population_state is not None else {"missing": True})

    lines.extend(["## Checkpoint manifest", ""])
    append_json_block(lines, checkpoint_view if checkpoint is not None else {"missing": True})

    lines.extend([
        "## Commit log",
        "",
        f"- non-empty records: {commit_count if commit_count is not None else '-'}",
        f"- latest episode: {latest_commit_episode if latest_commit_episode is not None else '-'}",
        f"- latest commit weight version: {latest_commit_version if latest_commit_version is not None else '-'}",
        "",
    ])
    append_json_block(lines, commit_tail)

    lines.extend([
        "## Trajectory",
        "",
        f"- non-empty records: {trajectory_count if trajectory_count is not None else '-'}",
        f"- latest episode: {latest_trajectory_episode if latest_trajectory_episode is not None else '-'}",
        "",
    ])
    append_json_block(lines, trajectory_tail)

    lines.extend(["## Consistency findings", ""])
    if evidence:
        lines.extend(f"- {item}" for item in evidence)
    else:
        lines.append("- no obvious artifact ordering mismatch detected")
    lines.append("")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"diagnostic_report={report}")
    print(f"summary_is_population={str(summary_is_population).lower()}")
    print(f"checkpoint_step={checkpoint_step if checkpoint_step is not None else '-'}")
    print(f"latest_commit_episode={latest_commit_episode if latest_commit_episode is not None else '-'}")
    print(f"latest_commit_weight_version={latest_commit_version if latest_commit_version is not None else '-'}")
    print(f"latest_trajectory_episode={latest_trajectory_episode if latest_trajectory_episode is not None else '-'}")
    print(f"consistency_findings={len(evidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
