#!/usr/bin/env python3
"""Quantify completion-order and staleness bias in Flyppy async shared-weight training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMIT_LOG = Path("artifacts/experiments/flyppy-v3/commit-log.jsonl")
DEFAULT_JSON = Path("reports/flyppy/diagnostics/async_commit_bias.json")
DEFAULT_REPORT = Path("reports/flyppy/diagnostics/async_commit_bias.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-log", type=Path, default=DEFAULT_COMMIT_LOG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-commit-version", type=int)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: int(row["commit_seq"]))
    return rows


def resolve_analysis_watermark(commit_log: Path, explicit: int | None) -> int | None:
    if explicit is not None:
        if explicit < 0:
            raise ValueError("max commit version must be >= 0")
        return explicit
    population_state = commit_log.parent / "population-state.json"
    if not population_state.exists():
        return None
    payload = json.loads(population_state.read_text(encoding="utf-8"))
    value = int(payload.get("global_weight_version", 0))
    if value < 0:
        raise ValueError("population-state global_weight_version must be >= 0")
    return value


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    x = list(xs)
    y = list(ys)
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = mean(x)
    my = mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / denom


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "reward_episodes": 0,
            "reward_events": 0,
            "aversive_events": 0,
            "mean_control_steps": 0.0,
            "mean_staleness": 0.0,
            "max_staleness": 0,
            "mean_commit_rank": 0.0,
        }
    return {
        "episodes": len(rows),
        "reward_episodes": sum(int(row.get("reward_events", 0)) > 0 for row in rows),
        "reward_events": sum(int(row.get("reward_events", 0)) for row in rows),
        "aversive_events": sum(int(row.get("aversive_events", 0)) for row in rows),
        "mean_control_steps": mean(int(row["control_steps"]) for row in rows),
        "mean_staleness": mean(int(row["staleness"]) for row in rows),
        "max_staleness": max(int(row["staleness"]) for row in rows),
        "mean_commit_rank": mean(float(row["window_commit_rank"]) for row in rows),
    }


def annotate_boundary_batches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a stable analysis batch key to old and new commit-log rows.

    New logs persist ``boundary_batch_number`` directly. Historical logs only
    store attempt IDs, but a new boundary batch cannot launch until the previous
    one has finished. Sorting by episode therefore lets us detect a new batch at
    the first repeated attempt ID without relying on completion order.
    """

    annotated = [dict(row) for row in rows]
    legacy_rows = [
        row
        for row in annotated
        if row.get("boundary_attempt_index") is not None
        and row.get("boundary_batch_number") is None
    ]
    seen_attempts: set[int] = set()
    legacy_batch = 0
    for row in sorted(legacy_rows, key=lambda item: int(item["episode"])):
        attempt = int(row["boundary_attempt_index"])
        if attempt in seen_attempts:
            legacy_batch += 1
            seen_attempts.clear()
        seen_attempts.add(attempt)
        row["_analysis_boundary_batch_key"] = f"legacy:{legacy_batch}"
    for row in annotated:
        if row.get("boundary_batch_number") is not None:
            row["_analysis_boundary_batch_key"] = f"stored:{int(row['boundary_batch_number'])}"
    return annotated


def source_version_round_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure source-CNS generation spread for comparable population rounds.

    Boundary-band runs are compared by the experimental condition round:
    ``boundary_attempt_index // population`` within one batch. This is the
    meaningful async-vs-wave unit because it contains one attempt from each
    course-seed slot at the same ease-level round. ``launch_round`` is only a
    fallback for curricula without boundary attempt identities.
    """

    empty = {
        "complete_rounds": 0,
        "mean_source_version_spread": None,
        "max_source_version_spread": None,
        "zero_spread_rounds": 0,
    }
    if not rows:
        return dict(empty)
    slots = sorted({int(row["slot"]) for row in rows})
    if not slots or slots != list(range(max(slots) + 1)):
        return dict(empty)
    population = len(slots)

    boundary_rows = [row for row in rows if row.get("boundary_attempt_index") is not None]
    if boundary_rows:
        if any("_analysis_boundary_batch_key" not in row for row in boundary_rows):
            rows = annotate_boundary_batches(rows)
        candidates = [
            row
            for row in rows
            if row.get("boundary_attempt_index") is not None
            and row.get("_analysis_boundary_batch_key") is not None
        ]
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in candidates:
            attempt = int(row["boundary_attempt_index"])
            batch_key = str(row["_analysis_boundary_batch_key"])
            grouped.setdefault((batch_key, attempt // population), []).append(row)
    else:
        launch_round_candidates = [row for row in rows if row.get("launch_round") is not None]
        if not launch_round_candidates:
            return dict(empty)
        grouped = {}
        for row in launch_round_candidates:
            grouped.setdefault(("launch", int(row["launch_round"])), []).append(row)

    spreads: list[int] = []
    for group_rows in grouped.values():
        if len(group_rows) != population:
            continue
        if {int(row["slot"]) for row in group_rows} != set(range(population)):
            continue
        source_versions = [int(row["source_weight_version"]) for row in group_rows]
        spreads.append(max(source_versions) - min(source_versions))

    return {
        "complete_rounds": len(spreads),
        "mean_source_version_spread": mean(spreads) if spreads else None,
        "max_source_version_spread": max(spreads) if spreads else None,
        "zero_spread_rounds": sum(spread == 0 for spread in spreads),
    }


def window_payload(name: str, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in source_rows]
    count = len(rows)
    for rank, row in enumerate(rows):
        row["window_commit_rank"] = rank / max(1, count - 1)
    reward_rows = [row for row in rows if int(row.get("reward_events", 0)) > 0]
    no_reward_rows = [row for row in rows if int(row.get("reward_events", 0)) == 0]
    slots = sorted({int(row["slot"]) for row in rows})
    return {
        "name": name,
        "episode_start": min((int(row["episode"]) for row in rows), default=None),
        "episode_end": max((int(row["episode"]) for row in rows), default=None),
        "all": summarize(rows),
        "reward": summarize(reward_rows),
        "no_reward": summarize(no_reward_rows),
        "correlation_control_steps_staleness": pearson(
            (float(row["control_steps"]) for row in rows),
            (float(row["staleness"]) for row in rows),
        ),
        "source_version_rounds": source_version_round_summary(rows),
        "slot_summaries": {
            str(slot): summarize([row for row in rows if int(row["slot"]) == slot])
            for slot in slots
        },
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Flyppy async shared-weight completion-order bias",
        "",
        f"- commit log: `{payload['commit_log']}`",
        f"- records: {payload['records']}",
        f"- episode coverage: {payload['episode_start']}..{payload['episode_end']}",
        f"- persisted max commit version: {payload.get('analysis_max_commit_version') if payload.get('analysis_max_commit_version') is not None else '-'}",
        f"- unpersisted commit rows ignored: {payload.get('unpersisted_commits_ignored', 0)}",
        "",
        "## Reward episode と no-reward episode",
        "",
        "| window | class | episodes | mean steps | mean staleness | max staleness | mean commit rank | reward events | aversive events |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in payload["windows"]:
        for key, label in (("reward", "PAMあり"), ("no_reward", "PAMなし")):
            item = window[key]
            lines.append(
                "| {window} | {label} | {episodes} | {steps:.2f} | {stale:.3f} | {max_stale} | {rank:.3f} | {reward} | {aversive} |".format(
                    window=window["name"],
                    label=label,
                    episodes=item["episodes"],
                    steps=float(item["mean_control_steps"]),
                    stale=float(item["mean_staleness"]),
                    max_stale=item["max_staleness"],
                    rank=float(item["mean_commit_rank"]),
                    reward=item["reward_events"],
                    aversive=item["aversive_events"],
                )
            )

    lines.extend(
        [
            "",
            "## Episode長とstaleness",
            "",
            "| window | Pearson r(control steps, staleness) |",
            "|---|---:|",
        ]
    )
    for window in payload["windows"]:
        correlation = window["correlation_control_steps_staleness"]
        text = "n/a" if correlation is None else f"{float(correlation):.4f}"
        lines.append(f"| {window['name']} | {text} |")

    lines.extend(
        [
            "",
            "## 同一attempt roundのsource weight version幅",
            "",
            "| window | complete rounds | mean spread | max spread | zero-spread rounds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for window in payload["windows"]:
        cohort = window["source_version_rounds"]
        mean_spread = cohort["mean_source_version_spread"]
        max_spread = cohort["max_source_version_spread"]
        lines.append(
            "| {name} | {rounds} | {mean_spread} | {max_spread} | {zero} |".format(
                name=window["name"],
                rounds=cohort["complete_rounds"],
                mean_spread="n/a" if mean_spread is None else f"{float(mean_spread):.3f}",
                max_spread="n/a" if max_spread is None else str(int(max_spread)),
                zero=cohort["zero_spread_rounds"],
            )
        )

    latest = payload["windows"][-1]
    lines.extend(
        [
            "",
            f"## 最新window ({latest['name']}) のslot別",
            "",
            "| slot | episodes | reward episodes | reward rate | mean steps | mean staleness | max staleness |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for slot, item in sorted(latest["slot_summaries"].items(), key=lambda pair: int(pair[0])):
        rate = item["reward_episodes"] / item["episodes"] if item["episodes"] else 0.0
        lines.append(
            "| {slot} | {episodes} | {reward} | {rate:.1%} | {steps:.2f} | {stale:.3f} | {max_stale} |".format(
                slot=slot,
                episodes=item["episodes"],
                reward=item["reward_episodes"],
                rate=rate,
                steps=float(item["mean_control_steps"]),
                stale=float(item["mean_staleness"]),
                max_stale=item["max_staleness"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "shared-weight transactionのrebaseはstale episodeの局所更新を捨てないが、そのepisode中のsensory/neural/body trajectoryはsource weight version上で生成される。",
            "したがって長いepisodeほどstalenessが大きい場合、成功経験と失敗経験が異なるweight世代を経験するschedule biasが残る。",
            "boundary-bandの固定slot割当はcourse seedごとのepisode数を揃えるが、async modeでは同一attempt roundのslotが異なるsource weight versionから出発し得る。",
            "wave modeの直接の狙いはcommit時stalenessをゼロにすることではなく、同一attempt roundのsource weight version幅を0にしてtrajectory生成世代を揃えることである。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    commit_log = absolute(args.commit_log)
    output_json = absolute(args.output_json)
    report = absolute(args.report)
    raw_rows = load_jsonl(commit_log)
    try:
        analysis_max_commit_version = resolve_analysis_watermark(
            commit_log,
            args.max_commit_version,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    rows = raw_rows
    if analysis_max_commit_version is not None:
        rows = [
            row
            for row in raw_rows
            if int(row["commit_weight_version"]) <= analysis_max_commit_version
        ]
    if not rows:
        raise SystemExit("commit log is empty after filtering")
    rows = annotate_boundary_batches(rows)

    windows = [
        window_payload("all", rows),
        window_payload("last48", rows[-48:]),
        window_payload("last24", rows[-24:]),
    ]
    payload = {
        "schema_version": 3,
        "commit_log": portable(commit_log),
        "records": len(rows),
        "analysis_max_commit_version": analysis_max_commit_version,
        "unpersisted_commits_ignored": len(raw_rows) - len(rows),
        "episode_start": min(int(row["episode"]) for row in rows),
        "episode_end": max(int(row["episode"]) for row in rows),
        "windows": windows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(payload), encoding="utf-8")

    for window in windows:
        reward = window["reward"]
        no_reward = window["no_reward"]
        correlation = window["correlation_control_steps_staleness"]
        print(
            "{} reward_staleness={:.3f} no_reward_staleness={:.3f} reward_steps={:.2f} no_reward_steps={:.2f} corr={}".format(
                window["name"],
                float(reward["mean_staleness"]),
                float(no_reward["mean_staleness"]),
                float(reward["mean_control_steps"]),
                float(no_reward["mean_control_steps"]),
                "n/a" if correlation is None else f"{float(correlation):.4f}",
            )
        )
    print(f"json={output_json}")
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
