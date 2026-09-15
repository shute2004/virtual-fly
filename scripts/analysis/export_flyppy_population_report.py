#!/usr/bin/env python3
"""Export a compact report for asynchronous shared-weight Flyppy training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v3/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/flyppy/population_latest.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    if not payload.get("shared_weight", False):
        raise SystemExit("summary is not a shared-weight population run")

    episodes = sorted(
        list(payload.get("episode_results", [])),
        key=lambda item: int(item.get("episode", -1)),
    )
    elapsed = float(payload.get("elapsed_seconds", 0.0))
    aggregate_steps = int(
        payload.get(
            "aggregate_control_steps",
            sum(int(item.get("control_steps", 0)) for item in episodes),
        )
    )
    throughput = (
        float(payload.get("aggregate_control_steps_per_second", aggregate_steps / elapsed))
        if elapsed > 0.0
        else 0.0
    )
    successes = sum(int(item.get("passed_gates", 0)) > 0 for item in episodes)
    max_passed = max((int(item.get("passed_gates", 0)) for item in episodes), default=0)

    vision_runtime = str(payload.get("vision_runtime", "unknown"))
    vision_rays = payload.get("vision_rays_per_ommatidium")
    vision_framebuffer = payload.get("vision_framebuffer")
    body_runtime = str(payload.get("body_runtime", "unknown"))
    body_processes = payload.get("body_processes")

    lines = [
        "# Flyppy shared-weight population latest run",
        "",
        f"- backend: `{payload.get('backend', '-')}`",
        f"- population: {payload.get('population', '-')}",
        f"- body runtime: `{body_runtime}`",
        f"- body processes: {body_processes if body_processes is not None else '-'}",
        f"- vision runtime: `{vision_runtime}`",
        f"- rays/ommatidium: {vision_rays if vision_rays is not None else '-'}",
        f"- RGB framebuffer: {str(bool(vision_framebuffer)).lower() if vision_framebuffer is not None else '-'}",
        f"- shared weight: {str(bool(payload.get('shared_weight'))).lower()}",
        f"- weight averaging: {str(bool(payload.get('weight_averaging'))).lower()}",
        f"- episodes: {len(episodes)}",
        f"- elapsed: {elapsed:.3f} s",
        f"- aggregate control steps: {aggregate_steps}",
        f"- aggregate control steps/s: {throughput:.3f}",
        f"- simulated biological seconds: {float(payload.get('simulated_seconds_this_run', 0.0)):.6f}",
        f"- first gate pass: {successes}/{len(episodes) if episodes else 0}",
        f"- max passed gates: {max_passed}",
        f"- global weight version: {payload.get('global_weight_version_start', '-')} -> {payload.get('global_weight_version_end', '-')}",
        f"- mean version staleness: {float(payload.get('mean_version_staleness', 0.0)):.3f}",
        f"- max version staleness: {int(payload.get('max_version_staleness', 0))}",
        "- commit semantics: episode-local additive+clamp transaction rebased onto latest global weight",
        "",
        "| ep | slot | source v | commit from | commit v | stale | steps | passed | collision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in episodes:
        lines.append(
            "| {episode} | {slot} | {source} | {from_v} | {commit_v} | {stale} | {steps} | {passed} | {collision} |".format(
                episode=int(item.get("episode", -1)),
                slot=int(item.get("slot", -1)),
                source=int(item.get("source_weight_version", -1)),
                from_v=int(item.get("commit_from_version", -1)),
                commit_v=int(item.get("commit_weight_version", -1)),
                stale=int(item.get("version_staleness", -1)),
                steps=int(item.get("control_steps", 0)),
                passed=int(item.get("passed_gates", 0)),
                collision=str(bool(item.get("collision", False))).lower(),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"flyppy_population_report={args.output}")
    print(f"vision_runtime={vision_runtime}")
    print(f"vision_rays_per_ommatidium={vision_rays}")
    print(f"aggregate_control_steps_per_second={throughput:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
