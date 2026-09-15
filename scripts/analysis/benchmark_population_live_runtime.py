#!/usr/bin/env python3
"""Benchmark dense-P versus eager-live Flyppy population runtimes.

This benchmark never mutates the production Flyppy experiment. Each case starts
from its own copy of the same production checkpoint/curriculum. The curriculum
step sizes are effectively frozen so runtime comparisons do not get confounded
by different reset conditions.

Cases by default:
  dense-P reference: N=1,2
  eager-live runtime: N=1,2,4,8

The dense-P reference uses the diagnostic frontier bridge, which has the same
outgoing propagation frontier but retains the previous dense-P plasticity
kernel. The eager-live runtime uses the normal production-facing population
bridge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_BENCH_ROOT = Path("artifacts/profiles/flyppy-live-runtime-benchmark")
DEFAULT_REPORT = Path("reports/flyppy/population_live_benchmark.md")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


@dataclass
class CaseResult:
    mode: str
    population: int
    returncode: int
    wall_seconds: float
    output_dir: Path
    summary: dict | None
    checkpoint_digest: str | None
    stdout_tail: str
    stderr_tail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-control-steps", type=int, default=32)
    parser.add_argument("--live-populations", default="1,2,4,8")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def parse_populations(text: str) -> list[int]:
    values = sorted({int(token.strip()) for token in text.split(",") if token.strip()})
    if not values or values[0] < 1 or values[-1] > 32:
        raise SystemExit("live populations must be integers in 1..32")
    return values


def tail(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return "[truncated to final output]\n" + text[-limit:]


def prepare_case(source: Path, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "checkpoint", output / "checkpoint")
    shutil.copy2(source / "curriculum-state.json", output / "curriculum-state.json")


def run_case(
    *,
    mode: str,
    population: int,
    source: Path,
    bench_root: Path,
    snapshot: Path,
    episodes: int,
    max_control_steps: int,
    base_env: dict[str, str],
) -> CaseResult:
    output = bench_root / f"{mode}-p{population}"
    prepare_case(source, output)
    env = dict(base_env)
    env["VF_POPULATION_BRIDGE_BIN"] = (
        "population_frontier_profile_bridge" if mode == "dense" else "population_neural_bridge"
    )

    command = [
        "uv", "run", "python", "scripts/embodiment/train_flyppy_population.py",
        "--episodes", str(episodes),
        "--population", str(population),
        "--snapshot", str(snapshot),
        "--groups", str(snapshot / "embodiment-groups-v0.json"),
        "--retinotopic-map", str(snapshot / "retinotopic-vision-v1.json"),
        "--wing-motor-map", str(snapshot / "wing-motor-neurons-v0.json"),
        "--body-motor-map", str(snapshot / "body-motor-neurons-v0.json"),
        "--output-dir", str(output),
        "--max-control-steps", str(max_control_steps),
        "--checkpoint-every", "999999",
        "--trajectory-stride", "1000000",
        "--curriculum-x-step-mm", "0.000000001",
        "--curriculum-failure-x-step-mm", "0.000000001",
        "--curriculum-z-step-mm", "0.000000001",
        "--curriculum-speed-step-mm-s", "0.000000001",
    ]

    print(f"\n== {mode} population={population} ==", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    summary_path = output / "summary.json"
    summary = None
    checkpoint_digest = None
    if proc.returncode == 0 and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint_digest = tree_digest(output / "checkpoint")
        print(
            "case=PASS mode={} population={} rate={:.6f} elapsed={:.3f}s".format(
                mode,
                population,
                float(summary.get("aggregate_control_steps_per_second", 0.0)),
                float(summary.get("elapsed_seconds", 0.0)),
            ),
            flush=True,
        )
    else:
        print(f"case=FAIL mode={mode} population={population} returncode={proc.returncode}", flush=True)
        if proc.stderr:
            print(tail(proc.stderr), file=sys.stderr, flush=True)

    return CaseResult(
        mode=mode,
        population=population,
        returncode=proc.returncode,
        wall_seconds=wall_seconds,
        output_dir=output,
        summary=summary,
        checkpoint_digest=checkpoint_digest,
        stdout_tail=tail(proc.stdout),
        stderr_tail=tail(proc.stderr),
    )


def rate(case: CaseResult) -> float:
    if case.summary is None:
        return 0.0
    return float(case.summary.get("aggregate_control_steps_per_second", 0.0))


def logical_signature(case: CaseResult) -> object:
    if case.summary is None:
        return None
    results = []
    for item in case.summary.get("episode_results", []):
        results.append(
            {
                key: item.get(key)
                for key in (
                    "episode",
                    "slot",
                    "control_steps",
                    "passed_gates",
                    "collision",
                    "collision_reason",
                    "finished",
                    "reward_events",
                    "aversive_events",
                    "source_weight_version",
                    "commit_from_version",
                    "commit_weight_version",
                    "version_staleness",
                    "max_x_mm",
                    "min_z_mm",
                    "max_z_mm",
                    "final_vx_mm_s",
                )
            }
        )
    return results


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    bench_root = absolute(args.bench_root)
    report = absolute(args.report)
    snapshot = absolute(args.snapshot)
    calibration_path = absolute(DEFAULT_CALIBRATION)
    live_populations = parse_populations(args.live_populations)

    if args.episodes < max(live_populations):
        raise SystemExit("episodes must be >= the largest live population so every slot receives work")
    if args.episodes < 2 or args.max_control_steps < 1:
        raise SystemExit("episodes must be >=2 and max-control-steps >=1")

    required = [
        production / "checkpoint" / "manifest.json",
        production / "curriculum-state.json",
        snapshot / "manifest.json",
        snapshot / "embodiment-groups-v0.json",
        snapshot / "retinotopic-vision-v1.json",
        snapshot / "wing-motor-neurons-v0.json",
        snapshot / "body-motor-neurons-v0.json",
        calibration_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing benchmark inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    base_env = dict(os.environ)
    base_env["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))

    print("building benchmark bridge binaries...", flush=True)
    build = subprocess.run(
        [
            "cargo", "build", "-q", "-p", "vf-runner", "--release",
            "--bin", "population_neural_bridge",
            "--bin", "population_frontier_profile_bridge",
        ],
        cwd=ROOT,
        env=base_env,
        text=True,
        capture_output=True,
        check=False,
    )
    if build.returncode != 0:
        print(build.stdout)
        print(build.stderr, file=sys.stderr)
        raise SystemExit(build.returncode)

    before = tree_digest(production / "checkpoint")
    if bench_root.exists():
        shutil.rmtree(bench_root)
    bench_root.mkdir(parents=True, exist_ok=True)

    cases: list[CaseResult] = []
    for population in (1, 2):
        cases.append(
            run_case(
                mode="dense",
                population=population,
                source=production,
                bench_root=bench_root,
                snapshot=snapshot,
                episodes=args.episodes,
                max_control_steps=args.max_control_steps,
                base_env=base_env,
            )
        )
    for population in live_populations:
        cases.append(
            run_case(
                mode="live",
                population=population,
                source=production,
                bench_root=bench_root,
                snapshot=snapshot,
                episodes=args.episodes,
                max_control_steps=args.max_control_steps,
                base_env=base_env,
            )
        )

    after = tree_digest(production / "checkpoint")
    production_unchanged = before == after
    by_key = {(case.mode, case.population): case for case in cases}
    all_passed = production_unchanged and all(case.returncode == 0 and case.summary is not None for case in cases)

    parity_rows: list[tuple[int, bool, bool]] = []
    for population in (1, 2):
        dense = by_key.get(("dense", population))
        live = by_key.get(("live", population))
        if dense is None or live is None or dense.summary is None or live.summary is None:
            parity_rows.append((population, False, False))
            continue
        checkpoint_equal = dense.checkpoint_digest == live.checkpoint_digest
        logical_equal = logical_signature(dense) == logical_signature(live)
        parity_rows.append((population, logical_equal, checkpoint_equal))
        all_passed = all_passed and logical_equal and checkpoint_equal

    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Flyppy eager-live population runtime benchmark",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if all_passed else 'FAIL'}",
        f"- episodes per case: {args.episodes}",
        f"- max control steps per episode: {args.max_control_steps}",
        "- curriculum drift: effectively frozen (1e-9 reset-step changes)",
        f"- production checkpoint unchanged: {'yes' if production_unchanged else 'NO'}",
        f"- production checkpoint digest before: `{before}`",
        f"- production checkpoint digest after: `{after}`",
        "",
        "## Throughput",
        "",
        "| runtime | population | aggregate control steps/s | trainer elapsed s | outer wall s | mean staleness | max staleness |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        if case.summary is None:
            lines.append(f"| {case.mode} | {case.population} | FAIL | - | {case.wall_seconds:.3f} | - | - |")
            continue
        lines.append(
            "| {} | {} | {:.6f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                case.mode,
                case.population,
                rate(case),
                float(case.summary.get("elapsed_seconds", 0.0)),
                case.wall_seconds,
                float(case.summary.get("mean_version_staleness", 0.0)),
                int(case.summary.get("max_version_staleness", 0)),
            )
        )

    lines.extend(["", "## Dense-P → eager-live speedup", ""])
    for population in (1, 2):
        dense = by_key.get(("dense", population))
        live = by_key.get(("live", population))
        rd = rate(dense) if dense else 0.0
        rl = rate(live) if live else 0.0
        ratio = rl / rd if rd > 0 else 0.0
        lines.append(f"- N={population}: {ratio:.3f}x ({rd:.6f} → {rl:.6f} aggregate control steps/s)")

    lines.extend(["", "## Live-runtime population scaling", ""])
    live1 = by_key.get(("live", 1))
    r1 = rate(live1) if live1 else 0.0
    for population in live_populations:
        case = by_key[("live", population)]
        rp = rate(case)
        lines.append(
            f"- N={population}: {rp:.6f} aggregate control steps/s; vs N=1 = {(rp / r1 if r1 > 0 else 0.0):.3f}x"
        )

    lines.extend([
        "",
        "## Numerical / trajectory parity",
        "",
        "Dense-P and eager-live cases use the same production checkpoint, frozen curriculum, seeds, body physics, and trainer. The following compares episode-result signatures and final checkpoint tree digests.",
        "",
        "| population | episode results equal | final checkpoint digest equal |",
        "|---:|---|---|",
    ])
    for population, logical_equal, checkpoint_equal in parity_rows:
        lines.append(
            f"| {population} | {'yes' if logical_equal else 'NO'} | {'yes' if checkpoint_equal else 'NO'} |"
        )

    failed = [case for case in cases if case.returncode != 0 or case.summary is None]
    if failed:
        lines.extend(["", "## Failed cases", ""])
        for case in failed:
            lines.extend(
                [
                    f"### {case.mode} N={case.population}",
                    "",
                    f"returncode: {case.returncode}",
                    "",
                    "stdout tail:",
                    "",
                    "```text",
                    case.stdout_tail.rstrip(),
                    "```",
                    "",
                    "stderr tail:",
                    "",
                    "```text",
                    case.stderr_tail.rstrip(),
                    "```",
                    "",
                ]
            )

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"population_live_benchmark={report}")
    print(f"overall={'PASS' if all_passed else 'FAIL'}")
    print(f"production_checkpoint_modified={'false' if production_unchanged else 'TRUE'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
