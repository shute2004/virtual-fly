#!/usr/bin/env python3
"""Benchmark local-body vs process-body Flyppy trainers on identical temp state.

Every case starts from a private copy of the production checkpoint/curriculum.
For each population size, the two trainers must produce the same final CNS
checkpoint and episode outcomes. Production state is never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-process-body-benchmark")
DEFAULT_REPORT = Path("reports/flyppy/process_body_benchmark.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-control-steps", type=int, default=32)
    return p.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def tree_digest(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def clone_state(production: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(production / "checkpoint", out / "checkpoint")
    shutil.copy2(production / "curriculum-state.json", out / "curriculum-state.json")
    population = production / "population-state.json"
    if population.exists():
        shutil.copy2(population, out / "population-state.json")


def comparable_episode(item: dict) -> dict:
    keys = (
        "episode", "slot", "source_weight_version", "commit_from_version",
        "commit_weight_version", "version_staleness", "control_steps",
        "passed_gates", "collision", "collision_reason", "finished",
        "max_x_mm", "min_z_mm", "max_z_mm", "final_vx_mm_s",
        "spawn_x_mm", "spawn_z_mm", "initial_speed_mm_s",
        "reward_events", "aversive_events",
    )
    return {key: item.get(key) for key in keys}


def run_case(
    *,
    script: Path,
    out: Path,
    population: int,
    episodes: int,
    max_steps: int,
    env: dict[str, str],
) -> dict:
    command = [
        sys.executable,
        str(script),
        "--episodes", str(episodes),
        "--population", str(population),
        "--output-dir", str(out),
        "--max-control-steps", str(max_steps),
        "--checkpoint-every", str(episodes + 1000),
        "--trajectory-stride", "1000000",
        "--curriculum-x-step-mm", "0.000000001",
        "--curriculum-failure-x-step-mm", "0.000000001",
        "--curriculum-z-step-mm", "0.000000001",
        "--curriculum-speed-step-mm-s", "0.000000001",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-120:])
        raise RuntimeError(
            f"{script.name} N={population} failed ({completed.returncode}):\n{tail}"
        )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "checkpoint_digest": tree_digest(out / "checkpoint"),
        "episodes": [
            comparable_episode(item)
            for item in sorted(summary["episode_results"], key=lambda row: int(row["episode"]))
        ],
    }


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    required = [
        production / "checkpoint/manifest.json",
        production / "curriculum-state.json",
        ROOT / DEFAULT_CALIBRATION,
        EMBODIMENT / "train_flyppy_population.py",
        EMBODIMENT / "train_flyppy_population_process.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing process-body benchmark inputs:\n  " + "\n  ".join(missing))
    if args.episodes < 8:
        raise SystemExit("--episodes must be >= 8 so N=8 has a full batch")

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    before = tree_digest(production / "checkpoint")

    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    rows = []
    all_equal = True
    for population in (1, 2, 4, 8):
        local_out = temp / f"local-n{population}"
        process_out = temp / f"process-n{population}"
        clone_state(production, local_out)
        clone_state(production, process_out)

        local = run_case(
            script=EMBODIMENT / "train_flyppy_population.py",
            out=local_out,
            population=population,
            episodes=args.episodes,
            max_steps=args.max_control_steps,
            env=env,
        )
        process = run_case(
            script=EMBODIMENT / "train_flyppy_population_process.py",
            out=process_out,
            population=population,
            episodes=args.episodes,
            max_steps=args.max_control_steps,
            env=env,
        )

        digest_equal = local["checkpoint_digest"] == process["checkpoint_digest"]
        episode_equal = local["episodes"] == process["episodes"]
        version_equal = (
            int(local["summary"]["global_weight_version_end"])
            == int(process["summary"]["global_weight_version_end"])
            and int(local["summary"]["checkpoint_neural_step"])
            == int(process["summary"]["checkpoint_neural_step"])
        )
        equal = digest_equal and episode_equal and version_equal
        all_equal = all_equal and equal

        local_sps = float(local["summary"]["aggregate_control_steps_per_second"])
        process_sps = float(process["summary"]["aggregate_control_steps_per_second"])
        rows.append(
            {
                "population": population,
                "local_sps": local_sps,
                "process_sps": process_sps,
                "speedup": process_sps / local_sps,
                "digest_equal": digest_equal,
                "episode_equal": episode_equal,
                "version_equal": version_equal,
                "local_elapsed": float(local["summary"]["elapsed_seconds"]),
                "process_elapsed": float(process["summary"]["elapsed_seconds"]),
            }
        )

    after = tree_digest(production / "checkpoint")
    production_unchanged = before == after
    overall = all_equal and production_unchanged

    n1 = rows[0]
    n8 = rows[-1]
    process_scaling = n8["process_sps"] / n1["process_sps"]
    local_scaling = n8["local_sps"] / n1["local_sps"]

    lines = [
        "# Flyppy process-body production trainer benchmark",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        f"- episodes per case: {args.episodes}",
        f"- max control steps per episode: {args.max_control_steps}",
        "- curriculum movement during benchmark: effectively frozen",
        f"- production checkpoint modified: {'no' if production_unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "| population | local steps/s | process steps/s | process/local | local elapsed s | process elapsed s | checkpoint equal | episode equal | version/step equal |",
        "|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {population} | {local_sps:.6f} | {process_sps:.6f} | {speedup:.3f}x | "
            "{local_elapsed:.3f} | {process_elapsed:.3f} | {digest_equal} | {episode_equal} | {version_equal} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Scaling",
            "",
            f"- local N=8 / N=1 aggregate throughput: {local_scaling:.3f}x",
            f"- process-body N=8 / N=1 aggregate throughput: {process_scaling:.3f}x",
            f"- process-body N=8 vs local N=8: {n8['speedup']:.3f}x",
            "",
            "The process-body trainer is acceptable for production cutover only if every same-N local/process case preserves checkpoint bytes and episode/commit outcomes.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"process_body_benchmark={report}")
    print(f"process_body_benchmark_pass={str(overall).lower()}")
    print(f"process_N8_vs_N1={process_scaling:.6f}")
    print(f"process_N8_vs_local_N8={n8['speedup']:.6f}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
