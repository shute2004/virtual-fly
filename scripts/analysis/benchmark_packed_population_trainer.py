#!/usr/bin/env python3
"""Benchmark packed-body Flyppy trainer against the local-body reference.

Every case starts from an independent copy of the production checkpoint,
curriculum state, and population state.  Same-N local and packed runs must
produce identical CNS checkpoint bytes, episode/commit outcomes, global weight
version, and neural step.  Production state is never modified.
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
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-packed-trainer-benchmark")
DEFAULT_REPORT = Path("reports/flyppy/packed_trainer_benchmark.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--max-control-steps", type=int, default=32)
    p.add_argument("--populations", type=int, nargs="+", default=(4, 8, 12))
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
        tail = "\n".join(completed.stdout.splitlines()[-160:])
        raise RuntimeError(
            f"{script.name} N={population} failed ({completed.returncode}):\n{tail}"
        )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "checkpoint_digest": tree_digest(out / "checkpoint"),
        "episodes": [
            comparable_episode(item)
            for item in sorted(
                summary["episode_results"],
                key=lambda row: int(row["episode"]),
            )
        ],
        "stdout": completed.stdout,
    }


def body_process_count(stdout: str, population: int) -> int:
    marker = "body_process_packing=packed "
    for line in stdout.splitlines():
        if not line.startswith(marker):
            continue
        for token in line.split():
            if token.startswith("body_processes="):
                return int(token.split("=", 1)[1])
    return min(population, 4)


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    populations = tuple(dict.fromkeys(int(v) for v in args.populations))
    if any(value < 1 or value > 32 for value in populations):
        raise SystemExit("populations must be in 1..32")
    if args.episodes < max(populations):
        raise SystemExit("--episodes must be >= the largest population")

    required = [
        production / "checkpoint/manifest.json",
        production / "curriculum-state.json",
        ROOT / DEFAULT_CALIBRATION,
        EMBODIMENT / "train_flyppy_population.py",
        EMBODIMENT / "train_flyppy_population_packed.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing packed trainer benchmark inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    env["VF_FLYPPY_BODY_PROCESSES"] = "auto"

    before = tree_digest(production / "checkpoint")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    all_equal = True
    for population in populations:
        local_out = temp / f"local-n{population}"
        packed_out = temp / f"packed-n{population}"
        clone_state(production, local_out)
        clone_state(production, packed_out)

        local = run_case(
            script=EMBODIMENT / "train_flyppy_population.py",
            out=local_out,
            population=population,
            episodes=args.episodes,
            max_steps=args.max_control_steps,
            env=env,
        )
        packed = run_case(
            script=EMBODIMENT / "train_flyppy_population_packed.py",
            out=packed_out,
            population=population,
            episodes=args.episodes,
            max_steps=args.max_control_steps,
            env=env,
        )

        digest_equal = local["checkpoint_digest"] == packed["checkpoint_digest"]
        episode_equal = local["episodes"] == packed["episodes"]
        version_equal = (
            int(local["summary"]["global_weight_version_end"])
            == int(packed["summary"]["global_weight_version_end"])
            and int(local["summary"]["checkpoint_neural_step"])
            == int(packed["summary"]["checkpoint_neural_step"])
        )
        equal = digest_equal and episode_equal and version_equal
        all_equal = all_equal and equal
        local_sps = float(local["summary"]["aggregate_control_steps_per_second"])
        packed_sps = float(packed["summary"]["aggregate_control_steps_per_second"])
        rows.append(
            {
                "population": population,
                "body_processes": body_process_count(packed["stdout"], population),
                "local_sps": local_sps,
                "packed_sps": packed_sps,
                "speedup": packed_sps / local_sps,
                "local_elapsed": float(local["summary"]["elapsed_seconds"]),
                "packed_elapsed": float(packed["summary"]["elapsed_seconds"]),
                "digest_equal": digest_equal,
                "episode_equal": episode_equal,
                "version_equal": version_equal,
            }
        )

    after = tree_digest(production / "checkpoint")
    production_unchanged = before == after
    overall = all_equal and production_unchanged

    lines = [
        "# Flyppy packed production-trainer benchmark",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        f"- populations: {', '.join(str(value) for value in populations)}",
        f"- episodes per case: {args.episodes}",
        f"- max control steps per episode: {args.max_control_steps}",
        "- packed body-process rule: min(population, 4)",
        f"- production checkpoint modified: {'no' if production_unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "| population | body processes | local steps/s | packed steps/s | packed/local | local elapsed s | packed elapsed s | checkpoint equal | episode equal | version/step equal |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {population} | {body_processes} | {local_sps:.6f} | {packed_sps:.6f} | "
            "{speedup:.3f}x | {local_elapsed:.3f} | {packed_elapsed:.3f} | "
            "{digest_equal} | {episode_equal} | {version_equal} |".format(**row)
        )

    if len(rows) >= 2:
        first = rows[0]
        last = rows[-1]
        lines.extend(
            [
                "",
                "## Scaling",
                "",
                "- packed N={} / N={} aggregate throughput: {:.3f}x".format(
                    last["population"],
                    first["population"],
                    float(last["packed_sps"]) / float(first["packed_sps"]),
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Packed cutover is acceptable only when every same-N case preserves checkpoint bytes, episode/commit outcomes, global weight version, and neural step.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"packed_trainer_benchmark={report}")
    print(f"packed_trainer_benchmark_pass={str(overall).lower()}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
