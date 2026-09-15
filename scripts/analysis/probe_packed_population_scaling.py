#!/usr/bin/env python3
"""Measure packed Flyppy body-worker scaling across larger populations.

This probe reuses the tested run_case implementation from
probe_body_worker_packing.py directly.  The older CLI remains an N=8 historical
probe; importing its functions lets this scaling probe vary population without
changing that historical contract.

Every case uses the real shared GPU MaleCNS runtime and the same checkpoint /
outcome parity contract. Production checkpoint/state are never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "scripts" / "analysis"
EMBODIMENT = ROOT / "scripts" / "embodiment"
for path in (ANALYSIS, EMBODIMENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-packed-population-scaling")
DEFAULT_REPORT = Path("reports/flyppy/packed_population_scaling.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--populations", type=int, nargs="+", default=(4, 8, 12))
    p.add_argument("--max-control-steps", type=int, default=48)
    p.add_argument("--physics-steps", type=int, default=10)
    p.add_argument("--timeout-s", type=float, default=120.0)
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


def process_counts_for(population: int) -> tuple[int, ...]:
    """Test physically distinct packings only.

    The historical packing probe used requested counts 1/2/4/8.  Here we keep
    that ladder but never request more owner processes than flies, avoiding the
    old N=4 alias where requested 8 silently collapsed to four non-empty groups.
    """

    return tuple(value for value in (1, 2, 4, 8) if value <= population)


def comparable_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["checkpoint_digest"],
        row["results"],
        row["final_version"],
        row["neural_step"],
    )


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    populations = tuple(dict.fromkeys(int(v) for v in args.populations))

    if any(v < 1 or v > 32 for v in populations):
        raise SystemExit("populations must be in 1..32")
    if args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("step counts must be positive")
    if args.timeout_s <= 0.0:
        raise SystemExit("--timeout-s must be positive")

    required = [
        production / "checkpoint/manifest.json",
        production / "curriculum-state.json",
        ROOT / DEFAULT_CALIBRATION,
        ANALYSIS / "probe_body_worker_packing.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing packed scaling inputs:\n  " + "\n  ".join(missing))

    # Importing does not execute the N=8-only CLI main().  We deliberately reuse
    # the already parity-tested worker/run_case implementation underneath it.
    import probe_body_worker_packing as packing
    from flyppy_course import FlyppyCourse
    from virtual_fly.training.curriculum import (
        SpawnCondition,
        current_adaptive_condition,
        load_state,
    )

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))

    population_state_path = production / "population-state.json"
    population_state = (
        json.loads(population_state_path.read_text(encoding="utf-8"))
        if population_state_path.exists()
        else {}
    )
    initial_global_version = int(population_state.get("global_weight_version", 0))

    course = FlyppyCourse(seed=0, gate_count=6, environment_version="v3")
    first_gate = course.gates[0]
    state = load_state(
        production / "curriculum-state.json",
        start=SpawnCondition(8.91, float(first_gate.center_z_mm), 400.0),
        target=SpawnCondition(0.0, 8.91, 300.0),
        checkpoint_exists=True,
    )
    condition = current_adaptive_condition(state)

    before = tree_digest(production / "checkpoint")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    all_rows: list[dict[str, object]] = []
    parity_ok = True

    for population in populations:
        reference_key: tuple[object, ...] | None = None
        for process_count in process_counts_for(population):
            row = packing.run_case(
                out=temp / f"n{population}-p{process_count}",
                production=production,
                population=population,
                process_count=process_count,
                max_steps=args.max_control_steps,
                physics_steps=args.physics_steps,
                timeout_s=args.timeout_s,
                condition=condition,
                initial_global_version=initial_global_version,
            )
            key = comparable_key(row)
            if reference_key is None:
                reference_key = key
                equal = True
            else:
                equal = key == reference_key
            parity_ok = parity_ok and equal

            record = dict(row)
            record["population"] = population
            record["processes"] = process_count
            record["flies_per_process"] = population / process_count
            record["equal"] = equal
            all_rows.append(record)

    after = tree_digest(production / "checkpoint")
    production_unchanged = before == after
    overall = parity_ok and production_unchanged

    best_by_population: dict[int, dict[str, object]] = {}
    for population in populations:
        candidates = [row for row in all_rows if int(row["population"]) == population]
        best_by_population[population] = max(
            candidates,
            key=lambda row: float(row["steady_steps_s"]),
        )

    lines = [
        "# Flyppy packed population scaling probe",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        f"- populations: {', '.join(str(v) for v in populations)}",
        f"- max control steps per slot: {args.max_control_steps}",
        "- CNS: real shared GPU MaleCNS runtime",
        f"- production checkpoint modified: {'no' if production_unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "| population | body processes | flies/process | steady steps/s | observe s | CNS s | act s | startup s | parity |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in all_rows:
        lines.append(
            "| {population} | {processes} | {flies_per_process:.2f} | {steady_steps_s:.3f} | "
            "{observe_s:.3f} | {cns_s:.3f} | {act_s:.3f} | {startup_s:.3f} | {equal} |".format(
                **row
            )
        )

    lines.extend(["", "## Best packing by population", ""])
    for population in populations:
        row = best_by_population[population]
        lines.append(
            f"- N={population}: {int(row['processes'])} body processes, "
            f"{float(row['flies_per_process']):.2f} flies/process, "
            f"{float(row['steady_steps_s']):.3f} steps/s"
        )

    if len(populations) >= 2:
        first = populations[0]
        last = populations[-1]
        first_sps = float(best_by_population[first]["steady_steps_s"])
        last_sps = float(best_by_population[last]["steady_steps_s"])
        lines.extend(
            [
                "",
                "## Best-packing scaling",
                "",
                f"- best N={last} / best N={first} aggregate throughput: {last_sps / first_sps:.3f}x",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation: population count and body-process count are independent variables. "
            "Each population is compared only across physically distinct process counts, and "
            "checkpoint/outcome parity is required within that population.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"packed_population_scaling={report}")
    print(f"packed_population_scaling_pass={str(overall).lower()}")
    for population in populations:
        row = best_by_population[population]
        print(
            f"N{population}_best_processes={int(row['processes'])} "
            f"N{population}_best_steps_per_second={float(row['steady_steps_s']):.6f}"
        )
    return 0 if overall else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
