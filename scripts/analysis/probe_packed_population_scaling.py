#!/usr/bin/env python3
"""Measure packed Flyppy body-worker scaling across larger populations.

This meta-probe reuses probe_body_worker_packing.py so every population is tested
with the real shared GPU MaleCNS and the same checkpoint/outcome parity contract.
It intentionally varies population independently from body-process count; the
result is used to choose a packing rule rather than baking in the N=8 optimum.

Production checkpoint/state are never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-packed-population-scaling")
DEFAULT_REPORT = Path("reports/flyppy/packed_population_scaling.md")
PACKING_PROBE = ROOT / "scripts/analysis/probe_body_worker_packing.py"

ROW_RE = re.compile(
    r"^\|\s*(?P<processes>\d+)\s*\|\s*(?P<flies>[0-9.]+)\s*\|\s*"
    r"(?P<steps>\d+)\s*\|\s*(?P<sps>[0-9.]+)\s*\|\s*"
    r"(?P<observe>[0-9.]+)\s*\|\s*(?P<cns>[0-9.]+)\s*\|\s*"
    r"(?P<act>[0-9.]+)\s*\|\s*(?P<startup>[0-9.]+)\s*\|\s*"
    r"(?P<equal>True|False)\s*\|$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--populations", type=int, nargs="+", default=(4, 8, 12))
    p.add_argument("--max-control-steps", type=int, default=48)
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


def parse_child_report(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line.strip())
        if not match:
            continue
        values = match.groupdict()
        rows.append(
            {
                "processes": int(values["processes"]),
                "flies_per_process": float(values["flies"]),
                "steps": int(values["steps"]),
                "sps": float(values["sps"]),
                "observe_s": float(values["observe"]),
                "cns_s": float(values["cns"]),
                "act_s": float(values["act"]),
                "startup_s": float(values["startup"]),
                "equal": values["equal"] == "True",
            }
        )
    if not rows:
        raise RuntimeError(f"no packing rows parsed from {path}")
    return rows


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    populations = tuple(dict.fromkeys(int(v) for v in args.populations))
    if any(v < 1 or v > 32 for v in populations):
        raise SystemExit("populations must be in 1..32")
    if args.max_control_steps < 1:
        raise SystemExit("--max-control-steps must be >= 1")
    required = [production / "checkpoint/manifest.json", PACKING_PROBE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing packed scaling inputs:\n  " + "\n  ".join(missing))

    temp.mkdir(parents=True, exist_ok=True)
    before = tree_digest(production / "checkpoint")
    all_rows: list[dict[str, object]] = []
    child_pass = True

    for population in populations:
        child_report = temp / f"packing-n{population}.md"
        child_temp = temp / f"n{population}"
        command = [
            sys.executable,
            str(PACKING_PROBE),
            "--production",
            str(production),
            "--temp",
            str(child_temp),
            "--report",
            str(child_report),
            "--population",
            str(population),
            "--max-control-steps",
            str(args.max_control_steps),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-120:])
            raise RuntimeError(f"packing probe N={population} failed:\n{tail}")
        text = child_report.read_text(encoding="utf-8")
        child_pass = child_pass and "- overall: PASS" in text
        rows = parse_child_report(child_report)
        # With process_count > population, the underlying assignment collapses
        # empty groups. Keep only physically distinct process counts here.
        seen_actual: set[int] = set()
        for row in rows:
            requested = int(row["processes"])
            actual = min(requested, population)
            if actual in seen_actual:
                continue
            seen_actual.add(actual)
            row = dict(row)
            row["population"] = population
            row["actual_processes"] = actual
            all_rows.append(row)

    after = tree_digest(production / "checkpoint")
    production_unchanged = before == after
    parity_ok = all(bool(row["equal"]) for row in all_rows)
    overall = child_pass and parity_ok and production_unchanged

    best_by_population: dict[int, dict[str, object]] = {}
    for population in populations:
        candidates = [row for row in all_rows if int(row["population"]) == population]
        best_by_population[population] = max(candidates, key=lambda row: float(row["sps"]))

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
            "| {population} | {actual_processes} | {flies_per_process:.2f} | {sps:.3f} | "
            "{observe_s:.3f} | {cns_s:.3f} | {act_s:.3f} | {startup_s:.3f} | {equal} |".format(**row)
        )

    lines.extend(["", "## Best packing by population", ""])
    for population in populations:
        row = best_by_population[population]
        lines.append(
            f"- N={population}: {int(row['actual_processes'])} body processes, "
            f"{float(row['flies_per_process']):.2f} flies/process, "
            f"{float(row['sps']):.3f} steps/s"
        )

    if len(populations) >= 2:
        first = populations[0]
        last = populations[-1]
        first_sps = float(best_by_population[first]["sps"])
        last_sps = float(best_by_population[last]["sps"])
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
            "The goal is to identify a packing rule that preserves independent flies while avoiding excessive MuJoCo/Python/renderer process duplication.",
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
            f"N{population}_best_processes={int(row['actual_processes'])} "
            f"N{population}_best_steps_per_second={float(row['sps']):.6f}"
        )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
