#!/usr/bin/env python3
"""Verify the PlasticFastGraph-backed population runtime without training.

Checks:
- vf-neural PlasticFastGraph/unit tests,
- population bridge compilation,
- real MaleCNS shared-weight smoke path (2 slots),
- production checkpoint byte-for-byte unchanged.

A Markdown report is written to reports/flyppy/sparse_population_runtime_verify.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path("artifacts/experiments/flyppy-v3/checkpoint")
DEFAULT_PLASTIC = Path("artifacts/malecns-v1.0/plastic-fast-graph-v1.json")
DEFAULT_REPORT = Path("reports/flyppy/sparse_population_runtime_verify.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--plastic-graph", type=Path, default=DEFAULT_PLASTIC)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def tree_digest(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return "MISSING"
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def run(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - started,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "ok": proc.returncode == 0,
    }


def main() -> int:
    args = parse_args()
    checkpoint = absolute(args.checkpoint)
    plastic_path = absolute(args.plastic_graph)
    report = absolute(args.report)

    if not (checkpoint / "manifest.json").exists():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    if not plastic_path.exists():
        raise SystemExit(f"missing PlasticFastGraph metadata: {plastic_path}")

    plastic = json.loads(plastic_path.read_text(encoding="utf-8"))
    before = tree_digest(checkpoint)

    stages = [
        run(
            "vf-neural tests",
            ["cargo", "test", "-q", "-p", "vf-neural", "plastic_graph::tests"],
        ),
        run(
            "population bridge check",
            ["cargo", "check", "-q", "-p", "vf-runner", "--bin", "population_neural_bridge"],
        ),
        run(
            "real MaleCNS population smoke",
            [sys.executable, "scripts/analysis/smoke_population_neural_bridge.py"],
        ),
    ]

    after = tree_digest(checkpoint)
    checkpoint_unchanged = before == after and before != "MISSING"
    ok = all(stage["ok"] for stage in stages) and checkpoint_unchanged

    n = int(plastic["source_neuron_count"])
    e = int(plastic["source_edge_count"])
    p = int(plastic["plastic_edge_count"])
    old_slot = 28 * n + 20 * e
    sparse_slot = 28 * n + 20 * p

    lines = [
        "# Sparse population runtime verification",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if ok else 'FAIL'}",
        "- training performed: false",
        f"- production checkpoint unchanged: {'yes' if checkpoint_unchanged else 'NO'}",
        f"- checkpoint digest before: `{before}`",
        f"- checkpoint digest after: `{after}`",
        "",
        "## Structural state reduction",
        "",
        f"- N: {n:,}",
        f"- E: {e:,}",
        f"- P: {p:,}",
        f"- P/E: {p / e:.8%}",
        f"- previous slot-local GPU state model: {old_slot:,} bytes ({old_slot / 1024**2:.2f} MiB)",
        f"- current P-backed slot-local GPU state model: {sparse_slot:,} bytes ({sparse_slot / 1024**2:.2f} MiB)",
        f"- structural reduction: {old_slot / sparse_slot:.3f}x",
        "",
        "The current sparse runtime still keeps dense transaction state over P. Dirty/sparse transaction storage is the next phase.",
        "",
        "## Verification stages",
        "",
        "| stage | result | seconds | returncode |",
        "|---|---|---:|---:|",
    ]
    for stage in stages:
        lines.append(
            f"| {stage['name']} | {'PASS' if stage['ok'] else 'FAIL'} | "
            f"{stage['seconds']:.3f} | {stage['returncode']} |"
        )

    for stage in stages:
        if stage["stdout"].strip() or stage["stderr"].strip():
            lines.extend([
                "",
                f"### {stage['name']}",
                "",
                "```text",
                (stage["stdout"] + stage["stderr"])[-8000:].rstrip(),
                "```",
            ])
    lines.append("")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"sparse_population_runtime_report={report}")
    print(f"sparse_population_runtime={'PASS' if ok else 'FAIL'}")
    print(f"checkpoint_modified={'false' if checkpoint_unchanged else 'true'}")
    print(f"slot_state_MiB={sparse_slot / 1024**2:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
