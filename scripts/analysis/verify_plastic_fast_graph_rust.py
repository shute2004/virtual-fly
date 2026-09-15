#!/usr/bin/env python3
"""Verify Rust PlasticFastGraph compilation against the Python structural analysis.

This script does not mutate training/checkpoint state. It runs the Rust probe on
exactly the released MaleCNS snapshot, compares its structural counts with the
Python-generated plastic-fast-graph-v1 metadata, and writes a compact report for
remote review.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_GRAPH = Path("artifacts/malecns-v1.0/plastic-fast-graph-v1.json")
DEFAULT_REPORT = Path("reports/flyppy/plastic_fast_graph_rust_parity.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object in {path}")
    return payload


def main() -> int:
    args = parse_args()
    snapshot = absolute(args.snapshot)
    graph = absolute(args.graph)
    report = absolute(args.report)

    if not graph.exists():
        raise SystemExit(
            f"missing {graph}; run scripts/data/prepare_plastic_fast_graph.py first"
        )

    expected = read_json(graph)
    command = [
        "cargo",
        "run",
        "-q",
        "-p",
        "vf-runner",
        "--bin",
        "plastic_fast_graph_probe",
        "--release",
        "--",
        "--snapshot",
        str(snapshot),
    ]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    actual: dict | None = None
    parse_error: str | None = None
    if proc.returncode == 0:
        try:
            actual = json.loads(proc.stdout)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

    checks: dict[str, tuple[object, object, bool]] = {}
    if actual is not None:
        mapping = {
            "neuron_count": "source_neuron_count",
            "edge_count": "source_edge_count",
            "dopamine_capable_post_count": "dopamine_capable_post_count",
            "plastic_edge_count": "plastic_edge_count",
        }
        for actual_key, expected_key in mapping.items():
            a = actual.get(actual_key)
            e = expected.get(expected_key)
            checks[actual_key] = (e, a, a == e)

    ok = (
        proc.returncode == 0
        and parse_error is None
        and bool(checks)
        and all(item[2] for item in checks.values())
    )

    lines = [
        "# PlasticFastGraph Rust/Python parity",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- mutates training/checkpoint state: false",
        f"- overall: {'PASS' if ok else 'FAIL'}",
        f"- rust_returncode: {proc.returncode}",
        "",
        "## Structural count parity",
        "",
        "| metric | Python expected | Rust actual | match |",
        "|---|---:|---:|---|",
    ]
    for key in (
        "neuron_count",
        "edge_count",
        "dopamine_capable_post_count",
        "plastic_edge_count",
    ):
        if key in checks:
            expected_value, actual_value, match = checks[key]
            lines.append(
                f"| {key} | {expected_value} | {actual_value} | {'yes' if match else 'NO'} |"
            )

    lines.extend(
        [
            "",
            "## Rust output",
            "",
            "```text",
            proc.stdout.rstrip() or "(empty)",
            "```",
        ]
    )
    if proc.stderr.strip() or parse_error:
        lines.extend(
            [
                "",
                "## Diagnostics",
                "",
                "```text",
                (parse_error + "\n" if parse_error else "") + proc.stderr.rstrip(),
                "```",
            ]
        )
    lines.append("")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"plastic_fast_graph_rust_parity_report={report}")
    print(f"plastic_fast_graph_rust_parity={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
