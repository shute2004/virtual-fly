#!/usr/bin/env python3
"""Verify and time population checkpoint save without touching production state."""

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
import time

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/checkpoint-io")
DEFAULT_REPORT = Path("reports/flyppy/checkpoint_io_verify.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def tree_digest(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode("utf-8"))
        h.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    checkpoint = production / "checkpoint"
    snapshot = ROOT / "artifacts/malecns-v1.0"
    groups = snapshot / "embodiment-groups-v0.json"
    calibration_path = ROOT / DEFAULT_CALIBRATION

    required = [
        checkpoint / "manifest.json",
        checkpoint / "weights.f32le",
        snapshot / "manifest.json",
        groups,
        calibration_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing checkpoint-I/O inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))

    before = tree_digest(checkpoint)
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    saved = temp / "checkpoint"

    test_started = time.perf_counter()
    test = subprocess.run(
        ["cargo", "test", "-q", "-p", "vf-runner", "checkpoint::tests::full_state_round_trip"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    test_seconds = time.perf_counter() - test_started
    if test.returncode != 0:
        raise RuntimeError(
            "checkpoint round-trip test failed\nstdout:\n{}\nstderr:\n{}".format(
                test.stdout, test.stderr
            )
        )

    from population_neural_bridge_client import PopulationNeuralBridgeClient

    with PopulationNeuralBridgeClient(snapshot=snapshot, groups=groups, slots=1) as client:
        load_started = time.perf_counter()
        client.load_checkpoint(checkpoint)
        load_seconds = time.perf_counter() - load_started

        save_started = time.perf_counter()
        client.save_checkpoint(saved)
        save_seconds = time.perf_counter() - save_started

    after = tree_digest(checkpoint)
    production_unchanged = before == after
    source_weights_digest = file_digest(checkpoint / "weights.f32le")
    saved_weights_digest = file_digest(saved / "weights.f32le")
    weights_equal = source_weights_digest == saved_weights_digest

    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    sizes = {
        path.name: path.stat().st_size
        for path in sorted(saved.iterdir())
        if path.is_file()
    }
    total_bytes = sum(sizes.values())
    passed = production_unchanged and weights_equal and test.returncode == 0

    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Population checkpoint I/O verification",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if passed else 'FAIL'}",
        f"- production checkpoint unchanged: {'yes' if production_unchanged else 'NO'}",
        f"- production checkpoint digest: `{before}`",
        f"- checkpoint round-trip unit test: PASS ({test_seconds:.3f} s)",
        f"- population checkpoint load wall time: {load_seconds:.3f} s",
        f"- population checkpoint save wall time: {save_seconds:.3f} s",
        f"- saved total bytes: {total_bytes:,}",
        f"- source/saved weight bytes identical: {'yes' if weights_equal else 'NO'}",
        f"- source weights SHA-256: `{source_weights_digest}`",
        f"- saved weights SHA-256: `{saved_weights_digest}`",
        f"- schema_version: {manifest.get('schema_version')}",
        "",
        "## Saved files",
        "",
        "| file | bytes |",
        "|---|---:|",
    ]
    for name, size in sizes.items():
        lines.append(f"| {name} | {size:,} |")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"checkpoint_io_verify={report}")
    print(f"overall={'PASS' if passed else 'FAIL'}")
    print(f"checkpoint_save_seconds={save_seconds:.6f}")
    print(f"production_checkpoint_modified={'false' if production_unchanged else 'TRUE'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
