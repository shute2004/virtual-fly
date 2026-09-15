#!/usr/bin/env python3
"""Verify outgoing-frontier + eager live-plasticity population runtime.

Read-only with respect to the production Flyppy experiment. It checks:

1. synthetic per-step full-state bitwise parity against the dense reference,
2. two-slot parity when only one slot receives extra plasticity teaching steps,
3. real MaleCNS all-neuron event parity on every step and final full-state parity,
4. asynchronous shared-weight stale-rebase smoke semantics,
5. production checkpoint digest stability.

No production training is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path("artifacts/experiments/flyppy-v3/checkpoint")
DEFAULT_REPORT = Path("reports/flyppy/population_live_verify.md")


@dataclass
class Stage:
    name: str
    command: list[str]
    returncode: int
    seconds: float
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
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


def run_stage(name: str, command: list[str]) -> Stage:
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return Stage(
        name=name,
        command=command,
        returncode=proc.returncode,
        seconds=time.perf_counter() - started,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def fenced(text: str, limit: int = 12000) -> str:
    if len(text) > limit:
        text = "[truncated to final output]\n" + text[-limit:]
    return "```text\n" + text.rstrip() + "\n```"


def main() -> int:
    args = parse_args()
    checkpoint = absolute(args.checkpoint)
    report = absolute(args.report)
    if not (checkpoint / "manifest.json").exists():
        raise SystemExit(f"missing checkpoint: {checkpoint}")

    before = tree_digest(checkpoint)
    stages = [
        run_stage(
            "synthetic per-step full-state golden",
            [
                "cargo",
                "test",
                "-p",
                "vf-neural",
                "outgoing_frontier_is_bitwise_equal_to_dense_incoming_reference",
                "--",
                "--nocapture",
            ],
        ),
        run_stage(
            "two-slot async teaching parity",
            [
                "cargo",
                "test",
                "-p",
                "vf-neural",
                "live_plasticity_bitmap_preserves_idle_slot_across_async_teaching_steps",
                "--",
                "--nocapture",
            ],
        ),
        run_stage(
            "real MaleCNS live-runtime parity",
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "vf-runner",
                "--bin",
                "population_frontier_malecns_parity",
                "--release",
                "--",
                "--snapshot",
                "artifacts/malecns-v1.0",
            ],
        ),
        run_stage(
            "shared-weight population smoke",
            [
                "uv",
                "run",
                "python",
                "scripts/analysis/smoke_population_neural_bridge.py",
            ],
        ),
    ]
    after = tree_digest(checkpoint)
    checkpoint_unchanged = before == after
    passed = checkpoint_unchanged and all(stage.returncode == 0 for stage in stages)

    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eager live-plasticity population runtime verification",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if passed else 'FAIL'}",
        "- training performed: false",
        f"- production checkpoint unchanged: {'yes' if checkpoint_unchanged else 'NO'}",
        f"- checkpoint digest before: `{before}`",
        f"- checkpoint digest after: `{after}`",
        "",
        "## Contract",
        "",
        "Propagation uses outgoing adjacency only to discover candidate posts; actual current accumulation preserves original incoming-CSR order. Plasticity uses an eager live bitmap: every edge with non-zero incoming eligibility remains live, and edges whose current local term can become non-zero are added before applying the unchanged eligibility/weight/transaction equations. The two physical bitmap buffers are mirrored after every active plasticity step so slots that are idle during another slot's extra teaching steps preserve their frontier exactly. No lazy decay approximation is used.",
        "",
        "## Expected structural cost",
        "",
        "- MaleCNS PlasticFastGraph P: 10,871,322 edges",
        "- dense plasticity dispatch: P edge invocations / active slot / neural step",
        "- live bitmap dispatch: ceil(P/32) = 339,729 word invocations / active slot / neural step, plus equation work only for live bits",
        "- two mirrored live bitmaps: about 2.59 MiB / slot",
        "",
        "## Verification stages",
        "",
        "| stage | result | seconds | returncode |",
        "|---|---|---:|---:|",
    ]
    for stage in stages:
        lines.append(
            f"| {stage.name} | {'PASS' if stage.returncode == 0 else 'FAIL'} | {stage.seconds:.3f} | {stage.returncode} |"
        )

    for stage in stages:
        lines.extend(
            [
                "",
                f"### {stage.name}",
                "",
                "Command:",
                "",
                fenced(" ".join(stage.command)),
                "",
                "stdout:",
                "",
                fenced(stage.stdout or "<empty>"),
            ]
        )
        if stage.stderr:
            lines.extend(["", "stderr:", "", fenced(stage.stderr)])

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"population_live_verify={report}")
    print(f"overall={'PASS' if passed else 'FAIL'}")
    print(f"production_checkpoint_modified={'false' if checkpoint_unchanged else 'TRUE'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
