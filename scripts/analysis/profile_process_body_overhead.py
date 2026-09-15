#!/usr/bin/env python3
"""Profile parent-side overhead of the process-isolated Flyppy trainer.

This diagnostic keeps production state immutable.  Each population case starts
from a private copy of the production checkpoint/curriculum and runs the real
process-body trainer in-process with lightweight monkey-patched timers around
body-worker waits, CNS calls, reset, checkpoint I/O, and worker teardown.

The purpose is to distinguish steady-state training cost from short-benchmark
fixed costs such as destroying eight MuJoCo renderers at shutdown.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
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
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-process-body-overhead")
DEFAULT_REPORT = Path("reports/flyppy/process_body_overhead_profile.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-control-steps", type=int, default=32)
    p.add_argument("--long-max-control-steps", type=int, default=128)
    p.add_argument("--case-population", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--case-output", type=Path, default=None, help=argparse.SUPPRESS)
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


class Timings:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - started
            self.calls[name] += 1

    def record_call(self, name: str, fn):
        def wrapped(*args, **kwargs):
            with self.measure(name):
                return fn(*args, **kwargs)
        return wrapped


def run_case_in_process(population: int, out: Path, max_steps: int) -> dict[str, object]:
    import train_flyppy_population_process as trainer
    import flyppy_body_worker
    import population_neural_bridge_client

    timings = Timings()

    original_receive = flyppy_body_worker.FlyppyBodyProcess.receive
    original_close = flyppy_body_worker.FlyppyBodyProcess.close
    original_reset = trainer.reset_slot
    bridge_cls = population_neural_bridge_client.PopulationNeuralBridgeClient
    original_step_batch = bridge_cls.step_batch
    original_step_slot = bridge_cls.step_slot
    original_commit_slot = bridge_cls.commit_slot
    original_load_checkpoint = bridge_cls.load_checkpoint
    original_save_checkpoint = bridge_cls.save_checkpoint

    def timed_receive(self, phase: str):
        name = f"body_receive_{phase}"
        with timings.measure(name):
            return original_receive(self, phase)

    def timed_close(self):
        with timings.measure("body_worker_close"):
            return original_close(self)

    def timed_reset(*args, **kwargs):
        with timings.measure("body_worker_reset"):
            return original_reset(*args, **kwargs)

    flyppy_body_worker.FlyppyBodyProcess.receive = timed_receive
    flyppy_body_worker.FlyppyBodyProcess.close = timed_close
    # train_flyppy_population_process imported the class object, so patching methods
    # above affects its instances. reset_slot itself is rebound explicitly.
    trainer.reset_slot = timed_reset
    bridge_cls.step_batch = timings.record_call("brain_step_batch", original_step_batch)
    bridge_cls.step_slot = timings.record_call("brain_step_slot", original_step_slot)
    bridge_cls.commit_slot = timings.record_call("brain_commit_slot", original_commit_slot)
    bridge_cls.load_checkpoint = timings.record_call("brain_checkpoint_load", original_load_checkpoint)
    bridge_cls.save_checkpoint = timings.record_call("brain_checkpoint_save", original_save_checkpoint)

    argv_before = sys.argv[:]
    try:
        sys.argv = [
            str(EMBODIMENT / "train_flyppy_population_process.py"),
            "--episodes", str(8),
            "--population", str(population),
            "--output-dir", str(out),
            "--max-control-steps", str(max_steps),
            "--checkpoint-every", str(1008),
            "--trajectory-stride", "1000000",
            "--curriculum-x-step-mm", "0.000000001",
            "--curriculum-failure-x-step-mm", "0.000000001",
            "--curriculum-z-step-mm", "0.000000001",
            "--curriculum-speed-step-mm-s", "0.000000001",
        ]
        outer_started = time.perf_counter()
        rc = trainer.main()
        outer_elapsed = time.perf_counter() - outer_started
        if rc != 0:
            raise RuntimeError(f"process trainer returned {rc}")
    finally:
        sys.argv = argv_before
        flyppy_body_worker.FlyppyBodyProcess.receive = original_receive
        flyppy_body_worker.FlyppyBodyProcess.close = original_close
        trainer.reset_slot = original_reset
        bridge_cls.step_batch = original_step_batch
        bridge_cls.step_slot = original_step_slot
        bridge_cls.commit_slot = original_commit_slot
        bridge_cls.load_checkpoint = original_load_checkpoint
        bridge_cls.save_checkpoint = original_save_checkpoint

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    aggregate_steps = int(summary["aggregate_control_steps"])
    trainer_elapsed = float(summary["elapsed_seconds"])
    teardown = float(timings.seconds.get("body_worker_close", 0.0))
    steady_elapsed = max(1e-12, trainer_elapsed - teardown)
    return {
        "population": population,
        "max_control_steps": max_steps,
        "aggregate_control_steps": aggregate_steps,
        "trainer_elapsed_s": trainer_elapsed,
        "outer_elapsed_s": outer_elapsed,
        "body_worker_close_s": teardown,
        "steady_elapsed_s": steady_elapsed,
        "reported_steps_s": aggregate_steps / trainer_elapsed,
        "steady_steps_s": aggregate_steps / steady_elapsed,
        "timing_seconds": dict(timings.seconds),
        "timing_calls": dict(timings.calls),
        "checkpoint_digest": tree_digest(out / "checkpoint"),
    }


def case_main(args: argparse.Namespace) -> int:
    if args.case_output is None:
        raise SystemExit("--case-output is required with --case-population")
    production = absolute(args.production)
    case_root = absolute(args.case_output)
    run_out = case_root / "run"
    clone_state(production, run_out)
    payload = run_case_in_process(args.case_population, run_out, args.max_control_steps)
    result_path = case_root / "profile.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def launch_case(
    *,
    production: Path,
    case_root: Path,
    population: int,
    max_steps: int,
    env: dict[str, str],
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--production", str(production),
        "--case-population", str(population),
        "--case-output", str(case_root),
        "--max-control-steps", str(max_steps),
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
            f"profile N={population} max_steps={max_steps} failed ({completed.returncode}):\n{tail}"
        )
    return json.loads((case_root / "profile.json").read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if args.case_population:
        return case_main(args)

    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    calibration_path = ROOT / DEFAULT_CALIBRATION
    required = [
        production / "checkpoint/manifest.json",
        production / "curriculum-state.json",
        calibration_path,
        EMBODIMENT / "train_flyppy_population_process.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing process-body overhead profile inputs:\n  " + "\n  ".join(missing))
    if args.episodes != 8:
        raise SystemExit("this diagnostic currently requires --episodes 8")

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    before = tree_digest(production / "checkpoint")

    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    rows = []
    # Short run reproduces the previous benchmark. Long N=8 tests whether fixed
    # teardown cost was the reason apparent scaling reversed.
    for population in (1, 2, 4, 8):
        rows.append(
            launch_case(
                production=production,
                case_root=temp / f"short-n{population}",
                population=population,
                max_steps=args.max_control_steps,
                env=env,
            )
        )
    long_n8 = launch_case(
        production=production,
        case_root=temp / "long-n8",
        population=8,
        max_steps=args.long_max_control_steps,
        env=env,
    )

    after = tree_digest(production / "checkpoint")
    unchanged = before == after

    lines = [
        "# Flyppy process-body parent overhead profile",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- production checkpoint modified: {'no' if unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "- short-run episode count: 8",
        f"- short-run max control steps: {args.max_control_steps}",
        f"- long N=8 max control steps: {args.long_max_control_steps}",
        "",
        "## Short-run scaling",
        "",
        "| N | aggregate steps | reported steps/s | teardown s | steady steps/s | observe wait s | act wait s | CNS batch s | commit s | checkpoint save s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        timing = row["timing_seconds"]
        lines.append(
            "| {population} | {aggregate_control_steps} | {reported_steps_s:.3f} | "
            "{body_worker_close_s:.3f} | {steady_steps_s:.3f} | {observe:.3f} | "
            "{act:.3f} | {brain:.3f} | {commit:.3f} | {save:.3f} |".format(
                population=row["population"],
                aggregate_control_steps=row["aggregate_control_steps"],
                reported_steps_s=row["reported_steps_s"],
                body_worker_close_s=row["body_worker_close_s"],
                steady_steps_s=row["steady_steps_s"],
                observe=float(timing.get("body_receive_observe", 0.0)),
                act=float(timing.get("body_receive_act", 0.0)),
                brain=float(timing.get("brain_step_batch", 0.0)),
                commit=float(timing.get("brain_commit_slot", 0.0)),
                save=float(timing.get("brain_checkpoint_save", 0.0)),
            )
        )

    lt = long_n8["timing_seconds"]
    lines.extend(
        [
            "",
            "## Long N=8 run",
            "",
            f"- aggregate control steps: {long_n8['aggregate_control_steps']}",
            f"- reported throughput: {long_n8['reported_steps_s']:.3f} steps/s",
            f"- worker teardown: {long_n8['body_worker_close_s']:.3f} s",
            f"- teardown-excluded throughput: {long_n8['steady_steps_s']:.3f} steps/s",
            f"- observe wait: {float(lt.get('body_receive_observe', 0.0)):.3f} s",
            f"- act wait: {float(lt.get('body_receive_act', 0.0)):.3f} s",
            f"- CNS batch: {float(lt.get('brain_step_batch', 0.0)):.3f} s",
            f"- reinforcement CNS: {float(lt.get('brain_step_slot', 0.0)):.3f} s",
            f"- commit: {float(lt.get('brain_commit_slot', 0.0)):.3f} s",
            f"- checkpoint load: {float(lt.get('brain_checkpoint_load', 0.0)):.3f} s",
            f"- checkpoint save: {float(lt.get('brain_checkpoint_save', 0.0)):.3f} s",
            "",
            "Interpretation: reported throughput includes worker teardown because the production-equivalent trainer closes body workers before computing elapsed_seconds. The steady figure subtracts only measured body-worker close time; all CNS, IPC, checkpoint and episode work remains included.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"process_body_overhead_profile={report}")
    print(f"production_checkpoint_unchanged={str(unchanged).lower()}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
