#!/usr/bin/env python3
"""Break down end-to-end Flyppy live-population wall time for N=1 and N=8.

The profiler runs only on temporary copies of the production checkpoint and
curriculum. It times the existing Python/CNS boundaries without changing the
numerical path: retina encoding, batched CNS steps, reinforcement CNS steps,
periphery, MuJoCo stepping, commits, and checkpoint load/save.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-live-bottlenecks")
DEFAULT_REPORT = Path("reports/flyppy/population_live_bottlenecks.md")
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
        with item.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def timed(stats: dict[str, list[float]], name: str, fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            stats[name][0] += time.perf_counter() - start
            stats[name][1] += 1
    return wrapper


def run_case(trainer, bridge_module, production: Path, temp_root: Path, population: int, episodes: int, max_steps: int):
    out = temp_root / f"n{population}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(production / "checkpoint", out / "checkpoint")
    shutil.copy2(production / "curriculum-state.json", out / "curriculum-state.json")
    copy_if_exists(production / "population-state.json", out / "population-state.json")

    stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])

    # Patch only call boundaries; each wrapper delegates immediately to the
    # original implementation and therefore does not alter simulation state.
    retina_cls = trainer.MaleCNSRetina
    periphery_cls = trainer.WholeBodyPeriphery
    body_cls = trainer.FlyBodyV3NeuromuscularAdapter
    orig_retina = retina_cls.encode
    orig_periphery = periphery_cls.step
    orig_body = body_cls.step_muscles
    retina_cls.encode = timed(stats, "retina", orig_retina)
    periphery_cls.step = timed(stats, "periphery", orig_periphery)
    body_cls.step_muscles = timed(stats, "physics", orig_body)

    BaseClient = bridge_module.PopulationNeuralBridgeClient

    class ProfilingClient(BaseClient):
        def step_batch(self, *args, **kwargs):
            return timed(stats, "brain_batch", super().step_batch)(*args, **kwargs)
        def step_slot(self, *args, **kwargs):
            return timed(stats, "brain_reinforcement", super().step_slot)(*args, **kwargs)
        def commit_slot(self, *args, **kwargs):
            return timed(stats, "brain_commit", super().commit_slot)(*args, **kwargs)
        def load_checkpoint(self, *args, **kwargs):
            return timed(stats, "brain_checkpoint_load", super().load_checkpoint)(*args, **kwargs)
        def save_checkpoint(self, *args, **kwargs):
            return timed(stats, "brain_checkpoint_save", super().save_checkpoint)(*args, **kwargs)

    old_client = trainer.PopulationNeuralBridgeClient
    trainer.PopulationNeuralBridgeClient = ProfilingClient
    old_argv = sys.argv[:]
    started = time.perf_counter()
    try:
        sys.argv = [
            str(Path(trainer.__file__).resolve()),
            "--episodes", str(episodes),
            "--population", str(population),
            "--output-dir", str(out),
            "--max-control-steps", str(max_steps),
            "--checkpoint-every", str(episodes + 1),
            "--trajectory-stride", "1000000",
            "--curriculum-x-step-mm", "0.000000001",
            "--curriculum-failure-x-step-mm", "0.000000001",
            "--curriculum-z-step-mm", "0.000000001",
            "--curriculum-speed-step-mm-s", "0.000000001",
        ]
        rc = int(trainer.main())
    finally:
        wall = time.perf_counter() - started
        sys.argv = old_argv
        trainer.PopulationNeuralBridgeClient = old_client
        retina_cls.encode = orig_retina
        periphery_cls.step = orig_periphery
        body_cls.step_muscles = orig_body

    if rc != 0:
        raise RuntimeError(f"population={population} trainer returned {rc}")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    controls = int(summary["aggregate_control_steps"])
    return {
        "population": population,
        "wall": wall,
        "trainer_elapsed": float(summary["elapsed_seconds"]),
        "control_steps": controls,
        "control_steps_per_second": controls / float(summary["elapsed_seconds"]),
        "stats": {k: {"seconds": v[0], "calls": int(v[1])} for k, v in stats.items()},
    }


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    temp_root = absolute(args.temp)
    report = absolute(args.report)
    checkpoint = production / "checkpoint"
    required = [
        checkpoint / "manifest.json",
        production / "curriculum-state.json",
        ROOT / DEFAULT_CALIBRATION,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing bottleneck-profile inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))

    before = tree_digest(checkpoint)
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)

    import population_neural_bridge_client as bridge_module
    import train_flyppy_population as trainer

    cases = [
        run_case(trainer, bridge_module, production, temp_root, n, args.episodes, args.max_control_steps)
        for n in (1, 8)
    ]
    after = tree_digest(checkpoint)
    if before != after:
        raise RuntimeError("production checkpoint changed during bottleneck profiling")

    stages = [
        "retina", "brain_batch", "brain_reinforcement", "periphery", "physics",
        "brain_commit", "brain_checkpoint_load", "brain_checkpoint_save",
    ]
    lines = [
        "# Flyppy eager-live population bottleneck profile",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- overall: PASS",
        f"- episodes per case: {args.episodes}",
        f"- max control steps per episode: {args.max_control_steps}",
        "- production checkpoint modified: no",
        f"- production checkpoint digest: `{before}`",
        "",
        "## End-to-end",
        "",
        "| population | control steps | trainer elapsed s | aggregate steps/s | outer wall s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        lines.append(f"| {c['population']} | {c['control_steps']} | {c['trainer_elapsed']:.3f} | {c['control_steps_per_second']:.6f} | {c['wall']:.3f} |")

    lines.extend([
        "",
        "## Timed boundaries",
        "",
        "Times are inclusive wall time at Python/CNS boundaries. Percentages use trainer elapsed and are diagnostic; small uninstrumented Python/course/I/O work appears as remainder.",
        "",
        "| population | stage | seconds | calls | ms/call | ms/control-step | % trainer elapsed |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for c in cases:
        controls = c["control_steps"]
        elapsed = c["trainer_elapsed"]
        for stage in stages:
            s = c["stats"].get(stage, {"seconds": 0.0, "calls": 0})
            sec = float(s["seconds"])
            calls = int(s["calls"])
            lines.append(
                f"| {c['population']} | {stage} | {sec:.3f} | {calls} | "
                f"{(1000*sec/calls if calls else 0):.3f} | {(1000*sec/controls if controls else 0):.3f} | "
                f"{(100*sec/elapsed if elapsed else 0):.2f}% |"
            )

    n1, n8 = cases
    lines.extend([
        "",
        "## Scaling signal",
        "",
        f"- N=8 / N=1 aggregate throughput: {n8['control_steps_per_second']/n1['control_steps_per_second']:.3f}x",
        f"- N=8 / N=1 brain_batch total time for the same aggregate episode budget: {n8['stats'].get('brain_batch', {'seconds':0})['seconds']/max(n1['stats'].get('brain_batch', {'seconds':0})['seconds'], 1e-12):.3f}x",
        f"- N=8 / N=1 retina total time: {n8['stats'].get('retina', {'seconds':0})['seconds']/max(n1['stats'].get('retina', {'seconds':0})['seconds'], 1e-12):.3f}x",
        f"- N=8 / N=1 physics total time: {n8['stats'].get('physics', {'seconds':0})['seconds']/max(n1['stats'].get('physics', {'seconds':0})['seconds'], 1e-12):.3f}x",
        "",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"population_live_bottleneck_profile={report}")
    print("production_checkpoint_modified=false")
    for c in cases:
        print(f"N{c['population']}_aggregate_steps_per_second={c['control_steps_per_second']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
