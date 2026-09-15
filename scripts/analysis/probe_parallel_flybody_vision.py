#!/usr/bin/env python3
"""Probe thread-parallel FlyBody compound-eye rendering across independent slots.

Each slot keeps its own FlyBody/MuJoCo Simulation.  A dedicated worker thread owns
that slot's lazily-created eye renderer for the whole probe, so a renderer/context
is never migrated between threads.  The probe compares static-scene ommatidia
readouts bit-for-bit between sequential and concurrent scheduling and measures
aggregate readout throughput for concurrency 1/2/4/8.

No production checkpoint, CNS state, or training state is modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_REPORT = Path("reports/flyppy/parallel_vision_probe.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


class VisionWorker:
    def __init__(self, slot) -> None:
        self.slot = slot
        self.requests: queue.Queue[tuple[str, Future]] = queue.Queue()
        self.thread = threading.Thread(
            target=self._run,
            name=f"flyppy-vision-{slot.slot}",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        while True:
            command, future = self.requests.get()
            try:
                if command == "read":
                    result = self.slot.vision._eye_readouts(
                        self.slot.body.sim, self.slot.body.fly
                    )
                    # Copy while still on the owner thread so later renderer reuse
                    # cannot alias a mutable backing array.
                    future.set_result(
                        {side: np.asarray(values).copy() for side, values in result.items()}
                    )
                elif command == "close":
                    # Destroy the MuJoCo renderer on the same thread that created it.
                    self.slot.body.sim.eye_renderer = None
                    future.set_result(None)
                    return
                else:
                    raise RuntimeError(f"unknown vision worker command {command!r}")
            except BaseException as exc:  # propagate worker/backend failures exactly
                future.set_exception(exc)

    def submit_read(self) -> Future:
        future: Future = Future()
        self.requests.put(("read", future))
        return future

    def close(self) -> None:
        future: Future = Future()
        self.requests.put(("close", future))
        future.result()
        self.thread.join()


def make_args():
    snapshot = ROOT / "artifacts/malecns-v1.0"
    return SimpleNamespace(
        seed=0,
        gate_count=6,
        wing_motor_map=snapshot / "wing-motor-neurons-v0.json",
        body_motor_map=snapshot / "body-motor-neurons-v0.json",
        retinotopic_map=snapshot / "retinotopic-vision-v1.json",
        photoreceptor_current_gain=2.0,
    )


def same_readout(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return all(
        side in a
        and side in b
        and a[side].dtype == b[side].dtype
        and a[side].shape == b[side].shape
        and np.array_equal(a[side], b[side])
        for side in ("L", "R")
    )


def timed_sequential(workers: list[VisionWorker], iterations: int):
    latest = None
    started = time.perf_counter()
    for _ in range(iterations):
        current = []
        for worker in workers:
            current.append(worker.submit_read().result())
        latest = current
    elapsed = time.perf_counter() - started
    return elapsed, latest


def timed_parallel(workers: list[VisionWorker], iterations: int):
    latest = None
    started = time.perf_counter()
    for _ in range(iterations):
        futures = [worker.submit_read() for worker in workers]
        latest = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    return elapsed, latest


def main() -> int:
    args = parse_args()
    if args.slots < 1 or args.slots > 8:
        raise SystemExit("--slots must be in 1..8 for this probe")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")

    required = [
        ROOT / "artifacts/malecns-v1.0/retinotopic-vision-v1.json",
        ROOT / "artifacts/malecns-v1.0/wing-motor-neurons-v0.json",
        ROOT / "artifacts/malecns-v1.0/body-motor-neurons-v0.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing parallel-vision probe inputs:\n  " + "\n  ".join(missing))

    import train_flyppy_population as trainer

    trainer_args = make_args()
    slots = [trainer.make_slot(trainer_args, index) for index in range(args.slots)]
    workers = [VisionWorker(slot) for slot in slots]
    try:
        # Warm every renderer + Numba path before timing.  Renderer construction is
        # deliberately performed on its permanent owner thread.
        warm = [worker.submit_read().result() for worker in workers]

        rows = []
        overall_equal = True
        for concurrency in (1, 2, 4, 8):
            if concurrency > args.slots:
                continue
            selected = workers[:concurrency]
            sequential_s, sequential_latest = timed_sequential(selected, args.iterations)
            parallel_s, parallel_latest = timed_parallel(selected, args.iterations)
            equal = all(
                same_readout(warm[index], sequential_latest[index])
                and same_readout(sequential_latest[index], parallel_latest[index])
                for index in range(concurrency)
            )
            overall_equal = overall_equal and equal
            reads = concurrency * args.iterations
            rows.append(
                {
                    "concurrency": concurrency,
                    "reads": reads,
                    "sequential_s": sequential_s,
                    "parallel_s": parallel_s,
                    "sequential_reads_s": reads / sequential_s,
                    "parallel_reads_s": reads / parallel_s,
                    "speedup": sequential_s / parallel_s,
                    "bitwise_equal": equal,
                }
            )
    finally:
        for worker in workers:
            try:
                worker.close()
            except Exception:
                pass

    report = absolute(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FlyBody parallel compound-eye rendering probe",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall_equal else 'FAIL'}",
        "- training performed: false",
        "- production checkpoint touched: false",
        f"- slots constructed: {args.slots}",
        f"- measured iterations per scheduling mode: {args.iterations}",
        "- renderer ownership: one permanent worker thread per FlyBody slot",
        "- numerical contract: static-scene L/R ommatidia arrays must be bitwise identical",
        "",
        "| concurrency | aggregate eye-readout calls | sequential s | parallel s | sequential readouts/s | parallel readouts/s | speedup | bitwise equal |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {concurrency} | {reads} | {sequential_s:.4f} | {parallel_s:.4f} | "
            "{sequential_reads_s:.2f} | {parallel_reads_s:.2f} | {speedup:.3f}x | {bitwise_equal} |".format(
                **row
            )
        )
    lines.extend([
        "",
        "Interpretation: this probe changes scheduling only. Each independent MuJoCo/FlyBody slot keeps its own physics state and renderer; CNS state and body dynamics are not merged or approximated.",
        "",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"parallel_vision_probe={report}")
    print(f"parallel_vision_bitwise_equal={str(overall_equal).lower()}")
    for row in rows:
        print(
            f"N{row['concurrency']}_parallel_speedup={row['speedup']:.6f} "
            f"parallel_readouts_per_second={row['parallel_reads_s']:.6f}"
        )
    return 0 if overall_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
