#!/usr/bin/env python3
"""Probe process-parallel FlyBody compound-eye rendering across independent slots.

macOS OpenGL/CGL rendering is not a safe target for the previous Python-thread
probe: a native renderer/context failure can terminate the whole interpreter.
This probe therefore uses multiprocessing with the ``spawn`` start method. Each
child process constructs and owns one complete FlyBody/MuJoCo slot and performs
all rendering from that child process's main thread.

The parent compares static-scene ommatidia readouts bit-for-bit between sequential
and concurrent scheduling and measures aggregate readout throughput for
concurrency 1/2/4/8. If a child dies or hangs in native rendering, the parent
records a FAIL report instead of intentionally propagating the native crash.

No production checkpoint, CNS state, or training state is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback
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
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


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


def _worker_main(connection, slot_index: int) -> None:
    """Own one complete FlyBody slot and renderer inside a spawned process."""

    try:
        # Import and construct inside the child so MuJoCo/CGL objects are never
        # inherited from or created on the parent's rendering thread.
        import train_flyppy_population as trainer

        slot = trainer.make_slot(make_args(), slot_index)
        connection.send(("ready", None))
        while True:
            command = connection.recv()
            if command == "read":
                result = slot.vision._eye_readouts(slot.body.sim, slot.body.fly)
                payload = {
                    side: np.asarray(values).copy()
                    for side, values in result.items()
                }
                connection.send(("ok", payload))
            elif command == "close":
                # Release renderer in the same process/main thread that used it.
                slot.body.sim.eye_renderer = None
                connection.send(("closed", None))
                return
            else:
                raise RuntimeError(f"unknown vision worker command {command!r}")
    except BaseException as exc:
        try:
            connection.send(
                (
                    "error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        except BaseException:
            pass
    finally:
        try:
            connection.close()
        except BaseException:
            pass


class VisionProcess:
    def __init__(self, context, slot_index: int, timeout_s: float) -> None:
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(
            target=_worker_main,
            args=(child, slot_index),
            name=f"flyppy-vision-{slot_index}",
        )
        self.slot_index = slot_index
        self.timeout_s = timeout_s
        self.process.start()
        child.close()
        status, payload = self._recv("startup")
        if status != "ready":
            raise RuntimeError(f"slot {slot_index} failed startup: {status}: {payload}")

    def _recv(self, phase: str):
        if not self.connection.poll(self.timeout_s):
            exitcode = self.process.exitcode
            if exitcode is not None:
                raise RuntimeError(
                    f"slot {self.slot_index} process died during {phase}; exitcode={exitcode}"
                )
            self.process.terminate()
            self.process.join(timeout=5.0)
            raise RuntimeError(
                f"slot {self.slot_index} timed out during {phase} after {self.timeout_s:.1f}s"
            )
        try:
            return self.connection.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError) as exc:
            self.process.join(timeout=1.0)
            raise RuntimeError(
                f"slot {self.slot_index} native process ended during {phase}; "
                f"exitcode={self.process.exitcode}"
            ) from exc

    def request_read(self) -> None:
        if not self.process.is_alive():
            raise RuntimeError(
                f"slot {self.slot_index} process is not alive; exitcode={self.process.exitcode}"
            )
        self.connection.send("read")

    def receive_read(self) -> dict[str, np.ndarray]:
        status, payload = self._recv("render")
        if status == "error":
            raise RuntimeError(f"slot {self.slot_index} worker error: {payload}")
        if status != "ok":
            raise RuntimeError(f"slot {self.slot_index} unexpected response {status!r}")
        return payload

    def read(self) -> dict[str, np.ndarray]:
        self.request_read()
        return self.receive_read()

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.connection.send("close")
                self._recv("close")
            except Exception:
                self.process.terminate()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2.0)
        try:
            self.connection.close()
        except Exception:
            pass


def same_readout(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return all(
        side in a
        and side in b
        and a[side].dtype == b[side].dtype
        and a[side].shape == b[side].shape
        and np.array_equal(a[side], b[side])
        for side in ("L", "R")
    )


def timed_sequential(workers: list[VisionProcess], iterations: int):
    latest = None
    started = time.perf_counter()
    for _ in range(iterations):
        latest = [worker.read() for worker in workers]
    return time.perf_counter() - started, latest


def timed_parallel(workers: list[VisionProcess], iterations: int):
    latest = None
    started = time.perf_counter()
    for _ in range(iterations):
        for worker in workers:
            worker.request_read()
        latest = [worker.receive_read() for worker in workers]
    return time.perf_counter() - started, latest


def main() -> int:
    args = parse_args()
    if args.slots < 1 or args.slots > 8:
        raise SystemExit("--slots must be in 1..8 for this probe")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.timeout_s <= 0.0:
        raise SystemExit("--timeout-s must be positive")

    required = [
        ROOT / "artifacts/malecns-v1.0/retinotopic-vision-v1.json",
        ROOT / "artifacts/malecns-v1.0/wing-motor-neurons-v0.json",
        ROOT / "artifacts/malecns-v1.0/body-motor-neurons-v0.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing parallel-vision probe inputs:\n  " + "\n  ".join(missing))

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    overall_equal = True
    context = mp.get_context("spawn")
    workers: list[VisionProcess] = []
    try:
        for index in range(args.slots):
            workers.append(VisionProcess(context, index, args.timeout_s))

        # Warm renderer + Numba paths sequentially so compilation/initialization is
        # excluded from both scheduling measurements.
        warm = [worker.read() for worker in workers]

        for concurrency in (1, 2, 4, 8):
            if concurrency > args.slots:
                continue
            selected = workers[:concurrency]
            try:
                sequential_s, sequential_latest = timed_sequential(
                    selected, args.iterations
                )
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
            except Exception as exc:
                overall_equal = False
                errors.append(f"N={concurrency}: {type(exc).__name__}: {exc}")
                break
    except Exception as exc:
        overall_equal = False
        errors.append(f"setup/warmup: {type(exc).__name__}: {exc}")
    finally:
        for worker in workers:
            try:
                worker.close()
            except Exception as exc:
                errors.append(
                    f"close slot {worker.slot_index}: {type(exc).__name__}: {exc}"
                )

    report = absolute(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    overall_pass = overall_equal and not errors and bool(rows)
    lines = [
        "# FlyBody process-parallel compound-eye rendering probe",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall_pass else 'FAIL'}",
        "- training performed: false",
        "- production checkpoint touched: false",
        f"- requested slots: {args.slots}",
        f"- measured iterations per scheduling mode: {args.iterations}",
        "- multiprocessing start method: spawn",
        "- ownership: one complete FlyBody/MuJoCo/renderer per child process",
        "- renderer thread: child-process main thread only",
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
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.extend(
        [
            "",
            "Interpretation: process isolation is deliberate on macOS. A child-native rendering failure is reported here rather than moving an OpenGL context across Python threads or terminating the parent probe.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"parallel_vision_probe={report}")
    print(f"parallel_vision_pass={str(overall_pass).lower()}")
    for row in rows:
        print(
            f"N{row['concurrency']}_parallel_speedup={row['speedup']:.6f} "
            f"parallel_readouts_per_second={row['parallel_reads_s']:.6f}"
        )
    for error in errors:
        print(f"parallel_vision_error={error}", file=sys.stderr)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
