#!/usr/bin/env python3
"""Profile the packed Flyppy compound-eye render path at the MuJoCo call level.

Each spawned child owns one complete FlyBody slot and its renderer on that
process's main thread, matching the macOS-safe ownership model used by the packed
runtime.  The probe keeps four logical flies active concurrently and splits eye
readout time into:

    MuJoCo Renderer.update_scene
    MuJoCo Renderer.render
    FlyGym fisheye correction
    FlyGym raw-image -> hex/ommatidia aggregation
    remaining Python/allocation overhead

No CNS/checkpoint/training state is modified.
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_REPORT = Path("reports/flyppy/packed_vision_render_stages.md")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


class TimedRenderer:
    def __init__(self, inner, stats: dict[str, float]) -> None:
        self._inner = inner
        self._stats = stats

    def update_scene(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._inner.update_scene(*args, **kwargs)
        finally:
            self._stats["update_scene_s"] += time.perf_counter() - started
            self._stats["update_scene_calls"] += 1.0

    def render(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._inner.render(*args, **kwargs)
        finally:
            self._stats["render_s"] += time.perf_counter() - started
            self._stats["render_calls"] += 1.0

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class TimedRetina:
    def __init__(self, inner, stats: dict[str, float]) -> None:
        self._inner = inner
        self._stats = stats

    def correct_fisheye(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._inner.correct_fisheye(*args, **kwargs)
        finally:
            self._stats["fisheye_s"] += time.perf_counter() - started
            self._stats["fisheye_calls"] += 1.0

    def raw_image_to_hex_pxls(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._inner.raw_image_to_hex_pxls(*args, **kwargs)
        finally:
            self._stats["hex_s"] += time.perf_counter() - started
            self._stats["hex_calls"] += 1.0

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def worker_args(snapshot: Path, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=int(seed),
        gate_count=6,
        wing_motor_map=snapshot / "wing-motor-neurons-v0.json",
        body_motor_map=snapshot / "body-motor-neurons-v0.json",
        retinotopic_map=snapshot / "retinotopic-vision-v1.json",
        photoreceptor_current_gain=2.0,
    )


def checksum_eyes(eyes: dict[str, np.ndarray]) -> tuple[tuple[str, tuple[int, ...], str, bytes], ...]:
    rows = []
    for key in sorted(eyes):
        value = np.ascontiguousarray(eyes[key])
        rows.append((str(key), tuple(int(v) for v in value.shape), str(value.dtype), value.tobytes()))
    return tuple(rows)


def worker_main(connection, slot_index: int, snapshot_text: str, warmup: int) -> None:
    try:
        import train_flyppy_population as trainer

        snapshot = Path(snapshot_text)
        slot = trainer.make_slot(worker_args(snapshot, slot_index), slot_index)
        sim = slot.body.sim

        # Create the lazy renderer and Retina in the same child main thread that
        # will own them for the entire probe.
        baseline = slot.vision._eye_readouts(sim, slot.body.fly)
        baseline_checksum = checksum_eyes(baseline)
        for _ in range(max(0, int(warmup) - 1)):
            slot.vision._eye_readouts(sim, slot.body.fly)

        stats: dict[str, float] = {
            "readout_s": 0.0,
            "readout_calls": 0.0,
            "update_scene_s": 0.0,
            "update_scene_calls": 0.0,
            "render_s": 0.0,
            "render_calls": 0.0,
            "fisheye_s": 0.0,
            "fisheye_calls": 0.0,
            "hex_s": 0.0,
            "hex_calls": 0.0,
        }

        renderer_wrapped = False
        retina_wrapped = False
        if getattr(sim, "eye_renderer", None) is not None:
            sim.eye_renderer = TimedRenderer(sim.eye_renderer, stats)
            renderer_wrapped = True
        if getattr(sim, "retina", None) is not None:
            sim.retina = TimedRetina(sim.retina, stats)
            retina_wrapped = True

        connection.send(
            (
                "ready",
                {
                    "renderer_wrapped": renderer_wrapped,
                    "retina_wrapped": retina_wrapped,
                },
            )
        )

        while True:
            request = connection.recv()
            command = str(request[0])
            if command == "sample":
                count = int(request[1])
                equal = True
                started_batch = time.perf_counter()
                for _ in range(count):
                    started = time.perf_counter()
                    eyes = slot.vision._eye_readouts(sim, slot.body.fly)
                    stats["readout_s"] += time.perf_counter() - started
                    stats["readout_calls"] += 1.0
                    equal = equal and checksum_eyes(eyes) == baseline_checksum
                batch_s = time.perf_counter() - started_batch
                connection.send(
                    (
                        "ok",
                        {
                            "stats": dict(stats),
                            "batch_s": batch_s,
                            "bitwise_static_equal": bool(equal),
                        },
                    )
                )
                continue
            if command == "close":
                # Drop wrappers and renderer on their owner thread/process.
                if isinstance(getattr(sim, "eye_renderer", None), TimedRenderer):
                    sim.eye_renderer = sim.eye_renderer._inner
                sim.eye_renderer = None
                connection.send(("closed", None))
                return
            raise RuntimeError(f"unknown command {command!r}")
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


class Worker:
    def __init__(self, context, slot: int, snapshot: Path, warmup: int, timeout_s: float) -> None:
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.timeout_s = float(timeout_s)
        self.slot = int(slot)
        self.process = context.Process(
            target=worker_main,
            args=(child, self.slot, str(snapshot), int(warmup)),
            name=f"flyppy-render-profile-{slot}",
        )
        self.process.start()
        child.close()
        status, payload = self.recv("startup")
        if status != "ready":
            raise RuntimeError(f"worker {slot} startup failed: {status}: {payload}")
        self.startup = payload

    def recv(self, phase: str):
        if not self.connection.poll(self.timeout_s):
            raise RuntimeError(f"worker {self.slot} timed out during {phase}")
        try:
            return self.connection.recv()
        except EOFError as exc:
            self.process.join(timeout=1.0)
            raise RuntimeError(
                f"worker {self.slot} ended during {phase}; exitcode={self.process.exitcode}"
            ) from exc

    def request(self, command: str, payload=None) -> None:
        self.connection.send((command,) if payload is None else (command, payload))

    def receive_ok(self, phase: str):
        status, payload = self.recv(phase)
        if status == "error":
            raise RuntimeError(f"worker {self.slot} error: {payload}")
        if status not in {"ok", "closed"}:
            raise RuntimeError(f"worker {self.slot} unexpected status {status!r}")
        return payload

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.request("close")
                self.receive_ok("close")
            except Exception:
                self.process.terminate()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2.0)
        self.connection.close()


def main() -> int:
    args = parse_args()
    if args.population < 1 or args.population > 8:
        raise SystemExit("--population must be in 1..8 for this render-stage probe")
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    snapshot = absolute(args.snapshot)
    report = absolute(args.report)
    required = [
        snapshot / "manifest.json",
        snapshot / "retinotopic-vision-v1.json",
        snapshot / "wing-motor-neurons-v0.json",
        snapshot / "body-motor-neurons-v0.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing render-stage inputs:\n  " + "\n  ".join(missing))

    context = mp.get_context("spawn")
    workers: list[Worker] = []
    try:
        for slot in range(args.population):
            workers.append(Worker(context, slot, snapshot, args.warmup, args.timeout_s))

        started = time.perf_counter()
        for worker in workers:
            worker.request("sample", args.samples)
        payloads = [worker.receive_ok("sample") for worker in workers]
        wall_s = time.perf_counter() - started

        totals = {
            key: sum(float(payload["stats"].get(key, 0.0)) for payload in payloads)
            for key in (
                "readout_s", "readout_calls",
                "update_scene_s", "update_scene_calls",
                "render_s", "render_calls",
                "fisheye_s", "fisheye_calls",
                "hex_s", "hex_calls",
            )
        }
        readout_s = totals["readout_s"]
        known_s = (
            totals["update_scene_s"]
            + totals["render_s"]
            + totals["fisheye_s"]
            + totals["hex_s"]
        )
        other_s = max(0.0, readout_s - known_s)
        aggregate_readouts = args.population * args.samples
        static_equal = all(bool(payload["bitwise_static_equal"]) for payload in payloads)
        wrappers_ok = all(
            bool(worker.startup.get("renderer_wrapped"))
            and bool(worker.startup.get("retina_wrapped"))
            for worker in workers
        )
        overall = static_equal and wrappers_ok

        def per_readout(seconds: float) -> float:
            return 1000.0 * seconds / aggregate_readouts

        lines = [
            "# Flyppy packed vision render-stage profile",
            "",
            f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- overall: {'PASS' if overall else 'FAIL'}",
            f"- concurrent body processes: {args.population}",
            f"- samples per process: {args.samples}",
            f"- aggregate eye-readout calls: {aggregate_readouts}",
            f"- parent wall time: {wall_s:.6f} s",
            f"- aggregate readouts/s: {aggregate_readouts / wall_s:.3f}",
            f"- renderer proxy installed in every child: {wrappers_ok}",
            f"- static ommatidia output bitwise stable: {static_equal}",
            "",
            "## Stage totals",
            "",
            "Times below are summed across concurrent child processes. Nested stage times therefore describe work distribution, not parent wall-clock addends.",
            "",
            "| stage | summed seconds | calls | ms/eye-readout | share of readout work |",
            "|---|---:|---:|---:|---:|",
        ]
        stages = [
            ("MuJoCo update_scene", totals["update_scene_s"], int(totals["update_scene_calls"])),
            ("MuJoCo render", totals["render_s"], int(totals["render_calls"])),
            ("fisheye correction", totals["fisheye_s"], int(totals["fisheye_calls"])),
            ("hex/ommatidia aggregation", totals["hex_s"], int(totals["hex_calls"])),
            ("other readout overhead", other_s, aggregate_readouts),
        ]
        for label, seconds, calls in stages:
            share = 100.0 * seconds / readout_s if readout_s else 0.0
            lines.append(
                f"| {label} | {seconds:.6f} | {calls} | {per_readout(seconds):.3f} | {share:.2f}% |"
            )
        lines.extend(
            [
                "",
                "## Per-eye call cost",
                "",
                f"- update_scene: {1000.0 * totals['update_scene_s'] / max(totals['update_scene_calls'], 1.0):.3f} ms/call",
                f"- render: {1000.0 * totals['render_s'] / max(totals['render_calls'], 1.0):.3f} ms/call",
                f"- fisheye: {1000.0 * totals['fisheye_s'] / max(totals['fisheye_calls'], 1.0):.3f} ms/call",
                f"- hex aggregation: {1000.0 * totals['hex_s'] / max(totals['hex_calls'], 1.0):.3f} ms/call",
                "",
                "Each eye-readout call contains the left and right compound-eye paths. The probe changes no physical state and requires bitwise-stable ommatidia output across all timed samples.",
                "",
            ]
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines), encoding="utf-8")
        print(f"packed_vision_render_stages={report}")
        print(f"packed_vision_render_stages_pass={str(overall).lower()}")
        print(f"aggregate_readouts_per_second={aggregate_readouts / wall_s:.6f}")
        print(f"update_scene_ms_per_readout={per_readout(totals['update_scene_s']):.6f}")
        print(f"render_ms_per_readout={per_readout(totals['render_s']):.6f}")
        return 0 if overall else 1
    finally:
        for worker in workers:
            try:
                worker.close()
            except Exception:
                pass


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
