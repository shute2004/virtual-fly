#!/usr/bin/env python3
"""Probe process-parallel Flyppy body workers with the real CNS/body boundary.

This is the architectural probe for moving each independent physical fly out of
the trainer process.  Every spawned child owns one complete FlyBody slot:

    observe: body -> compound eye -> MaleCNS retinal currents
    act:     motor spikes -> peripheral state -> MuJoCo physics -> course event

The parent deliberately does *not* run a CNS in this probe.  Instead it feeds a
deterministic synthetic motor-spike schedule between observe and act, allowing us
to compare exactly the same body trajectory under sequential and parallel process
scheduling.  The IPC shape therefore matches the intended production boundary:
retinal currents travel child -> parent and individual motor-neuron spikes travel
parent -> child.

No production checkpoint or training state is read or modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_REPORT = Path("reports/flyppy/process_body_worker_probe.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--physics-steps", type=int, default=10)
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


def _worker_main(connection, slot_index: int, physics_steps: int) -> None:
    try:
        import train_flyppy_population as trainer
        from virtual_fly.training.curriculum import SpawnCondition

        slot = trainer.make_slot(make_args(), slot_index)
        control_dt_s = float(slot.body.timestep) * int(physics_steps)
        body_ids = tuple(int(value) for value in slot.periphery.body_ids)
        condition = SpawnCondition(8.91, 8.91, 400.0)

        def reset() -> dict[str, object]:
            trainer.begin_episode(
                slot,
                episode=0,
                source_weight_version=0,
                condition=condition,
            )
            return {
                "body_ids": body_ids,
                "control_dt_s": control_dt_s,
            }

        connection.send(("ready", reset()))
        while True:
            request = connection.recv()
            command = str(request[0])
            if command == "reset":
                connection.send(("ok", reset()))
                continue

            if command == "observe":
                retinal = slot.vision.encode(slot.body.sim, slot.body.fly)
                payload = {
                    "body_currents": retinal.body_currents,
                    "active_photoreceptors": int(retinal.active_photoreceptors),
                    "active_columns": int(retinal.active_columns),
                    "mean_current": float(retinal.mean_current),
                    "max_current": float(retinal.max_current),
                }
                connection.send(("ok", payload))
                continue

            if command == "act":
                active_ids = frozenset(int(value) for value in request[1])
                spikes = {body_id: body_id in active_ids for body_id in body_ids}
                peripheral = slot.periphery.step(spikes, dt_s=control_dt_s)
                slot.body.step_muscles(peripheral, physics_steps=physics_steps)

                position = slot.body.thorax_position_mm()
                velocity = slot.body.root_linear_velocity_mm_s()
                physical_collision = slot.world.physical_collision_reason(slot.body.sim)
                event = slot.course.update(
                    float(position[0]),
                    float(position[2]),
                    physical_collision_reason=physical_collision,
                    analytic_body_collision=False,
                )
                payload = {
                    "position": tuple(float(value) for value in position),
                    "velocity": tuple(float(value) for value in velocity),
                    "passed_gate": bool(event.passed_gate),
                    "collision": bool(event.collision),
                    "collision_reason": event.collision_reason,
                    "finished": bool(event.finished),
                    "next_gate": int(
                        getattr(
                            slot.course,
                            "absolute_next_gate_index",
                            slot.course.next_gate_index,
                        )
                    ),
                    "motor": peripheral.compact_diagnostics(),
                }
                connection.send(("ok", payload))
                continue

            if command == "close":
                slot.body.sim.eye_renderer = None
                connection.send(("closed", None))
                return

            raise RuntimeError(f"unknown body-worker command {command!r}")
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


class BodyProcess:
    def __init__(self, context, slot_index: int, physics_steps: int, timeout_s: float):
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(
            target=_worker_main,
            args=(child, slot_index, physics_steps),
            name=f"flyppy-body-{slot_index}",
        )
        self.slot_index = slot_index
        self.timeout_s = timeout_s
        self.process.start()
        child.close()
        status, payload = self._recv("startup")
        if status != "ready":
            raise RuntimeError(f"slot {slot_index} failed startup: {status}: {payload}")
        self.body_ids = tuple(int(value) for value in payload["body_ids"])
        self.control_dt_s = float(payload["control_dt_s"])

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

    def request(self, command: str, payload=None) -> None:
        if not self.process.is_alive():
            raise RuntimeError(
                f"slot {self.slot_index} process is not alive; exitcode={self.process.exitcode}"
            )
        if payload is None:
            self.connection.send((command,))
        else:
            self.connection.send((command, payload))

    def receive(self, phase: str):
        status, payload = self._recv(phase)
        if status == "error":
            raise RuntimeError(f"slot {self.slot_index} worker error: {payload}")
        if status not in {"ok", "closed"}:
            raise RuntimeError(f"slot {self.slot_index} unexpected response {status!r}")
        return payload

    def call(self, command: str, payload=None):
        self.request(command, payload)
        return self.receive(command)

    def reset(self) -> None:
        self.call("reset")

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.request("close")
                self.receive("close")
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


def active_ids_for_step(body_ids: tuple[int, ...], step: int) -> tuple[int, ...]:
    """Deterministic individual-MN activity used only as probe input."""

    if step % 5 == 4:
        return ()
    width = min(3, len(body_ids))
    return tuple(body_ids[(step * 7 + offset * 11) % len(body_ids)] for offset in range(width))


def digest_update(hasher, slot_index: int, step: int, observe, act) -> None:
    # Pickle protocol 5 is deterministic here because both paths run the same Python
    # code and preserve dictionary insertion order.  It also preserves exact IEEE
    # float payloads instead of string-rounding them.
    hasher.update(slot_index.to_bytes(4, "little", signed=False))
    hasher.update(step.to_bytes(4, "little", signed=False))
    hasher.update(pickle.dumps((observe, act), protocol=5))


def reset_all(workers: list[BodyProcess]) -> None:
    for worker in workers:
        worker.reset()


def run_sequential(workers: list[BodyProcess], steps: int):
    transcript = [hashlib.sha256() for _ in workers]
    started = time.perf_counter()
    for step in range(steps):
        for index, worker in enumerate(workers):
            observe = worker.call("observe")
            active = active_ids_for_step(worker.body_ids, step)
            act = worker.call("act", active)
            digest_update(transcript[index], worker.slot_index, step, observe, act)
    elapsed = time.perf_counter() - started
    return elapsed, [h.hexdigest() for h in transcript]


def run_parallel(workers: list[BodyProcess], steps: int):
    transcript = [hashlib.sha256() for _ in workers]
    started = time.perf_counter()
    for step in range(steps):
        for worker in workers:
            worker.request("observe")
        observations = [worker.receive("observe") for worker in workers]

        for worker in workers:
            worker.request("act", active_ids_for_step(worker.body_ids, step))
        acts = [worker.receive("act") for worker in workers]

        for index, worker in enumerate(workers):
            digest_update(
                transcript[index],
                worker.slot_index,
                step,
                observations[index],
                acts[index],
            )
    elapsed = time.perf_counter() - started
    return elapsed, [h.hexdigest() for h in transcript]


def main() -> int:
    args = parse_args()
    if args.slots < 1 or args.slots > 8:
        raise SystemExit("--slots must be in 1..8 for this probe")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.physics_steps < 1:
        raise SystemExit("--physics-steps must be >= 1")
    if args.timeout_s <= 0.0:
        raise SystemExit("--timeout-s must be positive")

    required = [
        ROOT / "artifacts/malecns-v1.0/retinotopic-vision-v1.json",
        ROOT / "artifacts/malecns-v1.0/wing-motor-neurons-v0.json",
        ROOT / "artifacts/malecns-v1.0/body-motor-neurons-v0.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing body-worker probe inputs:\n  " + "\n  ".join(missing))

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    context = mp.get_context("spawn")
    workers: list[BodyProcess] = []
    try:
        for index in range(args.slots):
            workers.append(BodyProcess(context, index, args.physics_steps, args.timeout_s))

        # Warm all renderer/Numba paths outside timed regions.
        for worker in workers:
            worker.call("observe")
        reset_all(workers)

        for concurrency in (1, 2, 4, 8):
            if concurrency > args.slots:
                continue
            selected = workers[:concurrency]
            try:
                reset_all(selected)
                sequential_s, sequential_digest = run_sequential(selected, args.steps)
                reset_all(selected)
                parallel_s, parallel_digest = run_parallel(selected, args.steps)
                equal = sequential_digest == parallel_digest
                control_steps = concurrency * args.steps
                rows.append(
                    {
                        "concurrency": concurrency,
                        "control_steps": control_steps,
                        "sequential_s": sequential_s,
                        "parallel_s": parallel_s,
                        "sequential_steps_s": control_steps / sequential_s,
                        "parallel_steps_s": control_steps / parallel_s,
                        "speedup": sequential_s / parallel_s,
                        "trajectory_equal": equal,
                    }
                )
                if not equal:
                    errors.append(f"N={concurrency}: sequential/parallel transcript mismatch")
                    break
            except Exception as exc:
                errors.append(f"N={concurrency}: {type(exc).__name__}: {exc}")
                break
    except Exception as exc:
        errors.append(f"setup/warmup: {type(exc).__name__}: {exc}")
    finally:
        for worker in workers:
            try:
                worker.close()
            except Exception as exc:
                errors.append(
                    f"close slot {worker.slot_index}: {type(exc).__name__}: {exc}"
                )

    overall_pass = bool(rows) and not errors and all(
        bool(row["trajectory_equal"]) for row in rows
    )
    report = absolute(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Flyppy process-isolated body-worker probe",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall_pass else 'FAIL'}",
        "- training performed: false",
        "- production checkpoint touched: false",
        f"- requested slots: {args.slots}",
        f"- control steps per slot and scheduling mode: {args.steps}",
        f"- physics steps per control step: {args.physics_steps}",
        "- multiprocessing start method: spawn",
        "- child ownership: FlyBody + MuJoCo + eye renderer + MaleCNS retina seam + periphery + Flyppy course",
        "- parent/child boundary: retinal body currents out; individual motor-neuron spikes in",
        "- numerical contract: complete observe+act transcript digest must match between sequential and parallel scheduling",
        "",
        "| concurrency | aggregate control steps | sequential s | parallel s | sequential steps/s | parallel steps/s | body-side speedup | trajectory bitwise equal |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {concurrency} | {control_steps} | {sequential_s:.4f} | {parallel_s:.4f} | "
            "{sequential_steps_s:.2f} | {parallel_steps_s:.2f} | {speedup:.3f}x | {trajectory_equal} |".format(
                **row
            )
        )
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.extend(
        [
            "",
            "Interpretation: this probe measures only the body-side process boundary. It intentionally excludes CNS compute; production can place the shared GPU CNS batch between the observe and act phases.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"process_body_worker_probe={report}")
    print(f"process_body_worker_pass={str(overall_pass).lower()}")
    for row in rows:
        print(
            f"N{row['concurrency']}_body_parallel_speedup={row['speedup']:.6f} "
            f"parallel_control_steps_per_second={row['parallel_steps_s']:.6f}"
        )
    for error in errors:
        print(f"process_body_worker_error={error}", file=sys.stderr)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
