#!/usr/bin/env python3
"""Probe packing multiple Flyppy bodies into fewer spawned worker processes.

The current process-body prototype uses one Python/MuJoCo process per fly. On the
M1 unified-memory machine, body-only N=8 scales, but N=8 degrades once the large
shared GPU CNS is present. This probe keeps population=8 fixed while changing the
number of body processes: 1, 2, 4, or 8. Each worker owns several fully independent
FlyBody slots and executes those slots sequentially on its main thread; different
workers run concurrently.

Every packing starts from the same private copy of the production CNS checkpoint.
Final checkpoint bytes and episode/commit outcomes must match across all packings.
Production state is never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-body-worker-packing")
DEFAULT_REPORT = Path("reports/flyppy/body_worker_packing_probe.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--max-control-steps", type=int, default=64)
    p.add_argument("--physics-steps", type=int, default=10)
    p.add_argument("--timeout-s", type=float, default=120.0)
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


def clone_checkpoint(production: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(production / "checkpoint", out / "checkpoint")


def make_worker_args(config: dict[str, Any], slot_index: int) -> SimpleNamespace:
    snapshot = Path(config["snapshot"])
    return SimpleNamespace(
        seed=int(config["seed"]),
        gate_count=int(config["gate_count"]),
        wing_motor_map=snapshot / "wing-motor-neurons-v0.json",
        body_motor_map=snapshot / "body-motor-neurons-v0.json",
        retinotopic_map=snapshot / "retinotopic-vision-v1.json",
        photoreceptor_current_gain=float(config["photoreceptor_current_gain"]),
    )


def worker_main(connection, slot_ids: tuple[int, ...], physics_steps: int, config: dict[str, Any]) -> None:
    try:
        import train_flyppy_population as trainer
        from virtual_fly.training.curriculum import SpawnCondition

        slots = {
            slot_id: trainer.make_slot(make_worker_args(config, slot_id), slot_id)
            for slot_id in slot_ids
        }
        body_ids = {
            slot_id: tuple(int(value) for value in slot.periphery.body_ids)
            for slot_id, slot in slots.items()
        }
        control_dts = {
            round(float(slot.body.timestep) * int(physics_steps), 15)
            for slot in slots.values()
        }
        if len(control_dts) != 1:
            raise RuntimeError(f"packed slots disagree on control dt: {control_dts}")
        control_dt_s = float(next(iter(control_dts)))
        connection.send(("ready", {"body_ids": body_ids, "control_dt_s": control_dt_s}))

        while True:
            request = connection.recv()
            command = str(request[0])
            payload = request[1] if len(request) > 1 else None

            if command == "reset":
                replies = {}
                for raw_id, reset_payload in dict(payload).items():
                    slot_id = int(raw_id)
                    slot = slots[slot_id]
                    condition = SpawnCondition(
                        float(reset_payload["spawn_x_mm"]),
                        float(reset_payload["spawn_z_mm"]),
                        float(reset_payload["initial_speed_mm_s"]),
                    )
                    trainer.begin_episode(
                        slot,
                        episode=int(reset_payload["episode"]),
                        source_weight_version=int(reset_payload["source_weight_version"]),
                        condition=condition,
                    )
                    replies[slot_id] = {
                        "velocity": tuple(float(v) for v in slot.body.root_linear_velocity_mm_s())
                    }
                connection.send(("ok", replies))
                continue

            if command == "observe":
                replies = {}
                for slot_id in tuple(int(v) for v in payload):
                    slot = slots[slot_id]
                    retinal = slot.vision.encode(slot.body.sim, slot.body.fly)
                    replies[slot_id] = {
                        "body_currents": retinal.body_currents,
                        "active_photoreceptors": int(retinal.active_photoreceptors),
                        "active_columns": int(retinal.active_columns),
                        "mean_current": float(retinal.mean_current),
                        "max_current": float(retinal.max_current),
                    }
                connection.send(("ok", replies))
                continue

            if command == "act":
                replies = {}
                for raw_id, active_values in dict(payload).items():
                    slot_id = int(raw_id)
                    slot = slots[slot_id]
                    active_ids = frozenset(int(v) for v in active_values)
                    spikes = {body_id: body_id in active_ids for body_id in body_ids[slot_id]}
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
                    replies[slot_id] = {
                        "position": tuple(float(v) for v in position),
                        "velocity": tuple(float(v) for v in velocity),
                        "passed_gate": bool(event.passed_gate),
                        "collision": bool(event.collision),
                        "collision_reason": event.collision_reason,
                        "finished": bool(event.finished),
                    }
                connection.send(("ok", replies))
                continue

            if command == "close":
                for slot in slots.values():
                    slot.body.sim.eye_renderer = None
                connection.send(("closed", None))
                return

            raise RuntimeError(f"unknown packed worker command {command!r}")
    except BaseException as exc:
        try:
            connection.send(("error", {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }))
        except BaseException:
            pass
    finally:
        try:
            connection.close()
        except BaseException:
            pass


class PackedWorker:
    def __init__(self, context, slot_ids: tuple[int, ...], physics_steps: int, timeout_s: float, config: dict[str, Any]):
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.slot_ids = tuple(slot_ids)
        self.timeout_s = float(timeout_s)
        self.process = context.Process(
            target=worker_main,
            args=(child, self.slot_ids, physics_steps, dict(config)),
            name="flyppy-packed-" + "-".join(str(v) for v in self.slot_ids),
        )
        self.process.start()
        child.close()
        status, payload = self._recv("startup")
        if status != "ready":
            raise RuntimeError(f"packed worker {self.slot_ids} startup failed: {status}: {payload}")
        self.body_ids = {int(k): tuple(int(v) for v in values) for k, values in payload["body_ids"].items()}
        self.control_dt_s = float(payload["control_dt_s"])

    def _recv(self, phase: str):
        if not self.connection.poll(self.timeout_s):
            if self.process.exitcode is not None:
                raise RuntimeError(f"packed worker {self.slot_ids} died during {phase}; exitcode={self.process.exitcode}")
            self.process.terminate()
            self.process.join(timeout=5.0)
            raise RuntimeError(f"packed worker {self.slot_ids} timed out during {phase}")
        try:
            return self.connection.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError) as exc:
            self.process.join(timeout=1.0)
            raise RuntimeError(
                f"packed worker {self.slot_ids} native process ended during {phase}; exitcode={self.process.exitcode}"
            ) from exc

    def request(self, command: str, payload=None) -> None:
        self.connection.send((command,) if payload is None else (command, payload))

    def receive(self, phase: str):
        status, payload = self._recv(phase)
        if status == "error":
            raise RuntimeError(f"packed worker {self.slot_ids} error: {payload}")
        if status not in {"ok", "closed"}:
            raise RuntimeError(f"packed worker {self.slot_ids} unexpected status {status!r}")
        return payload

    def call(self, command: str, payload=None):
        self.request(command, payload)
        return self.receive(command)

    def close(self) -> float:
        started = time.perf_counter()
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
        return time.perf_counter() - started


def assignments(population: int, process_count: int) -> list[tuple[int, ...]]:
    groups = [[] for _ in range(process_count)]
    for slot_id in range(population):
        groups[slot_id % process_count].append(slot_id)
    return [tuple(group) for group in groups if group]


def run_case(
    *,
    out: Path,
    production: Path,
    population: int,
    process_count: int,
    max_steps: int,
    physics_steps: int,
    timeout_s: float,
    condition,
    initial_global_version: int,
) -> dict[str, Any]:
    from population_neural_bridge_client import PopulationNeuralBridgeClient

    clone_checkpoint(production, out)
    snapshot = ROOT / "artifacts/malecns-v1.0"
    config = {
        "snapshot": str(snapshot),
        "seed": 0,
        "gate_count": 6,
        "photoreceptor_current_gain": 2.0,
    }
    context = mp.get_context("spawn")
    workers: list[PackedWorker] = []
    startup_started = time.perf_counter()
    try:
        for group in assignments(population, process_count):
            workers.append(PackedWorker(context, group, physics_steps, timeout_s, config))
        startup_s = time.perf_counter() - startup_started

        control_dts = {round(worker.control_dt_s, 15) for worker in workers}
        if len(control_dts) != 1:
            raise RuntimeError(f"workers disagree on control dt: {control_dts}")

        slots = {
            slot_id: {
                "active": True,
                "source_weight_version": initial_global_version,
                "control_steps": 0,
                "passed_gates": 0,
                "collision": False,
                "collision_reason": None,
                "finished": False,
                "max_x_mm": float("-inf"),
                "min_z_mm": float("inf"),
                "max_z_mm": float("-inf"),
                "final_vx_mm_s": 0.0,
                "reward_events": 0,
                "aversive_events": 0,
            }
            for slot_id in range(population)
        }
        slot_to_worker = {slot_id: worker for worker in workers for slot_id in worker.slot_ids}

        observe_s = 0.0
        cns_s = 0.0
        act_s = 0.0
        reinforce_s = 0.0
        commit_s = 0.0
        checkpoint_s = 0.0
        aggregate_steps = 0
        commit_records = []

        with PopulationNeuralBridgeClient(
            snapshot=snapshot,
            groups=snapshot / "embodiment-groups-v0.json",
            slots=population,
        ) as brain:
            brain.ping()
            brain.load_checkpoint(out / "checkpoint", global_weight_version=initial_global_version)
            reset_by_worker: dict[PackedWorker, dict[int, dict[str, Any]]] = {worker: {} for worker in workers}
            for slot_id in range(population):
                slots[slot_id]["source_weight_version"] = brain.global_weight_version
                reset_by_worker[slot_to_worker[slot_id]][slot_id] = {
                    "episode": slot_id,
                    "source_weight_version": brain.global_weight_version,
                    "spawn_x_mm": float(condition.x_mm),
                    "spawn_z_mm": float(condition.z_mm),
                    "initial_speed_mm_s": float(condition.speed_mm_s),
                }
            for worker, payload in reset_by_worker.items():
                worker.request("reset", payload)
            for worker in workers:
                reply = worker.receive("reset")
                for slot_id, item in reply.items():
                    slots[int(slot_id)]["final_vx_mm_s"] = float(item["velocity"][0])

            loop_started = time.perf_counter()
            completed = 0
            while completed < population:
                active_ids = [slot_id for slot_id, state in slots.items() if state["active"]]
                if not active_ids:
                    raise RuntimeError("no active slots before completion")

                by_worker = {
                    worker: tuple(slot_id for slot_id in worker.slot_ids if slot_id in active_ids)
                    for worker in workers
                }
                t = time.perf_counter()
                for worker, ids in by_worker.items():
                    if ids:
                        worker.request("observe", ids)
                observations = {}
                for worker, ids in by_worker.items():
                    if ids:
                        observations.update({int(k): v for k, v in worker.receive("observe").items()})
                observe_s += time.perf_counter() - t

                requests = []
                for slot_id in active_ids:
                    requests.append({
                        "slot": slot_id,
                        "stimulate_body": observations[slot_id]["body_currents"],
                        "read_body": slot_to_worker[slot_id].body_ids[slot_id],
                    })
                t = time.perf_counter()
                batch_spikes = brain.step_batch(requests, plasticity=True)
                cns_s += time.perf_counter() - t

                act_by_worker: dict[PackedWorker, dict[int, tuple[int, ...]]] = {worker: {} for worker in workers}
                for slot_id in active_ids:
                    spikes = batch_spikes.get(slot_id, {})
                    act_by_worker[slot_to_worker[slot_id]][slot_id] = tuple(
                        int(body_id) for body_id, fired in spikes.items() if bool(fired)
                    )
                t = time.perf_counter()
                for worker, payload in act_by_worker.items():
                    if payload:
                        worker.request("act", payload)
                acts = {}
                for worker, payload in act_by_worker.items():
                    if payload:
                        acts.update({int(k): v for k, v in worker.receive("act").items()})
                act_s += time.perf_counter() - t

                terminal = []
                for slot_id in active_ids:
                    state = slots[slot_id]
                    act = acts[slot_id]
                    position = act["position"]
                    velocity = act["velocity"]
                    x_mm = float(position[0])
                    z_mm = float(position[2])
                    state["max_x_mm"] = max(float(state["max_x_mm"]), x_mm)
                    state["min_z_mm"] = min(float(state["min_z_mm"]), z_mm)
                    state["max_z_mm"] = max(float(state["max_z_mm"]), z_mm)
                    state["final_vx_mm_s"] = float(velocity[0])
                    state["control_steps"] = int(state["control_steps"]) + 1
                    aggregate_steps += 1

                    t = time.perf_counter()
                    if bool(act["passed_gate"]):
                        state["passed_gates"] = int(state["passed_gates"]) + 1
                        state["reward_events"] = int(state["reward_events"]) + 1
                        brain.step_slot(slot_id, stimulate={"reward_dan": 2.0}, plasticity=True, steps=4)
                    if bool(act["collision"]):
                        state["collision"] = True
                        state["collision_reason"] = act["collision_reason"]
                        state["aversive_events"] = int(state["aversive_events"]) + 1
                        brain.step_slot(slot_id, stimulate={"aversive_dan": 2.0}, plasticity=True, steps=4)
                    reinforce_s += time.perf_counter() - t
                    if bool(act["finished"]):
                        state["finished"] = True
                    if state["collision"] or state["finished"] or int(state["control_steps"]) >= max_steps:
                        terminal.append(slot_id)

                for slot_id in sorted(terminal):
                    state = slots[slot_id]
                    t = time.perf_counter()
                    commit = brain.commit_slot(slot_id, source_weight_version=int(state["source_weight_version"]))
                    commit_s += time.perf_counter() - t
                    state["active"] = False
                    completed += 1
                    commit_records.append({
                        "slot": slot_id,
                        "source_weight_version": int(commit["source_weight_version"]),
                        "commit_from_version": int(commit["commit_from_version"]),
                        "commit_weight_version": int(commit["commit_weight_version"]),
                        "staleness": int(commit["staleness"]),
                    })

            loop_s = time.perf_counter() - loop_started
            t = time.perf_counter()
            saved = brain.save_checkpoint(out / "checkpoint")
            checkpoint_s = time.perf_counter() - t
            final_version = brain.global_weight_version

        results = []
        for slot_id in range(population):
            state = slots[slot_id]
            commit = next(row for row in commit_records if row["slot"] == slot_id)
            results.append({
                "slot": slot_id,
                "source_weight_version": int(state["source_weight_version"]),
                "commit_from_version": int(commit["commit_from_version"]),
                "commit_weight_version": int(commit["commit_weight_version"]),
                "staleness": int(commit["staleness"]),
                "control_steps": int(state["control_steps"]),
                "passed_gates": int(state["passed_gates"]),
                "collision": bool(state["collision"]),
                "collision_reason": state["collision_reason"],
                "finished": bool(state["finished"]),
                "max_x_mm": float(state["max_x_mm"]),
                "min_z_mm": float(state["min_z_mm"]),
                "max_z_mm": float(state["max_z_mm"]),
                "final_vx_mm_s": float(state["final_vx_mm_s"]),
                "reward_events": int(state["reward_events"]),
                "aversive_events": int(state["aversive_events"]),
            })

        return {
            "process_count": process_count,
            "slots_per_process": population / process_count,
            "aggregate_steps": aggregate_steps,
            "startup_s": startup_s,
            "loop_s": loop_s,
            "steady_steps_s": aggregate_steps / loop_s,
            "observe_s": observe_s,
            "cns_s": cns_s,
            "act_s": act_s,
            "reinforce_s": reinforce_s,
            "commit_s": commit_s,
            "checkpoint_s": checkpoint_s,
            "checkpoint_digest": tree_digest(out / "checkpoint"),
            "results": results,
            "final_version": final_version,
            "neural_step": int(saved.get("step", -1)),
        }
    finally:
        teardown_s = 0.0
        for worker in workers:
            try:
                teardown_s += worker.close()
            except Exception:
                pass
        # teardown is deliberately excluded from returned steady-state timing.


def main() -> int:
    args = parse_args()
    if args.population != 8:
        raise SystemExit("this probe currently requires --population 8")
    if args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("step counts must be positive")

    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    required = [
        production / "checkpoint/manifest.json",
        production / "curriculum-state.json",
        ROOT / DEFAULT_CALIBRATION,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing packing probe inputs:\n  " + "\n  ".join(missing))

    from flyppy_course import FlyppyCourse
    from virtual_fly.training.curriculum import SpawnCondition, current_adaptive_condition, load_state

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    population_state_path = production / "population-state.json"
    population_state = json.loads(population_state_path.read_text(encoding="utf-8")) if population_state_path.exists() else {}
    initial_global_version = int(population_state.get("global_weight_version", 0))

    course = FlyppyCourse(seed=0, gate_count=6, environment_version="v3")
    first_gate = course.gates[0]
    state = load_state(
        production / "curriculum-state.json",
        start=SpawnCondition(8.91, float(first_gate.center_z_mm), 400.0),
        target=SpawnCondition(0.0, 8.91, 300.0),
        checkpoint_exists=True,
    )
    condition = current_adaptive_condition(state)

    before = tree_digest(production / "checkpoint")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    rows = []
    reference = None
    all_equal = True
    for process_count in (1, 2, 4, 8):
        row = run_case(
            out=temp / f"p{process_count}",
            production=production,
            population=args.population,
            process_count=process_count,
            max_steps=args.max_control_steps,
            physics_steps=args.physics_steps,
            timeout_s=args.timeout_s,
            condition=condition,
            initial_global_version=initial_global_version,
        )
        if reference is None:
            reference = row
            equal = True
        else:
            equal = (
                row["checkpoint_digest"] == reference["checkpoint_digest"]
                and row["results"] == reference["results"]
                and row["final_version"] == reference["final_version"]
                and row["neural_step"] == reference["neural_step"]
            )
        row["equal"] = equal
        all_equal = all_equal and equal
        rows.append(row)

    after = tree_digest(production / "checkpoint")
    unchanged = before == after
    overall = all_equal and unchanged
    best = max(rows, key=lambda row: float(row["steady_steps_s"]))

    lines = [
        "# Flyppy packed body-worker probe",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        f"- population: {args.population}",
        f"- max control steps: {args.max_control_steps}",
        "- CNS: real shared GPU MaleCNS runtime",
        "- production checkpoint modified: " + ("no" if unchanged else "YES"),
        f"- production checkpoint digest: `{before}`",
        f"- best process count: {best['process_count']}",
        f"- best steady throughput: {best['steady_steps_s']:.3f} steps/s",
        "",
        "| body processes | flies/process | aggregate steps | steady steps/s | observe s | CNS s | act s | startup s | checkpoint equal/outcomes equal |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {process_count} | {slots_per_process:.1f} | {aggregate_steps} | {steady_steps_s:.3f} | "
            "{observe_s:.3f} | {cns_s:.3f} | {act_s:.3f} | {startup_s:.3f} | {equal} |".format(**row)
        )
    lines.extend([
        "",
        "Interpretation: population remains fixed at eight independent flies. Only the number of Python/MuJoCo owner processes changes. This directly tests whether one-process-per-fly resource duplication is responsible for the N=8 regression seen with the shared CNS present.",
        "",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"body_worker_packing_probe={report}")
    print(f"body_worker_packing_pass={str(overall).lower()}")
    print(f"best_process_count={best['process_count']}")
    print(f"best_steady_steps_per_second={best['steady_steps_s']:.6f}")
    return 0 if overall else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
