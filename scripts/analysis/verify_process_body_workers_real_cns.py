#!/usr/bin/env python3
"""Verify process-isolated Flyppy bodies against the current local-body trainer.

Both paths start from independent copies of the same production CNS checkpoint,
curriculum state and population state.  The reference path is the existing
train_flyppy_population.py.  The candidate path keeps the shared GPU CNS in the
parent but moves each physical fly into a spawned FlyppyBodyProcess.

The verification requires matching final CNS checkpoint bytes and matching
per-episode/commit outcomes.  Production state is never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-process-body-real-cns")
DEFAULT_REPORT = Path("reports/flyppy/process_body_real_cns_verify.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--max-control-steps", type=int, default=32)
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


def clone_state(production: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(production / "checkpoint", out / "checkpoint")
    shutil.copy2(production / "curriculum-state.json", out / "curriculum-state.json")
    population = production / "population-state.json"
    if population.exists():
        shutil.copy2(population, out / "population-state.json")


def run_reference(out: Path, population: int, max_steps: int, physics_steps: int) -> dict:
    command = [
        sys.executable,
        str(EMBODIMENT / "train_flyppy_population.py"),
        "--episodes", str(population),
        "--population", str(population),
        "--output-dir", str(out),
        "--max-control-steps", str(max_steps),
        "--physics-steps", str(physics_steps),
        "--checkpoint-every", str(population + 1000),
        "--trajectory-stride", "1",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(f"reference trainer failed ({completed.returncode}):\n{tail}")
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def run_process_candidate(
    out: Path,
    population: int,
    max_steps: int,
    physics_steps: int,
    timeout_s: float,
) -> dict:
    import train_flyppy_population as trainer
    from flyppy_body_worker import spawn_body_processes
    from flyppy_course import FlyppyCourse
    from population_neural_bridge_client import PopulationNeuralBridgeClient
    from virtual_fly.training.curriculum import SpawnCondition, load_state, current_adaptive_condition

    snapshot = ROOT / "artifacts/malecns-v1.0"
    groups = snapshot / "embodiment-groups-v0.json"
    course = FlyppyCourse(seed=0, gate_count=6, environment_version="v3")
    first_gate = course.gates[0]
    start = SpawnCondition(8.91, float(first_gate.center_z_mm), 400.0)
    target = SpawnCondition(0.0, 8.91, 300.0)
    state = load_state(
        out / "curriculum-state.json",
        start=start,
        target=target,
        checkpoint_exists=True,
    )
    condition = current_adaptive_condition(state)

    population_state_path = out / "population-state.json"
    population_state = (
        json.loads(population_state_path.read_text(encoding="utf-8"))
        if population_state_path.exists()
        else {}
    )
    initial_global_version = int(population_state.get("global_weight_version", 0))

    config = {
        "seed": 0,
        "gate_count": 6,
        "wing_motor_map": str(snapshot / "wing-motor-neurons-v0.json"),
        "body_motor_map": str(snapshot / "body-motor-neurons-v0.json"),
        "retinotopic_map": str(snapshot / "retinotopic-vision-v1.json"),
        "photoreceptor_current_gain": 2.0,
    }
    workers = spawn_body_processes(
        population=population,
        physics_steps=physics_steps,
        timeout_s=timeout_s,
        config=config,
    )
    try:
        control_dt_values = {round(worker.control_dt_s, 15) for worker in workers}
        if len(control_dt_values) != 1:
            raise RuntimeError(f"body workers disagree on control dt: {control_dt_values}")

        slots = {
            worker.slot_index: {
                "active": True,
                "episode": worker.slot_index,
                "source_weight_version": initial_global_version,
                "control_steps": 0,
                "passed_gates": 0,
                "collision": False,
                "collision_reason": None,
                "finished": False,
                "max_x_mm": float("-inf"),
                "min_z_mm": float("inf"),
                "max_z_mm": float("-inf"),
                "final_velocity": (0.0, 0.0, 0.0),
                "reward_events": 0,
                "aversive_events": 0,
            }
            for worker in workers
        }

        with PopulationNeuralBridgeClient(
            snapshot=snapshot,
            groups=groups,
            slots=population,
        ) as brain:
            brain.ping()
            brain.load_checkpoint(
                out / "checkpoint",
                global_weight_version=initial_global_version,
            )
            # Use the actual loaded version as the episode source, matching the
            # reference trainer's begin_episode timing.
            for worker in workers:
                slot = slots[worker.slot_index]
                slot["source_weight_version"] = brain.global_weight_version
                reset = worker.reset(
                    episode=slot["episode"],
                    source_weight_version=brain.global_weight_version,
                    spawn_x_mm=float(condition.x_mm),
                    spawn_z_mm=float(condition.z_mm),
                    initial_speed_mm_s=float(condition.speed_mm_s),
                )
                slot["final_velocity"] = tuple(reset["velocity"])

            completed = 0
            commit_records: list[dict[str, object]] = []
            while completed < population:
                active_workers = [
                    worker for worker in workers if bool(slots[worker.slot_index]["active"])
                ]
                if not active_workers:
                    raise RuntimeError("candidate has no active body workers before completion")

                for worker in active_workers:
                    worker.request("observe")
                observations = {
                    worker.slot_index: worker.receive("observe") for worker in active_workers
                }

                requests = [
                    {
                        "slot": worker.slot_index,
                        "stimulate_body": observations[worker.slot_index]["body_currents"],
                        "read_body": worker.body_ids,
                    }
                    for worker in active_workers
                ]
                batch_spikes = brain.step_batch(requests, plasticity=True)

                for worker in active_workers:
                    spikes = batch_spikes.get(worker.slot_index, {})
                    active_ids = tuple(
                        int(body_id) for body_id, fired in spikes.items() if bool(fired)
                    )
                    worker.request("act", active_ids)
                acts = {
                    worker.slot_index: worker.receive("act") for worker in active_workers
                }

                terminal: list[int] = []
                for worker in active_workers:
                    slot_id = worker.slot_index
                    slot = slots[slot_id]
                    act = acts[slot_id]
                    position = act["position"]
                    velocity = act["velocity"]
                    x_mm = float(position[0])
                    z_mm = float(position[2])
                    slot["final_velocity"] = tuple(float(v) for v in velocity)
                    slot["max_x_mm"] = max(float(slot["max_x_mm"]), x_mm)
                    slot["min_z_mm"] = min(float(slot["min_z_mm"]), z_mm)
                    slot["max_z_mm"] = max(float(slot["max_z_mm"]), z_mm)
                    slot["control_steps"] = int(slot["control_steps"]) + 1

                    if bool(act["passed_gate"]):
                        slot["passed_gates"] = int(slot["passed_gates"]) + 1
                        slot["reward_events"] = int(slot["reward_events"]) + 1
                        brain.step_slot(
                            slot_id,
                            stimulate={"reward_dan": 2.0},
                            plasticity=True,
                            steps=4,
                        )
                    if bool(act["collision"]):
                        slot["collision"] = True
                        slot["collision_reason"] = act["collision_reason"]
                        slot["aversive_events"] = int(slot["aversive_events"]) + 1
                        brain.step_slot(
                            slot_id,
                            stimulate={"aversive_dan": 2.0},
                            plasticity=True,
                            steps=4,
                        )
                    if bool(act["finished"]):
                        slot["finished"] = True

                    if (
                        bool(slot["collision"])
                        or bool(slot["finished"])
                        or int(slot["control_steps"]) >= max_steps
                    ):
                        terminal.append(slot_id)

                for slot_id in sorted(terminal):
                    slot = slots[slot_id]
                    commit = brain.commit_slot(
                        slot_id,
                        source_weight_version=int(slot["source_weight_version"]),
                    )
                    slot["active"] = False
                    completed += 1
                    commit_records.append(
                        {
                            "slot": slot_id,
                            "episode": int(slot["episode"]),
                            "source_weight_version": int(commit["source_weight_version"]),
                            "commit_from_version": int(commit["commit_from_version"]),
                            "commit_weight_version": int(commit["commit_weight_version"]),
                            "staleness": int(commit["staleness"]),
                        }
                    )

            saved = brain.save_checkpoint(out / "checkpoint")
            final_global_version = brain.global_weight_version

        results = []
        for slot_id in sorted(slots):
            slot = slots[slot_id]
            commit = next(item for item in commit_records if int(item["slot"]) == slot_id)
            results.append(
                {
                    "episode": int(slot["episode"]),
                    "slot": slot_id,
                    "source_weight_version": int(slot["source_weight_version"]),
                    "commit_from_version": int(commit["commit_from_version"]),
                    "commit_weight_version": int(commit["commit_weight_version"]),
                    "version_staleness": int(commit["staleness"]),
                    "control_steps": int(slot["control_steps"]),
                    "passed_gates": int(slot["passed_gates"]),
                    "collision": bool(slot["collision"]),
                    "collision_reason": slot["collision_reason"],
                    "finished": bool(slot["finished"]),
                    "max_x_mm": float(slot["max_x_mm"]),
                    "min_z_mm": float(slot["min_z_mm"]),
                    "max_z_mm": float(slot["max_z_mm"]),
                    "final_vx_mm_s": float(slot["final_velocity"][0]),
                    "spawn_x_mm": float(condition.x_mm),
                    "spawn_z_mm": float(condition.z_mm),
                    "initial_speed_mm_s": float(condition.speed_mm_s),
                    "reward_events": int(slot["reward_events"]),
                    "aversive_events": int(slot["aversive_events"]),
                }
            )
        return {
            "episode_results": results,
            "global_weight_version_start": initial_global_version,
            "global_weight_version_end": final_global_version,
            "checkpoint_neural_step": saved.get("step"),
            "commit_records": commit_records,
        }
    finally:
        for worker in workers:
            try:
                worker.close()
            except Exception:
                pass


def comparable_result(item: dict) -> dict:
    keys = (
        "episode", "slot", "source_weight_version", "commit_from_version",
        "commit_weight_version", "version_staleness", "control_steps",
        "passed_gates", "collision", "collision_reason", "finished",
        "max_x_mm", "min_z_mm", "max_z_mm", "final_vx_mm_s",
        "spawn_x_mm", "spawn_z_mm", "initial_speed_mm_s",
        "reward_events", "aversive_events",
    )
    return {key: item.get(key) for key in keys}


def results_equal(reference: list[dict], candidate: list[dict]) -> bool:
    if len(reference) != len(candidate):
        return False
    for a, b in zip(
        sorted(reference, key=lambda item: int(item["episode"])),
        sorted(candidate, key=lambda item: int(item["episode"])),
        strict=True,
    ):
        aa = comparable_result(a)
        bb = comparable_result(b)
        for key in aa:
            av, bv = aa[key], bb[key]
            if isinstance(av, float) or isinstance(bv, float):
                if not math.isclose(float(av), float(bv), rel_tol=0.0, abs_tol=0.0):
                    return False
            elif av != bv:
                return False
    return True


def main() -> int:
    args = parse_args()
    if args.population < 1 or args.population > 8:
        raise SystemExit("population must be in 1..8 for this verifier")
    if args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("control/physics steps must be positive")

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
        raise SystemExit("missing real-CNS verifier inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))

    production_before = tree_digest(production / "checkpoint")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    reference_dir = temp / "reference"
    candidate_dir = temp / "process-body"
    clone_state(production, reference_dir)
    clone_state(production, candidate_dir)

    reference = run_reference(
        reference_dir,
        args.population,
        args.max_control_steps,
        args.physics_steps,
    )
    candidate = run_process_candidate(
        candidate_dir,
        args.population,
        args.max_control_steps,
        args.physics_steps,
        args.timeout_s,
    )

    production_after = tree_digest(production / "checkpoint")
    reference_digest = tree_digest(reference_dir / "checkpoint")
    candidate_digest = tree_digest(candidate_dir / "checkpoint")
    checkpoint_equal = reference_digest == candidate_digest
    episode_equal = results_equal(reference["episode_results"], candidate["episode_results"])
    version_equal = (
        int(reference["global_weight_version_start"]) == int(candidate["global_weight_version_start"])
        and int(reference["global_weight_version_end"]) == int(candidate["global_weight_version_end"])
        and int(reference["checkpoint_neural_step"]) == int(candidate["checkpoint_neural_step"])
    )
    production_unchanged = production_before == production_after
    overall = checkpoint_equal and episode_equal and version_equal and production_unchanged

    lines = [
        "# Flyppy process-body real-CNS parity verification",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        f"- population: {args.population}",
        f"- max control steps per episode: {args.max_control_steps}",
        f"- physics steps per control step: {args.physics_steps}",
        "- reference: current local-body train_flyppy_population.py",
        "- candidate: process-isolated bodies + shared parent GPU CNS",
        f"- final checkpoint byte/tree digest equal: {checkpoint_equal}",
        f"- episode and commit outcomes equal: {episode_equal}",
        f"- global version and neural step equal: {version_equal}",
        f"- production checkpoint modified: {not production_unchanged}",
        f"- production checkpoint digest: `{production_before}`",
        f"- reference final checkpoint digest: `{reference_digest}`",
        f"- candidate final checkpoint digest: `{candidate_digest}`",
        "",
        "## Reference episode results",
        "",
        "```json",
        json.dumps([comparable_result(x) for x in reference["episode_results"]], indent=2),
        "```",
        "",
        "## Process-body episode results",
        "",
        "```json",
        json.dumps([comparable_result(x) for x in candidate["episode_results"]], indent=2),
        "```",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"process_body_real_cns_verify={report}")
    print(f"checkpoint_equal={str(checkpoint_equal).lower()}")
    print(f"episode_equal={str(episode_equal).lower()}")
    print(f"version_equal={str(version_equal).lower()}")
    print(f"production_checkpoint_modified={str(not production_unchanged).lower()}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
