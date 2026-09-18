#!/usr/bin/env python3
"""Evaluate one Flyppy v3 checkpoint under fixed, read-only conditions.

This is deliberately not a trainer:

- the production checkpoint is only loaded;
- every neural step runs with plasticity disabled;
- no slot is committed;
- no checkpoint/curriculum/trajectory state is saved;
- every condition starts from reset CNS/body/periphery/retina state.

Gate-pass PAM stimulation and collision PPL stimulation remain present so the
closed-loop neural state transition at task events matches production, but those
steps also run with plasticity disabled.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from flyppy_packed_body_worker import (  # noqa: E402
    active_by_worker,
    spawn_packed_body_processes,
)
from population_neural_bridge_client import PopulationNeuralBridgeClient  # noqa: E402
from virtual_fly.training.fixed_evaluation import (  # noqa: E402
    FIXED_EVAL_SUITE_V1,
    SUITE_VERSION,
    render_markdown,
    summarize_condition,
)


DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
DEFAULT_JSON = Path("reports/flyppy/fixed_evaluation_latest.json")
DEFAULT_REPORT = Path("reports/flyppy/fixed_evaluation_latest.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument(
        "--initial-malecns",
        action="store_true",
        help="evaluate the untrained MaleCNS initial weights instead of loading a production checkpoint",
    )
    p.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--population", type=int, default=4, help="fixed course seeds evaluated per condition")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--body-processes", type=int, default=0, help="0 = measured default min(population, 4)")
    p.add_argument("--gate-count", type=int, default=6)
    p.add_argument("--environment-version", choices=("v3", "v4", "v5", "v6", "v7"), default="v3")
    p.add_argument("--max-control-steps", type=int, default=1800)
    p.add_argument("--physics-steps", type=int, default=10)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--photoreceptor-current-gain", type=float, default=2.0)
    p.add_argument("--reward-current", type=float, default=2.0)
    p.add_argument("--aversive-current", type=float, default=2.0)
    p.add_argument("--reinforcement-steps", type=int, default=4)
    return p.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def evaluation_conditions(environment_version: str):
    # v4 deliberately preserves v3's gate-center band, so fixed-suite vertical
    # spawn coordinates stay identical across the two environments.
    _ = environment_version
    return FIXED_EVAL_SUITE_V1


def portable_path(path: Path) -> str:
    """Avoid embedding a developer-specific absolute home path in reports."""

    parts = path.resolve().parts
    if "artifacts" in parts:
        index = parts.index("artifacts")
        return str(Path(*parts[index:]))
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def validate(args: argparse.Namespace, production: Path, snapshot: Path, calibration: Path) -> None:
    if args.population < 1 or args.population > 32:
        raise SystemExit("--population must be in 1..32")
    if args.body_processes < 0:
        raise SystemExit("--body-processes must be >= 0")
    if args.gate_count < 2:
        raise SystemExit("--gate-count must be >= 2 so second-gate ability is observable")
    if min(args.max_control_steps, args.physics_steps, args.reinforcement_steps) < 1:
        raise SystemExit("step counts must be positive")
    if args.reinforcement_steps % 2 != 0:
        raise SystemExit("--reinforcement-steps must be even to match the production bridge contract")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0.0:
        raise SystemExit("--timeout-s must be finite and positive")
    positives = (
        args.photoreceptor_current_gain,
        args.reward_current,
        args.aversive_current,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positives):
        raise SystemExit("current gains must be finite and positive")

    required = [
        snapshot / "manifest.json",
        snapshot / "embodiment-groups-v0.json",
        snapshot / "retinotopic-vision-v1.json",
        snapshot / "wing-motor-neurons-v0.json",
        snapshot / "body-motor-neurons-v0.json",
        calibration,
    ]
    if not args.initial_malecns:
        required.append(production / "checkpoint" / "manifest.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing fixed-evaluation inputs:\n  " + "\n  ".join(missing))


def reset_bodies(workers, condition, *, global_weight_version: int) -> None:
    for worker in workers:
        payload = {
            slot_id: {
                "episode": -1,
                "source_weight_version": global_weight_version,
                "spawn_x_mm": condition.x_mm,
                "spawn_z_mm": condition.z_mm,
                "initial_speed_mm_s": condition.speed_mm_s,
            }
            for slot_id in worker.slot_ids
        }
        worker.request("reset", payload)
    for worker in workers:
        worker.receive("reset")


def empty_slot_state(condition, course_seed: int) -> dict[str, object]:
    return {
        "condition": condition.name,
        "course_seed": course_seed,
        "control_steps": 0,
        "passed_gates": 0,
        "collision": False,
        "collision_reason": None,
        "finished": False,
        "max_x_mm": float("-inf"),
        "min_z_mm": float("inf"),
        "max_z_mm": float("-inf"),
        "final_vx_mm_s": condition.speed_mm_s,
        "reward_events": 0,
        "aversive_events": 0,
        "wing_spikes_total": 0,
        "somatic_spikes_total": 0,
        "active_wing_units_sum": 0.0,
        "active_somatic_units_sum": 0.0,
        "power_activation_sum": 0.0,
        "power_lr_abs_diff_sum": 0.0,
        "active_steering_channels_sum": 0.0,
        "abs_leg_drive_sum": 0.0,
    }


def run_condition(
    *,
    condition,
    workers,
    slot_to_worker,
    brain: PopulationNeuralBridgeClient,
    population: int,
    seed_start: int,
    max_control_steps: int,
    reward_current: float,
    aversive_current: float,
    reinforcement_steps: int,
) -> list[dict[str, object]]:
    for slot_id in range(population):
        version = brain.restart_slot(slot_id)
        if version != brain.global_weight_version:
            raise RuntimeError(
                f"slot {slot_id} restart returned global version {version}, expected {brain.global_weight_version}"
            )
    reset_bodies(workers, condition, global_weight_version=brain.global_weight_version)

    states = {
        slot_id: empty_slot_state(condition, seed_start + slot_id)
        for slot_id in range(population)
    }
    active = set(states)

    while active:
        grouped = active_by_worker(workers, active)
        for worker, slot_ids in grouped.items():
            if slot_ids:
                worker.request("observe", slot_ids)
        observations: dict[int, dict[str, Any]] = {}
        for worker, slot_ids in grouped.items():
            if slot_ids:
                observations.update(
                    {int(k): v for k, v in worker.receive("observe").items()}
                )

        requests = [
            {
                "slot": slot_id,
                "stimulate_body": observations[slot_id]["body_currents"],
                "read_body": slot_to_worker[slot_id].body_ids[slot_id],
            }
            for slot_id in sorted(active)
        ]
        batch_spikes = brain.step_batch(requests, plasticity=False)

        acts_by_worker: dict[object, dict[int, tuple[int, ...]]] = {
            worker: {} for worker in workers
        }
        for slot_id in sorted(active):
            spikes = batch_spikes.get(slot_id, {})
            acts_by_worker[slot_to_worker[slot_id]][slot_id] = tuple(
                int(body_id) for body_id, fired in spikes.items() if bool(fired)
            )
        for worker, payload in acts_by_worker.items():
            if payload:
                worker.request("act", payload)
        acts: dict[int, dict[str, Any]] = {}
        for worker, payload in acts_by_worker.items():
            if payload:
                acts.update({int(k): v for k, v in worker.receive("act").items()})

        terminal: list[int] = []
        for slot_id in sorted(active):
            state = states[slot_id]
            act = acts[slot_id]
            position = tuple(float(v) for v in act["position"])
            velocity = tuple(float(v) for v in act["velocity"])
            state["control_steps"] = int(state["control_steps"]) + 1
            state["max_x_mm"] = max(float(state["max_x_mm"]), position[0])
            state["min_z_mm"] = min(float(state["min_z_mm"]), position[2])
            state["max_z_mm"] = max(float(state["max_z_mm"]), position[2])
            state["final_vx_mm_s"] = velocity[0]

            motor = dict(act.get("motor") or {})
            state["wing_spikes_total"] = int(state["wing_spikes_total"]) + int(
                motor.get("spikes", 0)
            )
            state["somatic_spikes_total"] = int(state["somatic_spikes_total"]) + int(
                motor.get("somatic_spikes", 0)
            )
            state["active_wing_units_sum"] = float(state["active_wing_units_sum"]) + float(
                motor.get("active_motor_units", 0)
            )
            state["active_somatic_units_sum"] = float(
                state["active_somatic_units_sum"]
            ) + float(motor.get("active_somatic_units", 0))
            dlm_left = float(motor.get("dlm_left", 0.0))
            dlm_right = float(motor.get("dlm_right", 0.0))
            dvm_left = float(motor.get("dvm_left", 0.0))
            dvm_right = float(motor.get("dvm_right", 0.0))
            state["power_activation_sum"] = float(state["power_activation_sum"]) + (
                dlm_left + dlm_right + dvm_left + dvm_right
            ) / 4.0
            state["power_lr_abs_diff_sum"] = float(
                state["power_lr_abs_diff_sum"]
            ) + (abs(dlm_left - dlm_right) + abs(dvm_left - dvm_right)) / 2.0
            steering = dict(motor.get("steering") or {})
            state["active_steering_channels_sum"] = float(
                state["active_steering_channels_sum"]
            ) + len(steering)
            leg_drive = dict(motor.get("leg_drive") or {})
            state["abs_leg_drive_sum"] = float(state["abs_leg_drive_sum"]) + (
                sum(abs(float(value)) for value in leg_drive.values()) / 18.0
            )

            if bool(act["passed_gate"]):
                state["passed_gates"] = int(state["passed_gates"]) + 1
                state["reward_events"] = int(state["reward_events"]) + 1
                brain.step_slot(
                    slot_id,
                    stimulate={"reward_dan": reward_current},
                    plasticity=False,
                    steps=reinforcement_steps,
                )
            if bool(act["collision"]):
                state["collision"] = True
                state["collision_reason"] = act["collision_reason"]
                state["aversive_events"] = int(state["aversive_events"]) + 1
                brain.step_slot(
                    slot_id,
                    stimulate={"aversive_dan": aversive_current},
                    plasticity=False,
                    steps=reinforcement_steps,
                )
            if bool(act["finished"]):
                state["finished"] = True

            if (
                bool(state["collision"])
                or bool(state["finished"])
                or int(state["control_steps"]) >= max_control_steps
            ):
                terminal.append(slot_id)

        for slot_id in terminal:
            active.remove(slot_id)

    rows: list[dict[str, object]] = []
    for slot_id in range(population):
        state = states[slot_id]
        steps = max(1, int(state["control_steps"]))
        rows.append(
            {
                "condition": state["condition"],
                "condition_values": asdict(condition),
                "course_seed": int(state["course_seed"]),
                "slot": slot_id,
                "control_steps": int(state["control_steps"]),
                "passed_gates": int(state["passed_gates"]),
                "collision": bool(state["collision"]),
                "collision_reason": state["collision_reason"],
                "finished": bool(state["finished"]),
                "max_x_mm": float(state["max_x_mm"]),
                "min_z_mm": float(state["min_z_mm"]),
                "max_z_mm": float(state["max_z_mm"]),
                "max_altitude_gain_mm": float(state["max_z_mm"]) - condition.z_mm,
                "max_altitude_loss_mm": condition.z_mm - float(state["min_z_mm"]),
                "final_vx_mm_s": float(state["final_vx_mm_s"]),
                "reward_events": int(state["reward_events"]),
                "aversive_events": int(state["aversive_events"]),
                "wing_spikes_total": int(state["wing_spikes_total"]),
                "somatic_spikes_total": int(state["somatic_spikes_total"]),
                "wing_spikes_per_step": int(state["wing_spikes_total"]) / steps,
                "somatic_spikes_per_step": int(state["somatic_spikes_total"]) / steps,
                "mean_active_wing_motor_units": float(state["active_wing_units_sum"]) / steps,
                "mean_active_somatic_motor_units": float(state["active_somatic_units_sum"]) / steps,
                "mean_power_activation": float(state["power_activation_sum"]) / steps,
                "mean_power_lr_abs_diff": float(state["power_lr_abs_diff_sum"]) / steps,
                "mean_active_steering_channels": float(
                    state["active_steering_channels_sum"]
                ) / steps,
                "mean_abs_leg_drive": float(state["abs_leg_drive_sum"]) / steps,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    snapshot = absolute(args.snapshot)
    calibration = absolute(args.calibration)
    output_json = absolute(args.output_json)
    report = absolute(args.report)
    validate(args, production, snapshot, calibration)

    # Fixed suite v1 is tied to the finalized Part3 production sensory seam.
    os.environ["VF_FLYPPY_VISION_MODE"] = "direct-ray"
    os.environ["VF_FLYPPY_OMMATIDIA_RAYS"] = "13"
    os.environ.pop("VF_COURSE_START_GATE", None)
    calibration_payload = load_json(calibration)
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration_payload["synapse_scale"]))

    population_state_path = production / "population-state.json"
    population_state = (
        load_json(population_state_path)
        if not args.initial_malecns and population_state_path.exists()
        else {}
    )
    initial_global_version = int(population_state.get("global_weight_version", 0))
    production_summary_path = production / "summary.json"
    production_summary = (
        load_json(production_summary_path)
        if not args.initial_malecns and production_summary_path.exists()
        else {}
    )
    training_episode_end = production_summary.get("episode_end")

    process_count = args.body_processes or min(args.population, 4)
    config = {
        "seed": int(args.seed_start),
        "gate_count": int(args.gate_count),
        "environment_version": str(args.environment_version),
        "wing_motor_map": str(snapshot / "wing-motor-neurons-v0.json"),
        "body_motor_map": str(snapshot / "body-motor-neurons-v0.json"),
        "retinotopic_map": str(snapshot / "retinotopic-vision-v1.json"),
        "photoreceptor_current_gain": float(args.photoreceptor_current_gain),
    }

    workers = []
    started = time.perf_counter()
    episode_results: list[dict[str, object]] = []
    dirty_after: dict[int, int] = {}
    loaded_step = -1
    backend = "gpu-population"
    vision_runtime = "unknown"
    vision_rays = -1
    final_global_version = initial_global_version

    conditions = evaluation_conditions(args.environment_version)

    try:
        workers, slot_to_worker = spawn_packed_body_processes(
            population=args.population,
            process_count=process_count,
            physics_steps=args.physics_steps,
            timeout_s=args.timeout_s,
            config=config,
        )
        vision_modes = {worker.vision_mode for worker in workers}
        vision_ray_counts = {worker.vision_rays_per_ommatidium for worker in workers}
        if vision_modes != {"direct-ray"} or vision_ray_counts != {13}:
            raise RuntimeError(
                f"fixed suite requires direct-ray K=13, got modes={vision_modes}, rays={vision_ray_counts}"
            )
        vision_runtime = next(iter(vision_modes))
        vision_rays = next(iter(vision_ray_counts))

        with PopulationNeuralBridgeClient(
            snapshot=snapshot,
            groups=snapshot / "embodiment-groups-v0.json",
            slots=args.population,
        ) as brain:
            brain.ping()
            backend = str(brain.ready.get("backend", "gpu-population"))
            if args.initial_malecns:
                loaded_step = 0
            else:
                loaded = brain.load_checkpoint(
                    production / "checkpoint",
                    global_weight_version=initial_global_version,
                )
                loaded_step = int(loaded.get("step", -1))

            for condition in conditions:
                episode_results.extend(
                    run_condition(
                        condition=condition,
                        workers=workers,
                        slot_to_worker=slot_to_worker,
                        brain=brain,
                        population=args.population,
                        seed_start=args.seed_start,
                        max_control_steps=args.max_control_steps,
                        reward_current=args.reward_current,
                        aversive_current=args.aversive_current,
                        reinforcement_steps=args.reinforcement_steps,
                    )
                )
                for slot_id in range(args.population):
                    stats = brain.transaction_stats(slot_id)
                    dirty_after[slot_id] = max(
                        dirty_after.get(slot_id, 0), int(stats["dirty_edges"])
                    )

            final_global_version = brain.global_weight_version
    finally:
        for worker in workers:
            try:
                worker.close()
            except Exception:
                pass

    if final_global_version != initial_global_version:
        raise RuntimeError(
            "fixed evaluation changed global weight version: "
            f"{initial_global_version} -> {final_global_version}"
        )
    if any(value != 0 for value in dirty_after.values()):
        raise RuntimeError(f"fixed evaluation produced plasticity transactions: {dirty_after}")

    elapsed = time.perf_counter() - started
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_version": SUITE_VERSION,
        "evaluation_subject": "initial_malecns" if args.initial_malecns else "trained_checkpoint",
        "checkpoint": (
            "initial MaleCNS snapshot weights"
            if args.initial_malecns
            else portable_path(production / "checkpoint")
        ),
        "checkpoint_neural_step": loaded_step,
        "training_episode_end": (
            int(training_episode_end) if training_episode_end is not None else None
        ),
        "global_weight_version": initial_global_version,
        "global_weight_version_unchanged": final_global_version == initial_global_version,
        "backend": backend,
        "population": args.population,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.population - 1,
        "body_processes": len(workers),
        "environment_version": args.environment_version,
        "vision_runtime": vision_runtime,
        "vision_rays_per_ommatidium": vision_rays,
        "plasticity": False,
        "reinforcement_event_stimulation": True,
        "transaction_dirty_edges_after": dirty_after,
        "elapsed_seconds": elapsed,
        "conditions": [asdict(condition) for condition in conditions],
        "condition_summaries": [
            summarize_condition(
                condition,
                [row for row in episode_results if row["condition"] == condition.name],
            )
            for condition in conditions
        ],
        "episode_results": episode_results,
    }
    save_json(output_json, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown(payload), encoding="utf-8")

    print(f"fixed_eval_suite={SUITE_VERSION}")
    print(f"checkpoint_step={loaded_step} global_weight_version={initial_global_version}")
    for item in payload["condition_summaries"]:
        print(
            "condition={} first_gate={}/{} second_gate={}/{} collision={}/{} max_gates={}".format(
                item["condition"]["name"],
                item["first_gate_passes"],
                item["episodes"],
                item["second_gate_passes"],
                item["episodes"],
                item["collisions"],
                item["episodes"],
                item["max_passed_gates"],
            )
        )
    print(f"global_weight_version_unchanged={final_global_version == initial_global_version}")
    print(f"transaction_dirty_edges_after={dirty_after}")
    print(f"json={output_json}")
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
