"""Deterministic frozen Flyppy episode capture.

This module owns capture orchestration; CLI parsing remains in the thin script.
It intentionally preserves the existing evaluation semantics: weights may be
frozen while gate-triggered DAN stimulation still affects within-episode neural
state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from virtual_fly.playback.config import absolute, ensure_runtime_environment, load_json, load_viewer_ids
from virtual_fly.reproducibility import (
    dependency_lock_provenance,
    git_provenance,
    run_semantics,
    validate_haltere_map,
    validate_production_snapshot,
)
from virtual_fly.runtime.neural_bridge import PopulationNeuralBridgeClient
from virtual_fly.runtime.packed_body_worker import spawn_packed_body_processes
from virtual_fly.paths import REPO_ROOT


def keep_substep(index: int, count: int, stride: int) -> bool:
    return (index + 1) % stride == 0 or index + 1 == count


def capture_playback(args: Any) -> dict[str, object]:
    production = absolute(args.production)
    output = absolute(args.output)
    snapshot = absolute(args.snapshot)
    viewer_graph = absolute(args.viewer_graph)
    calibration = absolute(args.calibration)
    haltere_map = absolute(args.haltere_sensory_map)
    if min(args.physics_steps, args.physics_substep_stride, args.neural_telemetry_stride) <= 0:
        raise SystemExit("positive step/fps values required")
    if args.playback_fps <= 0:
        raise SystemExit("positive step/fps values required")

    snapshot_meta = validate_production_snapshot(snapshot)
    haltere_meta = validate_haltere_map(
        haltere_map, args.haltere_sensory_kind, snapshot=snapshot
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ensure_runtime_environment(calibration)
    viewer_ids = load_viewer_ids(viewer_graph)

    if args.fresh_brain:
        global_version = 0
    else:
        required = [production / "population-state.json", production / "checkpoint" / "manifest.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit("missing learned preview inputs:\n  " + "\n  ".join(missing))
        global_version = int(
            load_json(production / "population-state.json").get("global_weight_version", 0)
        )

    config = {
        "snapshot": str(snapshot),
        "seed": args.course_seed,
        "gate_count": args.gate_count,
        "environment_version": args.environment_version,
        "flight_body_version": args.flight_body_version,
        "vertical_steering_gain": args.vertical_steering_gain,
        "measured_steering_gain": args.measured_steering_gain,
        "neutral_trim_strength": args.neutral_trim_strength,
        "steering_tau_ms": args.steering_tau_ms,
        "steering_spike_increment": args.steering_spike_increment,
        "wing_motor_map": str(snapshot / "wing-motor-neurons-v0.json"),
        "body_motor_map": str(snapshot / "body-motor-neurons-v0.json"),
        "retinotopic_map": str(snapshot / "retinotopic-vision-v1.json"),
        "haltere_sensory_map": str(haltere_map),
        "haltere_sensory_kind": args.haltere_sensory_kind,
        "photoreceptor_current_gain": args.photoreceptor_current_gain,
        "haltere_current_gain": args.haltere_current_gain,
        "haltere_transduction": args.haltere_transduction,
        "capture_physics_trace": True,
    }
    workers = []
    frames: list[dict[str, object]] = []
    passed = 0
    terminal_reason: str | None = None
    control = 0
    try:
        workers, slotmap = spawn_packed_body_processes(
            population=1,
            process_count=1,
            physics_steps=args.physics_steps,
            timeout_s=args.timeout_s,
            config=config,
        )
        worker = slotmap[0]
        motor_ids = tuple(int(value) for value in worker.body_ids[0])
        read_ids = tuple(sorted(set(motor_ids) | set(viewer_ids)))
        with PopulationNeuralBridgeClient(
            snapshot=snapshot,
            groups=snapshot / "embodiment-groups-v0.json",
            slots=1,
        ) as brain:
            brain.ping()
            if args.fresh_brain:
                loaded = {"step": 0}
                if int(brain.global_weight_version) != 0:
                    raise RuntimeError(
                        "fresh MaleCNS bridge unexpectedly started at global weight version "
                        f"{brain.global_weight_version}"
                    )
            else:
                loaded = brain.load_checkpoint(
                    production / "checkpoint", global_weight_version=global_version
                )
            brain.restart_slot(0)
            reset_payload: dict[str, object] = {
                "episode": 0,
                "source_weight_version": brain.global_weight_version,
                "spawn_x_mm": args.spawn_x_mm,
                "spawn_z_mm": args.spawn_z_mm,
                "initial_speed_mm_s": args.initial_speed_mm_s,
                "initial_vz_mm_s": args.initial_vz_mm_s,
            }
            if args.gate2_center_z_mm is not None:
                reset_payload["gate_center_overrides"] = {1: float(args.gate2_center_z_mm)}
            worker.request("reset", {0: reset_payload})
            worker.receive("reset")
            initial = worker.call("snapshot", (0,))[0]
            frames.append(
                {
                    "control_step": 0,
                    "physics_substep": 0,
                    "sim_time_s": float(initial["sim_time_s"]),
                    "qpos": initial["qpos"],
                    "qvel": initial["qvel"],
                    "next_gate": 0,
                    "passed_gate": False,
                    "collision": False,
                    "collision_reason": None,
                    "reward": False,
                    "aversive": False,
                    "neural_step": int(brain.last_step),
                    "active_neural_body_ids": [],
                    "active_motor_body_ids": [],
                    "motor": {},
                    "retinal": {},
                }
            )
            terminal = False
            last_active: list[int] = []
            while not terminal and control < args.max_control_steps:
                observation = worker.call("observe", (0,))[0]
                neural_sample = control % args.neural_telemetry_stride == 0
                batch = brain.step_batch(
                    [
                        {
                            "slot": 0,
                            "stimulate_body": observation["body_currents"],
                            "read_body": read_ids if neural_sample else motor_ids,
                        }
                    ],
                    plasticity=bool(args.plasticity),
                )
                spikes = batch.get(0, {})
                if neural_sample:
                    last_active = [
                        body_id for body_id in viewer_ids if bool(spikes.get(body_id, False))
                    ]
                active_motor = tuple(
                    body_id for body_id in motor_ids if bool(spikes.get(body_id, False))
                )
                action = worker.call("act", {0: active_motor})[0]
                control += 1
                reward = bool(action["passed_gate"])
                aversive = bool(action["collision"])
                if reward:
                    passed += 1
                    if passed >= int(args.reward_from_gate):
                        brain.step_slot(
                            0,
                            stimulate={"reward_dan": args.reward_current},
                            plasticity=bool(args.plasticity),
                            steps=args.reinforcement_steps,
                        )
                if aversive:
                    brain.step_slot(
                        0,
                        stimulate={"aversive_dan": args.aversive_current},
                        plasticity=bool(args.plasticity),
                        steps=args.reinforcement_steps,
                    )
                    terminal_reason = str(action.get("collision_reason") or "collision")
                trace = list(action.get("physics_trace") or [])
                if not trace:
                    raise RuntimeError("missing physics_trace")
                count = len(trace)
                for index, pose in enumerate(trace):
                    if not keep_substep(index, count, args.physics_substep_stride):
                        continue
                    final = index + 1 == count
                    frames.append(
                        {
                            "control_step": control,
                            "physics_substep": index + 1,
                            "sim_time_s": float(pose["sim_time_s"]),
                            "qpos": pose["qpos"],
                            "qvel": pose["qvel"],
                            "next_gate": int(
                                action["next_gate"]
                                if final
                                else max(0, int(action["next_gate"]) - (1 if reward else 0))
                            ),
                            "passed_gate": bool(reward and final),
                            "collision": bool(aversive and final),
                            "collision_reason": (
                                action.get("collision_reason") if aversive and final else None
                            ),
                            "reward": bool(reward and final),
                            "aversive": bool(aversive and final),
                            "neural_step": int(brain.last_step),
                            "active_neural_body_ids": last_active,
                            "active_motor_body_ids": list(active_motor) if final else [],
                            "motor": action.get("motor") or {},
                            "physics": pose.get("diagnostics") or {},
                            "retinal": {
                                "active_photoreceptors": int(observation["active_photoreceptors"]),
                                "active_columns": int(observation["active_columns"]),
                                "mean_current": float(observation["mean_current"]),
                                "max_current": float(observation["max_current"]),
                            },
                        }
                    )
                terminal = bool(action["collision"] or action["finished"])

            payload: dict[str, object] = {
                "schema_version": 1,
                "kind": "flyppy-offline-preview-playback",
                "brain_state": "fresh-snapshot" if args.fresh_brain else "learned-checkpoint",
                "source_experiment": None if args.fresh_brain else str(production),
                "checkpoint_neural_step": int(loaded.get("step", 0 if args.fresh_brain else -1)),
                "global_weight_version": int(brain.global_weight_version),
                "environment_version": args.environment_version,
                "flight_body_version": args.flight_body_version,
                "vertical_steering_gain": args.vertical_steering_gain,
                "neutral_trim_strength": args.neutral_trim_strength,
                "steering_tau_ms": args.steering_tau_ms,
                "steering_spike_increment": args.steering_spike_increment,
                "haltere_current_gain": args.haltere_current_gain,
                "haltere_sensory_kind": args.haltere_sensory_kind,
                "haltere_sensory_map_sha256": haltere_meta["sha256"],
                "haltere_transduction": args.haltere_transduction,
                "vision_runtime": os.environ.get("VF_FLYPPY_VISION_MODE", "direct-ray"),
                "vision_rays_per_ommatidium": int(os.environ.get("VF_FLYPPY_OMMATIDIA_RAYS", "13")),
                "snapshot_semantics": snapshot_meta,
                "code": git_provenance(REPO_ROOT),
                "dependency_lock": dependency_lock_provenance(REPO_ROOT),
                "semantics": run_semantics(),
                "course_seed": args.course_seed,
                "gate_count": args.gate_count,
                "gate2_center_z_mm": args.gate2_center_z_mm,
                "spawn_x_mm": args.spawn_x_mm,
                "spawn_z_mm": args.spawn_z_mm,
                "initial_speed_mm_s": args.initial_speed_mm_s,
                "initial_vz_mm_s": args.initial_vz_mm_s,
                "physics_steps_per_control": args.physics_steps,
                "physics_substep_stride": args.physics_substep_stride,
                "playback_fps": args.playback_fps,
                "passed_gates": passed,
                "terminal_reason": terminal_reason,
                "control_steps": control,
                "plasticity": bool(args.plasticity),
                "reward_from_gate": int(args.reward_from_gate),
                "frame_count": len(frames),
                "frames": frames,
            }
            temp = output.with_name("." + output.name + ".tmp")
            temp.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
            os.replace(temp, output)
            return payload
    finally:
        for worker in workers:
            try:
                worker.close()
            except Exception:
                pass
