#!/usr/bin/env python3
"""Loop the strongest current Flyppy presentation condition without learning.

This is a viewer source, not a trainer. It loads the production MaleCNS
checkpoint, freezes plasticity, and repeatedly evaluates one fixed v4 condition
while a detached viewer is connected. No checkpoint, curriculum, transaction, or
video file is written.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from flyppy_packed_body_worker import spawn_packed_body_processes
from live_telemetry import LiveTelemetryPublisher
from population_neural_bridge_client import PopulationNeuralBridgeClient

from virtual_fly.playback.config import (
    DEFAULT_CALIBRATION,
    DEFAULT_COURSE_SEED,
    DEFAULT_PRODUCTION,
    DEFAULT_SNAPSHOT,
    DEFAULT_SPAWN_X_MM,
    DEFAULT_SPAWN_Z_MM,
    DEFAULT_SPEED_MM_S,
    DEFAULT_VIEWER_GRAPH,
    absolute,
    ensure_runtime_environment,
    load_json,
    load_viewer_ids,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREVIEW = Path("artifacts/experiments/flyppy-best-preview")


def wait_for_viewer(publisher: LiveTelemetryPublisher) -> None:
    while not publisher.requested():
        time.sleep(0.05)


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    preview_dir = absolute(args.preview_dir)
    snapshot = absolute(args.snapshot)
    viewer_graph = absolute(args.viewer_graph)
    calibration = absolute(args.calibration)

    required = [
        production / "checkpoint" / "manifest.json",
        production / "population-state.json",
        snapshot / "manifest.json",
        snapshot / "embodiment-groups-v0.json",
        snapshot / "retinotopic-vision-v1.json",
        snapshot / "wing-motor-neurons-v0.json",
        snapshot / "body-motor-neurons-v0.json",
        viewer_graph,
        calibration,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing preview inputs:\n  " + "\n  ".join(missing))
    if (
        args.gate_count < 2
        or args.physics_steps < 1
        or args.max_control_steps < 1
        or args.neural_telemetry_stride < 1
    ):
        raise SystemExit("gate-count must be >=2 and step counts/telemetry stride must be positive")
    if args.reinforcement_steps < 1 or args.reinforcement_steps % 2 != 0:
        raise SystemExit("reinforcement-steps must be a positive even integer")

    preview_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_environment(calibration)
    viewer_ids = load_viewer_ids(viewer_graph)
    population_state = load_json(production / "population-state.json")
    global_weight_version = int(population_state.get("global_weight_version", 0))

    config = {
        "seed": int(args.course_seed),
        "gate_count": int(args.gate_count),
        "environment_version": str(args.environment_version),
        "wing_motor_map": str(snapshot / "wing-motor-neurons-v0.json"),
        "body_motor_map": str(snapshot / "body-motor-neurons-v0.json"),
        "retinotopic_map": str(snapshot / "retinotopic-vision-v1.json"),
        "photoreceptor_current_gain": float(args.photoreceptor_current_gain),
    }

    workers = []
    publisher = LiveTelemetryPublisher(preview_dir, enabled=False, on_demand=True)
    publisher.publish_status(
        running=True,
        backend="waiting-for-viewer",
        episode=None,
        control_step=None,
        curriculum={
            "mode": "read-only-best-preview",
            "environment_version": args.environment_version,
            "plasticity": False,
            "course_seed": args.course_seed,
        },
    )

    try:
        workers, slot_to_worker = spawn_packed_body_processes(
            population=1,
            process_count=1,
            physics_steps=args.physics_steps,
            timeout_s=args.timeout_s,
            config=config,
        )
        worker = slot_to_worker[0]
        motor_ids = tuple(int(value) for value in worker.body_ids[0])
        motor_id_set = frozenset(motor_ids)
        read_ids = tuple(sorted(motor_id_set | frozenset(viewer_ids)))

        with PopulationNeuralBridgeClient(
            snapshot=snapshot,
            groups=snapshot / "embodiment-groups-v0.json",
            slots=1,
        ) as brain:
            brain.ping()
            loaded = brain.load_checkpoint(
                production / "checkpoint",
                global_weight_version=global_weight_version,
            )
            backend = str(brain.ready.get("backend", "gpu-population"))
            print(
                "preview_ready environment={} seed={} checkpoint_step={} global_v={} "
                "spawn=({:.4f},{:.4f}) speed={:.2f}".format(
                    args.environment_version,
                    args.course_seed,
                    int(loaded.get("step", -1)),
                    brain.global_weight_version,
                    args.spawn_x_mm,
                    args.spawn_z_mm,
                    args.initial_speed_mm_s,
                ),
                flush=True,
            )
            print(f"preview_dir={preview_dir}", flush=True)
            print("plasticity=false checkpoint_write=false video_write=false", flush=True)

            preview_episode = 0
            while True:
                wait_for_viewer(publisher)
                brain.restart_slot(0)
                worker.request(
                    "reset",
                    {
                        0: {
                            "episode": preview_episode,
                            "source_weight_version": brain.global_weight_version,
                            "spawn_x_mm": float(args.spawn_x_mm),
                            "spawn_z_mm": float(args.spawn_z_mm),
                            "initial_speed_mm_s": float(args.initial_speed_mm_s),
                        }
                    },
                )
                worker.receive("reset")

                control_step = 0
                passed_gates = 0
                terminal = False
                publisher.publish_status(
                    running=True,
                    backend=backend,
                    episode=preview_episode,
                    control_step=0,
                    curriculum={
                        "mode": "read-only-best-preview",
                        "environment_version": args.environment_version,
                        "plasticity": False,
                        "course_seed": args.course_seed,
                    },
                )

                initial_snapshot = worker.call("snapshot", (0,))[0]
                publisher.publish_body_state(
                    episode=preview_episode,
                    control_step=0,
                    sim_time_s=float(initial_snapshot["sim_time_s"]),
                    qpos=initial_snapshot["qpos"],
                    qvel=initial_snapshot["qvel"],
                    next_gate=0,
                    passed_gate=False,
                    collision=False,
                    collision_reason=None,
                    reward=False,
                    aversive=False,
                    motor={},
                    retinal={},
                )

                while not terminal and control_step < args.max_control_steps:
                    if not publisher.requested():
                        break

                    observation = worker.call("observe", (0,))[0]
                    neural_sample = control_step % args.neural_telemetry_stride == 0
                    batch = brain.step_batch(
                        [
                            {
                                "slot": 0,
                                "stimulate_body": observation["body_currents"],
                                "read_body": read_ids if neural_sample else motor_ids,
                            }
                        ],
                        plasticity=False,
                    )
                    spikes = batch.get(0, {})
                    active_motor = tuple(
                        body_id for body_id in motor_ids if bool(spikes.get(body_id, False))
                    )
                    act = worker.call("act", {0: active_motor})[0]
                    control_step += 1

                    reward = bool(act["passed_gate"])
                    aversive = bool(act["collision"])
                    if reward:
                        passed_gates += 1
                        brain.step_slot(
                            0,
                            stimulate={"reward_dan": float(args.reward_current)},
                            plasticity=False,
                            steps=args.reinforcement_steps,
                        )
                    if aversive:
                        brain.step_slot(
                            0,
                            stimulate={"aversive_dan": float(args.aversive_current)},
                            plasticity=False,
                            steps=args.reinforcement_steps,
                        )

                    if neural_sample or reward or aversive:
                        active_viewer = [
                            body_id for body_id in viewer_ids if bool(spikes.get(body_id, False))
                        ]
                        publisher.publish_neural(
                            episode=preview_episode,
                            control_step=control_step,
                            neural_step=brain.last_step,
                            depolarizing_body_ids=active_viewer,
                            hyperpolarizing_body_ids=(),
                            reward=reward,
                            aversive=aversive,
                        )
                    body_snapshot = worker.call("snapshot", (0,))[0]
                    publisher.publish_body_state(
                        episode=preview_episode,
                        control_step=control_step,
                        sim_time_s=float(body_snapshot["sim_time_s"]),
                        qpos=body_snapshot["qpos"],
                        qvel=body_snapshot["qvel"],
                        next_gate=int(act["next_gate"]),
                        passed_gate=reward,
                        collision=aversive,
                        collision_reason=act["collision_reason"],
                        reward=reward,
                        aversive=aversive,
                        motor=act.get("motor") or {},
                        retinal={
                            "active_photoreceptors": int(observation["active_photoreceptors"]),
                            "active_columns": int(observation["active_columns"]),
                            "mean_current": float(observation["mean_current"]),
                            "max_current": float(observation["max_current"]),
                        },
                    )
                    publisher.publish_status(
                        running=True,
                        backend=backend,
                        episode=preview_episode,
                        control_step=control_step,
                        curriculum={
                            "mode": "read-only-best-preview",
                            "environment_version": args.environment_version,
                            "plasticity": False,
                            "course_seed": args.course_seed,
                            "passed_gates": passed_gates,
                        },
                    )
                    terminal = bool(act["collision"] or act["finished"])

                if terminal and publisher.requested():
                    print(
                        f"preview_episode={preview_episode} passed_gates={passed_gates} "
                        f"control_steps={control_step}",
                        flush=True,
                    )
                    deadline = time.monotonic() + max(0.0, float(args.terminal_hold_s))
                    while publisher.requested() and time.monotonic() < deadline:
                        time.sleep(0.05)
                preview_episode += 1
    except KeyboardInterrupt:
        return 0
    finally:
        publisher.publish_status(
            running=False,
            backend="preview-stopped",
            episode=None,
            control_step=None,
            curriculum={"mode": "read-only-best-preview"},
        )
        publisher.close()
        for worker in workers:
            try:
                worker.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
