#!/usr/bin/env python3
"""Detached MuJoCo observer for a running Flyppy training process.

This process owns a separate read-only simulation. It never writes training
state. The training process publishes qpos/qvel snapshots atomically; this
viewer copies them into its own MuJoCo model and calls mj_forward without
stepping physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco as mj
import mujoco.viewer as mjviewer
import numpy as np

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v1"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--poll-hz", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_hz <= 0.0:
        raise SystemExit("poll-hz must be positive")

    course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count)
    world = FlyppyWorld(course)
    body = FlyBodyMuscleAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_observer_camera=False,
    )
    live_path = args.experiment / "live" / "body.json"
    last_key: tuple[int, int] | None = None
    period = 1.0 / args.poll_hz

    print(f"body_viewer_source={live_path}")
    print("body_viewer=waiting-for-telemetry")
    with mjviewer.launch_passive(body.sim.mj_model, body.sim.mj_data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            try:
                payload = json.loads(live_path.read_text(encoding="utf-8"))
                key = (int(payload["episode"]), int(payload["control_step"]))
                if key != last_key:
                    qpos = np.asarray(payload["qpos"], dtype=np.float64)
                    qvel = np.asarray(payload["qvel"], dtype=np.float64)
                    if qpos.shape != body.sim.mj_data.qpos.shape:
                        raise RuntimeError(
                            f"qpos shape mismatch: telemetry={qpos.shape} viewer={body.sim.mj_data.qpos.shape}"
                        )
                    if qvel.shape != body.sim.mj_data.qvel.shape:
                        raise RuntimeError(
                            f"qvel shape mismatch: telemetry={qvel.shape} viewer={body.sim.mj_data.qvel.shape}"
                        )
                    body.sim.mj_data.qpos[:] = qpos
                    body.sim.mj_data.qvel[:] = qvel
                    body.sim.mj_data.time = float(payload.get("sim_time_s", 0.0))
                    mj.mj_forward(body.sim.mj_model, body.sim.mj_data)
                    last_key = key
                    viewer.sync()
            except FileNotFoundError:
                pass
            except json.JSONDecodeError:
                # Atomic replacement should prevent partial reads, but a stale
                # external filesystem/cache must never crash the observer.
                pass

            remaining = period - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(min(remaining, period))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
