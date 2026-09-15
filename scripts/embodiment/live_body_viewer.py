#!/usr/bin/env python3
"""Detached read-only observer for a running Flyppy training process.

The training process publishes qpos/qvel snapshots atomically. This observer
keeps the two latest poses, interpolates them with MuJoCo-aware generalized
coordinate math, and renders the actual FlyBody at a stable display cadence.
Browser camera gestures are read from ``live/camera.json`` and affect only this
observer. Nothing in this process is fed back into training.
"""

from __future__ import annotations

import argparse
import binascii
import json
import math
import os
from pathlib import Path
import struct
import time
import zlib

import mujoco as mj
import numpy as np

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from flybody_v3_adapter import FlyBodyV3MuscleAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld


DEFAULT_CAMERA = {
    "azimuth": 90.0,
    "elevation": -18.0,
    "distance": 14.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v1"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument(
        "--environment-version",
        choices=("v1", "v2", "v3"),
        default=None,
        help="explicit physical environment; overrides missing/stale run metadata",
    )
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=30.0,
        help="observer render cadence; source poses are interpolated to this rate",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--native-window", action="store_true")
    return parser.parse_args()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack("!I", len(payload))
        + kind
        + payload
        + struct.pack("!I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_rgb_png(image: np.ndarray) -> bytes:
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB image, got {rgb.shape}")
    rgb = np.ascontiguousarray(rgb[:, :, :3])
    height, width, _ = rgb.shape
    rows = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, level=1))
        + _png_chunk(b"IEND", b"")
    )


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def read_camera_state(path: Path, previous: dict[str, float]) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return previous
    try:
        azimuth = float(payload["azimuth"])
        elevation = float(payload["elevation"])
        distance = float(payload["distance"])
    except (KeyError, TypeError, ValueError):
        return previous
    if not all(math.isfinite(value) for value in (azimuth, elevation, distance)):
        return previous
    return {
        "azimuth": azimuth % 360.0,
        "elevation": max(-89.0, min(89.0, elevation)),
        "distance": max(2.0, min(80.0, distance)),
    }


def infer_training_context(
    experiment: Path,
    explicit_environment_version: str | None = None,
) -> tuple[int, str]:
    """Mirror trainer stage and physical environment in the detached observer."""

    state_path = experiment / "curriculum-state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        gate_index = int(state.get("training_gate_index", 0))
        environment_version = str(
            explicit_environment_version
            or state.get("environment_version", "v1")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        gate_index = 0
        environment_version = str(explicit_environment_version or "v1")
    if environment_version not in {"v1", "v2", "v3"}:
        environment_version = "v1"
    return max(0, gate_index), environment_version


def read_pose(path: Path) -> tuple[tuple[int, int], float, np.ndarray, np.ndarray] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    try:
        key = (int(payload["episode"]), int(payload["control_step"]))
        sim_time_s = float(payload.get("sim_time_s", 0.0))
        qpos = np.asarray(payload["qpos"], dtype=np.float64)
        qvel = np.asarray(payload["qvel"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    return key, sim_time_s, qpos, qvel


def interpolate_qpos(
    model,
    qpos_a: np.ndarray,
    qpos_b: np.ndarray,
    alpha: float,
    scratch_velocity: np.ndarray,
) -> np.ndarray:
    """Interpolate generalized positions without linearly blending quaternions."""

    if alpha <= 0.0:
        return qpos_a.copy()
    if alpha >= 1.0:
        return qpos_b.copy()
    scratch_velocity.fill(0.0)
    mj.mj_differentiatePos(model, scratch_velocity, 1.0, qpos_a, qpos_b)
    result = qpos_a.copy()
    mj.mj_integratePos(model, result, scratch_velocity, alpha)
    return result


def main() -> int:
    args = parse_args()
    if args.poll_hz <= 0.0:
        raise SystemExit("poll-hz must be positive")
    if args.width < 160 or args.height < 120:
        raise SystemExit("render size is too small")

    training_gate_index, environment_version = infer_training_context(
        args.experiment,
        args.environment_version,
    )
    os.environ["VF_COURSE_START_GATE"] = str(training_gate_index)
    course = FlyppyCourse(
        seed=args.seed,
        gate_count=args.gate_count,
        environment_version=environment_version,
    )
    world = FlyppyWorld(course)
    spawn_z = (course.floor_z_mm + course.ceiling_z_mm) / 2.0
    body_cls = FlyBodyV3MuscleAdapter if environment_version == "v3" else FlyBodyMuscleAdapter
    body = body_cls(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, spawn_z),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_observer_camera=False,
    )

    live_dir = args.experiment / "live"
    live_path = live_dir / "body.json"
    frame_path = live_dir / "fly.png"
    camera_path = live_dir / "camera.json"
    # fly.png belongs to this detached renderer, not to training. Never display
    # a frame inherited from a previous observer process while waiting for the
    # current run's first body snapshot.
    try:
        frame_path.unlink(missing_ok=True)
    except OSError:
        pass

    last_key: tuple[int, int] | None = None
    last_camera_mtime: int | None = None
    camera_state = dict(DEFAULT_CAMERA)
    period = 1.0 / args.poll_hz
    renderer = mj.Renderer(
        body.sim.mj_model,
        height=args.height,
        width=args.width,
    )

    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1

    native_viewer = None
    if args.native_window:
        import mujoco.viewer as mjviewer

        native_viewer = mjviewer.launch_passive(body.sim.mj_model, body.sim.mj_data)

    model = body.sim.mj_model
    scratch_velocity = np.zeros(model.nv, dtype=np.float64)
    pose_a_qpos: np.ndarray | None = None
    pose_a_qvel: np.ndarray | None = None
    pose_a_time = 0.0
    pose_b_qpos: np.ndarray | None = None
    pose_b_qvel: np.ndarray | None = None
    pose_b_time = 0.0
    transition_started = time.perf_counter()
    transition_duration = period
    last_source_arrival: float | None = None
    waiting_for_telemetry = True

    print(f"body_viewer_source={live_path}")
    print(f"body_frame_output={frame_path}")
    print(f"body_camera_input={camera_path}")
    print(f"body_viewer_training_gate_index={training_gate_index}")
    print(f"body_viewer_environment_version={environment_version}")
    print(f"body_viewer_render_hz={args.poll_hz:.1f} interpolation=MuJoCo-generalized-position")
    print("body_viewer=waiting-for-telemetry")

    try:
        while True:
            frame_started = time.perf_counter()
            now = frame_started
            if native_viewer is not None and not native_viewer.is_running():
                native_viewer.close()
                native_viewer = None

            pose = read_pose(live_path)
            if pose is None and not live_path.exists():
                # A new training run intentionally removes prior live telemetry.
                # Do not keep rendering the previous run's pose from memory.
                if last_key is not None:
                    last_key = None
                    pose_a_qpos = None
                    pose_a_qvel = None
                    pose_b_qpos = None
                    pose_b_qvel = None
                    last_source_arrival = None
                    try:
                        frame_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    if not waiting_for_telemetry:
                        print("body_viewer=waiting-for-telemetry")
                    waiting_for_telemetry = True
            elif pose is not None:
                key, source_time, incoming_qpos, incoming_qvel = pose
                if incoming_qpos.shape != body.sim.mj_data.qpos.shape:
                    raise RuntimeError(
                        f"qpos shape mismatch: telemetry={incoming_qpos.shape} viewer={body.sim.mj_data.qpos.shape}"
                    )
                if incoming_qvel.shape != body.sim.mj_data.qvel.shape:
                    raise RuntimeError(
                        f"qvel shape mismatch: telemetry={incoming_qvel.shape} viewer={body.sim.mj_data.qvel.shape}"
                    )

                if waiting_for_telemetry:
                    print(f"body_viewer=live episode={key[0]} control_step={key[1]}")
                    waiting_for_telemetry = False

                if key != last_key:
                    episode_changed = last_key is None or key[0] != last_key[0]
                    arrival = now
                    if episode_changed or pose_b_qpos is None:
                        pose_a_qpos = incoming_qpos.copy()
                        pose_a_qvel = incoming_qvel.copy()
                        pose_a_time = source_time
                        pose_b_qpos = incoming_qpos.copy()
                        pose_b_qvel = incoming_qvel.copy()
                        pose_b_time = source_time
                        transition_duration = period
                    else:
                        pose_a_qpos = np.asarray(body.sim.mj_data.qpos, dtype=np.float64).copy()
                        pose_a_qvel = np.asarray(body.sim.mj_data.qvel, dtype=np.float64).copy()
                        pose_a_time = float(body.sim.mj_data.time)
                        pose_b_qpos = incoming_qpos.copy()
                        pose_b_qvel = incoming_qvel.copy()
                        pose_b_time = source_time
                        if last_source_arrival is None:
                            source_interval = period
                        else:
                            source_interval = arrival - last_source_arrival
                        transition_duration = max(period, min(0.25, source_interval * 0.90))
                    transition_started = arrival
                    last_source_arrival = arrival
                    last_key = key

            try:
                camera_mtime = camera_path.stat().st_mtime_ns
            except FileNotFoundError:
                camera_mtime = None
            if camera_mtime != last_camera_mtime:
                camera_state = read_camera_state(camera_path, camera_state)
                last_camera_mtime = camera_mtime

            if (
                pose_a_qpos is not None
                and pose_b_qpos is not None
                and pose_a_qvel is not None
                and pose_b_qvel is not None
            ):
                alpha = min(
                    1.0,
                    max(0.0, (now - transition_started) / max(transition_duration, 1e-6)),
                )
                render_qpos = interpolate_qpos(
                    model, pose_a_qpos, pose_b_qpos, alpha, scratch_velocity
                )
                render_qvel = pose_a_qvel * (1.0 - alpha) + pose_b_qvel * alpha
                render_time = pose_a_time * (1.0 - alpha) + pose_b_time * alpha

                body.sim.mj_data.qpos[:] = render_qpos
                body.sim.mj_data.qvel[:] = render_qvel
                body.sim.mj_data.time = render_time
                mj.mj_forward(model, body.sim.mj_data)

                camera.azimuth = camera_state["azimuth"]
                camera.elevation = camera_state["elevation"]
                camera.distance = camera_state["distance"]
                camera.lookat[:] = np.asarray(body.thorax_position_mm(), dtype=np.float64)

                renderer.update_scene(body.sim.mj_data, camera=camera)
                frame = renderer.render()
                write_bytes_atomic(frame_path, encode_rgb_png(frame))

                if native_viewer is not None:
                    native_viewer.sync()

            remaining = period - (time.perf_counter() - frame_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        return 0
    finally:
        renderer.close()
        if native_viewer is not None:
            native_viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
