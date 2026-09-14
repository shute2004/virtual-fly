#!/usr/bin/env python3
"""Detached read-only observer for a running Flyppy training process.

The training process publishes qpos/qvel snapshots atomically. This observer
copies them into a separate MuJoCo model, renders the actual FlyBody and writes
``live/fly.png`` for the browser viewer. Browser camera gestures are read from
``live/camera.json`` and affect only this observer. Nothing in this process is
fed back into training.
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
    parser.add_argument("--poll-hz", type=float, default=30.0)
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
        + _png_chunk(b"IDAT", zlib.compress(rows, level=3))
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


def main() -> int:
    args = parse_args()
    if args.poll_hz <= 0.0:
        raise SystemExit("poll-hz must be positive")
    if args.width < 160 or args.height < 120:
        raise SystemExit("render size is too small")

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

    live_dir = args.experiment / "live"
    live_path = live_dir / "body.json"
    frame_path = live_dir / "fly.png"
    camera_path = live_dir / "camera.json"
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

    print(f"body_viewer_source={live_path}")
    print(f"body_frame_output={frame_path}")
    print(f"body_camera_input={camera_path}")
    print("body_viewer=waiting-for-telemetry")

    try:
        while True:
            started = time.perf_counter()
            if native_viewer is not None and not native_viewer.is_running():
                native_viewer.close()
                native_viewer = None

            try:
                payload = json.loads(live_path.read_text(encoding="utf-8"))
                key = (int(payload["episode"]), int(payload["control_step"]))
                try:
                    camera_mtime = camera_path.stat().st_mtime_ns
                except FileNotFoundError:
                    camera_mtime = None
                camera_changed = camera_mtime != last_camera_mtime

                if key != last_key or camera_changed:
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

                    if camera_changed:
                        camera_state = read_camera_state(camera_path, camera_state)
                        last_camera_mtime = camera_mtime
                    camera.azimuth = camera_state["azimuth"]
                    camera.elevation = camera_state["elevation"]
                    camera.distance = camera_state["distance"]
                    camera.lookat[:] = np.asarray(body.thorax_position_mm(), dtype=np.float64)

                    renderer.update_scene(body.sim.mj_data, camera=camera)
                    frame = renderer.render()
                    write_bytes_atomic(frame_path, encode_rgb_png(frame))

                    if native_viewer is not None:
                        native_viewer.sync()
                    last_key = key
            except FileNotFoundError:
                pass
            except json.JSONDecodeError:
                pass

            remaining = period - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(min(remaining, period))
    except KeyboardInterrupt:
        return 0
    finally:
        renderer.close()
        if native_viewer is not None:
            native_viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
