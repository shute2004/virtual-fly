#!/usr/bin/env python3
"""Serve the detached learning viewer and request telemetry while it is open.

The viewer remains independent from training. It holds one tiny local Unix
stream connection to the trainer while it is open; that connection itself is the
viewer lease. Training generates body and neural snapshots only while the stream
is connected. Browser camera gestures are written atomically under
``<experiment>/live/camera.json`` and are consumed only by
``live_body_viewer.py``'s separate MuJoCo renderer.

The HTTP layer deliberately tolerates partially materialized telemetry. The
browser polls status, neural state and body state together, so a missing optional
JSON file must not prevent it from continuing to request the independently
rendered ``fly.png`` frame.
"""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlparse

from live_telemetry import viewer_control_socket_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v2"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temp, path)


def finite_number(payload: dict, key: str) -> float:
    value = float(payload[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


class ViewerTelemetryLease:
    """Hold a reconnecting Unix stream to the trainer while the viewer is open."""

    def __init__(self, experiment: Path, *, interval_s: float = 0.4) -> None:
        self.target = viewer_control_socket_path(experiment)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="flyppy-viewer-telemetry-lease",
            daemon=True,
        )

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def _run(self) -> None:
        connection: socket.socket | None = None
        try:
            while not self._stop.is_set():
                if connection is None:
                    candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    candidate.settimeout(max(0.1, self.interval_s))
                    try:
                        candidate.connect(str(self.target))
                    except OSError:
                        candidate.close()
                        self._connected.clear()
                        self._stop.wait(self.interval_s)
                        continue
                    candidate.settimeout(None)
                    connection = candidate
                    self._connected.set()

                try:
                    connection.sendall(b"watch\n")
                except OSError:
                    try:
                        connection.close()
                    except OSError:
                        pass
                    connection = None
                    self._connected.clear()
                    continue

                self._stop.wait(self.interval_s)
        finally:
            self._connected.clear()
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 3.0))


def make_handler(
    root: Path,
    camera_path: Path,
    live_url_prefix: str,
    lease: ViewerTelemetryLease,
):
    live_root = camera_path.parent
    status_url = live_url_prefix + "status.json"
    body_url = live_url_prefix + "body.json"
    neural_url = live_url_prefix + "neural.json"

    class ViewerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def end_headers(self) -> None:
            if self.path.startswith("/artifacts/experiments/") or self.path.startswith("/api/"):
                self.send_header("Cache-Control", "no-store, max-age=0")
            super().end_headers()

        def _send_json(self, payload: dict, *, status: int = 200) -> None:
            response = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        @staticmethod
        def _read_json(path: Path) -> dict | None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return None
            return payload if isinstance(payload, dict) else None

        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            request_path = urlparse(self.path).path

            if request_path == "/api/viewer-health":
                artifacts = {}
                for name in ("status.json", "body.json", "neural.json", "fly.png", "camera.json"):
                    path = live_root / name
                    try:
                        stat = path.stat()
                    except FileNotFoundError:
                        artifacts[name] = {"exists": False}
                    else:
                        artifacts[name] = {
                            "exists": True,
                            "bytes": int(stat.st_size),
                            "mtime_ns": int(stat.st_mtime_ns),
                        }
                self._send_json(
                    {
                        "schema_version": 1,
                        "telemetry_stream_connected": lease.connected,
                        "telemetry_socket": str(lease.target),
                        "artifacts": artifacts,
                    }
                )
                return

            if request_path == status_url:
                status_path = live_root / "status.json"
                body_path = live_root / "body.json"
                status = self._read_json(status_path)
                body = self._read_json(body_path)

                # status is normally published at run start/end, whereas body is
                # sampled continuously. If body is newer, synthesize the current
                # episode/step instead of serving stale status metadata.
                if body is not None:
                    try:
                        body_mtime = body_path.stat().st_mtime_ns
                        status_mtime = status_path.stat().st_mtime_ns if status_path.exists() else -1
                    except OSError:
                        body_mtime = 0
                        status_mtime = 0
                    if status is None or status_mtime < body_mtime:
                        self._send_json(
                            {
                                "schema_version": 1,
                                "running": True,
                                "backend": (status or {}).get("backend", "gpu-population"),
                                "episode": body.get("episode"),
                                "control_step": body.get("control_step"),
                                "curriculum": (status or {}).get("curriculum", {}),
                                "viewer_attached": lease.connected,
                                "served_at_unix_s": time.time(),
                            }
                        )
                        return

                if status is None:
                    self._send_json(
                        {
                            "schema_version": 1,
                            "running": False,
                            "backend": "waiting-for-training",
                            "episode": None,
                            "control_step": None,
                            "curriculum": {},
                            "viewer_attached": lease.connected,
                            "served_at_unix_s": time.time(),
                        }
                    )
                    return

            # The browser currently polls status, neural and body in one
            # Promise.all. Return harmless waiting payloads for the optional
            # streams so one missing file does not suppress fly.png polling.
            if request_path == body_url and not (live_root / "body.json").exists():
                self._send_json(
                    {
                        "schema_version": 1,
                        "episode": None,
                        "control_step": None,
                        "sim_time_s": 0.0,
                        "qpos": [],
                        "qvel": [],
                        "next_gate": None,
                        "passed_gate": False,
                        "collision": False,
                        "collision_reason": None,
                        "reward": False,
                        "aversive": False,
                        "motor": {},
                        "retinal": {},
                    }
                )
                return

            if request_path == neural_url and not (live_root / "neural.json").exists():
                self._send_json(
                    {
                        "schema_version": 1,
                        "episode": None,
                        "control_step": None,
                        "neural_step": None,
                        "depolarizing_body_ids": [],
                        "hyperpolarizing_body_ids": [],
                        "reward": False,
                        "aversive": False,
                    }
                )
                return

            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
            if self.path != "/api/camera":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 4096:
                    raise ValueError("invalid camera payload length")
                payload = json.loads(self.rfile.read(length))
                azimuth = finite_number(payload, "azimuth") % 360.0
                elevation = max(-89.0, min(89.0, finite_number(payload, "elevation")))
                distance = max(2.0, min(80.0, finite_number(payload, "distance")))
                write_json_atomic(
                    camera_path,
                    {
                        "schema_version": 1,
                        "azimuth": azimuth,
                        "elevation": elevation,
                        "distance": distance,
                    },
                )
                self._send_json({"ok": True})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def log_message(self, format: str, *args) -> None:
            if self.path.startswith("/artifacts/experiments/"):
                return
            super().log_message(format, *args)

    return ViewerHandler


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be in 1..65535")
    root = args.root.resolve()
    experiment = (root / args.experiment).resolve()
    camera_path = experiment / "live" / "camera.json"
    try:
        experiment_relative = experiment.relative_to(root)
    except ValueError as exc:
        raise SystemExit("experiment path must be inside viewer root") from exc
    live_url_prefix = "/" + experiment_relative.as_posix() + "/live/"

    lease = ViewerTelemetryLease(experiment)
    lease.start()
    handler = make_handler(root, camera_path, live_url_prefix, lease)
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"viewer_server=http://{args.bind}:{args.port}")
    print(f"observer_camera={camera_path}")
    print(f"telemetry_request={lease.target}")
    print("telemetry_transport=unix-stream")
    print("viewer_health=/api/viewer-health")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        lease.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
