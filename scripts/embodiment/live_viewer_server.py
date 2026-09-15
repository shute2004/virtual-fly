#!/usr/bin/env python3
"""Serve the detached learning viewer and request telemetry while it is open.

The viewer remains independent from training. It sends only a tiny local lease
heartbeat to the trainer's Unix datagram control socket. Training generates body
and neural snapshots only while that lease is alive. Browser camera gestures are
written atomically under ``<experiment>/live/camera.json`` and are consumed only
by ``live_body_viewer.py``'s separate MuJoCo renderer.
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
    """Heartbeat a running trainer without owning or blocking it."""

    def __init__(self, experiment: Path, *, interval_s: float = 0.4) -> None:
        self.target = viewer_control_socket_path(experiment)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="flyppy-viewer-telemetry-lease",
            daemon=True,
        )

    @staticmethod
    def _send(target: Path, payload: bytes) -> None:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.sendto(payload, str(target))
        except OSError:
            # The viewer may be opened before training. The next heartbeat will
            # connect automatically once the trainer's control socket exists.
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._send(self.target, b"watch")
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 3.0))
        self._send(self.target, b"stop")


def make_handler(root: Path, camera_path: Path):
    class ViewerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def end_headers(self) -> None:
            if self.path.startswith("/artifacts/experiments/") or self.path.startswith("/api/"):
                self.send_header("Cache-Control", "no-store, max-age=0")
            super().end_headers()

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
                response = b'{"ok":true}\n'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                response = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8") + b"\n"
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

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
        experiment.relative_to(root)
    except ValueError as exc:
        raise SystemExit("experiment path must be inside viewer root") from exc

    handler = make_handler(root, camera_path)
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    lease = ViewerTelemetryLease(experiment)
    lease.start()
    print(f"viewer_server=http://{args.bind}:{args.port}")
    print(f"observer_camera={camera_path}")
    print(f"telemetry_request={lease.target}")
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
