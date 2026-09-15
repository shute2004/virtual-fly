#!/usr/bin/env python3
"""Verify detached Flyppy telemetry activates only on viewer demand."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from live_telemetry import LiveTelemetryPublisher, ViewerDemandSwitch


def connect(target: Path) -> socket.socket:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(target))
    client.sendall(b"watch\n")
    return client


def main() -> int:
    temp_root = Path(tempfile.mkdtemp(prefix="virtual-fly-viewer-demand-"))
    experiment = temp_root / "experiment"
    live = experiment / "live"
    live.mkdir(parents=True, exist_ok=True)
    for name in ("status.json", "body.json", "neural.json", "fly.png"):
        (live / name).write_bytes(b"stale")

    switch = ViewerDemandSwitch(always=False)
    publisher = LiveTelemetryPublisher(experiment, enabled=switch)
    client: socket.socket | None = None
    try:
        stale_outputs_cleared = all(
            not (live / name).exists()
            for name in ("status.json", "body.json", "neural.json", "fly.png")
        )

        inactive_initial = not bool(switch)
        publisher.publish_status(
            running=True,
            backend="probe",
            episode=1,
            control_step=1,
        )
        status = live / "status.json"
        no_write_without_viewer = not status.exists()

        client = connect(publisher.control_path)
        active_after_connect = bool(switch)
        cached_status_materialized = status.exists()
        cached_payload = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
        cached_control_step_is_1 = cached_payload.get("control_step") == 1

        publisher.publish_status(
            running=True,
            backend="probe",
            episode=1,
            control_step=2,
        )
        watched_payload = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
        watched_control_step_is_2 = watched_payload.get("control_step") == 2

        before_mtime = status.stat().st_mtime_ns if status.exists() else -1
        client.close()
        client = None
        inactive_after_disconnect = not bool(switch)
        publisher.publish_status(
            running=True,
            backend="probe",
            episode=1,
            control_step=3,
        )
        after_disconnect_mtime = status.stat().st_mtime_ns if status.exists() else -2
        no_write_after_disconnect = before_mtime == after_disconnect_mtime

        client = connect(publisher.control_path)
        active_after_reconnect = bool(switch)

        checks = {
            "stale_outputs_cleared": stale_outputs_cleared,
            "inactive_initial": inactive_initial,
            "no_write_without_viewer": no_write_without_viewer,
            "active_after_connect": active_after_connect,
            "cached_status_materialized": cached_status_materialized,
            "cached_control_step_is_1": cached_control_step_is_1,
            "watched_control_step_is_2": watched_control_step_is_2,
            "inactive_after_disconnect": inactive_after_disconnect,
            "no_write_after_disconnect": no_write_after_disconnect,
            "active_after_reconnect": active_after_reconnect,
        }
        passed = all(checks.values())
        for name, value in checks.items():
            print(f"{name}={str(value).lower()}")
        print(f"viewer_demand_telemetry={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        if client is not None:
            try:
                client.close()
            except OSError:
                pass
        publisher.close()
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
