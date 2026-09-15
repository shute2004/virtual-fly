#!/usr/bin/env python3
"""Verify detached Flyppy telemetry activates only on viewer demand."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from live_telemetry import LiveTelemetryPublisher, ViewerDemandSwitch


def send(target: Path, payload: bytes) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
        client.sendto(payload, str(target))


def main() -> int:
    temp_root = Path(tempfile.mkdtemp(prefix="virtual-fly-viewer-demand-"))
    experiment = temp_root / "experiment"
    live = experiment / "live"
    live.mkdir(parents=True, exist_ok=True)
    for name in ("status.json", "body.json", "neural.json", "fly.png"):
        (live / name).write_bytes(b"stale")

    switch = ViewerDemandSwitch(always=False)
    publisher = LiveTelemetryPublisher(
        experiment,
        enabled=switch,
        lease_timeout_s=0.25,
    )
    try:
        stale_outputs_cleared = all(
            not (live / name).exists()
            for name in ("status.json", "body.json", "neural.json", "fly.png")
        )

        # Before publisher binding the switch is used once only to preload the
        # immutable viewer graph. After binding it must be inactive with no viewer.
        inactive_initial = not bool(switch)
        publisher.publish_status(
            running=True,
            backend="probe",
            episode=1,
            control_step=1,
        )
        status = live / "status.json"
        no_write_without_viewer = not status.exists()

        # Exact mid-run join contract: the status published before the viewer
        # existed must materialize immediately when the watch lease arrives.
        send(publisher.control_path, b"watch")
        active_after_watch = bool(switch)
        cached_status_materialized = status.exists()
        cached_payload = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
        cached_control_step_is_1 = cached_payload.get("control_step") == 1

        publisher.publish_status(
            running=True,
            backend="probe",
            episode=1,
            control_step=2,
        )
        wrote_with_viewer = status.exists()
        watched_payload = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
        watched_control_step_is_2 = watched_payload.get("control_step") == 2

        before_mtime = status.stat().st_mtime_ns if status.exists() else -1
        send(publisher.control_path, b"stop")
        inactive_after_stop = not bool(switch)
        publisher.publish_status(
            running=True,
            backend="probe",
            episode=1,
            control_step=3,
        )
        after_stop_mtime = status.stat().st_mtime_ns if status.exists() else -2
        no_write_after_stop = before_mtime == after_stop_mtime

        send(publisher.control_path, b"watch")
        active_for_timeout = bool(switch)
        time.sleep(0.35)
        inactive_after_timeout = not bool(switch)

        checks = {
            "stale_outputs_cleared": stale_outputs_cleared,
            "inactive_initial": inactive_initial,
            "no_write_without_viewer": no_write_without_viewer,
            "active_after_watch": active_after_watch,
            "cached_status_materialized": cached_status_materialized,
            "cached_control_step_is_1": cached_control_step_is_1,
            "wrote_with_viewer": wrote_with_viewer,
            "watched_control_step_is_2": watched_control_step_is_2,
            "inactive_after_stop": inactive_after_stop,
            "no_write_after_stop": no_write_after_stop,
            "active_for_timeout": active_for_timeout,
            "inactive_after_timeout": inactive_after_timeout,
        }
        passed = all(checks.values())
        for name, value in checks.items():
            print(f"{name}={str(value).lower()}")
        print(f"viewer_demand_telemetry={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        publisher.close()
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
