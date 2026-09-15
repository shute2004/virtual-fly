#!/usr/bin/env python3
"""Integration probe for detached viewer stream-lease transport."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from live_telemetry import LiveTelemetryPublisher, ViewerDemandSwitch
from live_viewer_server import ViewerTelemetryLease


def wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="virtual-fly-viewer-stream-"))
    try:
        # Case 1: trainer first, viewer joins later.
        exp_a = root / "trainer-first"
        switch_a = ViewerDemandSwitch(always=False)
        publisher_a = LiveTelemetryPublisher(exp_a, enabled=switch_a)
        publisher_a.publish_status(
            running=True,
            backend="probe",
            episode=1,
            control_step=7,
        )
        lease_a = ViewerTelemetryLease(exp_a, interval_s=0.05)
        lease_a.start()
        trainer_first_active = wait_until(lambda: bool(switch_a))
        trainer_first_status = (exp_a / "live" / "status.json").exists()
        lease_a.close()
        trainer_first_detached = wait_until(lambda: not bool(switch_a))
        publisher_a.close()

        # Case 2: viewer first, trainer appears later. This is the production
        # workflow the detached viewer must support.
        exp_b = root / "viewer-first"
        lease_b = ViewerTelemetryLease(exp_b, interval_s=0.05)
        lease_b.start()
        time.sleep(0.15)
        switch_b = ViewerDemandSwitch(always=False)
        publisher_b = LiveTelemetryPublisher(exp_b, enabled=switch_b)
        publisher_b.publish_status(
            running=True,
            backend="probe",
            episode=2,
            control_step=11,
        )
        viewer_first_active = wait_until(lambda: bool(switch_b))
        viewer_first_status = (exp_b / "live" / "status.json").exists()
        lease_b.close()
        viewer_first_detached = wait_until(lambda: not bool(switch_b))
        publisher_b.close()

        checks = {
            "trainer_first_active": trainer_first_active,
            "trainer_first_status": trainer_first_status,
            "trainer_first_detached": trainer_first_detached,
            "viewer_first_active": viewer_first_active,
            "viewer_first_status": viewer_first_status,
            "viewer_first_detached": viewer_first_detached,
        }
        passed = all(checks.values())
        for name, value in checks.items():
            print(f"{name}={str(value).lower()}")
        print(f"viewer_stream_lease={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
