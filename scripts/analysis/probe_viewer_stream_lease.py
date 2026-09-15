#!/usr/bin/env python3
"""Integration probe for detached viewer stream-lease transport."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from live_telemetry import (
    LiveTelemetryPublisher,
    ViewerDemandSwitch,
    viewer_control_socket_path,
)
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
    resources: list[tuple[ViewerTelemetryLease | None, LiveTelemetryPublisher | None]] = []
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
        resources.append((lease_a, publisher_a))
        lease_a.start()
        trainer_first_active = wait_until(lambda: bool(switch_a))
        trainer_first_status = (exp_a / "live" / "status.json").exists()
        lease_a.close()
        trainer_first_detached = wait_until(lambda: not bool(switch_a))
        publisher_a.close()

        # Case 2: viewer first, trainer appears later.
        exp_b = root / "viewer-first"
        lease_b = ViewerTelemetryLease(exp_b, interval_s=0.05)
        resources.append((lease_b, None))
        lease_b.start()
        time.sleep(0.15)
        switch_b = ViewerDemandSwitch(always=False)
        publisher_b = LiveTelemetryPublisher(exp_b, enabled=switch_b)
        resources[-1] = (lease_b, publisher_b)
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

        # Case 3: a symlink spelling of the same physical experiment directory
        # must rendezvous on exactly the same control socket.
        physical = root / "physical-experiment"
        physical.mkdir()
        symlink_alias = root / "experiment-link"
        symlink_alias.symlink_to(physical, target_is_directory=True)
        symlink_identity_equal = (
            viewer_control_socket_path(physical)
            == viewer_control_socket_path(symlink_alias)
        )
        switch_c = ViewerDemandSwitch(always=False)
        publisher_c = LiveTelemetryPublisher(physical, enabled=switch_c)
        lease_c = ViewerTelemetryLease(symlink_alias, interval_s=0.05)
        resources.append((lease_c, publisher_c))
        lease_c.start()
        symlink_alias_active = wait_until(lambda: bool(switch_c))
        lease_c.close()
        publisher_c.close()

        # Case 4: on a case-insensitive filesystem, different case spellings of
        # the same existing directory must also map to the same endpoint. Linux
        # and other case-sensitive filesystems skip this platform-specific case.
        mixed = root / "CaseAliasExperiment"
        mixed.mkdir()
        case_alias = root / "casealiasexperiment"
        case_alias_supported = False
        case_alias_identity_equal = True
        case_alias_active = True
        try:
            case_alias_supported = case_alias.exists() and os.path.samefile(mixed, case_alias)
        except OSError:
            case_alias_supported = False
        if case_alias_supported:
            case_alias_identity_equal = (
                viewer_control_socket_path(mixed)
                == viewer_control_socket_path(case_alias)
            )
            switch_d = ViewerDemandSwitch(always=False)
            publisher_d = LiveTelemetryPublisher(mixed, enabled=switch_d)
            lease_d = ViewerTelemetryLease(case_alias, interval_s=0.05)
            resources.append((lease_d, publisher_d))
            lease_d.start()
            case_alias_active = wait_until(lambda: bool(switch_d))
            lease_d.close()
            publisher_d.close()

        # Case 5: if a viewer starts before a --fresh-style recreation of the
        # experiment directory, reconnect attempts must recompute the inode-based
        # socket identity and find the trainer's newly created listener.
        recreated = root / "recreated-experiment"
        recreated.mkdir()
        lease_e = ViewerTelemetryLease(recreated, interval_s=0.05)
        resources.append((lease_e, None))
        lease_e.start()
        time.sleep(0.10)
        shutil.rmtree(recreated)
        recreated.mkdir()
        switch_e = ViewerDemandSwitch(always=False)
        publisher_e = LiveTelemetryPublisher(recreated, enabled=switch_e)
        resources[-1] = (lease_e, publisher_e)
        recreated_active = wait_until(lambda: bool(switch_e))
        lease_e.close()
        publisher_e.close()

        checks = {
            "trainer_first_active": trainer_first_active,
            "trainer_first_status": trainer_first_status,
            "trainer_first_detached": trainer_first_detached,
            "viewer_first_active": viewer_first_active,
            "viewer_first_status": viewer_first_status,
            "viewer_first_detached": viewer_first_detached,
            "symlink_identity_equal": symlink_identity_equal,
            "symlink_alias_active": symlink_alias_active,
            "case_alias_identity_equal": case_alias_identity_equal,
            "case_alias_active": case_alias_active,
            "recreated_directory_active": recreated_active,
        }
        passed = all(checks.values())
        print(f"case_alias_supported={str(case_alias_supported).lower()}")
        for name, value in checks.items():
            print(f"{name}={str(value).lower()}")
        print(f"viewer_stream_lease={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        for lease, publisher in resources:
            if lease is not None:
                try:
                    lease.close()
                except Exception:
                    pass
            if publisher is not None:
                try:
                    publisher.close()
                except Exception:
                    pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
