#!/usr/bin/env python3
"""Low-overhead live telemetry for detached Flyppy viewers.

Training never waits for a viewer and does not continuously write viewer state.
Instead it owns a tiny local Unix datagram control socket. A detached viewer
sends a short lease heartbeat while it is open; only then does training publish
body/neural/status JSON under ``<experiment>/live``. Closing the viewer sends an
explicit stop packet, and an abnormal viewer exit expires automatically after a
short lease timeout.

No renderer, browser, HTTP server, or viewer state is owned by training.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np


def viewer_control_socket_path(experiment_dir: Path) -> Path:
    """Return a short stable local socket path shared by trainer and viewer."""

    resolved = str(Path(experiment_dir).resolve()).encode("utf-8")
    token = hashlib.sha256(resolved).hexdigest()[:20]
    return Path("/tmp") / f"virtual-fly-viewer-{token}.sock"


class LiveTelemetryPublisher:
    def __init__(
        self,
        experiment_dir: Path,
        *,
        enabled: bool = False,
        on_demand: bool = True,
        lease_timeout_s: float = 2.0,
    ) -> None:
        self.always_enabled = bool(enabled)
        self.on_demand = bool(on_demand)
        self.lease_timeout_s = float(lease_timeout_s)
        if self.lease_timeout_s <= 0.0:
            raise ValueError("lease_timeout_s must be positive")

        self.root = Path(experiment_dir) / "live"
        self.control_path = viewer_control_socket_path(experiment_dir)
        self._last_request = float("-inf")
        self._control: socket.socket | None = None

        if self.on_demand:
            try:
                self.control_path.unlink(missing_ok=True)
                control = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                control.bind(str(self.control_path))
                control.setblocking(False)
                self._control = control
            except OSError:
                self.close()
                raise

    def close(self) -> None:
        control = self._control
        self._control = None
        if control is not None:
            try:
                control.close()
            except OSError:
                pass
        if self.on_demand:
            try:
                self.control_path.unlink(missing_ok=True)
            except OSError:
                pass

    def requested(self) -> bool:
        """Poll viewer control packets and return whether observation is active."""

        if self.always_enabled:
            return True
        control = self._control
        if control is None:
            return False

        while True:
            try:
                packet = control.recv(64)
            except BlockingIOError:
                break
            except OSError:
                break
            if packet.startswith(b"watch"):
                self._last_request = time.monotonic()
            elif packet.startswith(b"stop"):
                self._last_request = float("-inf")

        return (time.monotonic() - self._last_request) <= self.lease_timeout_s

    def _write(self, name: str, payload: Mapping[str, Any]) -> None:
        if not self.requested():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / name
        temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temp.write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, target)

    def publish_status(
        self,
        *,
        running: bool,
        backend: str,
        episode: int | None,
        control_step: int | None,
        curriculum: Mapping[str, Any] | None = None,
    ) -> None:
        self._write(
            "status.json",
            {
                "schema_version": 1,
                "running": bool(running),
                "backend": str(backend),
                "episode": episode,
                "control_step": control_step,
                "curriculum": dict(curriculum or {}),
            },
        )

    def publish_body_state(
        self,
        *,
        episode: int,
        control_step: int,
        sim_time_s: float,
        qpos: Sequence[float],
        qvel: Sequence[float],
        next_gate: int,
        passed_gate: bool,
        collision: bool,
        collision_reason: str | None,
        reward: bool,
        aversive: bool,
        motor: Mapping[str, Any],
        retinal: Mapping[str, Any],
    ) -> None:
        """Publish a body snapshot without requiring the MuJoCo object in-process."""

        self._write(
            "body.json",
            {
                "schema_version": 1,
                "episode": int(episode),
                "control_step": int(control_step),
                "sim_time_s": float(sim_time_s),
                "qpos": [float(value) for value in qpos],
                "qvel": [float(value) for value in qvel],
                "next_gate": int(next_gate),
                "passed_gate": bool(passed_gate),
                "collision": bool(collision),
                "collision_reason": collision_reason,
                "reward": bool(reward),
                "aversive": bool(aversive),
                "motor": dict(motor),
                "retinal": dict(retinal),
            },
        )

    def publish_body(
        self,
        *,
        episode: int,
        control_step: int,
        sim,
        next_gate: int,
        passed_gate: bool,
        collision: bool,
        collision_reason: str | None,
        reward: bool,
        aversive: bool,
        motor: Mapping[str, Any],
        retinal: Mapping[str, Any],
    ) -> None:
        qpos = np.asarray(sim.mj_data.qpos, dtype=np.float64)
        qvel = np.asarray(sim.mj_data.qvel, dtype=np.float64)
        self.publish_body_state(
            episode=episode,
            control_step=control_step,
            sim_time_s=float(sim.mj_data.time),
            qpos=qpos,
            qvel=qvel,
            next_gate=next_gate,
            passed_gate=passed_gate,
            collision=collision,
            collision_reason=collision_reason,
            reward=reward,
            aversive=aversive,
            motor=motor,
            retinal=retinal,
        )

    def publish_neural(
        self,
        *,
        episode: int,
        control_step: int,
        neural_step: int,
        depolarizing_body_ids: Sequence[int],
        hyperpolarizing_body_ids: Sequence[int],
        reward: bool,
        aversive: bool,
    ) -> None:
        self._write(
            "neural.json",
            {
                "schema_version": 1,
                "episode": int(episode),
                "control_step": int(control_step),
                "neural_step": int(neural_step),
                "depolarizing_body_ids": [int(value) for value in depolarizing_body_ids],
                "hyperpolarizing_body_ids": [int(value) for value in hyperpolarizing_body_ids],
                "reward": bool(reward),
                "aversive": bool(aversive),
            },
        )
