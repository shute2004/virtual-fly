#!/usr/bin/env python3
"""Low-overhead live telemetry for detached Flyppy viewers.

Training never waits for a viewer and does not continuously write viewer state.
Instead it owns a tiny local Unix stream listener. A detached viewer connects to
that socket while it is open; only then does training publish body/neural/status
JSON under ``<experiment>/live``. Closing or crashing the viewer closes the
stream and disables telemetry automatically.

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


def _filesystem_identity(path: Path) -> bytes:
    """Return a local physical identity for a path without depending on spelling.

    Existing paths are identified by the filesystem device/inode pair, so case
    aliases and symlink aliases of the same directory rendezvous on the same
    viewer socket. For a not-yet-created path, anchor the identity at the nearest
    existing ancestor and append only the missing suffix.
    """

    candidate = Path(path)
    try:
        stat = candidate.stat()
    except OSError:
        missing: list[str] = []
        current = candidate
        while True:
            try:
                stat = current.stat()
                break
            except OSError:
                parent = current.parent
                if parent == current:
                    return b"path\0" + os.fsencode(os.path.abspath(os.fspath(candidate)))
                missing.append(current.name)
                current = parent
        suffix = "/".join(reversed(missing))
        return (
            f"ancestor:{int(stat.st_dev)}:{int(stat.st_ino)}:".encode("ascii")
            + os.fsencode(suffix)
        )

    return f"inode:{int(stat.st_dev)}:{int(stat.st_ino)}".encode("ascii")


def viewer_control_socket_path(experiment_dir: Path) -> Path:
    """Return the short local socket path shared by trainer and viewer.

    The token is based on filesystem identity rather than the absolute path
    string. On case-insensitive macOS filesystems, ``Desktop`` and ``desktop``
    therefore map to the same socket when they name the same directory.
    """

    token = hashlib.sha256(_filesystem_identity(Path(experiment_dir))).hexdigest()[:20]
    return Path("/tmp") / f"virtual-fly-viewer-{token}.sock"


class ViewerDemandSwitch:
    """Bool-like trainer switch driven by a detached viewer connection."""

    def __init__(self, *, always: bool = False) -> None:
        self.always = bool(always)
        self._publisher: LiveTelemetryPublisher | None = None

    def bind(self, publisher: "LiveTelemetryPublisher") -> None:
        self._publisher = publisher

    def __bool__(self) -> bool:
        if self._publisher is None:
            return True
        return self.always or self._publisher.requested()

    def __str__(self) -> str:
        return "always" if self.always else "viewer-demand"


class LiveTelemetryPublisher:
    def __init__(
        self,
        experiment_dir: Path,
        *,
        enabled: bool | ViewerDemandSwitch = False,
        on_demand: bool = True,
        lease_timeout_s: float = 2.0,
    ) -> None:
        del lease_timeout_s
        switch = enabled if isinstance(enabled, ViewerDemandSwitch) else None
        self.always_enabled = switch.always if switch is not None else bool(enabled)
        self.enabled = self.always_enabled
        self.on_demand = bool(on_demand)

        self.root = Path(experiment_dir) / "live"
        self.control_path = viewer_control_socket_path(experiment_dir)
        self._listener: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._requested_last_poll = False
        self._latest_status: dict[str, Any] | None = None

        # requested() is called from the production control loop and again from
        # individual publish helpers. Polling the kernel on every call is
        # unnecessary while no observer is attached. A 50 ms cache keeps attach
        # and detach latency interactive while reducing idle socket syscalls to
        # at most 20 polling rounds/s.
        self._demand_poll_interval_s = 0.05
        self._last_demand_poll_s = float("-inf")
        self._cached_active = self.always_enabled

        # A new run must never inherit an old run's apparent live state.
        for name in ("status.json", "body.json", "neural.json", "fly.png", "fly.jpg"):
            try:
                (self.root / name).unlink(missing_ok=True)
            except OSError:
                pass

        if self.on_demand:
            try:
                self.control_path.unlink(missing_ok=True)
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                listener.bind(str(self.control_path))
                listener.listen(8)
                listener.setblocking(False)
                self._listener = listener
            except OSError:
                self.close()
                raise

        if switch is not None:
            switch.bind(self)

    def close(self) -> None:
        for client in self._clients:
            try:
                client.close()
            except OSError:
                pass
        self._clients.clear()
        self._cached_active = self.always_enabled

        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if self.on_demand:
            try:
                self.control_path.unlink(missing_ok=True)
            except OSError:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _write_unchecked(self, name: str, payload: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / name
        temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temp.write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, target)

    def _accept_clients(self) -> None:
        listener = self._listener
        if listener is None:
            return
        while True:
            try:
                client, _ = listener.accept()
            except BlockingIOError:
                break
            except OSError:
                break
            client.setblocking(False)
            self._clients.append(client)

    def _prune_clients(self) -> None:
        alive: list[socket.socket] = []
        for client in self._clients:
            keep = True
            while True:
                try:
                    packet = client.recv(256)
                except BlockingIOError:
                    break
                except OSError:
                    keep = False
                    break
                if not packet:
                    keep = False
                    break
                # Payload content is intentionally irrelevant. Receiving bytes
                # merely proves the stream is still connected.
            if keep:
                alive.append(client)
            else:
                try:
                    client.close()
                except OSError:
                    pass
        self._clients = alive

    def requested(self) -> bool:
        """Return whether at least one detached viewer is currently connected."""

        if self.always_enabled:
            active = True
        else:
            now = time.monotonic()
            if now - self._last_demand_poll_s >= self._demand_poll_interval_s:
                self._accept_clients()
                self._prune_clients()
                self._cached_active = bool(self._clients)
                self._last_demand_poll_s = now
            active = self._cached_active

        # publish_status() is normally called before a viewer exists. Materialize
        # the cached status immediately when the first viewer connects.
        if active and not self._requested_last_poll and self._latest_status is not None:
            self._write_unchecked("status.json", self._latest_status)
        self._requested_last_poll = active
        return active

    def _write(self, name: str, payload: Mapping[str, Any]) -> None:
        if not self.requested():
            return
        self._write_unchecked(name, payload)

    def publish_status(
        self,
        *,
        running: bool,
        backend: str,
        episode: int | None,
        control_step: int | None,
        curriculum: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": 1,
            "running": bool(running),
            "backend": str(backend),
            "episode": episode,
            "control_step": control_step,
            "curriculum": dict(curriculum or {}),
        }
        self._latest_status = payload
        self._write("status.json", payload)

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
        gate_height_center_z_mm: float | None = None,
    ) -> None:
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
                "gate_height_center_z_mm": (
                    None if gate_height_center_z_mm is None else float(gate_height_center_z_mm)
                ),
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
