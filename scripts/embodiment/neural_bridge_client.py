#!/usr/bin/env python3
"""Small synchronous client for the persistent Rust neural bridge."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


class NeuralBridgeError(RuntimeError):
    pass


class NeuralBridgeClient:
    def __init__(
        self,
        *,
        snapshot: Path,
        groups: Path,
        backend: str = "gpu",
        repo_root: Path | None = None,
    ) -> None:
        root = repo_root or Path(__file__).resolve().parents[2]
        command = [
            "cargo",
            "run",
            "-q",
            "-p",
            "vf-runner",
            "--bin",
            "neural_bridge",
            "--release",
            "--",
            "--snapshot",
            str(snapshot),
            "--groups",
            str(groups),
            "--backend",
            backend,
        ]
        self._proc = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise NeuralBridgeError("failed to open neural bridge pipes")
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self.ready = self._read_response()
        if self.ready.get("event") != "ready":
            self.close(force=True)
            raise NeuralBridgeError(f"unexpected bridge startup response: {self.ready}")

    def _read_response(self) -> dict:
        line = self._stdout.readline()
        if not line:
            code = self._proc.poll()
            raise NeuralBridgeError(f"neural bridge exited unexpectedly (code={code})")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NeuralBridgeError(f"invalid bridge response: {line!r}") from exc
        if not response.get("ok", False):
            raise NeuralBridgeError(str(response.get("error", response)))
        return response

    def _request(self, payload: Mapping) -> dict:
        if self._proc.poll() is not None:
            raise NeuralBridgeError(
                f"neural bridge is not running (code={self._proc.returncode})"
            )
        self._stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._stdin.flush()
        return self._read_response()

    def ping(self) -> None:
        response = self._request({"type": "ping"})
        if response.get("event") != "pong":
            raise NeuralBridgeError(f"unexpected ping response: {response}")

    def step_with_body_readout(
        self,
        *,
        stimulate: Mapping[str, float] | None = None,
        stimulate_body: Sequence[tuple[int, float]] = (),
        read: Sequence[str] = (),
        read_body: Sequence[int] = (),
        plasticity: bool = True,
        steps: int = 1,
    ) -> tuple[dict[str, dict], dict[int, bool]]:
        """Advance the CNS and return optional group diagnostics + exact body spikes.

        ``read_body`` is the target motor-boundary primitive: each requested
        released MaleCNS body ID is returned independently. No population mean,
        matrix decoder, or action value is computed by this method.
        """

        response = self._request(
            {
                "type": "step",
                "stimulate": dict(stimulate or {}),
                "stimulate_body": [
                    [int(body_id), float(current)]
                    for body_id, current in stimulate_body
                ],
                "read": list(read),
                "read_body": [int(body_id) for body_id in read_body],
                "plasticity": bool(plasticity),
                "steps": int(steps),
            }
        )
        groups = response.get("read", {})
        bodies = {
            int(item["body_id"]): bool(item["spike"])
            for item in response.get("read_body", [])
        }
        return groups, bodies

    def step(
        self,
        *,
        stimulate: Mapping[str, float] | None = None,
        stimulate_body: Sequence[tuple[int, float]] = (),
        read: Sequence[str] = (),
        plasticity: bool = True,
        steps: int = 1,
    ) -> dict[str, dict]:
        groups, _ = self.step_with_body_readout(
            stimulate=stimulate,
            stimulate_body=stimulate_body,
            read=read,
            plasticity=plasticity,
            steps=steps,
        )
        return groups

    def save_weights(self, path: Path) -> dict:
        response = self._request({"type": "save_weights", "path": str(path)})
        if response.get("event") != "weights_saved":
            raise NeuralBridgeError(f"unexpected weight checkpoint response: {response}")
        return response

    def save_checkpoint(self, path: Path) -> dict:
        response = self._request({"type": "save_checkpoint", "path": str(path)})
        if response.get("event") != "checkpoint_saved":
            raise NeuralBridgeError(f"unexpected state checkpoint response: {response}")
        return response

    def load_checkpoint(self, path: Path) -> dict:
        response = self._request({"type": "load_checkpoint", "path": str(path)})
        if response.get("event") != "checkpoint_loaded":
            raise NeuralBridgeError(f"unexpected checkpoint load response: {response}")
        return response

    def close(self, *, force: bool = False) -> None:
        if self._proc.poll() is not None:
            return
        if not force:
            try:
                self._request({"type": "quit"})
            except Exception:
                force = True
        if force and self._proc.poll() is None:
            self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def __enter__(self) -> "NeuralBridgeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(force=exc is not None)
