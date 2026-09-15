#!/usr/bin/env python3
"""Client for the shared-weight batched Flyppy population neural bridge."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


class PopulationNeuralBridgeError(RuntimeError):
    pass


class PopulationNeuralBridgeClient:
    def __init__(
        self,
        *,
        snapshot: Path,
        groups: Path,
        slots: int = 2,
        repo_root: Path | None = None,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be >= 1")
        root = repo_root or Path(__file__).resolve().parents[2]
        command = [
            "cargo",
            "run",
            "-q",
            "-p",
            "vf-runner",
            "--bin",
            "population_neural_bridge",
            "--release",
            "--",
            "--snapshot",
            str(snapshot),
            "--groups",
            str(groups),
            "--slots",
            str(slots),
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
            raise PopulationNeuralBridgeError("failed to open population bridge pipes")
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self.last_step = 0
        self.ready = self._read_response()
        if self.ready.get("event") != "ready":
            self.close(force=True)
            raise PopulationNeuralBridgeError(
                f"unexpected population bridge startup response: {self.ready}"
            )
        self.global_weight_version = int(self.ready.get("global_weight_version", 0))
        self.slots = int(self.ready.get("slots", slots))

    def _read_response(self) -> dict:
        line = self._stdout.readline()
        if not line:
            raise PopulationNeuralBridgeError(
                f"population bridge exited unexpectedly (code={self._proc.poll()})"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PopulationNeuralBridgeError(f"invalid population bridge response: {line!r}") from exc
        if not response.get("ok", False):
            raise PopulationNeuralBridgeError(str(response.get("error", response)))
        if "global_weight_version" in response:
            self.global_weight_version = int(response["global_weight_version"])
        return response

    def _request(self, payload: Mapping) -> dict:
        if self._proc.poll() is not None:
            raise PopulationNeuralBridgeError(
                f"population bridge is not running (code={self._proc.returncode})"
            )
        self._stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._stdin.flush()
        return self._read_response()

    def ping(self) -> None:
        response = self._request({"type": "ping"})
        if response.get("event") != "pong":
            raise PopulationNeuralBridgeError(f"unexpected ping response: {response}")

    def load_checkpoint(self, path: Path, *, global_weight_version: int = 0) -> dict:
        response = self._request(
            {
                "type": "load_checkpoint",
                "path": str(path),
                "global_weight_version": int(global_weight_version),
            }
        )
        if response.get("event") != "checkpoint_loaded":
            raise PopulationNeuralBridgeError(f"unexpected checkpoint load response: {response}")
        self.last_step = int(response.get("step", self.last_step))
        self.global_weight_version = int(response.get("global_weight_version", global_weight_version))
        return response

    def save_checkpoint(self, path: Path) -> dict:
        response = self._request({"type": "save_checkpoint", "path": str(path)})
        if response.get("event") != "checkpoint_saved":
            raise PopulationNeuralBridgeError(f"unexpected checkpoint save response: {response}")
        self.last_step = int(response.get("step", self.last_step))
        return response

    def step_batch(self, slot_requests: Sequence[Mapping], *, plasticity: bool = True) -> dict[int, dict[int, bool]]:
        slots = []
        for request in slot_requests:
            slots.append(
                {
                    "slot": int(request["slot"]),
                    "stimulate": {
                        str(name): float(value)
                        for name, value in dict(request.get("stimulate", {})).items()
                    },
                    "stimulate_body": [
                        [int(body_id), float(current)]
                        for body_id, current in request.get("stimulate_body", ())
                    ],
                    "read_body": [int(body_id) for body_id in request.get("read_body", ())],
                }
            )
        response = self._request(
            {"type": "step_batch", "slots": slots, "plasticity": bool(plasticity)}
        )
        self.last_step = int(response.get("step", self.last_step))
        result: dict[int, dict[int, bool]] = {}
        for slot_response in response.get("slots", []):
            slot = int(slot_response["slot"])
            result[slot] = {
                int(item["body_id"]): bool(item["spike"])
                for item in slot_response.get("read_body", [])
            }
        return result

    def step_slot(
        self,
        slot: int,
        *,
        stimulate: Mapping[str, float] | None = None,
        stimulate_body: Sequence[tuple[int, float]] = (),
        plasticity: bool = True,
        steps: int = 1,
    ) -> None:
        response = self._request(
            {
                "type": "step_slot",
                "slot": int(slot),
                "stimulate": dict(stimulate or {}),
                "stimulate_body": [
                    [int(body_id), float(current)] for body_id, current in stimulate_body
                ],
                "plasticity": bool(plasticity),
                "steps": int(steps),
            }
        )
        self.last_step = int(response.get("step", self.last_step))

    def commit_slot(self, slot: int, *, source_weight_version: int) -> dict:
        response = self._request(
            {
                "type": "commit_slot",
                "slot": int(slot),
                "source_weight_version": int(source_weight_version),
            }
        )
        if response.get("event") != "transaction_committed":
            raise PopulationNeuralBridgeError(f"unexpected commit response: {response}")
        self.global_weight_version = int(response["commit_weight_version"])
        return response

    def restart_slot(self, slot: int) -> int:
        response = self._request({"type": "restart_slot", "slot": int(slot)})
        if response.get("event") != "slot_restarted":
            raise PopulationNeuralBridgeError(f"unexpected restart response: {response}")
        return int(response.get("global_weight_version", self.global_weight_version))

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

    def __enter__(self) -> "PopulationNeuralBridgeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(force=exc is not None)
