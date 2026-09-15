#!/usr/bin/env python3
"""Low-overhead live telemetry for external Flyppy viewers.

Training never waits for a viewer. The producer atomically replaces a few small
JSON files under the experiment directory; any number of observers may poll
those files independently. No renderer, browser, socket server, or viewer state
is owned by the training process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class LiveTelemetryPublisher:
    def __init__(self, experiment_dir: Path, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.root = Path(experiment_dir) / "live"
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _write(self, name: str, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
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
