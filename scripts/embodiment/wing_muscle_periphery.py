#!/usr/bin/env python3
"""Peripheral wing motor-unit and muscle activation dynamics.

This module is intentionally downstream of the MaleCNS runtime. It receives only
spikes from individual released wing motor-neuron body IDs and advances a local
motor-unit / neuromuscular activation state. It does not inspect Flyppy state,
reward, gap position, or any neural population average, and it never chooses an
action.

Only MaleCNS neurons whose muscle mapping is currently ``identified`` or
``identified_group`` are admitted to the training boundary. Putative, variable,
and unresolved mappings remain in the inventory but are excluded here rather
than guessed.

The time constants and recruitment increments below are calibrated bootstrap
parameters, not measured MaleCNS constants. Power and steering muscles use
separate dynamics because Drosophila indirect flight muscles are asynchronous,
whereas direct steering muscles are synchronous.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping


TRAINING_MAPPING_STATUSES = frozenset({"identified", "identified_group"})


@dataclass(frozen=True)
class MotorUnit:
    body_id: int
    side: str
    neuron_type: str
    muscle_class: str
    target_muscle: str
    mapping_status: str


@dataclass(frozen=True)
class PeripheralSnapshot:
    """Read-only state passed from the neuromuscular layer to body mechanics."""

    activation_by_body: dict[int, float]
    muscle_activation: dict[str, float]
    dlm_activation: dict[str, float]
    dvm_activation: dict[str, float]
    active_spikes: int
    selected_motor_units: int

    @staticmethod
    def _key(side: str, muscle: str) -> str:
        return f"{side}:{muscle}"

    def muscle(self, side: str, muscle: str) -> float:
        return float(self.muscle_activation.get(self._key(side, muscle), 0.0))

    def power_dlm(self, side: str) -> float:
        return float(self.dlm_activation.get(side, 0.0))

    def power_dvm(self, side: str) -> float:
        return float(self.dvm_activation.get(side, 0.0))

    def compact_diagnostics(self) -> dict[str, object]:
        active_units = sum(value > 1e-4 for value in self.activation_by_body.values())
        steering = {
            key: value
            for key, value in self.muscle_activation.items()
            if any(
                token in key
                for token in (
                    ":b1",
                    ":b2",
                    ":b3",
                    ":i1",
                    ":i2",
                    ":iii1",
                    ":iii3",
                    ":hg1",
                    ":hg2",
                    ":hg3",
                    ":hg4",
                )
            )
            and value > 1e-4
        }
        return {
            "spikes": self.active_spikes,
            "active_motor_units": active_units,
            "dlm_left": self.power_dlm("left"),
            "dlm_right": self.power_dlm("right"),
            "dvm_left": self.power_dvm("left"),
            "dvm_right": self.power_dvm("right"),
            "steering": steering,
        }


class WingMusclePeriphery:
    """Individual MN spike -> motor-unit state -> anatomical muscle activation."""

    def __init__(
        self,
        motor_map: Path,
        *,
        power_tau_s: float = 0.25,
        steering_tau_s: float = 0.012,
        indirect_tau_s: float = 0.050,
        power_spike_increment: float = 0.12,
        steering_spike_increment: float = 0.85,
        indirect_spike_increment: float = 0.35,
        initial_power_activation: float = 0.35,
    ) -> None:
        payload = json.loads(Path(motor_map).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) < 2:
            raise ValueError("wing motor map schema >= 2 is required")

        if power_tau_s <= 0 or steering_tau_s <= 0 or indirect_tau_s <= 0:
            raise ValueError("peripheral time constants must be positive")
        for name, value in (
            ("power_spike_increment", power_spike_increment),
            ("steering_spike_increment", steering_spike_increment),
            ("indirect_spike_increment", indirect_spike_increment),
            ("initial_power_activation", initial_power_activation),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        units: list[MotorUnit] = []
        excluded: list[dict[str, object]] = []
        for row in payload.get("neurons", []):
            status = str(row.get("mapping_status", "unknown"))
            if status not in TRAINING_MAPPING_STATUSES:
                excluded.append(
                    {
                        "body_id": int(row["body_id"]),
                        "type": str(row.get("type", "")),
                        "mapping_status": status,
                    }
                )
                continue
            side_raw = str(row.get("side", "")).upper()
            if side_raw not in ("L", "R"):
                raise ValueError(
                    f"training motor unit {row.get('body_id')} has unresolved side {side_raw!r}"
                )
            target = str(row.get("target_muscle", "")).strip()
            muscle_class = str(row.get("muscle_class", "")).strip()
            if not target or muscle_class not in {
                "power",
                "direct_steering",
                "indirect_control",
            }:
                raise ValueError(
                    f"training motor unit {row.get('body_id')} has incomplete muscle mapping"
                )
            units.append(
                MotorUnit(
                    body_id=int(row["body_id"]),
                    side="left" if side_raw == "L" else "right",
                    neuron_type=str(row.get("type", "")),
                    muscle_class=muscle_class,
                    target_muscle=target,
                    mapping_status=status,
                )
            )

        units.sort(key=lambda unit: unit.body_id)
        if not units:
            raise ValueError("wing motor map selected zero training motor units")
        if len({unit.body_id for unit in units}) != len(units):
            raise ValueError("wing motor map contains duplicate body IDs")
        if not any(unit.side == "left" for unit in units) or not any(
            unit.side == "right" for unit in units
        ):
            raise ValueError("training motor boundary must contain both sides")
        if not any(unit.muscle_class == "power" for unit in units):
            raise ValueError("training motor boundary contains no power-muscle motor units")

        self.units = tuple(units)
        self.excluded = tuple(excluded)
        self.body_ids = tuple(unit.body_id for unit in self.units)
        self._by_body = {unit.body_id: unit for unit in self.units}
        self._activation = {unit.body_id: 0.0 for unit in self.units}
        self.power_tau_s = float(power_tau_s)
        self.steering_tau_s = float(steering_tau_s)
        self.indirect_tau_s = float(indirect_tau_s)
        self.power_spike_increment = float(power_spike_increment)
        self.steering_spike_increment = float(steering_spike_increment)
        self.indirect_spike_increment = float(indirect_spike_increment)
        self.initial_power_activation = float(initial_power_activation)
        self.reset()

    @property
    def selected_count(self) -> int:
        return len(self.units)

    @property
    def excluded_count(self) -> int:
        return len(self.excluded)

    def reset(self) -> None:
        # Flyppy episodes begin in already-established flight, just as the body is
        # given an initial forward velocity and wing phase. This is an initial
        # peripheral calcium/activation state, not a persistent external command.
        for unit in self.units:
            self._activation[unit.body_id] = (
                self.initial_power_activation if unit.muscle_class == "power" else 0.0
            )

    def _class_parameters(self, muscle_class: str) -> tuple[float, float]:
        if muscle_class == "power":
            return self.power_tau_s, self.power_spike_increment
        if muscle_class == "direct_steering":
            return self.steering_tau_s, self.steering_spike_increment
        if muscle_class == "indirect_control":
            return self.indirect_tau_s, self.indirect_spike_increment
        raise RuntimeError(f"unsupported muscle class {muscle_class!r}")

    @staticmethod
    def _saturating_sum(values: list[float]) -> float:
        """Combine independent motor-unit recruitment without taking an average."""

        remaining = 1.0
        for value in values:
            bounded = min(1.0, max(0.0, float(value)))
            remaining *= 1.0 - bounded
        return 1.0 - remaining

    def step(self, spikes: Mapping[int, bool], *, dt_s: float) -> PeripheralSnapshot:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        missing = [body_id for body_id in self.body_ids if body_id not in spikes]
        if missing:
            preview = ",".join(str(body_id) for body_id in missing[:8])
            raise RuntimeError(
                f"neural bridge omitted {len(missing)} requested wing MN body IDs: {preview}"
            )

        active_spikes = 0
        for unit in self.units:
            tau_s, increment = self._class_parameters(unit.muscle_class)
            activation = self._activation[unit.body_id] * math.exp(-dt_s / tau_s)
            if bool(spikes[unit.body_id]):
                active_spikes += 1
                # A spike recruits the remaining unactivated fraction. Keeping the
                # state per body ID avoids a population-rate motor decoder.
                activation = 1.0 - (1.0 - activation) * (1.0 - increment)
            self._activation[unit.body_id] = min(1.0, max(0.0, activation))

        per_muscle_units: dict[str, list[float]] = {}
        dlm_units: dict[str, list[float]] = {"left": [], "right": []}
        dvm_units: dict[str, list[float]] = {"left": [], "right": []}
        for unit in self.units:
            activation = self._activation[unit.body_id]
            key = f"{unit.side}:{unit.target_muscle}"
            per_muscle_units.setdefault(key, []).append(activation)
            if unit.muscle_class == "power":
                if unit.target_muscle.startswith("DLM"):
                    dlm_units[unit.side].append(activation)
                elif unit.target_muscle.startswith("DVM"):
                    dvm_units[unit.side].append(activation)

        muscle_activation = {
            key: self._saturating_sum(values)
            for key, values in per_muscle_units.items()
        }
        dlm_activation = {
            side: self._saturating_sum(values) for side, values in dlm_units.items()
        }
        dvm_activation = {
            side: self._saturating_sum(values) for side, values in dvm_units.items()
        }

        return PeripheralSnapshot(
            activation_by_body=dict(self._activation),
            muscle_activation=muscle_activation,
            dlm_activation=dlm_activation,
            dvm_activation=dvm_activation,
            active_spikes=active_spikes,
            selected_motor_units=len(self.units),
        )
