#!/usr/bin/env python3
"""Peripheral state for the grounded whole-body MaleCNS motor boundary.

This composes the existing individual wing-MN periphery with a conservative set
of identified leg and haltere motor neurons.  Each body ID keeps its own local
activation state.  Multiple MNs innervating the same anatomical action are
combined only at the physical muscle/joint boundary; no population activity is
used to choose an action or read environment state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

from wing_muscle_periphery import PeripheralSnapshot, WingMusclePeriphery


@dataclass(frozen=True)
class SomaticMotorUnit:
    body_id: int
    subclass: str
    side: str
    leg_prefix: str
    target: str
    joint_role: str
    action: str


@dataclass(frozen=True)
class WholeBodyPeripheralSnapshot:
    wing: PeripheralSnapshot
    somatic_activation_by_body: dict[int, float]
    leg_drive: dict[str, float]
    haltere_power: dict[str, float]
    somatic_spikes: int
    selected_somatic_units: int

    @property
    def selected_motor_units(self) -> int:
        return self.wing.selected_motor_units + self.selected_somatic_units

    def compact_diagnostics(self) -> dict[str, object]:
        payload = self.wing.compact_diagnostics()
        payload.update(
            {
                "whole_body": True,
                "somatic_spikes": self.somatic_spikes,
                "active_somatic_units": sum(
                    value > 1e-4 for value in self.somatic_activation_by_body.values()
                ),
                "leg_drive": {
                    key: value for key, value in self.leg_drive.items() if abs(value) > 1e-4
                },
                "haltere_power": dict(self.haltere_power),
            }
        )
        return payload


class WholeBodyPeriphery:
    """Individual MaleCNS MN events -> grounded peripheral activation state."""

    def __init__(
        self,
        wing_motor_map: Path,
        body_motor_map: Path,
        *,
        wing_steering_tau_s: float = 0.012,
        wing_steering_spike_increment: float = 0.85,
        leg_tau_s: float = 0.025,
        leg_spike_increment: float = 0.80,
        haltere_power_tau_s: float = 0.25,
        haltere_power_spike_increment: float = 0.12,
        initial_haltere_power_activation: float = 0.35,
    ) -> None:
        self.wing = WingMusclePeriphery(
            wing_motor_map,
            steering_tau_s=float(wing_steering_tau_s),
            steering_spike_increment=float(wing_steering_spike_increment),
        )
        payload = json.loads(Path(body_motor_map).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("body motor map schema 1 is required")
        if leg_tau_s <= 0.0 or haltere_power_tau_s <= 0.0:
            raise ValueError("somatic peripheral time constants must be positive")
        for name, value in (
            ("leg_spike_increment", leg_spike_increment),
            ("haltere_power_spike_increment", haltere_power_spike_increment),
            ("initial_haltere_power_activation", initial_haltere_power_activation),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        units: list[SomaticMotorUnit] = []
        excluded: list[dict[str, object]] = []
        for row in payload.get("neurons", []):
            if str(row.get("mechanical_status", "")) != "grounded":
                excluded.append(
                    {
                        "body_id": int(row["body_id"]),
                        "subclass": str(row.get("subclass", "")),
                        "target": str(row.get("target", "")),
                    }
                )
                continue
            action = str(row.get("mechanical_action", ""))
            joint_role = str(row.get("joint_role", ""))
            subclass = str(row.get("subclass", ""))
            leg_prefix = str(row.get("leg_prefix", ""))
            side = str(row.get("side", "")).upper()
            if not action or not joint_role:
                raise ValueError(f"grounded body MN {row.get('body_id')} lacks mechanics")
            if subclass in {"fl", "ml", "hl"} and leg_prefix not in {
                "lf", "lm", "lh", "rf", "rm", "rh"
            }:
                raise ValueError(
                    f"grounded leg MN {row.get('body_id')} has invalid leg prefix {leg_prefix!r}"
                )
            if subclass == "hm" and side not in {"L", "R"}:
                raise ValueError(f"grounded haltere MN {row.get('body_id')} has unresolved side")
            units.append(
                SomaticMotorUnit(
                    body_id=int(row["body_id"]),
                    subclass=subclass,
                    side=side,
                    leg_prefix=leg_prefix,
                    target=str(row.get("target", "")),
                    joint_role=joint_role,
                    action=action,
                )
            )

        units.sort(key=lambda unit: unit.body_id)
        if len({unit.body_id for unit in units}) != len(units):
            raise ValueError("body motor map contains duplicate grounded body IDs")
        if not units:
            raise ValueError("body motor map selected zero grounded units")

        self.units = tuple(units)
        self.excluded = tuple(excluded)
        self.somatic_body_ids = tuple(unit.body_id for unit in self.units)
        self.body_ids = tuple(dict.fromkeys((*self.wing.body_ids, *self.somatic_body_ids)))
        self._activation = {unit.body_id: 0.0 for unit in self.units}
        self.leg_tau_s = float(leg_tau_s)
        self.leg_spike_increment = float(leg_spike_increment)
        self.haltere_power_tau_s = float(haltere_power_tau_s)
        self.haltere_power_spike_increment = float(haltere_power_spike_increment)
        self.initial_haltere_power_activation = float(initial_haltere_power_activation)
        self.reset()

    @staticmethod
    def _saturating_sum(values: list[float]) -> float:
        remaining = 1.0
        for value in values:
            bounded = min(1.0, max(0.0, float(value)))
            remaining *= 1.0 - bounded
        return 1.0 - remaining

    def reset(self) -> None:
        self.wing.reset()
        for unit in self.units:
            self._activation[unit.body_id] = (
                self.initial_haltere_power_activation
                if unit.subclass == "hm" and unit.action == "power"
                else 0.0
            )

    def step(self, spikes: Mapping[int, bool], *, dt_s: float) -> WholeBodyPeripheralSnapshot:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        missing = [body_id for body_id in self.body_ids if body_id not in spikes]
        if missing:
            preview = ",".join(str(body_id) for body_id in missing[:8])
            raise RuntimeError(
                f"neural bridge omitted {len(missing)} requested whole-body MN IDs: {preview}"
            )

        wing_snapshot = self.wing.step(spikes, dt_s=dt_s)
        somatic_spikes = 0
        for unit in self.units:
            if unit.subclass == "hm" and unit.action == "power":
                tau = self.haltere_power_tau_s
                increment = self.haltere_power_spike_increment
            else:
                tau = self.leg_tau_s
                increment = self.leg_spike_increment
            activation = self._activation[unit.body_id] * math.exp(-dt_s / tau)
            if bool(spikes[unit.body_id]):
                somatic_spikes += 1
                activation = 1.0 - (1.0 - activation) * (1.0 - increment)
            self._activation[unit.body_id] = min(1.0, max(0.0, activation))

        action_values: dict[str, list[float]] = {}
        haltere_values: dict[str, list[float]] = {"left": [], "right": []}
        for unit in self.units:
            activation = self._activation[unit.body_id]
            if unit.subclass == "hm" and unit.action == "power":
                side = "left" if unit.side == "L" else "right"
                haltere_values[side].append(activation)
                continue
            if unit.leg_prefix:
                key = f"{unit.leg_prefix}:{unit.joint_role}:{unit.action}"
                action_values.setdefault(key, []).append(activation)

        collapsed = {
            key: self._saturating_sum(values) for key, values in action_values.items()
        }
        leg_drive: dict[str, float] = {}
        for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
            for role, positive, negative in (
                ("femur_tibia_pitch", "extend", "flex"),
                ("coxa_femur_pitch", "extend", "flex"),
                ("thorax_coxa_yaw", "protract", "retract"),
            ):
                positive_value = collapsed.get(f"{leg}:{role}:{positive}", 0.0)
                negative_value = collapsed.get(f"{leg}:{role}:{negative}", 0.0)
                leg_drive[f"{leg}:{role}"] = min(
                    1.0, max(-1.0, positive_value - negative_value)
                )

        haltere_power = {
            side: self._saturating_sum(values) for side, values in haltere_values.items()
        }
        return WholeBodyPeripheralSnapshot(
            wing=wing_snapshot,
            somatic_activation_by_body=dict(self._activation),
            leg_drive=leg_drive,
            haltere_power=haltere_power,
            somatic_spikes=somatic_spikes,
            selected_somatic_units=len(self.units),
        )
