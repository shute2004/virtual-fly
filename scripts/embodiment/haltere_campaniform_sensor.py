#!/usr/bin/env python3
"""Local physical transduction for MaleCNS haltere campaniform afferents.

The body already contains physically oscillating halteres driven by identified
hDVM motor output.  This module closes the missing sensory half of that loop:
actual MuJoCo haltere motion -> local inertial-strain proxy -> current into the
released MaleCNS haltere campaniform afferents.

Campaniform sensilla are strain sensors.  In this first local model, angular
acceleration of the haltere joint supplies an inertial-strain proxy.  The proxy
is normalized by the acceleration of the nominal 180-degree, 218-Hz haltere
stroke and half-wave rectified.  Individual sensillum orientation/field phase is
not yet available in MaleCNS annotations, so all afferents on one side share the
same preferred strain sign.  That assumption is isolated here rather than being
encoded as a steering/action rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import mujoco

from virtual_fly.reproducibility import HALTERE_FULL_KIND, validate_haltere_map


DEFAULT_MAP = Path("artifacts/malecns-v1.0/haltere-campaniform-sensory-v1.json")
NOMINAL_HALF_STROKE_RAD = math.radians(90.0)
# Calibrated from the current FlyBody haltere mechanics at the nominal
# 218-Hz, +/-90-degree stroke after discarding five transient wingbeats.
# The interaction-load v2 path uses the local haltere-body force-z component
# returned by MuJoCo's post-constraint RNE interaction wrench.
NOMINAL_LOCAL_FORCE_Z_MEAN = -0.00401078948225995
NOMINAL_LOCAL_FORCE_Z_HALF_AMPLITUDE = 0.06008164767518615
HALTERE_BODY_NAMES = {
    "left": "virtual_fly/l_haltere",
    "right": "virtual_fly/r_haltere",
}


@dataclass(frozen=True)
class HaltereSensorySnapshot:
    body_currents: tuple[tuple[int, float], ...]
    active_sensilla: int
    mean_current: float
    max_current: float
    strain_by_side: dict[str, float]
    angular_acceleration_by_side: dict[str, float]
    local_force_z_by_side: dict[str, float]
    transduction: str


class HaltereCampaniformSensor:
    def __init__(
        self,
        sensory_map: Path = DEFAULT_MAP,
        *,
        current_gain: float = 0.0,
        transduction: str = "angular-acceleration-v1",
        expected_kind: str = HALTERE_FULL_KIND,
        snapshot: Path | None = None,
    ) -> None:
        validation = validate_haltere_map(Path(sensory_map), expected_kind, snapshot=snapshot)
        payload = json.loads(Path(sensory_map).read_text(encoding="utf-8"))
        self.map_kind = str(validation["kind"])
        self.map_sha256 = str(validation["sha256"])
        by_side = payload.get("body_ids_by_side") or {}
        self.body_ids_by_side = {
            "left": tuple(int(v) for v in by_side.get("left", ())),
            "right": tuple(int(v) for v in by_side.get("right", ())),
        }
        if not self.body_ids_by_side["left"] or not self.body_ids_by_side["right"]:
            raise ValueError("haltere sensory map must contain both sides")
        if not math.isfinite(current_gain) or current_gain < 0.0:
            raise ValueError("haltere current gain must be finite and >= 0")
        if transduction not in {"angular-acceleration-v1", "interaction-load-v2"}:
            raise ValueError(
                "haltere transduction must be angular-acceleration-v1 or interaction-load-v2"
            )
        self.current_gain = float(current_gain)
        self.transduction = str(transduction)
        self._previous_velocity = {"left": 0.0, "right": 0.0}
        self._initialized = {"left": False, "right": False}
        self._haltere_body_ids: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.current_gain > 0.0

    @property
    def body_ids(self) -> tuple[int, ...]:
        return self.body_ids_by_side["left"] + self.body_ids_by_side["right"]

    @staticmethod
    def _joint_name(side: str) -> str:
        return "c_thorax-l_haltere-pitch" if side == "left" else "c_thorax-r_haltere-pitch"

    def reset(self, body) -> None:
        self._haltere_body_ids = {}
        for side in ("left", "right"):
            _, qvel_address = body.joint_state_addresses(self._joint_name(side))
            self._previous_velocity[side] = float(body.sim.mj_data.qvel[qvel_address])
            self._initialized[side] = True
            body_id = int(
                mujoco.mj_name2id(
                    body.sim.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    HALTERE_BODY_NAMES[side],
                )
            )
            if body_id < 0:
                raise RuntimeError(f"missing physical haltere body for {side}")
            self._haltere_body_ids[side] = body_id

    def encode(self, body, *, dt_s: float) -> HaltereSensorySnapshot:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("haltere sensory dt_s must be finite and positive")
        omega = 2.0 * math.pi * float(body.wingbeat_hz)
        reference_acceleration = NOMINAL_HALF_STROKE_RAD * omega * omega
        currents: list[tuple[int, float]] = []
        strains: dict[str, float] = {}
        accelerations: dict[str, float] = {}
        local_force_z: dict[str, float] = {}

        if self.transduction == "interaction-load-v2":
            # Compute the final body-level interaction wrench for the current
            # MuJoCo state. cfrc_int is the force/torque transmitted between a
            # body and its parent; rotate the force into the haltere body frame
            # before taking the local z component used by this first load model.
            mujoco.mj_rnePostConstraint(body.sim.mj_model, body.sim.mj_data)

        for side in ("left", "right"):
            _, qvel_address = body.joint_state_addresses(self._joint_name(side))
            velocity = float(body.sim.mj_data.qvel[qvel_address])
            if not self._initialized[side]:
                acceleration = 0.0
                self._initialized[side] = True
            else:
                acceleration = (velocity - self._previous_velocity[side]) / dt_s
            self._previous_velocity[side] = velocity
            accelerations[side] = acceleration

            if self.transduction == "interaction-load-v2":
                body_id = self._haltere_body_ids.get(side)
                if body_id is None:
                    raise RuntimeError("haltere sensor must be reset before load encoding")
                data = body.sim.mj_data
                xmat = data.xmat[body_id]
                force = data.cfrc_int[body_id][3:]
                # xmat is the local->world rotation.  The third column is the
                # local-z axis expressed in world coordinates, so dot(force,z)
                # gives the signed local load without side-specific hand signs.
                force_z = (
                    float(force[0]) * float(xmat[2])
                    + float(force[1]) * float(xmat[5])
                    + float(force[2]) * float(xmat[8])
                )
                local_force_z[side] = force_z
                normalized = (
                    force_z - NOMINAL_LOCAL_FORCE_Z_MEAN
                ) / NOMINAL_LOCAL_FORCE_Z_HALF_AMPLITUDE
            else:
                local_force_z[side] = 0.0
                # Historical local inertial-reaction proxy.  Retained as v1 for
                # exact reproduction of earlier experiments.
                normalized = -acceleration / max(reference_acceleration, 1e-12)

            # One preferred strain sign is used until field-specific orientation
            # data are available for these MaleCNS afferents.
            strain = min(1.0, max(0.0, normalized))
            strains[side] = strain
            current = self.current_gain * strain
            if current > 0.0:
                currents.extend((body_id, current) for body_id in self.body_ids_by_side[side])

        values = [current for _, current in currents]
        return HaltereSensorySnapshot(
            body_currents=tuple(currents),
            active_sensilla=len(currents),
            mean_current=(sum(values) / len(values) if values else 0.0),
            max_current=(max(values) if values else 0.0),
            strain_by_side=strains,
            angular_acceleration_by_side=accelerations,
            local_force_z_by_side=local_force_z,
            transduction=self.transduction,
        )
