#!/usr/bin/env python3
"""Biophysical normalization for the canonical virtual-fly body.

FlyGym 2.1.0 already supplies per-joint anatomical ranges, stiffness and damping
for FlyBody.  Those ranges remain the default.  Only two clearly documented
flight-related mismatches are overridden here:

* haltere pitch: FlyBody's +/-0.2 rad placeholder is much smaller than the
  roughly 140--220 degree peak-to-peak natural stroke reported for Drosophila;
  use +/-110 degrees as the mechanical range;
* head yaw: FlyBody's +/-0.2 rad is narrower than reported Drosophila flight
  head rotations; use +/-25 degrees as a conservative mechanical limit.

The body mass is normalized to the canonical adult-female value while preserving
FlyBody's relative segment mass and principal-inertia distribution.  We do not
invent independent masses for unresolved anatomical parts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco as mj
import numpy as np

from virtual_fly.physics import CANONICAL_FLY


HALTERE_HALF_RANGE_RAD = math.radians(110.0)
HEAD_YAW_HALF_RANGE_RAD = math.radians(25.0)

# Key format matches FlyBody joint-DOF names: parent-child-axis.
JOINT_RANGE_OVERRIDES_RAD: dict[str, tuple[float, float]] = {
    "c_thorax-l_haltere-pitch": (-HALTERE_HALF_RANGE_RAD, HALTERE_HALF_RANGE_RAD),
    "c_thorax-r_haltere-pitch": (-HALTERE_HALF_RANGE_RAD, HALTERE_HALF_RANGE_RAD),
    "c_thorax-c_head-yaw": (-HEAD_YAW_HALF_RANGE_RAD, HEAD_YAW_HALF_RANGE_RAD),
}

JOINT_RANGE_PROVENANCE: dict[str, str] = {
    "c_thorax-l_haltere-pitch": (
        "literature: Drosophila haltere stroke amplitude is approximately "
        "140-220 degrees peak-to-peak; mechanical limit set to +/-110 degrees"
    ),
    "c_thorax-r_haltere-pitch": (
        "literature: Drosophila haltere stroke amplitude is approximately "
        "140-220 degrees peak-to-peak; mechanical limit set to +/-110 degrees"
    ),
    "c_thorax-c_head-yaw": (
        "literature: tethered flying Drosophila can rotate the head to roughly "
        "+/-25 degrees in yaw; typical active excursions can be smaller"
    ),
}


@dataclass(frozen=True)
class MassNormalization:
    source_mass_g: float
    target_mass_g: float
    scale: float
    body_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "source_mass_g": self.source_mass_g,
            "source_mass_mg": self.source_mass_g * 1000.0,
            "target_mass_g": self.target_mass_g,
            "target_mass_mg": self.target_mass_g * 1000.0,
            "scale": self.scale,
            "body_count": self.body_count,
        }


def dof_key(dof) -> str:
    return f"{dof.parent.name}-{dof.child.name}-{dof.axis.value}"


def apply_researched_joint_ranges(fly) -> dict[str, dict[str, object]]:
    """Apply only literature-supported corrections to FlyBody joint ranges.

    All other biological joint ranges are retained exactly as supplied by
    FlyGym 2.1.0's ``joints.yaml``.  The returned dictionary is suitable for
    provenance logging.
    """

    applied: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for dof, joint in fly.jointdof_to_mjcfjoint.items():
        key = dof_key(dof)
        if key not in JOINT_RANGE_OVERRIDES_RAD:
            continue
        low, high = JOINT_RANGE_OVERRIDES_RAD[key]
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise RuntimeError(f"invalid researched joint range for {key}")
        # MjSpec exposes ``range`` as a mutable length-2 array.
        joint.range[0] = low
        joint.range[1] = high
        seen.add(key)
        applied[key] = {
            "range_rad": [low, high],
            "range_deg": [math.degrees(low), math.degrees(high)],
            "provenance": JOINT_RANGE_PROVENANCE[key],
        }

    missing = sorted(set(JOINT_RANGE_OVERRIDES_RAD) - seen)
    if missing:
        raise RuntimeError(f"researched FlyBody joints not found: {missing}")
    return applied


def _fly_body_ids(sim, fly) -> np.ndarray:
    ids: list[int] = []
    for segment in fly.get_bodysegs_order():
        mjcf_body = fly.bodyseg_to_mjcfbody[segment]
        body_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_BODY, mjcf_body.name)
        if body_id < 0:
            raise RuntimeError(f"compiled FlyBody segment not found: {mjcf_body.name}")
        ids.append(int(body_id))
    unique = np.asarray(sorted(set(ids)), dtype=np.int32)
    if unique.size == 0 or np.any(unique == 0):
        raise RuntimeError("failed to resolve non-world FlyBody body IDs")
    return unique


def normalize_fly_mass(sim, fly, *, target_mass_g: float | None = None) -> MassNormalization:
    """Normalize total FlyBody mass while preserving segment mass/inertia ratios.

    FlyBody's source flight model uses centimetres and g/cm^3 densities with
    gravity -981 cm/s^2.  FlyGym converts lengths to millimetres while mass is
    unchanged, so compiled body masses are in grams.  Both body mass and
    principal inertia are multiplied by the same scalar and MuJoCo constants are
    recomputed afterwards.
    """

    target = CANONICAL_FLY.total_mass_g if target_mass_g is None else float(target_mass_g)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_mass_g must be finite and positive")

    body_ids = _fly_body_ids(sim, fly)
    source = float(np.sum(np.asarray(sim.mj_model.body_mass)[body_ids]))
    if not math.isfinite(source) or source <= 0.0:
        raise RuntimeError(f"invalid compiled FlyBody mass: {source}")

    scale = target / source
    sim.mj_model.body_mass[body_ids] *= scale
    sim.mj_model.body_inertia[body_ids, :] *= scale
    mj.mj_setConst(sim.mj_model, sim.mj_data)
    mj.mj_forward(sim.mj_model, sim.mj_data)

    normalized = float(np.sum(np.asarray(sim.mj_model.body_mass)[body_ids]))
    if not math.isclose(normalized, target, rel_tol=2e-6, abs_tol=1e-12):
        raise RuntimeError(
            f"FlyBody mass normalization failed: expected {target}, got {normalized}"
        )
    return MassNormalization(
        source_mass_g=source,
        target_mass_g=target,
        scale=scale,
        body_count=int(body_ids.size),
    )


def compiled_joint_ranges(sim, fly) -> dict[str, tuple[float, float]]:
    """Return compiled limited biological hinge ranges for diagnostics."""

    result: dict[str, tuple[float, float]] = {}
    for dof, joint in fly.jointdof_to_mjcfjoint.items():
        joint_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_JOINT, joint.name)
        if joint_id < 0:
            raise RuntimeError(f"compiled FlyBody joint not found: {joint.name}")
        if bool(sim.mj_model.jnt_limited[joint_id]):
            low, high = sim.mj_model.jnt_range[joint_id]
            result[dof_key(dof)] = (float(low), float(high))
    return result
