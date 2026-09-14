#!/usr/bin/env python3
"""Whole-body FlyBody mechanics for grounded MaleCNS motor outputs.

Wing mechanics are inherited unchanged from ``FlyBodyMuscleAdapter``.  Identified
leg antagonist muscles apply generalized joint torque at the corresponding
FlyBody biological joint.  hDVM activity powers an antiphase haltere oscillator.
No environment state, target trajectory, action decoder, or neural population
average is used.
"""

from __future__ import annotations

import math

import mujoco as mj

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from whole_body_periphery import WholeBodyPeripheralSnapshot


# Calibrated mechanical scales, anchored to FlyBody 2.1.0's position-actuator
# gain magnitudes for the same leg joint families.  They are not measured muscle
# forces and are kept explicit for future replacement by moment-arm/force data.
LEG_TORQUE_SCALE = {
    "thorax_coxa_yaw": 80.0,
    "coxa_femur_pitch": 80.0,
    "femur_tibia_pitch": 40.0,
}
LEG_LIMIT_SOFT_ZONE_FRACTION = 0.10

HALTERE_NOMINAL_HALF_STROKE_RAD = math.radians(90.0)
HALTERE_VIRTUAL_KP = 20.0
HALTERE_VIRTUAL_KD = 0.02
HALTERE_MAX_ABS_TORQUE = 200.0


class FlyBodyNeuromuscularAdapter(FlyBodyMuscleAdapter):
    """Apply wing, grounded leg, and hDVM haltere motor output physically."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._protraction_sign = {
            leg: self._infer_protraction_sign(leg)
            for leg in ("lf", "lm", "lh", "rf", "rm", "rh")
        }
        self.last_somatic_torque: dict[str, float] = {}

    @staticmethod
    def _joint_name(leg: str, role: str) -> str:
        if role == "thorax_coxa_yaw":
            return f"c_thorax-{leg}_coxa-yaw"
        if role == "coxa_femur_pitch":
            return f"{leg}_coxa-{leg}_trochanterfemur-pitch"
        if role == "femur_tibia_pitch":
            return f"{leg}_trochanterfemur-{leg}_tibia-pitch"
        raise KeyError(role)

    def _infer_protraction_sign(self, leg: str) -> float:
        """Infer which local coxa-yaw sign moves the distal leg forward (+x)."""

        name = self._joint_name(leg, "thorax_coxa_yaw")
        qpos_address, _ = self.joint_state_addresses(name)
        low, high = self.joint_range_rad(name)
        original = float(self.sim.mj_data.qpos[qpos_address])
        epsilon = min(0.03, 0.05 * (high - low))
        plus = min(high - 1e-6, original + epsilon)
        minus = max(low + 1e-6, original - epsilon)
        if plus <= minus:
            raise RuntimeError(f"cannot perturb coxa yaw for {leg}")

        self.sim.mj_data.qpos[qpos_address] = plus
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)
        x_plus = float(self.body_segment_position_mm(f"{leg}_tarsus5")[0])
        self.sim.mj_data.qpos[qpos_address] = minus
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)
        x_minus = float(self.body_segment_position_mm(f"{leg}_tarsus5")[0])
        self.sim.mj_data.qpos[qpos_address] = original
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

        delta = x_plus - x_minus
        if not math.isfinite(delta) or abs(delta) < 1e-7:
            raise RuntimeError(
                f"FlyBody geometry could not resolve protraction sign for {leg}: dx={delta}"
            )
        return 1.0 if delta > 0.0 else -1.0

    def _limit_taper(self, joint_name: str, torque: float) -> float:
        if torque == 0.0:
            return 0.0
        qpos_address, _ = self.joint_state_addresses(joint_name)
        angle = float(self.sim.mj_data.qpos[qpos_address])
        low, high = self.joint_range_rad(joint_name)
        width = high - low
        soft = max(1e-6, width * LEG_LIMIT_SOFT_ZONE_FRACTION)
        distance = high - angle if torque > 0.0 else angle - low
        factor = min(1.0, max(0.0, distance / soft))
        return torque * factor

    def _apply_leg_torque(
        self, state: WholeBodyPeripheralSnapshot
    ) -> dict[str, float]:
        applied: dict[str, float] = {}
        for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
            for role, scale in LEG_TORQUE_SCALE.items():
                drive = float(state.leg_drive.get(f"{leg}:{role}", 0.0))
                if drive == 0.0:
                    continue
                joint_name = self._joint_name(leg, role)
                # For pitch DOFs, FlyBody's source classes define positive as
                # extension.  For coxa yaw, geometry determines the side-specific
                # direction that physically moves the distal leg forward.
                local_drive = (
                    drive * self._protraction_sign[leg]
                    if role == "thorax_coxa_yaw"
                    else drive
                )
                torque = self._limit_taper(joint_name, local_drive * scale)
                _, qvel_address = self.joint_state_addresses(joint_name)
                self.sim.mj_data.qfrc_applied[qvel_address] += torque
                applied[joint_name] = torque
        return applied

    def _apply_haltere_torque(
        self, state: WholeBodyPeripheralSnapshot, phase: float
    ) -> dict[str, float]:
        applied: dict[str, float] = {}
        omega = 2.0 * math.pi * self.wingbeat_hz
        # Natural haltere motion is approximately antiphase with the wings.
        target = HALTERE_NOMINAL_HALF_STROKE_RAD * math.sin(phase + math.pi)
        target_velocity = (
            HALTERE_NOMINAL_HALF_STROKE_RAD
            * omega
            * math.cos(phase + math.pi)
        )
        for side, prefix in (("left", "l"), ("right", "r")):
            power = float(state.haltere_power.get(side, 0.0))
            if power <= 0.0:
                continue
            name = f"c_thorax-{prefix}_haltere-pitch"
            qpos_address, qvel_address = self.joint_state_addresses(name)
            angle = float(self.sim.mj_data.qpos[qpos_address])
            velocity = float(self.sim.mj_data.qvel[qvel_address])
            torque = power * (
                HALTERE_VIRTUAL_KP * (target - angle)
                + HALTERE_VIRTUAL_KD * (target_velocity - velocity)
            )
            torque = min(HALTERE_MAX_ABS_TORQUE, max(-HALTERE_MAX_ABS_TORQUE, torque))
            # Mechanical joint limits are +/-110 degrees; tapering close to the
            # limit avoids repeatedly driving the hard constraint.
            low, high = self.joint_range_rad(name)
            soft = (high - low) * 0.08
            distance = high - angle if torque > 0.0 else angle - low
            torque *= min(1.0, max(0.0, distance / max(soft, 1e-6)))
            self.sim.mj_data.qfrc_applied[qvel_address] += torque
            applied[name] = torque
        return applied

    def _apply_additional_muscle_torque(
        self, state, phase: float
    ) -> dict[str, float]:
        if not isinstance(state, WholeBodyPeripheralSnapshot):
            return {}
        applied = self._apply_leg_torque(state)
        applied.update(self._apply_haltere_torque(state, phase))
        self.last_somatic_torque = applied
        return applied

    def reset(self) -> None:
        super().reset()
        self.last_somatic_torque = {}
