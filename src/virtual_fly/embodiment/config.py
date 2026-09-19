"""Typed configuration for one physical Flyppy body stack.

The configuration describes only body/sensory/peripheral construction. Training,
curriculum, reinforcement and shared-weight scheduling live in ``virtual_fly.training``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from virtual_fly.reproducibility import HALTERE_FULL_KIND
from virtual_fly.paths import PRODUCTION_NEUTRAL_TRIM_PATTERN


@dataclass(frozen=True)
class FlyppyBodyConfig:
    snapshot: Path
    seed: int
    fixed_course_seed: int | None
    gate_count: int
    environment_version: str
    flight_body_version: str
    vertical_steering_gain: float
    measured_steering_gain: float
    neutral_trim_strength: float
    neutral_trim_pattern: Path
    steering_tau_ms: float
    steering_spike_increment: float
    wing_motor_map: Path
    body_motor_map: Path
    retinotopic_map: Path
    haltere_sensory_map: Path
    haltere_sensory_kind: str
    photoreceptor_current_gain: float
    haltere_current_gain: float
    haltere_transduction: str

    @classmethod
    def from_namespace(cls, args: Any) -> "FlyppyBodyConfig":
        return cls(
            snapshot=Path(args.snapshot),
            seed=int(args.seed),
            fixed_course_seed=(
                None
                if getattr(args, "fixed_course_seed", None) is None
                else int(args.fixed_course_seed)
            ),
            gate_count=int(args.gate_count),
            environment_version=str(args.environment_version),
            flight_body_version=str(getattr(args, "flight_body_version", "v3")),
            vertical_steering_gain=float(getattr(args, "vertical_steering_gain", 1.0)),
            measured_steering_gain=float(getattr(args, "measured_steering_gain", 1.0)),
            neutral_trim_strength=float(getattr(args, "neutral_trim_strength", 1.0)),
            neutral_trim_pattern=Path(getattr(args, "neutral_trim_pattern", PRODUCTION_NEUTRAL_TRIM_PATTERN)),
            steering_tau_ms=float(getattr(args, "steering_tau_ms", 12.0)),
            steering_spike_increment=float(getattr(args, "steering_spike_increment", 0.85)),
            wing_motor_map=Path(args.wing_motor_map),
            body_motor_map=Path(args.body_motor_map),
            retinotopic_map=Path(args.retinotopic_map),
            haltere_sensory_map=Path(
                getattr(
                    args,
                    "haltere_sensory_map",
                    "artifacts/malecns-v1.0/haltere-campaniform-sensory-v1.json",
                )
            ),
            haltere_sensory_kind=str(
                getattr(args, "haltere_sensory_kind", HALTERE_FULL_KIND)
            ),
            photoreceptor_current_gain=float(args.photoreceptor_current_gain),
            haltere_current_gain=float(getattr(args, "haltere_current_gain", 0.0)),
            haltere_transduction=str(
                getattr(args, "haltere_transduction", "angular-acceleration-v1")
            ),
        )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "FlyppyBodyConfig":
        class Namespace:
            pass

        args = Namespace()
        defaults: dict[str, Any] = {
            "fixed_course_seed": None,
            "environment_version": "v3",
            "flight_body_version": "v3",
            "vertical_steering_gain": 1.0,
            "measured_steering_gain": 1.0,
            "neutral_trim_strength": 1.0,
            "neutral_trim_pattern": PRODUCTION_NEUTRAL_TRIM_PATTERN,
            "steering_tau_ms": 12.0,
            "steering_spike_increment": 0.85,
            "haltere_sensory_map": "artifacts/malecns-v1.0/haltere-campaniform-sensory-v1.json",
            "haltere_sensory_kind": HALTERE_FULL_KIND,
            "haltere_current_gain": 0.0,
            "haltere_transduction": "angular-acceleration-v1",
        }
        for key, value in {**defaults, **dict(config)}.items():
            setattr(args, key, value)
        return cls.from_namespace(args)

    def to_worker_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "snapshot",
            "wing_motor_map",
            "body_motor_map",
            "retinotopic_map",
            "haltere_sensory_map",
            "neutral_trim_pattern",
        ):
            payload[key] = str(payload[key])
        return payload
