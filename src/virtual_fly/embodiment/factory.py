"""Construction/reset of the physical Flyppy body stack.

This is the single production factory for body/environment/sensory/peripheral
composition. It contains no training policy, reinforcement schedule or neural
weight logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from virtual_fly.embodiment.body import BODY_ADAPTERS
from virtual_fly.embodiment.config import FlyppyBodyConfig
from virtual_fly.embodiment.course import FlyppyCourse
from virtual_fly.embodiment.haltere import HaltereCampaniformSensor
from virtual_fly.embodiment.periphery import WholeBodyPeriphery
from virtual_fly.embodiment.retina import MaleCNSRetina
from virtual_fly.embodiment.world import FlyppyWorld
from virtual_fly.physics import FLYPPY_GEOMETRY_V3, FLYPPY_GEOMETRY_V4
from virtual_fly.training.curriculum import SpawnCondition


@dataclass
class FlyppyBodyStack:
    course: FlyppyCourse
    world: FlyppyWorld
    body: Any
    periphery: WholeBodyPeriphery
    vision: MaleCNSRetina
    haltere_sensor: HaltereCampaniformSensor


def build_flyppy_stack(config: FlyppyBodyConfig, slot_id: int) -> FlyppyBodyStack:
    course_seed = (
        int(config.fixed_course_seed)
        if config.fixed_course_seed is not None
        else config.seed + int(slot_id)
    )
    course = FlyppyCourse(
        seed=course_seed,
        gate_count=config.gate_count,
        environment_version=config.environment_version,
    )
    world = FlyppyWorld(course)
    geometry = (
        FLYPPY_GEOMETRY_V4
        if config.environment_version in {"v4", "v5", "v6", "v7"}
        else FLYPPY_GEOMETRY_V3
    )
    body_version = config.flight_body_version
    body_cls = BODY_ADAPTERS[body_version]
    body_kwargs: dict[str, Any] = {
        "tethered": False,
        "world": world,
        "spawn_position_mm": (
            0.0,
            0.0,
            (geometry.corridor_low_z_mm + geometry.corridor_high_z_mm) / 2.0,
        ),
        "initial_linear_velocity_mm_s": (0.0, 0.0, 0.0),
        "enable_vision": True,
        "enable_observer_camera": False,
    }
    if body_version == "v5":
        body_kwargs["vertical_steering_gain"] = config.vertical_steering_gain
    if body_version in {"v6", "v8"}:
        body_kwargs["measured_steering_gain"] = config.measured_steering_gain
    if body_version in {"v7", "v8"}:
        body_kwargs["neutral_trim_strength"] = config.neutral_trim_strength
        body_kwargs["measured_wing_pattern"] = config.neutral_trim_pattern
    body = body_cls(**body_kwargs)
    return FlyppyBodyStack(
        course=course,
        world=world,
        body=body,
        periphery=WholeBodyPeriphery(
            config.wing_motor_map,
            config.body_motor_map,
            wing_steering_tau_s=config.steering_tau_ms / 1000.0,
            wing_steering_spike_increment=config.steering_spike_increment,
        ),
        vision=MaleCNSRetina(
            config.retinotopic_map,
            current_gain=config.photoreceptor_current_gain,
        ),
        haltere_sensor=HaltereCampaniformSensor(
            config.haltere_sensory_map,
            current_gain=config.haltere_current_gain,
            transduction=config.haltere_transduction,
            expected_kind=config.haltere_sensory_kind,
            snapshot=config.snapshot,
        ),
    )


def reset_flyppy_stack(
    stack: FlyppyBodyStack,
    condition: SpawnCondition,
    *,
    initial_vz_mm_s: float = 0.0,
    course_start_gate_index: int | None = None,
    gate_center_overrides: dict[int, float] | None = None,
) -> tuple[float, float, float]:
    if gate_center_overrides:
        for gate_index, center_z_mm in sorted(gate_center_overrides.items()):
            stack.world.set_gate_center_z_mm(
                stack.body.sim, int(gate_index), float(center_z_mm)
            )
    stack.course.reset(next_gate_index=course_start_gate_index)
    stack.body.reset()
    stack.body.set_root_position_mm((float(condition.x_mm), 0.0, float(condition.z_mm)))
    stack.body.set_root_linear_velocity_mm_s(
        (float(condition.speed_mm_s), 0.0, float(initial_vz_mm_s))
    )
    stack.periphery.reset()
    stack.vision.reset_adaptation()
    stack.haltere_sensor.reset(stack.body)
    return tuple(float(value) for value in stack.body.root_linear_velocity_mm_s())
