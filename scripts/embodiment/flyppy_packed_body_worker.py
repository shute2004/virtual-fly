#!/usr/bin/env python3
"""Packed process-isolated Flyppy body workers.

A small number of spawned processes each own several fully independent FlyBody
slots.  Every MuJoCo model, retinal state, peripheral state, and course state
remains slot-local.  Slots inside one worker are executed sequentially on that
worker's main thread; different workers run concurrently.

The parent owns the shared MaleCNS runtime.  The IPC boundary is unchanged:

    child -> parent: retinal body currents
    parent -> child: individual motor-neuron spikes

By default the child keeps the FlyGym raster compound-eye oracle.  Setting
``VF_FLYPPY_VISION_MODE=direct-ray`` switches only the sensory acquisition step
to the image-free weighted ``mj_multiRay`` implementation.  The direct ray path
excludes each eye camera's owner body so the camera marker cannot self-occlude
the receptor rays.  The existing MaleCNS adaptation/transduction path is reused
unchanged.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import traceback
from types import SimpleNamespace
from typing import Any, Iterable


def assignments(population: int, process_count: int) -> list[tuple[int, ...]]:
    if population < 1:
        raise ValueError("population must be positive")
    if process_count < 1:
        raise ValueError("process_count must be positive")
    actual = min(int(population), int(process_count))
    groups: list[list[int]] = [[] for _ in range(actual)]
    for slot_id in range(population):
        groups[slot_id % actual].append(slot_id)
    return [tuple(group) for group in groups if group]


def default_process_count(population: int) -> int:
    """Measured M1 packing rule: at most four MuJoCo owner processes."""

    return min(int(population), 4)


def _worker_args(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        seed=int(config["seed"]),
        fixed_course_seed=(
            None if config.get("fixed_course_seed") is None else int(config["fixed_course_seed"])
        ),
        gate_count=int(config["gate_count"]),
        environment_version=str(config.get("environment_version", "v3")),
        flight_body_version=str(config.get("flight_body_version", "v3")),
        vertical_steering_gain=float(config.get("vertical_steering_gain", 1.0)),
        measured_steering_gain=float(config.get("measured_steering_gain", 1.0)),
        neutral_trim_strength=float(config.get("neutral_trim_strength", 1.0)),
        steering_tau_ms=float(config.get("steering_tau_ms", 12.0)),
        steering_spike_increment=float(config.get("steering_spike_increment", 0.85)),
        wing_motor_map=Path(config["wing_motor_map"]),
        body_motor_map=Path(config["body_motor_map"]),
        retinotopic_map=Path(config["retinotopic_map"]),
        haltere_sensory_map=Path(config.get("haltere_sensory_map", "artifacts/malecns-v1.0/haltere-campaniform-sensory-v1.json")),
        photoreceptor_current_gain=float(config["photoreceptor_current_gain"]),
        haltere_current_gain=float(config.get("haltere_current_gain", 0.0)),
        haltere_transduction=str(config.get("haltere_transduction", "angular-acceleration-v1")),
        capture_physics_trace=bool(config.get("capture_physics_trace", False)),
    )


def _vision_config() -> tuple[str, int]:
    mode = os.environ.get("VF_FLYPPY_VISION_MODE", "raster").strip().lower()
    aliases = {
        "raster": "raster",
        "flygym": "raster",
        "reference": "raster",
        "direct": "direct-ray",
        "ray": "direct-ray",
        "direct-ray": "direct-ray",
    }
    try:
        resolved = aliases[mode]
    except KeyError as exc:
        raise ValueError(
            "VF_FLYPPY_VISION_MODE must be raster or direct-ray"
        ) from exc
    raw_rays = os.environ.get("VF_FLYPPY_OMMATIDIA_RAYS", "7").strip()
    try:
        rays = int(raw_rays)
    except ValueError as exc:
        raise ValueError("VF_FLYPPY_OMMATIDIA_RAYS must be a positive integer") from exc
    if rays < 1:
        raise ValueError("VF_FLYPPY_OMMATIDIA_RAYS must be a positive integer")
    return resolved, rays


def _worker_main(
    connection,
    slot_ids: tuple[int, ...],
    physics_steps: int,
    config: dict[str, Any],
) -> None:
    try:
        # MuJoCo/CGL objects are created and used only in this spawned process's
        # main thread. This avoids the macOS native crash seen with renderer
        # calls from Python worker threads.
        import train_flyppy_population as trainer
        from direct_ommatidia_sensor_bodyexclude import (
            BodyExcludedDirectOmmatidialSensor,
        )
        from virtual_fly.training.curriculum import SpawnCondition

        worker_args = _worker_args(config)
        slots = {
            slot_id: trainer.make_slot(worker_args, slot_id)
            for slot_id in slot_ids
        }
        vision_mode, rays_per_ommatidium = _vision_config()
        direct_sensors = (
            {
                slot_id: BodyExcludedDirectOmmatidialSensor(
                    slot.vision,
                    rays_per_ommatidium=rays_per_ommatidium,
                )
                for slot_id, slot in slots.items()
            }
            if vision_mode == "direct-ray"
            else {}
        )
        body_ids = {
            slot_id: tuple(int(value) for value in slot.periphery.body_ids)
            for slot_id, slot in slots.items()
        }
        control_dts = {
            round(float(slot.body.timestep) * int(physics_steps), 15)
            for slot in slots.values()
        }
        if len(control_dts) != 1:
            raise RuntimeError(f"packed slots disagree on control dt: {control_dts}")
        control_dt_s = float(next(iter(control_dts)))

        connection.send(
            (
                "ready",
                {
                    "slot_ids": slot_ids,
                    "body_ids": body_ids,
                    "control_dt_s": control_dt_s,
                    "vision_mode": vision_mode,
                    "vision_rays_per_ommatidium": rays_per_ommatidium,
                },
            )
        )

        while True:
            request = connection.recv()
            command = str(request[0])
            payload = request[1] if len(request) > 1 else None

            if command == "reset":
                replies: dict[int, dict[str, Any]] = {}
                for raw_id, reset_payload in dict(payload).items():
                    slot_id = int(raw_id)
                    slot = slots[slot_id]
                    condition = SpawnCondition(
                        float(reset_payload["spawn_x_mm"]),
                        float(reset_payload["spawn_z_mm"]),
                        float(reset_payload["initial_speed_mm_s"]),
                    )
                    trainer.begin_episode(
                        slot,
                        episode=int(reset_payload["episode"]),
                        source_weight_version=int(reset_payload["source_weight_version"]),
                        condition=condition,
                        initial_vz_mm_s=float(reset_payload.get("initial_vz_mm_s", 0.0)),
                        course_start_gate_index=(
                            None
                            if reset_payload.get("course_start_gate_index") is None
                            else int(reset_payload["course_start_gate_index"])
                        ),
                        gate_center_overrides={
                            int(key): float(value)
                            for key, value in dict(reset_payload.get("gate_center_overrides") or {}).items()
                        },
                    )
                    replies[slot_id] = {
                        "position": tuple(
                            float(value) for value in slot.body.thorax_position_mm()
                        ),
                        "velocity": tuple(
                            float(value)
                            for value in slot.body.root_linear_velocity_mm_s()
                        ),
                        "next_gate": int(slot.course.absolute_next_gate_index),
                    }
                connection.send(("ok", replies))
                continue

            if command == "observe":
                replies: dict[int, dict[str, Any]] = {}
                for slot_id in tuple(int(value) for value in payload):
                    slot = slots[slot_id]
                    if vision_mode == "direct-ray":
                        eyes = direct_sensors[slot_id].read_eye_readouts(
                            slot.body.sim,
                            slot.body.fly,
                        )
                        retinal = slot.vision.encode_from_eye_readouts(eyes)
                    else:
                        retinal = slot.vision.encode(slot.body.sim, slot.body.fly)
                    haltere = slot.haltere_sensor.encode(slot.body, dt_s=control_dt_s)
                    replies[slot_id] = {
                        "body_currents": (*retinal.body_currents, *haltere.body_currents),
                        "retinal_body_currents": tuple(retinal.body_currents),
                        "haltere_body_currents": tuple(haltere.body_currents),
                        "active_photoreceptors": int(retinal.active_photoreceptors),
                        "active_columns": int(retinal.active_columns),
                        "mean_current": float(retinal.mean_current),
                        "max_current": float(retinal.max_current),
                        "haltere_active_sensilla": int(haltere.active_sensilla),
                        "haltere_mean_current": float(haltere.mean_current),
                        "haltere_max_current": float(haltere.max_current),
                        "haltere_strain_by_side": dict(haltere.strain_by_side),
                        "haltere_angular_acceleration_by_side": dict(haltere.angular_acceleration_by_side),
                    }
                connection.send(("ok", replies))
                continue

            if command == "act":
                replies: dict[int, dict[str, Any]] = {}
                for raw_id, active_values in dict(payload).items():
                    slot_id = int(raw_id)
                    slot = slots[slot_id]
                    active_ids = frozenset(int(value) for value in active_values)
                    spikes = {
                        body_id: body_id in active_ids
                        for body_id in body_ids[slot_id]
                    }
                    peripheral = slot.periphery.step(spikes, dt_s=control_dt_s)
                    physics_trace: list[dict[str, object]] | None = (
                        [] if bool(worker_args.capture_physics_trace) else None
                    )
                    physical_collision, physics_steps_completed = slot.world.step_muscles_until_boundary_contact(
                        slot.body,
                        peripheral,
                        physics_steps=physics_steps,
                        trace=physics_trace,
                    )

                    position = slot.body.thorax_position_mm()
                    velocity = slot.body.root_linear_velocity_mm_s()
                    gate_observation = slot.course.observe(float(position[0]), float(position[2]))
                    gate_miss_distance_mm = max(
                        float(gate_observation.gap_low_dz_mm),
                        -float(gate_observation.gap_high_dz_mm),
                        0.0,
                    )
                    _, collision_geom = slot.world.physical_collision_detail(slot.body.sim)
                    body_min_x_mm = None
                    body_max_x_mm = None
                    if slot.course.needs_full_body_x_sample(float(position[0])):
                        body_min_x_mm, body_max_x_mm = slot.world.full_body_x_bounds_mm(slot.body.sim)
                    event = slot.course.update(
                        float(position[0]),
                        float(position[2]),
                        physical_collision_reason=physical_collision,
                        analytic_body_collision=False,
                        body_min_x_mm=body_min_x_mm,
                    )
                    replies[slot_id] = {
                        "position": tuple(float(value) for value in position),
                        "velocity": tuple(float(value) for value in velocity),
                        "passed_gate": bool(event.passed_gate),
                        "collision": bool(event.collision),
                        "collision_reason": event.collision_reason,
                        "gate_miss_distance_mm": float(gate_miss_distance_mm),
                        "finished": bool(event.finished),
                        "next_gate": int(
                            getattr(
                                slot.course,
                                "absolute_next_gate_index",
                                slot.course.next_gate_index,
                            )
                        ),
                        "motor": peripheral.compact_diagnostics(),
                        "physics_steps_completed": int(physics_steps_completed),
                        "body_min_x_mm": None if body_min_x_mm is None else float(body_min_x_mm),
                        "body_max_x_mm": None if body_max_x_mm is None else float(body_max_x_mm),
                        "collision_geom": collision_geom,
                        "physics_trace": physics_trace,
                    }
                connection.send(("ok", replies))
                continue

            if command == "snapshot":
                replies: dict[int, dict[str, Any]] = {}
                for slot_id in tuple(int(value) for value in payload):
                    sim = slots[slot_id].body.sim
                    replies[slot_id] = {
                        "sim_time_s": float(sim.mj_data.time),
                        "qpos": [float(value) for value in sim.mj_data.qpos],
                        "qvel": [float(value) for value in sim.mj_data.qvel],
                    }
                connection.send(("ok", replies))
                continue

            if command == "close":
                for slot in slots.values():
                    slot.body.sim.eye_renderer = None
                connection.send(("closed", None))
                return

            raise RuntimeError(f"unknown packed Flyppy worker command {command!r}")
    except BaseException as exc:
        try:
            connection.send(
                (
                    "error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        except BaseException:
            pass
    finally:
        try:
            connection.close()
        except BaseException:
            pass


class PackedFlyppyBodyProcess:
    def __init__(
        self,
        context: mp.context.BaseContext,
        *,
        slot_ids: tuple[int, ...],
        physics_steps: int,
        timeout_s: float,
        config: dict[str, Any],
    ) -> None:
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.slot_ids = tuple(int(value) for value in slot_ids)
        self.timeout_s = float(timeout_s)
        self.process = context.Process(
            target=_worker_main,
            args=(child, self.slot_ids, physics_steps, dict(config)),
            name="flyppy-packed-" + "-".join(str(value) for value in self.slot_ids),
        )
        self.process.start()
        child.close()
        status, payload = self._recv("startup")
        if status != "ready":
            raise RuntimeError(
                f"packed worker {self.slot_ids} failed startup: {status}: {payload}"
            )
        self.body_ids = {
            int(key): tuple(int(value) for value in values)
            for key, values in payload["body_ids"].items()
        }
        self.control_dt_s = float(payload["control_dt_s"])
        self.vision_mode = str(payload.get("vision_mode", "raster"))
        self.vision_rays_per_ommatidium = int(
            payload.get("vision_rays_per_ommatidium", 7)
        )

    def _recv(self, phase: str):
        if not self.connection.poll(self.timeout_s):
            exitcode = self.process.exitcode
            if exitcode is not None:
                raise RuntimeError(
                    f"packed worker {self.slot_ids} died during {phase}; exitcode={exitcode}"
                )
            self.process.terminate()
            self.process.join(timeout=5.0)
            raise RuntimeError(
                f"packed worker {self.slot_ids} timed out during {phase} "
                f"after {self.timeout_s:.1f}s"
            )
        try:
            return self.connection.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError) as exc:
            self.process.join(timeout=1.0)
            raise RuntimeError(
                f"packed worker {self.slot_ids} native process ended during {phase}; "
                f"exitcode={self.process.exitcode}"
            ) from exc

    def request(self, command: str, payload=None) -> None:
        if not self.process.is_alive():
            raise RuntimeError(
                f"packed worker {self.slot_ids} is not alive; "
                f"exitcode={self.process.exitcode}"
            )
        self.connection.send((command,) if payload is None else (command, payload))

    def receive(self, phase: str):
        status, payload = self._recv(phase)
        if status == "error":
            raise RuntimeError(f"packed worker {self.slot_ids} error: {payload}")
        if status not in {"ok", "closed"}:
            raise RuntimeError(
                f"packed worker {self.slot_ids} unexpected response {status!r}"
            )
        return payload

    def call(self, command: str, payload=None):
        self.request(command, payload)
        return self.receive(command)

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.request("close")
                self.receive("close")
            except Exception:
                self.process.terminate()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2.0)
        try:
            self.connection.close()
        except Exception:
            pass


def spawn_packed_body_processes(
    *,
    population: int,
    process_count: int | None,
    physics_steps: int,
    timeout_s: float,
    config: dict[str, Any],
) -> tuple[list[PackedFlyppyBodyProcess], dict[int, PackedFlyppyBodyProcess]]:
    count = default_process_count(population) if process_count is None else int(process_count)
    context = mp.get_context("spawn")
    workers = [
        PackedFlyppyBodyProcess(
            context,
            slot_ids=group,
            physics_steps=physics_steps,
            timeout_s=timeout_s,
            config=config,
        )
        for group in assignments(population, count)
    ]
    slot_to_worker = {
        slot_id: worker
        for worker in workers
        for slot_id in worker.slot_ids
    }
    return workers, slot_to_worker


def active_by_worker(
    workers: Iterable[PackedFlyppyBodyProcess],
    active_slot_ids: Iterable[int],
) -> dict[PackedFlyppyBodyProcess, tuple[int, ...]]:
    active = frozenset(int(value) for value in active_slot_ids)
    return {
        worker: tuple(slot_id for slot_id in worker.slot_ids if slot_id in active)
        for worker in workers
    }
