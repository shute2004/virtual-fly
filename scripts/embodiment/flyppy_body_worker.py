#!/usr/bin/env python3
"""Process-isolated Flyppy body worker for macOS-safe parallel embodiment.

Each spawned child owns one complete physical fly: FlyBody/MuJoCo, compound-eye
renderer, MaleCNS retinal transduction, whole-body periphery, and Flyppy course.
The parent owns the shared CNS runtime.  The IPC boundary is therefore exactly:

    child -> parent: retinal body currents
    parent -> child: individual motor-neuron spikes

No neural weights or CNS state live in a body worker.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import traceback
from types import SimpleNamespace
from typing import Any


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
        wing_motor_map=Path(config["wing_motor_map"]),
        body_motor_map=Path(config["body_motor_map"]),
        retinotopic_map=Path(config["retinotopic_map"]),
        haltere_sensory_map=Path(config.get("haltere_sensory_map", "artifacts/malecns-v1.0/haltere-campaniform-sensory-v1.json")),
        photoreceptor_current_gain=float(config["photoreceptor_current_gain"]),
        haltere_current_gain=float(config.get("haltere_current_gain", 0.0)),
        haltere_transduction=str(config.get("haltere_transduction", "angular-acceleration-v1")),
    )


def _worker_main(connection, slot_index: int, physics_steps: int, config: dict[str, Any]) -> None:
    try:
        # Import inside the spawned child so MuJoCo/CGL objects are created and
        # destroyed in this process's main thread.
        import train_flyppy_population as trainer
        from virtual_fly.training.curriculum import SpawnCondition

        slot = trainer.make_slot(_worker_args(config), slot_index)
        control_dt_s = float(slot.body.timestep) * int(physics_steps)
        body_ids = tuple(int(value) for value in slot.periphery.body_ids)

        def reset(payload: dict[str, Any]) -> dict[str, Any]:
            condition = SpawnCondition(
                float(payload["spawn_x_mm"]),
                float(payload["spawn_z_mm"]),
                float(payload["initial_speed_mm_s"]),
            )
            trainer.begin_episode(
                slot,
                episode=int(payload["episode"]),
                source_weight_version=int(payload["source_weight_version"]),
                condition=condition,
                initial_vz_mm_s=float(payload.get("initial_vz_mm_s", 0.0)),
                course_start_gate_index=(
                    None
                    if payload.get("course_start_gate_index") is None
                    else int(payload["course_start_gate_index"])
                ),
                gate_center_overrides={
                    int(key): float(value)
                    for key, value in dict(payload.get("gate_center_overrides") or {}).items()
                },
            )
            return {
                "body_ids": body_ids,
                "control_dt_s": control_dt_s,
                "position": tuple(float(value) for value in slot.body.thorax_position_mm()),
                "velocity": tuple(float(value) for value in slot.body.root_linear_velocity_mm_s()),
                "next_gate": int(slot.course.absolute_next_gate_index),
            }

        connection.send(("ready", {"body_ids": body_ids, "control_dt_s": control_dt_s}))
        while True:
            request = connection.recv()
            command = str(request[0])

            if command == "reset":
                connection.send(("ok", reset(dict(request[1]))))
                continue

            if command == "observe":
                retinal = slot.vision.encode(slot.body.sim, slot.body.fly)
                haltere = slot.haltere_sensor.encode(slot.body, dt_s=control_dt_s)
                connection.send(
                    (
                        "ok",
                        {
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
                        },
                    )
                )
                continue

            if command == "act":
                active_ids = frozenset(int(value) for value in request[1])
                spikes = {body_id: body_id in active_ids for body_id in body_ids}
                peripheral = slot.periphery.step(spikes, dt_s=control_dt_s)
                physical_collision, physics_steps_completed = slot.world.step_muscles_until_boundary_contact(
                    slot.body,
                    peripheral,
                    physics_steps=physics_steps,
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
                connection.send(
                    (
                        "ok",
                        {
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
                        },
                    )
                )
                continue

            if command == "snapshot":
                connection.send(
                    (
                        "ok",
                        {
                            "sim_time_s": float(slot.body.sim.mj_data.time),
                            "qpos": [float(value) for value in slot.body.sim.mj_data.qpos],
                            "qvel": [float(value) for value in slot.body.sim.mj_data.qvel],
                        },
                    )
                )
                continue

            if command == "close":
                slot.body.sim.eye_renderer = None
                connection.send(("closed", None))
                return

            raise RuntimeError(f"unknown Flyppy body-worker command {command!r}")
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


class FlyppyBodyProcess:
    """Parent-side handle for one process-isolated physical fly."""

    def __init__(
        self,
        context: mp.context.BaseContext,
        *,
        slot_index: int,
        physics_steps: int,
        timeout_s: float,
        config: dict[str, Any],
    ) -> None:
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(
            target=_worker_main,
            args=(child, slot_index, physics_steps, dict(config)),
            name=f"flyppy-body-{slot_index}",
        )
        self.slot_index = int(slot_index)
        self.timeout_s = float(timeout_s)
        self.process.start()
        child.close()
        status, payload = self._recv("startup")
        if status != "ready":
            raise RuntimeError(
                f"slot {self.slot_index} failed startup: {status}: {payload}"
            )
        self.body_ids = tuple(int(value) for value in payload["body_ids"])
        self.control_dt_s = float(payload["control_dt_s"])

    def _recv(self, phase: str):
        if not self.connection.poll(self.timeout_s):
            exitcode = self.process.exitcode
            if exitcode is not None:
                raise RuntimeError(
                    f"slot {self.slot_index} process died during {phase}; exitcode={exitcode}"
                )
            self.process.terminate()
            self.process.join(timeout=5.0)
            raise RuntimeError(
                f"slot {self.slot_index} timed out during {phase} after {self.timeout_s:.1f}s"
            )
        try:
            return self.connection.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError) as exc:
            self.process.join(timeout=1.0)
            raise RuntimeError(
                f"slot {self.slot_index} native process ended during {phase}; "
                f"exitcode={self.process.exitcode}"
            ) from exc

    def request(self, command: str, payload=None) -> None:
        if not self.process.is_alive():
            raise RuntimeError(
                f"slot {self.slot_index} process is not alive; exitcode={self.process.exitcode}"
            )
        self.connection.send((command,) if payload is None else (command, payload))

    def receive(self, phase: str):
        status, payload = self._recv(phase)
        if status == "error":
            raise RuntimeError(f"slot {self.slot_index} worker error: {payload}")
        if status not in {"ok", "closed"}:
            raise RuntimeError(f"slot {self.slot_index} unexpected response {status!r}")
        return payload

    def call(self, command: str, payload=None):
        self.request(command, payload)
        return self.receive(command)

    def reset(
        self,
        *,
        episode: int,
        source_weight_version: int,
        spawn_x_mm: float,
        spawn_z_mm: float,
        initial_speed_mm_s: float,
        initial_vz_mm_s: float = 0.0,
        course_start_gate_index: int | None = None,
        gate_center_overrides: dict[int, float] | None = None,
    ):
        return self.call(
            "reset",
            {
                "episode": int(episode),
                "source_weight_version": int(source_weight_version),
                "spawn_x_mm": float(spawn_x_mm),
                "spawn_z_mm": float(spawn_z_mm),
                "initial_speed_mm_s": float(initial_speed_mm_s),
                "initial_vz_mm_s": float(initial_vz_mm_s),
                "course_start_gate_index": (
                    None
                    if course_start_gate_index is None
                    else int(course_start_gate_index)
                ),
                "gate_center_overrides": {
                    int(key): float(value)
                    for key, value in (gate_center_overrides or {}).items()
                },
            },
        )

    def snapshot(self):
        return self.call("snapshot")

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


def spawn_body_processes(
    *,
    population: int,
    physics_steps: int,
    timeout_s: float,
    config: dict[str, Any],
) -> list[FlyppyBodyProcess]:
    context = mp.get_context("spawn")
    return [
        FlyppyBodyProcess(
            context,
            slot_index=index,
            physics_steps=physics_steps,
            timeout_s=timeout_s,
            config=config,
        )
        for index in range(population)
    ]
