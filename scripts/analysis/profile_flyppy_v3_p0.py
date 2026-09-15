#!/usr/bin/env python3
"""Read-only P0 wall-clock profiler for the current Flyppy v3 hot loop."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "scripts" / "embodiment"
sys.path.insert(0, str(EMB))

from virtual_fly.physics import FLYPPY_GEOMETRY_V3
from flybody_v3_adapter import FlyBodyV3NeuromuscularAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from live_telemetry import LiveTelemetryPublisher
from malecns_retina import MaleCNSRetina
from neural_bridge_client import NeuralBridgeClient, NeuralBridgeError
from whole_body_periphery import WholeBodyPeriphery


class Profiler:
    def __init__(self) -> None:
        self.enabled = False
        self.samples: dict[str, list[float]] = defaultdict(list)

    def call(self, key, fn, *args, **kwargs):
        if not self.enabled:
            return fn(*args, **kwargs)
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.samples[key].append(time.perf_counter() - started)

    def summary(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for key, values in sorted(self.samples.items()):
            if not values:
                continue
            ordered = sorted(values)
            p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
            output[key] = {
                "calls": len(ordered),
                "total_s": sum(ordered),
                "mean_ms": statistics.fmean(ordered) * 1e3,
                "p50_ms": statistics.median(ordered) * 1e3,
                "p95_ms": ordered[p95_index] * 1e3,
                "max_ms": ordered[-1] * 1e3,
            }
        return output


class ProfiledBridge(NeuralBridgeClient):
    def __init__(self, *args, profiler: Profiler, **kwargs) -> None:
        self.profiler = profiler
        self.count_protocol = False
        self.request_bytes: dict[str, int] = defaultdict(int)
        self.request_count: dict[str, int] = defaultdict(int)
        self.response_bytes = 0
        super().__init__(*args, **kwargs)

    def _read_response(self) -> dict:
        line = self._stdout.readline()
        if not line:
            raise NeuralBridgeError(
                f"neural bridge exited unexpectedly (code={self._proc.poll()})"
            )
        if self.count_protocol:
            self.response_bytes += len(line.encode("utf-8"))
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NeuralBridgeError(f"invalid bridge response: {line!r}") from exc
        if not response.get("ok", False):
            raise NeuralBridgeError(str(response.get("error", response)))
        return response

    def _request(self, payload) -> dict:
        if self._proc.poll() is not None:
            raise NeuralBridgeError(
                f"neural bridge is not running (code={self._proc.returncode})"
            )
        request_type = str(payload.get("type", "unknown"))
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        if self.count_protocol:
            self.request_bytes[request_type] += len(line.encode("utf-8"))
            self.request_count[request_type] += 1
        started = time.perf_counter() if self.count_protocol else None
        self._stdin.write(line)
        self._stdin.flush()
        response = self._read_response()
        if started is not None:
            self.profiler.samples[f"bridge_request:{request_type}"].append(
                time.perf_counter() - started
            )
        return response

    def protocol_summary(self) -> dict[str, object]:
        return {
            "request_bytes_total": sum(self.request_bytes.values()),
            "response_bytes_total": self.response_bytes,
            "request_bytes_by_type": dict(self.request_bytes),
            "request_count_by_type": dict(self.request_count),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/experiments/flyppy-v3"))
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument("--groups", type=Path, default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"))
    parser.add_argument("--retinotopic-map", type=Path, default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"))
    parser.add_argument("--wing-motor-map", type=Path, default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"))
    parser.add_argument("--body-motor-map", type=Path, default=Path("artifacts/malecns-v1.0/body-motor-neurons-v0.json"))
    parser.add_argument("--viewer-graph", type=Path, default=Path("artifacts/embodiment/neural-viewer-graph-v1.json"))
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--telemetry", choices=("on", "off"), default="on")
    parser.add_argument("--neural-telemetry-stride", type=int, default=10)
    parser.add_argument("--body-telemetry-stride", type=int, default=1)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path("reports/flyppy/profile_latest.md"))
    parser.add_argument("--json", type=Path, default=Path("reports/flyppy/profile_latest.json"))
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_viewer_body_ids(path: Path) -> tuple[int, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(sorted({int(node["body_id"]) for node in data.get("nodes", [])}))


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.steps < 1:
        raise SystemExit("warmup must be >= 0 and steps must be >= 1")
    if min(args.neural_telemetry_stride, args.body_telemetry_stride, args.trajectory_stride) < 1:
        raise SystemExit("profile strides must be >= 1")

    experiment = absolute(args.experiment)
    snapshot = absolute(args.snapshot)
    groups = absolute(args.groups)
    retinotopic_map = absolute(args.retinotopic_map)
    wing_motor_map = absolute(args.wing_motor_map)
    body_motor_map = absolute(args.body_motor_map)
    viewer_graph = absolute(args.viewer_graph)
    checkpoint = experiment / "checkpoint"
    curriculum_state = experiment / "curriculum-state.json"

    required = [
        snapshot / "manifest.json",
        groups,
        retinotopic_map,
        wing_motor_map,
        body_motor_map,
        viewer_graph,
        checkpoint / "manifest.json",
        curriculum_state,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required profile inputs:\n  " + "\n  ".join(missing))

    state = json.loads(curriculum_state.read_text(encoding="utf-8"))
    spawn_x = float(state["spawn_x_mm"])
    spawn_z = float(state["spawn_z_mm"])
    initial_speed = float(state["initial_speed_mm_s"])

    course = FlyppyCourse(seed=0, gate_count=6, environment_version="v3")
    world = FlyppyWorld(course)
    body = FlyBodyV3NeuromuscularAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, FLYPPY_GEOMETRY_V3.corridor_high_z_mm / 2.0),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=True,
        enable_observer_camera=False,
    )
    periphery = WholeBodyPeriphery(wing_motor_map, body_motor_map)
    vision = MaleCNSRetina(retinotopic_map, current_gain=2.0)
    viewer_ids = load_viewer_body_ids(viewer_graph) if args.telemetry == "on" else ()
    profile_root = ROOT / "artifacts" / "profiles" / "flyppy-v3-p0"
    publisher = LiveTelemetryPublisher(profile_root, enabled=args.telemetry == "on")
    profile_root.mkdir(parents=True, exist_ok=True)
    trajectory_path = profile_root / f"trajectory-{args.telemetry}.jsonl"

    profiler = Profiler()
    original_eye_readouts = vision._eye_readouts
    vision._eye_readouts = lambda sim, fly: profiler.call(
        "retina_eye_readout", original_eye_readouts, sim, fly
    )

    measured = 0
    episode = 0
    episode_step = 0
    retinal_pairs = 0
    read_ids = 0
    reinforcement_events = 0
    episode_resets = 0
    control_dt_s = body.timestep * 10

    with trajectory_path.open("w", encoding="utf-8") as trajectory:
        with ProfiledBridge(
            snapshot=snapshot,
            groups=groups,
            backend="gpu",
            repo_root=ROOT,
            profiler=profiler,
        ) as brain:
            brain.ping()
            brain.load_checkpoint(checkpoint)

            def reset_episode() -> None:
                nonlocal episode, episode_step
                course.reset()
                body.reset()
                body.set_root_position_mm((spawn_x, 0.0, spawn_z))
                body.set_root_linear_velocity_mm_s((initial_speed, 0.0, 0.0))
                periphery.reset()
                vision.reset_adaptation()
                brain.reset_dynamics()
                episode += 1
                episode_step = 0
                if args.telemetry == "on":
                    profiler.call(
                        "telemetry_status",
                        publisher.publish_status,
                        running=True,
                        backend=str(brain.ready.get("backend", "gpu")),
                        episode=episode,
                        control_step=0,
                        curriculum={
                            "mode": "adaptive",
                            "environment_version": "v3",
                            "motor_boundary": "whole-body",
                            "spawn_x_mm": spawn_x,
                            "spawn_z_mm": spawn_z,
                            "initial_speed_mm_s": initial_speed,
                        },
                    )

            reset_episode()
            warmup_left = args.warmup
            measured_wall_start: float | None = None

            while measured < args.steps:
                if warmup_left == 0 and not profiler.enabled:
                    profiler.enabled = True
                    brain.count_protocol = True
                    measured_wall_start = time.perf_counter()

                active = profiler.enabled
                control_started = time.perf_counter() if active else None
                neural_telemetry_sample = episode_step % args.neural_telemetry_stride == 0
                body_telemetry_sample = episode_step % args.body_telemetry_stride == 0
                read_body = (
                    tuple(dict.fromkeys((*periphery.body_ids, *viewer_ids)))
                    if neural_telemetry_sample and viewer_ids
                    else periphery.body_ids
                )

                retinal = profiler.call("retina_total", vision.encode, body.sim, body.fly)
                _, body_spikes = profiler.call(
                    "neural_control_total",
                    brain.step_with_body_readout,
                    stimulate_body=retinal.body_currents,
                    read=(),
                    read_body=read_body,
                    plasticity=True,
                )
                peripheral = profiler.call(
                    "periphery", periphery.step, body_spikes, dt_s=control_dt_s
                )
                profiler.call("physics", body.step_muscles, peripheral, physics_steps=10)

                position = body.thorax_position_mm()
                velocity = body.root_linear_velocity_mm_s()
                collision_reason = profiler.call(
                    "collision_query", world.physical_collision_reason, body.sim
                )
                event = profiler.call(
                    "course_update",
                    course.update,
                    float(position[0]),
                    float(position[2]),
                    physical_collision_reason=collision_reason,
                    analytic_body_collision=False,
                )

                reward = False
                aversive = False
                if event.passed_gate:
                    profiler.call(
                        "reinforcement_total",
                        brain.step,
                        stimulate={"reward_dan": 2.0},
                        read=(),
                        plasticity=True,
                        steps=4,
                    )
                    reward = True
                    if active:
                        reinforcement_events += 1
                if event.collision:
                    profiler.call(
                        "reinforcement_total",
                        brain.step,
                        stimulate={"aversive_dan": 2.0},
                        read=(),
                        plasticity=True,
                        steps=4,
                    )
                    aversive = True
                    if active:
                        reinforcement_events += 1

                next_gate = getattr(course, "absolute_next_gate_index", course.next_gate_index)
                if episode_step % args.trajectory_stride == 0 or reward or aversive or event.finished:
                    trajectory_payload = {
                        "episode": episode,
                        "control_step": episode_step,
                        "x_mm": float(position[0]),
                        "y_mm": float(position[1]),
                        "z_mm": float(position[2]),
                        "vx_mm_s": float(velocity[0]),
                        "vy_mm_s": float(velocity[1]),
                        "vz_mm_s": float(velocity[2]),
                        "next_gate": next_gate,
                        "motor_periphery": peripheral.compact_diagnostics(),
                        "retinal_input": {
                            "active_columns": retinal.active_columns,
                            "active_photoreceptors": retinal.active_photoreceptors,
                            "mean_current": retinal.mean_current,
                            "max_current": retinal.max_current,
                        },
                        "reward_stimulated": reward,
                        "aversive_stimulated": aversive,
                        "passed_gate": event.passed_gate,
                        "collision": event.collision,
                        "collision_reason": event.collision_reason,
                        "finished": event.finished,
                        "environment_version": "v3",
                        "motor_boundary": "whole-body",
                        "curriculum_mode": "adaptive",
                        "boundary_ease_level": None,
                    }
                    serialized = profiler.call(
                        "trajectory_serialize",
                        json.dumps,
                        trajectory_payload,
                        separators=(",", ":"),
                    )
                    profiler.call("trajectory_write", trajectory.write, serialized + "\n")

                if args.telemetry == "on":
                    if neural_telemetry_sample or reward or aversive or event.finished:
                        active_viewer = [
                            body_id for body_id in viewer_ids if body_spikes.get(body_id, False)
                        ]
                        profiler.call(
                            "telemetry_neural",
                            publisher.publish_neural,
                            episode=episode,
                            control_step=episode_step,
                            neural_step=brain.last_step,
                            depolarizing_body_ids=active_viewer,
                            hyperpolarizing_body_ids=(),
                            reward=reward,
                            aversive=aversive,
                        )
                    if body_telemetry_sample or reward or aversive or event.finished:
                        profiler.call(
                            "telemetry_body",
                            publisher.publish_body,
                            episode=episode,
                            control_step=episode_step,
                            sim=body.sim,
                            next_gate=next_gate,
                            passed_gate=event.passed_gate,
                            collision=event.collision,
                            collision_reason=event.collision_reason,
                            reward=reward,
                            aversive=aversive,
                            motor=peripheral.compact_diagnostics(),
                            retinal={
                                "active_columns": retinal.active_columns,
                                "active_photoreceptors": retinal.active_photoreceptors,
                                "mean_current": retinal.mean_current,
                                "max_current": retinal.max_current,
                            },
                        )

                if active:
                    retinal_pairs += len(retinal.body_currents)
                    read_ids += len(read_body)
                    measured += 1
                    assert control_started is not None
                    profiler.samples["control_step_total"].append(
                        time.perf_counter() - control_started
                    )
                else:
                    warmup_left -= 1

                episode_step += 1
                if event.collision or event.finished:
                    if active:
                        episode_resets += 1
                    profiler.call("trajectory_flush", trajectory.flush)
                    profiler.call("episode_reset_total", reset_episode)

            assert measured_wall_start is not None
            measured_wall_seconds = time.perf_counter() - measured_wall_start
            payload = {
                "schema_version": 2,
                "backend": brain.ready.get("backend"),
                "neurons": brain.ready.get("neurons"),
                "edges": brain.ready.get("edges"),
                "warmup_control_steps": args.warmup,
                "measured_control_steps": measured,
                "measured_wall_seconds": measured_wall_seconds,
                "control_steps_per_second": measured / measured_wall_seconds,
                "telemetry": args.telemetry,
                "neural_telemetry_stride": args.neural_telemetry_stride,
                "body_telemetry_stride": args.body_telemetry_stride,
                "trajectory_stride": args.trajectory_stride,
                "spawn_x_mm": spawn_x,
                "spawn_z_mm": spawn_z,
                "initial_speed_mm_s": initial_speed,
                "mean_retinal_stimulus_pairs": retinal_pairs / measured,
                "mean_motor_read_ids": read_ids / measured,
                "reinforcement_events": reinforcement_events,
                "episode_resets": episode_resets,
                "timing": profiler.summary(),
                "protocol": brain.protocol_summary(),
                "checkpoint": str(checkpoint),
                "checkpoint_modified": False,
                "paths": {
                    "snapshot": str(snapshot),
                    "groups": str(groups),
                    "retinotopic_map": str(retinotopic_map),
                    "wing_motor_map": str(wing_motor_map),
                    "body_motor_map": str(body_motor_map),
                    "viewer_graph": str(viewer_graph),
                },
            }

    json_path = absolute(args.json)
    report_path = absolute(args.report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rows = sorted(payload["timing"].items(), key=lambda item: item[1]["total_s"], reverse=True)
    markdown = [
        "# Flyppy P0 performance profile",
        "",
        f"- backend: `{payload['backend']}`",
        f"- measured control steps: {measured}",
        f"- telemetry: `{args.telemetry}`",
        f"- wall-clock: {measured_wall_seconds:.6f} s",
        f"- control steps/s: {measured / measured_wall_seconds:.3f}",
        f"- neurons / edges: {payload['neurons']} / {payload['edges']}",
        f"- neural/body/trajectory stride: {args.neural_telemetry_stride}/{args.body_telemetry_stride}/{args.trajectory_stride}",
        "",
        "| stage | calls | total s | mean ms | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in rows:
        markdown.append(
            f"| {key} | {value['calls']} | {value['total_s']:.6f} | "
            f"{value['mean_ms']:.3f} | {value['p50_ms']:.3f} | {value['p95_ms']:.3f} |"
        )
    markdown += [
        "",
        "## Protocol",
        "",
        f"- request bytes: {payload['protocol']['request_bytes_total']}",
        f"- response bytes: {payload['protocol']['response_bytes_total']}",
        f"- request count: `{json.dumps(payload['protocol']['request_count_by_type'], sort_keys=True)}`",
        f"- mean retinal stimulus pairs/step: {payload['mean_retinal_stimulus_pairs']:.1f}",
        f"- mean read_body IDs/step: {payload['mean_motor_read_ids']:.1f}",
        "",
        "`retina_total` includes `retina_eye_readout`. `bridge_request:step` overlaps conceptually with neural/reinforcement totals. "
        "This pass loads but never saves the production checkpoint. GPU kernel-level splitting is deferred until the wall-clock profile shows the neural bridge is dominant.",
        "",
    ]
    report_path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"profile_report={report_path}")
    print(f"profile_json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
