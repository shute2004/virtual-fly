#!/usr/bin/env python3
"""Asynchronous shared-weight Flyppy v3 population trainer.

Each Fly slot owns its episode-local CNS/body/plasticity state. All slots feed
one learned global weight history. At episode completion the slot's exact
additive+clamp plasticity transaction is rebased onto the newest global weights;
weights are never averaged and a stale slot never overwrites the global state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

from virtual_fly.physics import FLYBODY_V3, FLYPPY_GEOMETRY_V3, FLYPPY_GEOMETRY_V4
from virtual_fly.training.checkpointing import (
    persist_shared_checkpoint as persist_population_checkpoint,
    population_state_snapshot,
    prepare_population_resume,
    recover_population_storage,
)
from virtual_fly.training.population_schedule import checkpoint_can_flush, launch_round_for_index
from virtual_fly.training.curriculum import (
    AdaptiveCurriculumConfig,
    BoundaryBandConfig,
    SpawnCondition,
    boundary_condition_for_attempt,
    boundary_frontier_role,
    boundary_target_gates,
    frontier_focus_condition,
    frontier_focus_level,
    current_adaptive_condition,
    next_boundary_attempt_for_group,
    ensure_boundary_state,
    load_state as load_curriculum_state,
    record_adaptive_result,
    record_boundary_result,
)

from flybody_v3_adapter import FlyBodyV3NeuromuscularAdapter
from flybody_v4_adapter import FlyBodyV4NeuromuscularAdapter
from flybody_v5_adapter import FlyBodyV5NeuromuscularAdapter
from flybody_v6_adapter import FlyBodyV6NeuromuscularAdapter
from flybody_v7_adapter import FlyBodyV7NeuromuscularAdapter
from flybody_v8_adapter import FlyBodyV8NeuromuscularAdapter
from haltere_campaniform_sensor import HaltereCampaniformSensor
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from live_telemetry import LiveTelemetryPublisher
from malecns_retina import MaleCNSRetina
from population_neural_bridge_client import PopulationNeuralBridgeClient
from whole_body_periphery import WholeBodyPeriphery
from virtual_fly.reproducibility import (
    HALTERE_FULL_KIND,
    HALTERE_TIMING_KIND,
    build_run_provenance,
    compatibility_warnings,
    validate_derived_artifact,
    validate_haltere_map,
    validate_production_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]


def normalized_vision_config() -> tuple[str, int]:
    raw_mode = str(os.environ.get("VF_FLYPPY_VISION_MODE", "raster")).strip().lower()
    aliases = {
        "raster": "raster",
        "flygym": "raster",
        "reference": "raster",
        "direct": "direct-ray",
        "ray": "direct-ray",
        "direct-ray": "direct-ray",
    }
    mode = aliases.get(raw_mode, raw_mode)
    rays = int(os.environ.get("VF_FLYPPY_OMMATIDIA_RAYS", "7"))
    return mode, rays


def run_reproducibility_metadata(
    args: argparse.Namespace,
    *,
    body_runtime: str,
    vision_mode_override: str | None = None,
    vision_rays_override: int | None = None,
) -> dict[str, object]:
    vision_mode, vision_rays = normalized_vision_config()
    if vision_mode_override is not None:
        vision_mode = vision_mode_override
    if vision_rays_override is not None:
        vision_rays = vision_rays_override
    body_version = str(getattr(args, "flight_body_version", "v3"))
    artifacts: dict[str, Path] = {
        "embodiment_groups": args.groups,
        "retinotopic_map": args.retinotopic_map,
        "wing_motor_map": args.wing_motor_map,
        "body_motor_map": args.body_motor_map,
        "haltere_sensory_map": args.haltere_sensory_map,
        "viewer_graph": args.viewer_graph,
        "measured_wingbeat": Path("artifacts/flybody-data/wing_pattern_fmech.npy"),
    }
    calibration = os.environ.get("VF_NEURAL_CALIBRATION_PATH", "").strip()
    if calibration:
        artifacts["neural_calibration"] = Path(calibration)
    if body_version in {"v7", "v8"}:
        artifacts["neutral_trim_pattern"] = Path("artifacts/embodiment/wing-pattern-neutral-trim-v1.npy")
        artifacts["neutral_trim_metadata"] = Path("artifacts/embodiment/wing-pattern-neutral-trim-v1.json")
    if body_version == "v5":
        artifacts["measured_vertical_steering"] = Path("artifacts/embodiment/wing-steering-vertical-mode-v1.json")
    if body_version in {"v6", "v8"}:
        artifacts["measured_b2_hg3_steering"] = Path("artifacts/embodiment/wing-steering-b2-hg3-modes-v1.json")

    conditions: dict[str, object] = {
        "body": {
            "version": body_version,
            "vertical_steering_gain": float(args.vertical_steering_gain),
            "measured_steering_gain": float(args.measured_steering_gain),
            "neutral_trim_strength": float(args.neutral_trim_strength),
            "steering_tau_ms": float(args.steering_tau_ms),
            "steering_spike_increment": float(args.steering_spike_increment),
            "runtime": body_runtime,
        },
        "environment": {
            "version": str(args.environment_version),
            "gate_count": int(args.gate_count),
        },
        "vision": {
            "mode": vision_mode,
            "rays_per_ommatidium": vision_rays,
            "photoreceptor_current_gain": float(args.photoreceptor_current_gain),
        },
        "haltere": {
            "enabled": float(args.haltere_current_gain) > 0.0,
            "map_kind": str(args.haltere_sensory_kind),
            "current_gain": float(args.haltere_current_gain),
            "transduction": str(args.haltere_transduction),
        },
        "reinforcement": {
            "reward_current": float(args.reward_current),
            "aversive_current": float(args.aversive_current),
            "reinforcement_steps": int(args.reinforcement_steps),
            "gate_near_miss_aversive_floor_fraction": float(args.gate_near_miss_aversive_floor_fraction),
            "gate_near_miss_distance_mm": float(args.gate_near_miss_distance_mm),
            "path": "gate pass -> reward_dan current; collision -> aversive_dan current",
        },
        "rng": {
            "seed": int(args.seed),
            "fixed_course_seed": None if args.fixed_course_seed is None else int(args.fixed_course_seed),
        },
        "training": {
            "population": int(args.population),
            "launch_mode": str(args.launch_mode),
            "curriculum_mode": str(args.curriculum_mode),
            "shared_weight_commit_semantics": "episode-local additive+clamp transaction rebased onto latest global weight",
            "neural_synapse_scale": os.environ.get("VF_NEURAL_SYNAPSE_SCALE"),
        },
    }
    conditions["compatibility_warnings"] = compatibility_warnings(
        body_version=body_version,
        environment_version=str(args.environment_version),
        vision_mode=vision_mode,
        haltere_gain=float(args.haltere_current_gain),
        haltere_kind=str(args.haltere_sensory_kind),
    )
    return build_run_provenance(
        root=ROOT, snapshot=args.snapshot, artifacts=artifacts, conditions=conditions
    )


def write_run_provenance(
    output: Path,
    *,
    start_episode: int,
    initial_global_weight_version: int,
    payload: dict[str, object],
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output / "provenance" / (
        f"run-{start_episode:06d}-gv{initial_global_weight_version:06d}-{stamp}.json"
    )
    save_json_atomic(path, payload)
    return path


@dataclass
class SlotRuntime:
    slot: int
    course: FlyppyCourse
    world: FlyppyWorld
    body: Any
    periphery: WholeBodyPeriphery
    vision: MaleCNSRetina
    haltere_sensor: HaltereCampaniformSensor
    active: bool = False
    episode: int = -1
    source_weight_version: int = 0
    spawn_x_mm: float = 0.0
    spawn_z_mm: float = 0.0
    initial_speed_mm_s: float = 0.0
    initial_vz_mm_s: float = 0.0
    control_steps: int = 0
    passed_gates: int = 0
    collision: bool = False
    collision_reason: str | None = None
    finished: bool = False
    max_x_mm: float = float("-inf")
    min_z_mm: float = float("inf")
    max_z_mm: float = float("-inf")
    final_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_retinal: Any = None
    last_haltere: Any = None
    last_peripheral: Any = None
    last_body_spikes: dict[int, bool] | None = None
    reward_events: int = 0
    aversive_events: int = 0
    boundary_ease_level: float | None = None
    boundary_attempt_index: int | None = None
    boundary_batch_number: int | None = None
    frontier_role: str = "evaluate"
    frontier_target_gate: int | None = None
    course_start_gate_index: int | None = None
    launch_round: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=24, help="total episodes launched across all slots")
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument("--groups", type=Path, default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"))
    parser.add_argument("--retinotopic-map", type=Path, default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"))
    parser.add_argument("--haltere-sensory-map", type=Path, default=Path("artifacts/malecns-v1.0/haltere-campaniform-sensory-v1.json"))
    parser.add_argument(
        "--haltere-sensory-kind",
        choices=(HALTERE_FULL_KIND, HALTERE_TIMING_KIND),
        default=HALTERE_FULL_KIND,
        help="semantic kind required from --haltere-sensory-map; prevents full/timing map substitution",
    )
    parser.add_argument("--wing-motor-map", type=Path, default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"))
    parser.add_argument("--body-motor-map", type=Path, default=Path("artifacts/malecns-v1.0/body-motor-neurons-v0.json"))
    parser.add_argument("--viewer-graph", type=Path, default=Path("artifacts/embodiment/neural-viewer-graph-v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/flyppy-v3"))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument(
        "--environment-version",
        choices=("v3", "v4", "v5", "v6", "v7"),
        default="v3",
        help=(
            "v4 preserves the historical tall-corridor/narrow-hole-range course; "
            "v5 keeps v4 physics but uses the full vertical corridor for diverse gate openings"
        ),
    )
    parser.add_argument(
        "--flight-body-version",
        choices=("v3", "v4", "v5", "v6", "v7", "v8"),
        default="v3",
        help=(
            "v3 preserves the fixed-hover wing seam; v4 adds A-IFM power-modulated "
            "stroke amplitude; v5/v6 add measured b2/hg3 steering; v7 adds neutral trim; "
            "v8 combines v6 measured steering with v7 neutral trim"
        ),
    )
    parser.add_argument("--vertical-steering-gain", type=float, default=1.0)
    parser.add_argument("--measured-steering-gain", type=float, default=1.0)
    parser.add_argument("--steering-tau-ms", type=float, default=12.0)
    parser.add_argument("--steering-spike-increment", type=float, default=0.85)
    parser.add_argument("--neutral-trim-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fixed-course-seed",
        type=int,
        default=None,
        help="use one Flyppy course layout for every population slot while keeping spawn curriculum independent",
    )
    parser.add_argument("--fixed-spawn-x-mm", type=float, default=None)
    parser.add_argument("--fixed-spawn-z-mm", type=float, default=None)
    parser.add_argument("--fixed-spawn-speed-mm-s", type=float, default=None)
    parser.add_argument("--fixed-spawn-vz-mm-s", type=float, default=0.0)
    parser.add_argument("--max-control-steps", type=int, default=1800)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--photoreceptor-current-gain", type=float, default=2.0)
    parser.add_argument("--haltere-current-gain", type=float, default=0.0, help="current gain for physical haltere campaniform feedback; 0 preserves historical behavior")
    parser.add_argument(
        "--haltere-transduction",
        choices=("angular-acceleration-v1", "interaction-load-v2"),
        default="angular-acceleration-v1",
        help="physical haltere campaniform transduction model",
    )
    parser.add_argument("--reward-current", type=float, default=2.0)
    parser.add_argument("--aversive-current", type=float, default=2.0)
    parser.add_argument(
        "--gate-near-miss-aversive-floor-fraction",
        type=float,
        default=0.25,
        help="fraction of max PPL current for a gate collision whose thorax center reaches the aperture",
    )
    parser.add_argument(
        "--gate-near-miss-distance-mm",
        type=float,
        default=2.0,
        help="vertical miss distance at which gate-collision PPL current reaches --aversive-current",
    )
    parser.add_argument("--reinforcement-steps", type=int, default=4)
    parser.add_argument("--telemetry", action="store_true", help="publish detached viewer telemetry for one slot")
    parser.add_argument("--telemetry-slot", type=int, default=0)
    parser.add_argument("--telemetry-stride", type=int, default=10)
    parser.add_argument("--body-telemetry-stride", type=int, default=1)
    parser.add_argument(
        "--launch-mode",
        choices=("async", "wave"),
        default="async",
        help=(
            "async restarts each finished slot immediately; wave waits for all currently "
            "launched slots to finish before starting the next group from one shared weight version"
        ),
    )

    parser.add_argument(
        "--curriculum-mode",
        choices=("adaptive", "boundary-band", "gate2-height"),
        default="adaptive",
        help=(
            "adaptive preserves the historical per-episode policy; boundary-band updates only "
            "after a fixed batch; gate2-height keeps course/spawn fixed while moving gate 2 "
            "upward with retention rehearsal"
        ),
    )
    parser.add_argument("--boundary-batch-size", type=int, default=24)
    parser.add_argument("--boundary-harden-success-rate", type=float, default=0.80)
    parser.add_argument("--boundary-ease-success-rate", type=float, default=0.40)
    parser.add_argument("--gate2-height-start-z-mm", type=float, default=None)
    parser.add_argument("--gate2-height-target-z-mm", type=float, default=None)
    parser.add_argument("--gate2-height-step-mm", type=float, default=0.25)
    parser.add_argument("--gate2-height-current-success-rate", type=float, default=0.80)
    parser.add_argument("--gate2-height-retention-success-rate", type=float, default=0.80)
    parser.add_argument(
        "--gate2-height-acquisition-only",
        action="store_true",
        help="use every gate-height batch episode at the current frontier; intended for short capability acquisition before retention rehearsal",
    )

    parser.add_argument("--curriculum-start-x-mm", type=float, default=8.91)
    parser.add_argument("--curriculum-target-x-mm", type=float, default=0.0)
    parser.add_argument("--curriculum-target-z-mm", type=float, default=8.91)
    parser.add_argument("--curriculum-start-speed-mm-s", type=float, default=400.0)
    parser.add_argument("--curriculum-target-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--curriculum-x-step-mm", type=float, default=0.297)
    parser.add_argument("--curriculum-failure-x-step-mm", type=float, default=0.1485)
    parser.add_argument("--curriculum-z-step-mm", type=float, default=0.1485)
    parser.add_argument("--curriculum-speed-step-mm-s", type=float, default=12.5)
    parser.add_argument("--curriculum-max-easy-speed-mm-s", type=float, default=450.0)
    return parser.parse_args()


def save_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_viewer_body_ids(path: Path) -> tuple[int, ...]:
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    return tuple(sorted({int(node["body_id"]) for node in payload.get("nodes", [])}))


def target_condition(args: argparse.Namespace) -> SpawnCondition:
    return SpawnCondition(
        args.curriculum_target_x_mm,
        args.curriculum_target_z_mm,
        args.curriculum_target_speed_mm_s,
    )


def adaptive_config(args: argparse.Namespace, first_gate) -> AdaptiveCurriculumConfig:
    return AdaptiveCurriculumConfig(
        target=target_condition(args),
        x_step_mm=args.curriculum_x_step_mm,
        z_step_mm=args.curriculum_z_step_mm,
        speed_step_mm_s=args.curriculum_speed_step_mm_s,
        max_easy_speed_mm_s=args.curriculum_max_easy_speed_mm_s,
        failure_x_step_mm=args.curriculum_failure_x_step_mm,
        first_gate_x_mm=float(first_gate.x_mm),
        first_gate_z_mm=float(first_gate.center_z_mm),
    )


def fixed_spawn_condition(args: argparse.Namespace) -> SpawnCondition | None:
    values = (
        args.fixed_spawn_x_mm,
        args.fixed_spawn_z_mm,
        args.fixed_spawn_speed_mm_s,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise SystemExit("fixed spawn requires x, z, and speed together")
    x_mm, z_mm, speed_mm_s = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x_mm, z_mm, speed_mm_s)):
        raise SystemExit("fixed spawn values must be finite")
    if speed_mm_s <= 0.0:
        raise SystemExit("fixed spawn speed must be > 0")
    if not math.isfinite(float(args.fixed_spawn_vz_mm_s)):
        raise SystemExit("fixed spawn vertical speed must be finite")
    return SpawnCondition(x_mm, z_mm, speed_mm_s)


def validate(args: argparse.Namespace, first_gate) -> None:
    if args.episodes < 1:
        raise SystemExit("episodes must be >= 1")
    if args.population < 1:
        raise SystemExit("population must be >= 1")
    if args.population > 32:
        raise SystemExit("population > 32 exceeds the GPU runtime active-mask limit")
    if not 0 <= args.telemetry_slot < args.population:
        raise SystemExit("telemetry-slot must identify an existing population slot")
    if args.max_control_steps < 1 or args.physics_steps < 1:
        raise SystemExit("control/physics steps must be >= 1")
    if min(args.trajectory_stride, args.telemetry_stride, args.body_telemetry_stride) < 1:
        raise SystemExit("trajectory/telemetry strides must be >= 1")
    if args.checkpoint_every < 1:
        raise SystemExit("checkpoint-every must be >= 1")
    minimum_batch_size = (
        1
        if args.curriculum_mode == "gate2-height" and args.gate2_height_acquisition_only
        else 5
    )
    if args.boundary_batch_size < minimum_batch_size:
        raise SystemExit(
            f"boundary-batch-size must be >= {minimum_batch_size} for this curriculum mode"
        )
    if not (
        0.0
        <= args.boundary_ease_success_rate
        < args.boundary_harden_success_rate
        <= 1.0
    ):
        raise SystemExit("boundary success-rate thresholds are invalid")
    if args.curriculum_mode == "gate2-height":
        if args.gate_count < 2:
            raise SystemExit("gate2-height curriculum requires at least 2 gates")
        if args.fixed_course_seed is None:
            raise SystemExit("gate2-height curriculum requires --fixed-course-seed")
        if not math.isfinite(float(args.gate2_height_step_mm)) or float(args.gate2_height_step_mm) <= 0.0:
            raise SystemExit("gate2-height-step-mm must be finite and > 0")
        for name in ("gate2_height_current_success_rate", "gate2_height_retention_success_rate"):
            value = float(getattr(args, name))
            if not 0.0 < value <= 1.0:
                raise SystemExit(f"{name.replace('_','-')} must be in (0,1]")
        for name in ("gate2_height_start_z_mm", "gate2_height_target_z_mm"):
            value = getattr(args, name)
            if value is not None and not math.isfinite(float(value)):
                raise SystemExit(f"{name.replace('_','-')} must be finite when provided")

    if args.reinforcement_steps < 1 or args.reinforcement_steps % 2 != 0:
        raise SystemExit(
            "population prototype currently requires an even reinforcement-steps value; production v3 uses 4"
        )
    if args.curriculum_start_x_mm >= first_gate.x_mm:
        raise SystemExit("curriculum-start-x-mm must be before the first gate")
    fixed_spawn = fixed_spawn_condition(args)
    if fixed_spawn is not None and fixed_spawn.x_mm >= first_gate.x_mm:
        raise SystemExit("fixed-spawn-x-mm must be before the first gate")
    if not math.isfinite(float(args.haltere_current_gain)) or float(args.haltere_current_gain) < 0.0:
        raise SystemExit("haltere-current-gain must be finite and >= 0")
    if not math.isfinite(float(args.steering_tau_ms)) or float(args.steering_tau_ms) <= 0.0:
        raise SystemExit("steering-tau-ms must be finite and > 0")
    if not 0.0 <= float(args.steering_spike_increment) <= 1.0:
        raise SystemExit("steering-spike-increment must be in [0,1]")
    if not 0.0 <= float(args.gate_near_miss_aversive_floor_fraction) <= 1.0:
        raise SystemExit("gate-near-miss-aversive-floor-fraction must be in [0,1]")
    if not math.isfinite(float(args.gate_near_miss_distance_mm)) or float(args.gate_near_miss_distance_mm) <= 0.0:
        raise SystemExit("gate-near-miss-distance-mm must be finite and > 0")
    required = [
        args.snapshot / "manifest.json",
        args.groups,
        args.retinotopic_map,
        args.wing_motor_map,
        args.body_motor_map,
    ]
    if float(args.haltere_current_gain) > 0.0:
        required.append(args.haltere_sensory_map)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing population training inputs:\n  " + "\n  ".join(missing))
    try:
        validate_production_snapshot(args.snapshot)
        for derived in (args.groups, args.retinotopic_map, args.wing_motor_map, args.body_motor_map):
            validate_derived_artifact(derived, args.snapshot)
        validate_haltere_map(
            args.haltere_sensory_map,
            args.haltere_sensory_kind,
            snapshot=args.snapshot,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    positive = [
        args.photoreceptor_current_gain,
        args.reward_current,
        args.aversive_current,
        args.curriculum_x_step_mm,
        args.curriculum_failure_x_step_mm,
        args.curriculum_z_step_mm,
        args.curriculum_start_speed_mm_s,
        args.curriculum_target_speed_mm_s,
        args.curriculum_speed_step_mm_s,
        args.curriculum_max_easy_speed_mm_s,
    ]
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
        raise SystemExit("positive training parameters must be finite and > 0")


def gate_collision_aversive_current(
    args: argparse.Namespace,
    observation_or_miss,
    collision_reason: str | None,
) -> tuple[float, float | None]:
    """Return PPL current and vertical miss distance for one collision event.

    Gate collisions are graded only by how far the thorax center is vertically
    outside the aperture.  Floor/ceiling collisions keep the full aversive
    current.  This changes the experimenter-imposed DAN stimulation magnitude,
    not CNS action selection or motor commands.
    """

    maximum = float(args.aversive_current)
    if collision_reason != "gate" or observation_or_miss is None:
        return maximum, None
    if isinstance(observation_or_miss, (int, float)):
        miss_mm = max(0.0, float(observation_or_miss))
    else:
        miss_mm = max(
            float(observation_or_miss.gap_low_dz_mm),
            -float(observation_or_miss.gap_high_dz_mm),
            0.0,
        )
    floor = maximum * float(args.gate_near_miss_aversive_floor_fraction)
    fraction = min(1.0, miss_mm / float(args.gate_near_miss_distance_mm))
    return floor + (maximum - floor) * fraction, miss_mm


def population_boundary_config(
    args: argparse.Namespace,
    first_gate,
    state: dict[str, Any],
) -> BoundaryBandConfig:
    """Build a local v3 boundary band around the resumed adaptive condition.

    Existing ``boundary_band`` state, when present, owns the persisted endpoints.
    These derived endpoints are therefore only the migration/start defaults when
    a continuation experiment first switches from adaptive to boundary-band.
    """

    adaptive = adaptive_config(args, first_gate)
    hard_state = dict(state)
    easy_state = dict(state)
    record_adaptive_result(hard_state, adaptive, success=True)
    record_adaptive_result(easy_state, adaptive, success=False)
    hard = current_adaptive_condition(hard_state)
    easy = current_adaptive_condition(easy_state)

    def span(a: float, b: float, fallback: float) -> float:
        width = abs(float(b) - float(a))
        return width if width > 1e-12 else float(fallback)

    harden_step = SpawnCondition(
        span(hard.x_mm, easy.x_mm, args.curriculum_x_step_mm),
        span(hard.z_mm, easy.z_mm, args.curriculum_z_step_mm),
        span(hard.speed_mm_s, easy.speed_mm_s, args.curriculum_speed_step_mm_s),
    )
    ease_step = SpawnCondition(
        harden_step.x_mm / 2.0,
        harden_step.z_mm / 2.0,
        harden_step.speed_mm_s / 2.0,
    )
    return BoundaryBandConfig(
        hard=hard,
        easy=easy,
        target=target_condition(args),
        batch_size=args.boundary_batch_size,
        harden_success_rate=args.boundary_harden_success_rate,
        ease_success_rate=args.boundary_ease_success_rate,
        harden_step=harden_step,
        ease_step=ease_step,
        seed=args.seed,
    )


def make_slot(args: argparse.Namespace, slot_id: int) -> SlotRuntime:
    fixed_course_seed = getattr(args, "fixed_course_seed", None)
    course_seed = int(fixed_course_seed) if fixed_course_seed is not None else args.seed + slot_id
    course = FlyppyCourse(
        seed=course_seed,
        gate_count=args.gate_count,
        environment_version=args.environment_version,
    )
    world = FlyppyWorld(course)
    geometry = FLYPPY_GEOMETRY_V4 if args.environment_version in {"v4", "v5", "v6", "v7"} else FLYPPY_GEOMETRY_V3
    body_version = getattr(args, "flight_body_version", "v3")
    body_cls = {
        "v3": FlyBodyV3NeuromuscularAdapter,
        "v4": FlyBodyV4NeuromuscularAdapter,
        "v5": FlyBodyV5NeuromuscularAdapter,
        "v6": FlyBodyV6NeuromuscularAdapter,
        "v7": FlyBodyV7NeuromuscularAdapter,
        "v8": FlyBodyV8NeuromuscularAdapter,
    }[body_version]
    body_kwargs = {
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
        body_kwargs["vertical_steering_gain"] = float(
            getattr(args, "vertical_steering_gain", 1.0)
        )
    if body_version in {"v6", "v8"}:
        body_kwargs["measured_steering_gain"] = float(
            getattr(args, "measured_steering_gain", 1.0)
        )
    if body_version in {"v7", "v8"}:
        body_kwargs["neutral_trim_strength"] = float(
            getattr(args, "neutral_trim_strength", 1.0)
        )
    body = body_cls(**body_kwargs)
    return SlotRuntime(
        slot=slot_id,
        course=course,
        world=world,
        body=body,
        periphery=WholeBodyPeriphery(
            args.wing_motor_map,
            args.body_motor_map,
            wing_steering_tau_s=float(getattr(args, "steering_tau_ms", 12.0)) / 1000.0,
            wing_steering_spike_increment=float(
                getattr(args, "steering_spike_increment", 0.85)
            ),
        ),
        vision=MaleCNSRetina(
            args.retinotopic_map,
            current_gain=args.photoreceptor_current_gain,
        ),
        haltere_sensor=HaltereCampaniformSensor(
            args.haltere_sensory_map,
            current_gain=float(getattr(args, "haltere_current_gain", 0.0)),
            transduction=str(getattr(args, "haltere_transduction", "angular-acceleration-v1")),
            expected_kind=str(getattr(args, "haltere_sensory_kind", HALTERE_FULL_KIND)),
            snapshot=args.snapshot,
        ),
    )


def begin_episode(
    slot: SlotRuntime,
    *,
    episode: int,
    source_weight_version: int,
    condition: SpawnCondition,
    initial_vz_mm_s: float = 0.0,
    boundary_ease_level: float | None = None,
    boundary_attempt_index: int | None = None,
    boundary_batch_number: int | None = None,
    frontier_role: str = "evaluate",
    frontier_target_gate: int | None = None,
    course_start_gate_index: int | None = None,
    gate_center_overrides: dict[int, float] | None = None,
    launch_round: int = 0,
) -> None:
    slot.active = True
    slot.episode = episode
    slot.source_weight_version = source_weight_version
    slot.spawn_x_mm = float(condition.x_mm)
    slot.spawn_z_mm = float(condition.z_mm)
    slot.initial_speed_mm_s = float(condition.speed_mm_s)
    slot.initial_vz_mm_s = float(initial_vz_mm_s)
    slot.control_steps = 0
    slot.passed_gates = 0
    slot.collision = False
    slot.collision_reason = None
    slot.finished = False
    slot.max_x_mm = float("-inf")
    slot.min_z_mm = float("inf")
    slot.max_z_mm = float("-inf")
    slot.reward_events = 0
    slot.aversive_events = 0
    slot.boundary_ease_level = boundary_ease_level
    slot.boundary_attempt_index = boundary_attempt_index
    slot.boundary_batch_number = boundary_batch_number
    slot.frontier_role = str(frontier_role)
    slot.frontier_target_gate = frontier_target_gate
    slot.course_start_gate_index = (
        None if course_start_gate_index is None else int(course_start_gate_index)
    )
    slot.launch_round = int(launch_round)
    slot.last_retinal = None
    slot.last_haltere = None
    slot.last_peripheral = None
    slot.last_body_spikes = {}

    if gate_center_overrides:
        for gate_index, center_z_mm in sorted(gate_center_overrides.items()):
            slot.world.set_gate_center_z_mm(slot.body.sim, int(gate_index), float(center_z_mm))
    slot.course.reset(next_gate_index=slot.course_start_gate_index)
    slot.body.reset()
    slot.body.set_root_position_mm((slot.spawn_x_mm, 0.0, slot.spawn_z_mm))
    slot.body.set_root_linear_velocity_mm_s((slot.initial_speed_mm_s, 0.0, slot.initial_vz_mm_s))
    slot.periphery.reset()
    slot.vision.reset_adaptation()
    slot.haltere_sensor.reset(slot.body)
    velocity = slot.body.root_linear_velocity_mm_s()
    slot.final_velocity = tuple(float(value) for value in velocity)


def result_for_slot(
    slot: SlotRuntime,
    commit: dict,
    *,
    curriculum_mode: str = "adaptive",
) -> dict[str, object]:
    return {
        "episode": slot.episode,
        "slot": slot.slot,
        "source_weight_version": slot.source_weight_version,
        "commit_from_version": int(commit["commit_from_version"]),
        "commit_weight_version": int(commit["commit_weight_version"]),
        "version_staleness": int(commit["staleness"]),
        "control_steps": slot.control_steps,
        "passed_gates": slot.passed_gates,
        "collision": slot.collision,
        "collision_reason": slot.collision_reason,
        "finished": slot.finished,
        "max_x_mm": slot.max_x_mm,
        "min_z_mm": slot.min_z_mm,
        "max_z_mm": slot.max_z_mm,
        "final_vx_mm_s": float(slot.final_velocity[0]),
        "spawn_x_mm": slot.spawn_x_mm,
        "spawn_z_mm": slot.spawn_z_mm,
        "initial_speed_mm_s": slot.initial_speed_mm_s,
        "initial_vz_mm_s": slot.initial_vz_mm_s,
        "environment_version": slot.course.environment_version,
        "motor_boundary": "whole-body",
        "curriculum_mode": curriculum_mode,
        "boundary_ease_level": slot.boundary_ease_level,
        "boundary_attempt_index": slot.boundary_attempt_index,
        "boundary_batch_number": slot.boundary_batch_number,
        "frontier_role": slot.frontier_role,
        "frontier_target_gate": slot.frontier_target_gate,
        "course_start_gate_index": slot.course_start_gate_index,
        "launch_round": slot.launch_round,
        "reward_events": slot.reward_events,
        "aversive_events": slot.aversive_events,
    }


def main() -> int:
    args = parse_args()
    probe_course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count, environment_version=args.environment_version)
    first_gate = probe_course.gates[0]
    validate(args, first_gate)

    output = args.output_dir
    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint"
    state_path = output / "curriculum-state.json"
    trajectory_path = output / "trajectory.jsonl"
    commit_log_path = output / "commit-log.jsonl"
    summary_path = output / "summary.json"
    storage_recovery = recover_population_storage(output)
    checkpoint_dir_recovery = storage_recovery.checkpoint_directory
    checkpoint_exists = storage_recovery.checkpoint_exists
    if checkpoint_dir_recovery.action not in {"active_checkpoint", "no_checkpoint"}:
        print(f"checkpoint_directory_recovery action={checkpoint_dir_recovery.action}")
    staged_recovery = storage_recovery.staged_state
    if staged_recovery is not None and staged_recovery.action not in {
        "consistent",
        "legacy_checkpoint",
        "none",
    }:
        print(
            "checkpoint_state_recovery action={} checkpoint_v={} active_v={} pending_v={}".format(
                staged_recovery.action,
                staged_recovery.checkpoint_global_weight_version,
                staged_recovery.active_global_weight_version,
                staged_recovery.pending_global_weight_version,
            )
        )

    start_condition = SpawnCondition(
        args.curriculum_start_x_mm,
        float(first_gate.center_z_mm),
        args.curriculum_start_speed_mm_s,
    )
    state = load_curriculum_state(
        state_path,
        start=start_condition,
        target=target_condition(args),
        checkpoint_exists=checkpoint_exists,
    )
    state["curriculum_mode"] = args.curriculum_mode
    state["environment_version"] = args.environment_version
    state["flight_body_version"] = getattr(args, "flight_body_version", "v3")
    state["motor_boundary"] = "whole-body"
    adaptive = adaptive_config(args, first_gate)
    boundary = (
        population_boundary_config(args, first_gate, state)
        if args.curriculum_mode == "boundary-band"
        else None
    )
    boundary_issued_attempts: set[int] = set()
    if boundary is not None:
        ensure_boundary_state(state, boundary)
        state["consecutive_failures"] = 0

    try:
        resume_plan = prepare_population_resume(
            output,
            requested_launch_mode=args.launch_mode,
            checkpoint_exists=checkpoint_exists,
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    initial_global_version = resume_plan.initial_global_weight_version
    start_episode = resume_plan.start_episode
    reproducibility = run_reproducibility_metadata(
        args, body_runtime="in-process", vision_mode_override="raster", vision_rays_override=0
    )
    provenance_path = write_run_provenance(
        output,
        start_episode=start_episode,
        initial_global_weight_version=initial_global_version,
        payload=reproducibility,
    )
    for warning in reproducibility["conditions"].get("compatibility_warnings", []):
        print(f"compatibility_warning={warning}")
    reconciliation = resume_plan.reconciliation
    if reconciliation is not None and reconciliation.changed:
        print(
            "resume_reconciled stable_v={} removed_commits={} removed_trajectory_lines={} "
            "removed_episodes={} next_episode={}".format(
                reconciliation.stable_global_weight_version,
                reconciliation.removed_commits,
                reconciliation.removed_trajectory_lines,
                list(reconciliation.removed_episode_ids),
                reconciliation.next_episode,
            )
        )
    trajectory_mode = resume_plan.trajectory_mode
    commit_mode = resume_plan.commit_mode
    viewer_ids = load_viewer_body_ids(args.viewer_graph) if args.telemetry else ()
    publisher = LiveTelemetryPublisher(output, enabled=args.telemetry)
    slots = [make_slot(args, index) for index in range(args.population)]
    control_dt_s = slots[0].body.timestep * args.physics_steps

    print(
        "population_runtime=enabled population={} shared_weight=true weight_averaging=false "
        "environment={} motor_boundary=whole-body telemetry={} launch_mode={}".format(
            args.population,
            args.environment_version,
            args.telemetry,
            args.launch_mode,
        )
    )

    launched = 0
    completed = 0
    results: list[dict[str, object]] = []
    commit_records: list[dict[str, object]] = []
    saved_state: dict[str, object] = {}
    started = time.perf_counter()
    aggregate_control_steps = 0
    checkpoint_pending = False

    with trajectory_path.open(trajectory_mode, encoding="utf-8") as trajectory, commit_log_path.open(
        commit_mode, encoding="utf-8"
    ) as commit_log:
        with PopulationNeuralBridgeClient(
            snapshot=args.snapshot,
            groups=args.groups,
            slots=args.population,
        ) as brain:
            brain.ping()
            backend_name = str(brain.ready.get("backend", "gpu-population"))
            if checkpoint_exists:
                loaded = brain.load_checkpoint(
                    checkpoint,
                    global_weight_version=initial_global_version,
                )
                print(
                    f"resumed_checkpoint={loaded.get('path')} neural_step={loaded.get('step')} "
                    f"global_weight_version={brain.global_weight_version}"
                )

            def persist_shared_checkpoint() -> dict[str, Any]:
                return persist_population_checkpoint(
                    output_dir=output,
                    checkpoint_dir=checkpoint,
                    curriculum_state=state,
                    population_state=population_state_snapshot(
                        population=args.population,
                        global_weight_version=brain.global_weight_version,
                        curriculum_mode=args.curriculum_mode,
                        launch_mode=args.launch_mode,
                    ),
                    save_checkpoint=brain.save_checkpoint,
                )

            def launch_slot(slot: SlotRuntime) -> bool:
                nonlocal launched
                if launched >= args.episodes:
                    return False
                ease_level: float | None = None
                attempt_index: int | None = None
                batch_number: int | None = None
                frontier_role = "evaluate"
                frontier_target_gate: int | None = None
                course_start_gate_index: int | None = None
                if boundary is None:
                    condition = current_adaptive_condition(state)
                else:
                    attempt_index = next_boundary_attempt_for_group(
                        state,
                        boundary,
                        group_index=slot.slot,
                        group_count=args.population,
                        issued_attempts=boundary_issued_attempts,
                    )
                    if attempt_index is None:
                        return False
                    batch_number = int(dict(state["boundary_band"])["batch_number"])
                    condition, ease_level = boundary_condition_for_attempt(
                        state,
                        boundary,
                        attempt_index,
                        group_count=args.population,
                    )
                    frontier_target_gate = boundary_target_gates(
                        state,
                        boundary,
                        gate_count=args.gate_count,
                    )
                    frontier_role = boundary_frontier_role(
                        attempt_index,
                        group_count=args.population,
                    )
                    if frontier_role == "focus":
                        focus_level = frontier_focus_level(
                            state,
                            boundary,
                            gate_count=args.gate_count,
                        )
                        target_local_index = frontier_target_gate - 1
                        target_gate = slot.course.gates[target_local_index]
                        previous_gate = (
                            slot.course.gates[target_local_index - 1]
                            if target_local_index > 0
                            else None
                        )
                        condition = frontier_focus_condition(
                            condition,
                            target_gate_x_mm=float(target_gate.x_mm),
                            target_gate_z_mm=float(target_gate.center_z_mm),
                            previous_gate_x_mm=(
                                None if previous_gate is None else float(previous_gate.x_mm)
                            ),
                            previous_gate_z_mm=(
                                None
                                if previous_gate is None
                                else float(previous_gate.center_z_mm)
                            ),
                            previous_gate_half_gap_mm=(
                                None
                                if previous_gate is None
                                else float(previous_gate.half_gap_mm)
                            ),
                            focus_level=focus_level,
                        )
                        course_start_gate_index = (
                            slot.course.source_gate_offset + target_local_index
                        )
                    boundary_issued_attempts.add(attempt_index)
                fixed_spawn = fixed_spawn_condition(args)
                if fixed_spawn is not None:
                    condition = fixed_spawn
                episode = start_episode + launched
                launch_round = launch_round_for_index(launched, args.population)
                begin_episode(
                    slot,
                    episode=episode,
                    source_weight_version=brain.global_weight_version,
                    condition=condition,
                    initial_vz_mm_s=(args.fixed_spawn_vz_mm_s if fixed_spawn is not None else 0.0),
                    boundary_ease_level=ease_level,
                    boundary_attempt_index=attempt_index,
                    boundary_batch_number=batch_number,
                    frontier_role=frontier_role,
                    frontier_target_gate=frontier_target_gate,
                    course_start_gate_index=course_start_gate_index,
                    launch_round=launch_round,
                )
                launched += 1
                return True

            for slot in slots:
                if launched >= args.episodes:
                    break
                launch_slot(slot)

            while completed < args.episodes:
                active_slots = [slot for slot in slots if slot.active]
                if not active_slots:
                    raise RuntimeError("population trainer has no active slots before target completion")

                batch_requests = []
                for slot in active_slots:
                    retinal = slot.vision.encode(slot.body.sim, slot.body.fly)
                    haltere = slot.haltere_sensor.encode(slot.body, dt_s=control_dt_s)
                    slot.last_retinal = retinal
                    slot.last_haltere = haltere
                    sensory_currents = (*retinal.body_currents, *haltere.body_currents)
                    neural_sample = (
                        args.telemetry
                        and slot.slot == args.telemetry_slot
                        and slot.control_steps % args.telemetry_stride == 0
                    )
                    if neural_sample and viewer_ids:
                        read_body = tuple(dict.fromkeys((*slot.periphery.body_ids, *viewer_ids)))
                    else:
                        read_body = slot.periphery.body_ids
                    batch_requests.append(
                        {
                            "slot": slot.slot,
                            "stimulate_body": sensory_currents,
                            "read_body": read_body,
                        }
                    )

                batch_spikes = brain.step_batch(batch_requests, plasticity=True)
                terminal_slots: list[SlotRuntime] = []

                for slot in active_slots:
                    body_spikes = batch_spikes.get(slot.slot, {})
                    slot.last_body_spikes = body_spikes
                    peripheral = slot.periphery.step(body_spikes, dt_s=control_dt_s)
                    slot.last_peripheral = peripheral
                    physical_collision, _ = slot.world.step_muscles_until_boundary_contact(
                        slot.body,
                        peripheral,
                        physics_steps=args.physics_steps,
                    )

                    position = slot.body.thorax_position_mm()
                    velocity = slot.body.root_linear_velocity_mm_s()
                    slot.final_velocity = tuple(float(value) for value in velocity)
                    x_mm = float(position[0])
                    z_mm = float(position[2])
                    slot.max_x_mm = max(slot.max_x_mm, x_mm)
                    slot.min_z_mm = min(slot.min_z_mm, z_mm)
                    slot.max_z_mm = max(slot.max_z_mm, z_mm)
                    control_step = slot.control_steps
                    slot.control_steps += 1
                    aggregate_control_steps += 1

                    gate_observation = slot.course.observe(x_mm, z_mm)
                    body_min_x_mm = None
                    if slot.course.needs_full_body_x_sample(x_mm):
                        body_min_x_mm, _ = slot.world.full_body_x_bounds_mm(slot.body.sim)
                    event = slot.course.update(
                        x_mm,
                        z_mm,
                        physical_collision_reason=physical_collision,
                        analytic_body_collision=False,
                        body_min_x_mm=body_min_x_mm,
                    )
                    reward = False
                    aversive = False
                    aversive_current_applied = None
                    gate_miss_distance_mm = None
                    if event.passed_gate:
                        slot.passed_gates += 1
                        slot.reward_events += 1
                        brain.step_slot(
                            slot.slot,
                            stimulate={"reward_dan": args.reward_current},
                            plasticity=True,
                            steps=args.reinforcement_steps,
                        )
                        reward = True
                    if event.collision:
                        slot.collision = True
                        slot.collision_reason = event.collision_reason
                        slot.aversive_events += 1
                        aversive_current_applied, gate_miss_distance_mm = gate_collision_aversive_current(
                            args,
                            gate_observation,
                            event.collision_reason,
                        )
                        brain.step_slot(
                            slot.slot,
                            stimulate={"aversive_dan": aversive_current_applied},
                            plasticity=True,
                            steps=args.reinforcement_steps,
                        )
                        aversive = True
                    if event.finished:
                        slot.finished = True

                    next_gate = getattr(slot.course, "absolute_next_gate_index", slot.course.next_gate_index)
                    if control_step % args.trajectory_stride == 0 or reward or aversive or event.finished:
                        trajectory.write(
                            json.dumps(
                                {
                                    "episode": slot.episode,
                                    "slot": slot.slot,
                                    "source_weight_version": slot.source_weight_version,
                                    "control_step": control_step,
                                    "x_mm": x_mm,
                                    "y_mm": float(position[1]),
                                    "z_mm": z_mm,
                                    "vx_mm_s": float(velocity[0]),
                                    "vy_mm_s": float(velocity[1]),
                                    "vz_mm_s": float(velocity[2]),
                                    "next_gate": next_gate,
                                    "motor_periphery": peripheral.compact_diagnostics(),
                                    "retinal_input": {
                                        "active_columns": slot.last_retinal.active_columns,
                                        "active_photoreceptors": slot.last_retinal.active_photoreceptors,
                                        "mean_current": slot.last_retinal.mean_current,
                                        "max_current": slot.last_retinal.max_current,
                                    },
                                    "reward_stimulated": reward,
                                    "aversive_stimulated": aversive,
                                    "aversive_current": aversive_current_applied,
                                    "gate_miss_distance_mm": gate_miss_distance_mm,
                                    "passed_gate": event.passed_gate,
                                    "collision": event.collision,
                                    "collision_reason": event.collision_reason,
                                    "finished": event.finished,
                                    "environment_version": args.environment_version,
                                    "motor_boundary": "whole-body",
                                    "curriculum_mode": args.curriculum_mode,
                                    "launch_mode": args.launch_mode,
                                    "boundary_ease_level": slot.boundary_ease_level,
                                    "boundary_attempt_index": slot.boundary_attempt_index,
                                    "boundary_batch_number": slot.boundary_batch_number,
                                    "frontier_role": slot.frontier_role,
                                    "frontier_target_gate": slot.frontier_target_gate,
                                    "course_start_gate_index": slot.course_start_gate_index,
                                    "launch_round": slot.launch_round,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )

                    if args.telemetry and slot.slot == args.telemetry_slot:
                        neural_sample = control_step % args.telemetry_stride == 0
                        body_sample = control_step % args.body_telemetry_stride == 0
                        if neural_sample or reward or aversive or event.finished:
                            active_viewer = [
                                body_id for body_id in viewer_ids if body_spikes.get(body_id, False)
                            ]
                            publisher.publish_neural(
                                episode=slot.episode,
                                control_step=control_step,
                                neural_step=brain.last_step,
                                depolarizing_body_ids=active_viewer,
                                hyperpolarizing_body_ids=(),
                                reward=reward,
                                aversive=aversive,
                            )
                        if body_sample or reward or aversive or event.finished:
                            publisher.publish_body(
                                episode=slot.episode,
                                control_step=control_step,
                                sim=slot.body.sim,
                                next_gate=next_gate,
                                passed_gate=event.passed_gate,
                                collision=event.collision,
                                collision_reason=event.collision_reason,
                                reward=reward,
                                aversive=aversive,
                                motor=peripheral.compact_diagnostics(),
                                retinal={
                                    "active_columns": slot.last_retinal.active_columns,
                                    "active_photoreceptors": slot.last_retinal.active_photoreceptors,
                                    "mean_current": slot.last_retinal.mean_current,
                                    "max_current": slot.last_retinal.max_current,
                                },
                            )

                    if slot.collision or slot.finished or slot.control_steps >= args.max_control_steps:
                        terminal_slots.append(slot)

                trajectory.flush()

                for slot in sorted(terminal_slots, key=lambda item: item.slot):
                    commit = brain.commit_slot(
                        slot.slot,
                        source_weight_version=slot.source_weight_version,
                    )
                    completed += 1
                    state["curriculum_episodes"] = int(state["curriculum_episodes"]) + 1
                    batch_completed = None
                    if boundary is None:
                        record_adaptive_result(state, adaptive, success=slot.passed_gates > 0)
                    else:
                        if slot.boundary_attempt_index is None:
                            raise RuntimeError(
                                "boundary-band slot completed without an attempt index"
                            )
                        target_gates = boundary_target_gates(
                            state,
                            boundary,
                            gate_count=args.gate_count,
                        )
                        batch_completed = record_boundary_result(
                            state,
                            boundary,
                            passed_gates=slot.passed_gates,
                            gate_count=args.gate_count,
                            attempt_index=slot.boundary_attempt_index,
                            group=slot.slot,
                            frontier_role=slot.frontier_role,
                        )
                        if batch_completed is not None:
                            boundary_issued_attempts.clear()
                            print(
                                "boundary_batch_complete target_gates={} mastered={} next_target={} "
                                "success={}/{} rate={:.3f} raw_rate={:.3f} aggregation={} "
                                "adjustment={} gate_rates={} hard={} easy={}".format(
                                    batch_completed["evaluated_target_gates"],
                                    batch_completed["frontier_mastered_gates"],
                                    batch_completed["frontier_target_gates"],
                                    batch_completed["successes"],
                                    batch_completed["attempts"],
                                    batch_completed["success_rate"],
                                    batch_completed["raw_success_rate"],
                                    batch_completed["aggregation"],
                                    batch_completed["adjustment"],
                                    batch_completed["gate_pass_rates"],
                                    batch_completed["hard"],
                                    batch_completed["easy"],
                                )
                            )
                    result = result_for_slot(
                        slot,
                        commit,
                        curriculum_mode=args.curriculum_mode,
                    )
                    results.append(result)
                    commit_record = {
                        "commit_seq": int(commit["commit_weight_version"]),
                        "slot": slot.slot,
                        "episode": slot.episode,
                        "source_weight_version": int(commit["source_weight_version"]),
                        "commit_from_version": int(commit["commit_from_version"]),
                        "commit_weight_version": int(commit["commit_weight_version"]),
                        "staleness": int(commit["staleness"]),
                        "reward_events": slot.reward_events,
                        "aversive_events": slot.aversive_events,
                        "passed_gates": slot.passed_gates,
                        "collision": slot.collision,
                        "collision_reason": slot.collision_reason,
                        "control_steps": slot.control_steps,
                        "spawn_x_mm": slot.spawn_x_mm,
                        "spawn_z_mm": slot.spawn_z_mm,
                        "initial_speed_mm_s": slot.initial_speed_mm_s,
                        "curriculum_mode": args.curriculum_mode,
                        "launch_mode": args.launch_mode,
                        "boundary_ease_level": slot.boundary_ease_level,
                        "boundary_attempt_index": slot.boundary_attempt_index,
                        "boundary_batch_number": slot.boundary_batch_number,
                        "frontier_role": slot.frontier_role,
                        "frontier_target_gate": slot.frontier_target_gate,
                        "course_start_gate_index": slot.course_start_gate_index,
                        "launch_round": slot.launch_round,
                    }
                    commit_log.write(json.dumps(commit_record, separators=(",", ":")) + "\n")
                    commit_log.flush()
                    commit_records.append(commit_record)
                    print(
                        "population_commit episode={} slot={} source_v={} from_v={} -> v{} "
                        "staleness={} steps={} passed={} collision={}".format(
                            slot.episode,
                            slot.slot,
                            slot.source_weight_version,
                            commit["commit_from_version"],
                            commit["commit_weight_version"],
                            commit["staleness"],
                            slot.control_steps,
                            slot.passed_gates,
                            slot.collision,
                        )
                    )

                    if completed % args.checkpoint_every == 0:
                        checkpoint_pending = True

                    if (
                        args.launch_mode == "async"
                        and boundary is None
                        and launched < args.episodes
                    ):
                        launch_slot(slot)
                    else:
                        slot.active = False

                if checkpoint_can_flush(
                    launch_mode=args.launch_mode,
                    checkpoint_pending=checkpoint_pending,
                    active_slots=sum(slot.active for slot in slots),
                ):
                    saved_state = persist_shared_checkpoint()
                    checkpoint_pending = False

                if args.launch_mode == "wave" and launched < args.episodes:
                    if not any(slot.active for slot in slots):
                        for idle_slot in slots:
                            if launched >= args.episodes:
                                break
                            launch_slot(idle_slot)
                elif boundary is not None and launched < args.episodes:
                    for idle_slot in slots:
                        if launched >= args.episodes:
                            break
                        if idle_slot.active:
                            continue
                        launch_slot(idle_slot)

            saved_state = persist_shared_checkpoint()
            final_global_version = brain.global_weight_version
            backend_name = str(brain.ready.get("backend", "gpu-population"))

    elapsed = time.perf_counter() - started
    staleness_values = [int(item["staleness"]) for item in commit_records]
    summary = {
        "schema_version": 12,
        "experiment": f"flyppy_{args.environment_version}_shared_weight_population",
        "backend": backend_name,
        "persistent_runtime": True,
        "population": args.population,
        "shared_weight": True,
        "weight_averaging": False,
        "commit_semantics": "episode-local additive+clamp transaction rebased onto latest global weight",
        "launch_mode": args.launch_mode,
        "environment_version": args.environment_version,
        "flight_body_version": getattr(args, "flight_body_version", "v3"),
        "haltere_sensory_feedback": {
            "enabled": float(args.haltere_current_gain) > 0.0,
            "current_gain": float(args.haltere_current_gain),
            "map": str(args.haltere_sensory_map),
            "map_kind": str(args.haltere_sensory_kind),
            "transduction": str(args.haltere_transduction),
        },
        "motor_boundary": "whole-body",
        "physical_spec": {
            "morphology": FLYBODY_V3.morphology,
            "body_length_mm": FLYBODY_V3.body_length_mm,
            "wing_length_mm": FLYBODY_V3.wing_length_mm,
            "total_mass_mg": FLYBODY_V3.total_mass_mg,
            "neural_sex": FLYBODY_V3.neural_sex,
            "morphology_sex": FLYBODY_V3.morphology_sex,
        },
        "environment_spec": {
            "version": args.environment_version,
            "corridor_low_z_mm": probe_course.floor_z_mm,
            "corridor_high_z_mm": probe_course.ceiling_z_mm,
            "lateral_half_width_mm": probe_course.lateral_half_width_mm,
            "first_gate_x_mm": first_gate.x_mm,
            "gate_gap_height_mm": 2.0 * first_gate.half_gap_mm,
            "gate_half_thickness_mm": first_gate.half_thickness_mm,
        },
        "curriculum_mode": args.curriculum_mode,
        "episodes_this_run": args.episodes,
        "episode_start": start_episode,
        "episode_end": start_episode + args.episodes - 1,
        "control_dt_seconds": control_dt_s,
        "aggregate_control_steps": aggregate_control_steps,
        "aggregate_control_steps_per_second": aggregate_control_steps / elapsed,
        "simulated_seconds_this_run": aggregate_control_steps * control_dt_s,
        "total_passed_gates_this_run": sum(int(item["passed_gates"]) for item in results),
        "total_collisions_this_run": sum(bool(item["collision"]) for item in results),
        "curriculum": state,
        "episode_results": sorted(results, key=lambda item: int(item["episode"])),
        "population_weight_checkpoint": str(checkpoint),
        "checkpoint_semantics": "global-weights-only-v1",
        "checkpoint_resume_behavior": "population fast neural/plasticity state is reset; only global weights are restored",
        "checkpoint_neural_step": saved_state.get("step"),
        "checkpoint_neural_step_semantics": "aggregate-slot-neural-step-count-v1",
        "reproducibility": reproducibility,
        "provenance_file": str(provenance_path),
        "global_weight_version_start": initial_global_version,
        "global_weight_version_end": final_global_version,
        "mean_version_staleness": (
            sum(staleness_values) / len(staleness_values) if staleness_values else 0.0
        ),
        "max_version_staleness": max(staleness_values) if staleness_values else 0,
        "commit_log": str(commit_log_path),
        "telemetry_dir": str(output / "live"),
        "viewer_graph": str(args.viewer_graph),
        "elapsed_seconds": elapsed,
        "teaching_signal": "gate pass -> PAM reward DAN current; gate collision -> PPL current graded by vertical miss distance; floor/ceiling -> full PPL current; focused frontier episodes create local success experience but never count as mastery; only full-course evaluation episodes advance the arbitrary gate-count frontier",
    }
    save_json_atomic(summary_path, summary)
    publisher.publish_status(
        running=False,
        backend=backend_name,
        episode=results[-1]["episode"] if results else None,
        control_step=results[-1]["control_steps"] if results else None,
        curriculum=state,
    )
    print(f"summary={summary_path}")
    print(f"trajectory={trajectory_path}")
    print(f"commit_log={commit_log_path}")
    print(f"checkpoint={checkpoint}")
    print(f"elapsed={elapsed:.3f}s")
    print(f"aggregate_control_steps_per_second={aggregate_control_steps / elapsed:.3f}")
    print("flyppy_async_shared_weight_population=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
