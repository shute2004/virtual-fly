"""Shared Flyppy population-training configuration and semantic helpers.

This module contains pure/configuration logic shared by in-process, process-isolated,
and packed production runners. It must not own MuJoCo, GPU, or subprocess lifetime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

from virtual_fly.paths import REPO_ROOT
from virtual_fly.runtime.vision import VisionRuntimeConfig
from virtual_fly.semantics import (
    BODY_VERSIONS,
    HALTERE_FULL_KIND,
    HALTERE_TIMING_KIND,
    HALTERE_TRANSDUCTIONS,
    PRODUCTION_ENVIRONMENT_VERSIONS,
    SHARED_WEIGHT_COMMIT_SEMANTICS,
)
from virtual_fly.reproducibility import (
    build_run_provenance,
    compatibility_warnings,
    validate_derived_artifact,
    validate_haltere_map,
    validate_production_snapshot,
)
from virtual_fly.training.curriculum import (
    AdaptiveCurriculumConfig,
    BoundaryBandConfig,
    SpawnCondition,
    current_adaptive_condition,
    record_adaptive_result,
)

ROOT = REPO_ROOT

def normalized_vision_config() -> tuple[str, int]:
    config = VisionRuntimeConfig.from_environment(default_mode="raster", default_rays=7)
    return config.mode, config.rays_per_ommatidium


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
            "shared_weight_commit_semantics": SHARED_WEIGHT_COMMIT_SEMANTICS,
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
        choices=PRODUCTION_ENVIRONMENT_VERSIONS,
        default="v3",
        help=(
            "v4 preserves the historical tall-corridor/narrow-hole-range course; "
            "v5 keeps v4 physics but uses the full vertical corridor for diverse gate openings"
        ),
    )
    parser.add_argument(
        "--flight-body-version",
        choices=BODY_VERSIONS,
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
        choices=HALTERE_TRANSDUCTIONS,
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
