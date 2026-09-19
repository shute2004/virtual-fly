#!/usr/bin/env python3
"""Capture one frozen Flyppy episode for deterministic offline playback."""
from __future__ import annotations

import argparse
from pathlib import Path

from virtual_fly.playback.capture import capture_playback
from virtual_fly.paths import PRODUCTION_NEUTRAL_TRIM_PATTERN
from virtual_fly.playback.config import (
    DEFAULT_CALIBRATION,
    DEFAULT_COURSE_SEED,
    DEFAULT_PRODUCTION,
    DEFAULT_SNAPSHOT,
    DEFAULT_SPAWN_X_MM,
    DEFAULT_SPAWN_Z_MM,
    DEFAULT_SPEED_MM_S,
    DEFAULT_VIEWER_GRAPH,
)
from virtual_fly.semantics import (
    BODY_VERSIONS,
    HALTERE_FULL_KIND,
    HALTERE_TIMING_KIND,
    HALTERE_TRANSDUCTIONS,
    PRODUCTION_ENVIRONMENT_VERSIONS,
)

DEFAULT_OUTPUT = Path("artifacts/experiments/flyppy-best-playback/playback.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fresh-brain", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--viewer-graph", type=Path, default=DEFAULT_VIEWER_GRAPH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--haltere-sensory-map", type=Path, default=Path("artifacts/malecns-v1.0/haltere-campaniform-sensory-v1.json"))
    parser.add_argument("--haltere-sensory-kind", choices=(HALTERE_FULL_KIND, HALTERE_TIMING_KIND), default=HALTERE_FULL_KIND)
    parser.add_argument("--haltere-current-gain", type=float, default=0.0)
    parser.add_argument("--haltere-transduction", choices=HALTERE_TRANSDUCTIONS, default="angular-acceleration-v1")
    parser.add_argument("--environment-version", choices=PRODUCTION_ENVIRONMENT_VERSIONS, required=True)
    parser.add_argument("--flight-body-version", choices=BODY_VERSIONS, required=True)
    parser.add_argument("--vertical-steering-gain", type=float, default=1.0)
    parser.add_argument("--measured-steering-gain", type=float, default=1.0)
    parser.add_argument("--neutral-trim-strength", type=float, default=1.0)
    parser.add_argument("--neutral-trim-pattern", type=Path, default=PRODUCTION_NEUTRAL_TRIM_PATTERN)
    parser.add_argument("--steering-tau-ms", type=float, default=12.0)
    parser.add_argument("--steering-spike-increment", type=float, default=0.85)
    parser.add_argument("--course-seed", type=int, default=DEFAULT_COURSE_SEED)
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--spawn-x-mm", type=float, default=DEFAULT_SPAWN_X_MM)
    parser.add_argument("--spawn-z-mm", type=float, default=DEFAULT_SPAWN_Z_MM)
    parser.add_argument("--initial-speed-mm-s", type=float, default=DEFAULT_SPEED_MM_S)
    parser.add_argument("--initial-vz-mm-s", type=float, default=0.0)
    parser.add_argument("--gate2-center-z-mm", type=float, default=None)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--physics-substep-stride", type=int, default=3)
    parser.add_argument("--max-control-steps", type=int, default=1800)
    parser.add_argument("--photoreceptor-current-gain", type=float, default=2.0)
    parser.add_argument("--reward-current", type=float, default=2.0)
    parser.add_argument("--aversive-current", type=float, default=2.0)
    parser.add_argument("--reinforcement-steps", type=int, default=4)
    parser.add_argument("--neural-telemetry-stride", type=int, default=5)
    parser.add_argument("--playback-fps", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--plasticity", action="store_true")
    parser.add_argument("--reward-from-gate", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = capture_playback(args)
    print(
        "playback_capture=PASS checkpoint_step={} global_v={} frames={} controls={} passed_gates={} terminal={}".format(
            payload["checkpoint_neural_step"],
            payload["global_weight_version"],
            payload["frame_count"],
            payload["control_steps"],
            payload["passed_gates"],
            payload["terminal_reason"],
        )
    )
    print(f"playback_json={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
