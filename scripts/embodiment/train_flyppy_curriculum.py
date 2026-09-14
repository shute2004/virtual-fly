#!/usr/bin/env python3
"""Fast curriculum driver for Flyppy learning.

This is an implementation/training utility, not a scientific evaluation harness.
It repeatedly runs one Flyppy episode while preserving learned CNS weights. The
first curriculum stage makes successful gate passage easy enough to experience:
start close to the first gate, near its vertical center, and with a generous
one-shot initial forward velocity. Every successful first-gate passage moves the
initial condition toward the real target (x=0 mm, z=5 mm, vx=300 mm/s). Once the
target is reached, subsequent episodes are ordinary target-condition training.

Only episode-reset initial conditions are changed. There is no per-step position,
velocity, action, or policy controller. Reinforcement inside each episode remains
current injection into the released PAM/PPL DAN groups in flyppy_closed_loop.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from flyppy_course import FlyppyCourse


DEFAULT_OUTPUT = Path("artifacts/experiments/flyppy-v1")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"),
    )
    parser.add_argument(
        "--retinotopic-map",
        type=Path,
        default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"),
    )
    parser.add_argument(
        "--wing-motor-map",
        type=Path,
        default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"),
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--gate-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard the previous Flyppy checkpoint/trajectory and restart curriculum",
    )
    parser.add_argument("--curriculum-start-x-mm", type=float, default=5.5)
    parser.add_argument("--curriculum-target-x-mm", type=float, default=0.0)
    parser.add_argument("--curriculum-x-step-mm", type=float, default=0.75)
    parser.add_argument("--curriculum-target-z-mm", type=float, default=5.0)
    parser.add_argument("--curriculum-z-step-mm", type=float, default=0.25)
    parser.add_argument("--curriculum-start-speed-mm-s", type=float, default=400.0)
    parser.add_argument("--curriculum-target-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--curriculum-speed-step-mm-s", type=float, default=25.0)
    parser.add_argument("--curriculum-max-easy-speed-mm-s", type=float, default=500.0)
    parser.add_argument("--curriculum-failure-x-step-mm", type=float, default=0.5)
    args, forwarded = parser.parse_known_args()
    return args, forwarded


def move_toward(value: float, target: float, step: float) -> float:
    if value < target:
        return min(target, value + step)
    if value > target:
        return max(target, value - step)
    return target


def validate_forwarded(args: list[str]) -> None:
    controlled = {
        "--episodes",
        "--output-dir",
        "--snapshot",
        "--groups",
        "--retinotopic-map",
        "--wing-motor-map",
        "--backend",
        "--gate-count",
        "--seed",
        "--trajectory-stride",
        "--resume-checkpoint",
        "--checkpoint-dir",
        "--checkpoint-every",
        "--initial-forward-speed-mm-s",
        "--progress-reward-current",
        "--progress-reward-steps",
        "--progress-reward-start-mm",
        "--progress-reward-spacing-mm",
        "--no-progress-reward",
    }
    conflicts = sorted({item for item in args if item.split("=", 1)[0] in controlled})
    if conflicts:
        raise SystemExit(
            "curriculum driver controls these arguments internally: " + ", ".join(conflicts)
        )


def new_state(*, default_x: float, default_z: float, default_speed: float) -> dict[str, object]:
    return {
        "schema_version": 2,
        "spawn_x_mm": default_x,
        "spawn_z_mm": default_z,
        "initial_speed_mm_s": default_speed,
        "successful_first_gates": 0,
        "consecutive_failures": 0,
        "curriculum_episodes": 0,
    }


def load_state(
    path: Path, *, default_x: float, default_z: float, default_speed: float
) -> dict[str, object]:
    if not path.exists():
        return new_state(default_x=default_x, default_z=default_z, default_speed=default_speed)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to read curriculum state {path}: {exc}") from exc
    if int(state.get("schema_version", 0)) != 2:
        return new_state(default_x=default_x, default_z=default_z, default_speed=default_speed)
    return state


def save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args, forwarded = parse_args()
    validate_forwarded(forwarded)

    if args.episodes < 1:
        raise SystemExit("episodes must be >= 1")
    positive = (
        args.curriculum_x_step_mm,
        args.curriculum_z_step_mm,
        args.curriculum_start_speed_mm_s,
        args.curriculum_target_speed_mm_s,
        args.curriculum_speed_step_mm_s,
        args.curriculum_max_easy_speed_mm_s,
        args.curriculum_failure_x_step_mm,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise SystemExit("curriculum step/speed parameters must be finite and positive")
    for value in (
        args.curriculum_start_x_mm,
        args.curriculum_target_x_mm,
        args.curriculum_target_z_mm,
    ):
        if not math.isfinite(value):
            raise SystemExit("curriculum position parameters must be finite")
    if args.curriculum_max_easy_speed_mm_s < args.curriculum_start_speed_mm_s:
        raise SystemExit("curriculum-max-easy-speed-mm-s must be >= start speed")

    course = FlyppyCourse(seed=args.seed, gate_count=args.gate_count)
    first_gate = course.gates[0]
    if args.curriculum_start_x_mm >= first_gate.x_mm:
        raise SystemExit("curriculum-start-x-mm must be before the first gate")

    default_x = float(args.curriculum_start_x_mm)
    default_z = float(first_gate.center_z_mm)
    default_speed = float(args.curriculum_start_speed_mm_s)

    output_dir = args.output_dir
    checkpoint = output_dir / "checkpoint"
    state_path = output_dir / "curriculum-state.json"
    summary_path = output_dir / "summary.json"

    if args.fresh:
        state_path.unlink(missing_ok=True)

    state = load_state(
        state_path,
        default_x=default_x,
        default_z=default_z,
        default_speed=default_speed,
    )
    if not checkpoint.joinpath("manifest.json").exists() and not args.fresh:
        state = new_state(default_x=default_x, default_z=default_z, default_speed=default_speed)

    print(
        "curriculum first_gate_x={:.3f} gate_center_z={:.3f} target=(x={:.3f},z={:.3f},vx={:.1f})".format(
            float(first_gate.x_mm),
            float(first_gate.center_z_mm),
            args.curriculum_target_x_mm,
            args.curriculum_target_z_mm,
            args.curriculum_target_speed_mm_s,
        )
    )

    script = Path(__file__).with_name("flyppy_closed_loop.py")
    repo_root = Path(__file__).resolve().parents[2]

    for local_episode in range(args.episodes):
        spawn_x = float(state["spawn_x_mm"])
        spawn_z = float(state["spawn_z_mm"])
        speed = float(state["initial_speed_mm_s"])
        resume = checkpoint.joinpath("manifest.json").exists() and not (
            args.fresh and local_episode == 0
        )

        command = [
            sys.executable,
            str(script),
            "--episodes",
            "1",
            "--snapshot",
            str(args.snapshot),
            "--groups",
            str(args.groups),
            "--retinotopic-map",
            str(args.retinotopic_map),
            "--wing-motor-map",
            str(args.wing_motor_map),
            "--backend",
            args.backend,
            "--gate-count",
            str(args.gate_count),
            "--seed",
            str(args.seed),
            "--trajectory-stride",
            str(args.trajectory_stride),
            "--output-dir",
            str(output_dir),
            "--checkpoint-every",
            "1",
            "--initial-forward-speed-mm-s",
            f"{speed:.9g}",
            "--no-progress-reward",
            *forwarded,
        ]
        if resume:
            command.extend(["--resume-checkpoint", str(checkpoint)])

        env = os.environ.copy()
        env["VF_CURRICULUM_SPAWN_X"] = f"{spawn_x:.9g}"
        env["VF_CURRICULUM_SPAWN_Z"] = f"{spawn_z:.9g}"
        print(
            "curriculum_episode={} spawn=(x={:.3f},z={:.3f}) initial_vx={:.1f} resume={}".format(
                int(state["curriculum_episodes"]), spawn_x, spawn_z, speed, resume
            )
        )
        completed = subprocess.run(command, cwd=repo_root, env=env)
        if completed.returncode != 0:
            return completed.returncode

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results = summary.get("episode_results", [])
        if len(results) != 1:
            raise RuntimeError("one curriculum subprocess did not produce exactly one episode")
        result = results[0]
        passed = int(result.get("passed_gates", 0))

        state["curriculum_episodes"] = int(state["curriculum_episodes"]) + 1
        if passed > 0:
            state["successful_first_gates"] = int(state["successful_first_gates"]) + 1
            state["consecutive_failures"] = 0
            state["spawn_x_mm"] = move_toward(
                spawn_x, args.curriculum_target_x_mm, args.curriculum_x_step_mm
            )
            state["spawn_z_mm"] = move_toward(
                spawn_z, args.curriculum_target_z_mm, args.curriculum_z_step_mm
            )
            state["initial_speed_mm_s"] = move_toward(
                speed,
                args.curriculum_target_speed_mm_s,
                args.curriculum_speed_step_mm_s,
            )
        else:
            state["consecutive_failures"] = int(state["consecutive_failures"]) + 1
            # Make only the next episode's initial condition easier. This cannot
            # steer the fly after reset and is undone progressively after success.
            easiest_x = float(first_gate.x_mm) - 1.0
            state["spawn_x_mm"] = min(
                easiest_x, spawn_x + args.curriculum_failure_x_step_mm
            )
            state["initial_speed_mm_s"] = min(
                args.curriculum_max_easy_speed_mm_s,
                speed + args.curriculum_speed_step_mm_s,
            )

        state["target_x_mm"] = float(args.curriculum_target_x_mm)
        state["target_z_mm"] = float(args.curriculum_target_z_mm)
        state["target_speed_mm_s"] = float(args.curriculum_target_speed_mm_s)
        state["curriculum_complete"] = bool(
            math.isclose(float(state["spawn_x_mm"]), args.curriculum_target_x_mm, abs_tol=1e-9)
            and math.isclose(float(state["spawn_z_mm"]), args.curriculum_target_z_mm, abs_tol=1e-9)
            and math.isclose(
                float(state["initial_speed_mm_s"]),
                args.curriculum_target_speed_mm_s,
                abs_tol=1e-9,
            )
        )
        save_state(state_path, state)

        print(
            "curriculum_result passed={} max_x={:.3f} next_spawn=(x={:.3f},z={:.3f}) "
            "next_vx={:.1f} failures={} complete={}".format(
                passed,
                float(result.get("max_x_mm", 0.0)),
                float(state["spawn_x_mm"]),
                float(state["spawn_z_mm"]),
                float(state["initial_speed_mm_s"]),
                int(state["consecutive_failures"]),
                state["curriculum_complete"],
            )
        )

    print(f"curriculum_state={state_path}")
    print(f"checkpoint={checkpoint}")
    print("flyppy_curriculum=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
