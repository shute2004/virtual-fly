#!/usr/bin/env python3
"""End-to-end cutover benchmark for image-free Flyppy vision.

Raster and direct-ray cases start from independent copies of the same production
checkpoint.  The direct sensory semantics are intentionally different, so final
checkpoint equality is *not* required.  Instead this benchmark checks that both
paths complete real shared-CNS training, commit every episode, emit finite
retinal/motor outcomes, save a valid checkpoint, and leave production state
untouched while comparing wall-clock throughput.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-direct-vision-cutover")
DEFAULT_REPORT = Path("reports/flyppy/direct_vision_cutover_benchmark.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-control-steps", type=int, default=96)
    p.add_argument("--rays", type=int, default=13)
    p.add_argument("--trajectory-stride", type=int, default=8)
    return p.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def tree_digest(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def clone_state(production: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(production / "checkpoint", out / "checkpoint")
    shutil.copy2(production / "curriculum-state.json", out / "curriculum-state.json")
    population = production / "population-state.json"
    if population.exists():
        shutil.copy2(population, out / "population-state.json")


def retinal_stats(path: Path) -> dict[str, float | int]:
    count = 0
    active_columns = 0.0
    active_photoreceptors = 0.0
    mean_current = 0.0
    max_current = 0.0
    if not path.exists():
        return {
            "samples": 0,
            "active_columns": 0.0,
            "active_photoreceptors": 0.0,
            "mean_current": 0.0,
            "max_current": 0.0,
        }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        retinal = item.get("retinal_input")
        if not isinstance(retinal, dict):
            continue
        count += 1
        active_columns += float(retinal.get("active_columns", 0))
        active_photoreceptors += float(retinal.get("active_photoreceptors", 0))
        mean_current += float(retinal.get("mean_current", 0.0))
        max_current = max(max_current, float(retinal.get("max_current", 0.0)))
    denom = float(count) if count else 1.0
    return {
        "samples": count,
        "active_columns": active_columns / denom,
        "active_photoreceptors": active_photoreceptors / denom,
        "mean_current": mean_current / denom,
        "max_current": max_current,
    }


def finite_episode_results(summary: dict) -> bool:
    numeric = (
        "control_steps", "passed_gates", "max_x_mm", "min_z_mm", "max_z_mm",
        "final_vx_mm_s", "spawn_x_mm", "spawn_z_mm", "initial_speed_mm_s",
        "reward_events", "aversive_events", "version_staleness",
    )
    for item in summary.get("episode_results", []):
        for key in numeric:
            value = item.get(key)
            if value is None or not math.isfinite(float(value)):
                return False
    return True


def run_case(
    *, mode: str,
    rays: int,
    out: Path,
    population: int,
    episodes: int,
    max_steps: int,
    trajectory_stride: int,
    env: dict[str, str],
) -> dict[str, object]:
    case_env = dict(env)
    case_env["VF_FLYPPY_VISION_MODE"] = mode
    case_env["VF_FLYPPY_OMMATIDIA_RAYS"] = str(rays)
    command = [
        sys.executable,
        str(EMBODIMENT / "train_flyppy_population_packed.py"),
        "--episodes", str(episodes),
        "--population", str(population),
        "--output-dir", str(out),
        "--max-control-steps", str(max_steps),
        "--checkpoint-every", str(episodes + 1000),
        "--trajectory-stride", str(trajectory_stride),
        "--curriculum-x-step-mm", "0.000000001",
        "--curriculum-failure-x-step-mm", "0.000000001",
        "--curriculum-z-step-mm", "0.000000001",
        "--curriculum-speed-step-mm-s", "0.000000001",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=case_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-180:])
        raise RuntimeError(f"{mode} trainer failed ({completed.returncode}):\n{tail}")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    retina = retinal_stats(out / "trajectory.jsonl")
    version_start = int(summary["global_weight_version_start"])
    version_end = int(summary["global_weight_version_end"])
    checkpoint_step = int(summary["checkpoint_neural_step"])
    valid = (
        int(summary["episodes_this_run"]) == episodes
        and len(summary.get("episode_results", [])) == episodes
        and version_end - version_start == episodes
        and checkpoint_step > 0
        and int(summary["aggregate_control_steps"]) > 0
        and finite_episode_results(summary)
        and int(retina["samples"]) > 0
        and float(retina["active_photoreceptors"]) > 0.0
        and (out / "checkpoint/manifest.json").exists()
    )
    return {
        "valid": valid,
        "summary": summary,
        "retina": retina,
        "checkpoint_digest": tree_digest(out / "checkpoint"),
        "stdout": completed.stdout,
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.population <= 32:
        raise SystemExit("population must be in 1..32")
    if args.episodes < args.population:
        raise SystemExit("episodes must be >= population")
    if args.max_control_steps < 1 or args.rays < 1 or args.trajectory_stride < 1:
        raise SystemExit("max-control-steps, rays, and trajectory-stride must be positive")

    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    required = [
        production / "checkpoint/manifest.json",
        production / "curriculum-state.json",
        ROOT / DEFAULT_CALIBRATION,
        EMBODIMENT / "train_flyppy_population_packed.py",
        EMBODIMENT / "direct_ommatidia_sensor_bodyexclude.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing direct cutover inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    env["VF_FLYPPY_BODY_PROCESSES"] = "auto"

    before = tree_digest(production / "checkpoint")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    raster_out = temp / "raster"
    direct_out = temp / f"direct-k{args.rays}"
    clone_state(production, raster_out)
    clone_state(production, direct_out)

    raster = run_case(
        mode="raster",
        rays=args.rays,
        out=raster_out,
        population=args.population,
        episodes=args.episodes,
        max_steps=args.max_control_steps,
        trajectory_stride=args.trajectory_stride,
        env=env,
    )
    direct = run_case(
        mode="direct-ray",
        rays=args.rays,
        out=direct_out,
        population=args.population,
        episodes=args.episodes,
        max_steps=args.max_control_steps,
        trajectory_stride=args.trajectory_stride,
        env=env,
    )

    after = tree_digest(production / "checkpoint")
    unchanged = before == after
    direct_meta = direct["summary"]
    metadata_ok = (
        direct_meta.get("vision_runtime") == "direct-ray"
        and int(direct_meta.get("vision_rays_per_ommatidium", -1)) == args.rays
        and direct_meta.get("vision_framebuffer") is False
    )
    overall = bool(raster["valid"] and direct["valid"] and metadata_ok and unchanged)

    rs = raster["summary"]
    ds = direct["summary"]
    rr = raster["retina"]
    dr = direct["retina"]
    raster_sps = float(rs["aggregate_control_steps_per_second"])
    direct_sps = float(ds["aggregate_control_steps_per_second"])
    speedup = direct_sps / raster_sps

    lines = [
        "# Flyppy direct-vision production cutover benchmark",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        f"- population: {args.population}",
        f"- episodes per case: {args.episodes}",
        f"- max control steps per episode: {args.max_control_steps}",
        f"- direct rays/ommatidium: {args.rays}",
        "- direct RGB framebuffer: no",
        "- final checkpoint equality is intentionally not required because sensory semantics differ",
        f"- production checkpoint modified: {'no' if unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "| mode | valid | steps/s | elapsed s | aggregate steps | gates | collisions | global version end | neural step | retinal samples | mean active columns | mean active photoreceptors | mean retinal current | max retinal current |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, case in (("raster", raster), (f"direct-ray K={args.rays}", direct)):
        summary = case["summary"]
        retina = case["retina"]
        lines.append(
            "| {} | {} | {:.6f} | {:.3f} | {} | {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.6f} | {:.6f} |".format(
                name,
                case["valid"],
                float(summary["aggregate_control_steps_per_second"]),
                float(summary["elapsed_seconds"]),
                int(summary["aggregate_control_steps"]),
                int(summary["total_passed_gates_this_run"]),
                int(summary["total_collisions_this_run"]),
                int(summary["global_weight_version_end"]),
                int(summary["checkpoint_neural_step"]),
                int(retina["samples"]),
                float(retina["active_columns"]),
                float(retina["active_photoreceptors"]),
                float(retina["mean_current"]),
                float(retina["max_current"]),
            )
        )

    lines.extend(
        [
            "",
            "## Throughput",
            "",
            f"- direct/raster aggregate throughput: **{speedup:.3f}x**",
            f"- raster checkpoint: `{raster['checkpoint_digest']}`",
            f"- direct checkpoint: `{direct['checkpoint_digest']}`",
            f"- direct runtime metadata valid: {metadata_ok}",
            "",
            "A PASS means the image-free K-selected sensor completes the real shared-CNS learning loop with finite retinal drive, asynchronous commits, and a valid checkpoint while production state remains untouched. Behavioral equality to raster is not a cutover requirement.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"direct_vision_cutover_benchmark={report}")
    print(f"direct_vision_cutover_pass={str(overall).lower()}")
    print(f"direct_vs_raster_speedup={speedup:.6f}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
