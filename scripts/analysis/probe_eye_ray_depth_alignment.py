#!/usr/bin/env python3
"""Validate direct eye rays against MuJoCo's depth buffer, independent of colour/IDs.

The previous segmentation-ID probe produced 0% for every camera-frame convention,
which makes the object-ID comparison contract suspect.  This diagnostic therefore
compares geometry through metric depth only:

    raw eye pixel -> MuJoCo depth buffer z-depth
    same pixel    -> mj_multiRay Euclidean hit distance -> camera z-depth

For a perspective camera, z-depth = ray_distance * abs(unit_local_ray.z).  The
probe evaluates the same 16 matrix/sign conventions as the segmentation probe and
also reports hit/background classification agreement.  No training state is
modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from direct_ommatidia_sensor import DirectOmmatidialSensor, _EYE_GEOMGROUP
from flyppy_course import FlyppyCourse
from malecns_retina import MaleCNSRetina
import train_flyppy_population as trainer
from virtual_fly.training.curriculum import SpawnCondition, current_adaptive_condition, load_state

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
DEFAULT_REPORT = Path("reports/flyppy/eye_ray_depth_alignment.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def condition_for(production: Path) -> SpawnCondition:
    course = FlyppyCourse(seed=0, gate_count=6, environment_version="v3")
    state = load_state(
        production / "curriculum-state.json",
        start=SpawnCondition(8.91, float(course.gates[0].center_z_mm), 400.0),
        target=SpawnCondition(0.0, 8.91, 300.0),
        checkpoint_exists=(production / "checkpoint/manifest.json").exists(),
    )
    return current_adaptive_condition(state)


def make_args(snapshot: Path):
    from types import SimpleNamespace

    return SimpleNamespace(
        seed=0,
        gate_count=6,
        wing_motor_map=snapshot / "wing-motor-neurons-v0.json",
        body_motor_map=snapshot / "body-motor-neurons-v0.json",
        retinotopic_map=snapshot / "retinotopic-vision-v1.json",
        photoreceptor_current_gain=2.0,
    )


def representative_pixels(
    sensor: DirectOmmatidialSensor, side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ommatidia: list[int] = []
    rows_out: list[int] = []
    cols_out: list[int] = []
    for index in sensor.required_by_side[side]:
        rows, cols, _total = sensor._source_pixels[index]
        if len(rows) == 0:
            continue
        center_row = float(np.mean(rows))
        center_col = float(np.mean(cols))
        distance2 = (
            (rows.astype(np.float64) - center_row) ** 2
            + (cols.astype(np.float64) - center_col) ** 2
        )
        chosen = int(np.argmin(distance2))
        ommatidia.append(int(index))
        rows_out.append(int(rows[chosen]))
        cols_out.append(int(cols[chosen]))
    return (
        np.asarray(ommatidia, dtype=np.int32),
        np.asarray(rows_out, dtype=np.int32),
        np.asarray(cols_out, dtype=np.int32),
    )


def local_rays(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    nrows: int,
    ncols: int,
    fovy_deg: float,
    sx: int,
    sy: int,
    sz: int,
) -> np.ndarray:
    # This is algebraically equivalent to MuJoCo 3.9's cam_project inverse when
    # no explicit camera intrinsics/sensor size are supplied.  MuJoCo uses the
    # image height to define both fx and fy from vertical FOV.
    focal = 0.5 * float(nrows) / math.tan(math.radians(float(fovy_deg)) / 2.0)
    x = sx * ((cols.astype(np.float64) + 0.5) - 0.5 * ncols) / focal
    y = sy * (0.5 * nrows - (rows.astype(np.float64) + 0.5)) / focal
    z = np.full_like(x, float(sz))
    directions = np.column_stack((x, y, z))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)
    return directions


def ray_hits(model, data, origin: np.ndarray, directions_world: np.ndarray):
    directions_world = np.ascontiguousarray(directions_world, dtype=np.float64)
    geom_ids = np.empty(len(directions_world), dtype=np.int32)
    distances = np.empty(len(directions_world), dtype=np.float64)
    mj.mj_multiRay(
        model,
        data,
        np.ascontiguousarray(origin, dtype=np.float64),
        directions_world.reshape(-1),
        _EYE_GEOMGROUP,
        1,
        -1,
        geom_ids,
        distances,
        None,
        len(directions_world),
        10_000.0,
    )
    return geom_ids, distances


def main() -> int:
    args = parse_args()
    production = absolute(args.production)
    report = absolute(args.report)
    snapshot = ROOT / DEFAULT_SNAPSHOT
    required = [
        production / "checkpoint/manifest.json",
        production / "curriculum-state.json",
        ROOT / DEFAULT_CALIBRATION,
        snapshot / "manifest.json",
        snapshot / "retinotopic-vision-v1.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing depth-alignment inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    before = tree_digest(production / "checkpoint")

    slot = trainer.make_slot(make_args(snapshot), 0)
    trainer.begin_episode(
        slot,
        episode=0,
        source_weight_version=0,
        condition=condition_for(production),
    )
    retina = MaleCNSRetina(snapshot / "retinotopic-vision-v1.json")
    sensor = DirectOmmatidialSensor(retina, rays_per_ommatidium=1)
    sim = slot.body.sim
    fly = slot.body.fly
    camera_ids = sensor._camera_ids_by_side(sim, fly)

    renderer = mj.Renderer(
        sim.mj_model,
        height=int(sensor.retina.nrows),
        width=int(sensor.retina.ncols),
    )
    renderer.enable_depth_rendering()
    scene_option = mj.MjvOption()
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 0

    far_plane = float(sim.mj_model.vis.map.zfar) * float(sim.mj_model.stat.extent)
    side_payload: dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for side in ("L", "R"):
        camera_id = int(camera_ids[side])
        ommatidia, rows, cols = representative_pixels(sensor, side)
        renderer.update_scene(sim.mj_data, camera_id, scene_option=scene_option)
        depth = np.asarray(renderer.render(), dtype=np.float64)
        sampled_depth = np.asarray(depth[rows, cols], dtype=np.float64)
        side_payload[side] = (camera_id, ommatidia, rows, cols, sampled_depth)

    candidates: list[dict[str, object]] = []
    for matrix_mode in ("R.T", "R"):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    all_abs_error: list[float] = []
                    all_sq_error: list[float] = []
                    class_matches = 0
                    class_total = 0
                    paired_hits = 0
                    side_stats: dict[str, tuple[float, float, int, int]] = {}
                    for side in ("L", "R"):
                        camera_id, _omm, rows, cols, raster_depth = side_payload[side]
                        local = local_rays(
                            rows,
                            cols,
                            nrows=int(sensor.retina.nrows),
                            ncols=int(sensor.retina.ncols),
                            fovy_deg=float(sim.mj_model.cam_fovy[camera_id]),
                            sx=sx,
                            sy=sy,
                            sz=sz,
                        )
                        rotation = np.asarray(
                            sim.mj_data.cam_xmat[camera_id], dtype=np.float64
                        ).reshape(3, 3)
                        world = local @ (rotation.T if matrix_mode == "R.T" else rotation)
                        world /= np.maximum(
                            np.linalg.norm(world, axis=1, keepdims=True), 1e-15
                        )
                        geom_ids, distances = ray_hits(
                            sim.mj_model,
                            sim.mj_data,
                            sim.mj_data.cam_xpos[camera_id],
                            world,
                        )

                        raster_hit = np.isfinite(raster_depth) & (raster_depth < far_plane * 0.999)
                        ray_hit = (geom_ids >= 0) & np.isfinite(distances) & (distances >= 0.0)
                        class_same = raster_hit == ray_hit
                        class_matches += int(np.count_nonzero(class_same))
                        class_total += len(class_same)

                        both = raster_hit & ray_hit
                        predicted_z = distances[both] * np.abs(local[both, 2])
                        error = np.abs(predicted_z - raster_depth[both])
                        sq = (predicted_z - raster_depth[both]) ** 2
                        all_abs_error.extend(float(value) for value in error)
                        all_sq_error.extend(float(value) for value in sq)
                        paired_hits += int(np.count_nonzero(both))
                        side_stats[side] = (
                            float(np.mean(error)) if len(error) else float("inf"),
                            math.sqrt(float(np.mean(sq))) if len(sq) else float("inf"),
                            int(np.count_nonzero(class_same)),
                            len(class_same),
                        )

                    mae = float(np.mean(all_abs_error)) if all_abs_error else float("inf")
                    rmse = math.sqrt(float(np.mean(all_sq_error))) if all_sq_error else float("inf")
                    classification = class_matches / class_total if class_total else 0.0
                    candidates.append(
                        {
                            "matrix": matrix_mode,
                            "sx": sx,
                            "sy": sy,
                            "sz": sz,
                            "mae": mae,
                            "rmse": rmse,
                            "classification": classification,
                            "paired_hits": paired_hits,
                            "L": side_stats["L"],
                            "R": side_stats["R"],
                        }
                    )

    candidates.sort(
        key=lambda item: (
            -float(item["classification"]),
            float(item["rmse"]),
            float(item["mae"]),
            str(item["matrix"]),
        )
    )
    current = next(
        item
        for item in candidates
        if item["matrix"] == "R.T"
        and item["sx"] == 1
        and item["sy"] == 1
        and item["sz"] == -1
    )
    best = candidates[0]

    after = tree_digest(production / "checkpoint")
    unchanged = before == after
    overall = unchanged

    def fmt(value: float) -> str:
        return "inf" if not math.isfinite(value) else f"{value:.6f}"

    lines = [
        "# Flyppy eye ray depth alignment",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        "- oracle: MuJoCo metric depth rendering at FlyGym Retina raw resolution",
        "- comparison: representative raw source pixel per required ommatidium",
        f"- required samples: L={len(side_payload['L'][1])}, R={len(side_payload['R'][1])}",
        f"- far plane: {far_plane:.6f}",
        f"- current convention: matrix=R.T, sx=1, sy=1, sz=-1",
        f"- current hit/background agreement: {100.0 * float(current['classification']):.3f}%",
        f"- current paired-hit depth MAE: {fmt(float(current['mae']))}",
        f"- current paired-hit depth RMSE: {fmt(float(current['rmse']))}",
        f"- best convention: matrix={best['matrix']}, sx={best['sx']}, sy={best['sy']}, sz={best['sz']}",
        f"- best hit/background agreement: {100.0 * float(best['classification']):.3f}%",
        f"- best paired-hit depth MAE: {fmt(float(best['mae']))}",
        f"- best paired-hit depth RMSE: {fmt(float(best['rmse']))}",
        f"- production checkpoint modified: {'no' if unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "## All camera-frame conventions",
        "",
        "| rank | matrix | sx | sy | sz | hit/bg agreement | paired hits | depth MAE | depth RMSE | L agreement | R agreement |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(candidates, 1):
        left = item["L"]
        right = item["R"]
        lines.append(
            f"| {rank} | {item['matrix']} | {item['sx']} | {item['sy']} | {item['sz']} | "
            f"{100.0 * float(item['classification']):.3f}% | {int(item['paired_hits'])} | "
            f"{fmt(float(item['mae']))} | {fmt(float(item['rmse']))} | "
            f"{100.0 * left[2] / left[3]:.3f}% | {100.0 * right[2] / right[3]:.3f}% |"
        )

    lines.extend(
        [
            "",
            "Depth alignment avoids segmentation object-ID semantics entirely. A correct camera/ray convention should simultaneously maximize hit/background agreement and minimize paired-hit metric-depth error.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

    try:
        renderer.close()
    except Exception:
        pass
    try:
        slot.body.sim.eye_renderer = None
    except Exception:
        pass

    print(f"eye_ray_depth_alignment={report}")
    print(f"current_classification={float(current['classification']):.6f}")
    print(f"current_rmse={float(current['rmse']):.6f}")
    print(f"best_classification={float(best['classification']):.6f}")
    print(f"best_rmse={float(best['rmse']):.6f}")
    print(
        f"best_matrix={best['matrix']} best_sx={best['sx']} "
        f"best_sy={best['sy']} best_sz={best['sz']}"
    )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
