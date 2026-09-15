#!/usr/bin/env python3
"""Compare eye-camera raster geometry IDs against direct MuJoCo ray hits.

This diagnostic deliberately ignores colour and lighting.  For one representative raw
camera pixel from every currently required ommatidium, it renders MuJoCo segmentation
and asks mj_multiRay which geom the corresponding camera ray hits.  Several camera-frame
conventions are evaluated so a transform/sign mistake can be separated from photometry.

No training or production checkpoint mutation occurs.
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
DEFAULT_REPORT = Path("reports/flyppy/eye_ray_geometry_alignment.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
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


def representative_pixels(sensor: DirectOmmatidialSensor, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omm: list[int] = []
    rows_out: list[int] = []
    cols_out: list[int] = []
    for index in sensor.required_by_side[side]:
        rows, cols, _total = sensor._source_pixels[index]
        if len(rows) == 0:
            continue
        center = np.asarray([float(np.mean(rows)), float(np.mean(cols))], dtype=np.float64)
        d2 = (rows.astype(np.float64) - center[0]) ** 2 + (cols.astype(np.float64) - center[1]) ** 2
        chosen = int(np.argmin(d2))
        omm.append(int(index))
        rows_out.append(int(rows[chosen]))
        cols_out.append(int(cols[chosen]))
    return (
        np.asarray(omm, dtype=np.int32),
        np.asarray(rows_out, dtype=np.int32),
        np.asarray(cols_out, dtype=np.int32),
    )


def local_rays(rows: np.ndarray, cols: np.ndarray, nrows: int, ncols: int, fovy: float, sx: int, sy: int, sz: int) -> np.ndarray:
    tan_half_y = math.tan(math.radians(float(fovy)) / 2.0)
    tan_half_x = tan_half_y * (float(ncols) / float(nrows))
    x = sx * (2.0 * (cols.astype(np.float64) + 0.5) / ncols - 1.0) * tan_half_x
    y = sy * (1.0 - 2.0 * (rows.astype(np.float64) + 0.5) / nrows) * tan_half_y
    z = np.full_like(x, float(sz))
    dirs = np.column_stack((x, y, z))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-15)
    return dirs


def ray_hits(model, data, origin, dirs_world):
    dirs_world = np.ascontiguousarray(dirs_world, dtype=np.float64)
    geom_ids = np.empty(len(dirs_world), dtype=np.int32)
    distances = np.empty(len(dirs_world), dtype=np.float64)
    mj.mj_multiRay(
        model,
        data,
        np.ascontiguousarray(origin, dtype=np.float64),
        dirs_world.reshape(-1),
        _EYE_GEOMGROUP,
        1,
        -1,
        geom_ids,
        distances,
        None,
        len(dirs_world),
        10_000.0,
    )
    return geom_ids


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
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing geometry-alignment inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    before = tree_digest(production / "checkpoint")

    slot = trainer.make_slot(make_args(snapshot), 0)
    trainer.begin_episode(slot, episode=0, source_weight_version=0, condition=condition_for(production))
    retina = MaleCNSRetina(snapshot / "retinotopic-vision-v1.json")
    sensor = DirectOmmatidialSensor(retina, rays_per_ommatidium=1)
    sim = slot.body.sim
    fly = slot.body.fly
    camera_ids = sensor._camera_ids_by_side(sim, fly)

    # Use a dedicated segmentation renderer at exactly the Retina raw resolution.
    renderer = mj.Renderer(sim.mj_model, height=sensor.retina.nrows, width=sensor.retina.ncols)
    renderer.enable_segmentation_rendering()
    scene_option = mj.MjvOption()
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 0

    candidates = []
    side_payload = {}
    for side in ("L", "R"):
        camera_id = camera_ids[side]
        omm, rows, cols = representative_pixels(sensor, side)
        renderer.update_scene(sim.mj_data, camera_id, scene_option=scene_option)
        seg = renderer.render()
        sampled = seg[rows, cols]
        geom_type = int(mj.mjtObj.mjOBJ_GEOM)
        oracle = np.where(sampled[:, 1] == geom_type, sampled[:, 0], -1).astype(np.int32)
        side_payload[side] = (camera_id, omm, rows, cols, oracle)

    for matrix_mode in ("R.T", "R"):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    total = 0
                    matches = 0
                    side_rates = {}
                    for side in ("L", "R"):
                        camera_id, _omm, rows, cols, oracle = side_payload[side]
                        local = local_rays(rows, cols, sensor.retina.nrows, sensor.retina.ncols, float(sim.mj_model.cam_fovy[camera_id]), sx, sy, sz)
                        rotation = np.asarray(sim.mj_data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
                        world = local @ (rotation.T if matrix_mode == "R.T" else rotation)
                        world /= np.maximum(np.linalg.norm(world, axis=1, keepdims=True), 1e-15)
                        hits = ray_hits(sim.mj_model, sim.mj_data, sim.mj_data.cam_xpos[camera_id], world)
                        same = hits == oracle
                        count = len(same)
                        matched = int(np.count_nonzero(same))
                        total += count
                        matches += matched
                        side_rates[side] = matched / count if count else 0.0
                    candidates.append({
                        "matrix": matrix_mode,
                        "sx": sx,
                        "sy": sy,
                        "sz": sz,
                        "matches": matches,
                        "total": total,
                        "rate": matches / total if total else 0.0,
                        "L": side_rates["L"],
                        "R": side_rates["R"],
                    })

    candidates.sort(key=lambda x: (-x["rate"], x["matrix"], -x["sx"], -x["sy"], x["sz"]))
    current = next(c for c in candidates if c["matrix"] == "R.T" and c["sx"] == 1 and c["sy"] == 1 and c["sz"] == -1)
    best = candidates[0]
    after = tree_digest(production / "checkpoint")
    unchanged = before == after
    overall = unchanged

    lines = [
        "# Flyppy eye ray geometry alignment",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        "- oracle: MuJoCo segmentation rendering at FlyGym Retina raw resolution",
        "- comparison: representative raw source pixel per required ommatidium",
        f"- required samples: L={len(side_payload['L'][1])}, R={len(side_payload['R'][1])}",
        f"- current convention match: {100.0 * current['rate']:.3f}% (L={100.0*current['L']:.3f}%, R={100.0*current['R']:.3f}%)",
        f"- best convention: matrix={best['matrix']}, sx={best['sx']}, sy={best['sy']}, sz={best['sz']}",
        f"- best match: {100.0 * best['rate']:.3f}% (L={100.0*best['L']:.3f}%, R={100.0*best['R']:.3f}%)",
        f"- production checkpoint modified: {'no' if unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "## All camera-frame conventions",
        "",
        "| rank | matrix | sx | sy | sz | combined match | L match | R match |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(candidates, 1):
        lines.append(
            f"| {rank} | {item['matrix']} | {item['sx']} | {item['sy']} | {item['sz']} | "
            f"{100.0*item['rate']:.3f}% | {100.0*item['L']:.3f}% | {100.0*item['R']:.3f}% |"
        )

    lines.extend([
        "",
        "A high segmentation-ID match means camera projection/pose and ray geometry agree independently of lighting. If the current convention is not the best, direct sensor geometry should be corrected before further photometry work.",
        "",
    ])
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
    print(f"eye_ray_geometry_alignment={report}")
    print(f"current_match={current['rate']:.6f}")
    print(f"best_match={best['rate']:.6f}")
    print(f"best_matrix={best['matrix']} best_sx={best['sx']} best_sy={best['sy']} best_sz={best['sz']}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
