#!/usr/bin/env python3
"""Diagnose direct Flyppy eye-ray hit geometry independently of photometry.

The depth-alignment probe showed identical aggregate metrics for every camera-frame
convention.  This diagnostic inspects the raw mj_multiRay outputs themselves:

- candidate-specific direction fingerprints;
- geom-id / distance fingerprints;
- hit geom names, owning bodies and geom groups;
- very-near hits that can reveal self-intersection at the eye origin.

No training state or production checkpoint is modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
DEFAULT_REPORT = Path("reports/flyppy/eye_ray_hit_diagnostics.md")


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


def representative_pixels(sensor: DirectOmmatidialSensor, side: str) -> tuple[np.ndarray, np.ndarray]:
    rows_out: list[int] = []
    cols_out: list[int] = []
    for index in sensor.required_by_side[side]:
        rows, cols, _total = sensor._source_pixels[index]
        if len(rows) == 0:
            continue
        center_row = float(np.mean(rows))
        center_col = float(np.mean(cols))
        d2 = (rows.astype(np.float64) - center_row) ** 2 + (cols.astype(np.float64) - center_col) ** 2
        chosen = int(np.argmin(d2))
        rows_out.append(int(rows[chosen]))
        cols_out.append(int(cols[chosen]))
    return np.asarray(rows_out, dtype=np.int32), np.asarray(cols_out, dtype=np.int32)


def local_rays(rows: np.ndarray, cols: np.ndarray, *, nrows: int, ncols: int, fovy: float, sx: int, sy: int, sz: int) -> np.ndarray:
    focal = 0.5 * float(nrows) / math.tan(math.radians(float(fovy)) / 2.0)
    x = sx * ((cols.astype(np.float64) + 0.5) - 0.5 * ncols) / focal
    y = sy * (0.5 * nrows - (rows.astype(np.float64) + 0.5)) / focal
    z = np.full_like(x, float(sz))
    rays = np.column_stack((x, y, z))
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-15)
    return rays


def fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()[:16]


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


def geom_label(model, geom_id: int) -> tuple[str, str, int]:
    if geom_id < 0:
        return ("<background>", "<none>", -1)
    geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or f"geom#{geom_id}"
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or f"body#{body_id}"
    return geom_name, body_name, int(model.geom_group[geom_id])


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
        raise SystemExit("missing ray-hit diagnostic inputs:\n  " + "\n  ".join(missing))

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

    # Four deliberately distant conventions are enough to prove whether the
    # binding/output changes with direction. The original convention is first.
    candidates = [
        ("current", "R.T", 1, 1, -1),
        ("flip-z", "R.T", 1, 1, 1),
        ("flip-xy", "R.T", -1, -1, -1),
        ("matrix-R", "R", 1, 1, -1),
    ]

    results: list[dict[str, object]] = []
    for label, matrix_mode, sx, sy, sz in candidates:
        combined_dirs: list[np.ndarray] = []
        combined_ids: list[np.ndarray] = []
        combined_distances: list[np.ndarray] = []
        geom_counter: Counter[tuple[str, str, int]] = Counter()
        group_counter: Counter[int] = Counter()
        near_hits = 0
        hit_count = 0
        side_rows: dict[str, dict[str, object]] = {}

        for side in ("L", "R"):
            camera_id = int(camera_ids[side])
            rows, cols = representative_pixels(sensor, side)
            local = local_rays(
                rows,
                cols,
                nrows=int(sensor.retina.nrows),
                ncols=int(sensor.retina.ncols),
                fovy=float(sim.mj_model.cam_fovy[camera_id]),
                sx=sx,
                sy=sy,
                sz=sz,
            )
            rotation = np.asarray(sim.mj_data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
            world = local @ (rotation.T if matrix_mode == "R.T" else rotation)
            world /= np.maximum(np.linalg.norm(world, axis=1, keepdims=True), 1e-15)
            geom_ids, distances = ray_hits(
                sim.mj_model,
                sim.mj_data,
                np.asarray(sim.mj_data.cam_xpos[camera_id], dtype=np.float64),
                world,
            )
            combined_dirs.append(world)
            combined_ids.append(geom_ids)
            combined_distances.append(distances)

            valid = (geom_ids >= 0) & np.isfinite(distances) & (distances >= 0.0)
            side_hit_count = int(np.count_nonzero(valid))
            hit_count += side_hit_count
            near_hits += int(np.count_nonzero(valid & (distances <= 1e-4)))
            for geom_id in geom_ids[valid]:
                info = geom_label(sim.mj_model, int(geom_id))
                geom_counter[info] += 1
                group_counter[info[2]] += 1
            side_rows[side] = {
                "samples": len(geom_ids),
                "hits": side_hit_count,
                "direction_fp": fingerprint(world),
                "output_fp": fingerprint(geom_ids, np.round(distances, 9)),
                "min_distance": float(np.min(distances[valid])) if side_hit_count else float("nan"),
                "median_distance": float(np.median(distances[valid])) if side_hit_count else float("nan"),
            }

        all_dirs = np.concatenate(combined_dirs, axis=0)
        all_ids = np.concatenate(combined_ids, axis=0)
        all_dist = np.concatenate(combined_distances, axis=0)
        results.append(
            {
                "label": label,
                "matrix": matrix_mode,
                "sx": sx,
                "sy": sy,
                "sz": sz,
                "samples": len(all_ids),
                "hits": hit_count,
                "near_hits": near_hits,
                "direction_fp": fingerprint(all_dirs),
                "output_fp": fingerprint(all_ids, np.round(all_dist, 9)),
                "geom_counter": geom_counter,
                "group_counter": group_counter,
                "side_rows": side_rows,
            }
        )

    after = tree_digest(production / "checkpoint")
    unchanged = before == after
    direction_fps = {str(item["direction_fp"]) for item in results}
    output_fps = {str(item["output_fp"]) for item in results}

    lines = [
        "# Flyppy eye ray hit diagnostics",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if unchanged else 'FAIL'}",
        f"- candidate direction fingerprints unique: {len(direction_fps)}/{len(results)}",
        f"- candidate mj_multiRay output fingerprints unique: {len(output_fps)}/{len(results)}",
        f"- production checkpoint modified: {'no' if unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "## Camera ownership",
        "",
    ]
    for side in ("L", "R"):
        camera_id = int(camera_ids[side])
        body_id = int(sim.mj_model.cam_bodyid[camera_id])
        camera_name = mj.mj_id2name(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, camera_id) or f"camera#{camera_id}"
        body_name = mj.mj_id2name(sim.mj_model, mj.mjtObj.mjOBJ_BODY, body_id) or f"body#{body_id}"
        pos = np.asarray(sim.mj_data.cam_xpos[camera_id], dtype=np.float64)
        lines.append(f"- {side}: camera={camera_name}, body={body_name}, body_id={body_id}, pos={pos.tolist()}")

    lines.extend([
        "",
        "## Candidate fingerprints",
        "",
        "| candidate | matrix | sx | sy | sz | hits | near<=1e-4 | direction fp | output fp |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for item in results:
        lines.append(
            f"| {item['label']} | {item['matrix']} | {item['sx']} | {item['sy']} | {item['sz']} | "
            f"{item['hits']}/{item['samples']} | {item['near_hits']} | `{item['direction_fp']}` | `{item['output_fp']}` |"
        )

    for item in results:
        lines.extend([
            "",
            f"## {item['label']} top hits",
            "",
            f"- geom groups: {dict(sorted(item['group_counter'].items()))}",
        ])
        for side in ("L", "R"):
            row = item["side_rows"][side]
            lines.append(
                f"- {side}: hits={row['hits']}/{row['samples']}, min_distance={row['min_distance']:.9f}, "
                f"median_distance={row['median_distance']:.9f}, dir_fp=`{row['direction_fp']}`, out_fp=`{row['output_fp']}`"
            )
        lines.extend([
            "",
            "| count | geom | body | group |",
            "|---:|---|---|---:|",
        ])
        for (geom_name, body_name, group), count in item["geom_counter"].most_common(15):
            lines.append(f"| {count} | `{geom_name}` | `{body_name}` | {group} |")

    lines.extend([
        "",
        "If direction fingerprints differ but mj_multiRay output fingerprints do not, the issue is downstream of direction construction (for example origin-inside/self geometry or API filtering). If both differ, the previous aggregate tie was caused by the depth/classification contract rather than an inert ray transform.",
        "",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

    try:
        slot.body.sim.eye_renderer = None
    except Exception:
        pass

    print(f"eye_ray_hit_diagnostics={report}")
    print(f"direction_fingerprints={len(direction_fps)}")
    print(f"output_fingerprints={len(output_fps)}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
