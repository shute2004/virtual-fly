#!/usr/bin/env python3
"""Compare image-free ommatidial ray sensing with the FlyGym raster oracle.

The probe never trains or writes the production checkpoint.  It instantiates one
Flyppy v3 body at the current curriculum condition and evaluates exactly the same
physical states with:

* FlyGym's reference RGB render -> fisheye -> 721-ommatidia path; and
* direct weighted ``mj_multiRay`` receptor sampling for K=1/3/7/13 rays.

Accuracy is reported both at the required ommatidial-light boundary and after the
existing MaleCNS adaptation/transduction seam.  Direct-sensor timings exclude
one-time receptive-field construction; all paths are warmed before measurement.
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
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from direct_ommatidia_sensor import DirectOmmatidialSensor
from flyppy_course import FlyppyCourse
from malecns_retina import MaleCNSRetina
import train_flyppy_population as trainer
from virtual_fly.training.curriculum import (
    SpawnCondition,
    current_adaptive_condition,
    load_state,
)

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_REPORT = Path("reports/flyppy/direct_ommatidia_ray_probe.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--physics-steps", type=int, default=10)
    p.add_argument("--rays", type=int, nargs="+", default=(1, 3, 7, 13))
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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def current_vector(retina: MaleCNSRetina, drive) -> tuple[np.ndarray, np.ndarray]:
    body_ids = sorted(
        {
            int(body_id)
            for side in ("L", "R")
            for column in retina._runtime_columns[side]
            for body_id in column.body_ids
        }
    )
    lookup = {int(body_id): float(value) for body_id, value in drive.body_currents}
    values = np.asarray([lookup.get(body_id, 0.0) for body_id in body_ids], dtype=np.float64)
    return np.asarray(body_ids, dtype=np.int64), values


def make_body_args(snapshot: Path, gain: float) -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        gate_count=6,
        wing_motor_map=snapshot / "wing-motor-neurons-v0.json",
        body_motor_map=snapshot / "body-motor-neurons-v0.json",
        retinotopic_map=snapshot / "retinotopic-vision-v1.json",
        photoreceptor_current_gain=float(gain),
    )


def current_condition(production: Path) -> SpawnCondition:
    course = FlyppyCourse(seed=0, gate_count=6, environment_version="v3")
    first_gate = course.gates[0]
    state = load_state(
        production / "curriculum-state.json",
        start=SpawnCondition(8.91, float(first_gate.center_z_mm), 400.0),
        target=SpawnCondition(0.0, 8.91, 300.0),
        checkpoint_exists=(production / "checkpoint/manifest.json").exists(),
    )
    return current_adaptive_condition(state)


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.physics_steps < 1:
        raise SystemExit("samples and physics-steps must be positive")
    rays = tuple(dict.fromkeys(int(value) for value in args.rays))
    if not rays or any(value < 1 for value in rays):
        raise SystemExit("--rays values must be positive integers")

    production = absolute(args.production)
    report = absolute(args.report)
    snapshot = ROOT / DEFAULT_SNAPSHOT
    required = [
        production / "checkpoint/manifest.json",
        production / "curriculum-state.json",
        ROOT / DEFAULT_CALIBRATION,
        snapshot / "manifest.json",
        snapshot / "retinotopic-vision-v1.json",
        snapshot / "wing-motor-neurons-v0.json",
        snapshot / "body-motor-neurons-v0.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing direct-ray probe inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    gain = 2.0
    before = tree_digest(production / "checkpoint")

    condition = current_condition(production)
    body_args = make_body_args(snapshot, gain)
    slot = trainer.make_slot(body_args, 0)
    trainer.begin_episode(
        slot,
        episode=0,
        source_weight_version=0,
        condition=condition,
    )

    mapping = snapshot / "retinotopic-vision-v1.json"
    reference_retina = MaleCNSRetina(mapping, current_gain=gain)
    direct_retinas = {
        k: MaleCNSRetina(mapping, current_gain=gain)
        for k in rays
    }
    sensors = {
        k: DirectOmmatidialSensor(direct_retinas[k], rays_per_ommatidium=k)
        for k in rays
    }

    # Warm OpenGL/Numba and direct receptive-field bundles before timing.
    _ = reference_retina._eye_readouts(slot.body.sim, slot.body.fly)
    for k in rays:
        _ = sensors[k].read_eye_readouts(slot.body.sim, slot.body.fly)
    reference_retina.reset_adaptation()
    for retina in direct_retinas.values():
        retina.reset_adaptation()

    reference_sensor_s = 0.0
    reference_transduction_s = 0.0
    direct_sensor_s = {k: 0.0 for k in rays}
    direct_transduction_s = {k: 0.0 for k in rays}
    light_abs_error: dict[int, list[float]] = {k: [] for k in rays}
    current_abs_error: dict[int, list[float]] = {k: [] for k in rays}
    current_sign_mismatch: dict[int, int] = {k: 0 for k in rays}
    current_sign_compared: dict[int, int] = {k: 0 for k in rays}
    finite_ok = True

    body_ids = tuple(int(value) for value in slot.periphery.body_ids)
    quiet_spikes = {body_id: False for body_id in body_ids}

    for _sample in range(args.samples):
        started = time.perf_counter()
        reference_eyes = reference_retina._eye_readouts(slot.body.sim, slot.body.fly)
        reference_sensor_s += time.perf_counter() - started
        started = time.perf_counter()
        reference_drive = reference_retina.encode_from_eye_readouts(reference_eyes)
        reference_transduction_s += time.perf_counter() - started
        ref_ids, ref_current = current_vector(reference_retina, reference_drive)

        reference_light = {
            side: reference_retina._all_local_achromatic(reference_eyes[side])
            for side in ("L", "R")
        }

        for k in rays:
            sensor = sensors[k]
            retina = direct_retinas[k]
            started = time.perf_counter()
            direct_eyes = sensor.read_eye_readouts(slot.body.sim, slot.body.fly)
            direct_sensor_s[k] += time.perf_counter() - started
            started = time.perf_counter()
            direct_drive = retina.encode_from_eye_readouts(direct_eyes)
            direct_transduction_s[k] += time.perf_counter() - started

            for side in ("L", "R"):
                indices = np.asarray(sensor.required_by_side[side], dtype=np.int32)
                direct_light = retina._all_local_achromatic(direct_eyes[side])
                delta = np.abs(reference_light[side][indices] - direct_light[indices])
                light_abs_error[k].extend(float(value) for value in delta)
                finite_ok = finite_ok and bool(np.all(np.isfinite(direct_light[indices])))

            direct_ids, direct_current = current_vector(retina, direct_drive)
            if not np.array_equal(ref_ids, direct_ids):
                raise RuntimeError(f"body-current index mismatch for K={k}")
            delta_current = np.abs(ref_current - direct_current)
            current_abs_error[k].extend(float(value) for value in delta_current)
            active = (np.abs(ref_current) > 1e-12) | (np.abs(direct_current) > 1e-12)
            current_sign_compared[k] += int(np.count_nonzero(active))
            current_sign_mismatch[k] += int(
                np.count_nonzero(np.sign(ref_current[active]) != np.sign(direct_current[active]))
            )
            finite_ok = finite_ok and bool(np.all(np.isfinite(direct_current)))

        peripheral = slot.periphery.step(quiet_spikes, dt_s=float(slot.body.timestep) * args.physics_steps)
        slot.body.step_muscles(peripheral, physics_steps=args.physics_steps)

    after = tree_digest(production / "checkpoint")
    unchanged = before == after
    unique_left = len(sensors[rays[0]].required_by_side["L"])
    unique_right = len(sensors[rays[0]].required_by_side["R"])
    required_count_ok = unique_left > 0 and unique_right > 0
    overall = unchanged and finite_ok and required_count_ok

    ref_sensor_ms = 1000.0 * reference_sensor_s / args.samples
    ref_total_ms = 1000.0 * (reference_sensor_s + reference_transduction_s) / args.samples

    lines = [
        "# Flyppy direct ommatidia ray probe",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        f"- samples: {args.samples}",
        f"- physics steps between samples: {args.physics_steps}",
        f"- spawn condition: x={condition.x_mm:.6f} mm, z={condition.z_mm:.6f} mm, vx={condition.speed_mm_s:.6f} mm/s",
        f"- required unique ommatidia: L={unique_left}, R={unique_right}, total={unique_left + unique_right}",
        "- direct path creates RGB framebuffer: no",
        "- direct visibility query: MuJoCo mj_multiRay",
        "- direct photometry: geom/material base channel + white sky + analytic ground checker approximation",
        f"- production checkpoint modified: {'no' if unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "## Performance",
        "",
        f"- reference raster sensor: {ref_sensor_ms:.3f} ms/readout",
        f"- reference raster sensor + transduction: {ref_total_ms:.3f} ms/control observation",
        "",
        "| rays/ommatidium | total rays/readout | direct sensor ms | sensor speedup vs raster | direct sensor+transduction ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in rays:
        sensor_ms = 1000.0 * direct_sensor_s[k] / args.samples
        total_ms = 1000.0 * (direct_sensor_s[k] + direct_transduction_s[k]) / args.samples
        lines.append(
            f"| {k} | {sensors[k].last_stats.total_rays} | {sensor_ms:.3f} | "
            f"{(ref_sensor_ms / sensor_ms if sensor_ms > 0 else float('inf')):.3f}x | {total_ms:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Compatibility against FlyGym raster oracle",
            "",
            "Errors cover only ommatidia that currently feed released MaleCNS retinal columns. "
            "The direct path intentionally does not attempt exact OpenGL lighting/texture filtering yet.",
            "",
            "| rays/ommatidium | light MAE | light p95 abs | light max abs | MaleCNS current MAE | current p95 abs | current sign mismatch |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k in rays:
        light = light_abs_error[k]
        currents = current_abs_error[k]
        sign_total = current_sign_compared[k]
        sign_rate = current_sign_mismatch[k] / sign_total if sign_total else 0.0
        lines.append(
            f"| {k} | {float(np.mean(light)) if light else 0.0:.6f} | {percentile(light, 95):.6f} | "
            f"{max(light, default=0.0):.6f} | {float(np.mean(currents)) if currents else 0.0:.6f} | "
            f"{percentile(currents, 95):.6f} | {100.0 * sign_rate:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation contract",
            "",
            "This probe validates the image-free sensor mechanics and measures the remaining photometric approximation gap. "
            "Production should not switch from the raster oracle solely because this probe executes; the next decision is based on both wall-clock gain and MaleCNS-current compatibility.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

    # Avoid keeping the offscreen renderer alive after the probe.
    try:
        slot.body.sim.eye_renderer = None
    except Exception:
        pass

    print(f"direct_ommatidia_ray_probe={report}")
    print(f"direct_ommatidia_ray_probe_pass={str(overall).lower()}")
    print(f"required_ommatidia={unique_left + unique_right}")
    for k in rays:
        sensor_ms = 1000.0 * direct_sensor_s[k] / args.samples
        print(
            f"K{k}_rays={sensors[k].last_stats.total_rays} "
            f"K{k}_sensor_ms={sensor_ms:.6f} "
            f"K{k}_speedup={ref_sensor_ms / max(sensor_ms, 1e-12):.6f}"
        )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
