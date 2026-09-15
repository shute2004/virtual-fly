#!/usr/bin/env python3
"""Separate direct-eye photometric error from adaptation-state divergence.

The body-excluded image-free sensor is already much closer to FlyGym's raster
oracle, but signed MaleCNS currents can still disagree because retinal
transduction is temporal: a small persistent light bias changes the adapted
baseline and can later flip local contrast sign.

For each physical state and each K rays/ommatidium this probe compares:

1. raster oracle light -> raster-owned adaptation state;
2. direct light -> direct-owned adaptation state (natural direct path);
3. direct light -> *the raster oracle's pre-step adaptation baseline*.

If (3) is much closer to (1) than (2), accumulated adaptation divergence is the
main amplifier. If both remain similarly different, instantaneous direct
photometry is the main source. The probe also fits an affine light calibration
``raster ~= a * direct + b`` only as a diagnostic; it is not applied to the
production sensor.

No training state is modified.
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
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from direct_ommatidia_sensor_bodyexclude import BodyExcludedDirectOmmatidialSensor
from flyppy_course import FlyppyCourse
from malecns_retina import MaleCNSRetina
import train_flyppy_population as trainer
from virtual_fly.training.curriculum import SpawnCondition, current_adaptive_condition, load_state

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
DEFAULT_REPORT = Path("reports/flyppy/direct_ommatidia_adaptation_gap.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--rays", type=int, nargs="+", default=(3, 7, 13))
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


def current_condition(production: Path) -> SpawnCondition:
    course = FlyppyCourse(seed=0, gate_count=6, environment_version="v3")
    state = load_state(
        production / "curriculum-state.json",
        start=SpawnCondition(8.91, float(course.gates[0].center_z_mm), 400.0),
        target=SpawnCondition(0.0, 8.91, 300.0),
        checkpoint_exists=(production / "checkpoint/manifest.json").exists(),
    )
    return current_adaptive_condition(state)


def make_body_args(snapshot: Path, gain: float) -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        gate_count=6,
        wing_motor_map=snapshot / "wing-motor-neurons-v0.json",
        body_motor_map=snapshot / "body-motor-neurons-v0.json",
        retinotopic_map=snapshot / "retinotopic-vision-v1.json",
        photoreceptor_current_gain=float(gain),
    )


def body_id_axis(retina: MaleCNSRetina) -> np.ndarray:
    values = sorted(
        {
            int(body_id)
            for side in ("L", "R")
            for column in retina._runtime_columns[side]
            for body_id in column.body_ids
        }
    )
    return np.asarray(values, dtype=np.int64)


def current_vector(body_ids: np.ndarray, drive) -> np.ndarray:
    lookup = {int(body_id): float(value) for body_id, value in drive.body_currents}
    return np.asarray(
        [lookup.get(int(body_id), 0.0) for body_id in body_ids],
        dtype=np.float64,
    )


def append_current_metrics(reference: np.ndarray, candidate: np.ndarray, target: dict[str, object]) -> None:
    delta = np.abs(reference - candidate)
    target["abs"].extend(float(value) for value in delta)
    active = (np.abs(reference) > 1e-12) | (np.abs(candidate) > 1e-12)
    target["sign_total"] += int(np.count_nonzero(active))
    target["sign_mismatch"] += int(
        np.count_nonzero(np.sign(reference[active]) != np.sign(candidate[active]))
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if len(x) < 2:
        return 1.0, 0.0, 0.0, 0.0
    design = np.column_stack((x, np.ones_like(x)))
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    a = float(coeff[0])
    b = float(coeff[1])
    pred = a * x + b
    residual = y - pred
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual * residual))
    centered = y - float(np.mean(y))
    ss_tot = float(np.sum(centered * centered))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    return a, b, mae, r2


def main() -> int:
    args = parse_args()
    if args.samples < 2 or args.physics_steps < 1:
        raise SystemExit("samples must be >=2 and physics-steps must be positive")
    rays = tuple(dict.fromkeys(int(value) for value in args.rays))
    if not rays or any(value < 1 for value in rays):
        raise SystemExit("--rays must contain positive integers")

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
        raise SystemExit("missing adaptation-gap inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    before = tree_digest(production / "checkpoint")

    gain = 2.0
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
    natural_retinas = {k: MaleCNSRetina(mapping, current_gain=gain) for k in rays}
    shared_baseline_retinas = {k: MaleCNSRetina(mapping, current_gain=gain) for k in rays}
    sensors = {
        k: BodyExcludedDirectOmmatidialSensor(
            natural_retinas[k],
            rays_per_ommatidium=k,
        )
        for k in rays
    }
    body_ids = body_id_axis(reference_retina)
    for retina in list(natural_retinas.values()) + list(shared_baseline_retinas.values()):
        if not np.array_equal(body_ids, body_id_axis(retina)):
            raise RuntimeError("retinal body-ID axis mismatch")

    # Warm renderer and direct receptive-field bundles outside timing/metrics.
    _ = reference_retina._eye_readouts(slot.body.sim, slot.body.fly)
    for sensor in sensors.values():
        _ = sensor.read_eye_readouts(slot.body.sim, slot.body.fly)
    reference_retina.reset_adaptation()
    for retina in natural_retinas.values():
        retina.reset_adaptation()
    for retina in shared_baseline_retinas.values():
        retina.reset_adaptation()

    metrics: dict[int, dict[str, object]] = {}
    for k in rays:
        metrics[k] = {
            "direct_light": [],
            "reference_light": [],
            "signed_light_error": [],
            "natural": {"abs": [], "sign_total": 0, "sign_mismatch": 0},
            "shared": {"abs": [], "sign_total": 0, "sign_mismatch": 0},
            "sensor_s": 0.0,
        }

    quiet_spikes = {int(body_id): False for body_id in slot.periphery.body_ids}
    control_dt_s = float(slot.body.timestep) * args.physics_steps

    for _sample in range(args.samples):
        reference_eyes = reference_retina._eye_readouts(slot.body.sim, slot.body.fly)
        baseline_before = {
            side: np.array(reference_retina._adapted_light[side], copy=True)
            for side in ("L", "R")
        }
        reference_drive = reference_retina.encode_from_eye_readouts(reference_eyes)
        reference_current = current_vector(body_ids, reference_drive)
        reference_light = {
            side: reference_retina._all_local_achromatic(reference_eyes[side])
            for side in ("L", "R")
        }

        for k in rays:
            started = time.perf_counter()
            direct_eyes = sensors[k].read_eye_readouts(slot.body.sim, slot.body.fly)
            metrics[k]["sensor_s"] += time.perf_counter() - started

            direct_light = {
                side: natural_retinas[k]._all_local_achromatic(direct_eyes[side])
                for side in ("L", "R")
            }
            for side in ("L", "R"):
                indices = np.asarray(sensors[k].required_by_side[side], dtype=np.int32)
                ref = reference_light[side][indices]
                direct = direct_light[side][indices]
                metrics[k]["reference_light"].extend(float(value) for value in ref)
                metrics[k]["direct_light"].extend(float(value) for value in direct)
                metrics[k]["signed_light_error"].extend(float(value) for value in (direct - ref))

            natural_drive = natural_retinas[k].encode_from_eye_readouts(direct_eyes)
            natural_current = current_vector(body_ids, natural_drive)
            append_current_metrics(reference_current, natural_current, metrics[k]["natural"])

            shared_retina = shared_baseline_retinas[k]
            for side in ("L", "R"):
                shared_retina._adapted_light[side][:] = baseline_before[side]
            shared_drive = shared_retina.encode_from_eye_readouts(direct_eyes)
            shared_current = current_vector(body_ids, shared_drive)
            append_current_metrics(reference_current, shared_current, metrics[k]["shared"])

        peripheral = slot.periphery.step(quiet_spikes, dt_s=control_dt_s)
        slot.body.step_muscles(peripheral, physics_steps=args.physics_steps)

    after = tree_digest(production / "checkpoint")
    unchanged = before == after
    overall = unchanged

    lines = [
        "# Flyppy direct ommatidia adaptation-gap probe",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        f"- samples: {args.samples}",
        f"- physics steps between samples: {args.physics_steps}",
        f"- spawn condition: x={condition.x_mm:.6f} mm, z={condition.z_mm:.6f} mm, vx={condition.speed_mm_s:.6f} mm/s",
        "- direct sensor: body-excluded mj_multiRay; no RGB framebuffer",
        f"- production checkpoint modified: {'no' if unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "## Temporal transduction separation",
        "",
        "`natural` lets raster and direct paths accumulate their own adaptation histories. "
        "`shared-baseline` injects direct light while restoring the raster path's pre-step adaptation baseline first.",
        "",
        "| rays/ommatidium | sensor ms | light bias (direct-ref) | light MAE | affine a | affine b | affine MAE | affine R2 | natural current MAE | natural sign mismatch | shared-baseline current MAE | shared-baseline sign mismatch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for k in rays:
        ref = np.asarray(metrics[k]["reference_light"], dtype=np.float64)
        direct = np.asarray(metrics[k]["direct_light"], dtype=np.float64)
        signed = np.asarray(metrics[k]["signed_light_error"], dtype=np.float64)
        a, b, affine_mae, affine_r2 = affine_fit(direct, ref)
        natural = metrics[k]["natural"]
        shared = metrics[k]["shared"]
        natural_abs = natural["abs"]
        shared_abs = shared["abs"]
        natural_sign = (
            natural["sign_mismatch"] / natural["sign_total"]
            if natural["sign_total"]
            else 0.0
        )
        shared_sign = (
            shared["sign_mismatch"] / shared["sign_total"]
            if shared["sign_total"]
            else 0.0
        )
        sensor_ms = 1000.0 * float(metrics[k]["sensor_s"]) / args.samples
        lines.append(
            f"| {k} | {sensor_ms:.3f} | {float(np.mean(signed)):.6f} | "
            f"{float(np.mean(np.abs(signed))):.6f} | {a:.6f} | {b:.6f} | "
            f"{affine_mae:.6f} | {affine_r2:.6f} | "
            f"{float(np.mean(natural_abs)) if natural_abs else 0.0:.6f} | {100.0*natural_sign:.3f}% | "
            f"{float(np.mean(shared_abs)) if shared_abs else 0.0:.6f} | {100.0*shared_sign:.3f}% |"
        )

    lines.extend([
        "",
        "## Current-error tails",
        "",
        "| rays/ommatidium | natural p95 abs | shared-baseline p95 abs |",
        "|---:|---:|---:|",
    ])
    for k in rays:
        lines.append(
            f"| {k} | {percentile(metrics[k]['natural']['abs'], 95):.6f} | "
            f"{percentile(metrics[k]['shared']['abs'], 95):.6f} |"
        )

    lines.extend([
        "",
        "The affine fit is diagnostic only. A low affine residual would indicate a mostly global photometric calibration gap; a large residual means geometry/material/local lighting differences remain.",
        "",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

    try:
        slot.body.sim.eye_renderer = None
    except Exception:
        pass

    print(f"direct_ommatidia_adaptation_gap={report}")
    print(f"direct_ommatidia_adaptation_gap_pass={str(overall).lower()}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
