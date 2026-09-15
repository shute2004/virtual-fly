#!/usr/bin/env python3
"""Compare image-free ommatidial photometry models against the raster oracle.

This probe isolates the remaining compatibility gap after direct ``mj_multiRay``
visibility was shown to be fast.  It compares:

* base-colour direct sensing; and
* headlight-Lambert direct sensing using ray hit normals.

Both paths remain image-free and use the same K-ray receptive-field geometry.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
ANALYSIS = ROOT / "scripts" / "analysis"
for path in (EMBODIMENT, ANALYSIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from direct_ommatidia_photometry import HeadlightDirectOmmatidialSensor
from direct_ommatidia_sensor import DirectOmmatidialSensor
from malecns_retina import MaleCNSRetina
from probe_direct_ommatidia_rays import (
    current_condition,
    current_vector,
    make_body_args,
    percentile,
)
import train_flyppy_population as trainer

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_REPORT = Path("reports/flyppy/direct_ommatidia_photometry_probe.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--physics-steps", type=int, default=10)
    p.add_argument("--rays", type=int, default=1)
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


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.physics_steps < 1 or args.rays < 1:
        raise SystemExit("samples, physics-steps and rays must be positive")

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
        raise SystemExit("missing photometry-probe inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    gain = 2.0
    before = tree_digest(production / "checkpoint")

    condition = current_condition(production)
    slot = trainer.make_slot(make_body_args(snapshot, gain), 0)
    trainer.begin_episode(
        slot,
        episode=0,
        source_weight_version=0,
        condition=condition,
    )

    mapping = snapshot / "retinotopic-vision-v1.json"
    reference_retina = MaleCNSRetina(mapping, current_gain=gain)
    base_retina = MaleCNSRetina(mapping, current_gain=gain)
    lit_retina = MaleCNSRetina(mapping, current_gain=gain)
    base_sensor = DirectOmmatidialSensor(base_retina, rays_per_ommatidium=args.rays)
    lit_sensor = HeadlightDirectOmmatidialSensor(lit_retina, rays_per_ommatidium=args.rays)

    # Warm OpenGL/Numba/direct bundles outside timing.
    _ = reference_retina._eye_readouts(slot.body.sim, slot.body.fly)
    _ = base_sensor.read_eye_readouts(slot.body.sim, slot.body.fly)
    _ = lit_sensor.read_eye_readouts(slot.body.sim, slot.body.fly)
    reference_retina.reset_adaptation()
    base_retina.reset_adaptation()
    lit_retina.reset_adaptation()

    methods = {
        "base-colour": (base_retina, base_sensor),
        "headlight-lambert": (lit_retina, lit_sensor),
    }
    sensor_seconds = {name: 0.0 for name in methods}
    transduction_seconds = {name: 0.0 for name in methods}
    light_errors = {name: [] for name in methods}
    current_errors = {name: [] for name in methods}
    sign_mismatch = {name: 0 for name in methods}
    sign_compared = {name: 0 for name in methods}
    reference_sensor_seconds = 0.0
    reference_transduction_seconds = 0.0

    body_ids = tuple(int(value) for value in slot.periphery.body_ids)
    quiet_spikes = {body_id: False for body_id in body_ids}

    for _ in range(args.samples):
        started = time.perf_counter()
        reference_eyes = reference_retina._eye_readouts(slot.body.sim, slot.body.fly)
        reference_sensor_seconds += time.perf_counter() - started
        started = time.perf_counter()
        reference_drive = reference_retina.encode_from_eye_readouts(reference_eyes)
        reference_transduction_seconds += time.perf_counter() - started
        ref_ids, ref_current = current_vector(reference_retina, reference_drive)
        reference_light = {
            side: reference_retina._all_local_achromatic(reference_eyes[side])
            for side in ("L", "R")
        }

        for name, (retina, sensor) in methods.items():
            started = time.perf_counter()
            eyes = sensor.read_eye_readouts(slot.body.sim, slot.body.fly)
            sensor_seconds[name] += time.perf_counter() - started
            started = time.perf_counter()
            drive = retina.encode_from_eye_readouts(eyes)
            transduction_seconds[name] += time.perf_counter() - started

            for side in ("L", "R"):
                indices = np.asarray(sensor.required_by_side[side], dtype=np.int32)
                light = retina._all_local_achromatic(eyes[side])
                light_errors[name].extend(
                    float(value)
                    for value in np.abs(reference_light[side][indices] - light[indices])
                )

            body_index, current = current_vector(retina, drive)
            if not np.array_equal(ref_ids, body_index):
                raise RuntimeError(f"body-current index mismatch for {name}")
            current_errors[name].extend(float(value) for value in np.abs(ref_current - current))
            active = (np.abs(ref_current) > 1e-12) | (np.abs(current) > 1e-12)
            sign_compared[name] += int(np.count_nonzero(active))
            sign_mismatch[name] += int(
                np.count_nonzero(np.sign(ref_current[active]) != np.sign(current[active]))
            )

        peripheral = slot.periphery.step(
            quiet_spikes,
            dt_s=float(slot.body.timestep) * args.physics_steps,
        )
        slot.body.step_muscles(peripheral, physics_steps=args.physics_steps)

    after = tree_digest(production / "checkpoint")
    unchanged = before == after
    model = slot.body.sim.mj_model
    headlight = model.vis.headlight
    reference_ms = 1000.0 * reference_sensor_seconds / args.samples
    reference_total_ms = 1000.0 * (
        reference_sensor_seconds + reference_transduction_seconds
    ) / args.samples

    lines = [
        "# Flyppy direct ommatidia photometry probe",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if unchanged else 'FAIL'}",
        f"- samples: {args.samples}",
        f"- rays per required ommatidium: {args.rays}",
        f"- total rays/readout: {base_sensor.last_stats.total_rays}",
        "- RGB framebuffer used by direct methods: no",
        f"- MuJoCo explicit lights: {int(model.nlight)}",
        f"- headlight active: {bool(headlight.active)}",
        f"- headlight ambient: {list(np.asarray(headlight.ambient, dtype=float))}",
        f"- headlight diffuse: {list(np.asarray(headlight.diffuse, dtype=float))}",
        f"- headlight specular: {list(np.asarray(headlight.specular, dtype=float))}",
        f"- production checkpoint modified: {'no' if unchanged else 'YES'}",
        f"- production checkpoint digest: `{before}`",
        "",
        "## Performance and raster-oracle compatibility",
        "",
        f"- reference raster sensor: {reference_ms:.3f} ms/readout",
        f"- reference raster sensor + transduction: {reference_total_ms:.3f} ms/control observation",
        "",
        "| method | sensor ms | speedup vs raster | sensor+transduction ms | light MAE | light p95 | current MAE | current p95 | sign mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in methods:
        sensor_ms = 1000.0 * sensor_seconds[name] / args.samples
        total_ms = 1000.0 * (
            sensor_seconds[name] + transduction_seconds[name]
        ) / args.samples
        light = light_errors[name]
        current = current_errors[name]
        sign_rate = sign_mismatch[name] / max(sign_compared[name], 1)
        lines.append(
            f"| {name} | {sensor_ms:.3f} | {reference_ms/max(sensor_ms, 1e-12):.3f}x | "
            f"{total_ms:.3f} | {float(np.mean(light)):.6f} | {percentile(light, 95):.6f} | "
            f"{float(np.mean(current)):.6f} | {percentile(current, 95):.6f} | "
            f"{100.0*sign_rate:.3f}% |"
        )

    lines.extend(
        [
            "",
            "The raster path is used here only as a compatibility oracle. The direct sensor remains the architectural target; this probe determines whether the remaining gap is primarily photometric rather than geometric.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

    try:
        slot.body.sim.eye_renderer = None
    except Exception:
        pass

    print(f"direct_ommatidia_photometry_probe={report}")
    print(f"direct_ommatidia_photometry_probe_pass={str(unchanged).lower()}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
