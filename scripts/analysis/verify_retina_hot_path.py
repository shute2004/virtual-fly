#!/usr/bin/env python3
"""Verify the optimized MaleCNS retinal transduction path.

The optimized path computes all 721 achromatic ommatidial values once per eye,
but must preserve the historical MaleCNS column order because several columns may
map to the same FlyBody ommatidium and therefore share adaptation state.

This verifier compares the optimized path with a scalar reference implementation
of the previous encode loop over a deterministic sequence of synthetic compound-
eye readouts. It compares every emitted body current, diagnostic value, and final
adaptation state without touching the production experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from malecns_retina import MaleCNSRetina, RetinalDrive

DEFAULT_MAP = Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json")
DEFAULT_REPORT = Path("reports/flyppy/retina_hot_path_verify.md")


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def f64_bits(value: float) -> bytes:
    return struct.pack("<d", float(value))


def scalar_reference(
    retina: MaleCNSRetina,
    adapted: dict[str, np.ndarray],
    eyes: dict[str, np.ndarray],
) -> RetinalDrive:
    body_currents: list[tuple[int, float]] = []
    active_columns = 0
    magnitudes: list[float] = []

    for side in ("L", "R"):
        eye = eyes[side]
        if len(eye) != retina.flybody_ommatidia_per_eye:
            raise RuntimeError("synthetic eye shape mismatch")
        for column in retina._columns_by_side[side]:
            index = int(column["ommatidium_index"])
            light = retina._local_achromatic(eye, index)
            baseline = float(adapted[side][index])
            if not np.isfinite(baseline):
                adapted[side][index] = light
                current = light * retina.current_gain
            else:
                denominator = max(abs(baseline), retina.contrast_denominator_floor)
                contrast = (light - baseline) / denominator
                contrast = float(np.clip(contrast, -retina.contrast_clip, retina.contrast_clip))
                adapted[side][index] = baseline + retina._adaptation_alpha * (light - baseline)
                current = contrast * retina.current_gain
            if abs(current) <= retina.current_floor:
                continue
            active_columns += 1
            magnitude = abs(current)
            for body_id in column["body_ids"]:
                body_currents.append((int(body_id), current))
                magnitudes.append(magnitude)

    return RetinalDrive(
        body_currents=tuple(body_currents),
        active_photoreceptors=len(body_currents),
        active_columns=active_columns,
        mean_current=float(np.mean(magnitudes)) if magnitudes else 0.0,
        max_current=float(np.max(magnitudes)) if magnitudes else 0.0,
    )


def assert_drive_equal(step: int, a: RetinalDrive, b: RetinalDrive) -> None:
    if a.active_photoreceptors != b.active_photoreceptors:
        raise AssertionError(f"step {step}: active photoreceptor count mismatch")
    if a.active_columns != b.active_columns:
        raise AssertionError(f"step {step}: active column count mismatch")
    if f64_bits(a.mean_current) != f64_bits(b.mean_current):
        raise AssertionError(f"step {step}: mean current mismatch {a.mean_current} != {b.mean_current}")
    if f64_bits(a.max_current) != f64_bits(b.max_current):
        raise AssertionError(f"step {step}: max current mismatch {a.max_current} != {b.max_current}")
    if len(a.body_currents) != len(b.body_currents):
        raise AssertionError(f"step {step}: body current length mismatch")
    for index, ((body_a, current_a), (body_b, current_b)) in enumerate(
        zip(a.body_currents, b.body_currents, strict=True)
    ):
        if body_a != body_b or f64_bits(current_a) != f64_bits(current_b):
            raise AssertionError(
                f"step {step} body current {index}: ({body_a},{current_a}) != ({body_b},{current_b})"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--bench-repeats", type=int, default=3)
    args = parser.parse_args()

    mapping = absolute(args.map)
    report = absolute(args.report)
    if not mapping.exists():
        raise SystemExit(f"missing retinotopic map: {mapping}")
    if args.steps < 2 or args.bench_repeats < 1:
        raise SystemExit("--steps must be >=2 and --bench-repeats >=1")

    retina = MaleCNSRetina(mapping)
    n = retina.flybody_ommatidia_per_eye
    rng = np.random.default_rng(20260915)
    sequence: list[dict[str, np.ndarray]] = []
    for step in range(args.steps):
        # Mix stable illumination, smooth changes and clipped bright values so
        # first-sample, adaptation and contrast-clamp paths all execute.
        base = np.float32(0.15 + 0.7 * (step / max(1, args.steps - 1)))
        left = np.clip(base + rng.normal(0.0, 0.18, size=(n, 2)), 0.0, 1.0).astype(np.float32)
        right = np.clip((1.0 - base) + rng.normal(0.0, 0.18, size=(n, 2)), 0.0, 1.0).astype(np.float32)
        sequence.append({"L": left, "R": right})

    reference_adapted = {
        side: np.full(n, np.nan, dtype=np.float64) for side in ("L", "R")
    }
    retina.reset_adaptation()
    for step, eyes in enumerate(sequence):
        expected = scalar_reference(retina, reference_adapted, eyes)
        actual = retina.encode_from_eye_readouts(eyes)
        assert_drive_equal(step, expected, actual)
        for side in ("L", "R"):
            if not np.array_equal(
                reference_adapted[side].view(np.uint64),
                retina._adapted_light[side].view(np.uint64),
            ):
                raise AssertionError(f"step {step}: {side} adaptation state is not bitwise equal")

    duplicate_stats = {}
    for side in ("L", "R"):
        indices = [int(c["ommatidium_index"]) for c in retina._columns_by_side[side]]
        unique = len(set(indices))
        duplicate_stats[side] = {
            "columns": len(indices),
            "unique_ommatidia": unique,
            "duplicate_assignments": len(indices) - unique,
        }

    def time_reference() -> float:
        best = math.inf
        for _ in range(args.bench_repeats):
            adapted = {side: np.full(n, np.nan, dtype=np.float64) for side in ("L", "R")}
            started = time.perf_counter()
            for eyes in sequence:
                scalar_reference(retina, adapted, eyes)
            best = min(best, time.perf_counter() - started)
        return best

    def time_optimized() -> float:
        best = math.inf
        for _ in range(args.bench_repeats):
            retina.reset_adaptation()
            started = time.perf_counter()
            for eyes in sequence:
                retina.encode_from_eye_readouts(eyes)
            best = min(best, time.perf_counter() - started)
        return best

    ref_s = time_reference()
    opt_s = time_optimized()
    speedup = ref_s / opt_s if opt_s > 0 else math.inf

    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MaleCNS retinal hot-path verification",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- overall: PASS",
        f"- synthetic compound-eye steps: {args.steps}",
        f"- ommatidia per eye: {n}",
        "- emitted body currents: bitwise equal on every step",
        "- diagnostics: bitwise equal on every step",
        "- final adaptation state: bitwise equal",
        f"- scalar reference transduction wall time: {ref_s:.6f} s",
        f"- optimized transduction wall time: {opt_s:.6f} s",
        f"- transduction-only speedup: {speedup:.3f}x",
        "",
        "## Mapping multiplicity",
        "",
        "| side | MaleCNS columns | unique FlyBody ommatidia | duplicate assignments |",
        "|---|---:|---:|---:|",
    ]
    for side in ("L", "R"):
        s = duplicate_stats[side]
        lines.append(
            f"| {side} | {s['columns']} | {s['unique_ommatidia']} | {s['duplicate_assignments']} |"
        )
    lines.extend([
        "",
        "Duplicate assignments are intentionally not collapsed. They share the same per-ommatidium adaptation state and are processed in the original MaleCNS column order.",
        "",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"retina_hot_path_verify={report}")
    print("overall=PASS")
    print(f"transduction_speedup={speedup:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
