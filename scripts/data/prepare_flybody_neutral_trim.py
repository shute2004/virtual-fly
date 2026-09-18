#!/usr/bin/env python3
"""Generate the task-independent neutral-flight wingbeat trim used by body v7.

The coefficients were obtained by minimizing the mean root pitch fluid torque of
FlyBody's published measured hover cycle while preserving its mean translational
fluid wrench.  No Flyppy geometry, reward, CNS state, or learned policy enters
this calibration.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/embodiment"))
from flybody_measured_wingbeat import DEFAULT_PATTERN, MeasuredWingbeatCycle
from virtual_fly.reproducibility import (
    NEUTRAL_TRIM_KIND,
    NEUTRAL_TRIM_SEMANTICS,
    git_provenance,
    sha256_file,
)


OUTPUT_PATTERN = ROOT / "artifacts/embodiment/wing-pattern-neutral-trim-v1.npy"
OUTPUT_METADATA = ROOT / "artifacts/embodiment/wing-pattern-neutral-trim-v1.json"
OFFSETS_RAD = np.asarray(
    [-0.11274463928429765, 0.08681837949639278, -0.10731191762522763],
    dtype=np.float64,
)
AMPLITUDE_SCALES = np.asarray(
    [1.0009058776550996, 1.0670814658837553, 0.9935552950128496],
    dtype=np.float64,
)
PHASE_SHIFTS_RAD = np.asarray(
    [0.03403023064568882, -0.17199874277543603, -0.012566486484526245],
    dtype=np.float64,
)


def main() -> int:
    source = ROOT / DEFAULT_PATTERN
    cycle = MeasuredWingbeatCycle(source)
    trimmed = np.empty_like(cycle.pattern, dtype=np.float64)
    for index in range(cycle.samples):
        phase = 2.0 * math.pi * index / cycle.samples
        for axis in range(3):
            raw = cycle._sample_vector(phase + float(PHASE_SHIFTS_RAD[axis]))
            trimmed[index, axis] = (
                float(cycle.mean[axis])
                + float(AMPLITUDE_SCALES[axis])
                * (float(raw[axis]) - float(cycle.mean[axis]))
                + float(OFFSETS_RAD[axis])
            )

    OUTPUT_PATTERN.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATTERN, trimmed, allow_pickle=False)
    payload = {
        "schema_version": 1,
        "kind": NEUTRAL_TRIM_KIND,
        "runtime_semantics": NEUTRAL_TRIM_SEMANTICS,
        "source_pattern": str(source.relative_to(ROOT)),
        "output_pattern": str(OUTPUT_PATTERN.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(OUTPUT_PATTERN),
        "axes": ["yaw", "roll", "pitch"],
        "offsets_rad": OFFSETS_RAD.tolist(),
        "amplitude_scales": AMPLITUDE_SCALES.tolist(),
        "phase_shifts_rad": PHASE_SHIFTS_RAD.tolist(),
        "provenance": {
            "generator": "scripts/data/prepare_flybody_neutral_trim.py",
            "generator_git": git_provenance(ROOT),
        },
        "calibration": {
            "objective": "preserve mean source translational fluid wrench while minimizing mean root pitch fluid torque and kinematic departure",
            "root_pose": "FlyBody v3 source -47.5 degree flight pose",
            "wingbeat_hz": 218.0,
            "flyppy_state_used": False,
            "cns_state_used": False,
            "reward_used": False,
            "optimizer_result_96_phase_samples": {
                "baseline_force_to_weight": [0.19308673056290085, -1.897842597506781e-17, 1.0251236565499555],
                "trimmed_force_to_weight": [0.19485890385209212, 1.3453920104244854e-16, 1.028016643707966],
                "baseline_pitch_torque": 4.494704487389817,
                "trimmed_pitch_torque": 0.00042849447695325615,
            },
        },
    }
    OUTPUT_METADATA.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"neutral_trim=PASS samples={cycle.samples} pattern={OUTPUT_PATTERN} metadata={OUTPUT_METADATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
