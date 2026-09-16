#!/usr/bin/env python3
"""Compare motor output from two frozen Flyppy fixed-evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from virtual_fly.training.fixed_evaluation import (
    compare_motor_outputs,
    render_motor_comparison_markdown,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = Path("reports/flyppy/fixed_evaluation_initial_malecns.json")
DEFAULT_TRAINED = Path("reports/flyppy/evaluations/history/fixed_evaluation_episode255.json")
DEFAULT_JSON = Path("reports/flyppy/diagnostics/fixed_evaluation_motor_comparison.json")
DEFAULT_REPORT = Path("reports/flyppy/diagnostics/fixed_evaluation_motor_comparison.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--trained", type=Path, default=DEFAULT_TRAINED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    baseline_path = absolute(args.baseline)
    trained_path = absolute(args.trained)
    output_json = absolute(args.output_json)
    report = absolute(args.report)

    payload = compare_motor_outputs(load_json(baseline_path), load_json(trained_path))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_motor_comparison_markdown(payload), encoding="utf-8")

    print(f"baseline={baseline_path}")
    print(f"trained={trained_path}")
    for row in payload["condition_comparison"]:
        motor = row["motor"]
        wing = motor["mean_wing_spikes_per_step"]
        somatic = motor["mean_somatic_spikes_per_step"]
        steering = motor["mean_active_steering_channels"]
        print(
            "condition={} first_gate={}->{} wing={:.4f}->{:.4f} somatic={:.4f}->{:.4f} steering={:.3f}->{:.3f}".format(
                row["condition"],
                row["baseline_first_gate_passes"],
                row["trained_first_gate_passes"],
                float(wing["baseline"]),
                float(wing["trained"]),
                float(somatic["baseline"]),
                float(somatic["trained"]),
                float(steering["baseline"]),
                float(steering["trained"]),
            )
        )
    print(f"json={output_json}")
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
