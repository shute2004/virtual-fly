#!/usr/bin/env python3
"""Read-only probe for the Flyppy v3 shared-weight population resume path.

The probe reproduces the production trainer's initialization path through:

1. argument/input validation,
2. two FlyBody slot construction,
3. population neural bridge startup,
4. loading the existing production checkpoint.

It does not open trajectory/commit logs for writing, does not save a checkpoint,
and does not modify curriculum/population state. A compact report is written
under reports/flyppy/ for remote review.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from flyppy_course import FlyppyCourse  # noqa: E402
from population_neural_bridge_client import PopulationNeuralBridgeClient  # noqa: E402
from train_flyppy_population import make_slot  # noqa: E402


DEFAULT_EXPERIMENT = Path("artifacts/experiments/flyppy-v3")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_REPORT = Path("reports/flyppy/population_resume_probe.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object in {path}, got {type(payload).__name__}")
    return payload


def record_stage(stages: list[dict[str, Any]], name: str, fn) -> Any:
    started = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:
        stages.append(
            {
                "stage": name,
                "ok": False,
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    stages.append(
        {
            "stage": name,
            "ok": True,
            "seconds": time.perf_counter() - started,
            "error": None,
        }
    )
    return value


def markdown(stages: list[dict[str, Any]], details: dict[str, Any], error: str | None) -> str:
    lines = [
        "# Flyppy population resume probe",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- mutates training artifacts: false",
        f"- overall: {'FAIL' if error else 'PASS'}",
        "",
        "## Stages",
        "",
        "| stage | ok | seconds | error |",
        "|---|---|---:|---|",
    ]
    for item in stages:
        err = str(item.get("error") or "-").replace("|", "\\|")
        lines.append(
            f"| {item['stage']} | {'yes' if item['ok'] else 'no'} | "
            f"{float(item['seconds']):.3f} | {err} |"
        )
    lines.extend(
        [
            "",
            "## Details",
            "",
            "```json",
            json.dumps(details, indent=2, sort_keys=True),
            "```",
        ]
    )
    if error:
        lines.extend(["", "## Failure", "", "```text", error.rstrip(), "```"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    experiment = absolute(args.experiment)
    snapshot = absolute(args.snapshot)
    report = absolute(args.report)
    checkpoint = experiment / "checkpoint"
    groups = snapshot / "embodiment-groups-v0.json"
    curriculum_path = experiment / "curriculum-state.json"
    population_state_path = experiment / "population-state.json"

    stages: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "experiment": str(experiment),
        "snapshot": str(snapshot),
        "population": args.population,
    }
    failure: str | None = None

    try:
        def validate_inputs() -> None:
            if args.population not in (1, 2):
                raise RuntimeError("probe supports population=1 or 2")
            required = [
                snapshot / "manifest.json",
                groups,
                checkpoint / "manifest.json",
                curriculum_path,
            ]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError("missing inputs: " + ", ".join(missing))

        record_stage(stages, "validate_inputs", validate_inputs)

        checkpoint_manifest = record_stage(
            stages, "read_checkpoint_manifest", lambda: read_json(checkpoint / "manifest.json")
        )
        curriculum = record_stage(
            stages, "read_curriculum_state", lambda: read_json(curriculum_path)
        )
        if population_state_path.exists():
            population_state = record_stage(
                stages, "read_population_state", lambda: read_json(population_state_path)
            )
        else:
            population_state = {}
            stages.append(
                {
                    "stage": "read_population_state",
                    "ok": True,
                    "seconds": 0.0,
                    "error": None,
                }
            )

        details["checkpoint_step"] = checkpoint_manifest.get("step")
        details["curriculum_episodes"] = curriculum.get("curriculum_episodes")
        initial_global_version = int(population_state.get("global_weight_version", 0))
        details["initial_global_weight_version"] = initial_global_version

        # Reuse the trainer's actual slot-construction path, but do not begin an episode.
        class SlotArgs:
            gate_count = 6
            seed = 0
            wing_motor_map = snapshot / "wing-motor-neurons-v0.json"
            body_motor_map = snapshot / "body-motor-neurons-v0.json"
            retinotopic_map = snapshot / "retinotopic-vision-v1.json"
            photoreceptor_current_gain = 2.0

        slot_args = SlotArgs()
        slots = record_stage(
            stages,
            "construct_flybody_slots",
            lambda: [make_slot(slot_args, index) for index in range(args.population)],
        )
        details["constructed_slots"] = len(slots)

        with ExitStack() as stack:
            brain = record_stage(
                stages,
                "start_population_bridge",
                lambda: stack.enter_context(
                    PopulationNeuralBridgeClient(
                        snapshot=snapshot,
                        groups=groups,
                        slots=args.population,
                    )
                ),
            )
            pong = record_stage(stages, "bridge_ping", brain.ping)
            details["bridge_backend"] = brain.ready.get("backend")
            details["bridge_ping"] = pong
            loaded = record_stage(
                stages,
                "load_production_checkpoint",
                lambda: brain.load_checkpoint(
                    checkpoint,
                    global_weight_version=initial_global_version,
                ),
            )
            details["loaded_checkpoint"] = loaded
            details["bridge_global_weight_version"] = brain.global_weight_version

    except Exception:
        failure = traceback.format_exc()

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown(stages, details, failure), encoding="utf-8")
    print(f"population_resume_probe_report={report}")
    print(f"population_resume_probe={'FAIL' if failure else 'PASS'}")
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
