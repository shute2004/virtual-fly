#!/usr/bin/env python3
"""Verify that viewer telemetry becomes quiescent after detaching.

This probe runs the real packed production trainer in a throw-away experiment,
attaches the same ViewerTelemetryLease used by the detached viewer, confirms
body/neural telemetry is advancing, closes the lease, waits for the trainer's
short demand cache to settle, and then verifies that training continues while
body.json/neural.json mtimes remain unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from live_viewer_server import ViewerTelemetryLease

EXPERIMENT = ROOT / "artifacts" / "experiments" / "flyppy-viewer-detach-probe"
REPORT = ROOT / "reports" / "flyppy" / "viewer_detach_quiescence.md"


def line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0


def mtime(path: Path) -> int | None:
    try:
        return int(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return None


def wait_until(predicate, timeout_s: float, *, interval_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return bool(predicate())


def stop_process_group(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=8.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def tail(path: Path, lines: int = 50) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<missing>"
    return "\n".join(content[-lines:])


def main() -> int:
    snapshot = ROOT / "artifacts" / "malecns-v1.0"
    calibration = ROOT / "artifacts" / "embodiment" / "neural-runtime-calibration-v1.json"
    groups = snapshot / "embodiment-groups-v0.json"
    retinotopic = snapshot / "retinotopic-vision-v1.json"
    wing_motor = snapshot / "wing-motor-neurons-v0.json"
    body_motor = snapshot / "body-motor-neurons-v0.json"
    viewer_graph = ROOT / "artifacts" / "embodiment" / "neural-viewer-graph-v1.json"

    required = [
        snapshot / "manifest.json",
        calibration,
        groups,
        retinotopic,
        wing_motor,
        body_motor,
        viewer_graph,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing detach-probe inputs:\n  " + "\n  ".join(missing))

    import shutil

    shutil.rmtree(EXPERIMENT, ignore_errors=True)
    EXPERIMENT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    exp_rel = EXPERIMENT.relative_to(ROOT)
    trajectory = EXPERIMENT / "trajectory.jsonl"
    live = EXPERIMENT / "live"
    body_path = live / "body.json"
    neural_path = live / "neural.json"
    trainer_log = ROOT / "artifacts" / "viewer-detach-probe-trainer.log"

    synapse_scale = float(json.loads(calibration.read_text(encoding="utf-8"))["synapse_scale"])
    env = os.environ.copy()
    env["VF_NEURAL_SYNAPSE_SCALE"] = str(synapse_scale)
    env["VF_FLYPPY_VISION_MODE"] = "direct-ray"
    env["VF_FLYPPY_OMMATIDIA_RAYS"] = "13"

    trainer: subprocess.Popen | None = None
    lease: ViewerTelemetryLease | None = None
    checks: dict[str, bool] = {}
    observations: dict[str, int | None] = {}
    exception_text = "none"

    try:
        with trainer_log.open("w", encoding="utf-8") as trainer_out:
            trainer = subprocess.Popen(
                [
                    "uv", "run", "python",
                    "scripts/embodiment/train_flyppy_population_packed.py",
                    "--episodes", "10000",
                    "--population", "4",
                    "--snapshot", str(snapshot),
                    "--groups", str(groups),
                    "--retinotopic-map", str(retinotopic),
                    "--wing-motor-map", str(wing_motor),
                    "--body-motor-map", str(body_motor),
                    "--viewer-graph", str(viewer_graph),
                    "--output-dir", str(exp_rel),
                    "--trajectory-stride", "1",
                    "--checkpoint-every", "10000",
                    "--fresh",
                ],
                cwd=ROOT,
                env=env,
                stdout=trainer_out,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

            checks["trainer_warmed"] = wait_until(
                lambda: line_count(trajectory) >= 80 or trainer.poll() is not None,
                240.0,
            ) and line_count(trajectory) >= 80
            if not checks["trainer_warmed"]:
                raise RuntimeError("trainer ended before warmup")

            checks["headless_no_body_json"] = not body_path.exists()
            checks["headless_no_neural_json"] = not neural_path.exists()

            lease = ViewerTelemetryLease(EXPERIMENT, interval_s=0.05)
            lease.start()
            checks["lease_connected"] = wait_until(lambda: lease.connected, 5.0)
            checks["telemetry_materialized"] = wait_until(
                lambda: body_path.exists() and neural_path.exists(),
                30.0,
            )

            first_body = mtime(body_path)
            first_neural = mtime(neural_path)
            checks["body_advanced_while_attached"] = wait_until(
                lambda: mtime(body_path) not in (None, first_body),
                20.0,
            )
            checks["neural_advanced_while_attached"] = wait_until(
                lambda: mtime(neural_path) not in (None, first_neural),
                20.0,
            )

            lease.close()
            lease = None

            # The trainer intentionally caches viewer demand for ~50 ms. Allow
            # any final in-flight publication to complete, then define that
            # settled mtime as the quiescent baseline.
            time.sleep(0.35)
            settled_body = mtime(body_path)
            settled_neural = mtime(neural_path)
            detached_start_records = line_count(trajectory)

            checks["training_advanced_after_detach"] = wait_until(
                lambda: line_count(trajectory) >= detached_start_records + 120
                or trainer.poll() is not None,
                120.0,
            ) and line_count(trajectory) >= detached_start_records + 120

            final_body = mtime(body_path)
            final_neural = mtime(neural_path)
            checks["body_quiescent_after_settle"] = final_body == settled_body
            checks["neural_quiescent_after_settle"] = final_neural == settled_neural
            checks["trainer_still_running"] = trainer.poll() is None

            observations.update(
                {
                    "settled_body_mtime_ns": settled_body,
                    "final_body_mtime_ns": final_body,
                    "settled_neural_mtime_ns": settled_neural,
                    "final_neural_mtime_ns": final_neural,
                    "detached_start_records": detached_start_records,
                    "final_records": line_count(trajectory),
                }
            )
    except Exception as exc:
        exception_text = f"{type(exc).__name__}: {exc}"
        checks["probe_exception"] = False
    finally:
        if lease is not None:
            lease.close()
        stop_process_group(trainer)

    passed = bool(checks) and all(checks.values())
    diagnosis = "VIEWER_DETACH_QUIESCENCE_OK" if passed else "VIEWER_DETACH_QUIESCENCE_FAILED"

    lines = [
        "# Flyppy viewer detach quiescence probe",
        "",
        f"- diagnosis: **{diagnosis}**",
        "- trainer: production `train_flyppy_population_packed.py`",
        "- population: 4",
        "- vision: direct-ray K=13",
        "- viewer demand: real `ViewerTelemetryLease` Unix stream",
        f"- exception: `{exception_text}`",
        "",
        "## Checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")

    lines.extend(["", "## Observations", ""])
    for name, value in observations.items():
        lines.append(f"- {name}: `{value}`")

    lines.extend(
        [
            "",
            "## Trainer log tail",
            "",
            "```text",
            tail(trainer_log),
            "```",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"diagnosis={diagnosis}")
    for name, value in checks.items():
        print(f"{name}={str(value).lower()}")
    print(f"report={REPORT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
