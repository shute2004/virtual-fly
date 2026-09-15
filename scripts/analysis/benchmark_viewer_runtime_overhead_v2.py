#!/usr/bin/env python3
"""Measure detached viewer overhead with a constant four-fly production population.

Unlike the original benchmark, this probe deliberately gives the trainer far more
episodes than the measurement can consume, so all four population slots remain
replenishable throughout warmup, headless-before, viewer-attached and
headless-after phases.  Throughput is measured from trajectory records with
--trajectory-stride=1, therefore each record corresponds to one production
control step.

The benchmark uses a throw-away experiment and interrupts the trainer after the
three measured phases; it never touches the normal flyppy-v3 checkpoint/history.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "artifacts" / "experiments" / "flyppy-viewer-overhead-v2"
REPORT = ROOT / "reports" / "flyppy" / "viewer_runtime_overhead_v2.md"

WARMUP_RECORDS = 120
PHASE_RECORDS = 180
EPISODE_BUDGET = 10_000


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def terminate(proc: subprocess.Popen | None, *, group: bool = False) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if group and hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGINT)
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=8.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if proc.poll() is None:
        try:
            if group and hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=5.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                try:
                    if group and hasattr(os, "killpg"):
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass


def line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0


def wait_for_records(
    trainer: subprocess.Popen,
    trajectory: Path,
    target: int,
    *,
    timeout_s: float = 180.0,
) -> tuple[bool, float, int]:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_s
    current = line_count(trajectory)
    while current < target and time.monotonic() < deadline:
        if trainer.poll() is not None:
            return False, time.perf_counter() - started, current
        time.sleep(0.02)
        current = line_count(trajectory)
    return current >= target, time.perf_counter() - started, current


def measure_phase(
    trainer: subprocess.Popen,
    trajectory: Path,
    *,
    records: int,
) -> tuple[float | None, int, int, float]:
    start_count = line_count(trajectory)
    ok, elapsed, end_count = wait_for_records(
        trainer,
        trajectory,
        start_count + records,
    )
    advanced = max(0, end_count - start_count)
    if not ok or elapsed <= 0.0:
        return None, start_count, end_count, elapsed
    return advanced / elapsed, start_count, end_count, elapsed


def read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def fetch_json(url: str) -> dict | None:
    try:
        with urlopen(url, timeout=0.4) as response:  # noqa: S310 - loopback only
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError, TimeoutError):
        return None
    return payload if isinstance(payload, dict) else None


def mtime(path: Path) -> int | None:
    try:
        return int(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return None


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
        raise SystemExit("missing benchmark inputs:\n  " + "\n  ".join(missing))

    synapse_scale = float(json.loads(calibration.read_text(encoding="utf-8"))["synapse_scale"])
    shutil.rmtree(EXPERIMENT, ignore_errors=True)
    EXPERIMENT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    exp_rel = EXPERIMENT.relative_to(ROOT)
    trajectory = EXPERIMENT / "trajectory.jsonl"
    live = EXPERIMENT / "live"
    trainer_log = ROOT / "artifacts" / "viewer-overhead-v2-trainer.log"
    server_log = ROOT / "artifacts" / "viewer-overhead-v2-server.log"
    body_log = ROOT / "artifacts" / "viewer-overhead-v2-body.log"

    env = os.environ.copy()
    env["VF_NEURAL_SYNAPSE_SCALE"] = str(synapse_scale)
    env["VF_FLYPPY_VISION_MODE"] = "direct-ray"
    env["VF_FLYPPY_OMMATIDIA_RAYS"] = "13"

    trainer: subprocess.Popen | None = None
    server: subprocess.Popen | None = None
    body: subprocess.Popen | None = None
    checks: dict[str, bool] = {}
    phases: dict[str, tuple[float | None, int, int, float]] = {}
    exception_text = "none"

    try:
        with trainer_log.open("w", encoding="utf-8") as trainer_out:
            trainer = subprocess.Popen(
                [
                    "uv", "run", "python",
                    "scripts/embodiment/train_flyppy_population_packed.py",
                    "--episodes", str(EPISODE_BUDGET),
                    "--population", "4",
                    "--snapshot", str(snapshot),
                    "--groups", str(groups),
                    "--retinotopic-map", str(retinotopic),
                    "--wing-motor-map", str(wing_motor),
                    "--body-motor-map", str(body_motor),
                    "--viewer-graph", str(viewer_graph),
                    "--output-dir", str(exp_rel),
                    "--trajectory-stride", "1",
                    "--checkpoint-every", str(EPISODE_BUDGET),
                    "--fresh",
                ],
                cwd=ROOT,
                env=env,
                stdout=trainer_out,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

            warmed, _, warm_count = wait_for_records(
                trainer, trajectory, WARMUP_RECORDS, timeout_s=240.0
            )
            checks["trainer_warmed"] = warmed
            if not warmed:
                raise RuntimeError("trainer ended before warmup completed")

            checks["headless_no_body_json"] = not (live / "body.json").exists()
            checks["headless_no_neural_json"] = not (live / "neural.json").exists()
            phases["headless_before"] = measure_phase(
                trainer, trajectory, records=PHASE_RECORDS
            )
            checks["headless_before_completed"] = phases["headless_before"][0] is not None

            port = free_port()
            health_url = f"http://127.0.0.1:{port}/api/viewer-health"
            with server_log.open("w", encoding="utf-8") as server_out, body_log.open(
                "w", encoding="utf-8"
            ) as body_out:
                server = subprocess.Popen(
                    [
                        "uv", "run", "python",
                        "scripts/embodiment/live_viewer_server.py",
                        "--port", str(port),
                        "--bind", "127.0.0.1",
                        "--root", str(ROOT),
                        "--experiment", str(exp_rel),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=server_out,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                body_launcher = ["uv", "run", "python"]
                if platform.system() == "Darwin":
                    body_launcher = ["uv", "run", "mjpython"]
                body = subprocess.Popen(
                    body_launcher
                    + [
                        "scripts/embodiment/live_body_viewer.py",
                        "--experiment", str(exp_rel),
                        "--environment-version", "v3",
                        "--poll-hz", "30",
                        "--width", "960",
                        "--height", "540",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=body_out,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                deadline = time.monotonic() + 30.0
                connected = False
                while time.monotonic() < deadline:
                    health = fetch_json(health_url)
                    connected = bool(health and health.get("telemetry_stream_connected"))
                    if connected and (live / "body.json").exists() and (live / "neural.json").exists():
                        break
                    if trainer.poll() is not None or server.poll() is not None or body.poll() is not None:
                        break
                    time.sleep(0.05)

                checks["viewer_connected"] = connected
                checks["viewer_body_json"] = (live / "body.json").exists()
                checks["viewer_neural_json"] = (live / "neural.json").exists()
                checks["body_renderer_alive"] = body.poll() is None
                checks["viewer_server_alive"] = server.poll() is None

                phases["viewer_attached"] = measure_phase(
                    trainer, trajectory, records=PHASE_RECORDS
                )
                checks["viewer_phase_completed"] = phases["viewer_attached"][0] is not None

                body_mtime_before_detach = mtime(live / "body.json")
                neural_mtime_before_detach = mtime(live / "neural.json")
                terminate(body)
                terminate(server)
                body = None
                server = None

            # Let the trainer observe stream closure and expire its 50 ms cache.
            time.sleep(0.30)
            detached_start_count = line_count(trajectory)
            phases["headless_after"] = measure_phase(
                trainer, trajectory, records=PHASE_RECORDS
            )
            checks["headless_after_completed"] = phases["headless_after"][0] is not None
            checks["training_advanced_after_detach"] = line_count(trajectory) > detached_start_count
            checks["body_json_stopped_after_detach"] = mtime(live / "body.json") == body_mtime_before_detach
            checks["neural_json_stopped_after_detach"] = mtime(live / "neural.json") == neural_mtime_before_detach
            checks["trainer_still_running_after_measurement"] = trainer.poll() is None

            # The very large episode budget is intentional: reaching it during
            # the short benchmark would mean population replenishment was not
            # guaranteed for all phases.
            checks["episode_budget_not_exhausted"] = trainer.poll() is None
            terminate(trainer, group=True)
            trainer = None

    except Exception as exc:
        exception_text = f"{type(exc).__name__}: {exc}"
        checks["benchmark_exception"] = False
    finally:
        terminate(body)
        terminate(server)
        terminate(trainer, group=True)

    before = phases.get("headless_before", (None, 0, 0, 0.0))[0]
    attached = phases.get("viewer_attached", (None, 0, 0, 0.0))[0]
    after = phases.get("headless_after", (None, 0, 0, 0.0))[0]
    attached_ratio = attached / before if before and attached else None
    recovered_ratio = after / before if before and after else None

    passed = all(checks.values()) and all(
        phases.get(name, (None, 0, 0, 0.0))[0] is not None
        for name in ("headless_before", "viewer_attached", "headless_after")
    )
    diagnosis = "VIEWER_RUNTIME_OVERHEAD_V2_MEASURED" if passed else "VIEWER_RUNTIME_OVERHEAD_V2_FAILED"

    lines = [
        "# Flyppy viewer runtime overhead benchmark v2",
        "",
        f"- diagnosis: **{diagnosis}**",
        "- trainer: production `train_flyppy_population_packed.py`",
        "- population: 4, replenished throughout all measured phases",
        "- vision: direct-ray K=13",
        "- measurement: trajectory records/s with `--trajectory-stride=1` (one record per production control step)",
        f"- warmup records: {WARMUP_RECORDS}",
        f"- records per phase: {PHASE_RECORDS}",
        f"- episode budget: {EPISODE_BUDGET} (intentionally not completed)",
        "- attached phase: Unix stream lease + 960x540 detached MuJoCo renderer at 30 Hz",
        "- browser/Three.js rendering: excluded",
        f"- exception: `{exception_text}`",
        "",
        "## Throughput",
        "",
        f"- headless before attach: {before:.3f} control steps/s" if before is not None else "- headless before attach: unavailable",
        f"- viewer attached: {attached:.3f} control steps/s" if attached is not None else "- viewer attached: unavailable",
        f"- attached/headless-before: {attached_ratio:.3f}x" if attached_ratio is not None else "- attached/headless-before: unavailable",
        f"- headless after detach: {after:.3f} control steps/s" if after is not None else "- headless after detach: unavailable",
        f"- after/before: {recovered_ratio:.3f}x" if recovered_ratio is not None else "- after/before: unavailable",
        "",
        "The before/after ratio is reported as an observation, not a correctness threshold; learning state evolves during the run. Correct detach behavior is established separately by telemetry mtimes stopping while training continues.",
        "",
        "## State checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")

    for name in ("headless_before", "viewer_attached", "headless_after"):
        rate, start_count, end_count, elapsed = phases.get(name, (None, 0, 0, 0.0))
        lines.extend(
            [
                "",
                f"### {name}",
                f"- start trajectory records: {start_count}",
                f"- end trajectory records: {end_count}",
                f"- elapsed: {elapsed:.3f}s",
                f"- rate: {rate:.3f} control steps/s" if rate is not None else "- rate: unavailable",
            ]
        )

    lines.extend(
        [
            "",
            "## Trainer log tail",
            "```text",
            tail(trainer_log),
            "```",
            "",
            "## Viewer server log tail",
            "```text",
            tail(server_log),
            "```",
            "",
            "## Body renderer log tail",
            "```text",
            tail(body_log),
            "```",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"diagnosis={diagnosis}")
    if before is not None:
        print(f"headless_before_steps_per_second={before:.3f}")
    if attached is not None:
        print(f"viewer_attached_steps_per_second={attached:.3f}")
    if after is not None:
        print(f"headless_after_steps_per_second={after:.3f}")
    if attached_ratio is not None:
        print(f"attached_over_headless_before={attached_ratio:.3f}")
    if recovered_ratio is not None:
        print(f"headless_after_over_before={recovered_ratio:.3f}")
    for name, value in checks.items():
        print(f"{name}={str(value).lower()}")
    print(f"report={REPORT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
