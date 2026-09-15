#!/usr/bin/env python3
"""Measure detached viewer overhead inside one production Flyppy training run.

The benchmark keeps one packed production trainer alive and measures three
consecutive phases from trajectory growth:

1. headless (no viewer process),
2. detached viewer backend attached (lease + body renderer),
3. viewer detached again.

It uses a throw-away experiment and never touches the normal flyppy-v3
checkpoint/history. The browser/Three.js UI is intentionally excluded from the
throughput measurement; the measured attached phase includes the trainer-side
viewer reads/snapshots plus the detached MuJoCo renderer, which are the pieces
that can contend with training on the same machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "artifacts" / "experiments" / "flyppy-viewer-overhead-benchmark"
REPORT = ROOT / "reports" / "flyppy" / "viewer_runtime_overhead.md"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


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


def line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0


def wait_for_count(
    proc: subprocess.Popen,
    path: Path,
    target: int,
    *,
    timeout_s: float = 120.0,
) -> tuple[bool, float, int]:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_s
    current = line_count(path)
    while current < target and time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, time.perf_counter() - started, current
        time.sleep(0.02)
        current = line_count(path)
    return current >= target, time.perf_counter() - started, current


def stat_mtime(path: Path) -> int | None:
    try:
        return int(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return None


def tail(path: Path, lines: int = 60) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<missing>"
    return "\n".join(content[-lines:])


def phase_rate(
    trainer: subprocess.Popen,
    trajectory: Path,
    *,
    records: int,
) -> tuple[float | None, int, int]:
    start_count = line_count(trajectory)
    ok, elapsed, end_count = wait_for_count(
        trainer,
        trajectory,
        start_count + records,
    )
    advanced = max(0, end_count - start_count)
    if not ok or elapsed <= 0.0:
        return None, start_count, end_count
    return advanced / elapsed, start_count, end_count


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
        raise SystemExit("missing viewer benchmark inputs:\n  " + "\n  ".join(missing))

    synapse_scale = float(json.loads(calibration.read_text(encoding="utf-8"))["synapse_scale"])
    shutil.rmtree(EXPERIMENT, ignore_errors=True)
    EXPERIMENT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    exp_rel = EXPERIMENT.relative_to(ROOT)
    trajectory = EXPERIMENT / "trajectory.jsonl"
    live = EXPERIMENT / "live"
    trainer_log = ROOT / "artifacts" / "viewer-overhead-trainer.log"
    server_log = ROOT / "artifacts" / "viewer-overhead-server.log"
    body_log = ROOT / "artifacts" / "viewer-overhead-body.log"

    env = os.environ.copy()
    env["VF_NEURAL_SYNAPSE_SCALE"] = str(synapse_scale)
    env["VF_FLYPPY_VISION_MODE"] = "direct-ray"
    env["VF_FLYPPY_OMMATIDIA_RAYS"] = "13"

    trainer: subprocess.Popen | None = None
    server: subprocess.Popen | None = None
    body: subprocess.Popen | None = None

    checks: dict[str, bool] = {}
    rates: dict[str, float | None] = {
        "headless": None,
        "viewer_attached": None,
        "viewer_detached_again": None,
    }

    try:
        with trainer_log.open("w", encoding="utf-8") as trainer_out:
            trainer = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/embodiment/train_flyppy_population_packed.py",
                    "--episodes",
                    "40",
                    "--population",
                    "4",
                    "--snapshot",
                    str(snapshot),
                    "--groups",
                    str(groups),
                    "--retinotopic-map",
                    str(retinotopic),
                    "--wing-motor-map",
                    str(wing_motor),
                    "--body-motor-map",
                    str(body_motor),
                    "--viewer-graph",
                    str(viewer_graph),
                    "--output-dir",
                    str(exp_rel),
                    "--trajectory-stride",
                    "1",
                    "--checkpoint-every",
                    "40",
                    "--fresh",
                ],
                cwd=ROOT,
                env=env,
                stdout=trainer_out,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Ignore startup/JIT effects before the first measured phase.
            warmed, _, _ = wait_for_count(trainer, trajectory, 40, timeout_s=180.0)
            checks["trainer_warmed"] = warmed
            if not warmed:
                raise RuntimeError("trainer ended before benchmark warmup completed")

            checks["headless_no_body_json"] = not (live / "body.json").exists()
            checks["headless_no_neural_json"] = not (live / "neural.json").exists()
            rates["headless"], _, _ = phase_rate(trainer, trajectory, records=60)
            checks["headless_phase_completed"] = rates["headless"] is not None

            port = free_port()
            health_url = f"http://127.0.0.1:{port}/api/viewer-health"
            with server_log.open("w", encoding="utf-8") as server_out, body_log.open(
                "w", encoding="utf-8"
            ) as body_out:
                server = subprocess.Popen(
                    [
                        "uv",
                        "run",
                        "python",
                        "scripts/embodiment/live_viewer_server.py",
                        "--port",
                        str(port),
                        "--bind",
                        "127.0.0.1",
                        "--root",
                        str(ROOT),
                        "--experiment",
                        str(exp_rel),
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
                        "--experiment",
                        str(exp_rel),
                        "--environment-version",
                        "v3",
                        "--poll-hz",
                        "30",
                        "--width",
                        "960",
                        "--height",
                        "540",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=body_out,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                deadline = time.monotonic() + 20.0
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

                rates["viewer_attached"], _, _ = phase_rate(trainer, trajectory, records=60)
                checks["viewer_phase_completed"] = rates["viewer_attached"] is not None

                body_mtime_before_detach = stat_mtime(live / "body.json")
                neural_mtime_before_detach = stat_mtime(live / "neural.json")
                terminate(body)
                terminate(server)
                body = None
                server = None

            # Allow the trainer's 50 ms demand cache to observe the disconnect.
            time.sleep(0.25)
            detached_start = line_count(trajectory)
            rates["viewer_detached_again"], _, _ = phase_rate(
                trainer,
                trajectory,
                records=60,
            )
            checks["detached_phase_completed"] = rates["viewer_detached_again"] is not None
            checks["training_advanced_after_detach"] = line_count(trajectory) > detached_start
            checks["body_json_stopped_after_detach"] = (
                stat_mtime(live / "body.json") == body_mtime_before_detach
            )
            checks["neural_json_stopped_after_detach"] = (
                stat_mtime(live / "neural.json") == neural_mtime_before_detach
            )

            try:
                trainer.wait(timeout=180.0)
            except subprocess.TimeoutExpired:
                checks["trainer_completed"] = False
            else:
                checks["trainer_completed"] = trainer.returncode == 0
    except Exception as exc:
        checks["benchmark_exception"] = False
        exception_text = f"{type(exc).__name__}: {exc}"
    else:
        exception_text = "none"
    finally:
        terminate(body)
        terminate(server)
        terminate(trainer)

    headless = rates["headless"]
    attached = rates["viewer_attached"]
    detached = rates["viewer_detached_again"]
    attached_ratio = attached / headless if headless and attached else None
    detached_ratio = detached / headless if headless and detached else None

    passed = all(checks.values()) and all(value is not None for value in rates.values())
    diagnosis = "VIEWER_RUNTIME_OVERHEAD_MEASURED" if passed else "VIEWER_RUNTIME_OVERHEAD_BENCHMARK_FAILED"

    lines = [
        "# Flyppy viewer runtime overhead benchmark",
        "",
        f"- diagnosis: **{diagnosis}**",
        "- trainer: production `train_flyppy_population_packed.py`",
        "- population: 4",
        "- vision: direct-ray K=13",
        "- measurement: trajectory records/s (aggregate control-step samples)",
        "- attached phase: Unix stream lease + 960x540 detached MuJoCo renderer at 30 Hz",
        "- browser/Three.js rendering: excluded from throughput measurement",
        f"- exception: `{exception_text}`",
        "",
        "## Throughput",
        "",
        f"- headless: {headless:.3f} records/s" if headless is not None else "- headless: unavailable",
        f"- viewer attached: {attached:.3f} records/s" if attached is not None else "- viewer attached: unavailable",
        (
            f"- attached/headless: {attached_ratio:.3f}x"
            if attached_ratio is not None
            else "- attached/headless: unavailable"
        ),
        (
            f"- viewer detached again: {detached:.3f} records/s"
            if detached is not None
            else "- viewer detached again: unavailable"
        ),
        (
            f"- detached/headless: {detached_ratio:.3f}x"
            if detached_ratio is not None
            else "- detached/headless: unavailable"
        ),
        "",
        "## State checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")

    lines.extend(
        [
            "",
            "## Trainer log tail",
            "",
            "```text",
            tail(trainer_log),
            "```",
            "",
            "## Viewer server log tail",
            "",
            "```text",
            tail(server_log),
            "```",
            "",
            "## Body renderer log tail",
            "",
            "```text",
            tail(body_log),
            "```",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"diagnosis={diagnosis}")
    if headless is not None:
        print(f"headless_records_per_second={headless:.3f}")
    if attached is not None:
        print(f"viewer_attached_records_per_second={attached:.3f}")
    if detached is not None:
        print(f"viewer_detached_again_records_per_second={detached:.3f}")
    if attached_ratio is not None:
        print(f"attached_over_headless={attached_ratio:.3f}")
    if detached_ratio is not None:
        print(f"detached_over_headless={detached_ratio:.3f}")
    for name, value in checks.items():
        print(f"{name}={str(value).lower()}")
    print(f"report={REPORT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
