#!/usr/bin/env python3
"""Exercise the real packed Flyppy trainer -> detached viewer pipeline end to end.

This probe deliberately uses the production packed trainer, shared MaleCNS bridge,
viewer lease server and detached MuJoCo body renderer. It writes into a dedicated
throw-away experiment so the production checkpoint/curriculum/history are not
modified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "flyppy" / "viewer_production_e2e.md"
EXPERIMENT = ROOT / "artifacts" / "experiments" / "flyppy-viewer-e2e-probe"


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


def tail(path: Path, lines: int = 80) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<missing>"
    return "\n".join(data[-lines:])


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
        raise SystemExit("missing viewer E2E inputs:\n  " + "\n  ".join(missing))

    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    synapse_scale = float(calibration_payload["synapse_scale"])

    shutil.rmtree(EXPERIMENT, ignore_errors=True)
    EXPERIMENT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    port = free_port()
    exp_rel = EXPERIMENT.relative_to(ROOT)
    server_log = ROOT / "artifacts" / "viewer-e2e-server.log"
    body_log = ROOT / "artifacts" / "viewer-e2e-body.log"
    trainer_log = ROOT / "artifacts" / "viewer-e2e-trainer.log"

    env = os.environ.copy()
    env["VF_NEURAL_SYNAPSE_SCALE"] = str(synapse_scale)
    env["VF_FLYPPY_VISION_MODE"] = "direct-ray"
    env["VF_FLYPPY_OMMATIDIA_RAYS"] = "13"

    server: subprocess.Popen | None = None
    body: subprocess.Popen | None = None
    trainer: subprocess.Popen | None = None

    checks: dict[str, bool] = {}
    body_keys: set[tuple[int, int]] = set()
    frame_mtimes: set[int] = set()
    stream_connected_seen = False
    health_samples = 0
    diagnosis = "UNKNOWN"

    try:
        with server_log.open("w", encoding="utf-8") as server_out, body_log.open(
            "w", encoding="utf-8"
        ) as body_out, trainer_log.open("w", encoding="utf-8") as trainer_out:
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

            deadline = time.monotonic() + 15.0
            health_url = f"http://127.0.0.1:{port}/api/viewer-health"
            while time.monotonic() < deadline and fetch_json(health_url) is None:
                if server.poll() is not None:
                    break
                time.sleep(0.05)
            checks["viewer_server_started"] = fetch_json(health_url) is not None

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
                    "15",
                    "--width",
                    "320",
                    "--height",
                    "180",
                ],
                cwd=ROOT,
                env=env,
                stdout=body_out,
                stderr=subprocess.STDOUT,
                text=True,
            )

            time.sleep(0.8)
            checks["body_renderer_started"] = body.poll() is None

            trainer = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/embodiment/train_flyppy_population_packed.py",
                    "--episodes",
                    "2",
                    "--population",
                    "1",
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
                    "--max-control-steps",
                    "40",
                    "--checkpoint-every",
                    "1",
                    "--telemetry-stride",
                    "1",
                    "--body-telemetry-stride",
                    "1",
                    "--fresh",
                ],
                cwd=ROOT,
                env=env,
                stdout=trainer_out,
                stderr=subprocess.STDOUT,
                text=True,
            )

            live = EXPERIMENT / "live"
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                health = fetch_json(health_url)
                if health is not None:
                    health_samples += 1
                    stream_connected_seen = stream_connected_seen or bool(
                        health.get("telemetry_stream_connected", False)
                    )

                body_payload = read_json(live / "body.json")
                if body_payload is not None:
                    try:
                        body_keys.add(
                            (
                                int(body_payload["episode"]),
                                int(body_payload["control_step"]),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        pass

                try:
                    frame_mtimes.add((live / "fly.png").stat().st_mtime_ns)
                except FileNotFoundError:
                    pass

                if trainer.poll() is not None:
                    # Give the detached renderer a moment to consume the final
                    # body snapshot before evaluating the result.
                    time.sleep(0.5)
                    try:
                        frame_mtimes.add((live / "fly.png").stat().st_mtime_ns)
                    except FileNotFoundError:
                        pass
                    break

                if server.poll() is not None or body.poll() is not None:
                    break
                time.sleep(0.05)

            if trainer.poll() is None:
                terminate(trainer)
            trainer_rc = trainer.returncode

            checks.update(
                {
                    "trainer_completed": trainer_rc == 0,
                    "viewer_stream_connected": stream_connected_seen,
                    "body_json_seen": (EXPERIMENT / "live" / "body.json").exists(),
                    "body_control_step_advanced": len(body_keys) >= 2,
                    "neural_json_seen": (EXPERIMENT / "live" / "neural.json").exists(),
                    "status_json_seen": (EXPERIMENT / "live" / "status.json").exists(),
                    "fly_png_seen": (EXPERIMENT / "live" / "fly.png").exists(),
                    "fly_png_updated": len(frame_mtimes) >= 2,
                    "viewer_server_survived": server.poll() is None,
                    "body_renderer_survived": body.poll() is None,
                }
            )

            exp_url = "/" + exp_rel.as_posix() + "/live/"
            for name in ("status.json", "body.json", "neural.json", "fly.png"):
                try:
                    with urlopen(  # noqa: S310 - loopback only
                        f"http://127.0.0.1:{port}{exp_url}{name}", timeout=1.0
                    ) as response:
                        checks[f"http_{name}"] = response.status == 200
                except OSError:
                    checks[f"http_{name}"] = False

            if all(checks.values()):
                diagnosis = "LIVE_VIEWER_PRODUCTION_E2E_OK"
            elif not checks.get("viewer_stream_connected", False):
                diagnosis = "VIEWER_TO_TRAINER_DEMAND_PATH_FAILED"
            elif not checks.get("body_json_seen", False):
                diagnosis = "TRAINER_TELEMETRY_BODY_PATH_FAILED"
            elif not checks.get("fly_png_seen", False):
                diagnosis = "DETACHED_MUJOCO_RENDER_PATH_FAILED"
            elif not checks.get("fly_png_updated", False):
                diagnosis = "DETACHED_MUJOCO_FRAME_NOT_ADVANCING"
            else:
                diagnosis = "VIEWER_E2E_PARTIAL_FAILURE"
    finally:
        terminate(trainer)
        terminate(body)
        terminate(server)

    lines = [
        "# Flyppy production viewer E2E probe",
        "",
        f"- diagnosis: **{diagnosis}**",
        f"- experiment: `{EXPERIMENT}`",
        "- production trainer: `train_flyppy_population_packed.py`",
        "- vision: `direct-ray`, K=13",
        f"- health samples: {health_samples}",
        f"- distinct body keys observed: {len(body_keys)}",
        f"- distinct fly.png mtimes observed: {len(frame_mtimes)}",
        "",
        "## Checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")

    lines.extend(
        [
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
    return 0 if diagnosis == "LIVE_VIEWER_PRODUCTION_E2E_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
