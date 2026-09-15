#!/usr/bin/env python3
"""Diagnose a running Flyppy trainer/viewer telemetry session.

Run this while training is active. It checks that the trainer control socket
exists, sends the same viewer heartbeat as live_viewer_server.py, and observes
whether status/body/neural/fly outputs appear and advance. This does not alter
training state beyond requesting the same read-only observer telemetry as the
viewer itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from live_telemetry import viewer_control_socket_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v3"),
    )
    parser.add_argument("--duration", type=float, default=3.0)
    return parser.parse_args()


def send_watch(target: Path) -> tuple[bool, str]:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.sendto(b"watch", str(target))
        return True, "ok"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def read_key(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return "-"
    episode = payload.get("episode", "-")
    step = payload.get("control_step", "-")
    neural_step = payload.get("neural_step")
    return f"{episode}:{step}" + (f":{neural_step}" if neural_step is not None else "")


def stamp(path: Path) -> tuple[bool, int, int]:
    try:
        stat = path.stat()
        return True, stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return False, -1, -1


def main() -> int:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")

    experiment = args.experiment
    if not experiment.is_absolute():
        experiment = (ROOT / experiment).resolve()
    else:
        experiment = experiment.resolve()
    control = viewer_control_socket_path(experiment)
    live = experiment / "live"
    paths = {
        "status": live / "status.json",
        "body": live / "body.json",
        "neural": live / "neural.json",
        "frame": live / "fly.png",
    }

    print(f"experiment={experiment}")
    print(f"control_socket={control}")
    print(f"control_socket_exists={str(control.exists()).lower()}")

    before = {name: stamp(path) for name, path in paths.items()}
    started = time.monotonic()
    sends = 0
    send_failures: list[str] = []
    while time.monotonic() - started < args.duration:
        ok, detail = send_watch(control)
        if ok:
            sends += 1
        else:
            send_failures.append(detail)
        time.sleep(0.2)
    after = {name: stamp(path) for name, path in paths.items()}

    print(f"watch_packets_sent={sends}")
    if send_failures:
        print(f"watch_send_error={send_failures[-1]}")

    for name, path in paths.items():
        existed_before, mtime_before, size_before = before[name]
        existed_after, mtime_after, size_after = after[name]
        changed = existed_after and (not existed_before or mtime_after != mtime_before)
        print(
            f"{name}_exists={str(existed_after).lower()} "
            f"{name}_changed={str(changed).lower()} "
            f"{name}_size={size_after} path={path}"
        )

    print(f"status_key={read_key(paths['status'])}")
    print(f"body_key={read_key(paths['body'])}")
    print(f"neural_key={read_key(paths['neural'])}")

    socket_ok = control.exists() and sends > 0
    telemetry_ok = after["body"][0] and after["neural"][0]
    telemetry_live = (
        (after["body"][1] != before["body"][1])
        or (after["neural"][1] != before["neural"][1])
    )

    if not control.exists():
        diagnosis = "TRAINER_CONTROL_SOCKET_MISSING"
    elif sends == 0:
        diagnosis = "VIEWER_HEARTBEAT_SEND_FAILED"
    elif not telemetry_ok:
        diagnosis = "TRAINER_NOT_PUBLISHING_ON_VIEWER_DEMAND"
    elif not telemetry_live:
        diagnosis = "TELEMETRY_PRESENT_BUT_NOT_ADVANCING"
    elif not after["frame"][0]:
        diagnosis = "TELEMETRY_LIVE_BODY_RENDERER_NOT_WRITING_FRAME"
    elif after["frame"][1] == before["frame"][1]:
        diagnosis = "BODY_FRAME_PRESENT_BUT_NOT_ADVANCING"
    else:
        diagnosis = "LIVE_VIEWER_PIPELINE_OK"

    print(f"socket_ok={str(socket_ok).lower()}")
    print(f"telemetry_ok={str(telemetry_ok).lower()}")
    print(f"telemetry_live={str(telemetry_live).lower()}")
    print(f"diagnosis={diagnosis}")
    return 0 if diagnosis == "LIVE_VIEWER_PIPELINE_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
