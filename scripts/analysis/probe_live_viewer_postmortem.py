#!/usr/bin/env python3
"""Postmortem diagnosis for a completed Flyppy detached-viewer session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/flyppy/live_viewer_postmortem.md"),
    )
    parser.add_argument(
        "--viewer-log",
        type=Path,
        default=Path("/tmp/virtual-fly-live-viewer-8765.log"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def describe(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"exists": False, "size": 0, "mtime": "-"}
    stamp = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return {"exists": True, "size": stat.st_size, "mtime": stamp}


def key(payload: dict | None) -> str:
    if not payload:
        return "-"
    episode = payload.get("episode", "-")
    step = payload.get("control_step", "-")
    neural_step = payload.get("neural_step")
    return f"{episode}:{step}" + (f":{neural_step}" if neural_step is not None else "")


def main() -> int:
    args = parse_args()
    experiment = args.experiment if args.experiment.is_absolute() else ROOT / args.experiment
    experiment = experiment.resolve()
    live = experiment / "live"
    summary_path = experiment / "summary.json"

    paths = {
        "camera": live / "camera.json",
        "status": live / "status.json",
        "body": live / "body.json",
        "neural": live / "neural.json",
        "frame": live / "fly.png",
    }
    meta = {name: describe(path) for name, path in paths.items()}
    status = load_json(paths["status"])
    body = load_json(paths["body"])
    neural = load_json(paths["neural"])
    summary = load_json(summary_path) or {}

    camera_seen = bool(meta["camera"]["exists"])
    body_seen = bool(meta["body"]["exists"])
    neural_seen = bool(meta["neural"]["exists"])
    frame_seen = bool(meta["frame"]["exists"])
    status_seen = bool(meta["status"]["exists"])

    if camera_seen and not body_seen and not neural_seen:
        diagnosis = "VIEWER_HTTP_ACTIVE_BUT_TRAINER_TELEMETRY_NEVER_MATERIALIZED"
    elif (body_seen or neural_seen) and not frame_seen:
        diagnosis = "TRAINER_TELEMETRY_EXISTED_BUT_BODY_RENDERER_NEVER_WROTE_FRAME"
    elif frame_seen and not status_seen:
        diagnosis = "FRAME_EXISTED_BUT_STATUS_PATH_MISSING"
    elif frame_seen and body_seen and neural_seen:
        diagnosis = "TELEMETRY_AND_FRAME_EXISTED_BROWSER_OR_TIMING_PATH_SUSPECT"
    elif not camera_seen and not body_seen and not neural_seen and not frame_seen:
        diagnosis = "NO_VIEWER_SESSION_ARTIFACTS_FOUND"
    else:
        diagnosis = "PARTIAL_VIEWER_ARTIFACTS_REQUIRE_TARGETED_FOLLOWUP"

    log_tail = "(viewer log not found)"
    if args.viewer_log.exists():
        try:
            lines = args.viewer_log.read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(lines[-80:]) or "(viewer log empty)"
        except OSError as exc:
            log_tail = f"(viewer log read failed: {exc})"

    lines = [
        "# Flyppy live viewer postmortem",
        "",
        f"- experiment: `{experiment}`",
        f"- diagnosis: **{diagnosis}**",
        f"- run episode range: {summary.get('episode_start', '-')}..{summary.get('episode_end', '-')}",
        f"- population: {summary.get('population', '-')}",
        f"- vision runtime: `{summary.get('vision_runtime', '-')}`",
        f"- rays/ommatidium: {summary.get('vision_rays_per_ommatidium', '-')}",
        f"- viewer telemetry mode: `{summary.get('viewer_telemetry', '-')}`",
        "",
        "## Residual live artifacts",
        "",
        "| artifact | exists | bytes | modified UTC | logical key |",
        "|---|---|---:|---|---|",
    ]
    payloads = {"status": status, "body": body, "neural": neural}
    for name in ("camera", "status", "body", "neural", "frame"):
        item = meta[name]
        logical_key = key(payloads.get(name)) if name in payloads else "-"
        lines.append(
            f"| {name} | {str(bool(item['exists'])).lower()} | {item['size']} | {item['mtime']} | {logical_key} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `camera.json` exists only if the browser successfully reached the viewer server's camera POST endpoint.",
        "- `body.json` / `neural.json` require the trainer to observe an active viewer lease and enter its telemetry path.",
        "- `fly.png` requires `live_body_viewer.py` to consume `body.json` and render a frame.",
        "- The trainer control socket is expected to be absent after training exits, so its current absence is not treated as a failure.",
        "",
        "## Viewer server log tail",
        "",
        "```text",
        log_tail,
        "```",
        "",
    ])

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"viewer_postmortem={output}")
    print(f"diagnosis={diagnosis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
