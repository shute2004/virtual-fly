#!/usr/bin/env python3
"""Replay or follow virtual-fly learning traces as a schematic 3D neural view.

This viewer intentionally does *not* claim anatomical coordinates. Known functional
populations are placed at fixed schematic locations and arbitrary endpoints of the
largest changed synapses are placed deterministically from their body IDs. The
purpose is to make learning dynamics inspectable without slowing the headless
simulation hot path.

Inputs come from a Flyppy experiment directory:

- trajectory.jsonl: body state, sensory stimulation, motor population activity,
  and reinforcement events sampled during the closed loop.
- synapse-snapshots.jsonl: optional low-frequency summaries containing only the
  largest changed synapses plus aggregate plasticity statistics.

Run this while training with --follow, or replay it afterwards. Synapse snapshots
are only available when training was started with --synapse-trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


GROUP_POSITIONS: dict[str, tuple[float, float, float]] = {
    "vision_t4c_left": (-3.2, -2.0, 1.5),
    "vision_t5c_left": (-3.2, -1.1, 0.7),
    "vision_t4d_left": (-3.2, -2.0, -1.5),
    "vision_t5d_left": (-3.2, -1.1, -0.7),
    "vision_t4c_right": (-3.2, 2.0, 1.5),
    "vision_t5c_right": (-3.2, 1.1, 0.7),
    "vision_t4d_right": (-3.2, 2.0, -1.5),
    "vision_t5d_right": (-3.2, 1.1, -0.7),
    "reward_dan": (0.0, -0.8, 2.5),
    "aversive_dan": (0.0, 0.8, -2.5),
    "flight_thrust_left": (3.3, -1.2, 0.0),
    "flight_thrust_right": (3.3, 1.2, 0.0),
}

GROUP_LABELS = {
    "vision_t4c_left": "T4c L",
    "vision_t5c_left": "T5c L",
    "vision_t4d_left": "T4d L",
    "vision_t5d_left": "T5d L",
    "vision_t4c_right": "T4c R",
    "vision_t5c_right": "T5c R",
    "vision_t4d_right": "T4d R",
    "vision_t5d_right": "T5d R",
    "reward_dan": "PAM08",
    "aversive_dan": "PPL1",
    "flight_thrust_left": "DNg02 L",
    "flight_thrust_right": "DNg02 R",
}


class TraceError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v0"),
    )
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="save replay as .gif or .mp4 instead of only opening a window",
    )
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--max-synapses", type=int, default=48)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                # A training process can be between write() and flush() while --follow
                # is watching. Ignore only an incomplete final line.
                if line_number > 1 and file.tell() == path.stat().st_size:
                    break
                raise TraceError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def stable_position(body_id: int) -> np.ndarray:
    digest = hashlib.blake2b(str(int(body_id)).encode("ascii"), digest_size=16).digest()
    values = np.frombuffer(digest[:12], dtype=np.uint32).astype(np.float64)
    unit = (values / np.float64(np.iinfo(np.uint32).max)) * 2.0 - 1.0
    norm = float(np.linalg.norm(unit))
    if norm < 1e-9:
        unit = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    unit /= norm
    radius = 0.8 + (digest[12] / 255.0) * 1.8
    # Brain-like ellipsoid, but still purely schematic.
    return unit * radius * np.array([1.5, 1.0, 0.85])


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def group_activity(frame: dict) -> dict[str, float]:
    visual = frame.get("visual_stimuli") or {}
    activity: dict[str, float] = {}
    for name in GROUP_POSITIONS:
        if name.startswith("vision_"):
            # These are the external currents delivered to the visual populations,
            # not an assertion that every member fired at this exact rate.
            activity[name] = clip01(float(visual.get(name, 0.0)) / 2.0)
    activity["flight_thrust_left"] = clip01(frame.get("motor_left", 0.0))
    activity["flight_thrust_right"] = clip01(frame.get("motor_right", 0.0))
    activity["reward_dan"] = 1.0 if frame.get("reward_stimulated") else 0.0
    activity["aversive_dan"] = 1.0 if frame.get("aversive_stimulated") else 0.0
    return activity


def snapshot_for_episode(snapshots: Iterable[dict], episode: int) -> dict | None:
    selected = None
    for snapshot in snapshots:
        if int(snapshot.get("episode", -1)) <= episode:
            selected = snapshot
        else:
            break
    return selected


def edge_nodes(snapshot: dict | None, max_synapses: int) -> dict[int, np.ndarray]:
    if snapshot is None:
        return {}
    result: dict[int, np.ndarray] = {}
    for edge in (snapshot.get("top_changed") or [])[:max_synapses]:
        for key in ("pre_body_id", "post_body_id"):
            body_id = int(edge[key])
            result.setdefault(body_id, stable_position(body_id))
    return result


def setup_figure():
    fig = plt.figure(figsize=(13.5, 7.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_info = fig.add_subplot(1, 2, 2)
    return fig, ax3d, ax_info


def draw_frame(
    ax3d,
    ax_info,
    frame: dict,
    snapshots: list[dict],
    *,
    max_synapses: int,
) -> None:
    ax3d.cla()
    ax_info.cla()

    episode = int(frame.get("episode", 0))
    step = int(frame.get("control_step", 0))
    activity = group_activity(frame)
    snapshot = snapshot_for_episode(snapshots, episode)
    arbitrary_nodes = edge_nodes(snapshot, max_synapses)

    ax3d.set_title("Neural activity / plasticity (schematic 3D layout)")
    ax3d.set_xlim(-5.0, 5.0)
    ax3d.set_ylim(-4.2, 4.2)
    ax3d.set_zlim(-3.6, 3.6)
    ax3d.set_xlabel("schematic sensory -> motor axis")
    ax3d.set_ylabel("left / right")
    ax3d.set_zlabel("vertical-motion / modulation")
    ax3d.view_init(elev=19, azim=-61)

    # Draw the strongest learned changes first so population nodes sit on top.
    if snapshot is not None:
        top = (snapshot.get("top_changed") or [])[:max_synapses]
        max_delta = max((float(edge.get("abs_delta", 0.0)) for edge in top), default=0.0)
        for edge in top:
            pre = arbitrary_nodes[int(edge["pre_body_id"])]
            post = arbitrary_nodes[int(edge["post_body_id"])]
            magnitude = float(edge.get("abs_delta", 0.0))
            strength = magnitude / max_delta if max_delta > 0.0 else 0.0
            delta = float(edge.get("delta", 0.0))
            color = "tab:orange" if delta >= 0.0 else "tab:cyan"
            xyz = np.vstack((pre, post))
            # Broad translucent stroke + narrow core approximates a glow.
            ax3d.plot(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color=color,
                alpha=0.08 + 0.18 * strength,
                linewidth=3.0 + 5.0 * strength,
            )
            ax3d.plot(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color=color,
                alpha=0.35 + 0.55 * strength,
                linewidth=0.6 + 1.8 * strength,
            )

        if arbitrary_nodes:
            points = np.vstack(list(arbitrary_nodes.values()))
            ax3d.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                s=11,
                alpha=0.28,
                depthshade=True,
            )

    for name, pos in GROUP_POSITIONS.items():
        intensity = activity.get(name, 0.0)
        size = 55.0 + 420.0 * intensity
        if name == "reward_dan":
            color = "tab:green"
        elif name == "aversive_dan":
            color = "tab:red"
        elif name.startswith("vision_"):
            color = "tab:blue"
        else:
            color = "tab:purple"
        ax3d.scatter(*pos, s=size, color=color, alpha=0.25 + 0.75 * intensity)
        ax3d.text(*pos, GROUP_LABELS[name], fontsize=7)

    names = list(GROUP_POSITIONS)
    values = [activity.get(name, 0.0) for name in names]
    short = [GROUP_LABELS[name] for name in names]
    ax_info.barh(np.arange(len(names)), values)
    ax_info.set_yticks(np.arange(len(names)), labels=short)
    ax_info.set_xlim(0.0, 1.0)
    ax_info.set_xlabel("normalized signal")
    ax_info.set_title(f"Episode {episode}  step {step}")
    ax_info.grid(axis="x", alpha=0.2)

    synapse_text = "synapse snapshot: unavailable"
    if snapshot is not None:
        synapse_text = (
            f"changed synapses: {int(snapshot.get('changed_synapses', 0)):,}\n"
            f"mean |Δw|: {float(snapshot.get('mean_abs_delta', 0.0)):.3e}\n"
            f"max |Δw|: {float(snapshot.get('max_abs_delta', 0.0)):.3e}\n"
            f"showing top: {min(max_synapses, len(snapshot.get('top_changed') or []))}"
        )
    event_bits = []
    if frame.get("passed_gate"):
        event_bits.append("gate passed / PAM")
    if frame.get("collision"):
        event_bits.append("collision / PPL1")
    if frame.get("finished"):
        event_bits.append("course finished")
    event_text = ", ".join(event_bits) if event_bits else "none"
    body_text = (
        f"body: x={float(frame.get('x_mm', 0.0)):.2f} mm  "
        f"z={float(frame.get('z_mm', 0.0)):.2f} mm\n"
        f"event: {event_text}\n\n{synapse_text}\n\n"
        "Layout note: node coordinates are schematic, not anatomical.\n"
        "Visual bars are delivered T4/T5 currents; DNg02 bars are spike fractions."
    )
    ax_info.text(
        0.02, -0.18, body_text,
        transform=ax_info.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )


def filtered_frames(records: list[dict], episode: int | None) -> list[dict]:
    if episode is None:
        return records
    return [record for record in records if int(record.get("episode", -1)) == episode]


def save_animation(fig, anim, output: Path, fps: float, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix in {".mp4", ".m4v"}:
        if not animation.writers.is_available("ffmpeg"):
            raise SystemExit("ffmpeg is required to save MP4; save as .gif or install ffmpeg")
        writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
    else:
        raise SystemExit("--save must end in .gif, .mp4, or .m4v")
    anim.save(output, writer=writer, dpi=dpi)
    print(f"saved={output}")


def replay(args: argparse.Namespace) -> int:
    trajectory_path = args.experiment_dir / "trajectory.jsonl"
    synapse_path = args.experiment_dir / "synapse-snapshots.jsonl"
    frames = filtered_frames(load_jsonl(trajectory_path), args.episode)
    if not frames:
        raise SystemExit(f"no trajectory frames found in {trajectory_path}")
    snapshots = load_jsonl(synapse_path)

    fig, ax3d, ax_info = setup_figure()
    interval = 1000.0 / args.fps

    def update(index: int):
        draw_frame(
            ax3d,
            ax_info,
            frames[index],
            snapshots,
            max_synapses=args.max_synapses,
        )
        fig.suptitle("virtual-fly neural replay", fontsize=13)
        return ()

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        repeat=False,
        blit=False,
    )
    if args.save is not None:
        save_animation(fig, anim, args.save, args.fps, args.dpi)
        plt.close(fig)
    else:
        plt.show()
    return 0


def follow(args: argparse.Namespace) -> int:
    if args.save is not None:
        raise SystemExit("--follow and --save cannot be used together")
    trajectory_path = args.experiment_dir / "trajectory.jsonl"
    synapse_path = args.experiment_dir / "synapse-snapshots.jsonl"
    fig, ax3d, ax_info = setup_figure()
    plt.ion()
    plt.show(block=False)
    next_index = 0
    last_count = -1
    interval = 1.0 / args.fps

    while plt.fignum_exists(fig.number):
        records = filtered_frames(load_jsonl(trajectory_path), args.episode)
        snapshots = load_jsonl(synapse_path)
        if len(records) != last_count:
            last_count = len(records)
        if next_index < len(records):
            frame = records[next_index]
            draw_frame(
                ax3d,
                ax_info,
                frame,
                snapshots,
                max_synapses=args.max_synapses,
            )
            fig.suptitle("virtual-fly neural live trace", fontsize=13)
            fig.canvas.draw_idle()
            plt.pause(max(0.001, interval))
            next_index += 1
        else:
            plt.pause(0.20)
            time.sleep(0.02)
    return 0


def main() -> int:
    args = parse_args()
    if args.fps <= 0.0 or args.dpi <= 0 or args.max_synapses < 1:
        raise SystemExit("fps/dpi must be positive and max-synapses must be >= 1")
    if args.follow:
        return follow(args)
    return replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
