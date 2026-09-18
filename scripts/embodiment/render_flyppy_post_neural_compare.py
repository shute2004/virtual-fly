#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

ACTIVE_COLOR = np.array([0x56, 0xD6, 0xFF], dtype=np.float64) / 255.0
REWARD_COLOR = np.array([0x69, 0xF0, 0xA5], dtype=np.float64) / 255.0
AVERSIVE_COLOR = np.array([0xFF, 0x6F, 0x7D], dtype=np.float64) / 255.0
EDGE_IDLE = np.array([0x18, 0x21, 0x2D], dtype=np.float64) / 255.0
EDGE_ACTIVE = np.array([0x55, 0xD7, 0xFF], dtype=np.float64) / 255.0
MAX_PULSES = 320
MAX_NEW_PULSES = 96


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Flyppy Before/After with the live MaleCNS viewer visual logic.")
    p.add_argument("--before-playback", type=Path, required=True)
    p.add_argument("--after-playback", type=Path, required=True)
    # Retained for command-line compatibility and provenance metadata.
    p.add_argument("--before-checkpoint", type=Path, required=True)
    p.add_argument("--after-checkpoint", type=Path, required=True)
    p.add_argument("--before-frames", type=Path, required=True)
    p.add_argument("--after-frames", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    p.add_argument("--viewer-graph", type=Path, default=Path("artifacts/embodiment/neural-viewer-graph-v1.json"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/flyppy-post-final-video"))
    p.add_argument("--top-changed", type=int, default=400, help=argparse.SUPPRESS)
    p.add_argument("--neural-width", type=int, default=480)
    p.add_argument("--fps", type=float, default=60.0)
    return p.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("/System/Library/Fonts/SFNS.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ):
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def category_color(node: dict) -> np.ndarray:
    nt = int(node.get("nt") or 0)
    if nt == 4:
        value = 0xB08CFF
    elif nt == 7:
        value = 0x7FA9FF
    elif str(node.get("region") or "") == "optic_lobe":
        value = 0x879FB8
    elif str(node.get("region") or "") == "ventral_nerve_cord":
        value = 0x9F9CA7
    else:
        value = 0x93A4B8
    return np.array([(value >> 16) & 255, (value >> 8) & 255, value & 255], dtype=np.float64) / 255.0


def perspective_projector(graph: dict, width: int, height: int):
    """Mirror live-neural-viewer's fitBrainCameraToGraph + THREE perspective view."""
    anatomy = np.asarray(graph.get("anatomy_points") or [], dtype=np.float64)
    node_points = np.asarray([node["position"] for node in graph["nodes"]], dtype=np.float64)
    fit_points = anatomy if len(anatomy) else node_points
    mins = fit_points.min(axis=0)
    maxs = fit_points.max(axis=0)
    center = (mins + maxs) * 0.5
    span = float(np.max(maxs - mins))
    fov_deg = 48.0
    distance = max(18.0, span / (2.0 * math.tan(math.radians(fov_deg * 0.5))) * 1.18)
    camera = np.array([center[0], center[1] + span * 0.10, center[2] + distance], dtype=np.float64)
    target = center
    forward = target - camera
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    focal = (height * 0.5) / math.tan(math.radians(fov_deg * 0.5))

    def project(point) -> tuple[float, float, float] | None:
        rel = np.asarray(point, dtype=np.float64) - camera
        depth = float(np.dot(rel, forward))
        if depth <= 1e-6:
            return None
        x = float(np.dot(rel, right))
        y = float(np.dot(rel, up))
        px = width * 0.5 + focal * x / depth
        py = height * 0.5 - focal * y / depth
        return px, py, depth

    return project


class NeuralViewerRenderer:
    """Offline counterpart of live-neural-viewer updateActivity/updateNeuralVisuals."""

    def __init__(self, graph: dict, width: int, height: int, fps: float):
        self.graph = graph
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1.0 / fps
        self.nodes = graph["nodes"]
        self.node_index = {int(node["body_id"]): i for i, node in enumerate(self.nodes)}
        self.project = perspective_projector(graph, width, height)
        self.node_proj = [self.project(node["position"]) for node in self.nodes]
        self.base_colors = np.asarray([category_color(node) for node in self.nodes], dtype=np.float64)
        self.node_energy = np.zeros(len(self.nodes), dtype=np.float64)
        self.node_event_energy = np.zeros(len(self.nodes), dtype=np.float64)
        self.event_tint = REWARD_COLOR.copy()
        self.dopamine_mask = np.asarray([int(node.get("nt") or 0) == 4 for node in self.nodes], dtype=bool)

        raw_edges = graph["edges"]
        counts = np.asarray([max(1, int(edge.get("synapse_count", 1))) for edge in raw_edges], dtype=np.float64)
        max_log = max(1e-9, float(np.log1p(counts).max()))
        self.edges = []
        self.edges_by_pre: dict[int, list[int]] = {}
        for edge_i, (edge, count) in enumerate(zip(raw_edges, counts)):
            pre = int(edge["pre"])
            post = int(edge["post"])
            a = self.node_index.get(pre)
            b = self.node_index.get(post)
            if a is None or b is None or self.node_proj[a] is None or self.node_proj[b] is None:
                continue
            record = {
                "pre": pre,
                "post": post,
                "a": a,
                "b": b,
                "strength": float(math.log1p(count) / max_log),
                "pre_nt": int(self.nodes[a].get("nt") or 0),
            }
            idx = len(self.edges)
            self.edges.append(record)
            self.edges_by_pre.setdefault(pre, []).append(idx)
        self.edge_energy = np.zeros(len(self.edges), dtype=np.float64)
        self.pulses: list[dict] = []

        self.shell = Image.new("RGB", (width, height), (8, 11, 16))
        shell_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        shell_draw = ImageDraw.Draw(shell_overlay, "RGBA")
        # Same released CNS-coordinate backdrop as live viewer.
        for point in graph.get("anatomy_points") or []:
            p = self.project(point)
            if p is None:
                continue
            x, y, _ = p
            if -1 <= x < width + 1 and -1 <= y < height + 1:
                shell_draw.point((round(x), round(y)), fill=(100, 119, 137, 70))
        self.shell = Image.alpha_composite(self.shell.convert("RGBA"), shell_overlay).convert("RGB")

    def _spawn_activity(self, frame: dict, frame_index: int) -> None:
        active_ids = {int(v) for v in (frame.get("active_neural_body_ids") or [])}
        for body_id in active_ids:
            i = self.node_index.get(body_id)
            if i is not None:
                self.node_energy[i] = 1.0

        reward = bool(frame.get("reward"))
        aversive = bool(frame.get("aversive"))
        if reward or aversive:
            self.event_tint = REWARD_COLOR.copy() if reward else AVERSIVE_COLOR.copy()
            self.node_event_energy[self.dopamine_mask] = 1.0

        candidates: list[tuple[float, int]] = []
        for pre in active_ids:
            for edge_i in self.edges_by_pre.get(pre, ()):
                edge = self.edges[edge_i]
                post_active = edge["post"] in active_ids
                score = edge["strength"] * (1.25 if post_active else 1.0)
                self.edge_energy[edge_i] = max(
                    self.edge_energy[edge_i], min(1.0, 0.32 + 0.78 * score)
                )
                candidates.append((score, edge_i))
        candidates.sort(reverse=True)
        available = max(0, MAX_PULSES - len(self.pulses))
        for rank, (score, edge_i) in enumerate(candidates[: min(available, MAX_NEW_PULSES)]):
            edge = self.edges[edge_i]
            event_driven = edge["pre_nt"] == 4 and (reward or aversive)
            color = (
                REWARD_COLOR.copy()
                if event_driven and reward
                else AVERSIVE_COLOR.copy()
                if event_driven and aversive
                else ACTIVE_COLOR.copy()
            )
            duration_s = (220.0 + (1.0 - edge["strength"]) * 420.0) / 1000.0
            # Deterministic substitute for the viewer's tiny random phase jitter.
            phase = ((edge_i * 0.754877666 + frame_index * 0.037 + rank * 0.011) % 1.0) * 0.12
            self.pulses.append(
                {
                    "edge": edge_i,
                    "age": 0.0,
                    "duration": duration_s,
                    "phase": phase,
                    "color": color,
                }
            )

    def render(self, frame: dict, frame_index: int) -> Image.Image:
        node_decay = math.exp(-self.dt / 0.32)
        event_decay = math.exp(-self.dt / 0.48)
        edge_decay = math.exp(-self.dt / 0.42)
        self.node_energy *= node_decay
        self.node_event_energy *= event_decay
        self.edge_energy *= edge_decay
        self._spawn_activity(frame, frame_index)

        image = self.shell.copy().convert("RGBA")
        edge_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        edge_draw = ImageDraw.Draw(edge_overlay, "RGBA")
        for i, edge in enumerate(self.edges):
            pa = self.node_proj[edge["a"]]
            pb = self.node_proj[edge["b"]]
            if pa is None or pb is None:
                continue
            strength = edge["strength"]
            active = self.edge_energy[i]
            idle_scale = 0.17 + 0.25 * strength
            color = np.clip(EDGE_IDLE * idle_scale + EDGE_ACTIVE * active, 0.0, 1.0)
            rgba = tuple(int(round(v * 255)) for v in color) + (148,)
            edge_draw.line((pa[0], pa[1], pb[0], pb[1]), fill=rgba, width=1)
        image = Image.alpha_composite(image, edge_overlay)

        node_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        node_draw = ImageDraw.Draw(node_overlay, "RGBA")
        for i, projected in enumerate(self.node_proj):
            if projected is None:
                continue
            x, y, _ = projected
            if x < -8 or y < -8 or x > self.width + 8 or y > self.height + 8:
                continue
            energy = self.node_energy[i]
            event_energy = self.node_event_energy[i]
            base = self.base_colors[i]
            color = np.clip(
                base * (0.52 + 0.20 * energy)
                + ACTIVE_COLOR * 0.82 * energy
                + self.event_tint * 0.72 * event_energy,
                0.0,
                1.0,
            )
            rgb = tuple(int(round(v * 255)) for v in color)
            node_draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=rgb + (225,))
            glow = min(1.0, energy * 0.95 + event_energy * 0.85)
            if glow > 0.03:
                glow_color = np.clip(
                    ACTIVE_COLOR * energy + self.event_tint * event_energy, 0.0, 1.0
                )
                grgb = tuple(int(round(v * 255)) for v in glow_color)
                radius = 3.0 + 2.0 * glow
                node_draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    outline=grgb + (int(150 * glow),),
                    width=1,
                )
        image = Image.alpha_composite(image, node_overlay)

        pulse_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        pulse_draw = ImageDraw.Draw(pulse_overlay, "RGBA")
        next_pulses = []
        for pulse in self.pulses:
            pulse["age"] += self.dt
            progress = pulse["age"] / pulse["duration"] + pulse["phase"]
            if progress >= 1.0:
                continue
            t = max(0.0, min(1.0, progress))
            edge = self.edges[pulse["edge"]]
            pa = self.node_proj[edge["a"]]
            pb = self.node_proj[edge["b"]]
            if pa is None or pb is None:
                continue
            x = pa[0] + (pb[0] - pa[0]) * t
            y = pa[1] + (pb[1] - pa[1]) * t
            envelope = math.sin(math.pi * t)
            color = np.clip(pulse["color"] * envelope, 0.0, 1.0)
            rgb = tuple(int(round(v * 255)) for v in color)
            radius = 1.4 + 1.2 * envelope
            pulse_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=rgb + (235,),
            )
            next_pulses.append(pulse)
        self.pulses = next_pulses[:MAX_PULSES]
        return Image.alpha_composite(image, pulse_overlay).convert("RGB")


def label_body(image: Image.Image, label: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.text(
        (24, 22),
        label,
        font=font(31),
        fill=(250, 251, 253),
        stroke_width=2,
        stroke_fill=(18, 20, 24),
    )
    return out


def frame_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("frame-*.jpg"))


def main() -> int:
    args = parse_args()
    before_playback = read_json(absolute(args.before_playback))
    after_playback = read_json(absolute(args.after_playback))
    graph = read_json(absolute(args.viewer_graph))
    before_frames_dir = absolute(args.before_frames)
    after_frames_dir = absolute(args.after_frames)
    output = absolute(args.output_dir)
    output_frames = output / "frames"
    if output_frames.exists():
        shutil.rmtree(output_frames)
    output_frames.mkdir(parents=True)

    neural_width = int(args.neural_width)
    if neural_width < 320:
        raise ValueError("--neural-width must be >= 320")
    body_width = 960
    row_height = 540
    before_neural = NeuralViewerRenderer(graph, neural_width, row_height, float(args.fps))
    after_neural = NeuralViewerRenderer(graph, neural_width, row_height, float(args.fps))

    before_images = frame_files(before_frames_dir)
    after_images = frame_files(after_frames_dir)
    if len(before_images) != len(before_playback["frames"]):
        raise RuntimeError(f"before frame mismatch {len(before_images)} != {len(before_playback['frames'])}")
    if len(after_images) != len(after_playback["frames"]):
        raise RuntimeError(f"after frame mismatch {len(after_images)} != {len(after_playback['frames'])}")
    total = max(len(before_images), len(after_images))

    # Render each real playback once, then hold its final rendered state after it ends.
    before_neural_frames: list[Image.Image] = []
    after_neural_frames: list[Image.Image] = []
    for i, frame in enumerate(before_playback["frames"]):
        before_neural_frames.append(before_neural.render(frame, i))
    for i, frame in enumerate(after_playback["frames"]):
        after_neural_frames.append(after_neural.render(frame, i))

    for i in range(total):
        bi = min(i, len(before_images) - 1)
        ai = min(i, len(after_images) - 1)
        body_before = label_body(Image.open(before_images[bi]).convert("RGB"), "Before")
        body_after = label_body(Image.open(after_images[ai]).convert("RGB"), "After")
        neural_before = before_neural_frames[bi]
        neural_after = after_neural_frames[ai]
        canvas = Image.new("RGB", (body_width + neural_width, row_height * 2), (0, 0, 0))
        canvas.paste(body_before, (0, 0))
        canvas.paste(neural_before, (body_width, 0))
        canvas.paste(body_after, (0, row_height))
        canvas.paste(neural_after, (body_width, row_height))
        canvas.save(output_frames / f"frame-{i:05d}.jpg", quality=95, subsampling=0)
        if i % 120 == 0:
            print(f"rendered {i}/{total}", flush=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(float(args.fps)),
            "-i",
            str(output_frames / "frame-%05d.jpg"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "before-after-neural.mp4"),
        ],
        check=True,
    )

    metadata = {
        "schema_version": 2,
        "visualization_mode": "live-neural-viewer-emulation",
        "before_global_weight_version": int(before_playback["global_weight_version"]),
        "after_global_weight_version": int(after_playback["global_weight_version"]),
        "before_passed_gates": int(before_playback["passed_gates"]),
        "after_passed_gates": int(after_playback["passed_gates"]),
        "gate2_center_z_mm": before_playback.get("gate2_center_z_mm"),
        "evaluation_plasticity": False,
        "viewer_nodes": len(graph["nodes"]),
        "viewer_edges": len(graph["edges"]),
        "frame_count": total,
        "fps": float(args.fps),
        "video_width": body_width + neural_width,
        "video_height": row_height * 2,
        "body_width": body_width,
        "neural_width": neural_width,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))
    print(f"video={output / 'before-after-neural.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
