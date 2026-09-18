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


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render posting-oriented Flyppy before/after video with MaleCNS activity and learned synapse changes.")
    p.add_argument("--before-playback", type=Path, required=True)
    p.add_argument("--after-playback", type=Path, required=True)
    p.add_argument("--before-checkpoint", type=Path, required=True)
    p.add_argument("--after-checkpoint", type=Path, required=True)
    p.add_argument("--before-frames", type=Path, required=True)
    p.add_argument("--after-frames", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    p.add_argument("--viewer-graph", type=Path, default=Path("artifacts/embodiment/neural-viewer-graph-v1.json"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/flyppy-post-final-video"))
    p.add_argument("--top-changed", type=int, default=400)
    p.add_argument("--neural-width", type=int, default=480)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--jpeg-quality", type=int, default=3)
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


def checkpoint_weights(checkpoint: Path) -> np.memmap:
    manifest = read_json(checkpoint / "manifest.json")
    return np.memmap(checkpoint / manifest["weights_file"], dtype="<f4", mode="r")


def viewer_edge_deltas(snapshot: Path, graph: dict, before_checkpoint: Path, after_checkpoint: Path):
    manifest = read_json(snapshot / "manifest.json")
    body_ids = np.memmap(snapshot / manifest["body_ids_file"], dtype="<u8", mode="r")
    row_offsets = np.memmap(snapshot / manifest["row_offsets_file"], dtype="<u4", mode="r")
    pre_indices = np.memmap(snapshot / manifest["pre_indices_file"], dtype="<u4", mode="r")
    body_index = {int(body_id): i for i, body_id in enumerate(body_ids)}
    before = checkpoint_weights(before_checkpoint)
    after = checkpoint_weights(after_checkpoint)
    rows = []
    for edge in graph["edges"]:
        pre_body = int(edge["pre"])
        post_body = int(edge["post"])
        pre_index = body_index[pre_body]
        post_index = body_index[post_body]
        start = int(row_offsets[post_index])
        end = int(row_offsets[post_index + 1])
        local = np.flatnonzero(np.asarray(pre_indices[start:end]) == pre_index)
        if len(local) != 1:
            continue
        edge_index = start + int(local[0])
        delta = float(after[edge_index] - before[edge_index])
        rows.append({
            "pre": pre_body,
            "post": post_body,
            "delta": delta,
            "abs_delta": abs(delta),
            "before": float(before[edge_index]),
            "after": float(after[edge_index]),
            "synapse_count": int(edge.get("synapse_count", 1)),
        })
    rows.sort(key=lambda row: row["abs_delta"], reverse=True)
    return rows


def make_projection(graph: dict, width: int, height: int):
    positions = {int(node["body_id"]): tuple(float(x) for x in node["position"]) for node in graph["nodes"]}
    points = np.array(list(positions.values()), dtype=np.float64)
    xmin, ymin = points[:, 0].min(), points[:, 1].min()
    xmax, ymax = points[:, 0].max(), points[:, 1].max()
    left, right = 85.0, width - 85.0
    top, bottom = 90.0, height - 55.0
    x_span = max(1e-9, xmax - xmin)
    y_span = max(1e-9, ymax - ymin)
    available_w = right - left
    available_h = bottom - top
    # Preserve the MaleCNS viewer's anatomical aspect ratio.  Using separate
    # x/y scales makes the CNS look artificially stretched to fill the panel.
    scale = min(available_w / x_span, available_h / y_span)
    draw_w = x_span * scale
    draw_h = y_span * scale
    origin_x = left + (available_w - draw_w) * 0.5
    origin_y = top + (available_h - draw_h) * 0.5

    def project(position):
        x, y, _ = position
        px = origin_x + (x - xmin) * scale
        py = origin_y + draw_h - (y - ymin) * scale
        return int(round(px)), int(round(py))

    return positions, {body_id: project(position) for body_id, position in positions.items()}, project


def static_neural_panel(graph: dict, projected: dict[int, tuple[int, int]], changed_edges: list[dict], *, after: bool, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), (8, 10, 14))
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Viewer subgraph topology.
    for edge in graph["edges"]:
        a = projected.get(int(edge["pre"])); b = projected.get(int(edge["post"]))
        if a is not None and b is not None:
            draw.line((a, b), fill=(125, 135, 150, 22), width=1)
    for node in graph["nodes"]:
        p = projected[int(node["body_id"])]
        draw.ellipse((p[0]-1, p[1]-1, p[0]+1, p[1]+1), fill=(168, 176, 190, 95))

    if after and changed_edges:
        max_delta = max(edge["abs_delta"] for edge in changed_edges)
        for edge in reversed(changed_edges):
            a = projected.get(edge["pre"]); b = projected.get(edge["post"])
            if a is None or b is None:
                continue
            strength = math.sqrt(edge["abs_delta"] / max(1e-12, max_delta))
            alpha = int(70 + 175 * strength)
            line_width = 1 + int(2.5 * strength)
            color = (73, 221, 145, alpha) if edge["delta"] > 0 else (244, 145, 72, alpha)
            draw.line((a, b), fill=color, width=line_width)

    panel = Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(panel)
    compact = width < 700
    title = font(24 if compact else 27)
    small = font(17 if compact else 19)
    tiny = font(14 if compact else 16)
    d.text((20 if compact else 24, 18), "MaleCNS activity" if compact else "MaleCNS neural activity", font=title, fill=(238, 242, 247))
    d.text((20 if compact else 24, 50), "2,448 neurons / 5,868 edges" if compact else "2,448-neuron viewer subset / 5,868 connections", font=tiny, fill=(158, 168, 182))
    if after:
        if compact:
            d.text((20, 72), "synaptic Δw  v240 → v966", font=small, fill=(220, 226, 234))
            d.line((22, 103, 52, 103), fill=(73, 221, 145), width=3)
            d.text((60, 94), "strengthened", font=tiny, fill=(174, 184, 196))
            d.line((178, 103, 208, 103), fill=(244, 145, 72), width=3)
            d.text((216, 94), "weakened", font=tiny, fill=(174, 184, 196))
        else:
            d.text((width-355, 20), "learned synaptic Δw: v240 → v966", font=small, fill=(220, 226, 234))
            d.line((width-340, 52, width-305, 52), fill=(73, 221, 145), width=3)
            d.text((width-295, 41), "strengthened", font=tiny, fill=(174, 184, 196))
            d.line((width-178, 52, width-143, 52), fill=(244, 145, 72), width=3)
            d.text((width-133, 41), "weakened", font=tiny, fill=(174, 184, 196))
    else:
        d.text((20 if compact else width-250, 72 if compact else 20), "baseline connectivity", font=small, fill=(185, 194, 207))
    d.text((20 if compact else 24, height-30), "bright = active · plasticity OFF" if compact else "bright nodes = active neurons    evaluation plasticity = OFF", font=tiny, fill=(163, 174, 188))
    return panel


def active_panel(base: Image.Image, frame: dict, projected: dict[int, tuple[int, int]]) -> Image.Image:
    image = base.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    active = frame.get("active_neural_body_ids") or []
    for body_id in active:
        p = projected.get(int(body_id))
        if p is None:
            continue
        draw.ellipse((p[0]-3, p[1]-3, p[0]+3, p[1]+3), fill=(255, 245, 180, 245))
        draw.ellipse((p[0]-6, p[1]-6, p[0]+6, p[1]+6), outline=(255, 245, 180, 85), width=1)
    active_x = max(20, image.width - (145 if image.width < 700 else 190))
    draw.text((active_x, image.height-30), f"active: {len(active):4d}", font=font(15 if image.width < 700 else 17), fill=(226, 231, 239, 255))
    return image


def status_series(frames: list[dict]) -> list[str]:
    state = "approaching Gate 2"
    values = []
    for frame in frames:
        if frame.get("passed_gate") and int(frame.get("next_gate", 0)) >= 2:
            state = "Gate 2 PASSED"
        elif frame.get("collision"):
            state = "Gate 2 COLLISION"
        values.append(state)
    return values


def label_body(image: Image.Image, *, title: str, version: int, status: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    draw.rounded_rectangle((18, 16, 420, 90), radius=12, fill=(4, 7, 11, 190))
    draw.text((34, 25), title, font=font(29), fill=(250, 251, 253, 255))
    draw.text((34, 59), f"checkpoint v{version}  ·  plasticity OFF", font=font(17), fill=(194, 204, 218, 255))
    status_color = (92, 221, 143, 235) if "PASSED" in status else ((242, 103, 83, 235) if "COLLISION" in status else (232, 236, 242, 220))
    draw.rounded_rectangle((18, image.height-52, 285, image.height-14), radius=10, fill=(4, 7, 11, 190))
    draw.text((32, image.height-44), status, font=font(19), fill=status_color)
    return out


def frame_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("frame-*.jpg"))


def main() -> int:
    args = parse_args()
    before_playback = read_json(absolute(args.before_playback))
    after_playback = read_json(absolute(args.after_playback))
    before_checkpoint = absolute(args.before_checkpoint)
    after_checkpoint = absolute(args.after_checkpoint)
    snapshot = absolute(args.snapshot)
    graph = read_json(absolute(args.viewer_graph))
    before_frames_dir = absolute(args.before_frames)
    after_frames_dir = absolute(args.after_frames)
    output = absolute(args.output_dir)
    output_frames = output / "frames"
    if output_frames.exists():
        shutil.rmtree(output_frames)
    output_frames.mkdir(parents=True)

    changed = viewer_edge_deltas(snapshot, graph, before_checkpoint, after_checkpoint)
    changed = [row for row in changed if row["abs_delta"] > 1e-7][: max(1, int(args.top_changed))]
    neural_width = int(args.neural_width)
    if neural_width < 320:
        raise ValueError("--neural-width must be >= 320")
    body_width = 960
    row_height = 540
    _, projected, _ = make_projection(graph, neural_width, row_height)
    before_panel = static_neural_panel(graph, projected, changed, after=False, width=neural_width, height=row_height)
    after_panel = static_neural_panel(graph, projected, changed, after=True, width=neural_width, height=row_height)

    before_images = frame_files(before_frames_dir)
    after_images = frame_files(after_frames_dir)
    if len(before_images) != len(before_playback["frames"]):
        raise RuntimeError(f"before frame mismatch {len(before_images)} != {len(before_playback['frames'])}")
    if len(after_images) != len(after_playback["frames"]):
        raise RuntimeError(f"after frame mismatch {len(after_images)} != {len(after_playback['frames'])}")
    before_status = status_series(before_playback["frames"])
    after_status = status_series(after_playback["frames"])
    total = max(len(before_images), len(after_images))

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    for i in range(total):
        bi = min(i, len(before_images)-1)
        ai = min(i, len(after_images)-1)
        body_before = label_body(Image.open(before_images[bi]).convert("RGB"), title="BEFORE · untrained height adaptation", version=int(before_playback["global_weight_version"]), status=before_status[bi])
        body_after = label_body(Image.open(after_images[ai]).convert("RGB"), title="AFTER · learned CNS state", version=int(after_playback["global_weight_version"]), status=after_status[ai])
        neural_before = active_panel(before_panel, before_playback["frames"][bi], projected)
        neural_after = active_panel(after_panel, after_playback["frames"][ai], projected)
        canvas = Image.new("RGB", (body_width + neural_width, row_height * 2), (0, 0, 0))
        canvas.paste(body_before, (0, 0)); canvas.paste(neural_before, (body_width, 0))
        canvas.paste(body_after, (0, row_height)); canvas.paste(neural_after, (body_width, row_height))
        canvas.save(output_frames / f"frame-{i:05d}.jpg", quality=95, subsampling=0)
        if i % 120 == 0:
            print(f"rendered {i}/{total}", flush=True)
    code = subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(float(args.fps)), "-i", str(output_frames / "frame-%05d.jpg"),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output / "before-after-neural.mp4"),
    ], check=False).returncode
    if code != 0:
        raise RuntimeError(f"ffmpeg returned {code}")

    metadata = {
        "schema_version": 1,
        "before_global_weight_version": int(before_playback["global_weight_version"]),
        "after_global_weight_version": int(after_playback["global_weight_version"]),
        "before_passed_gates": int(before_playback["passed_gates"]),
        "after_passed_gates": int(after_playback["passed_gates"]),
        "gate2_center_z_mm": before_playback.get("gate2_center_z_mm"),
        "evaluation_plasticity": False,
        "viewer_nodes": len(graph["nodes"]),
        "viewer_edges": len(graph["edges"]),
        "changed_viewer_edges": sum(row["abs_delta"] > 1e-7 for row in viewer_edge_deltas(snapshot, graph, before_checkpoint, after_checkpoint)),
        "highlighted_changed_edges": len(changed),
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
