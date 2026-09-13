#!/usr/bin/env python3
"""Low-frequency synapse-change snapshots for optional training visualization.

The monitor is deliberately off the hot path. When requested, it asks the
persistent CNS process to dump the current synaptic weights, compares them with
the deterministic initial weights implied by the released synapse counts, keeps
only aggregate statistics plus the largest changes, and deletes the temporary
full dump.

This avoids transferring ~25M edge weights every control step while still
providing enough information to visualize where learning is changing the
connectome.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SynapseMonitorConfig:
    top_n: int = 64
    chunk_edges: int = 1_000_000
    min_abs_delta: float = 1e-7
    synapse_scale: float = 0.02


class SynapseMonitor:
    def __init__(
        self,
        *,
        snapshot: Path,
        output: Path,
        config: SynapseMonitorConfig = SynapseMonitorConfig(),
    ) -> None:
        if config.top_n < 1 or config.chunk_edges < 1:
            raise ValueError("top_n and chunk_edges must be >= 1")
        if config.min_abs_delta < 0.0 or config.synapse_scale <= 0.0:
            raise ValueError("invalid synapse monitor thresholds")

        self.snapshot = Path(snapshot)
        self.output = Path(output)
        self.config = config
        manifest_path = self.snapshot / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.edge_count = int(manifest["edge_count"])
        self.neuron_count = int(manifest["neuron_count"])
        self.body_ids = np.memmap(
            self.snapshot / manifest["body_ids_file"], dtype="<u8", mode="r"
        )
        self.row_offsets = np.memmap(
            self.snapshot / manifest["row_offsets_file"], dtype="<u4", mode="r"
        )
        self.pre_indices = np.memmap(
            self.snapshot / manifest["pre_indices_file"], dtype="<u4", mode="r"
        )
        self.synapse_counts = np.memmap(
            self.snapshot / manifest["synapse_counts_file"], dtype="<u4", mode="r"
        )

        if len(self.body_ids) != self.neuron_count:
            raise RuntimeError("body-id array does not match snapshot manifest")
        if len(self.pre_indices) != self.edge_count or len(self.synapse_counts) != self.edge_count:
            raise RuntimeError("edge arrays do not match snapshot manifest")
        if len(self.row_offsets) != self.neuron_count + 1:
            raise RuntimeError("row-offset array does not match snapshot manifest")

        self.labels = self._load_labels()
        self.output.parent.mkdir(parents=True, exist_ok=True)

    def _load_labels(self) -> list[str]:
        annotations_path = self.snapshot / "annotations.feather"
        if not annotations_path.exists():
            return [str(int(body_id)) for body_id in self.body_ids]

        frame = pd.read_feather(annotations_path)
        if len(frame) != self.neuron_count:
            raise RuntimeError("annotation table does not match snapshot neuron count")
        if "bodyId" in frame.columns:
            annotation_ids = frame["bodyId"].astype(np.uint64).to_numpy()
            if not np.array_equal(annotation_ids, np.asarray(self.body_ids)):
                raise RuntimeError("annotation table order does not match snapshot body IDs")

        preferred = [
            name
            for name in ("type", "flywireType", "mancType", "instance")
            if name in frame.columns
        ]
        labels: list[str] = []
        for index, row in frame.iterrows():
            label = ""
            for column in preferred:
                value = row[column]
                if pd.notna(value) and str(value).strip():
                    label = str(value).strip()
                    break
            if not label:
                label = str(int(self.body_ids[index]))
            labels.append(label)
        return labels

    def capture(self, brain, *, episode: int, control_step: int) -> dict[str, object]:
        temp = self.output.with_suffix(f".episode-{episode:04d}.weights.tmp")
        checkpoint = brain.save_weights(temp)
        try:
            summary = self._summarize(temp)
        finally:
            temp.unlink(missing_ok=True)

        record = {
            "episode": int(episode),
            "control_step": int(control_step),
            "checkpoint_weight_count": int(checkpoint.get("weights", 0)),
            **summary,
        }
        with self.output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record

    def _summarize(self, weight_path: Path) -> dict[str, object]:
        current = np.memmap(weight_path, dtype="<f4", mode="r")
        if len(current) != self.edge_count:
            raise RuntimeError(
                f"weight dump contains {len(current)} values; expected {self.edge_count}"
            )

        changed = 0
        abs_sum = 0.0
        max_abs = 0.0
        candidates: list[tuple[float, int, float, float]] = []
        cfg = self.config

        for start in range(0, self.edge_count, cfg.chunk_edges):
            end = min(self.edge_count, start + cfg.chunk_edges)
            counts = np.asarray(self.synapse_counts[start:end], dtype=np.float32)
            initial = counts * np.float32(cfg.synapse_scale)
            now = np.asarray(current[start:end], dtype=np.float32)
            delta = now - initial
            abs_delta = np.abs(delta)

            changed += int(np.count_nonzero(abs_delta > cfg.min_abs_delta))
            abs_sum += float(np.sum(abs_delta, dtype=np.float64))
            if len(abs_delta):
                max_abs = max(max_abs, float(np.max(abs_delta)))

            local_n = min(cfg.top_n, len(abs_delta))
            if local_n == 0:
                continue
            if local_n == len(abs_delta):
                local = np.arange(local_n, dtype=np.int64)
            else:
                local = np.argpartition(abs_delta, -local_n)[-local_n:]
            for local_index in local:
                magnitude = float(abs_delta[local_index])
                if magnitude <= cfg.min_abs_delta:
                    continue
                edge = start + int(local_index)
                candidates.append(
                    (
                        magnitude,
                        edge,
                        float(initial[local_index]),
                        float(delta[local_index]),
                    )
                )

        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[: cfg.top_n]
        top = [self._edge_record(*item) for item in candidates]

        return {
            "edge_count": self.edge_count,
            "changed_synapses": changed,
            "mean_abs_delta": abs_sum / self.edge_count if self.edge_count else 0.0,
            "max_abs_delta": max_abs,
            "top_changed": top,
        }

    def _edge_record(
        self,
        abs_delta: float,
        edge: int,
        initial_weight: float,
        delta: float,
    ) -> dict[str, object]:
        pre_index = int(self.pre_indices[edge])
        post_index = int(np.searchsorted(self.row_offsets, edge, side="right") - 1)
        current_weight = initial_weight + delta
        return {
            "edge": edge,
            "pre_index": pre_index,
            "post_index": post_index,
            "pre_body_id": int(self.body_ids[pre_index]),
            "post_body_id": int(self.body_ids[post_index]),
            "pre_label": self.labels[pre_index],
            "post_label": self.labels[post_index],
            "synapse_count": int(self.synapse_counts[edge]),
            "initial_weight": initial_weight,
            "current_weight": current_weight,
            "delta": delta,
            "abs_delta": abs_delta,
        }
