#!/usr/bin/env python3
"""Local optical stimulation of MaleCNS R1-R6 photoreceptors.

This module deliberately does not detect motion, edges, obstacles, gap position, or
any other visual feature outside the nervous system. Each observed MaleCNS lamina
cartridge samples one local point in the corresponding FlyBody eye image; that
local light is converted to current for the actual R1-R6 body IDs that connect to
the cartridge's L1 neuron. All subsequent spatial/temporal integration is left to
the MaleCNS network.

Provenance boundaries:
- L1 optic-column coordinates and R1-R6 -> L1 wiring: MaleCNS-derived map;
- hex-lattice unrolling into the FlyBody eye camera: calibrated geometric seam;
- local green-channel irradiance -> injected current gain: calibrated transduction seam.

The Flyppy world is achromatic, so one local rendered color channel is sufficient
to represent local brightness without combining spatial samples or extracting a
visual feature. A future spectral/phototransduction model can replace only this
local conversion without changing the retinotopic boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RetinalDrive:
    body_currents: tuple[tuple[int, float], ...]
    active_photoreceptors: int
    active_columns: int
    mean_current: float
    max_current: float


class MaleCNSRetina:
    def __init__(
        self,
        mapping_path: Path,
        *,
        current_gain: float = 2.0,
        current_floor: float = 1e-6,
    ) -> None:
        if not np.isfinite(current_gain) or current_gain <= 0.0:
            raise ValueError("current_gain must be finite and positive")
        if not np.isfinite(current_floor) or current_floor < 0.0:
            raise ValueError("current_floor must be finite and non-negative")

        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2:
            raise ValueError("unsupported retinotopic vision map schema")
        if data.get("assignment_method") != (
            "observed_R1-R6_to_L1_connectivity_with_observed_L1_hex"
        ):
            raise ValueError(
                "retinotopic map was not built from observed R1-R6 -> L1 wiring"
            )
        columns = data.get("columns")
        if not isinstance(columns, list) or not columns:
            raise ValueError("retinotopic vision map contains no columns")

        self.current_gain = float(current_gain)
        self.current_floor = float(current_floor)
        self.mapping_path = mapping_path
        self._columns_by_side: dict[str, list[dict[str, object]]] = {"L": [], "R": []}
        assigned_body: dict[int, tuple[str, int, int]] = {}
        for column in columns:
            side = str(column["side"])
            if side not in self._columns_by_side:
                raise ValueError(f"invalid eye side in retinotopic map: {side}")
            h1 = int(column["hex1"])
            h2 = int(column["hex2"])
            body_ids = tuple(int(value) for value in column["r1_r6_body_ids"])
            if not body_ids:
                continue
            key = (side, h1, h2)
            for body_id in body_ids:
                previous = assigned_body.get(body_id)
                if previous is not None and previous != key:
                    raise ValueError(
                        f"R1-R6 body ID {body_id} is assigned to multiple optical columns: "
                        f"{previous} and {key}"
                    )
                assigned_body[body_id] = key
            self._columns_by_side[side].append(
                {"hex1": h1, "hex2": h2, "body_ids": body_ids}
            )

        for side in ("L", "R"):
            if not self._columns_by_side[side]:
                raise ValueError(f"retinotopic vision map has no {side} eye columns")
            self._attach_uv(self._columns_by_side[side])

    @staticmethod
    def _attach_uv(columns: list[dict[str, object]]) -> None:
        # Standard axial-hex planar unrolling. This is a calibrated optical seam:
        # it preserves every MaleCNS column independently but does not claim that
        # normalized (u,v) is a measured ommatidial optical-axis calibration.
        h1 = np.asarray([int(item["hex1"]) for item in columns], dtype=np.float64)
        h2 = np.asarray([int(item["hex2"]) for item in columns], dtype=np.float64)
        x = h1 + 0.5 * h2
        y = h2 * (np.sqrt(3.0) / 2.0)
        x_span = float(np.ptp(x))
        y_span = float(np.ptp(y))
        if x_span <= 0.0 or y_span <= 0.0:
            raise ValueError("degenerate MaleCNS optic-lobe hex coordinates")
        u = (x - float(np.min(x))) / x_span
        v = (y - float(np.min(y))) / y_span
        for item, u_value, v_value in zip(columns, u, v, strict=True):
            item["u"] = float(u_value)
            item["v"] = float(v_value)

    @staticmethod
    def _eye_frames(sim, fly) -> dict[str, np.ndarray]:
        frames = np.asarray(sim.get_raw_vision(fly.name))
        names = list(fly.eyecameraname_to_mjcfcamera.keys())
        if len(names) != len(frames):
            raise RuntimeError(
                f"eye camera/frame mismatch: names={names}, frames={len(frames)}"
            )
        by_name = {
            name: np.asarray(frame)
            for name, frame in zip(names, frames, strict=True)
        }
        try:
            return {"L": by_name["l_eye_cam"], "R": by_name["r_eye_cam"]}
        except KeyError as exc:
            raise RuntimeError(f"FlyBody eye camera names are unexpected: {names}") from exc

    @staticmethod
    def _local_green(frame: np.ndarray, u: float, v: float) -> float:
        if frame.ndim != 3 or frame.shape[2] < 2:
            raise ValueError(f"expected RGB eye frame, got shape {frame.shape}")
        height, width = frame.shape[:2]
        # The orientation between MaleCNS hex axes and the FlyBody camera plane is
        # part of the calibrated seam. No neighboring pixels are pooled here.
        col = int(round(np.clip(u, 0.0, 1.0) * (width - 1)))
        row = int(round((1.0 - np.clip(v, 0.0, 1.0)) * (height - 1)))
        value = float(frame[row, col, 1])
        if np.issubdtype(frame.dtype, np.integer):
            value /= float(np.iinfo(frame.dtype).max)
        return float(np.clip(value, 0.0, 1.0))

    def encode(self, sim, fly) -> RetinalDrive:
        frames = self._eye_frames(sim, fly)
        body_currents: list[tuple[int, float]] = []
        active_columns = 0
        currents: list[float] = []

        for side in ("L", "R"):
            frame = frames[side]
            for column in self._columns_by_side[side]:
                light = self._local_green(
                    frame,
                    float(column["u"]),
                    float(column["v"]),
                )
                current = light * self.current_gain
                if current <= self.current_floor:
                    continue
                active_columns += 1
                for body_id in column["body_ids"]:
                    body_currents.append((int(body_id), current))
                    currents.append(current)

        return RetinalDrive(
            body_currents=tuple(body_currents),
            active_photoreceptors=len(body_currents),
            active_columns=active_columns,
            mean_current=float(np.mean(currents)) if currents else 0.0,
            max_current=float(np.max(currents)) if currents else 0.0,
        )
