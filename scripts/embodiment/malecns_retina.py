#!/usr/bin/env python3
"""Local optical stimulation of released MaleCNS R1-R6 photoreceptors.

This module deliberately does not detect motion, edges, obstacles, gap position, or
any other visual feature outside the nervous system. Each inferred MaleCNS optical
column is associated with the nearest FlyBody ommatidium in a calibrated 2-D eye
lattice seam; that local ommatidium intensity is converted to current for the
released R1-R6 body IDs assigned to the column. All subsequent spatial/temporal
integration is left to the MaleCNS network.

R1-R6 photoreceptors adapt strongly to mean luminance while retaining responses
to local temporal contrast. The transduction seam therefore keeps one independent
adaptation state per FlyBody ommatidium. The first sample provides the ordinary
light-on response; subsequent samples inject only that ommatidium's signed local
contrast relative to its exponentially adapting baseline. This is photoreceptor
transduction, not external visual feature extraction.

Provenance boundaries:
- R1-R6 identity, root side, lamina wiring and assignedOlHex coordinates:
  MaleCNS-derived map;
- MaleCNS hex-lattice -> FlyBody ommatidium-lattice registration:
  calibrated geometric seam;
- local irradiance -> contrast-current gain, adaptation time constant and contrast
  clamp: calibrated transduction seam.

FlyBody already performs the raw-camera -> compound-eye conversion using its 721
ommatidia per eye. Using those readouts is important: the raw camera image is an
internal rendering surface and is not itself the compound-eye sensory output.
The two FlyBody channels encode pale/yellow ommatidia; because the Flyppy world is
achromatic, their local sum is used as a brightness proxy without spatial pooling
or external feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from flygym.vision.retina import Retina


EXPECTED_ASSIGNMENT_METHOD = (
    "dominant_R1-R6_contacts_to_coordinate_L1_L2_L3_with_R1-R6_rootSide"
)


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
        sample_interval_s: float = 5.0e-4,
        adaptation_tau_s: float = 5.0e-2,
        contrast_denominator_floor: float = 5.0e-2,
        contrast_clip: float = 1.0,
    ) -> None:
        if not np.isfinite(current_gain) or current_gain <= 0.0:
            raise ValueError("current_gain must be finite and positive")
        if not np.isfinite(current_floor) or current_floor < 0.0:
            raise ValueError("current_floor must be finite and non-negative")
        if not np.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
            raise ValueError("sample_interval_s must be finite and positive")
        if not np.isfinite(adaptation_tau_s) or adaptation_tau_s <= 0.0:
            raise ValueError("adaptation_tau_s must be finite and positive")
        if (
            not np.isfinite(contrast_denominator_floor)
            or contrast_denominator_floor <= 0.0
        ):
            raise ValueError("contrast_denominator_floor must be finite and positive")
        if not np.isfinite(contrast_clip) or contrast_clip <= 0.0:
            raise ValueError("contrast_clip must be finite and positive")

        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 3:
            raise ValueError("unsupported retinotopic vision map schema")
        if data.get("assignment_method") != EXPECTED_ASSIGNMENT_METHOD:
            raise ValueError(
                "retinotopic map was not built from the expected MaleCNS lamina "
                "contact-distribution inference"
            )
        columns = data.get("columns")
        if not isinstance(columns, list) or not columns:
            raise ValueError("retinotopic vision map contains no columns")

        self.current_gain = float(current_gain)
        self.current_floor = float(current_floor)
        self.sample_interval_s = float(sample_interval_s)
        self.adaptation_tau_s = float(adaptation_tau_s)
        self.contrast_denominator_floor = float(contrast_denominator_floor)
        self.contrast_clip = float(contrast_clip)
        self._adaptation_alpha = 1.0 - math.exp(
            -self.sample_interval_s / self.adaptation_tau_s
        )
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

        retina = Retina()
        self.flybody_ommatidia_per_eye = int(retina.num_ommatidia_per_eye)
        ommatidia_uv = self._flybody_ommatidia_uv(retina.ommatidia_id_map)
        for side in ("L", "R"):
            if not self._columns_by_side[side]:
                raise ValueError(f"retinotopic vision map has no {side} eye columns")
            self._attach_uv(self._columns_by_side[side])
            self._attach_ommatidia(self._columns_by_side[side], ommatidia_uv)

        self._adapted_light: dict[str, np.ndarray] = {
            side: np.full(self.flybody_ommatidia_per_eye, np.nan, dtype=np.float64)
            for side in ("L", "R")
        }

    def reset_adaptation(self) -> None:
        """Reset local photoreceptor adaptation state at an explicit trial boundary."""

        for values in self._adapted_light.values():
            values.fill(np.nan)

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
    def _flybody_ommatidia_uv(ommatidia_id_map: np.ndarray) -> np.ndarray:
        """Return normalized centroids for FlyBody's numbered ommatidia."""

        ids = np.asarray(ommatidia_id_map, dtype=np.int64)
        if ids.ndim != 2 or int(ids.max(initial=0)) < 1:
            raise ValueError("FlyBody ommatidia id map is invalid")
        height, width = ids.shape
        flat_ids = ids.ravel()
        valid = flat_ids > 0
        positive_ids = flat_ids[valid]
        n = int(positive_ids.max())
        rows, cols = np.indices(ids.shape, dtype=np.float64)
        counts = np.bincount(positive_ids, minlength=n + 1)[1:].astype(np.float64)
        if np.any(counts <= 0.0):
            raise ValueError("FlyBody ommatidia IDs are not contiguous")
        row_sum = np.bincount(
            positive_ids, weights=rows.ravel()[valid], minlength=n + 1
        )[1:]
        col_sum = np.bincount(
            positive_ids, weights=cols.ravel()[valid], minlength=n + 1
        )[1:]
        center_row = row_sum / counts
        center_col = col_sum / counts
        u = center_col / max(1.0, float(width - 1))
        v = 1.0 - center_row / max(1.0, float(height - 1))
        return np.column_stack((u, v))

    @staticmethod
    def _attach_ommatidia(
        columns: list[dict[str, object]], ommatidia_uv: np.ndarray
    ) -> None:
        for item in columns:
            point = np.asarray([float(item["u"]), float(item["v"])])
            distance2 = np.sum((ommatidia_uv - point) ** 2, axis=1)
            item["ommatidium_index"] = int(np.argmin(distance2))

    @staticmethod
    def _eye_readouts(sim, fly) -> dict[str, np.ndarray]:
        readouts = np.asarray(sim.get_ommatidia_readouts(fly.name), dtype=np.float32)
        names = list(fly.eyecameraname_to_mjcfcamera.keys())
        if readouts.ndim != 3 or readouts.shape[2] != 2:
            raise RuntimeError(
                f"unexpected FlyBody ommatidia readout shape: {readouts.shape}"
            )
        if len(names) != len(readouts):
            raise RuntimeError(
                f"eye camera/readout mismatch: names={names}, eyes={len(readouts)}"
            )
        by_name = {
            name: np.asarray(readout, dtype=np.float32)
            for name, readout in zip(names, readouts, strict=True)
        }
        try:
            return {"L": by_name["l_eye_cam"], "R": by_name["r_eye_cam"]}
        except KeyError as exc:
            raise RuntimeError(f"FlyBody eye camera names are unexpected: {names}") from exc

    @staticmethod
    def _local_achromatic(readouts: np.ndarray, ommatidium_index: int) -> float:
        if not 0 <= ommatidium_index < len(readouts):
            raise IndexError(
                f"ommatidium index {ommatidium_index} outside 0..{len(readouts) - 1}"
            )
        value = float(np.sum(readouts[ommatidium_index], dtype=np.float64))
        return float(np.clip(value, 0.0, 1.0))

    def _transduce(self, side: str, ommatidium_index: int, light: float) -> float:
        baseline = float(self._adapted_light[side][ommatidium_index])
        if not np.isfinite(baseline):
            # A light-on transient seeds the local state. Repeated presentation of
            # the same luminance does not continue injecting an absolute DC current.
            self._adapted_light[side][ommatidium_index] = light
            return light * self.current_gain

        denominator = max(abs(baseline), self.contrast_denominator_floor)
        contrast = (light - baseline) / denominator
        contrast = float(np.clip(contrast, -self.contrast_clip, self.contrast_clip))
        self._adapted_light[side][ommatidium_index] = (
            baseline + self._adaptation_alpha * (light - baseline)
        )
        return contrast * self.current_gain

    def encode(self, sim, fly) -> RetinalDrive:
        eyes = self._eye_readouts(sim, fly)
        body_currents: list[tuple[int, float]] = []
        active_columns = 0
        current_magnitudes: list[float] = []

        for side in ("L", "R"):
            eye = eyes[side]
            if len(eye) != self.flybody_ommatidia_per_eye:
                raise RuntimeError(
                    f"FlyBody {side} eye has {len(eye)} ommatidia; "
                    f"expected {self.flybody_ommatidia_per_eye}"
                )
            for column in self._columns_by_side[side]:
                ommatidium_index = int(column["ommatidium_index"])
                light = self._local_achromatic(eye, ommatidium_index)
                current = self._transduce(side, ommatidium_index, light)
                if abs(current) <= self.current_floor:
                    continue
                active_columns += 1
                magnitude = abs(current)
                for body_id in column["body_ids"]:
                    body_currents.append((int(body_id), current))
                    current_magnitudes.append(magnitude)

        return RetinalDrive(
            body_currents=tuple(body_currents),
            active_photoreceptors=len(body_currents),
            active_columns=active_columns,
            mean_current=(
                float(np.mean(current_magnitudes)) if current_magnitudes else 0.0
            ),
            max_current=(
                float(np.max(current_magnitudes)) if current_magnitudes else 0.0
            ),
        )
