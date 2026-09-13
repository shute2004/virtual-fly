#!/usr/bin/env python3
"""Compound-eye -> MaleCNS vertical-motion population encoder.

This is intentionally a population-level approximation, not a claimed
ommatidium-to-neuron retinotopic reconstruction. FlyGym provides one intensity
sample per ommatidium; a small Reichardt-like temporal correlator estimates
upward/downward ON and OFF motion energy, which is delivered to released
MaleCNS T4c/T4d and T5c/T5d populations.

The approximation is isolated in this file so an individual-cell retinotopic
mapping can replace it without changing the CNS runtime or FlyBody adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from flygym.vision.retina import Retina


@dataclass(frozen=True)
class VerticalMotionReadout:
    t4c_left: float
    t4d_left: float
    t5c_left: float
    t5d_left: float
    t4c_right: float
    t4d_right: float
    t5c_right: float
    t5d_right: float

    def as_stimuli(self) -> dict[str, float]:
        return {
            "vision_t4c_left": self.t4c_left,
            "vision_t4d_left": self.t4d_left,
            "vision_t5c_left": self.t5c_left,
            "vision_t5d_left": self.t5d_left,
            "vision_t4c_right": self.t4c_right,
            "vision_t4d_right": self.t4d_right,
            "vision_t5c_right": self.t5c_right,
            "vision_t5d_right": self.t5d_right,
        }


class VerticalMotionEncoder:
    def __init__(
        self,
        ommatidia_id_map: np.ndarray | None = None,
        *,
        trace_decay: float = 0.65,
        current_gain: float = 18.0,
        max_current: float = 2.5,
        vertical_flip: bool = False,
    ) -> None:
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError("trace_decay must be in [0, 1)")
        if current_gain <= 0.0 or max_current <= 0.0:
            raise ValueError("current scales must be positive")

        if ommatidia_id_map is None:
            ommatidia_id_map = Retina().ommatidia_id_map
        id_map = np.asarray(ommatidia_id_map, dtype=np.int32)
        if id_map.ndim != 2 or int(id_map.max(initial=0)) <= 1:
            raise ValueError("ommatidia_id_map must be a non-trivial 2D ID map")

        self.trace_decay = float(trace_decay)
        self.current_gain = float(current_gain)
        self.max_current = float(max_current)
        self.vertical_flip = bool(vertical_flip)
        self.num_ommatidia = int(id_map.max())
        self.centers = self._centers_from_id_map(id_map, self.num_ommatidia)
        self._up_src, self._up_dst = self._neighbor_pairs(self.centers, upward=True)
        self._down_src, self._down_dst = self._neighbor_pairs(
            self.centers, upward=False
        )
        if len(self._up_src) < self.num_ommatidia // 3:
            raise RuntimeError("too few vertical ommatidial neighbors were resolved")

        self._previous_luminance: np.ndarray | None = None
        self._on_trace: np.ndarray | None = None
        self._off_trace: np.ndarray | None = None

    @staticmethod
    def _centers_from_id_map(id_map: np.ndarray, count: int) -> np.ndarray:
        rows, cols = np.indices(id_map.shape)
        centers = np.empty((count, 2), dtype=np.float64)
        for ommatidium in range(1, count + 1):
            mask = id_map == ommatidium
            if not np.any(mask):
                raise ValueError(f"ommatidium ID {ommatidium} is absent from ID map")
            centers[ommatidium - 1] = (
                float(np.mean(rows[mask])),
                float(np.mean(cols[mask])),
            )
        return centers

    @staticmethod
    def _neighbor_pairs(
        centers: np.ndarray, *, upward: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        src: list[int] = []
        dst: list[int] = []
        for i, (row_i, col_i) in enumerate(centers):
            dr = centers[:, 0] - row_i
            dc = centers[:, 1] - col_i
            axial = -dr if upward else dr
            valid = axial > 0.25
            # Prefer a nearby vertical neighbour while allowing the stagger of
            # the hexagonal lattice. A large lateral displacement is rejected.
            valid &= np.abs(dc) <= axial * 1.35 + 1.0
            if not np.any(valid):
                continue
            cost = axial + 2.5 * np.abs(dc)
            cost = np.where(valid, cost, np.inf)
            j = int(np.argmin(cost))
            if np.isfinite(cost[j]):
                src.append(i)
                dst.append(j)
        return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)

    def reset(self) -> None:
        self._previous_luminance = None
        self._on_trace = None
        self._off_trace = None

    @staticmethod
    def _direction_energy(
        trace: np.ndarray,
        event: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
    ) -> np.ndarray:
        forward = trace[:, src] * event[:, dst]
        reverse = trace[:, dst] * event[:, src]
        opponent = np.maximum(forward - reverse, 0.0)
        if opponent.shape[1] == 0:
            return np.zeros(2, dtype=np.float64)
        return np.mean(opponent, axis=1)

    def _to_current(self, energy: np.ndarray) -> np.ndarray:
        # sqrt keeps weak but spatially coherent motion visible while retaining
        # a hard upper bound on injected current.
        return np.minimum(
            self.max_current,
            self.current_gain * np.sqrt(np.maximum(energy, 0.0)),
        )

    def encode(self, ommatidia_readouts: np.ndarray) -> VerticalMotionReadout:
        readouts = np.asarray(ommatidia_readouts, dtype=np.float64)
        expected = (2, self.num_ommatidia, 2)
        if readouts.shape != expected:
            raise ValueError(f"expected ommatidia readouts {expected}, got {readouts.shape}")
        if not np.all(np.isfinite(readouts)):
            raise ValueError("ommatidia readouts contain non-finite values")

        luminance = np.clip(readouts.sum(axis=2), 0.0, 1.0)
        if self._previous_luminance is None:
            self._previous_luminance = luminance.copy()
            self._on_trace = np.zeros_like(luminance)
            self._off_trace = np.zeros_like(luminance)
            return VerticalMotionReadout(*(0.0 for _ in range(8)))

        on_event = np.maximum(luminance - self._previous_luminance, 0.0)
        off_event = np.maximum(self._previous_luminance - luminance, 0.0)
        assert self._on_trace is not None and self._off_trace is not None

        up_on = self._direction_energy(
            self._on_trace, on_event, self._up_src, self._up_dst
        )
        down_on = self._direction_energy(
            self._on_trace, on_event, self._down_src, self._down_dst
        )
        up_off = self._direction_energy(
            self._off_trace, off_event, self._up_src, self._up_dst
        )
        down_off = self._direction_energy(
            self._off_trace, off_event, self._down_src, self._down_dst
        )

        if self.vertical_flip:
            up_on, down_on = down_on, up_on
            up_off, down_off = down_off, up_off

        t4c = self._to_current(up_on)
        t4d = self._to_current(down_on)
        t5c = self._to_current(up_off)
        t5d = self._to_current(down_off)

        self._on_trace = self.trace_decay * self._on_trace + on_event
        self._off_trace = self.trace_decay * self._off_trace + off_event
        self._previous_luminance = luminance.copy()

        return VerticalMotionReadout(
            t4c_left=float(t4c[0]),
            t4d_left=float(t4d[0]),
            t5c_left=float(t5c[0]),
            t5d_left=float(t5d[0]),
            t4c_right=float(t4c[1]),
            t4d_right=float(t4d[1]),
            t5c_right=float(t5c[1]),
            t5d_right=float(t5d[1]),
        )


def _synthetic_readouts(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros((2, len(values), 2), dtype=np.float64)
    out[:, :, 0] = values
    return out


def _self_test() -> None:
    # Four vertically stacked artificial ommatidia. A bright point moves from
    # bottom to top; the upward ON channel must dominate the downward channel.
    id_map = np.arange(1, 5, dtype=np.int32).reshape(4, 1)
    encoder = VerticalMotionEncoder(
        id_map, trace_decay=0.5, current_gain=10.0, max_current=3.0
    )
    encoder.encode(_synthetic_readouts(np.zeros(4)))
    encoder.encode(_synthetic_readouts(np.array([0.0, 0.0, 0.0, 1.0])))
    result = encoder.encode(
        _synthetic_readouts(np.array([0.0, 0.0, 1.0, 0.0]))
    )
    if result.t4c_left <= result.t4d_left:
        raise RuntimeError(
            "synthetic upward motion did not dominate the upward T4 channel"
        )
    print("visual_motion_encoder=PASS")


if __name__ == "__main__":
    _self_test()
