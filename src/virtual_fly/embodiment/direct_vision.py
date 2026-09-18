#!/usr/bin/env python3
"""Image-free compound-eye sampling for Flyppy.

The production FlyGym path renders two full RGB eye images, applies a fisheye
remap, then averages pixels into ommatidia.  This module keeps the *sensor*
semantics at the ommatidium boundary but removes the framebuffer from the hot
path:

    MuJoCo scene -> weighted ommatidial ray bundles -> receptor values

The ray footprints are not arbitrary cones.  They are derived from FlyGym 2.1's
actual ``ommatidia_id_map`` and the exact integer fisheye remap used by
``Retina.correct_fisheye``.  Each ommatidium's corrected-image footprint is
mapped back to raw camera pixels, converted to camera-space ray directions and
compressed deterministically to K weighted representative rays.

Only ommatidia currently referenced by the MaleCNS retinotopic seam are queried.
The full 721-per-eye biological sensor definition remains intact; this is lazy
execution of causally required receptors, not a smaller virtual eye.

This first direct sensor deliberately separates geometry from photometry.  Ray
visibility is exact with MuJoCo's collision geometry/group filtering; radiance is
a lightweight achromatic approximation using geom/material base colour, a white
sky, and an analytic checker approximation for FlyGym's flat ground.  The
reference raster path remains the oracle while this approximation is measured.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mujoco as mj
import numpy as np
from flygym.vision.retina import Retina


_EYE_NAME_BY_SIDE = {"L": "l_eye_cam", "R": "r_eye_cam"}
# FlyGym get_raw_vision disables geom groups 1 and 2 for eye rendering.
_EYE_GEOMGROUP = np.asarray([1, 0, 0, 1, 1, 1], dtype=np.uint8)


@dataclass(frozen=True)
class RayBundle:
    """One eye's packed set of representative receptor rays."""

    directions_local: np.ndarray
    ommatidium_indices: np.ndarray
    weights: np.ndarray
    channels: np.ndarray

    @property
    def ray_count(self) -> int:
        return int(self.ommatidium_indices.size)


@dataclass(frozen=True)
class DirectEyeReadoutStats:
    left_rays: int
    right_rays: int
    required_left: int
    required_right: int

    @property
    def total_rays(self) -> int:
        return self.left_rays + self.right_rays


class DirectOmmatidialSensor:
    """Evaluate FlyGym ommatidial receptive fields without rendering RGB images."""

    def __init__(
        self,
        malecns_retina: Any,
        *,
        rays_per_ommatidium: int = 7,
        cutoff_mm: float = 10_000.0,
        cluster_iterations: int = 8,
    ) -> None:
        if rays_per_ommatidium < 1:
            raise ValueError("rays_per_ommatidium must be positive")
        if not math.isfinite(cutoff_mm) or cutoff_mm <= 0.0:
            raise ValueError("cutoff_mm must be finite and positive")
        if cluster_iterations < 1:
            raise ValueError("cluster_iterations must be positive")

        self.rays_per_ommatidium = int(rays_per_ommatidium)
        self.cutoff_mm = float(cutoff_mm)
        self.cluster_iterations = int(cluster_iterations)
        self.retina = Retina()
        self.num_ommatidia_per_eye = int(self.retina.num_ommatidia_per_eye)

        runtime_columns = getattr(malecns_retina, "_runtime_columns", None)
        if not isinstance(runtime_columns, dict):
            raise TypeError("malecns_retina does not expose compiled retinal columns")
        self.required_by_side: dict[str, tuple[int, ...]] = {}
        for side in ("L", "R"):
            columns = runtime_columns.get(side)
            if columns is None:
                raise ValueError(f"missing {side} retinal runtime columns")
            required = tuple(
                sorted({int(column.ommatidium_index) for column in columns})
            )
            if not required:
                raise ValueError(f"no required ommatidia for side {side}")
            if required[0] < 0 or required[-1] >= self.num_ommatidia_per_eye:
                raise ValueError(f"invalid {side} ommatidium index range")
            self.required_by_side[side] = required

        self._source_pixels = self._build_source_pixel_footprints()
        self._bundles: dict[tuple[str, int, float], RayBundle] = {}
        self.last_stats = DirectEyeReadoutStats(
            left_rays=0,
            right_rays=0,
            required_left=len(self.required_by_side["L"]),
            required_right=len(self.required_by_side["R"]),
        )

    def _build_source_pixel_footprints(
        self,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
        """Map corrected ommatidium pixels back to raw-camera source pixels.

        The returned tuple per ommatidium is ``(raw_rows, raw_cols, total_pixels)``.
        ``total_pixels`` includes corrected pixels that map outside the raw frame;
        those pixels are black in FlyGym's fisheye image and therefore retain zero
        weight in the direct sensor.
        """

        ids = np.asarray(self.retina.ommatidia_id_map, dtype=np.int32)
        nrows = int(self.retina.nrows)
        ncols = int(self.retina.ncols)
        if ids.shape != (nrows, ncols):
            raise ValueError(
                "FlyGym ommatidia map/image geometry mismatch: "
                f"map={ids.shape}, retina={(nrows, ncols)}"
            )

        rows, cols = np.indices((nrows, ncols), dtype=np.float64)
        dst_row_norm = ((2.0 * rows - nrows) / nrows) / float(self.retina.zoom)
        dst_col_norm = ((2.0 * cols - ncols) / ncols) / float(self.retina.zoom)
        radius2 = dst_col_norm * dst_col_norm + dst_row_norm * dst_row_norm
        denom = 1.0 - float(self.retina.distortion_coefficient) * radius2 + 1e-6
        src_row = (((dst_row_norm / denom + 1.0) * nrows) / 2.0).astype(np.int32)
        src_col = (((dst_col_norm / denom + 1.0) * ncols) / 2.0).astype(np.int32)
        in_bounds = (
            (src_row >= 0)
            & (src_row < nrows)
            & (src_col >= 0)
            & (src_col < ncols)
        )

        required = set(self.required_by_side["L"]) | set(self.required_by_side["R"])
        result: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        for index in sorted(required):
            corrected_mask = ids == (index + 1)
            total = int(np.count_nonzero(corrected_mask))
            if total < 1:
                raise RuntimeError(f"ommatidium {index} has no Retina footprint")
            valid = corrected_mask & in_bounds
            result[index] = (
                np.ascontiguousarray(src_row[valid], dtype=np.int32),
                np.ascontiguousarray(src_col[valid], dtype=np.int32),
                total,
            )
        return result

    @staticmethod
    def _pixel_rays(
        rows: np.ndarray,
        cols: np.ndarray,
        *,
        nrows: int,
        ncols: int,
        fovy_deg: float,
    ) -> np.ndarray:
        """Convert raw MuJoCo framebuffer pixels to camera-local unit rays."""

        tan_half_y = math.tan(math.radians(float(fovy_deg)) / 2.0)
        tan_half_x = tan_half_y * (float(ncols) / float(nrows))
        x = (2.0 * (cols.astype(np.float64) + 0.5) / ncols - 1.0) * tan_half_x
        y = (1.0 - 2.0 * (rows.astype(np.float64) + 0.5) / nrows) * tan_half_y
        z = -np.ones_like(x)
        directions = np.column_stack((x, y, z))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.maximum(norms, 1e-15)
        return directions

    @staticmethod
    def _farthest_initial_centers(samples: np.ndarray, k: int) -> np.ndarray:
        mean = np.sum(samples, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-15:
            first = 0
        else:
            mean /= norm
            first = int(np.argmax(samples @ mean))
        chosen = [first]
        best_similarity = samples @ samples[first]
        while len(chosen) < k:
            # Smallest similarity to every selected centre = largest angle.
            next_index = int(np.argmin(best_similarity))
            chosen.append(next_index)
            best_similarity = np.maximum(
                best_similarity, samples @ samples[next_index]
            )
        return np.ascontiguousarray(samples[np.asarray(chosen, dtype=np.int32)])

    def _cluster_rays(
        self,
        directions: np.ndarray,
        *,
        total_pixel_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic spherical k-means with area weights from Retina pixels."""

        valid_count = len(directions)
        if valid_count == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
        k = min(self.rays_per_ommatidium, valid_count)
        if k == valid_count:
            return (
                np.ascontiguousarray(directions, dtype=np.float64),
                np.full(k, 1.0 / float(total_pixel_count), dtype=np.float64),
            )

        centers = self._farthest_initial_centers(directions, k)
        labels = np.zeros(valid_count, dtype=np.int32)
        for _ in range(self.cluster_iterations):
            labels = np.argmax(directions @ centers.T, axis=1).astype(np.int32)
            next_centers = centers.copy()
            for cluster in range(k):
                members = directions[labels == cluster]
                if len(members) == 0:
                    continue
                center = np.sum(members, axis=0)
                norm = float(np.linalg.norm(center))
                if norm > 1e-15:
                    next_centers[cluster] = center / norm
            if np.allclose(next_centers, centers, rtol=0.0, atol=1e-12):
                centers = next_centers
                break
            centers = next_centers

        labels = np.argmax(directions @ centers.T, axis=1).astype(np.int32)
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        weights = counts / float(total_pixel_count)
        return (
            np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
        )

    def _bundle_for_camera(
        self,
        side: str,
        camera_id: int,
        fovy_deg: float,
    ) -> RayBundle:
        key = (side, int(camera_id), round(float(fovy_deg), 12))
        existing = self._bundles.get(key)
        if existing is not None:
            return existing

        all_dirs: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        all_weights: list[np.ndarray] = []
        all_channels: list[np.ndarray] = []
        pale_mask = np.asarray(self.retina.pale_type_mask, dtype=np.int32)
        for index in self.required_by_side[side]:
            rows, cols, total = self._source_pixels[index]
            directions = self._pixel_rays(
                rows,
                cols,
                nrows=int(self.retina.nrows),
                ncols=int(self.retina.ncols),
                fovy_deg=float(fovy_deg),
            )
            centers, weights = self._cluster_rays(
                directions,
                total_pixel_count=total,
            )
            if len(centers) == 0:
                continue
            all_dirs.append(centers)
            all_indices.append(np.full(len(centers), index, dtype=np.int32))
            all_weights.append(weights)
            all_channels.append(
                np.full(len(centers), int(pale_mask[index]) + 1, dtype=np.int8)
            )

        if not all_dirs:
            raise RuntimeError(f"no valid direct rays for eye {side}")
        bundle = RayBundle(
            directions_local=np.ascontiguousarray(np.concatenate(all_dirs, axis=0)),
            ommatidium_indices=np.ascontiguousarray(np.concatenate(all_indices)),
            weights=np.ascontiguousarray(np.concatenate(all_weights)),
            channels=np.ascontiguousarray(np.concatenate(all_channels)),
        )
        self._bundles[key] = bundle
        return bundle

    @staticmethod
    def _camera_ids_by_side(sim: Any, fly: Any) -> dict[str, int]:
        try:
            camera_ids = tuple(int(value) for value in sim._intern_eye_camera_ids_by_fly[fly.name])
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(f"Fly {fly.name!r} has no FlyGym eye-camera IDs") from exc
        names = list(fly.eyecameraname_to_mjcfcamera.keys())
        if len(names) != len(camera_ids):
            raise RuntimeError(
                f"eye camera name/id mismatch: names={names}, ids={camera_ids}"
            )
        by_name = dict(zip(names, camera_ids, strict=True))
        try:
            return {side: int(by_name[name]) for side, name in _EYE_NAME_BY_SIDE.items()}
        except KeyError as exc:
            raise RuntimeError(f"unexpected FlyBody eye camera names: {names}") from exc

    @staticmethod
    def _base_geom_rgb(model: Any, geom_id: int) -> np.ndarray:
        mat_id = int(model.geom_matid[geom_id])
        if mat_id >= 0:
            return np.asarray(model.mat_rgba[mat_id, :3], dtype=np.float64)
        return np.asarray(model.geom_rgba[geom_id, :3], dtype=np.float64)

    @staticmethod
    def _ground_checker_channel(
        model: Any,
        geom_id: int,
        point: np.ndarray,
        channel: int,
    ) -> float:
        """Approximate FlyGym's achromatic checker material analytically."""

        # FlatGroundWorld uses rgb1=.3, rgb2=.4 and texrepeat=(250,250).
        # The compiled plane half-size is 200 mm in FlyppyWorld.  Mapping this to
        # repeat count gives a 1.6 mm nominal checker period.  Exact OpenGL texture
        # filtering is intentionally left to the raster reference path.
        size = np.asarray(model.geom_size[geom_id, :2], dtype=np.float64)
        mat_id = int(model.geom_matid[geom_id])
        repeat = np.asarray([250.0, 250.0], dtype=np.float64)
        if mat_id >= 0 and hasattr(model, "mat_texrepeat"):
            candidate = np.asarray(model.mat_texrepeat[mat_id, :2], dtype=np.float64)
            if np.all(candidate > 0.0):
                repeat = candidate
        period = np.maximum(2.0 * size / repeat, 1e-9)
        parity = int(math.floor(float(point[0]) / float(period[0]))) + int(
            math.floor(float(point[1]) / float(period[1]))
        )
        return 0.3 if (parity & 1) == 0 else 0.4

    def _surface_channel_values(
        self,
        model: Any,
        origin: np.ndarray,
        directions_world: np.ndarray,
        geom_ids: np.ndarray,
        distances: np.ndarray,
        channels: np.ndarray,
    ) -> np.ndarray:
        values = np.ones(len(geom_ids), dtype=np.float64)  # white FlyGym skybox
        hit_indices = np.flatnonzero((geom_ids >= 0) & (distances >= 0.0))
        for ray_index in hit_indices:
            geom_id = int(geom_ids[ray_index])
            channel = int(channels[ray_index])
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name == "ground_plane":
                point = origin + float(distances[ray_index]) * directions_world[ray_index]
                values[ray_index] = self._ground_checker_channel(
                    model, geom_id, point, channel
                )
            else:
                rgb = self._base_geom_rgb(model, geom_id)
                values[ray_index] = float(rgb[channel])
        np.clip(values, 0.0, 1.0, out=values)
        return values

    def _read_side(
        self,
        sim: Any,
        *,
        side: str,
        camera_id: int,
    ) -> tuple[np.ndarray, int]:
        model = sim.mj_model
        data = sim.mj_data
        fovy = float(model.cam_fovy[camera_id])
        bundle = self._bundle_for_camera(side, camera_id, fovy)

        rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
        origin = np.ascontiguousarray(data.cam_xpos[camera_id], dtype=np.float64)
        directions_world = np.ascontiguousarray(
            bundle.directions_local @ rotation.T,
            dtype=np.float64,
        )
        norms = np.linalg.norm(directions_world, axis=1, keepdims=True)
        directions_world /= np.maximum(norms, 1e-15)

        geom_ids = np.empty(bundle.ray_count, dtype=np.int32)
        distances = np.empty(bundle.ray_count, dtype=np.float64)
        mj.mj_multiRay(
            model,
            data,
            origin,
            directions_world.reshape(-1),
            _EYE_GEOMGROUP,
            1,
            -1,
            geom_ids,
            distances,
            None,
            bundle.ray_count,
            self.cutoff_mm,
        )

        ray_values = self._surface_channel_values(
            model,
            origin,
            directions_world,
            geom_ids,
            distances,
            bundle.channels,
        )
        light = np.zeros(self.num_ommatidia_per_eye, dtype=np.float64)
        np.add.at(
            light,
            bundle.ommatidium_indices,
            bundle.weights * ray_values,
        )
        np.clip(light, 0.0, 1.0, out=light)

        readout = np.zeros((self.num_ommatidia_per_eye, 2), dtype=np.float32)
        pale_mask = np.asarray(self.retina.pale_type_mask, dtype=np.int32)
        required = np.asarray(self.required_by_side[side], dtype=np.int32)
        readout[required, pale_mask[required]] = light[required].astype(np.float32)
        return readout, bundle.ray_count

    def read_eye_readouts(self, sim: Any, fly: Any) -> dict[str, np.ndarray]:
        """Return sparse FlyGym-shaped receptor values without an RGB framebuffer."""

        camera_ids = self._camera_ids_by_side(sim, fly)
        left, left_rays = self._read_side(sim, side="L", camera_id=camera_ids["L"])
        right, right_rays = self._read_side(sim, side="R", camera_id=camera_ids["R"])
        self.last_stats = DirectEyeReadoutStats(
            left_rays=left_rays,
            right_rays=right_rays,
            required_left=len(self.required_by_side["L"]),
            required_right=len(self.required_by_side["R"]),
        )
        return {"L": left, "R": right}
