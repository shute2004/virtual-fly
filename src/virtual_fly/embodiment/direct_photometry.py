#!/usr/bin/env python3
"""Photometric models for the image-free Flyppy compound-eye sensor.

The base :mod:`direct_ommatidia_sensor` implementation deliberately separates
visibility from photometry.  This module adds a renderer-independent headlight
Lambert model while keeping the same weighted ommatidial ray bundles and
``mj_multiRay`` visibility queries.

This is still an image-free sensor: no RGB framebuffer, fisheye image, or pixel
aggregation is created at runtime.
"""

from __future__ import annotations

from typing import Any

import mujoco as mj
import numpy as np

from virtual_fly.embodiment.direct_vision import (
    DirectOmmatidialSensor,
    _EYE_GEOMGROUP,
)


class HeadlightDirectOmmatidialSensor(DirectOmmatidialSensor):
    """Direct ommatidial sensor with MuJoCo-headlight Lambert photometry.

    MuJoCo's default visualization headlight has ambient and diffuse terms.  The
    OpenGL renderer also includes specular/texture/filtering details; this model
    intentionally does not recreate the renderer.  It uses the physically
    relevant subset available directly from ray queries: hit material/albedo,
    surface normal, and incidence direction.
    """

    @staticmethod
    def _headlight_terms(model: Any) -> tuple[np.ndarray, np.ndarray, bool]:
        headlight = model.vis.headlight
        ambient = np.asarray(headlight.ambient, dtype=np.float64).reshape(3)
        diffuse = np.asarray(headlight.diffuse, dtype=np.float64).reshape(3)
        return ambient, diffuse, bool(headlight.active)

    def _surface_channel_values_lit(
        self,
        model: Any,
        origin: np.ndarray,
        directions_world: np.ndarray,
        geom_ids: np.ndarray,
        distances: np.ndarray,
        normals: np.ndarray,
        channels: np.ndarray,
    ) -> np.ndarray:
        # The FlyGym world skybox is white.  Rays that miss geometry therefore
        # remain white and are not shaded by scene lighting.
        values = np.ones(len(geom_ids), dtype=np.float64)
        ambient, diffuse, active = self._headlight_terms(model)
        hit_indices = np.flatnonzero((geom_ids >= 0) & (distances >= 0.0))
        for ray_index in hit_indices:
            geom_id = int(geom_ids[ray_index])
            channel = int(channels[ray_index])
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""

            if name == "ground_plane":
                point = origin + float(distances[ray_index]) * directions_world[ray_index]
                albedo = self._ground_checker_channel(model, geom_id, point, channel)
            else:
                rgb = self._base_geom_rgb(model, geom_id)
                albedo = float(rgb[channel])

            if not active:
                values[ray_index] = albedo
                continue

            normal = np.asarray(normals[ray_index], dtype=np.float64)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 1e-15:
                normal = normal / normal_norm
            # The headlight is attached to the viewing camera.  At the hit point,
            # light arrives from the camera direction, i.e. opposite the cast ray.
            incidence = max(
                0.0,
                float(np.dot(normal, -directions_world[ray_index])),
            )
            illumination = float(ambient[channel]) + float(diffuse[channel]) * incidence
            values[ray_index] = albedo * illumination

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
        normals = np.empty((bundle.ray_count, 3), dtype=np.float64)
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
            normals.reshape(-1),
            bundle.ray_count,
            self.cutoff_mm,
        )

        ray_values = self._surface_channel_values_lit(
            model,
            origin,
            directions_world,
            geom_ids,
            distances,
            normals,
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
