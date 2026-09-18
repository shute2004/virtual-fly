#!/usr/bin/env python3
"""Direct ommatidial sensor variant that excludes the eye-camera owner body.

The FlyGym eye renderer does not visually occlude itself with the small camera
marker geometry. MuJoCo ray queries, however, see that marker unless the owning
camera body is explicitly excluded. This subclass keeps the existing receptive
field and photometry implementation unchanged and only aligns the visibility
query with the eye camera by passing ``model.cam_bodyid[camera_id]`` as
``bodyexclude`` to ``mj_multiRay``.
"""

from __future__ import annotations

from typing import Any

import mujoco as mj
import numpy as np

from virtual_fly.embodiment.direct_vision import DirectOmmatidialSensor, _EYE_GEOMGROUP


class BodyExcludedDirectOmmatidialSensor(DirectOmmatidialSensor):
    """Image-free ommatidial sensor without self-hits on eye-camera markers."""

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
        camera_body_id = int(model.cam_bodyid[camera_id])
        mj.mj_multiRay(
            model,
            data,
            origin,
            directions_world.reshape(-1),
            _EYE_GEOMGROUP,
            1,
            camera_body_id,
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
