"""Explicit runtime configuration for compound-eye acquisition."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

from virtual_fly.semantics import VISION_MODES


@dataclass(frozen=True)
class VisionRuntimeConfig:
    mode: str
    rays_per_ommatidium: int

    @classmethod
    def normalized(cls, mode: str, rays_per_ommatidium: int) -> "VisionRuntimeConfig":
        aliases = {
            "raster": "raster",
            "flygym": "raster",
            "reference": "raster",
            "direct": "direct-ray",
            "ray": "direct-ray",
            "direct-ray": "direct-ray",
        }
        resolved = aliases.get(str(mode).strip().lower())
        if resolved not in VISION_MODES:
            raise ValueError("vision mode must be raster or direct-ray")
        rays = int(rays_per_ommatidium)
        if rays < 1:
            raise ValueError("rays_per_ommatidium must be a positive integer")
        return cls(resolved, rays)

    @classmethod
    def from_environment(
        cls,
        *,
        default_mode: str = "raster",
        default_rays: int = 7,
        environ: Mapping[str, str] | None = None,
    ) -> "VisionRuntimeConfig":
        source = os.environ if environ is None else environ
        mode = source.get("VF_FLYPPY_VISION_MODE", default_mode)
        raw_rays = source.get("VF_FLYPPY_OMMATIDIA_RAYS", str(default_rays))
        try:
            rays = int(raw_rays)
        except ValueError as exc:
            raise ValueError("VF_FLYPPY_OMMATIDIA_RAYS must be a positive integer") from exc
        return cls.normalized(mode, rays)

    @classmethod
    def from_worker_mapping(
        cls,
        config: Mapping[str, object],
        *,
        allow_environment_fallback: bool = True,
    ) -> "VisionRuntimeConfig":
        mode = config.get("vision_mode")
        rays = config.get("vision_rays_per_ommatidium")
        if mode is not None and rays is not None:
            return cls.normalized(str(mode), int(rays))
        if allow_environment_fallback:
            # Historical diagnostic callers may still construct packed workers
            # directly. Production population_packed always passes explicit values.
            return cls.from_environment()
        raise ValueError("packed worker requires explicit vision runtime configuration")

    def apply_to_worker_mapping(self, config: Mapping[str, object]) -> dict[str, object]:
        result = dict(config)
        result["vision_mode"] = self.mode
        result["vision_rays_per_ommatidium"] = self.rays_per_ommatidium
        return result
