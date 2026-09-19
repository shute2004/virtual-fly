from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from virtual_fly.runtime.vision import VisionRuntimeConfig


class VisionRuntimeConfigTests(unittest.TestCase):
    def test_production_values_can_be_made_explicit(self) -> None:
        config = VisionRuntimeConfig.from_environment(
            default_mode="raster",
            default_rays=7,
            environ={"VF_FLYPPY_VISION_MODE": "direct-ray", "VF_FLYPPY_OMMATIDIA_RAYS": "13"},
        )
        self.assertEqual(config.mode, "direct-ray")
        self.assertEqual(config.rays_per_ommatidium, 13)
        worker = config.apply_to_worker_mapping({"seed": 0})
        self.assertEqual(worker["vision_mode"], "direct-ray")
        self.assertEqual(worker["vision_rays_per_ommatidium"], 13)

    def test_explicit_worker_values_do_not_depend_on_environment(self) -> None:
        config = VisionRuntimeConfig.from_worker_mapping(
            {"vision_mode": "direct-ray", "vision_rays_per_ommatidium": 13},
            allow_environment_fallback=False,
        )
        self.assertEqual(config, VisionRuntimeConfig("direct-ray", 13))

    def test_alias_normalization_preserves_historical_cli_behavior(self) -> None:
        self.assertEqual(VisionRuntimeConfig.normalized("ray", 7).mode, "direct-ray")
        self.assertEqual(VisionRuntimeConfig.normalized("flygym", 7).mode, "raster")


    def test_packed_main_passes_runtime_semantics_before_execution(self) -> None:
        from virtual_fly.training import population_packed

        with patch.dict(
            os.environ,
            {"VF_FLYPPY_VISION_MODE": "direct-ray", "VF_FLYPPY_OMMATIDIA_RAYS": "13"},
            clear=False,
        ), patch.object(population_packed.trainer, "main", return_value=1) as mocked:
            self.assertEqual(population_packed.main(), 1)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["body_runtime"], "packed-process")
        self.assertEqual(kwargs["vision_mode_override"], "direct-ray")
        self.assertEqual(kwargs["vision_rays_override"], 13)

if __name__ == "__main__":
    unittest.main()
