from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
