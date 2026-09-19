from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))


class CompatibilityShimTests(unittest.TestCase):
    def test_course_shim_exports_package_class(self) -> None:
        shim = importlib.import_module("flyppy_course")
        from virtual_fly.embodiment.course import FlyppyCourse

        self.assertIs(shim.FlyppyCourse, FlyppyCourse)

    def test_v7_shim_exports_package_classes(self) -> None:
        shim = importlib.import_module("flybody_v7_adapter")
        from virtual_fly.embodiment.body.v7 import (
            FlyBodyV7MuscleAdapter,
            FlyBodyV7NeuromuscularAdapter,
        )

        self.assertIs(shim.FlyBodyV7MuscleAdapter, FlyBodyV7MuscleAdapter)
        self.assertIs(shim.FlyBodyV7NeuromuscularAdapter, FlyBodyV7NeuromuscularAdapter)

    def test_neural_bridge_shim_exports_runtime_client(self) -> None:
        shim = importlib.import_module("population_neural_bridge_client")
        from virtual_fly.runtime.neural_bridge import PopulationNeuralBridgeClient

        self.assertIs(shim.PopulationNeuralBridgeClient, PopulationNeuralBridgeClient)

    def test_body_worker_shim_exports_runtime_worker(self) -> None:
        shim = importlib.import_module("flyppy_body_worker")
        from virtual_fly.runtime.body_worker import FlyppyBodyProcess

        self.assertIs(shim.FlyppyBodyProcess, FlyppyBodyProcess)


if __name__ == "__main__":
    unittest.main()
