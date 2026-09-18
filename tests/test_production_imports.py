from __future__ import annotations

import unittest


class ProductionImportTests(unittest.TestCase):
    def test_production_orchestration_modules_import(self) -> None:
        from virtual_fly.training import population_packed, population_process

        self.assertTrue(callable(population_process.main))
        self.assertTrue(callable(population_packed.main))

    def test_production_runtime_modules_import(self) -> None:
        from virtual_fly.runtime import body_worker, neural_bridge, packed_body_worker

        self.assertTrue(hasattr(body_worker, "spawn_body_processes"))
        self.assertTrue(hasattr(packed_body_worker, "spawn_packed_body_processes"))
        self.assertTrue(hasattr(neural_bridge, "PopulationNeuralBridgeClient"))


if __name__ == "__main__":
    unittest.main()
