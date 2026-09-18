from __future__ import annotations

import unittest

from virtual_fly.semantics import (
    BODY_LINEAGE,
    BODY_VERSIONS,
    POPULATION_CHECKPOINT_SEMANTICS,
    POPULATION_NEURAL_STEP_SEMANTICS,
    PRODUCTION_DIRECT_RAY_COUNT,
)


class SemanticContractTests(unittest.TestCase):
    def test_body_lineage_is_not_interpreted_as_linear(self) -> None:
        self.assertEqual(BODY_VERSIONS, ("v3", "v4", "v5", "v6", "v7", "v8"))
        self.assertEqual(BODY_LINEAGE["v5"], ("v4",))
        self.assertEqual(BODY_LINEAGE["v6"], ("v4",))
        self.assertEqual(BODY_LINEAGE["v7"], ("v4",))
        self.assertEqual(BODY_LINEAGE["v8"], ("v6", "v7:neutral-trim-mixin"))

    def test_population_checkpoint_contract_is_stable(self) -> None:
        self.assertEqual(POPULATION_CHECKPOINT_SEMANTICS, "global-weights-only-v1")
        self.assertEqual(POPULATION_NEURAL_STEP_SEMANTICS, "aggregate-slot-neural-step-count-v1")

    def test_production_direct_ray_count_remains_k13(self) -> None:
        self.assertEqual(PRODUCTION_DIRECT_RAY_COUNT, 13)


if __name__ == "__main__":
    unittest.main()
