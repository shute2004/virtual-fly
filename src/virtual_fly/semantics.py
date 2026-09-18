"""Version and semantic contracts shared across virtual-fly runtime layers.

These constants name existing behavior; changing a value here is a scientific
compatibility change, not a cosmetic refactor.
"""
from __future__ import annotations

MALECNS_DATASET = "male-cns:v1.0"
MALECNS_RUNTIME_SEMANTICS = "male-cns-v1-class-dan-v1"
MODULATOR_ROLE_DEFINITION = "consensus_nt=dopamine AND released annotation class=DAN"
NEURAL_RUNTIME_SEMANTICS = "signed-activity-local-dynamics-v1"
PLASTICITY_SEMANTICS = "local-three-factor-dopamine-eligibility-v1"
POPULATION_CHECKPOINT_SEMANTICS = "global-weights-only-v1"
POPULATION_NEURAL_STEP_SEMANTICS = "aggregate-slot-neural-step-count-v1"

HALTERE_FULL_KIND = "male-cns-haltere-campaniform-afferents"
HALTERE_TIMING_KIND = "male-cns-haltere-timing-afferents-inferred-v1"
HALTERE_EXPECTED_COUNTS = {HALTERE_FULL_KIND: 195, HALTERE_TIMING_KIND: 97}
HALTERE_TRANSDUCTIONS = ("angular-acceleration-v1", "interaction-load-v2")

NEUTRAL_TRIM_KIND = "flybody-task-independent-neutral-wing-trim"
NEUTRAL_TRIM_SEMANTICS = "flybody-neutral-trim-v1"

BODY_VERSIONS = ("v3", "v4", "v5", "v6", "v7", "v8")
PRODUCTION_ENVIRONMENT_VERSIONS = ("v3", "v4", "v5", "v6", "v7")
ALL_ENVIRONMENT_VERSIONS = ("v1", "v2", *PRODUCTION_ENVIRONMENT_VERSIONS)
VISION_MODES = ("raster", "direct-ray")
PRODUCTION_DIRECT_RAY_COUNT = 13

# Historical lineage. This is deliberately not a numeric successor chain.
BODY_LINEAGE: dict[str, tuple[str, ...]] = {
    "v3": (),
    "v4": ("v3",),
    "v5": ("v4",),
    "v6": ("v4",),
    "v7": ("v4",),
    # v8 combines v6 mechanics with the v7 neutral-trim mixin.
    "v8": ("v6", "v7:neutral-trim-mixin"),
}

SHARED_WEIGHT_COMMIT_SEMANTICS = (
    "episode-local additive+clamp transaction rebased onto latest global weight"
)
