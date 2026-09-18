"""Public curriculum API for Flyppy training.

Implementation is split by policy family. The neural learning rule lives in the
CNS runtime; curriculum code only selects episode reset conditions and records
training outcomes.
"""
from virtual_fly.training.curriculum_core import (
    AdaptiveCurriculumConfig,
    SpawnCondition,
    current_adaptive_condition,
    initial_state,
    load_state,
    move_toward,
    record_adaptive_result,
)
from virtual_fly.training.curriculum_boundary import (
    BoundaryBandConfig,
    boundary_condition_for_attempt,
    boundary_frontier_role,
    boundary_target_gates,
    current_boundary_condition,
    ensure_boundary_state,
    frontier_focus_condition,
    frontier_focus_level,
    next_boundary_attempt_for_group,
    record_boundary_result,
)
from virtual_fly.training.curriculum_gate_height import (
    GateHeightCurriculumConfig,
    ensure_gate_height_state,
    gate_height_condition_for_attempt,
    gate_height_role,
    next_gate_height_attempt_for_group,
    record_gate_height_result,
)

__all__ = [
    "AdaptiveCurriculumConfig", "BoundaryBandConfig", "GateHeightCurriculumConfig",
    "SpawnCondition", "move_toward", "initial_state", "load_state",
    "current_adaptive_condition", "record_adaptive_result",
    "ensure_boundary_state", "boundary_condition_for_attempt",
    "next_boundary_attempt_for_group", "current_boundary_condition",
    "boundary_frontier_role", "frontier_focus_level", "frontier_focus_condition",
    "boundary_target_gates", "record_boundary_result",
    "ensure_gate_height_state", "next_gate_height_attempt_for_group",
    "gate_height_role", "gate_height_condition_for_attempt", "record_gate_height_result",
]
