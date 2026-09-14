"""Physical-scale specifications for virtual-fly.

Two scales are intentionally kept separate:

* ``CANONICAL_FLY`` is the historical/provisional Flyppy-v2 scale used by the
  already-recorded v2 experiments.  It is retained unchanged for reproducibility.
* ``FLYBODY_REFERENCE`` records the published physical scale of the FlyBody model
  whose flight physics were calibrated by Vaxenburg et al. (2025).

The distinction matters because the earlier v2 implementation incorrectly treated
2.75 mm / 1.09 mg literature means from another Drosophila measurement as if they
were FlyBody's own calibration values.  FlyBody reports a 2.97 mm model body length,
6.04 mm wing span and 0.983 mg total mass.  A successor flight task should be
validated against ``FLYBODY_REFERENCE`` before its environment geometry is locked.

Environment dimensions are task-design choices expressed relative to the v2 body
length.  They are not biological measurements.  Keeping them in one immutable
specification prevents the visual world, analytic course and MuJoCo collision world
from silently drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlyPhysicalSpec:
    morphology: str
    body_length_mm: float
    wing_length_mm: float
    total_mass_mg: float
    neural_sex: str
    morphology_sex: str
    provenance: str

    @property
    def total_mass_g(self) -> float:
        return self.total_mass_mg / 1000.0


@dataclass(frozen=True)
class FlyBodyPublishedReference:
    body_length_mm: float
    wing_span_mm: float
    total_mass_mg: float
    morphology_sex: str
    provenance: str

    @property
    def total_mass_g(self) -> float:
        return self.total_mass_mg / 1000.0


@dataclass(frozen=True)
class FlyppyGeometrySpec:
    body_length_mm: float
    corridor_height_body_lengths: float
    gate_gap_body_lengths: float
    first_gate_body_lengths: float
    gate_spacing_body_lengths: float
    lateral_half_width_body_lengths: float
    gate_half_thickness_body_lengths: float
    center_margin_body_lengths: float

    @property
    def corridor_low_z_mm(self) -> float:
        return 0.0

    @property
    def corridor_high_z_mm(self) -> float:
        return self.body_length_mm * self.corridor_height_body_lengths

    @property
    def gate_gap_height_mm(self) -> float:
        return self.body_length_mm * self.gate_gap_body_lengths

    @property
    def gate_gap_half_height_mm(self) -> float:
        return self.gate_gap_height_mm / 2.0

    @property
    def first_gate_x_mm(self) -> float:
        return self.body_length_mm * self.first_gate_body_lengths

    @property
    def gate_spacing_mm(self) -> float:
        return self.body_length_mm * self.gate_spacing_body_lengths

    @property
    def lateral_half_width_mm(self) -> float:
        return self.body_length_mm * self.lateral_half_width_body_lengths

    @property
    def gate_half_thickness_mm(self) -> float:
        return self.body_length_mm * self.gate_half_thickness_body_lengths

    @property
    def center_margin_mm(self) -> float:
        return self.body_length_mm * self.center_margin_body_lengths


# Published FlyBody model scale used by its flight-physics calibration.  The paper
# reports head/thorax/abdomen/leg/wing masses summing to 0.983 mg, model body length
# 0.297 cm and wing span 0.604 cm.
FLYBODY_REFERENCE = FlyBodyPublishedReference(
    body_length_mm=2.97,
    wing_span_mm=6.04,
    total_mass_mg=0.983,
    morphology_sex="female",
    provenance=(
        "Vaxenburg et al. 2025 FlyBody measured segment masses and model dimensions; "
        "flight fluid coefficients were calibrated on this physical model"
    ),
)


# Historical Flyppy-v2 provisional scale.  Do not reinterpret these values as
# FlyBody-native calibration values.  They are retained so old v2 checkpoints,
# reports and gate geometry remain exactly reproducible.
CANONICAL_FLY = FlyPhysicalSpec(
    morphology="FlyGym 2.1.0 FlyBody with historical Flyppy-v2 provisional scale",
    body_length_mm=2.75,
    wing_length_mm=2.13,
    total_mass_mg=1.09,
    neural_sex="male",
    morphology_sex="female",
    provenance=(
        "historical Flyppy-v2 provisional adult-female literature means; not the "
        "published FlyBody flight-calibration scale"
    ),
)


# Version-2 Flyppy world, preserved as an experimental baseline.  These are task
# design dimensions, not measured Drosophila ecology.  A future corrected flight
# environment should be versioned rather than silently changing these values.
FLYPPY_GEOMETRY = FlyppyGeometrySpec(
    body_length_mm=CANONICAL_FLY.body_length_mm,
    corridor_height_body_lengths=6.0,       # 16.50 mm
    gate_gap_body_lengths=2.5,              # 6.875 mm
    first_gate_body_lengths=4.0,            # 11.00 mm
    gate_spacing_body_lengths=4.0,          # 11.00 mm
    lateral_half_width_body_lengths=3.0,    # 8.25 mm each side
    gate_half_thickness_body_lengths=0.10,  # 0.275 mm; full 0.55 mm
    center_margin_body_lengths=0.50,        # 1.375 mm
)
