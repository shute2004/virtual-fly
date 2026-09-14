"""Canonical physical scale for the virtual-fly embodiment.

The numerical body is FlyGym 2.1.0 FlyBody, whose morphology is based on a
micro-CT scan of an adult female *Drosophila melanogaster*.  MaleCNS is a male
connectome, so morphology/CNS sex mismatch is explicit rather than hidden by an
unsupported uniform rescaling.

Environment dimensions are task-design choices expressed relative to the
canonical body length.  They are not biological measurements.  Keeping them in
one immutable specification prevents the visual world, analytic course and
MuJoCo collision world from silently drifting apart.
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


# Morphometric values are adult-female means from published D. melanogaster
# measurements.  FlyBody 2.1.0 itself is derived from an adult-female micro-CT
# morphology, so these values define the physical individual represented by the
# current body model.  They do NOT claim the MaleCNS connectome is female.
CANONICAL_FLY = FlyPhysicalSpec(
    morphology="FlyGym 2.1.0 FlyBody adult-female morphology",
    body_length_mm=2.75,
    wing_length_mm=2.13,
    total_mass_mg=1.09,
    neural_sex="male",
    morphology_sex="female",
    provenance=(
        "literature body/wing length and body mass; FlyGym 2.1.0 FlyBody "
        "morphology provenance"
    ),
)

# Version-2 Flyppy world.  These are explicit task-design dimensions, not
# measured Drosophila ecology.  The 2.5-body-length vertical opening is chosen
# to stop the old 4 mm aperture from being a nearly body-sized slit while still
# requiring substantial altitude control.  A runtime envelope diagnostic must
# verify that the complete swept FlyBody fits with positive clearance.
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
