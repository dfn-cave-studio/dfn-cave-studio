"""Typed access models for multi-source geological observations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DomainAssociationSegment(BaseModel):
    """One explicit overlap between an original interval and a structural domain."""

    from_depth: float
    to_depth: float
    domain_id: int | None = None
    assignment_source: str


class FractureSpacingObservation(BaseModel):
    """An interval-average along-hole spacing observation, not a fracture list."""

    observation_id: str
    record_id: str
    hole_id: str
    from_depth: float
    to_depth: float
    fracture_spacing: float = Field(gt=0)
    spacing_unit: str
    measurement_basis: Literal["BOREHOLE_ALONG_HOLE", "TRUE_NORMAL", "SCANLINE_APPARENT"] = (
        "BOREHOLE_ALONG_HOLE"
    )
    derived_p10: float = Field(gt=0)
    derivation: Literal["along_hole_mean_spacing_reciprocal"] = "along_hole_mean_spacing_reciprocal"
    set_id: int | None = None
    domain_segments: list[DomainAssociationSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_interval(self) -> "FractureSpacingObservation":
        if self.to_depth <= self.from_depth:
            raise ValueError("to_depth must be greater than from_depth")
        return self


class AxisPlaneAngleObservation(BaseModel):
    """A plane angle measured relative to the local borehole axis."""

    observation_id: str
    record_id: str
    hole_id: str
    measured_depth: float = Field(ge=0)
    axis_plane_angle: float = Field(ge=0, le=90)
    reference_frame: Literal["BOREHOLE_RELATIVE"] = "BOREHOLE_RELATIVE"
    angle_definition: Literal["acute_angle_between_plane_and_local_borehole_axis"] = (
        "acute_angle_between_plane_and_local_borehole_axis"
    )
    x: float
    y: float
    z: float
    set_id: int | None = None


class OrientationPointObservation(BaseModel):
    """A located orientation measurement without implied finite fracture area."""

    observation_id: str
    point_id: str
    point_key: str
    record_id: str
    source_kind: Literal["POINT_CLOUD", "BOREHOLE_CAMERA"]
    x: float
    y: float
    z: float
    dip: float | None = Field(default=None, ge=0, le=90)
    dip_direction: float | None = Field(default=None, ge=0, lt=360)
    local_set_id: str
    component_type: Literal["LOCAL_DOMINANT_SET", "RANDOM_BACKGROUND"]
    orientation_status: Literal["COMPLETE", "MISSING"]
    joint_spacing_m: float | None = Field(default=None, gt=0)
    measurement_basis: Literal["BOREHOLE_ALONG_HOLE", "TRUE_NORMAL", "SCANLINE_APPARENT"] | None = None
    joint_num: int | None = Field(default=None, gt=0)
    domain_id: int | None = None
    set_id: int | None = None
    site_id: str | None = None
    source: str | None = None
    quality: str | None = None
    observation_kind: Literal["MEASURED", "DOMINANT_SUMMARY"] = "MEASURED"
    calibration_role: Literal["UNASSIGNED"] = "UNASSIGNED"
    source_file: str
    source_row: int = Field(ge=0)
    import_batch_id: str
    audit_metadata: dict[str, Any] = Field(default_factory=dict)

    def approximate_p32_from_true_normal_spacing(self) -> float:
        """Return 1/S only when spacing is explicitly true-normal.

        The method is intentionally never invoked by the Phase 2A generator;
        borehole/scanline apparent spacing requires orientation correction
        before it can be interpreted as volumetric fracture intensity.
        """
        if self.joint_spacing_m is None:
            raise ValueError("joint_spacing_m is required")
        if self.measurement_basis != "TRUE_NORMAL":
            raise ValueError("P32 ≈ 1/spacing is only valid for TRUE_NORMAL measurement_basis")
        return 1.0 / self.joint_spacing_m


class OrientationPointSummary(BaseModel):
    """Auditable spacing-derived point summary; not a measured RQD sample."""

    point_key: str
    input_observation_ids: list[str]
    includes_random: bool
    random_component_status: Literal["REPORTED_PRESENT", "NOT_REPORTED", "REPORTED_ABSENT"] = "NOT_REPORTED"
    jv_estimated: float = Field(ge=0)
    rqd_from_jv: float = Field(ge=0, le=100)
    method_code: Literal["PALMSTROM_2005_SPACING_RECIPROCAL"] = "PALMSTROM_2005_SPACING_RECIPROCAL"
    rounding: str = "none"
    provenance: dict[str, str] = Field(default_factory=dict)
