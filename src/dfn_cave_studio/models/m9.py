"""Persistent models for the M9 local DFN parameter field workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DensityMethod(StrEnum):
    """Supported M9 density models."""

    GLOBAL_CONSTANT = "global_constant"
    IDW = "idw"
    ORDINARY_KRIGING = "ordinary_kriging"


class VariogramModel(StrEnum):
    """Supported isotropic semivariogram models."""

    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"


class VariogramMode(StrEnum):
    """Whether variogram parameters are fitted or supplied by the user."""

    AUTO = "auto"
    MANUAL = "manual"


class NonNegativePolicy(StrEnum):
    """Treatment of negative predictions for non-negative quantities."""

    REJECT = "reject"
    CLIP_WITH_AUDIT = "clip_with_audit"


class KrigingSettings(BaseModel):
    """Configuration for three-dimensional isotropic ordinary kriging."""

    mode: VariogramMode = VariogramMode.AUTO
    model: VariogramModel = VariogramModel.SPHERICAL
    nugget: float = Field(default=0.0, ge=0.0)
    sill: float = Field(default=1.0, gt=0.0)
    range: float = Field(default=100.0, gt=0.0)
    lag_count: int = Field(default=12, ge=3, le=100)
    minimum_neighbors: int = Field(default=3, ge=2)
    maximum_neighbors: int = Field(default=24, ge=2)
    search_radius: float | None = Field(default=None, gt=0.0)
    non_negative_policy: NonNegativePolicy = NonNegativePolicy.REJECT
    regularization: float = Field(default=1e-10, ge=0.0)

    @model_validator(mode="after")
    def validate_parameters(self) -> "KrigingSettings":
        if self.maximum_neighbors < self.minimum_neighbors:
            raise ValueError("maximum_neighbors must be greater than or equal to minimum_neighbors")
        if self.sill < self.nugget:
            raise ValueError("sill must be greater than or equal to nugget")
        return self


class ObservabilityState(StrEnum):
    """Whether an orientation can be observed reliably by a borehole."""

    ADEQUATE = "adequate"
    LOW_OBSERVABILITY = "low_observability"
    NO_DATA = "no_data"
    INSUFFICIENT_ORIENTATION_DATA = "insufficient_orientation_data"


class SizeModelSource(StrEnum):
    """Scientific provenance of fracture-size parameters."""

    FITTED = "fitted"
    SIZE_PROXY = "size_proxy"
    ASSUMED = "assumed"
    USER_DEFINED = "user_defined"
    EXPERIMENTAL = "experimental"


class ValidationState(StrEnum):
    """Availability of an independent validation result."""

    COMPLETE = "complete"
    INSUFFICIENT_VALIDATION = "insufficient_validation"
    NOT_VALIDATED = "not_validated"


class DensitySettings(BaseModel):
    """P10 interval, directional correction, and interpolation settings."""

    interval_mode: str = Field(default="fixed", pattern="^(fixed|domain)$")
    interval_length: float = Field(default=10.0, gt=0.0)
    method: DensityMethod = DensityMethod.GLOBAL_CONSTANT
    power: float = Field(default=2.0, gt=0.0)
    search_radius: float | None = Field(default=None, gt=0.0)
    min_neighbors: int = Field(default=1, ge=1)
    max_neighbors: int = Field(default=12, ge=1)
    anisotropy_x: float = Field(default=1.0, gt=0.0)
    anisotropy_y: float = Field(default=1.0, gt=0.0)
    anisotropy_z: float = Field(default=1.0, gt=0.0)
    global_fallback: bool = False
    monte_carlo_samples: int = Field(default=20_000, ge=100)
    low_observability_threshold: float = Field(default=0.05, gt=0.0, le=1.0)
    kriging: KrigingSettings = Field(default_factory=KrigingSettings)

    @model_validator(mode="after")
    def validate_neighbor_range(self) -> "DensitySettings":
        """Require a coherent nearest-neighbour range."""
        if self.max_neighbors < self.min_neighbors:
            raise ValueError("max_neighbors must be greater than or equal to min_neighbors")
        return self


class P10Interval(BaseModel):
    """Observed linear fracture intensity in one borehole interval."""

    hole_id: str
    from_depth: float
    to_depth: float
    domain_id: int | None = None
    set_id: int | None = None
    observation_count: int = Field(ge=0)
    sample_length: float = Field(ge=0.0)
    p10: float | None = Field(default=None, ge=0.0)
    role: str = "calibration"
    data_state: str = "modeled_value"
    center_x: float | None = None
    center_y: float | None = None
    center_z: float | None = None
    segment_directions: list[tuple[float, float, float, float]] = Field(default_factory=list)
    source: str = "formal_fracture_observations"
    provenance: dict[str, Any] = Field(default_factory=dict)
    full_orientation_count: int = Field(default=0, ge=0)
    dip_only_count: int = Field(default=0, ge=0)
    unassigned_dip_only_count: int = Field(default=0, ge=0)


class P32Estimate(BaseModel):
    """Poisson maximum-likelihood P32 estimate for one domain/set."""

    domain_id: int | None = None
    set_id: int
    fracture_count: int = Field(ge=0)
    raw_sample_length: float = Field(ge=0.0)
    effective_sample_length: float = Field(ge=0.0)
    mean_exposure: float = Field(ge=0.0, le=1.0)
    p32: float | None = Field(default=None, ge=0.0)
    standard_error: float | None = Field(default=None, ge=0.0)
    ci95_low: float | None = Field(default=None, ge=0.0)
    ci95_high: float | None = Field(default=None, ge=0.0)
    observability: ObservabilityState = ObservabilityState.NO_DATA
    random_seed: int
    calibration_holes: list[str] = Field(default_factory=list)
    method: str = "poisson_direction_corrected_mle"
    full_orientation_count: int = Field(default=0, ge=0)
    dip_only_count: int = Field(default=0, ge=0)
    orientation_model_source: str | None = None
    eligibility_status: str = "no_data"
    provenance: dict[str, Any] = Field(default_factory=dict)


class DomainOrientationModel(BaseModel):
    """Calibration-only Fisher orientation statistics for one domain/set."""

    domain_id: int | None = None
    set_id: int
    mean_dip_direction: float = Field(ge=0.0, lt=360.0)
    mean_dip: float = Field(ge=0.0, le=90.0)
    kappa: float = Field(gt=0.0)
    observation_count: int = Field(ge=1)
    source: str = "calibration_formal_fractures"
    full_orientation_count: int = Field(default=0, ge=0)
    dip_only_count: int = Field(default=0, ge=0)
    orientation_fit_eligible: bool = True


class SizeFitCandidate(BaseModel):
    """One candidate size-distribution fit."""

    distribution_type: str
    parameters: dict[str, float]
    log_likelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    sample_count: int = Field(ge=0)
    converged: bool = False
    optimizer_message: str = ""
    parameter_count: int = Field(ge=0)


class SizeModel(BaseModel):
    """Selected fracture-size model with explicit provenance."""

    domain_id: int | None = None
    set_id: int
    distribution_type: str = "fixed"
    parameters: dict[str, float] = Field(default_factory=lambda: {"radius": 1.0})
    min_radius: float = Field(default=1.0, gt=0.0)
    max_radius: float = Field(default=1.000001, gt=0.0)
    mean_radius: float = Field(default=1.0, gt=0.0)
    mean_squared_radius: float = Field(default=1.0, gt=0.0)
    source: SizeModelSource = SizeModelSource.ASSUMED
    measurement_field: str | None = None
    sample_count: int = Field(default=0, ge=0)
    candidates: list[SizeFitCandidate] = Field(default_factory=list)
    fit_status: str = "not_fitted"
    converged: bool | None = None
    optimizer_message: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class ParameterFieldMetadata(BaseModel):
    """Small JSON metadata stored alongside compressed parameter arrays."""

    shape: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    field_names: list[str]
    set_ids: list[int]
    density_method: DensityMethod
    random_seed: int
    coordinate_system: str = "X=Easting,Y=Northing,Z=Elevation"
    length_unit: str = "m"
    p32_unit: str = "m^-1"
    estimated_bytes: int = Field(ge=0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provenance: dict[str, Any] = Field(default_factory=dict)


class ValidationIntervalResult(BaseModel):
    """Observed-versus-predicted result for a held-out interval."""

    hole_id: str
    from_depth: float
    to_depth: float
    domain_id: int | None = None
    set_id: int
    observed_count: int = Field(ge=0)
    observed_p10: float | None = Field(default=None, ge=0.0)
    predicted_p10: float | None = Field(default=None, ge=0.0)
    predicted_p32: float | None = Field(default=None, ge=0.0)
    absolute_error: float | None = Field(default=None, ge=0.0)
    relative_error: float | None = None
    calibration_or_validation: str = "validation"
    data_state: str = "modeled_value"


class ValidationSummary(BaseModel):
    """Aggregate independent-validation metrics."""

    state: ValidationState = ValidationState.NOT_VALIDATED
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None
    r_squared: float | None = None
    correlation: float | None = None
    valid_interval_count: int = 0
    no_data_interval_count: int = 0


class VariogramLag(BaseModel):
    """One auditable experimental-variogram distance bin."""

    distance: float = Field(ge=0.0)
    pair_count: int = Field(ge=0)
    experimental_semivariance: float | None = Field(default=None, ge=0.0)
    fitted_semivariance: float | None = Field(default=None, ge=0.0)


class VariogramDiagnostics(BaseModel):
    """Fitted model and numerical diagnostics for one scalar field."""

    mode: VariogramMode
    model: VariogramModel
    nugget: float = Field(ge=0.0)
    sill: float = Field(gt=0.0)
    range: float = Field(gt=0.0)
    lags: list[VariogramLag] = Field(default_factory=list)
    fit_status: str = "manual"
    optimizer_message: str = ""
    sample_count: int = Field(ge=0)
    pair_count: int = Field(ge=0)
    regularized_solve_count: int = Field(default=0, ge=0)
    pseudoinverse_solve_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ScalarParameterSample(BaseModel):
    """One continuous-parameter interval located along a real borehole trajectory."""

    sample_id: str
    borehole_id: str
    from_depth: float = Field(ge=0.0)
    to_depth: float = Field(gt=0.0)
    parameter_name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    source_dataset: str = ""
    quality_flag: str = ""
    midpoint_x: float
    midpoint_y: float
    midpoint_z: float
    domain_id: int | None = None
    role: str = "calibration"

    @model_validator(mode="after")
    def validate_interval(self) -> "ScalarParameterSample":
        if self.to_depth <= self.from_depth:
            raise ValueError("to_depth must be greater than from_depth")
        return self


class ScalarValidationResult(BaseModel):
    """Independent prediction at one held-out borehole interval midpoint."""

    sample_id: str
    borehole_id: str
    from_depth: float = 0.0
    to_depth: float = 0.0
    domain_id: int | None = None
    observed: float
    predicted: float | None = None
    residual: float | None = None
    kriging_variance: float | None = Field(default=None, ge=0.0)
    standardized_residual: float | None = None
    prediction_interval_low: float | None = None
    prediction_interval_high: float | None = None
    status: str = "predicted"


class ScalarValidationSummary(BaseModel):
    """Validation metrics without interval-level spatial leakage."""

    sample_count: int = Field(default=0, ge=0)
    predicted_count: int = Field(default=0, ge=0)
    mean_error: float | None = None
    mae: float | None = Field(default=None, ge=0.0)
    rmse: float | None = Field(default=None, ge=0.0)
    r_squared: float | None = None


class ScalarFieldMetadata(BaseModel):
    """Persistent metadata for one independently stored physical-parameter field."""

    field_id: str
    parameter_name: str
    unit: str
    method: DensityMethod
    shape: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    array_names: list[str]
    config_hash: str
    algorithm_version: str = "m9-scalar-kriging-1"
    software_version: str = "0.11.0"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    variograms: dict[str, VariogramDiagnostics] = Field(default_factory=dict)
    rejected_voxel_count: int = Field(default=0, ge=0)
    clipped_voxel_count: int = Field(default=0, ge=0)
    pre_clip_minimum: float | None = None
    pre_adjustment_minimum: float | None = None
    pre_adjustment_maximum: float | None = None
    clipped_total_change: float = Field(default=0.0, ge=0.0)
    parameter_bounds: tuple[float | None, float | None] = (None, None)
    adjustment_policy: NonNegativePolicy = NonNegativePolicy.REJECT
    provenance: dict[str, Any] = Field(default_factory=dict)


class ScalarFieldResult(BaseModel):
    """Session/project result whose large arrays are persisted outside JSON."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: ScalarFieldMetadata
    settings: DensitySettings
    arrays: dict[str, np.ndarray] = Field(default_factory=dict, exclude=True)
    validation_results: list[ScalarValidationResult] = Field(default_factory=list)
    validation_summary: ScalarValidationSummary = Field(default_factory=ScalarValidationSummary)


class M9State(BaseModel):
    """The single project-owned state container for all M9 outputs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    density_settings: DensitySettings = Field(default_factory=DensitySettings)
    p10_intervals: list[P10Interval] = Field(default_factory=list)
    p32_estimates: list[P32Estimate] = Field(default_factory=list)
    orientation_models: list[DomainOrientationModel] = Field(default_factory=list)
    size_models: list[SizeModel] = Field(default_factory=list)
    parameter_field_metadata: ParameterFieldMetadata | None = None
    parameter_field_arrays: dict[str, np.ndarray] = Field(default_factory=dict, exclude=True)
    validation_results: list[ValidationIntervalResult] = Field(default_factory=list)
    validation_summary: ValidationSummary = Field(default_factory=ValidationSummary)
    random_seed: int = 42
    provenance: dict[str, Any] = Field(default_factory=dict)
    scalar_samples: list[ScalarParameterSample] = Field(default_factory=list)
    scalar_fields: list[ScalarFieldResult] = Field(default_factory=list)
