"""Persistent M11.1 second-voxelization result models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class M11TolerancePolicy(BaseModel):
    """Scale-relative tolerance policy used by the analytic geometry kernel."""

    name: Literal["float64-scale-relative-128eps"] = "float64-scale-relative-128eps"
    machine_epsilon_multiplier: int = 128
    zero_area_rule: str = "area <= length_tolerance^2 is zero"


class M11DomainSetMetrics(BaseModel):
    """Realized-versus-target P32 metrics for one domain and joint set."""

    domain_id: int | None = None
    set_id: int
    voxel_count: int = Field(default=0, ge=0)
    target_mean_p32: float | None = None
    realized_mean_p32: float | None = None
    bias: float | None = None
    mae: float | None = None
    rmse: float | None = None
    mean_relative_error: float | None = None
    error_p50: float | None = None
    error_p95: float | None = None
    boundary_voxel_count: int = Field(default=0, ge=0)
    interior_voxel_count: int = Field(default=0, ge=0)
    boundary_bias: float | None = None
    boundary_mae: float | None = None
    boundary_rmse: float | None = None
    interior_bias: float | None = None
    interior_mae: float | None = None
    interior_rmse: float | None = None


class M11ConservationSummary(BaseModel):
    """Per-fracture disk-area conservation audit."""

    target_area_total: float = Field(default=0.0, ge=0.0)
    analysis_domain_target_area_total: float = Field(default=0.0, ge=0.0)
    generation_area_outside_analysis_total: float = Field(default=0.0, ge=0.0)
    intersection_area_total: float = Field(default=0.0, ge=0.0)
    absolute_error_total: float = Field(default=0.0, ge=0.0)
    relative_error_total: float | None = None
    maximum_absolute_error: float = Field(default=0.0, ge=0.0)
    error_p50: float = Field(default=0.0, ge=0.0)
    error_p95: float = Field(default=0.0, ge=0.0)
    error_p99: float = Field(default=0.0, ge=0.0)
    over_tolerance_count: int = Field(default=0, ge=0)
    over_tolerance_ordinals: list[int] = Field(default_factory=list)
    m10_clipped_area_absolute_difference: float = Field(default=0.0, ge=0.0)
    absolute_area_tolerance: float = Field(default=1e-12, gt=0.0)
    relative_area_tolerance: float = Field(default=1e-10, gt=0.0)


class M11SecondVoxelizationResult(BaseModel):
    """One complete sparse intersection result for an M10 realization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    realization_id: str
    algorithm_version: str = "m11-second-voxelization-1"
    geometry_kernel_version: str = "analytic-circle-convex-polygon-1"
    tolerance_policy: M11TolerancePolicy = Field(default_factory=M11TolerancePolicy)
    voxel_ownership_policy: str = "half-open-min-inclusive-outer-max-inclusive"
    source_m10_realization_id: str
    source_m10_config_hash: str
    source_parameter_field_hash: str
    complete: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    candidate_pair_count: int = Field(default=0, ge=0)
    positive_intersection_count: int = Field(default=0, ge=0)
    rejected_candidate_count: int = Field(default=0, ge=0)
    sparse_bytes: int = Field(default=0, ge=0)
    total_array_bytes: int = Field(default=0, ge=0)
    p32_unresolved_orientation_reference: float = Field(default=0.0, ge=0.0)
    conservation: M11ConservationSummary = Field(default_factory=M11ConservationSummary)
    domain_set_metrics: list[M11DomainSetMetrics] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    arrays: dict[str, np.ndarray] = Field(default_factory=dict, exclude=True)

    @property
    def pair_count(self) -> int:
        """Return the number of positive-area sparse fracture/voxel pairs."""
        values = self.arrays.get("intersection_area")
        return 0 if values is None else int(len(values))


class M11State(BaseModel):
    """Project-owned M11.1 state; later M11 phases must use separate extensions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: list[M11SecondVoxelizationResult] = Field(default_factory=list)
    algorithm_version: str = "m11-second-voxelization-1"
    provenance: dict[str, Any] = Field(default_factory=dict)

    def result_for(self, realization_id: str) -> M11SecondVoxelizationResult | None:
        """Return the stored result for one M10 realization, if present."""
        return next((item for item in self.results if item.source_m10_realization_id == realization_id), None)
