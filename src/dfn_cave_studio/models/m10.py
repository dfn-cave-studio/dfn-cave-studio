"""Persistent models for M10 conditional explicit DFN generation."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class M10FractureSource(StrEnum):
    """Auditable source of an explicit fracture."""

    STOCHASTIC = "STOCHASTIC"
    CONDITIONED_OBSERVATION = "CONDITIONED_OBSERVATION"
    DETERMINISTIC_STRUCTURE = "DETERMINISTIC_STRUCTURE"


class LocalP32Status(StrEnum):
    """Scientific status of M10 voxel-level generated P32."""

    CENTER_ASSIGNED_PRELIMINARY = "CENTER_ASSIGNED_PRELIMINARY"


class M10GenerationConfig(BaseModel):
    """Configuration shared by one or more explicit-DFN realizations."""

    base_seed: int = 42
    realization_count: int = Field(default=1, ge=1, le=100)
    seed_strategy: str = Field(default="base_plus_index", pattern="^base_plus_index$")
    condition_calibration_observations: bool = True
    deterministic_structures_reduce_budget: bool = False
    experimental_size_models_confirmed: bool = False
    maximum_fractures: int = Field(default=1_000_000, ge=1)
    memory_warning_bytes: int = Field(default=512 * 1024**2, ge=1)
    disk_sides: int = Field(default=32, ge=12, le=128)
    validation_warning_acknowledged: bool = False
    worker_count: Literal[1, 2, 4, 8] = 1
    fracture_batch_size: int = Field(default=65_536, ge=1_024, le=1_000_000)
    memory_safety_fraction: float = Field(default=0.70, gt=0.0, le=0.95)
    size_threshold_mode: Literal["auto", "manual"] = "auto"
    small_area_share: float = 0.10
    medium_large_cumulative_share: float = 0.70
    manual_small_medium_radius: float = 0.5
    manual_medium_large_radius: float = 2.0
    enabled_size_classes: list[Literal["SMALL", "MEDIUM", "LARGE"]] = Field(
        default_factory=lambda: ["MEDIUM", "LARGE"]
    )
    retain_outside_deterministic: bool = False

    @model_validator(mode="after")
    def validate_multiscale(self) -> "M10GenerationConfig":
        """Validate resolution thresholds without silently changing user input."""
        if self.size_threshold_mode == "auto":
            if not 0.0 < self.small_area_share < self.medium_large_cumulative_share < 1.0:
                raise ValueError("Auto shares require 0 < small share < cumulative medium/large share < 1")
        elif not (
            self.manual_small_medium_radius >= 0.0
            and self.manual_small_medium_radius < self.manual_medium_large_radius
        ):
            raise ValueError("Manual thresholds require 0 <= r_sm < r_ml")
        if len(set(self.enabled_size_classes)) != len(self.enabled_size_classes):
            raise ValueError("enabled_size_classes must not contain duplicates")
        return self

    def seed_for(self, realization_index: int) -> int:
        """Return the deterministic seed for a zero-based realization index."""
        if realization_index < 0:
            raise ValueError("realization_index must be non-negative")
        return self.base_seed + realization_index


class DeterministicStructure(BaseModel):
    """Minimal parameterized deterministic plane/disc imported in M10."""

    structure_id: str
    center_x: float
    center_y: float
    center_z: float
    dip_direction: float = Field(ge=0.0, lt=360.0)
    dip: float = Field(ge=0.0, le=90.0)
    radius: float = Field(gt=0.0)
    structure_type: str
    domain_id: int | None = None
    set_id: int | None = None
    source_file: str = ""
    source_row: int = Field(default=0, ge=0)
    provenance: dict[str, Any] = Field(default_factory=dict)


class M10DomainSetSummary(BaseModel):
    """Generation diagnostics for one voxel/domain/set target aggregate."""

    domain_id: int | None = None
    set_id: int
    target_area: float = Field(default=0.0, ge=0.0)
    target_count_expectation: float = Field(default=0.0, ge=0.0)
    conditioned_area_deducted: float = Field(default=0.0, ge=0.0)
    deterministic_area_deducted: float = Field(default=0.0, ge=0.0)
    stochastic_original_area: float = Field(default=0.0, ge=0.0)
    stochastic_clipped_area: float = Field(default=0.0, ge=0.0)
    fracture_count: int = Field(default=0, ge=0)
    over_conditioned_cell_count: int = Field(default=0, ge=0)
    skipped_cell_count: int = Field(default=0, ge=0)
    status: str = "generated"
    target_mean_dip_direction: float | None = None
    target_mean_dip: float | None = None
    target_kappa: float | None = None
    generated_mean_dip_direction: float | None = None
    generated_mean_dip: float | None = None
    generated_resultant_length: float | None = None
    small_medium_radius: float | None = None
    medium_large_radius: float | None = None
    size_threshold_mode: str = "auto"
    size_class_budgets: dict[str, dict[str, float | bool | None]] = Field(default_factory=dict)
    p32_explicit_target: float = Field(default=0.0, ge=0.0)
    p32_subgrid: float = Field(default=0.0, ge=0.0)
    p32_unresolved_orientation: float = Field(default=0.0, ge=0.0)
    p32_conservation_error: float = 0.0


class M10QualitySummary(BaseModel):
    """Auditable per-realization M10 quality report."""

    target_fracture_count_expectation: float = Field(default=0.0, ge=0.0)
    fracture_count: int = Field(default=0, ge=0)
    stochastic_count: int = Field(default=0, ge=0)
    conditioned_count: int = Field(default=0, ge=0)
    deterministic_count: int = Field(default=0, ge=0)
    target_p32: float = Field(default=0.0, ge=0.0)
    generated_original_p32: float = Field(default=0.0, ge=0.0)
    generated_clipped_p32: float = Field(default=0.0, ge=0.0)
    p32_explicit_target: float = Field(default=0.0, ge=0.0)
    p32_subgrid: float = Field(default=0.0, ge=0.0)
    p32_unresolved_orientation: float = Field(default=0.0, ge=0.0)
    p32_target_conservation_error: float = 0.0
    mean_radius: float | None = None
    radius_q05: float | None = None
    radius_q50: float | None = None
    radius_q95: float | None = None
    domain_set: list[M10DomainSetSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model_sources: dict[str, Any] = Field(default_factory=dict)
    skipped_regions: list[dict[str, Any]] = Field(default_factory=list)
    over_conditioned_regions: list[dict[str, Any]] = Field(default_factory=list)
    local_p32_status: LocalP32Status = LocalP32Status.CENTER_ASSIGNED_PRELIMINARY
    local_p32_notice: str = (
        "Local voxel P32 has not yet been recomputed by exact fracture–voxel intersection. "
        "This will be completed in M11."
    )


class M10Realization(BaseModel):
    """One complete explicit-DFN realization and its compressed geometry arrays."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    realization_id: str
    realization_index: int = Field(ge=0)
    seed: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_hash: str
    complete: bool = True
    quality: M10QualitySummary = Field(default_factory=M10QualitySummary)
    geometry_arrays: dict[str, np.ndarray] = Field(default_factory=dict, exclude=True)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def fracture_count(self) -> int:
        """Return the number of stored explicit fractures."""
        centers = self.geometry_arrays.get("center")
        return 0 if centers is None else int(len(centers))

    def fracture_id(self, ordinal: int) -> str:
        """Construct a stable user-facing fracture ID without storing UUID strings per row."""
        if ordinal < 0 or ordinal >= self.fracture_count:
            raise IndexError("fracture ordinal is outside this realization")
        return f"{self.realization_id}:f{ordinal:09d}"


class M10State(BaseModel):
    """Single project-owned state for M10 configuration and realizations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: M10GenerationConfig = Field(default_factory=M10GenerationConfig)
    deterministic_structures: list[DeterministicStructure] = Field(default_factory=list)
    realizations: list[M10Realization] = Field(default_factory=list)
    last_input_hash: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
