"""Persistent Phase 2A models for constrained borehole-fracture realizations."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RandomComponentStatus(StrEnum):
    """Whether a point-cloud site reported its random fracture component."""

    REPORTED_PRESENT = "REPORTED_PRESENT"
    NOT_REPORTED = "NOT_REPORTED"
    REPORTED_ABSENT = "REPORTED_ABSENT"


class BoreholeFractureComponent(IntEnum):
    """Compact component code stored for every generated fracture."""

    DOMINANT_SET = 1
    RANDOM_BACKGROUND = 2


class LocalOrientationComponent(BaseModel):
    """One immutable imported local component referenced by generated rows."""

    component_id: str
    observation_id: str
    point_key: str
    point_id: str
    local_set_id: str
    source_kind: str
    x: float
    y: float
    z: float
    dip: float | None = Field(default=None, ge=0, le=90)
    dip_direction: float | None = Field(default=None, ge=0, lt=360)
    joint_spacing_m: float | None = Field(default=None, gt=0)
    joint_num: int | None = Field(default=None, gt=0)
    component_type: Literal["LOCAL_DOMINANT_SET", "RANDOM_BACKGROUND"]
    global_set_id: int | None = Field(default=None, gt=0)
    source_file: str
    source_row: int = Field(ge=0)


class GlobalSetMapping(BaseModel):
    """Stable mapping from one representative site orientation to a mine-wide set."""

    observation_id: str
    component_id: str = ""
    point_key: str
    local_set_id: str
    source_kind: str
    global_set_id: int = Field(gt=0)
    sample_weight: int = Field(gt=0)


class GlobalJointSetModel(BaseModel):
    """One fitted mine-wide axial orientation model."""

    global_set_id: int = Field(gt=0)
    mean_dip_direction: float = Field(ge=0, lt=360)
    mean_dip: float = Field(ge=0, le=90)
    kappa: float = Field(gt=0)
    kappa_status: Literal[
        "UNRESOLVED",
        "SITE_MEAN_DISPERSION",
        "MEASURED_WITHIN_SET",
        "MANUAL",
        "ASSUMED",
        "LEGACY_UNSPECIFIED",
    ] = "LEGACY_UNSPECIFIED"
    representative_count: int = Field(ge=0)
    weighted_count: int = Field(ge=0)


class GlobalJointSetFit(BaseModel):
    """Complete, reproducible Phase 2A global-set fit."""

    number_of_sets: int = Field(gt=0)
    random_seed: int
    sets: list[GlobalJointSetModel]
    mappings: list[GlobalSetMapping]
    local_components: list[LocalOrientationComponent] = Field(default_factory=list)
    input_hash: str
    algorithm: str = "AXIAL_SPHERICAL_KMEANS_WEIGHTED_REPRESENTATIVES"
    provenance: dict[str, Any] = Field(default_factory=dict)


class BoreholeFractureGenerationConfig(BaseModel):
    """Configuration for Phase 2A constrained along-hole generation."""

    number_of_sets: int = Field(default=3, ge=1, le=50)
    use_confirmed_global_fit: bool = False
    realization_count: int = Field(default=1, ge=1, le=100)
    master_seed: int = 42
    position_sampler: Literal["HOMOGENEOUS_POISSON"] = "HOMOGENEOUS_POISSON"
    idw_power: float = Field(default=2.0, gt=0)
    search_radius: float = Field(default=1_000.0, gt=0)
    idw_search_mode: Literal["RADIUS", "ALL_WITHIN_DOMAIN"] = "RADIUS"
    max_neighbors: int = Field(default=12, ge=1)
    min_neighbors: int = Field(default=1, ge=1)
    direction_mode: Literal[
        "LOCAL_REPRESENTATIVE",
        "SPATIALLY_FITTED_WITHIN_GLOBAL_SET",
        "FIXED_GLOBAL_SET_MEAN",
    ] = "LOCAL_REPRESENTATIVE"
    random_background_strategy: Literal["ISOTROPIC_AXIAL_HEMISPHERE"] = "ISOTROPIC_AXIAL_HEMISPHERE"
    memory_budget_bytes: int = Field(default=512 * 1024**2, ge=1)
    generation_chunk_size: int = Field(default=65_536, ge=256, le=1_000_000)

    @model_validator(mode="before")
    @classmethod
    def migrate_direction_mode(cls, data: object) -> object:
        """Map Phase 2A v2 display names without changing stored science arrays."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        values["direction_mode"] = {
            "FIXED_GROUP_MEAN": "FIXED_GLOBAL_SET_MEAN",
            "FITTED_FROM_SITE_MEANS": "SPATIALLY_FITTED_WITHIN_GLOBAL_SET",
        }.get(values.get("direction_mode"), values.get("direction_mode", "LOCAL_REPRESENTATIVE"))
        return values

    @model_validator(mode="after")
    def validate_neighbors(self) -> "BoreholeFractureGenerationConfig":
        if self.min_neighbors > self.max_neighbors:
            raise ValueError("min_neighbors must be less than or equal to max_neighbors")
        return self

    def seed_for(self, realization_index: int) -> int:
        """Derive a deterministic realization seed without global RNG state."""
        if realization_index < 0:
            raise ValueError("realization_index must be non-negative")
        sequence = np.random.SeedSequence([self.master_seed & ((1 << 64) - 1), realization_index, 0x2A])
        return int(sequence.generate_state(1, dtype=np.uint64)[0])

    def rng_for(self, realization_index: int, interval_index: int, stream_id: int) -> np.random.Generator:
        """Return one stable, independent interval-operation substream."""
        if realization_index < 0 or interval_index < 0 or stream_id < 0:
            raise ValueError("realization, interval and stream indices must be non-negative")
        sequence = np.random.SeedSequence(
            [self.master_seed & ((1 << 64) - 1), realization_index, interval_index, stream_id, 0x2A]
        )
        return np.random.default_rng(sequence)


class BoreholeIntervalDiagnostic(BaseModel):
    """Generation status and counts for one original spacing interval."""

    interval_index: int = Field(ge=0)
    observation_id: str
    hole_id: str
    from_depth: float
    to_depth: float
    spacing_m: float = Field(gt=0)
    expected_count: float = Field(ge=0)
    generated_count: int = Field(ge=0)
    status: Literal["GENERATED", "BLOCKED_NO_LOCAL_INTENSITY"]
    global_set_probabilities: dict[int, float] = Field(default_factory=dict)
    local_component_probabilities: dict[int, float] = Field(default_factory=dict)
    random_probability: float = Field(default=0, ge=0, le=1)
    random_component_diagnostic: str = ""
    selected_point_count: int = Field(default=0, ge=0)
    nearest_constraint_distance: float | None = Field(default=None, ge=0)
    farthest_constraint_distance: float | None = Field(default=None, ge=0)
    is_long_range_extrapolation: bool = False


class BoreholeFractureRealization(BaseModel):
    """Columnar along-hole realization; no per-fracture Python objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    realization_id: str
    realization_index: int = Field(ge=0)
    master_seed: int
    derived_seed: int
    input_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    generator_name: str = "HOMOGENEOUS_POISSON"
    generator_version: str = "borehole-phase2a-3"
    complete: bool = True
    fracture_count: int = Field(ge=0)
    interval_diagnostics: list[BoreholeIntervalDiagnostic] = Field(default_factory=list)
    global_set_counts: dict[int, int] = Field(default_factory=dict)
    random_background_count: int = Field(default=0, ge=0)
    array_member: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    arrays: dict[str, np.ndarray] = Field(default_factory=dict, exclude=True)


class BoreholeFractureState(BaseModel):
    """Project-owned fit and multi-realization Phase 2A state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: BoreholeFractureGenerationConfig = Field(default_factory=BoreholeFractureGenerationConfig)
    global_fit: GlobalJointSetFit | None = None
    random_component_status: dict[str, RandomComponentStatus] = Field(default_factory=dict)
    realizations: list[BoreholeFractureRealization] = Field(default_factory=list)
    selected_realization_id: str | None = None
    archive_source: str | None = Field(default=None, exclude=True)
