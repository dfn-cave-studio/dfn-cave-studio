"""M8 spatial-domain and lightweight voxel-grid definition models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from dfn_cave_studio.models.bounds import ModelBounds


class GridMemoryField(BaseModel):
    """A field included in the grid memory estimate."""

    name: str
    bytes_per_voxel: int = Field(gt=0)


class VoxelCellState(StrEnum):
    """Non-conflated semantic state of an analysis cell."""

    OUTSIDE_MODEL = "outside_model"
    NO_DATA = "no_data"
    TRUE_ZERO = "true_zero"
    MODELED_VALUE = "modeled_value"


class SpatialGridConfig(BaseModel):
    """Persisted analysis/generation domains and preview settings."""

    analysis_domain: ModelBounds
    generation_domain: ModelBounds
    boundary_mode: str = Field(default="manual", pattern="^(auto|manual)$")
    outward_margin: float = Field(default=0.0, ge=0.0)
    buffer_layers: int = Field(default=2, ge=0)
    maximum_fracture_radius: float = Field(default=0.0, ge=0.0)
    clipping_acknowledged: bool = False
    preview_opacity: float = Field(default=0.35, ge=0.0, le=1.0)
    preview_slice_axis: str | None = Field(default=None, pattern="^(x|y|z)$")
    preview_slice_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def generation_contains_analysis(self) -> SpatialGridConfig:
        """Require the DFN generation domain to contain the analysis domain."""
        a = self.analysis_domain
        g = self.generation_domain
        if not (
            g.x_min <= a.x_min
            and g.x_max >= a.x_max
            and g.y_min <= a.y_min
            and g.y_max >= a.y_max
            and g.z_min <= a.z_min
            and g.z_max >= a.z_max
        ):
            raise ValueError("DFN generation domain must contain the voxel analysis domain")
        return self


class BoundaryViolationReport(BaseModel):
    """Detailed result of checking spatial data against a boundary."""

    outside_borehole_count: int = 0
    outside_trajectory_point_count: int = 0
    outside_observation_point_count: int = 0
    maximum_exceedance: dict[str, float] = Field(
        default_factory=lambda: {
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
            "z_max": 0.0,
        }
    )
    affected_holes: list[str] = Field(default_factory=list)
    recommended_domain: ModelBounds

    @property
    def has_violations(self) -> bool:
        """Whether any trajectory or observation lies outside."""
        return self.outside_trajectory_point_count > 0 or self.outside_observation_point_count > 0


class VoxelGridSummary(BaseModel):
    """Allocation-free voxel-grid dimensions and memory estimate."""

    nx: int = Field(gt=0)
    ny: int = Field(gt=0)
    nz: int = Field(gt=0)
    total_voxels: int = Field(gt=0)
    active_voxels: int | None = Field(default=None, ge=0)
    estimated_bytes: int = Field(ge=0)
    fields: list[GridMemoryField]
    state_semantics: list[VoxelCellState] = Field(default_factory=lambda: list(VoxelCellState))
    warning: str | None = None
