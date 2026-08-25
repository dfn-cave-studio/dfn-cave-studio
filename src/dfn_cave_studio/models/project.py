"""
Project data model — the top-level container for all DFN Cave Studio data.

A Project is the serializable unit: it contains or references ALL data
needed to reproduce a DFN study. This includes:
  - Metadata and version info
  - Model bounds and coordinate system
  - Unit and random seed configuration
  - Boreholes, structural domains, fracture sets
  - Deterministic and stochastic fractures
  - Voxel and mask configuration
  - Mechanical property library
  - Surface model
  - Analysis results (connectivity, fragmentation)

The project model supports versioning for forward/backward compatibility.
Project files use JSON format with the .dfncs extension.

References:
  - SCIENTIFIC_SPEC.md Section 2, 10.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture import DeterministicFracture, StochasticFracture
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.dfn_realization import DFNRealization
from dfn_cave_studio.models.borehole import BoreholeCollection
from dfn_cave_studio.models.structural_domain import StructuralDomainCollection
from dfn_cave_studio.models.rock_mask import RockMask, ExcavationMask, SurfaceModel, SpatialAttributeConfig
from dfn_cave_studio.models.mechanical_properties import MechanicalPropertyLibrary
from dfn_cave_studio.models.enums import ProjectStatus
from dfn_cave_studio.models.borehole_database import BoreholeDatabase
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig
from dfn_cave_studio.models.m9 import M9State


# =============================================================================
# Project Metadata
# =============================================================================

class ProjectMetadata(BaseModel):
    """Project-level metadata."""

    name: str = Field(default="Untitled Project")
    description: str = ""
    author: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    software_version: str = "0.9.1"
    project_version: int = 1  # Schema version for migration

    tags: List[str] = Field(default_factory=list)
    notes: str = ""


# =============================================================================
# Project Configuration
# =============================================================================

class ProjectConfig(BaseModel):
    """Project-level configuration that isn't in AppConfig.

    This is the persisted project configuration (units, seeds, tolerances).
    The running app config (AppConfig) is separate and not persisted in project.
    """

    master_seed: int = 42
    dfn_seed_override: Optional[int] = None
    voxel_seed_override: Optional[int] = None
    block_seed_override: Optional[int] = None

    # P32 convergence
    p32_tolerance: float = Field(default=0.05, ge=0.0, le=1.0, description="Relative P32 tolerance")

    # Coordinate system
    coordinate_system: str = "local_cartesian"  # "local_cartesian", "utm", "geographic"
    utm_zone: Optional[int] = None


# =============================================================================
# Project (Root Model)
# =============================================================================

CURRENT_PROJECT_VERSION = 3


class Project(BaseModel):
    """The complete, serializable DFN Cave Studio project.

    This is the root object that is saved/loaded from .dfncs files.
    All sub-models are owned by the Project.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Identity ──────────────────────────────────────────────────────────
    project_id: UUID = Field(default_factory=uuid4)
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)

    # ── Spatial ───────────────────────────────────────────────────────────
    model_bounds: ModelBounds = Field(default_factory=lambda: ModelBounds(
        x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=100,
    ))
    voxel_config: VoxelConfig = Field(default_factory=VoxelConfig)
    spatial_attr_config: SpatialAttributeConfig = Field(default_factory=SpatialAttributeConfig)

    # ── Configuration ─────────────────────────────────────────────────────
    config: ProjectConfig = Field(default_factory=ProjectConfig)

    # ── Surfaces & Masks ──────────────────────────────────────────────────
    surface_model: Optional[SurfaceModel] = None
    rock_mask: RockMask = Field(default_factory=RockMask)
    excavation_mask: ExcavationMask = Field(default_factory=ExcavationMask)

    # ── Geological Data ───────────────────────────────────────────────────
    borehole_collection: BoreholeCollection = Field(default_factory=BoreholeCollection)
    borehole_database: BoreholeDatabase = Field(default_factory=BoreholeDatabase)
    spatial_grid_config: Optional[SpatialGridConfig] = None
    m9_state: M9State = Field(default_factory=M9State)
    structural_domains: StructuralDomainCollection = Field(
        default_factory=StructuralDomainCollection
    )

    # ── DFN ───────────────────────────────────────────────────────────────
    joint_sets: List[JointSetConfig] = Field(default_factory=list)
    deterministic_fractures: List[DeterministicFracture] = Field(default_factory=list)
    dfn_realizations: List[DFNRealization] = Field(default_factory=list)

    # ── Mechanical ────────────────────────────────────────────────────────
    mechanical_library: MechanicalPropertyLibrary = Field(
        default_factory=MechanicalPropertyLibrary
    )

    # ── Status ────────────────────────────────────────────────────────────
    project_status: ProjectStatus = ProjectStatus.NEW

    # ── Results (cached) ──────────────────────────────────────────────────
    connectivity_results: Optional[Dict[str, Any]] = None
    fragmentation_results: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    voxel_p32_results: Optional[Dict[str, Any]] = None  # per-voxel P32 values
    connectivity_clusters: Optional[List[int]] = None    # per-fracture component IDs

    # ── Version Control ───────────────────────────────────────────────────
    schema_version: int = CURRENT_PROJECT_VERSION
    vcs_commit_sha: Optional[str] = None  # Git commit at save time

    # ── Computed ──────────────────────────────────────────────────────────
    @property
    def model_volume(self) -> float:
        """Model volume in m³."""
        return self.model_bounds.volume

    @property
    def total_deterministic_fractures(self) -> int:
        """Total number of deterministic fractures."""
        return len(self.deterministic_fractures)

    @property
    def total_stochastic_fractures(self) -> int:
        """Total number of stochastic fractures across all realizations."""
        return sum(r.fracture_count for r in self.dfn_realizations)

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the project to a JSON-compatible dictionary.

        Returns:
            Dictionary representation suitable for json.dump().
        """
        return json.loads(self.model_dump_json(indent=2))

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Deserialize a project from a dictionary.

        Handles schema version migration automatically.

        Args:
            data: Dictionary from json.load().

        Returns:
            Project instance.
        """
        schema_version = data.get("schema_version", 1)

        # Migrate if needed
        if schema_version < CURRENT_PROJECT_VERSION:
            data = cls._migrate(data, schema_version)

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Project":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: Path) -> "Project":
        """Load a project from a .dfncs JSON file.

        Args:
            path: Path to the project file.

        Returns:
            Project instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not valid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        project = cls.from_dict(data)

        # Verify the file was valid
        if not isinstance(data, dict):
            raise ValueError(f"Invalid project file: {path}")

        return project

    def save_to_file(self, path: Path) -> None:
        """Save the project to a .dfncs JSON file.

        Automatically updates modified_at timestamp.

        Args:
            path: Target file path.
        """
        path = Path(path)
        self.metadata.modified_at = datetime.now(timezone.utc)
        self.metadata.software_version = "0.9.1"

        # Try to get git commit SHA
        self.vcs_commit_sha = self._get_git_sha()

        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _get_git_sha(self) -> Optional[str]:
        """Try to get the current git commit SHA."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None

    # ── Version Migration ─────────────────────────────────────────────────

    @classmethod
    def _migrate(cls, data: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """Migrate project data from an older schema version.

        Args:
            data: Project data dictionary.
            from_version: Source schema version.

        Returns:
            Migrated data dictionary.
        """
        # Migration chain: each step handles one version bump
        migrated = dict(data)

        if from_version < 1:
            # Future: v0 → v1 migration
            pass

        if from_version < 2:
            migrated.setdefault("borehole_database", {"schema_version": 1, "records": []})
            migrated.setdefault("spatial_grid_config", None)

        if from_version < 3:
            migrated.setdefault("m9_state", {})

        # Future versions:
        # if from_version < 2:
        #     migrated = cls._migrate_v1_to_v2(migrated)

        migrated["schema_version"] = CURRENT_PROJECT_VERSION
        return migrated

    # ── Convenience ───────────────────────────────────────────────────────

    def create_sample_project(self) -> None:
        """Populate with reasonable sample data for quick start."""
        from dfn_cave_studio.models.fracture_set import (
            OrientationDistribution, SizeDistribution, SpatialDistribution
        )
        from dfn_cave_studio.models.enums import SizeDistributionType

        self.metadata.name = "Sample Project"
        self.metadata.description = "Auto-generated sample DFN project"

        # Add three typical joint sets
        self.joint_sets = [
            JointSetConfig(
                set_id=1, name="Joint Set 1 (Main)",
                color="#1976d2",
                orientation=OrientationDistribution(
                    mean_dip_direction=45, mean_dip=60, kappa=25,
                ),
                size=SizeDistribution(
                    distribution_type=SizeDistributionType.LOGNORMAL,
                    lognormal_mu=1.0, lognormal_sigma=0.5,
                    min_radius=1.0, max_radius=10.0,
                ),
                target_p32=0.5,
            ),
            JointSetConfig(
                set_id=2, name="Joint Set 2 (Secondary)",
                color="#388e3c",
                orientation=OrientationDistribution(
                    mean_dip_direction=135, mean_dip=70, kappa=20,
                ),
                size=SizeDistribution(
                    distribution_type=SizeDistributionType.LOGNORMAL,
                    lognormal_mu=0.8, lognormal_sigma=0.6,
                    min_radius=0.5, max_radius=8.0,
                ),
                target_p32=0.3,
            ),
            JointSetConfig(
                set_id=3, name="Joint Set 3 (Minor)",
                color="#f57c00",
                orientation=OrientationDistribution(
                    mean_dip_direction=270, mean_dip=30, kappa=15,
                ),
                size=SizeDistribution(
                    distribution_type=SizeDistributionType.POWER_LAW,
                    power_law_exponent=3.0,
                    min_radius=0.3, max_radius=5.0,
                ),
                target_p32=0.2,
            ),
        ]

        self.project_status = ProjectStatus.DATA_LOADED
