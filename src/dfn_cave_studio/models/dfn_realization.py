"""
DFN realization and project-level DFN configuration models.

A DFNRealization is one complete stochastic realization of the fracture network.
It contains:
  - All generated stochastic fractures
  - Deterministic fractures (shared across realizations)
  - Metadata: seeds, P32 achieved, generation parameters

References:
  - SCIENTIFIC_SPEC.md Section 5, 6.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict

from dfn_cave_studio.models.fracture import StochasticFracture, DeterministicFracture
from dfn_cave_studio.models.fracture_set import JointSetConfig


class DFNGenerationConfig(BaseModel):
    """Configuration for a DFN generation run."""

    master_seed: int = 42
    joint_sets: List[JointSetConfig] = Field(default_factory=list)
    model_volume: float = 0.0  # m³
    boundary_buffer: float = Field(default=1.0, description="Extra distance for fracture generation beyond bounds (m)")
    max_fractures_per_set: int = Field(default=100000, description="Safety cap on fracture count")

    # Deterministic fractures included in this DFN
    deterministic_fracture_ids: List[str] = Field(default_factory=list)


class DFNGenerationResult(BaseModel):
    """Results from a DFN generation run."""

    realization_id: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Generation parameters used
    config: DFNGenerationConfig = Field(default_factory=DFNGenerationConfig)

    # Generated fractures
    stochastic_fractures: List[StochasticFracture] = Field(default_factory=list)

    # Per-set statistics
    set_statistics: Dict[int, Dict[str, Any]] = Field(default_factory=dict)

    # Global statistics
    total_fractures: int = 0
    total_fracture_area: float = 0.0  # m²
    achieved_p32: float = 0.0  # m²/m³
    target_p32: float = 0.0  # m²/m³
    p32_error_percent: float = 0.0
    convergence_achieved: bool = False
    elapsed_seconds: float = 0.0
    cancelled: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DFNRealization(BaseModel):
    """A complete DFN realization.

    Each realization represents one possible configuration of the fracture
    network consistent with the statistical input parameters.
    """

    realization_id: UUID = Field(default_factory=uuid4)
    realization_number: int = 0
    name: str = "Realization 1"

    # Fractures
    deterministic_fractures: List[DeterministicFracture] = Field(default_factory=list)
    stochastic_fractures: List[StochasticFracture] = Field(default_factory=list)

    # Generation metadata
    generation_config: Optional[DFNGenerationConfig] = None
    generation_result: Optional[DFNGenerationResult] = None

    # Cached statistics
    p32_by_set: Dict[int, float] = Field(default_factory=dict)
    total_p32: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def all_fractures(self) -> List:
        """Combined list of all fractures (deterministic + stochastic)."""
        return list(self.deterministic_fractures) + list(self.stochastic_fractures)

    @property
    def fracture_count(self) -> int:
        """Total number of fractures."""
        return len(self.stochastic_fractures) + len(self.deterministic_fractures)

    def get_fractures_by_set(self, set_id: int) -> List[StochasticFracture]:
        """Get all stochastic fractures belonging to a specific set."""
        return [f for f in self.stochastic_fractures if f.set_id == set_id]
