"""
Fracture data models for DFN Cave Studio.

Defines the core fracture representations:
  - DeterministicFracture: Mapped/known individual fractures (faults, joints, contacts)
  - StochasticFracture: Generated fractures from statistical distributions
  - FractureGeometry: Disk, polygon, or rectangular plane geometry

References:
  - Baecher, G.B. et al. (1977). Statistical description of rock properties and sampling.
  - SCIENTIFIC_SPEC.md Section 4.
"""

from __future__ import annotations

from typing import Optional, List, Any
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ConfigDict

from dfn_cave_studio.models.enums import FractureType, FractureSource


# =============================================================================
# Geometry Types
# =============================================================================

class FractureGeometry(BaseModel):
    """Geometric representation of a single fracture.

    Supports three representations:
      - disk: Circular disk (center, normal, radius)
      - polygon: Planar polygon (set of 3D vertices)
      - rectangle: Finite rectangular plane (center, normal, width, height)
    """

    geometry_type: str = Field(default="disk", description="'disk', 'polygon', or 'rectangle'")

    # Disk parameters
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    radius: Optional[float] = None

    # Polygon parameters
    vertices: Optional[NDArray[np.float64]] = Field(default=None, exclude=True)

    # Rectangle parameters
    width: Optional[float] = None
    height: Optional[float] = None

    # Common
    normal_x: float = 0.0
    normal_y: float = 0.0
    normal_z: float = 1.0
    dip_direction: float = 0.0  # degrees
    dip: float = 0.0  # degrees
    area: float = 0.0  # m²

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def center(self) -> NDArray[np.float64]:
        """Fracture center as a numpy array."""
        return np.array([self.center_x, self.center_y, self.center_z])

    @property
    def normal(self) -> NDArray[np.float64]:
        """Fracture normal as a numpy array."""
        return np.array([self.normal_x, self.normal_y, self.normal_z])


# =============================================================================
# Fracture Models
# =============================================================================

class Fracture(BaseModel):
    """Base fracture model common to all fracture types."""

    fracture_id: UUID = Field(default_factory=uuid4)
    name: str = ""
    fracture_type: FractureType = FractureType.JOINT
    source: FractureSource = FractureSource.USER_DEFINED

    # Geometry
    geometry: FractureGeometry = Field(default_factory=FractureGeometry)

    # Spatial domain
    domain_id: int = 0
    realization_id: Optional[int] = None

    # Mechanical property reference
    mechanical_property_id: Optional[str] = None

    # Metadata
    random_seed: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DeterministicFracture(Fracture):
    """A known, mapped fracture (fault, shear zone, major joint, contact).

    Deterministic fractures are distinguished from stochastic fractures:
      - They have known (measured or interpreted) geometry
      - They have documented data provenance
      - They are NOT generated from statistical distributions
    """

    # Data provenance
    data_source: str = ""  # e.g., "borehole BH-001", "surface mapping", "geophysics"
    data_quality: str = ""  # e.g., "measured", "interpreted", "inferred"
    reference_document: str = ""  # e.g., report filename, survey date

    # Optional additional geometry
    surface_mesh_path: Optional[str] = None  # Path to triangulated surface file


class StochasticFracture(Fracture):
    """A fracture generated from statistical distributions.

    Stochastic fractures belong to a specific fracture set and realization.
    Each fracture stores its own seed for reproducibility.
    """

    set_id: int = 0
    set_name: str = ""

    # Generated parameters
    radius: float = 0.0
    vertices_list: Optional[List[List[float]]] = None

    def polygon_vertices(self) -> Optional[NDArray[np.float64]]:
        """Get polygon vertices as numpy array (if polygon type)."""
        if self.vertices_list is None:
            return None
        return np.array(self.vertices_list, dtype=np.float64)


# =============================================================================
# Fracture Creation Helpers
# =============================================================================

def create_fracture_from_dip(
    center: tuple[float, float, float],
    dip_direction_deg: float,
    dip_deg: float,
    radius: float,
    set_id: int = 0,
    realization_id: int = 0,
    random_seed: Optional[int] = None,
) -> StochasticFracture:
    """Create a stochastic fracture from dip direction and dip.

    Args:
        center: (x, y, z) center point in meters.
        dip_direction_deg: Dip direction in degrees.
        dip_deg: Dip angle in degrees.
        radius: Disk radius in meters.
        set_id: Fracture set identifier.
        realization_id: DFN realization number.
        random_seed: Seed for reproducibility.

    Returns:
        StochasticFracture instance.
    """
    from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal

    normal = dip_dir_dip_to_normal(dip_direction_deg, dip_deg)
    area = np.pi * radius ** 2

    geometry = FractureGeometry(
        geometry_type="disk",
        center_x=center[0],
        center_y=center[1],
        center_z=center[2],
        radius=radius,
        normal_x=float(normal[0]),
        normal_y=float(normal[1]),
        normal_z=float(normal[2]),
        dip_direction=dip_direction_deg,
        dip=dip_deg,
        area=area,
    )

    return StochasticFracture(
        geometry=geometry,
        set_id=set_id,
        realization_id=realization_id,
        random_seed=random_seed,
        radius=radius,
        source=FractureSource.STOCHASTIC,
    )
