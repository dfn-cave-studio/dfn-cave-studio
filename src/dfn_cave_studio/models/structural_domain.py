"""
Structural domain models for DFN Cave Studio.

A structural domain is a 3D region with distinct rock mass properties
and DFN parameters. Domains can be defined by:
  - Manual box/polygon boundaries
  - Fault buffer distances
  - Borehole cluster analysis
  - Spatial interpolation from borehole data

The global model is a special case: a single domain covering the entire model.
Domain partitioning MUST respect: global domain_id = 0, sub-domains > 0.

References:
  - SCIENTIFIC_SPEC.md Section 7.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator

from dfn_cave_studio.models.fracture_set import JointSetConfig


# =============================================================================
# Domain Boundary Types
# =============================================================================

class DomainBoundaryType:
    """Enum-like constants for domain boundary definition methods."""
    GLOBAL = "global"              # Entire model (single domain)
    MANUAL_BOX = "manual_box"      # User-defined axis-aligned box
    MANUAL_POLYGON = "manual_polygon"  # User-defined 3D polygon
    FAULT_BUFFER = "fault_buffer"    # Distance buffer around faults
    BOREHOLE_CLUSTER = "borehole_cluster"  # Clustered from borehole data
    DISTANCE_FIELD = "distance_field"  # Signed distance from reference
    INTERPOLATED = "interpolated"    # Interpolated from point data

    ALL = frozenset([
        GLOBAL, MANUAL_BOX, MANUAL_POLYGON, FAULT_BUFFER,
        BOREHOLE_CLUSTER, DISTANCE_FIELD, INTERPOLATED,
    ])


# =============================================================================
# Polygon Boundary
# =============================================================================

class DomainPolygonBoundary(BaseModel):
    """A 3D polygon boundary for structural domain definition."""

    vertices: List[List[float]] = Field(
        default_factory=list,
        description="List of 3D points [[x,y,z], ...] defining the polygon"
    )
    z_min: float = Field(default=-1e10, description="Lower Z bound")
    z_max: float = Field(default=1e10, description="Upper Z bound")

    @property
    def vertex_array(self) -> NDArray[np.float64]:
        """Vertices as numpy array (N, 2) for 2D check, or (N, 3)."""
        return np.array(self.vertices, dtype=np.float64)

    def contains_xy(self, x: float, y: float) -> bool:
        """Check if (x,y) is inside the 2D projection of the polygon.

        Uses ray-casting algorithm.
        """
        verts = self.vertex_array
        if verts.shape[0] < 3:
            return False

        inside = False
        n = verts.shape[0]
        j = n - 1
        for i in range(n):
            xi, yi = verts[i, 0], verts[i, 1]
            xj, yj = verts[j, 0], verts[j, 1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def contains_3d(self, x: float, y: float, z: float) -> bool:
        """Check if point is inside the 3D extruded polygon."""
        if z < self.z_min or z > self.z_max:
            return False
        return self.contains_xy(x, y)


# =============================================================================
# Structural Domain
# =============================================================================

class StructuralDomain(BaseModel):
    """A spatial region with distinct DFN parameters.

    Each domain has:
      - A spatial boundary definition
      - Domain-specific joint set configurations
      - Domain-specific mechanical property assignments
      - Display properties

    Domain 0 is reserved for the global (default) domain.
    """

    domain_id: int = Field(default=0, ge=0, description="Unique domain identifier (0 = global)")
    name: str = Field(default="Global Domain")
    description: str = ""
    color: str = Field(default="#66bb6a", description="Display color")

    # Boundary definition
    boundary_type: str = Field(default=DomainBoundaryType.GLOBAL)
    polygon_boundary: Optional[DomainPolygonBoundary] = None

    # Box boundary (for manual_box type)
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None

    # Buffer-based (for fault_buffer type)
    reference_fault_ids: List[str] = Field(default_factory=list)
    buffer_distance: float = Field(default=20.0, description="Buffer distance in meters")

    # Domain-specific DFN parameters (overrides global)
    joint_sets: List[JointSetConfig] = Field(default_factory=list)

    # Display
    visible: bool = True
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)

    # Version tracking
    model_version: int = 1

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a 3D point is within this domain.

        Args:
            x, y, z: Point coordinates in meters.

        Returns:
            True if the point is inside this domain.
        """
        if self.boundary_type == DomainBoundaryType.GLOBAL:
            return True

        elif self.boundary_type == DomainBoundaryType.MANUAL_BOX:
            if any(v is None for v in [self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max]):
                return False
            return (
                self.x_min <= x <= self.x_max
                and self.y_min <= y <= self.y_max
                and self.z_min <= z <= self.z_max
            )

        elif self.boundary_type == DomainBoundaryType.MANUAL_POLYGON:
            if self.polygon_boundary is None:
                return False
            return self.polygon_boundary.contains_3d(x, y, z)

        elif self.boundary_type == DomainBoundaryType.FAULT_BUFFER:
            # Placeholder: requires reference to fault geometries
            return False

        else:
            # Default: not in domain
            return False


# =============================================================================
# Structural Domain Collection
# =============================================================================

class StructuralDomainCollection(BaseModel):
    """Collection of structural domains with query methods.

    The global domain (domain_id=0) is always present as a fallback.
    Domain lookup follows priority: higher domain_id = higher priority.
    """

    domains: List[StructuralDomain] = Field(default_factory=list)
    model_version: int = 1

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure global domain exists
        if not any(d.domain_id == 0 for d in self.domains):
            self.domains.insert(0, StructuralDomain(domain_id=0, name="Global Domain"))

    def get_domain(self, domain_id: int) -> Optional[StructuralDomain]:
        """Get a domain by ID."""
        for d in self.domains:
            if d.domain_id == domain_id:
                return d
        return None

    def find_domain(self, x: float, y: float, z: float) -> StructuralDomain:
        """Find which domain contains a point.

        Returns the highest-priority domain containing the point,
        falling back to the global domain.

        Args:
            x, y, z: Point coordinates in meters.

        Returns:
            The StructuralDomain containing the point.
        """
        # Check domains in reverse priority order (higher ID = higher priority)
        sorted_domains = sorted(
            [d for d in self.domains if d.domain_id > 0],
            key=lambda d: d.domain_id,
            reverse=True,
        )
        for domain in sorted_domains:
            if domain.contains_point(x, y, z):
                return domain
        # Fallback to global
        return self.get_domain(0) or StructuralDomain(domain_id=0)

    def get_joint_sets_for_point(
        self, x: float, y: float, z: float
    ) -> Tuple[int, List[JointSetConfig]]:
        """Get the effective joint sets for a point in space.

        Args:
            x, y, z: Point coordinates.

        Returns:
            Tuple of (domain_id, list of JointSetConfig).
        """
        domain = self.find_domain(x, y, z)
        if domain.joint_sets:
            return domain.domain_id, domain.joint_sets
        # If domain has no specific joint sets, use global
        global_domain = self.get_domain(0)
        if global_domain and global_domain.joint_sets:
            return 0, global_domain.joint_sets
        return domain.domain_id, []
