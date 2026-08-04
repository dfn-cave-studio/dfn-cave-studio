"""
Rock mass and excavation mask models for DFN Cave Studio.

Defines spatial masks that partition the model volume into:
  - Valid rock mass (rock_mask)
  - Air/outside (everything else)
  - Excavations/caves (excavation_mask)
  - Surface boundary (surface_model)

All masks MUST distinguish between:
  - Active rock (data_valid AND within rock_mask AND NOT excavated)
  - Invalid/missing data (data_valid = False)
  - Air/void (outside rock_mask)
  - Excavated regions (inside excavation_mask)

References:
  - SCIENTIFIC_SPEC.md Section 7.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


# =============================================================================
# Mask Types
# =============================================================================

class MaskType(str, Enum):
    """Type of spatial mask."""
    SURFACE_BASED = "surface_based"   # Defined by upper/lower triangulated surfaces
    BOX = "box"                       # Simple axis-aligned box
    POLYGON = "polygon"               # 2D polygon extruded in Z
    VOXEL_FIELD = "voxel_field"       # Pre-computed voxel array
    DISTANCE_FIELD = "distance_field" # Signed distance function


# =============================================================================
# Surface Model
# =============================================================================

class SurfaceModel(BaseModel):
    """A triangulated surface representing topography or a geological contact.

    The surface is stored as vertices (N×3) and faces (M×3).
    The surface defines the boundary between "above" (air/void) and "below" (rock).
    """

    name: str = "Surface"
    vertices: Optional[List[List[float]]] = Field(
        default=None, description="N×3 vertex coordinates"
    )
    faces: Optional[List[List[int]]] = Field(
        default=None, description="M×3 triangle indices"
    )
    above_is_rock: bool = Field(
        default=False, description="True if rock is ABOVE the surface (e.g., hanging wall)"
    )
    surface_type: str = Field(
        default="topography", description="'topography', 'geological_contact', 'excavation_boundary'"
    )

    model_version: int = 1

    @property
    def vertex_array(self) -> Optional[NDArray[np.float64]]:
        """Vertices as numpy array (N, 3)."""
        if self.vertices is None:
            return None
        return np.array(self.vertices, dtype=np.float64)

    @property
    def face_array(self) -> Optional[NDArray[np.int64]]:
        """Faces as numpy array (M, 3)."""
        if self.faces is None:
            return None
        return np.array(self.faces, dtype=np.int64)

    def elevation_at(self, x: float, y: float) -> Optional[float]:
        """Query the surface elevation at a given (x, y) location.

        Uses barycentric interpolation on the triangulated surface.

        Args:
            x: X coordinate (easting, m).
            y: Y coordinate (northing, m).

        Returns:
            Z elevation at (x, y), or None if outside the mesh.
        """
        verts = self.vertex_array
        faces = self.face_array

        if verts is None or faces is None:
            return None

        point = np.array([x, y])

        for face in faces:
            if len(face) < 3:
                continue
            tri = verts[face[:3]]
            tri_2d = tri[:, :2]

            # Barycentric coordinates
            v0 = tri_2d[2] - tri_2d[0]
            v1 = tri_2d[1] - tri_2d[0]
            v2 = point - tri_2d[0]

            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)

            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-15:
                continue

            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w

            if -1e-12 <= u <= 1.0 + 1e-12 and -1e-12 <= v <= 1.0 + 1e-12 and -1e-12 <= w <= 1.0 + 1e-12:
                z = u * tri[0, 2] + v * tri[1, 2] + w * tri[2, 2]
                return float(z)

        return None

    def above_surface(self, x: float, y: float, z: float) -> Optional[bool]:
        """Check if a point is above the surface.

        Returns:
            True if above, False if below, None if outside mesh.
            Respects above_is_rock setting.
        """
        surf_z = self.elevation_at(x, y)
        if surf_z is None:
            return None

        is_above = z > surf_z
        if self.above_is_rock:
            return not is_above
        return is_above

    def to_trimesh(self):
        """Convert to a trimesh.Trimesh object for advanced processing."""
        import trimesh
        verts = self.vertex_array
        faces = self.face_array
        if verts is None or faces is None:
            return None
        return trimesh.Trimesh(vertices=verts, faces=faces)

    @classmethod
    def from_trimesh(cls, mesh, name: str = "Surface") -> "SurfaceModel":
        """Create a SurfaceModel from a trimesh.Trimesh object."""
        return cls(
            name=name,
            vertices=mesh.vertices.tolist(),
            faces=mesh.faces.tolist(),
        )


# =============================================================================
# Rock Mask
# =============================================================================

class RockMask(BaseModel):
    """Boolean mask defining valid rock mass within the model volume.

    Points inside the mask are valid rock; points outside are air/void.
    """

    name: str = "Rock Mask"
    mask_type: MaskType = MaskType.BOX
    enabled: bool = True

    # Surface-based mask
    upper_surface: Optional[SurfaceModel] = None
    lower_surface: Optional[SurfaceModel] = None

    # Box mask
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None

    model_version: int = 1

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a point is within the rock mass.

        Args:
            x, y, z: Point coordinates.

        Returns:
            True if the point is valid rock.
        """
        if not self.enabled:
            return True  # No mask: everything is rock

        if self.mask_type == MaskType.BOX:
            if any(v is None for v in [self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max]):
                return True
            return (
                self.x_min <= x <= self.x_max
                and self.y_min <= y <= self.y_max
                and self.z_min <= z <= self.z_max
            )

        elif self.mask_type == MaskType.SURFACE_BASED:
            # Check upper surface
            if self.upper_surface is not None:
                above_upper = self.upper_surface.above_surface(x, y, z)
                if above_upper is not None and above_upper:
                    return False

            # Check lower surface
            if self.lower_surface is not None:
                above_lower = self.lower_surface.above_surface(x, y, z)
                if above_lower is not None and not above_lower:
                    return False

            return True

        return True

    def contains_aabb(self, box_min: NDArray[np.float64], box_max: NDArray[np.float64]) -> int:
        """Check if an AABB is inside, outside, or partially inside the mask.

        Args:
            box_min: Minimum corner (3,).
            box_max: Maximum corner (3,).

        Returns:
            2 = fully inside, 1 = partially inside, 0 = fully outside.
        """
        corners = np.array([
            [box_min[0], box_min[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_max[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_max[0], box_max[1], box_max[2]],
        ])

        results = [self.contains_point(*c) for c in corners]
        n_inside = sum(results)

        if n_inside == 8:
            return 2  # Fully inside
        elif n_inside == 0:
            return 0  # Fully outside
        else:
            return 1  # Partial


# =============================================================================
# Excavation Mask
# =============================================================================

class ExcavationMask(BaseModel):
    """Mask defining excavated/caved regions (voids) within the rock mass.

    Excavations include:
      - Mine development (drifts, ramps)
      - Undercut / drawbell excavations
      - Cave zone (propagating upward from undercut)
      - Historical stopes

    Points inside the excavation mask are void (not rock).
    """

    name: str = "Excavation Mask"
    enabled: bool = True
    excavation_type: str = Field(default="undercut", description="'undercut', 'development', 'cave', 'stope'")

    # Box-based excavations (simplest)
    excavation_boxes: List[Tuple[float, float, float, float, float, float]] = Field(
        default_factory=list,
        description="List of (x_min, x_max, y_min, y_max, z_min, z_max) boxes"
    )

    # Surface-based
    excavation_meshes: List[SurfaceModel] = Field(default_factory=list)

    model_version: int = 1

    def is_excavated(self, x: float, y: float, z: float) -> bool:
        """Check if a point is within an excavation.

        Returns:
            True if the point has been excavated (is void).
        """
        if not self.enabled:
            return False

        for box in self.excavation_boxes:
            x_min, x_max, y_min, y_max, z_min, z_max = box
            if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                return True

        for mesh in self.excavation_meshes:
            # Simplified: check if point is inside the mesh
            # Full implementation would use trimesh.contains()
            pass

        return False


# =============================================================================
# Spatial Attributes (per-voxel metadata container)
# =============================================================================

class SpatialAttributeConfig(BaseModel):
    """Configuration for spatial attributes stored per voxel.

    Defines which masks and attributes are computed and stored.
    """

    compute_active_mask: bool = True
    compute_material_id: bool = True
    compute_domain_id: bool = True
    compute_excavation_mask: bool = True
    compute_data_valid_mask: bool = True

    # Default values (MUST NOT be 0 if 0 means "missing")
    active_default: int = 1        # 1 = active rock
    inactive_default: int = 0      # 0 = inactive
    void_default: int = -1         # -1 = void/outside
    invalid_default: int = -2      # -2 = data invalid
