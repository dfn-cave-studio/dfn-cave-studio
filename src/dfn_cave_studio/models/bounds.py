"""Model bounds and spatial reference models."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ModelBounds(BaseModel):
    """Axis-aligned bounding box defining the model domain.

    All coordinates in meters. The coordinate system is right-handed:
    X = Easting, Y = Northing, Z = Elevation.
    """

    x_min: float = Field(..., description="Minimum X coordinate (easting, m)")
    x_max: float = Field(..., description="Maximum X coordinate (easting, m)")
    y_min: float = Field(..., description="Minimum Y coordinate (northing, m)")
    y_max: float = Field(..., description="Maximum Y coordinate (northing, m)")
    z_min: float = Field(..., description="Minimum Z coordinate (elevation, m)")
    z_max: float = Field(..., description="Maximum Z coordinate (elevation, m)")

    origin_x: float = Field(default=0.0, description="Origin X offset for local coordinates")
    origin_y: float = Field(default=0.0, description="Origin Y offset for local coordinates")
    origin_z: float = Field(default=0.0, description="Origin Z offset for local coordinates")

    @model_validator(mode="after")
    def validate_bounds(self) -> "ModelBounds":
        """Validate that min < max for all dimensions."""
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be < x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be < y_max ({self.y_max})")
        if self.z_min >= self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be < z_max ({self.z_max})")
        return self

    # ------------------------------------------------------------------
    # Derived Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        """Model extent in X direction (m)."""
        return self.x_max - self.x_min

    @property
    def depth(self) -> float:
        """Model extent in Y direction (m)."""
        return self.y_max - self.y_min

    @property
    def height(self) -> float:
        """Model extent in Z direction (m)."""
        return self.z_max - self.z_min

    @property
    def volume(self) -> float:
        """Model volume (m³)."""
        return self.width * self.depth * self.height

    @property
    def center(self) -> tuple[float, float, float]:
        """Center point of the bounding box (m)."""
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
            (self.z_min + self.z_max) / 2.0,
        )

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def contains_point(
        self, x: float, y: float, z: float, tol: float = 1e-10
    ) -> bool:
        """Check if a point is within the model bounds.

        Args:
            x: X coordinate (m).
            y: Y coordinate (m).
            z: Z coordinate (m).
            tol: Numerical tolerance (m).

        Returns:
            True if the point is within bounds.
        """
        return (
            (self.x_min - tol) <= x <= (self.x_max + tol)
            and (self.y_min - tol) <= y <= (self.y_max + tol)
            and (self.z_min - tol) <= z <= (self.z_max + tol)
        )

    def to_array(self) -> tuple[float, float, float, float, float, float]:
        """Return bounds as a flat tuple (x_min, x_max, y_min, y_max, z_min, z_max)."""
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def to_numpy(self):
        """Return bounds as a numpy array of shape (3, 2)."""
        import numpy as np
        return np.array([
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        ])


class VoxelConfig(BaseModel):
    """Configuration for voxel grid discretization."""

    cell_size_x: float = Field(default=1.0, gt=0.0, description="Voxel cell size in X (m)")
    cell_size_y: float = Field(default=1.0, gt=0.0, description="Voxel cell size in Y (m)")
    cell_size_z: float = Field(default=1.0, gt=0.0, description="Voxel cell size in Z (m)")

    @property
    def nx(self) -> Optional[int]:
        """Number of voxels in X. Set after bounds are provided."""
        return None  # Must be computed with bounds

    @property
    def cell_volume(self) -> float:
        """Volume of a single voxel cell (m³)."""
        return self.cell_size_x * self.cell_size_y * self.cell_size_z

    def compute_grid_dimensions(self, bounds: ModelBounds) -> tuple[int, int, int]:
        """Compute the number of voxels in each dimension given model bounds.

        Uses math.ceil to ensure the entire model volume is covered —
        int() would truncate and leave a gap.

        Args:
            bounds: Model bounding box.

        Returns:
            Tuple of (nx, ny, nz).
        """
        import math
        nx = max(1, math.ceil(bounds.width / self.cell_size_x))
        ny = max(1, math.ceil(bounds.depth / self.cell_size_y))
        nz = max(1, math.ceil(bounds.height / self.cell_size_z))
        return nx, ny, nz

    def estimate_memory_mb(self, bounds: ModelBounds, n_attributes: int = 1) -> float:
        """Estimate memory usage for a dense grid.

        Args:
            bounds: Model bounds.
            n_attributes: Number of attributes per voxel.

        Returns:
            Estimated memory in MB (float32 per attribute).
        """
        nx, ny, nz = self.compute_grid_dimensions(bounds)
        n_voxels = nx * ny * nz
        bytes_per_voxel = 4 * n_attributes  # float32
        return (n_voxels * bytes_per_voxel) / (1024 * 1024)

    def is_dangerous_resolution(self, bounds: ModelBounds, max_memory_gb: float = 2.0) -> bool:
        """Check if the resolution would require dangerous amounts of memory.

        Args:
            bounds: Model bounds.
            max_memory_gb: Maximum acceptable memory in GB.

        Returns:
            True if the resolution is dangerous.
        """
        mem_mb = self.estimate_memory_mb(bounds, n_attributes=8)  # 8 typical attributes
        return mem_mb > (max_memory_gb * 1024)
