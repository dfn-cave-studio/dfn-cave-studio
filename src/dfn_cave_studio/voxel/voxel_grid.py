"""
Voxel grid with chunked and sparse storage for DFN Cave Studio.

Key design decisions:
  - Chunked storage: large grids split into manageable blocks (default 32³)
  - Sparse mode: only store non-air voxels
  - Multi-attribute: per-voxel arrays with named attributes
  - Mask system: active != material_id != void (no zero ambiguity)
  - Lazy loading: chunks loaded on demand
  - Memory warning: estimate before allocating

DO NOT:
  - Use 0 to mean both "active" and "void" simultaneously
  - Allocate dense 1000³ arrays without warning
  - Treat voxel boundaries as fracture boundaries

References:
  - SCIENTIFIC_SPEC.md Section 7, 8.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.rock_mask import RockMask, ExcavationMask, SpatialAttributeConfig


# =============================================================================
# Constants
# =============================================================================

# Sentinel values for mask attributes (must NOT overlap with valid data)
VOID_VALUE = -1       # Outside rock mass (air)
INVALID_VALUE = -2    # Data invalid or uninitialized
INACTIVE_VALUE = 0    # Inactive (inside rock but excluded from computation)
ACTIVE_VALUE = 1      # Active rock mass

# Valid material IDs start at 1; 0 means "no material assigned"
MATERIAL_VOID = 0


# =============================================================================
# Voxel Chunk
# =============================================================================

@dataclass
class VoxelChunk:
    """A single chunk of the voxel grid.

    Each chunk is a dense 3D block of size (chunk_size × chunk_size × chunk_size).
    The chunk stores multiple named attribute arrays.

    Chunks are identified by their origin indices (ix0, iy0, iz0) in the
    global voxel grid coordinate system.
    """

    ix0: int  # Starting X index of this chunk in global grid
    iy0: int  # Starting Y index
    iz0: int  # Starting Z index
    chunk_size: int = 32

    # Attribute arrays (lazy-allocated)
    _attrs: Dict[str, NDArray[np.int32]] = field(default_factory=dict)

    # Spatial metadata
    x_min: float = 0.0
    y_min: float = 0.0
    z_min: float = 0.0
    cell_size: float = 1.0

    def ensure_attr(self, name: str, default_value: int = INVALID_VALUE) -> NDArray[np.int32]:
        """Get or create an attribute array for this chunk."""
        if name not in self._attrs:
            self._attrs[name] = np.full(
                (self.chunk_size, self.chunk_size, self.chunk_size),
                default_value, dtype=np.int32,
            )
        return self._attrs[name]

    def has_attr(self, name: str) -> bool:
        return name in self._attrs

    def get_attr(self, name: str) -> Optional[NDArray[np.int32]]:
        return self._attrs.get(name)

    def set_voxel(self, i: int, j: int, k: int, attr_name: str, value: int) -> None:
        """Set a single voxel attribute value (local chunk coords)."""
        if 0 <= i < self.chunk_size and 0 <= j < self.chunk_size and 0 <= k < self.chunk_size:
            arr = self.ensure_attr(attr_name)
            arr[i, j, k] = value

    def get_voxel(self, i: int, j: int, k: int, attr_name: str) -> Optional[int]:
        """Get a single voxel attribute value."""
        arr = self._attrs.get(attr_name)
        if arr is not None and 0 <= i < self.chunk_size and 0 <= j < self.chunk_size and 0 <= k < self.chunk_size:
            return int(arr[i, j, k])
        return None

    def active_count(self) -> int:
        """Count active voxels in this chunk."""
        active = self._attrs.get("active_mask")
        if active is None:
            return 0
        return int(np.sum(active == ACTIVE_VALUE))

    def is_empty(self) -> bool:
        """Check if chunk has any active voxels."""
        return self.active_count() == 0

    def memory_bytes(self) -> int:
        """Estimate memory usage of this chunk."""
        total = 0
        for arr in self._attrs.values():
            total += arr.nbytes
        return total


# =============================================================================
# Voxel Grid
# =============================================================================

class VoxelGrid:
    """Chunked, multi-attribute voxel grid.

    The grid organizes voxels into chunks for memory efficiency.
    Supports sparse mode where only non-empty chunks are stored.

    Usage:
        grid = VoxelGrid.from_bounds(bounds, voxel_config)
        grid.set_voxel(10, 20, 30, "active_mask", ACTIVE_VALUE)
        value = grid.get_voxel(10, 20, 30, "active_mask")
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        cell_size: float = 1.0,
        x_min: float = 0.0,
        y_min: float = 0.0,
        z_min: float = 0.0,
        cell_size_x: Optional[float] = None,
        cell_size_y: Optional[float] = None,
        cell_size_z: Optional[float] = None,
        chunk_size: int = 32,
        sparse: bool = True,
    ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        # Support non-uniform cell sizes (dx, dy, dz may differ)
        self.cell_size_x = cell_size_x if cell_size_x is not None else cell_size
        self.cell_size_y = cell_size_y if cell_size_y is not None else cell_size
        self.cell_size_z = cell_size_z if cell_size_z is not None else cell_size
        # Backwards-compatible alias
        self.cell_size = cell_size
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.chunk_size = chunk_size
        self.sparse = sparse

        # Compute chunk layout
        self.n_chunks_x = max(1, math.ceil(nx / chunk_size))
        self.n_chunks_y = max(1, math.ceil(ny / chunk_size))
        self.n_chunks_z = max(1, math.ceil(nz / chunk_size))

        self._chunks: Dict[Tuple[int, int, int], VoxelChunk] = {}
        self._attr_config = SpatialAttributeConfig()

    # ── Factory Methods ──────────────────────────────────────────────────

    @classmethod
    def from_bounds(
        cls,
        bounds: ModelBounds,
        voxel_config: VoxelConfig,
        chunk_size: int = 32,
        sparse: bool = True,
    ) -> "VoxelGrid":
        """Create a VoxelGrid from model bounds and voxel configuration.

        Args:
            bounds: Model bounding box.
            voxel_config: Voxel discretization parameters.
            chunk_size: Side length of each chunk in voxels.
            sparse: If True, only allocate chunks when data is written.

        Returns:
            Configured VoxelGrid instance.
        """
        nx, ny, nz = voxel_config.compute_grid_dimensions(bounds)
        return cls(
            nx=nx, ny=ny, nz=nz,
            cell_size=voxel_config.cell_size_x,
            cell_size_x=voxel_config.cell_size_x,
            cell_size_y=voxel_config.cell_size_y,
            cell_size_z=voxel_config.cell_size_z,
            x_min=bounds.x_min, y_min=bounds.y_min, z_min=bounds.z_min,
            chunk_size=chunk_size, sparse=sparse,
        )

    @classmethod
    def estimate_memory_mb(
        cls,
        bounds: ModelBounds,
        voxel_config: VoxelConfig,
        n_attributes: int = 5,
        sparse_fraction: float = 0.3,
    ) -> float:
        """Estimate memory usage for a grid with given parameters.

        Args:
            bounds: Model bounds.
            voxel_config: Voxel configuration.
            n_attributes: Number of attributes per voxel.
            sparse_fraction: Fraction of voxels expected to be active.

        Returns:
            Estimated memory in MB.
        """
        nx, ny, nz = voxel_config.compute_grid_dimensions(bounds)
        total_voxels = nx * ny * nz
        active_voxels = total_voxels * sparse_fraction
        bytes_per_voxel = 4 * n_attributes  # int32 per attribute
        return (active_voxels * bytes_per_voxel) / (1024 * 1024)

    @classmethod
    def is_dangerous(
        cls,
        bounds: ModelBounds,
        voxel_config: VoxelConfig,
        max_memory_gb: float = 2.0,
    ) -> Tuple[bool, str]:
        """Check if the grid configuration would be dangerous.

        Returns:
            Tuple of (is_dangerous, reason_string).
        """
        nx, ny, nz = voxel_config.compute_grid_dimensions(bounds)
        total_voxels = nx * ny * nz

        if total_voxels > 1_000_000_000:
            return True, f"{total_voxels:,} total voxels > 1 billion — extreme resolution"
        if total_voxels > 100_000_000:
            return True, f"{total_voxels:,} total voxels > 100M — consider coarser grid"

        mem_mb = cls.estimate_memory_mb(bounds, voxel_config, 8, 0.5)
        if mem_mb > max_memory_gb * 1024:
            return True, f"Estimated {mem_mb:.0f} MB > {max_memory_gb} GB threshold"

        return False, "OK"

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def total_voxels(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def active_voxel_count(self) -> int:
        return sum(chunk.active_count() for chunk in self._chunks.values())

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Model bounds in meters (x_min, x_max, y_min, y_max, z_min, z_max)."""
        return (
            self.x_min, self.x_min + self.nx * self.cell_size_x,
            self.y_min, self.y_min + self.ny * self.cell_size_y,
            self.z_min, self.z_min + self.nz * self.cell_size_z,
        )

    @property
    def cell_volume(self) -> float:
        """Volume of a single voxel cell (m³)."""
        return self.cell_size_x * self.cell_size_y * self.cell_size_z

    # ── Index Conversion ─────────────────────────────────────────────────

    def world_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates (m) to voxel indices.

        Returns:
            Tuple of (ix, iy, iz). May be outside grid.
        """
        ix = int((x - self.x_min) / self.cell_size_x)
        iy = int((y - self.y_min) / self.cell_size_y)
        iz = int((z - self.z_min) / self.cell_size_z)
        return ix, iy, iz

    def voxel_to_world(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        """Convert voxel indices to world coordinates (voxel center).

        Returns:
            Tuple of (x, y, z) in meters.
        """
        x = self.x_min + (ix + 0.5) * self.cell_size_x
        y = self.y_min + (iy + 0.5) * self.cell_size_y
        z = self.z_min + (iz + 0.5) * self.cell_size_z
        return x, y, z

    def is_inside(self, ix: int, iy: int, iz: int) -> bool:
        """Check if voxel indices are within the grid."""
        return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz

    # ── Chunk Management ─────────────────────────────────────────────────

    def _chunk_key(self, ix: int, iy: int, iz: int) -> Tuple[int, int, int]:
        """Get the chunk key for given global voxel indices."""
        return (ix // self.chunk_size, iy // self.chunk_size, iz // self.chunk_size)

    def _get_or_create_chunk(self, cx: int, cy: int, cz: int) -> VoxelChunk:
        """Get or create a chunk by chunk coordinates."""
        key = (cx, cy, cz)
        if key not in self._chunks:
            chunk = VoxelChunk(
                ix0=cx * self.chunk_size,
                iy0=cy * self.chunk_size,
                iz0=cz * self.chunk_size,
                chunk_size=self.chunk_size,
                x_min=self.x_min, y_min=self.y_min, z_min=self.z_min,
                cell_size=self.cell_size,
            )
            if not self.sparse:
                # Pre-allocate standard attributes
                for attr in ["active_mask", "material_id", "domain_id", "excavation_mask", "data_valid"]:
                    chunk.ensure_attr(attr)
            self._chunks[key] = chunk
        return self._chunks[key]

    def _get_chunk(self, cx: int, cy: int, cz: int) -> Optional[VoxelChunk]:
        """Get an existing chunk, or None if sparse and not yet created."""
        return self._chunks.get((cx, cy, cz))

    # ── Data Access ──────────────────────────────────────────────────────

    def set_voxel(self, ix: int, iy: int, iz: int, attr_name: str, value: int) -> None:
        """Set an attribute for a single voxel.

        Args:
            ix, iy, iz: Global voxel indices.
            attr_name: Attribute name (e.g., "active_mask", "material_id").
            value: Integer value to set.
        """
        if not self.is_inside(ix, iy, iz):
            return
        cx, cy, cz = self._chunk_key(ix, iy, iz)
        chunk = self._get_or_create_chunk(cx, cy, cz)
        li = ix - chunk.ix0
        lj = iy - chunk.iy0
        lk = iz - chunk.iz0
        chunk.set_voxel(li, lj, lk, attr_name, value)

    def get_voxel(self, ix: int, iy: int, iz: int, attr_name: str) -> Optional[int]:
        """Get an attribute for a single voxel.

        Returns:
            Attribute value, or None if chunk not allocated (sparse mode).
        """
        if not self.is_inside(ix, iy, iz):
            return None
        cx, cy, cz = self._chunk_key(ix, iy, iz)
        chunk = self._get_chunk(cx, cy, cz)
        if chunk is None:
            return None  # Sparse: chunk not yet created → void
        li = ix - chunk.ix0
        lj = iy - chunk.iy0
        lk = iz - chunk.iz0
        return chunk.get_voxel(li, lj, lk, attr_name)

    def is_active(self, ix: int, iy: int, iz: int) -> bool:
        """Check if a voxel is active rock mass."""
        val = self.get_voxel(ix, iy, iz, "active_mask")
        return val == ACTIVE_VALUE

    def is_void(self, ix: int, iy: int, iz: int) -> bool:
        """Check if a voxel is void (outside rock mass)."""
        return not self.is_active(ix, iy, iz)

    # ── Bulk Operations ──────────────────────────────────────────────────

    def apply_mask(
        self,
        rock_mask: RockMask,
        excavation_mask: Optional[ExcavationMask] = None,
    ) -> int:
        """Apply rock and excavation masks to the entire grid.

        Sets active_mask for every voxel based on whether it's in valid rock
        and not excavated.

        Args:
            rock_mask: Rock mass mask.
            excavation_mask: Optional excavation mask.

        Returns:
            Number of active voxels.
        """
        count = 0
        for iz in range(self.nz):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    x, y, z = self.voxel_to_world(ix, iy, iz)

                    if not rock_mask.contains_point(x, y, z):
                        self.set_voxel(ix, iy, iz, "active_mask", INACTIVE_VALUE)
                        continue

                    if excavation_mask and excavation_mask.is_excavated(x, y, z):
                        self.set_voxel(ix, iy, iz, "active_mask", INACTIVE_VALUE)
                        self.set_voxel(ix, iy, iz, "excavation_mask", 1)
                        continue

                    self.set_voxel(ix, iy, iz, "active_mask", ACTIVE_VALUE)
                    count += 1

        return count

    def apply_domain_ids(
        self,
        domain_collection,
    ) -> None:
        """Assign domain_id to each voxel based on spatial domain lookup.

        Args:
            domain_collection: StructuralDomainCollection instance.
        """
        for iz in range(self.nz):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    if not self.is_active(ix, iy, iz):
                        continue
                    x, y, z = self.voxel_to_world(ix, iy, iz)
                    domain = domain_collection.find_domain(x, y, z)
                    self.set_voxel(ix, iy, iz, "domain_id", domain.domain_id)

    # ── Iteration ────────────────────────────────────────────────────────

    def iter_active_voxels(self):
        """Yield (ix, iy, iz) for all active voxels."""
        for key, chunk in self._chunks.items():
            active = chunk.get_attr("active_mask")
            if active is None:
                continue
            for li in range(chunk.chunk_size):
                for lj in range(chunk.chunk_size):
                    for lk in range(chunk.chunk_size):
                        if active[li, lj, lk] == ACTIVE_VALUE:
                            yield li + chunk.ix0, lj + chunk.iy0, lk + chunk.iz0

    def iter_chunks(self):
        """Yield (cx, cy, cz, chunk) for all allocated chunks."""
        for key, chunk in self._chunks.items():
            yield key[0], key[1], key[2], chunk

    # ── Statistics ───────────────────────────────────────────────────────

    def memory_estimate_mb(self) -> float:
        """Estimate current memory usage."""
        total = 0
        for chunk in self._chunks.values():
            total += chunk.memory_bytes()
        return total / (1024 * 1024)

    def summary(self) -> str:
        """Return a human-readable summary of the grid."""
        return (
            f"VoxelGrid: {self.nx}×{self.ny}×{self.nz} = {self.total_voxels:,} total voxels\n"
            f"  Cell sizes: dx={self.cell_size_x:.2f} dy={self.cell_size_y:.2f} dz={self.cell_size_z:.2f} m\n"
            f"  Chunks: {self.chunk_count} ({self.n_chunks_x}×{self.n_chunks_y}×{self.n_chunks_z} layout)\n"
            f"  Active voxels: {self.active_voxel_count:,}\n"
            f"  Sparse mode: {self.sparse}\n"
            f"  Memory: ~{self.memory_estimate_mb():.1f} MB"
        )
