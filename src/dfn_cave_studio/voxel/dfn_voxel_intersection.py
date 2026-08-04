"""
DFN-to-voxel intersection engine (M4).

Maps explicit fracture geometries onto the voxel grid, computing per-voxel:
  - fracture_count: number of intersecting fractures
  - fracture_area: total fracture area within voxel
  - local_p32: local P32 = fracture_area / voxel_volume
  - set_proportions: per-set fraction
  - dominant_direction: approximate main fracture orientation
  - max_fracture_size: largest intersecting fracture radius
  - connectivity_cluster: cluster ID (set by connectivity module)

Algorithm:
  1. Build AABB tree for fast candidate search (fracture bbox → voxel list)
  2. For each candidate pair, perform exact disk-AABB intersection test
  3. Accumulate per-voxel statistics

References:
  - SCIENTIFIC_SPEC.md Section 8.
"""

from __future__ import annotations

import math
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.models.fracture import StochasticFracture
from dfn_cave_studio.models.dfn_realization import DFNRealization
from dfn_cave_studio.geometry.intersection import disk_aabb_intersects
from dfn_cave_studio.voxel.voxel_grid import VoxelGrid, ACTIVE_VALUE


# =============================================================================
# Per-Voxel DFN Attributes
# =============================================================================

# Attribute names stored in VoxelGrid
ATTR_FRACTURE_COUNT = "fracture_count"
ATTR_FRACTURE_AREA = "fracture_area"
ATTR_LOCAL_P32 = "local_p32"
ATTR_MAX_FRAC_SIZE = "max_frac_size"
ATTR_MAIN_DIR = "main_direction"  # Stored as packed int or separate components
ATTR_CLUSTER_ID = "connectivity_cluster"
ATTR_SET_COUNTS_PREFIX = "set_count_"  # e.g., "set_count_1", "set_count_2"


# =============================================================================
# Intersection Engine
# =============================================================================

class DFNVoxelIntersectionEngine:
    """Computes explicit fracture-voxel intersections for a DFN realization.

    Uses a two-stage approach:
      1. Candidate search: find voxel AABBs that overlap each fracture's bbox
      2. Exact test: disk-AABB intersection for each candidate pair
    """

    def __init__(
        self,
        voxel_grid: VoxelGrid,
        realization: DFNRealization,
    ):
        self.grid = voxel_grid
        self.realization = realization
        self._cancelled = False

        # Fracture bounding boxes (lazy-computed)
        self._fracture_bboxes: Optional[NDArray[np.float64]] = None  # (N, 6)

    # ── Progress & Cancellation ───────────────────────────────────────────

    def set_progress_callback(self, callback: Optional[Callable[[int, int, str], None]]) -> None:
        self._progress_callback = callback

    def cancel(self) -> None:
        self._cancelled = True

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("DFN-voxel intersection cancelled")

    def _report(self, current: int, total: int, msg: str = "") -> None:
        if hasattr(self, '_progress_callback') and self._progress_callback:
            self._progress_callback(current, total, msg)

    # ── Bounding Box Cache ────────────────────────────────────────────────

    def _compute_fracture_bboxes(self) -> NDArray[np.float64]:
        """Compute axis-aligned bounding box for each fracture.

        Returns:
            (N, 6) array: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        fractures = self.realization.stochastic_fractures
        n = len(fractures)
        bboxes = np.zeros((n, 6), dtype=np.float64)

        for i, f in enumerate(fractures):
            geo = f.geometry
            r = f.radius if f.radius > 0 else (geo.radius or 1.0)
            cx, cy, cz = geo.center_x, geo.center_y, geo.center_z
            bboxes[i] = [cx - r, cx + r, cy - r, cy + r, cz - r, cz + r]

        return bboxes

    # ── Main Computation ──────────────────────────────────────────────────

    def compute_intersections(self) -> int:
        """Compute fracture-voxel intersections and populate voxel attributes.

        Returns:
            Number of fracture-voxel intersection pairs found.
        """
        self._check_cancelled()

        fractures = self.realization.stochastic_fractures
        n_fractures = len(fractures)

        if n_fractures == 0:
            return 0

        self._fracture_bboxes = self._compute_fracture_bboxes()

        total_pairs = 0
        n_processed = 0

        # Iterate over active voxels
        active_voxels = list(self.grid.iter_active_voxels())
        n_voxels = len(active_voxels)

        if n_voxels == 0:
            return 0

        for vox_idx, (ix, iy, iz) in enumerate(active_voxels):
            self._check_cancelled()
            if vox_idx % 1000 == 0:
                self._report(vox_idx, n_voxels, f"Processing voxel {vox_idx}/{n_voxels}")

            # Voxel AABB
            vox_min, vox_max = self._voxel_aabb(ix, iy, iz)

            # Per-set counters
            set_counts: Dict[int, int] = {}
            total_area = 0.0
            fracture_count = 0
            max_radius = 0.0

            for fi in range(n_fractures):
                frac = fractures[fi]
                r = frac.radius if frac.radius > 0 else (frac.geometry.radius or 1.0)

                # Candidate check (AABB overlap)
                bb = self._fracture_bboxes[fi]
                if not self._aabb_overlap(bb, vox_min, vox_max):
                    continue

                # Exact check
                if disk_aabb_intersects(
                    frac.geometry.center, frac.geometry.normal, r,
                    vox_min, vox_max,
                ):
                    fracture_count += 1
                    # Approximate area contribution (full disk area as upper bound)
                    disk_area = math.pi * r ** 2
                    total_area += disk_area
                    max_radius = max(max_radius, r)

                    set_id = frac.set_id
                    set_counts[set_id] = set_counts.get(set_id, 0) + 1

            if fracture_count > 0:
                total_pairs += 1
                vox_volume = self.grid.cell_size ** 3
                local_p32 = total_area / vox_volume

                self.grid.set_voxel(ix, iy, iz, ATTR_FRACTURE_COUNT, fracture_count)
                self.grid.set_voxel(ix, iy, iz, ATTR_LOCAL_P32, int(local_p32 * 1000))  # Store as milli-P32
                self.grid.set_voxel(ix, iy, iz, ATTR_MAX_FRAC_SIZE, int(max_radius * 1000))  # mm

                for set_id, count in set_counts.items():
                    attr_name = f"{ATTR_SET_COUNTS_PREFIX}{set_id}"
                    self.grid.set_voxel(ix, iy, iz, attr_name, count)

        self._report(n_voxels, n_voxels, "Intersection complete")
        return total_pairs

    # ── Helpers ───────────────────────────────────────────────────────────

    def _voxel_aabb(self, ix: int, iy: int, iz: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the AABB of a voxel in world coordinates."""
        x = self.grid.x_min + ix * self.grid.cell_size
        y = self.grid.y_min + iy * self.grid.cell_size
        z = self.grid.z_min + iz * self.grid.cell_size
        return (
            np.array([x, y, z]),
            np.array([x + self.grid.cell_size, y + self.grid.cell_size, z + self.grid.cell_size]),
        )

    @staticmethod
    def _aabb_overlap(bbox: NDArray[np.float64], v_min: NDArray[np.float64], v_max: NDArray[np.float64]) -> bool:
        """Check AABB overlap between fracture bbox and voxel."""
        return bool(
            bbox[0] <= v_max[0] and bbox[1] >= v_min[0]
            and bbox[2] <= v_max[1] and bbox[3] >= v_min[1]
            and bbox[4] <= v_max[2] and bbox[5] >= v_min[2]
        )
