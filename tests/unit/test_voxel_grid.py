"""Tests for voxel grid module (M3)."""

import math
import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.rock_mask import RockMask, MaskType
from dfn_cave_studio.voxel.voxel_grid import (
    VoxelGrid, VoxelChunk,
    ACTIVE_VALUE, INACTIVE_VALUE, INVALID_VALUE, VOID_VALUE,
)


class TestVoxelChunk:
    def test_create_chunk(self):
        chunk = VoxelChunk(ix0=0, iy0=0, iz0=0, chunk_size=16)
        assert chunk.chunk_size == 16
        assert chunk.is_empty()

    def test_ensure_attr(self):
        chunk = VoxelChunk(ix0=0, iy0=0, iz0=0, chunk_size=8)
        arr = chunk.ensure_attr("active_mask")
        assert arr.shape == (8, 8, 8)
        assert arr.dtype == np.int32

    def test_set_get_voxel(self):
        chunk = VoxelChunk(ix0=0, iy0=0, iz0=0, chunk_size=8)
        chunk.set_voxel(3, 4, 5, "active_mask", ACTIVE_VALUE)
        assert chunk.get_voxel(3, 4, 5, "active_mask") == ACTIVE_VALUE

    def test_out_of_bounds(self):
        chunk = VoxelChunk(ix0=0, iy0=0, iz0=0, chunk_size=8)
        chunk.set_voxel(100, 100, 100, "active_mask", ACTIVE_VALUE)
        assert chunk.get_voxel(100, 100, 100, "active_mask") is None

    def test_active_count(self):
        chunk = VoxelChunk(ix0=0, iy0=0, iz0=0, chunk_size=4)
        chunk.set_voxel(0, 0, 0, "active_mask", ACTIVE_VALUE)
        chunk.set_voxel(1, 1, 1, "active_mask", ACTIVE_VALUE)
        assert chunk.active_count() == 2

    def test_is_empty(self):
        chunk = VoxelChunk(ix0=0, iy0=0, iz0=0, chunk_size=4)
        assert chunk.is_empty()
        chunk.set_voxel(0, 0, 0, "active_mask", ACTIVE_VALUE)
        assert not chunk.is_empty()

    def test_multiple_attributes(self):
        chunk = VoxelChunk(ix0=0, iy0=0, iz0=0, chunk_size=8)
        chunk.set_voxel(2, 2, 2, "active_mask", ACTIVE_VALUE)
        chunk.set_voxel(2, 2, 2, "material_id", 3)
        chunk.set_voxel(2, 2, 2, "domain_id", 1)
        assert chunk.get_voxel(2, 2, 2, "active_mask") == ACTIVE_VALUE
        assert chunk.get_voxel(2, 2, 2, "material_id") == 3
        assert chunk.get_voxel(2, 2, 2, "domain_id") == 1


class TestVoxelGrid:
    @pytest.fixture
    def small_grid(self) -> VoxelGrid:
        return VoxelGrid(nx=10, ny=10, nz=10, cell_size=1.0, chunk_size=4, sparse=True)

    @pytest.fixture
    def grid_from_bounds(self) -> VoxelGrid:
        bounds = ModelBounds(x_min=0, x_max=50, y_min=0, y_max=50, z_min=0, z_max=50)
        vc = VoxelConfig(cell_size_x=2.0, cell_size_y=2.0, cell_size_z=2.0)
        return VoxelGrid.from_bounds(bounds, vc, chunk_size=8, sparse=True)

    def test_create_small_grid(self, small_grid):
        assert small_grid.nx == 10
        assert small_grid.total_voxels == 1000

    def test_chunk_layout(self, small_grid):
        # 10 voxels, chunk_size=4 → 3 chunks per axis (4+4+2)
        assert small_grid.n_chunks_x == 3  # ceil(10/4) = 3

    def test_world_to_voxel(self, small_grid):
        ix, iy, iz = small_grid.world_to_voxel(0.5, 0.5, 0.5)
        assert ix == 0
        assert iy == 0
        assert iz == 0

    def test_voxel_to_world(self, small_grid):
        x, y, z = small_grid.voxel_to_world(0, 0, 0)
        assert x == 0.5
        assert y == 0.5
        assert z == 0.5

    def test_roundtrip(self, small_grid):
        ix, iy, iz = 5, 5, 5
        x, y, z = small_grid.voxel_to_world(ix, iy, iz)
        ix2, iy2, iz2 = small_grid.world_to_voxel(x, y, z)
        assert ix == ix2
        assert iy == iy2

    def test_is_inside(self, small_grid):
        assert small_grid.is_inside(0, 0, 0)
        assert small_grid.is_inside(9, 9, 9)
        assert not small_grid.is_inside(10, 0, 0)

    def test_set_get_voxel(self, small_grid):
        small_grid.set_voxel(5, 5, 5, "active_mask", ACTIVE_VALUE)
        assert small_grid.get_voxel(5, 5, 5, "active_mask") == ACTIVE_VALUE

    def test_sparse_no_chunk_for_unused(self, small_grid):
        # In sparse mode, voxels that haven't been set should return None
        assert small_grid.get_voxel(0, 0, 0, "active_mask") is None
        assert small_grid.chunk_count == 0

    def test_chunk_created_on_write(self, small_grid):
        small_grid.set_voxel(0, 0, 0, "active_mask", ACTIVE_VALUE)
        assert small_grid.chunk_count >= 1

    def test_is_active(self, small_grid):
        small_grid.set_voxel(3, 3, 3, "active_mask", ACTIVE_VALUE)
        assert small_grid.is_active(3, 3, 3)
        assert not small_grid.is_active(0, 0, 0)  # None → not active

    def test_apply_box_mask(self):
        # 20³ grid spanning 0-100m with 5m cells. Mask covers 0-50m → 10³ active voxels.
        grid = VoxelGrid(nx=20, ny=20, nz=20, cell_size=5.0,
                         x_min=0, y_min=0, z_min=0, chunk_size=8, sparse=False)
        mask = RockMask(
            mask_type=MaskType.BOX,
            x_min=0, x_max=50, y_min=0, y_max=50, z_min=0, z_max=50,
        )
        count = grid.apply_mask(mask)
        assert count == 1000  # 10³ voxels with centers in [0,50]

    def test_apply_partial_mask(self):
        grid = VoxelGrid(nx=10, ny=10, nz=10, cell_size=1.0,
                         x_min=0, y_min=0, z_min=0, chunk_size=5, sparse=False)
        mask = RockMask(
            mask_type=MaskType.BOX,
            x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5,
        )
        count = grid.apply_mask(mask)
        assert count < 1000  # Only fraction of voxels active

    def test_iter_active_voxels(self):
        grid = VoxelGrid(nx=5, ny=5, nz=5, cell_size=1.0,
                         x_min=0, y_min=0, z_min=0, chunk_size=4, sparse=False)
        mask = RockMask(
            mask_type=MaskType.BOX,
            x_min=0, x_max=3, y_min=0, y_max=3, z_min=0, z_max=3,
        )
        grid.apply_mask(mask)
        active = list(grid.iter_active_voxels())
        assert len(active) > 0

    def test_memory_estimate(self, small_grid):
        mem = small_grid.memory_estimate_mb()
        assert mem >= 0.0

    def test_summary(self, small_grid):
        s = small_grid.summary()
        assert "10×10×10" in s
        assert "Sparse mode: True" in s

    def test_grid_from_bounds(self, grid_from_bounds):
        assert grid_from_bounds.nx == 25  # 50m / 2m
        assert grid_from_bounds.cell_size == 2.0

    def test_memory_estimation_static(self):
        bounds = ModelBounds(x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=100)
        vc = VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)
        mem = VoxelGrid.estimate_memory_mb(bounds, vc, n_attributes=5, sparse_fraction=0.3)
        assert mem > 0  # Should estimate some memory

    def test_dangerous_detection(self):
        bounds = ModelBounds(x_min=0, x_max=1000, y_min=0, y_max=1000, z_min=0, z_max=1000)
        vc = VoxelConfig(cell_size_x=0.1, cell_size_y=0.1, cell_size_z=0.1)
        is_dangerous, reason = VoxelGrid.is_dangerous(bounds, vc)
        assert is_dangerous  # 10000³ = 1 billion voxels

    def test_safe_resolution(self):
        bounds = ModelBounds(x_min=0, x_max=50, y_min=0, y_max=50, z_min=0, z_max=50)
        vc = VoxelConfig(cell_size_x=5.0, cell_size_y=5.0, cell_size_z=5.0)
        is_dangerous, reason = VoxelGrid.is_dangerous(bounds, vc)
        assert not is_dangerous  # 10³ = 1000 voxels
