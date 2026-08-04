"""Tests for geometric intersection algorithms.

Scientific validation cases SV-04, SV-07, SV-08.
"""

import numpy as np
import pytest

from dfn_cave_studio.geometry.intersection import (
    line_plane_intersection,
    point_in_aabb,
    aabb_aabb_overlap,
    plane_aabb_intersects,
    disk_aabb_intersects,
    fracture_fracture_intersects,
)


# =============================================================================
# Line-Plane Intersection
# =============================================================================

class TestLinePlaneIntersection:
    def test_simple_intersection(self):
        """Line through XY plane at z=0."""
        result = line_plane_intersection(
            line_point=np.array([0, 0, 1]),
            line_dir=np.array([0, 0, -1]),
            plane_point=np.array([0, 0, 0]),
            plane_normal=np.array([0, 0, 1]),
        )
        assert result is not None
        point, t = result
        np.testing.assert_array_almost_equal(point, [0, 0, 0])
        assert abs(t - 1.0) < 1e-12

    def test_parallel_no_intersection(self):
        """Line parallel to plane → None."""
        result = line_plane_intersection(
            line_point=np.array([0, 0, 1]),
            line_dir=np.array([1, 0, 0]),
            plane_point=np.array([0, 0, 0]),
            plane_normal=np.array([0, 0, 1]),
        )
        assert result is None

    def test_line_in_plane(self):
        """Line in the plane → None (parallel)."""
        result = line_plane_intersection(
            line_point=np.array([1, 0, 0]),
            line_dir=np.array([1, 0, 0]),
            plane_point=np.array([0, 0, 0]),
            plane_normal=np.array([0, 0, 1]),
        )
        assert result is None


# =============================================================================
# Point in AABB
# =============================================================================

class TestPointInAABB:
    def test_inside(self):
        assert point_in_aabb(np.array([5, 5, 5]), np.array([0, 0, 0]), np.array([10, 10, 10]))

    def test_outside(self):
        assert not point_in_aabb(np.array([15, 5, 5]), np.array([0, 0, 0]), np.array([10, 10, 10]))

    def test_on_boundary(self):
        assert point_in_aabb(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([10, 10, 10]))


# =============================================================================
# AABB-AABB Overlap
# =============================================================================

class TestAABBAABBOverlap:
    def test_overlapping(self):
        assert aabb_aabb_overlap(
            np.array([0, 0, 0]), np.array([5, 5, 5]),
            np.array([3, 3, 3]), np.array([8, 8, 8]),
        )

    def test_disjoint(self):
        assert not aabb_aabb_overlap(
            np.array([0, 0, 0]), np.array([1, 1, 1]),
            np.array([2, 2, 2]), np.array([3, 3, 3]),
        )

    def test_touching(self):
        assert aabb_aabb_overlap(
            np.array([0, 0, 0]), np.array([1, 1, 1]),
            np.array([1, 0, 0]), np.array([2, 1, 1]),
        )


# =============================================================================
# Plane-AABB Intersection
# =============================================================================

class TestPlaneAABBIntersects:
    def test_plane_through_center(self):
        """Plane through center of box."""
        assert plane_aabb_intersects(
            np.array([5, 5, 5]), np.array([0, 0, 1]),
            np.array([0, 0, 0]), np.array([10, 10, 10]),
        )

    def test_plane_outside(self):
        """Plane completely outside box."""
        assert not plane_aabb_intersects(
            np.array([5, 5, 15]), np.array([0, 0, 1]),
            np.array([0, 0, 0]), np.array([10, 10, 10]),
        )


# =============================================================================
# Disk-AABB Intersection (Fracture-Voxel)
# =============================================================================

class TestDiskAABBIntersects:
    """SV-04: Single fracture intersecting a cube."""

    def test_disk_through_cube_center(self):
        """Horizontal disk through cube center → intersects."""
        assert disk_aabb_intersects(
            np.array([5, 5, 5]),  # center
            np.array([0, 0, 1]),  # normal (horizontal)
            4.0,  # radius
            np.array([0, 0, 0]),
            np.array([10, 10, 10]),
        )

    def test_disk_outside(self):
        """Disk completely outside cube."""
        assert not disk_aabb_intersects(
            np.array([50, 50, 50]),
            np.array([0, 0, 1]),
            1.0,
            np.array([0, 0, 0]),
            np.array([10, 10, 10]),
        )

    def test_disk_touching(self):
        """Disk just touching the cube surface."""
        # Disk center at (10, 5, 5), radius 2 → touches box edge at x=10
        assert disk_aabb_intersects(
            np.array([10, 5, 5]),
            np.array([0, 0, 1]),
            0.5,
            np.array([0, 0, 0]),
            np.array([10, 10, 10]),
        )

    def test_disk_barely_inside(self):
        """Small disk at corner of box."""
        assert disk_aabb_intersects(
            np.array([0.1, 0.1, 0.1]),
            np.array([0, 0, 1]),
            0.2,
            np.array([0, 0, 0]),
            np.array([10, 10, 10]),
        )

    def test_dipping_fracture_multiple_voxels(self):
        """SV-05: Dipping fracture through multiple voxels."""
        # Model: 20x20x20, voxels: 2x2x2 (10 voxels in each direction)
        # Fracture: dip_direction=0 (north), dip=45°, center=(10,10,10), radius=10
        from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal
        normal = dip_dir_dip_to_normal(0, 45)

        # Check intersection with 4 voxels in different quadrants
        voxel_size = 2.0
        voxels_tested = 0
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    vmin = np.array([i * voxel_size, j * voxel_size, k * voxel_size])
                    vmax = vmin + voxel_size
                    intersects = disk_aabb_intersects(
                        np.array([10, 10, 10]), normal, 10.0,
                        vmin, vmax,
                    )
                    if intersects:
                        voxels_tested += 1

        # A 45° dipping disk of radius 10 through a 20x20x20 box
        # should intersect multiple voxels
        assert voxels_tested > 0, "Dipping fracture should intersect some voxels"
        assert voxels_tested < 1000, "Should not intersect ALL voxels"


# =============================================================================
# Fracture-Fracture Intersection
# =============================================================================

class TestFractureFractureIntersects:
    """SV-07, SV-08: Fracture-fracture intersection tests."""

    def test_two_intersecting(self):
        """Two perpendicular fractures that intersect."""
        assert fracture_fracture_intersects(
            np.array([5, 5, 5]), np.array([1, 0, 0]), 3.0,  # X-normal disk
            np.array([5, 5, 5]), np.array([0, 1, 0]), 3.0,  # Y-normal disk
        )

    def test_two_separated(self):
        """Two parallel fractures that don't intersect."""
        assert not fracture_fracture_intersects(
            np.array([0, 0, 0]), np.array([0, 0, 1]), 1.0,
            np.array([10, 0, 0]), np.array([0, 0, 1]), 1.0,
        )

    def test_coplanar_overlapping(self):
        """Two coplanar fractures that overlap."""
        assert fracture_fracture_intersects(
            np.array([0, 0, 0]), np.array([0, 0, 1]), 3.0,
            np.array([2, 0, 0]), np.array([0, 0, 1]), 3.0,
        )

    def test_coplanar_separated(self):
        """Two coplanar fractures that don't overlap."""
        assert not fracture_fracture_intersects(
            np.array([0, 0, 0]), np.array([0, 0, 1]), 1.0,
            np.array([10, 0, 0]), np.array([0, 0, 1]), 1.0,
        )

    def test_intersecting_at_45_degrees(self):
        """Two fractures intersecting at 45°."""
        assert fracture_fracture_intersects(
            np.array([5, 5, 5]), np.array([1, 0, 0]), 3.0,  # X-normal
            np.array([5, 5, 5]), np.array([1, 1, 0]), 3.0,  # 45° rotated
        )

    def test_edges_nearly_miss(self):
        """Two fractures whose edges nearly miss each other."""
        # Center-to-center distance > r1 + r2
        assert not fracture_fracture_intersects(
            np.array([0, 0, 0]), np.array([0, 0, 1]), 1.0,
            np.array([3, 0, 0]), np.array([0, 0, 1]), 1.0,
        )
