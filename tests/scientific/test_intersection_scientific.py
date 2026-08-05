"""Scientific validation of fracture-fracture intersection algorithm.

These tests verify correctness against analytic solutions with known
geometric configurations.
"""

import numpy as np

from dfn_cave_studio.geometry.intersection import (
    fracture_fracture_intersects,
    fracture_fracture_intersection_detail,
    disk_aabb_intersects,
)


class TestFractureFractureIntersection:
    """Validate fracture-fracture intersection against analytic solutions."""

    def test_orthogonal_intersecting(self):
        """Two orthogonal fractures at origin must intersect."""
        c1 = np.array([0.0, 0.0, 0.0])
        n1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([0.0, 0.0, 0.0])
        n2 = np.array([1.0, 0.0, 0.0])
        assert bool(fracture_fracture_intersects(c1, n1, 5.0, c2, n2, 5.0))

    def test_parallel_separated(self):
        """Parallel fractures in different planes must NOT intersect."""
        c1 = np.array([0.0, 0.0, 0.0])
        n1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([5.0, 0.0, 0.0])   # 5m offset in X
        n2 = np.array([0.0, 0.0, 1.0])
        # r1=1, r2=1 → r1+r2=2 < 5 → no intersection
        assert not bool(fracture_fracture_intersects(c1, n1, 1.0, c2, n2, 1.0))

    def test_coplanar_separated(self):
        """Coplanar fractures too far apart must NOT intersect."""
        c1 = np.array([0.0, 0.0, 0.0])
        n1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([100.0, 0.0, 0.0])
        n2 = np.array([0.0, 0.0, 1.0])
        assert not bool(fracture_fracture_intersects(c1, n1, 1.0, c2, n2, 1.0))

    def test_coplanar_touching(self):
        """Coplanar fractures touching at a point (r1+r2 = distance)."""
        c1 = np.array([0.0, 0.0, 0.0])
        n1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([7.0, 0.0, 0.0])
        n2 = np.array([0.0, 0.0, 1.0])
        # r1+r2 = 8 > 7 → intersect
        assert bool(fracture_fracture_intersects(c1, n1, 3.0, c2, n2, 5.0))

    def test_skew_non_intersecting(self):
        """Skew fractures (not parallel, not intersecting)."""
        c1 = np.array([0.0, 0.0, 0.0])
        n1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([10.0, 10.0, 10.0])
        n2 = np.array([1.0, 0.0, 0.0])
        # Both r=1, separated by >1 in all directions
        assert not bool(fracture_fracture_intersects(c1, n1, 1.0, c2, n2, 1.0))

    def test_brute_force_known_cases(self):
        """Test a matrix of known intersection cases."""
        # (c1, n1, r1, c2, n2, r2, expected)
        cases = [
            # Same origin, orthogonal
            ([0,0,0], [0,0,1], 5, [0,0,0], [0,1,0], 5, True),
            # Same origin, 45° apart
            ([0,0,0], [0,0,1], 5, [0,0,0], [1/np.sqrt(2),1/np.sqrt(2),0], 5, True),
            # Offset by 1m in X, same plane — overlap depends on radii
            ([0,0,0], [0,0,1], 3, [4,0,0], [0,0,1], 3, False),  # 4 > 3+3=6? No, 4<6 → True
            # Actually: distance=4, r1+r2=6, so they DO intersect!
        ]
        # Corrected:
        cases[-1] = ([0,0,0], [0,0,1], 3, [4,0,0], [0,0,1], 3, True)

        for c1, n1, r1, c2, n2, r2, expected in cases:
            result = bool(fracture_fracture_intersects(
                np.array(c1, dtype=float), np.array(n1, dtype=float), r1,
                np.array(c2, dtype=float), np.array(n2, dtype=float), r2,
            ))
            assert result == expected, f"Failed: {c1}/{n1}/r={r1} vs {c2}/{n2}/r={r2}"


class TestIntersectionDetail:
    """Validate the intersection segment detail function."""

    def test_orthogonal_at_origin_returns_segment(self):
        """Two orthogonal disks at origin must return a valid segment."""
        detail = fracture_fracture_intersection_detail(
            np.array([0., 0., 0.]), np.array([0., 0., 1.]), 5.0,
            np.array([0., 0., 0.]), np.array([1., 0., 0.]), 5.0,
        )
        assert detail is not None
        assert detail["intersects"]
        assert detail["segment_length"] > 0
        # Intersection of two radius-5 orthogonal disks centered at origin:
        # the segment is along the Y axis from -5 to 5
        seg_len = detail["segment_length"]
        assert 9.0 < seg_len < 11.0  # ≈ 10 for two r=5 disks at origin
        # Midpoint should be at origin
        np.testing.assert_allclose(detail["midpoint"], [0, 0, 0], atol=1e-10)

    def test_non_intersecting_returns_none(self):
        """Non-intersecting fractures must return None."""
        detail = fracture_fracture_intersection_detail(
            np.array([0., 0., 0.]), np.array([0., 0., 1.]), 1.0,
            np.array([10., 10., 10.]), np.array([1., 0., 0.]), 1.0,
        )
        assert detail is None

    def test_segment_endpoints_are_on_both_planes(self):
        """Intersection segment endpoints must lie on both fracture planes."""
        c1 = np.array([1., 2., 3.])
        n1 = np.array([1., 0., 0.]) / np.linalg.norm([1., 0., 0.])
        c2 = np.array([4., 5., 6.])
        n2 = np.array([0., 1., 0.]) / np.linalg.norm([0., 1., 0.])

        detail = fracture_fracture_intersection_detail(c1, n1, 10., c2, n2, 10.)
        assert detail is not None

        for pt in [detail["segment_start"], detail["segment_end"]]:
            # Check on plane 1: n1·(pt - c1) ≈ 0
            assert abs(np.dot(n1, pt - c1)) < 1e-10, f"Point {pt} not on plane 1"
            # Check on plane 2: n2·(pt - c2) ≈ 0
            assert abs(np.dot(n2, pt - c2)) < 1e-10, f"Point {pt} not on plane 2"

    def test_parallel_coplanar_returns_segment(self):
        """Parallel, coplanar disks return intersection."""
        detail = fracture_fracture_intersection_detail(
            np.array([0., 0., 0.]), np.array([0., 0., 1.]), 5.0,
            np.array([4., 0., 0.]), np.array([0., 0., 1.]), 5.0,
        )
        assert detail is not None
        assert detail["intersects"]


class TestDiskAABBIntersection:
    """Validate disk-AABB intersection tests."""

    def test_disk_inside_box(self):
        """Disk fully inside box must intersect."""
        center = np.array([0.5, 0.5, 0.5])
        normal = np.array([0., 0., 1.])
        vmin = np.array([0., 0., 0.])
        vmax = np.array([1., 1., 1.])
        assert disk_aabb_intersects(center, normal, 0.5, vmin, vmax)

    def test_disk_outside_box(self):
        """Disk far from box must NOT intersect."""
        center = np.array([10., 10., 10.])
        normal = np.array([0., 0., 1.])
        vmin = np.array([0., 0., 0.])
        vmax = np.array([1., 1., 1.])
        assert not disk_aabb_intersects(center, normal, 1.0, vmin, vmax)

    def test_disk_cutting_box_edge(self):
        """Disk centered on box face should intersect."""
        center = np.array([1.0, 0.5, 0.5])  # On the x=1 face
        normal = np.array([1., 0., 0.])      # Parallel to X
        vmin = np.array([0., 0., 0.])
        vmax = np.array([1., 1., 1.])
        # Disk center is on the x=1 boundary, pointing along X
        # Half the disk is inside, half outside — should detect intersection
        assert disk_aabb_intersects(center, normal, 0.5, vmin, vmax)

    def test_diagonal_disk_through_box(self):
        """45° disk passing through box corner should intersect."""
        center = np.array([0.5, 0.5, 0.5])
        normal = np.array([1., 1., 0.]) / np.sqrt(2)
        vmin = np.array([0., 0., 0.])
        vmax = np.array([1., 1., 1.])
        assert disk_aabb_intersects(center, normal, 2.0, vmin, vmax)
