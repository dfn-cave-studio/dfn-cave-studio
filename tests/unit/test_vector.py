"""Tests for vector operations."""

import math
import numpy as np
import pytest

from dfn_cave_studio.geometry.vector import (
    normalize,
    length,
    dot,
    cross,
    angle_between,
    distance,
    midpoint,
    project_point_to_line,
    project_point_to_plane,
    are_parallel,
    are_orthogonal,
    plane_from_three_points,
    polygon_area_3d,
)


class TestNormalize:
    def test_unit_x(self):
        result = normalize([3, 0, 0])
        np.testing.assert_array_almost_equal(result, [1, 0, 0])

    def test_already_unit(self):
        result = normalize([0, 1, 0])
        np.testing.assert_array_almost_equal(result, [0, 1, 0])

    def test_arbitrary_vector(self):
        v = np.array([1, 2, 3])
        result = normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-12

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError):
            normalize([0, 0, 0])


class TestLength:
    def test_simple(self):
        assert abs(length([3, 4, 0]) - 5.0) < 1e-12

    def test_zero(self):
        assert length([0, 0, 0]) == 0.0


class TestDot:
    def test_orthogonal(self):
        assert abs(dot([1, 0, 0], [0, 1, 0])) < 1e-12

    def test_parallel(self):
        assert abs(dot([1, 0, 0], [2, 0, 0]) - 2.0) < 1e-12


class TestCross:
    def test_right_hand_rule(self):
        result = cross([1, 0, 0], [0, 1, 0])
        np.testing.assert_array_almost_equal(result, [0, 0, 1])

    def test_anticommutative(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        np.testing.assert_array_almost_equal(cross(a, b), -cross(b, a))


class TestAngleBetween:
    def test_orthogonal(self):
        assert abs(angle_between([1, 0, 0], [0, 0, 1]) - math.pi / 2) < 1e-12

    def test_acute(self):
        # Two vectors at 45 degrees
        a = [1, 0, 0]
        b = [1, 1, 0]
        assert abs(angle_between(a, b) - math.pi / 4) < 1e-12

    def test_parallel(self):
        assert angle_between([1, 0, 0], [2, 0, 0]) < 1e-12


class TestDistance:
    def test_simple(self):
        assert abs(distance([0, 0, 0], [3, 4, 0]) - 5.0) < 1e-12

    def test_same_point(self):
        assert distance([1, 1, 1], [1, 1, 1]) == 0.0


class TestMidpoint:
    def test_simple(self):
        result = midpoint([0, 0, 0], [2, 2, 2])
        np.testing.assert_array_almost_equal(result, [1, 1, 1])


class TestProjectPointToLine:
    def test_projection(self):
        point = [1, 1, 0]
        line_pt = [0, 0, 0]
        line_dir = [1, 0, 0]
        projected, t = project_point_to_line(point, line_pt, line_dir)
        np.testing.assert_array_almost_equal(projected, [1, 0, 0])
        assert abs(t - 1.0) < 1e-12


class TestProjectPointToPlane:
    def test_to_xy_plane(self):
        point = [0, 0, 5]
        plane_pt = [0, 0, 0]
        normal = [0, 0, 1]
        result = project_point_to_plane(point, plane_pt, normal)
        np.testing.assert_array_almost_equal(result, [0, 0, 0])


class TestAreParallel:
    def test_parallel(self):
        assert are_parallel([1, 2, 3], [2, 4, 6])

    def test_not_parallel(self):
        assert not are_parallel([1, 0, 0], [0, 1, 0])


class TestAreOrthogonal:
    def test_orthogonal(self):
        assert are_orthogonal([1, 0, 0], [0, 1, 0])

    def test_not_orthogonal(self):
        assert not are_orthogonal([1, 0, 0], [1, 1, 0])


class TestPlaneFromThreePoints:
    def test_xy_plane(self):
        p1 = [0, 0, 0]
        p2 = [1, 0, 0]
        p3 = [0, 1, 0]
        point, normal = plane_from_three_points(p1, p2, p3)
        # Normal should be +Z
        np.testing.assert_array_almost_equal(normal, [0, 0, 1])

    def test_collinear_raises(self):
        with pytest.raises(ValueError):
            plane_from_three_points([0, 0, 0], [1, 0, 0], [2, 0, 0])


class TestPolygonArea3D:
    def test_unit_square(self):
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ])
        area = polygon_area_3d(vertices)
        assert abs(area - 1.0) < 1e-12

    def test_triangle(self):
        vertices = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
        ])
        area = polygon_area_3d(vertices)
        assert abs(area - 2.0) < 1e-12

    def test_fewer_than_3_vertices(self):
        assert polygon_area_3d(np.array([[0, 0, 0], [1, 0, 0]])) == 0.0
