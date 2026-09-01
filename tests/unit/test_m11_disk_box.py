"""Scientific unit tests for the analytic M11 disk/AABB kernel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dfn_cave_studio.geometry.disk_box import (
    disk_aabb_intersection_area_exact,
    disk_world_aabb,
)


@pytest.mark.parametrize(
    ("box_min", "box_max", "expected"),
    [
        ((-2.0, -2.0, -1.0), (2.0, 2.0, 1.0), math.pi),
        ((0.0, -2.0, -1.0), (2.0, 2.0, 1.0), math.pi / 2.0),
        ((0.0, 0.0, -1.0), (2.0, 2.0, 1.0), math.pi / 4.0),
    ],
)
def test_known_horizontal_circle_areas(box_min, box_max, expected) -> None:
    area = disk_aabb_intersection_area_exact(
        np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0, np.asarray(box_min), np.asarray(box_max)
    )
    assert area == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "normal",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 3.0]),
    ],
)
def test_full_disk_is_orientation_independent(normal: np.ndarray) -> None:
    area = disk_aabb_intersection_area_exact(
        np.zeros(3), normal, 0.75, np.full(3, -2.0), np.full(3, 2.0)
    )
    assert area == pytest.approx(math.pi * 0.75**2, rel=2e-12)


def test_tangent_and_outside_disks_have_zero_area() -> None:
    bounds_min = np.zeros(3)
    bounds_max = np.ones(3)
    assert disk_aabb_intersection_area_exact(
        np.array([-0.25, 0.5, 0.5]), np.array([0.0, 0.0, 1.0]), 0.25, bounds_min, bounds_max
    ) == pytest.approx(0.0)
    assert disk_aabb_intersection_area_exact(
        np.array([2.0, 2.0, 2.0]), np.array([0.0, 0.0, 1.0]), 0.25, bounds_min, bounds_max
    ) == pytest.approx(0.0)


def test_tight_world_aabb_uses_disk_plane_extent() -> None:
    lower, upper = disk_world_aabb(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]), 2.0)
    np.testing.assert_allclose(lower, [1.0, 0.0, 1.0])
    np.testing.assert_allclose(upper, [1.0, 4.0, 5.0])


@pytest.mark.parametrize(
    ("normal", "radius"),
    [(np.zeros(3), 1.0), (np.ones(3), 0.0), (np.ones(3), -1.0), (np.ones(3), float("nan"))],
)
def test_invalid_disks_are_rejected(normal: np.ndarray, radius: float) -> None:
    with pytest.raises(ValueError):
        disk_aabb_intersection_area_exact(
            np.zeros(3), normal, radius, np.zeros(3), np.ones(3)
        )
