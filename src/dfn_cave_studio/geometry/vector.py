"""
Vector operations for 3D geometry in DFN Cave Studio.

All vectors are right-handed Cartesian: X=Easting, Y=Northing, Z=Elevation.
All lengths in meters, angles in radians (internal).
"""

from __future__ import annotations

import math
from typing import Union, Tuple, Optional

import numpy as np
from numpy.typing import NDArray


# Type aliases
Vec3 = Union[NDArray[np.float64], Tuple[float, float, float]]
Vec3f = NDArray[np.float64]


def normalize(v: Vec3) -> Vec3f:
    """Normalize a 3D vector to unit length.

    Args:
        v: 3D vector (array-like of length 3).

    Returns:
        Unit vector of shape (3,).

    Raises:
        ValueError: If input has near-zero length.

    Example:
        >>> normalize([3, 0, 0])
        array([1., 0., 0.])
    """
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        raise ValueError("Cannot normalize vector with near-zero length")
    return v / norm


def length(v: Vec3) -> float:
    """Compute the Euclidean length of a 3D vector.

    Args:
        v: 3D vector.

    Returns:
        Vector length in meters.
    """
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def dot(a: Vec3, b: Vec3) -> float:
    """Compute dot product of two 3D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Dot product.
    """
    return float(np.dot(np.asarray(a), np.asarray(b)))


def cross(a: Vec3, b: Vec3) -> Vec3f:
    """Compute cross product of two 3D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cross product vector (right-hand rule).
    """
    return np.cross(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def angle_between(a: Vec3, b: Vec3) -> float:
    """Compute the acute angle between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Angle in radians [0, π/2].
    """
    a = normalize(np.asarray(a, dtype=np.float64))
    b = normalize(np.asarray(b, dtype=np.float64))
    cos_theta = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = math.acos(abs(cos_theta))
    # Return acute angle
    return min(theta, math.pi - theta)


def distance(p1: Vec3, p2: Vec3) -> float:
    """Compute Euclidean distance between two 3D points.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        Distance in meters.
    """
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def midpoint(p1: Vec3, p2: Vec3) -> Vec3f:
    """Compute the midpoint between two 3D points.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        Midpoint coordinates.
    """
    return (np.asarray(p1, dtype=np.float64) + np.asarray(p2, dtype=np.float64)) / 2.0


def project_point_to_line(
    point: Vec3, line_point: Vec3, line_dir: Vec3
) -> Tuple[Vec3f, float]:
    """Project a point onto a line.

    Args:
        point: Point to project.
        line_point: A point on the line.
        line_dir: Direction vector of the line (need not be unit).

    Returns:
        Tuple of (projected_point, parameter_t) where:
          projected_point = line_point + t * line_dir
    """
    point = np.asarray(point, dtype=np.float64)
    line_point = np.asarray(line_point, dtype=np.float64)
    line_dir = np.asarray(line_dir, dtype=np.float64)
    dir_unit = normalize(line_dir)
    v = point - line_point
    t = np.dot(v, dir_unit)
    return line_point + t * dir_unit, t


def project_point_to_plane(
    point: Vec3, plane_point: Vec3, plane_normal: Vec3
) -> Vec3f:
    """Project a point onto a plane.

    Args:
        point: Point to project.
        plane_point: A point on the plane.
        plane_normal: Normal vector of the plane (need not be unit).

    Returns:
        Projected point on the plane.
    """
    point = np.asarray(point, dtype=np.float64)
    plane_point = np.asarray(plane_point, dtype=np.float64)
    normal = normalize(np.asarray(plane_normal, dtype=np.float64))
    v = point - plane_point
    dist = np.dot(v, normal)
    return point - dist * normal


def are_parallel(v1: Vec3, v2: Vec3, tol: float = 1e-10) -> bool:
    """Check if two vectors are parallel.

    Args:
        v1: First vector.
        v2: Second vector.
        tol: Tolerance for cross product magnitude.

    Returns:
        True if vectors are parallel within tolerance.
    """
    c = cross(normalize(v1), normalize(v2))
    return float(np.linalg.norm(c)) < tol


def are_orthogonal(v1: Vec3, v2: Vec3, tol: float = 1e-10) -> bool:
    """Check if two vectors are orthogonal.

    Args:
        v1: First vector.
        v2: Second vector.
        tol: Tolerance for dot product.

    Returns:
        True if vectors are orthogonal within tolerance.
    """
    return abs(dot(normalize(v1), normalize(v2))) < tol


def plane_from_three_points(
    p1: Vec3, p2: Vec3, p3: Vec3
) -> Tuple[Vec3f, Vec3f]:
    """Compute a plane (point, normal) from three non-collinear points.

    Args:
        p1, p2, p3: Three points defining the plane.

    Returns:
        Tuple of (point_on_plane, unit_normal_vector).
        Normal follows right-hand rule from p1→p2→p3.

    Raises:
        ValueError: If points are collinear.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    v1 = p2 - p1
    v2 = p3 - p1
    normal = cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-15:
        raise ValueError("Points are collinear; cannot define a plane")
    return p1, normal / norm


def polygon_area_3d(vertices: NDArray[np.float64]) -> float:
    """Compute the area of a 3D polygon using Newell's method.

    Args:
        vertices: Nx3 array of polygon vertices (ordered, may be non-planar).

    Returns:
        Area in square meters.
    """
    if vertices.shape[0] < 3:
        return 0.0

    n = vertices.shape[0]
    normal = np.zeros(3)
    for i in range(n):
        j = (i + 1) % n
        normal[0] += (vertices[i, 1] - vertices[j, 1]) * (vertices[i, 2] + vertices[j, 2])
        normal[1] += (vertices[i, 2] - vertices[j, 2]) * (vertices[i, 0] + vertices[j, 0])
        normal[2] += (vertices[i, 0] - vertices[j, 0]) * (vertices[i, 1] + vertices[j, 1])

    return 0.5 * float(np.linalg.norm(normal))
