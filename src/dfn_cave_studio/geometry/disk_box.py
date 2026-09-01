"""Analytic circular-disk/AABB intersection area for M11 second voxelization.

The AABB is projected into the disk's local orthonormal basis.  Its six
half-space constraints clip a square containing the circle.  The final
circle/convex-polygon area is integrated with exact line and circular-arc
terms; no fixed-sided disk polygon is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DiskBoxIntersection:
    """Result of one disk/AABB area query."""

    area: float
    local_polygon: FloatArray
    tolerance: float

    @property
    def has_positive_area(self) -> bool:
        """Whether the intersection contains positive two-dimensional area."""
        return self.area > self.tolerance * self.tolerance


def geometry_tolerance(*values: FloatArray | float) -> float:
    """Return a scale-relative floating-point length tolerance."""
    scale = 1.0
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if array.size:
            scale = max(scale, float(np.max(np.abs(array))))
    return 128.0 * np.finfo(np.float64).eps * scale


def validate_disk_box(
    center: FloatArray,
    normal: FloatArray,
    radius: float,
    box_min: FloatArray,
    box_max: FloatArray,
) -> tuple[FloatArray, FloatArray, float, FloatArray, FloatArray, float]:
    """Validate and normalize one disk/AABB query."""
    center = np.asarray(center, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    box_min = np.asarray(box_min, dtype=np.float64)
    box_max = np.asarray(box_max, dtype=np.float64)
    if center.shape != (3,) or normal.shape != (3,) or box_min.shape != (3,) or box_max.shape != (3,):
        raise ValueError("center, normal, box_min and box_max must each have shape (3,)")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(normal)):
        raise ValueError("disk center and normal must be finite")
    if not np.all(np.isfinite(box_min)) or not np.all(np.isfinite(box_max)):
        raise ValueError("box bounds must be finite")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("disk radius must be finite and greater than zero")
    if np.any(box_max <= box_min):
        raise ValueError("box_max must be greater than box_min on every axis")
    normal_length = float(np.linalg.norm(normal))
    if not math.isfinite(normal_length) or normal_length <= 0.0:
        raise ValueError("disk normal must have positive finite length")
    normal = normal / normal_length
    tolerance = geometry_tolerance(center, radius, box_min, box_max)
    return center, normal, float(radius), box_min, box_max, tolerance


def disk_world_aabb(center: FloatArray, normal: FloatArray, radius: float) -> tuple[FloatArray, FloatArray]:
    """Return the tight world-coordinate AABB of a circular disk."""
    center = np.asarray(center, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if center.shape != (3,) or normal.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center and normal must be finite vectors with shape (3,)")
    length = float(np.linalg.norm(normal))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("normal must have positive finite length")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and greater than zero")
    unit = normal / length
    extent = float(radius) * np.sqrt(np.maximum(0.0, 1.0 - unit * unit))
    return center - extent, center + extent


def plane_basis(normal: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return a deterministic right-handed orthonormal basis for a plane."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    axis = int(np.argmin(np.abs(normal)))
    reference = np.zeros(3, dtype=np.float64)
    reference[axis] = 1.0
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _clip_half_plane(polygon: FloatArray, normal: FloatArray, bound: float, tolerance: float) -> FloatArray:
    if len(polygon) == 0:
        return polygon
    norm = float(np.linalg.norm(normal))
    if norm <= tolerance:
        return polygon if bound >= -tolerance else np.empty((0, 2), dtype=np.float64)
    output: list[FloatArray] = []
    previous = polygon[-1]
    previous_distance = float(np.dot(normal, previous) - bound)
    previous_inside = previous_distance <= tolerance
    for current in polygon:
        current_distance = float(np.dot(normal, current) - bound)
        current_inside = current_distance <= tolerance
        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            if abs(denominator) > tolerance:
                fraction = previous_distance / denominator
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_distance = current_distance
        previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    result = np.asarray(output, dtype=np.float64)
    if len(result) > 1:
        keep = np.ones(len(result), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(result, axis=0), axis=1) > tolerance
        result = result[keep]
        if len(result) > 1 and np.linalg.norm(result[0] - result[-1]) <= tolerance:
            result = result[:-1]
    return result


def disk_plane_box_polygon(
    center: FloatArray,
    normal: FloatArray,
    radius: float,
    box_min: FloatArray,
    box_max: FloatArray,
) -> tuple[FloatArray, float]:
    """Return the local convex polygon constraining disk points inside an AABB."""
    center, normal, radius, box_min, box_max, tolerance = validate_disk_box(
        center, normal, radius, box_min, box_max
    )
    u, v = plane_basis(normal)
    polygon = np.asarray(
        [[-radius, -radius], [radius, -radius], [radius, radius], [-radius, radius]], dtype=np.float64
    )
    for axis in range(3):
        direction = np.asarray([u[axis], v[axis]], dtype=np.float64)
        polygon = _clip_half_plane(polygon, -direction, center[axis] - box_min[axis], tolerance)
        polygon = _clip_half_plane(polygon, direction, box_max[axis] - center[axis], tolerance)
        if len(polygon) < 3:
            return np.empty((0, 2), dtype=np.float64), tolerance
    return polygon, tolerance


def _cross_2d(a: FloatArray, b: FloatArray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _circle_edge_integral(a: FloatArray, b: FloatArray, radius: float, tolerance: float) -> float:
    delta = b - a
    aa = float(np.dot(delta, delta))
    split = [0.0, 1.0]
    if aa > tolerance * tolerance:
        bb = 2.0 * float(np.dot(a, delta))
        cc = float(np.dot(a, a) - radius * radius)
        discriminant = bb * bb - 4.0 * aa * cc
        if discriminant > tolerance * tolerance:
            root = math.sqrt(discriminant)
            for value in ((-bb - root) / (2.0 * aa), (-bb + root) / (2.0 * aa)):
                if tolerance < value < 1.0 - tolerance:
                    split.append(value)
    split = sorted(set(split))
    contribution = 0.0
    for start, stop in zip(split[:-1], split[1:]):
        p = a + start * delta
        q = a + stop * delta
        midpoint = 0.5 * (p + q)
        cross_value = _cross_2d(p, q)
        if float(np.dot(midpoint, midpoint)) < radius * radius - tolerance * radius:
            contribution += 0.5 * cross_value
        else:
            contribution += 0.5 * radius * radius * math.atan2(cross_value, float(np.dot(p, q)))
    return contribution


def circle_convex_polygon_intersection_area(
    polygon: FloatArray, radius: float, tolerance: float | None = None
) -> float:
    """Return the analytic area shared by an origin-centred circle and convex polygon."""
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,):
        raise ValueError("polygon must have shape (N, 2)")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and greater than zero")
    if len(polygon) < 3:
        return 0.0
    tolerance = geometry_tolerance(polygon, radius) if tolerance is None else float(tolerance)
    area = sum(
        _circle_edge_integral(polygon[index], polygon[(index + 1) % len(polygon)], radius, tolerance)
        for index in range(len(polygon))
    )
    area = min(math.pi * radius * radius, max(0.0, abs(float(area))))
    return 0.0 if area <= tolerance * tolerance else area


def disk_aabb_intersection(
    center: FloatArray,
    normal: FloatArray,
    radius: float,
    box_min: FloatArray,
    box_max: FloatArray,
) -> DiskBoxIntersection:
    """Compute analytic positive-area intersection of a circular disk and AABB."""
    polygon, tolerance = disk_plane_box_polygon(center, normal, radius, box_min, box_max)
    area = circle_convex_polygon_intersection_area(polygon, radius, tolerance) if len(polygon) >= 3 else 0.0
    return DiskBoxIntersection(area=area, local_polygon=polygon, tolerance=tolerance)


def disk_aabb_intersection_area_exact(
    center: FloatArray,
    normal: FloatArray,
    radius: float,
    box_min: FloatArray,
    box_max: FloatArray,
) -> float:
    """Return analytic circular-disk/AABB intersection area in square metres."""
    return disk_aabb_intersection(center, normal, radius, box_min, box_max).area
