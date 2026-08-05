"""
Geometric intersection algorithms for DFN Cave Studio.

Implements:
  - Line-plane intersection
  - Plane-AABB (box) intersection
  - Fracture-voxel intersection (candidate + exact)
  - Fracture-fracture intersection

References:
  - Eberly, D. (2008). Dynamic Collision Detection.
  - SCIENTIFIC_SPEC.md Section 7, 8.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.geometry.vector import normalize, dot, cross


# =============================================================================
# Line-Plane Intersection
# =============================================================================

def line_plane_intersection(
    line_point: NDArray[np.float64],
    line_dir: NDArray[np.float64],
    plane_point: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
    tol: float = 1e-12,
) -> Optional[Tuple[NDArray[np.float64], float]]:
    """Compute intersection of a line with an infinite plane.

    Line: P = line_point + t * line_dir
    Plane: (P - plane_point) · plane_normal = 0

    Args:
        line_point: Point on the line (3,).
        line_dir: Direction vector of the line (3,), need not be unit.
        plane_point: Point on the plane (3,).
        plane_normal: Normal vector of the plane (3,), need not be unit.
        tol: Tolerance for parallelism check.

    Returns:
        Tuple of (intersection_point, t) or None if line is parallel to plane.
    """
    line_dir = np.asarray(line_dir, dtype=np.float64)
    plane_normal = normalize(np.asarray(plane_normal, dtype=np.float64))
    line_dir_u = normalize(line_dir)

    denom = dot(line_dir_u, plane_normal)

    if abs(denom) < tol:
        # Line is parallel to plane
        return None

    v = np.asarray(plane_point, dtype=np.float64) - np.asarray(line_point, dtype=np.float64)
    t = dot(v, plane_normal) / denom
    intersection = np.asarray(line_point, dtype=np.float64) + t * line_dir_u

    return intersection, t


# =============================================================================
# Box (AABB) Intersection Tests
# =============================================================================

def point_in_aabb(
    point: NDArray[np.float64],
    box_min: NDArray[np.float64],
    box_max: NDArray[np.float64],
    tol: float = 1e-12,
) -> bool:
    """Check if a point is inside an axis-aligned bounding box.

    Args:
        point: 3D point (3,).
        box_min: Minimum corner of AABB (3,).
        box_max: Maximum corner of AABB (3,).
        tol: Numerical tolerance.

    Returns:
        True if point is inside or on boundary of AABB.
    """
    return bool(np.all(point >= box_min - tol) and np.all(point <= box_max + tol))


def aabb_aabb_overlap(
    a_min: NDArray[np.float64], a_max: NDArray[np.float64],
    b_min: NDArray[np.float64], b_max: NDArray[np.float64],
) -> bool:
    """Check if two AABBs overlap.

    Args:
        a_min, a_max: First AABB corners.
        b_min, b_max: Second AABB corners.

    Returns:
        True if boxes overlap.
    """
    return bool(
        np.all(a_min <= b_max) and np.all(b_min <= a_max)
    )


# =============================================================================
# Plane-AABB Intersection
# =============================================================================

def plane_aabb_intersects(
    plane_point: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
    box_min: NDArray[np.float64],
    box_max: NDArray[np.float64],
) -> bool:
    """Test if an infinite plane intersects an AABB.

    Uses the separating axis theorem: the plane intersects the box if
    the signed distances from the box corners to the plane have opposite signs,
    OR one corner lies on the plane.

    Args:
        plane_point: Point on the plane (3,).
        plane_normal: Normal vector of the plane (3,).
        box_min: Minimum corner of AABB (3,).
        box_max: Maximum corner of AABB (3,).

    Returns:
        True if plane intersects AABB.
    """
    normal = normalize(np.asarray(plane_normal, dtype=np.float64))
    plane_point = np.asarray(plane_point, dtype=np.float64)

    # Compute signed distances for all 8 corners
    corners = np.array([
        [box_min[0], box_min[1], box_min[2]],
        [box_min[0], box_min[1], box_max[2]],
        [box_min[0], box_max[1], box_min[2]],
        [box_min[0], box_max[1], box_max[2]],
        [box_max[0], box_min[1], box_min[2]],
        [box_max[0], box_min[1], box_max[2]],
        [box_max[0], box_max[1], box_min[2]],
        [box_max[0], box_max[1], box_max[2]],
    ], dtype=np.float64)

    distances = np.dot(corners - plane_point, normal)
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Intersection if signs differ (plane passes through box)
    return min_dist * max_dist <= 0.0


# =============================================================================
# Disk (Circular Fracture) - AABB Intersection
# =============================================================================

def disk_aabb_intersects(
    disk_center: NDArray[np.float64],
    disk_normal: NDArray[np.float64],
    disk_radius: float,
    box_min: NDArray[np.float64],
    box_max: NDArray[np.float64],
) -> bool:
    """Test if a circular disk intersects an AABB.

    First checks if the disk's plane intersects the AABB (fast reject).
    Then finds the closest point on the AABB to the disk center,
    and checks if it's within the disk radius (projected onto the plane).

    Args:
        disk_center: Center of disk (3,).
        disk_normal: Normal vector of disk plane (3,).
        disk_radius: Radius of disk.
        box_min: Minimum corner of AABB (3,).
        box_max: Maximum corner of AABB (3,).

    Returns:
        True if disk intersects AABB.
    """
    center = np.asarray(disk_center, dtype=np.float64)
    normal = normalize(np.asarray(disk_normal, dtype=np.float64))

    # Fast reject: plane doesn't intersect box
    if not plane_aabb_intersects(center, normal, box_min, box_max):
        # However, the disk might still touch even if the infinite plane doesn't
        # intersect the box (disk is finite, its plane might be outside).
        # Quick distance check:
        box_center = (np.asarray(box_min) + np.asarray(box_max)) / 2.0
        dist_plane_to_box_center = abs(dot(box_center - center, normal))
        if dist_plane_to_box_center > disk_radius:
            return False

    # Find closest point on AABB to disk center
    closest = np.clip(center, box_min, box_max)

    # Distance from closest point to disk plane
    dist_to_plane = dot(closest - center, normal)

    # Distance in plane from center to closest point's projection
    if abs(np.linalg.norm(closest - center)) < 1e-12:
        return True  # Center is inside box

    # Project closest point onto disk plane
    projected = closest - dist_to_plane * normal
    in_plane_dist = np.linalg.norm(projected - center)

    return in_plane_dist <= disk_radius


# =============================================================================
# Fracture-Fracture Intersection
# =============================================================================

def fracture_fracture_intersects(
    center1: NDArray[np.float64],
    normal1: NDArray[np.float64],
    radius1: float,
    center2: NDArray[np.float64],
    normal2: NDArray[np.float64],
    radius2: float,
) -> bool:
    """Test if two circular disk fractures intersect.

    Computes the line of intersection between the two fracture planes,
    then checks if the intersection segment overlaps both disks.

    Args:
        center1, normal1, radius1: Parameters of fracture 1.
        center2, normal2, radius2: Parameters of fracture 2.

    Returns:
        True if the two fractures intersect.
    """
    c1 = np.asarray(center1, dtype=np.float64)
    n1 = normalize(np.asarray(normal1, dtype=np.float64))
    c2 = np.asarray(center2, dtype=np.float64)
    n2 = normalize(np.asarray(normal2, dtype=np.float64))

    # Direction of intersection line = n1 × n2
    line_dir = cross(n1, n2)
    dir_norm = np.linalg.norm(line_dir)

    if dir_norm < 1e-12:
        # Planes are parallel; check if coplanar and overlapping
        dist_between_planes = abs(dot(c2 - c1, n1))
        if dist_between_planes > 1e-10:
            return False  # Parallel but not coplanar
        # Coplanar: check 2D circle overlap
        center_dist = np.linalg.norm(c2 - c1)
        return center_dist <= (radius1 + radius2)

    line_dir = line_dir / dir_norm

    # Find a point on the intersection line.
    # Parametrization: P = c1 + α·n1 + β·n2
    # Constraints: n1·(P - c1) = 0  →  α + β·(n1·n2) = 0
    #              n2·(P - c2) = 0  →  α·(n1·n2) + β + n2·(c1 - c2) = 0
    #                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                                     The constant term comes from
    #                                     n2·(c1 + α·n1 + β·n2 - c2) = 0
    #                                     = n2·(c1-c2) + α·(n1·n2) + β = 0
    # Therefore: dα + β = -n2·(c1-c2)
    d12 = dot(n1, n2)  # n1·n2
    # Build 2×2 system: [1, d12; d12, 1] · [α; β] = [0; -n2·(c1-c2)]
    b_val = -dot(n2, c1 - c2)  # negative sign from constraint derivation

    A = np.array([[1.0, d12], [d12, 1.0]])
    b = np.array([0.0, b_val])

    # Handle degenerate case (parallel normals → planes nearly parallel)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]  # = 1 - d12²
    if abs(det) < 1e-12:
        return False

    alpha = (b[0] * A[1, 1] - A[0, 1] * b[1]) / det
    beta = (A[0, 0] * b[1] - b[0] * A[1, 0]) / det

    line_point = c1 + alpha * n1 + beta * n2

    # Compute shortest distance from each fracture center to the
    # intersection line, using the standard point-to-line formula.
    c1_to_line = c1 - line_point
    c2_to_line = c2 - line_point
    dist1_to_line = float(np.linalg.norm(cross(c1_to_line, line_dir)))
    dist2_to_line = float(np.linalg.norm(cross(c2_to_line, line_dir)))

    if dist1_to_line > radius1 or dist2_to_line > radius2:
        return False

    # Parameterize positions along the intersection line.
    # t = 0 at line_point; t_i = projection of (c_i - line_point) onto line_dir.
    t1 = float(dot(c1_to_line, line_dir))
    t2 = float(dot(c2_to_line, line_dir))

    # Half-length of the intersection chord within each disk
    h1 = np.sqrt(max(0.0, radius1 ** 2 - dist1_to_line ** 2))
    h2 = np.sqrt(max(0.0, radius2 ** 2 - dist2_to_line ** 2))

    # Intersection segments along the line:
    #   disk 1: [t1 - h1, t1 + h1]
    #   disk 2: [t2 - h2, t2 + h2]
    # Overlap exists iff the two intervals intersect.
    seg1_min, seg1_max = t1 - h1, t1 + h1
    seg2_min, seg2_max = t2 - h2, t2 + h2

    return seg1_min <= seg2_max and seg2_min <= seg1_max


def fracture_fracture_intersection_detail(
    center1: NDArray[np.float64],
    normal1: NDArray[np.float64],
    radius1: float,
    center2: NDArray[np.float64],
    normal2: NDArray[np.float64],
    radius2: float,
) -> Optional[dict]:
    """Compute detailed intersection between two circular disk fractures.

    Returns the intersection segment endpoints, length, and geometric
    properties for scientific use (fragmentation analysis, block theory).

    Args:
        center1, normal1, radius1: Parameters of fracture 1.
        center2, normal2, radius2: Parameters of fracture 2.

    Returns:
        None if no intersection, or dict with:
          - intersects: bool (True)
          - segment_start: (3,) array — first endpoint
          - segment_end: (3,) array — second endpoint
          - segment_length: float — length of intersection segment (m)
          - midpoint: (3,) array — midpoint of intersection segment
          - line_direction: (3,) array — unit vector along intersection line
          - line_point: (3,) array — reference point on intersection line
    """
    c1 = np.asarray(center1, dtype=np.float64)
    n1 = normalize(np.asarray(normal1, dtype=np.float64))
    c2 = np.asarray(center2, dtype=np.float64)
    n2 = normalize(np.asarray(normal2, dtype=np.float64))

    # Direction of intersection line
    line_dir = cross(n1, n2)
    dir_norm = np.linalg.norm(line_dir)

    if dir_norm < 1e-12:
        # Parallel planes
        dist_between = abs(dot(c2 - c1, n1))
        if dist_between > 1e-10:
            return None  # Not coplanar
        # Coplanar: 2D circle-circle intersection
        center_dist = float(np.linalg.norm(c2 - c1))
        if center_dist > radius1 + radius2:
            return None
        # For coplanar disks, intersection is a lens, not a segment.
        # Return the chord endpoints.
        if center_dist < 1e-12:
            # Concentric — intersection is the smaller disk
            r_min = min(radius1, radius2)
            # Arbitrary direction
            seg_dir = np.array([1.0, 0.0, 0.0])
            # Reject if parallel to n1
            if abs(dot(seg_dir, n1)) > 0.99:
                seg_dir = np.array([0.0, 1.0, 0.0])
            seg_dir = seg_dir - dot(seg_dir, n1) * n1
            seg_dir = seg_dir / np.linalg.norm(seg_dir)
            return {
                "intersects": True,
                "segment_start": c1 - seg_dir * r_min,
                "segment_end": c1 + seg_dir * r_min,
                "segment_length": 2.0 * r_min,
                "midpoint": c1.copy(),
                "line_direction": seg_dir,
                "line_point": c1.copy(),
            }
        # General coplanar case: chord of circle-circle intersection
        d = center_dist
        a = (radius1**2 - radius2**2 + d**2) / (2.0 * d)
        h_sq = radius1**2 - a**2
        if h_sq <= 0:
            return None  # Tangent or no intersection
        h = np.sqrt(h_sq)
        # Midpoint of intersection chord
        mid = c1 + a * (c2 - c1) / d
        # Chord direction: perpendicular to c2-c1 in the plane
        chord_dir = cross(n1, c2 - c1)
        chord_dir = chord_dir / np.linalg.norm(chord_dir)
        return {
            "intersects": True,
            "segment_start": mid - chord_dir * h,
            "segment_end": mid + chord_dir * h,
            "segment_length": 2.0 * h,
            "midpoint": mid,
            "line_direction": chord_dir,
            "line_point": mid,
        }

    line_dir = line_dir / dir_norm

    # Find reference point on intersection line
    d12 = dot(n1, n2)
    b_val = -dot(n2, c1 - c2)
    det = 1.0 - d12 ** 2

    if abs(det) < 1e-12:
        return None

    beta = b_val / det  # alpha = -d12 * beta
    alpha = -d12 * beta
    line_point = c1 + alpha * n1 + beta * n2

    # Distances from centers to the line
    dist1 = float(np.linalg.norm(cross(c1 - line_point, line_dir)))
    dist2 = float(np.linalg.norm(cross(c2 - line_point, line_dir)))

    if dist1 > radius1 or dist2 > radius2:
        return None

    # Half-chord lengths
    h1 = np.sqrt(max(0.0, radius1**2 - dist1**2))
    h2 = np.sqrt(max(0.0, radius2**2 - dist2**2))

    # Projection parameters
    t1 = float(dot(c1 - line_point, line_dir))
    t2 = float(dot(c2 - line_point, line_dir))

    # Overlap interval
    t_min = max(t1 - h1, t2 - h2)
    t_max = min(t1 + h1, t2 + h2)

    if t_min > t_max:
        return None

    seg_start = line_point + t_min * line_dir
    seg_end = line_point + t_max * line_dir
    seg_length = float(t_max - t_min)
    midpoint = (seg_start + seg_end) / 2.0

    return {
        "intersects": True,
        "segment_start": seg_start,
        "segment_end": seg_end,
        "segment_length": seg_length,
        "midpoint": midpoint,
        "line_direction": line_dir,
        "line_point": line_point,
    }


# =============================================================================
# Disk-AABB Intersection Area (Exact Polygon Clipping)
# =============================================================================

def _build_disk_polygon_2d(radius: float, n_sides: int = 64) -> NDArray[np.float64]:
    """Build a regular n-gon approximating a unit disk in the XY plane.

    Returns:
        (n_sides, 2) array of (x, y) vertex coordinates.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _clip_polygon_by_half_plane(
    poly: NDArray[np.float64],
    plane_normal_2d: NDArray[np.float64],
    plane_d: float,
) -> NDArray[np.float64]:
    """Clip a 2D polygon against the half-plane n·x ≤ d.

    Uses Sutherland-Hodgman algorithm.

    Args:
        poly: (N, 2) array of vertex coordinates.
        plane_normal_2d: (2,) unit normal pointing OUT of the valid region.
        plane_d: Signed distance from origin to the clipping line.

    Returns:
        (M, 2) array of clipped polygon vertices (may be empty).
    """
    if len(poly) == 0:
        return poly

    n = np.asarray(plane_normal_2d, dtype=np.float64)
    clipped = []

    n_verts = len(poly)
    for i in range(n_verts):
        p_curr = poly[i]
        p_next = poly[(i + 1) % n_verts]

        d_curr = np.dot(n, p_curr) - plane_d
        d_next = np.dot(n, p_next) - plane_d

        curr_inside = d_curr <= 0.0
        next_inside = d_next <= 0.0

        if curr_inside:
            clipped.append(p_curr)
            if not next_inside:
                # Leaving: add intersection point
                t = d_curr / (d_curr - d_next + 1e-30)
                intersection = p_curr + t * (p_next - p_curr)
                clipped.append(intersection)
        elif next_inside:
            # Entering: add intersection point
            t = d_curr / (d_curr - d_next + 1e-30)
            intersection = p_curr + t * (p_next - p_curr)
            clipped.append(intersection)

    if len(clipped) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(clipped, dtype=np.float64)


def _build_orthonormal_basis(normal: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build orthonormal basis (u, v) for the plane perpendicular to `normal`.

    Returns:
        (u, v) — two unit vectors spanning the plane, each (3,).
    """
    n = normalize(np.asarray(normal, dtype=np.float64))
    # Choose a reference vector not parallel to n
    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = normalize(cross(n, ref))  # perpendicular to n
    v = cross(n, u)               # perpendicular to n and u
    return u, v


def disk_aabb_intersection_area(
    disk_center: NDArray[np.float64],
    disk_normal: NDArray[np.float64],
    disk_radius: float,
    box_min: NDArray[np.float64],
    box_max: NDArray[np.float64],
    n_sides: int = 64,
) -> float:
    """Compute the area of intersection between a circular disk and an AABB.

    Uses polygon clipping: approximates the disk as a regular n-gon,
    clips it against the 6 half-planes of the AABB faces, and computes
    the area of the resulting polygon using Newell's method.

    This is the EXACT reference implementation. The previous
    _estimate_clipped_area() first-order falloff has been removed.

    Args:
        disk_center: Center of the disk (3,).
        disk_normal: Normal vector of the disk plane (3,). Need not be unit.
        disk_radius: Radius of the disk (m).
        box_min: Minimum corner of AABB (3,).
        box_max: Maximum corner of AABB (3,).
        n_sides: Number of polygon sides for disk approximation (default 64).

    Returns:
        Intersection area in m². Returns 0.0 if no intersection.
    """
    center = np.asarray(disk_center, dtype=np.float64)
    normal = normalize(np.asarray(disk_normal, dtype=np.float64))
    box_min = np.asarray(box_min, dtype=np.float64)
    box_max = np.asarray(box_max, dtype=np.float64)

    # Fast reject: if the disk's bounding sphere doesn't intersect the AABB
    if not disk_aabb_intersects(center, normal, disk_radius, box_min, box_max):
        return 0.0

    # Build orthonormal basis for the disk plane
    u, v = _build_orthonormal_basis(normal)

    # Build the disk polygon in 3D, then project to 2D
    disk_poly_2d = _build_disk_polygon_2d(disk_radius, n_sides)

    # Convert 2D disk polygon to 3D:  center + x*u + y*v
    disk_poly_3d = center + np.outer(disk_poly_2d[:, 0], u) + np.outer(disk_poly_2d[:, 1], v)

    # Project disk polygon to 2D using (u, v) basis
    # For any 3D point P, its 2D coords are: (dot(P-center, u), dot(P-center, v))
    centered = disk_poly_3d - center
    poly_2d = np.column_stack([
        np.dot(centered, u),
        np.dot(centered, v),
    ])

    # Compute the 6 half-plane constraints from the AABB.
    # Each box face defines a half-plane: the valid region is INSIDE the box.
    # Face plane: (x - box_min) · (+e_i) ≥ 0  →  -e_i · x ≤ -box_min[i]
    # Face plane: (x - box_max) · (-e_i) ≥ 0  →  +e_i · x ≤ +box_max[i]
    #
    # We need the half-plane equation in 2D: n_2d · (x,y) ≤ d_2d
    # For a 3D constraint: n_3d · P ≤ bound
    # Substituting P = center + x*u + y*v:
    #   n_3d · (center + x*u + y*v) ≤ bound
    #   x·(n_3d·u) + y·(n_3d·v) ≤ bound - n_3d·center
    #
    # So: n_2d = [n_3d·u, n_3d·v],  d_2d = bound - n_3d·center

    half_planes = []
    for dim in range(3):
        for bound, sign in [(box_min[dim], -1.0), (box_max[dim], 1.0)]:
            n_3d = np.zeros(3, dtype=np.float64)
            n_3d[dim] = sign
            # n_3d points inward (negative sign for min, positive for max)
            # The valid region is n_3d · P ≥ bound  (for min: P ≥ min; for max: P ≤ max)
            # Rewrite as: -sign·P ≤ -sign·bound if sign = -1 (min)
            #             sign·P ≤ sign·bound   if sign = +1 (max)
            # Both become: sign·P ≤ sign·bound for max, -sign·P ≤ -sign·bound for min
            # Which is equivalent to: n_valid · P ≤ bound_valid where
            #   n_valid = sign, bound_valid = sign * bound for max
            #   n_valid = -sign, bound_valid = -sign * bound for min
            #
            # Simpler: out of 6 faces, the inward normal always points toward
            # the box interior. The constraint is n_face · P ≤ n_face · face_point
            #
            # For min face: normal = -e_i, point on face = box_min
            #   -e_i · P ≤ -e_i · box_min → -P_i ≤ -box_min_i → P_i ≥ box_min_i ✓
            #
            # For max face: normal = +e_i, point on face = box_max
            #   +e_i · P ≤ +e_i · box_max → P_i ≤ box_max_i ✓

            if sign == -1.0:  # min face: normal points inward = -e_i
                n_face_3d = np.zeros(3, dtype=np.float64)
                n_face_3d[dim] = -1.0
                bound_3d = -bound  # = -box_min[dim]
            else:  # max face: normal points inward = +e_i
                n_face_3d = np.zeros(3, dtype=np.float64)
                n_face_3d[dim] = 1.0
                bound_3d = bound  # = box_max[dim]

            n_2d = np.array([np.dot(n_face_3d, u), np.dot(n_face_3d, v)])
            d_2d = bound_3d - np.dot(n_face_3d, center)
            half_planes.append((n_2d, d_2d))

    # Clip the disk polygon against all 6 half-planes
    for n_2d, d_2d in half_planes:
        if len(poly_2d) == 0:
            break
        poly_2d = _clip_polygon_by_half_plane(poly_2d, n_2d, d_2d)

    if len(poly_2d) < 3:
        return 0.0

    # Convert clipped 2D polygon back to 3D and compute area via Newell's method
    poly_3d = center + np.outer(poly_2d[:, 0], u) + np.outer(poly_2d[:, 1], v)

    from dfn_cave_studio.geometry.vector import polygon_area_3d
    return float(polygon_area_3d(poly_3d))
