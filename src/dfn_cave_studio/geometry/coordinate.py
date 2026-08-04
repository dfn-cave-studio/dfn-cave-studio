"""
Coordinate transforms and orientation conversions for DFN Cave Studio.

Conventions:
  - Dip direction: 0°–360° (clockwise from north / +Y axis)
  - Dip: 0°–90° (from horizontal)
  - Internal trig functions use radians
  - UI/IO uses degrees

Normal vector convention:
  - For a plane with dip direction α and dip β:
    - The normal points OUT of the rock mass (into the fracture)
    - Normal = [-sin(α)*sin(β), -cos(α)*sin(β), cos(β)]
    - The normal has positive Z component (points upward) for dips < 90°

References:
  - Priest, S.D. (1993). Discontinuity Analysis for Rock Engineering.
  - SCIENTIFIC_SPEC.md Section 2.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.geometry.vector import normalize


# =============================================================================
# Dip Direction / Dip ↔ Normal Vector
# =============================================================================

def dip_dir_dip_to_normal(dip_direction_deg: float, dip_deg: float) -> NDArray[np.float64]:
    """Convert dip direction and dip to a unit normal vector.

    The normal vector points into the fracture (away from intact rock).
    For standard upper-hemisphere convention, the Z component is positive.

    Args:
        dip_direction_deg: Dip direction in degrees (0–360°, clockwise from north).
        dip_deg: Dip angle in degrees (0–90°, from horizontal).

    Returns:
        Unit normal vector (3,) with Z ≥ 0.

    Example:
        >>> dip_dir_dip_to_normal(0, 90)    # Horizontal plane, dip dir north
        array([0., 0., 1.])
        >>> dip_dir_dip_to_normal(90, 0)    # Vertical plane, dip dir east
        array([-1., 0., 0.])
    """
    alpha = math.radians(dip_direction_deg)
    beta = math.radians(dip_deg)

    nx = -math.sin(alpha) * math.sin(beta)
    ny = -math.cos(alpha) * math.sin(beta)
    nz = math.cos(beta)

    n = np.array([nx, ny, nz], dtype=np.float64)
    # Ensure positive Z for upper hemisphere
    if n[2] < 0:
        n = -n

    return n


def normal_to_dip_dir_dip(normal: NDArray[np.float64]) -> Tuple[float, float]:
    """Convert a unit normal vector to dip direction and dip.

    Args:
        normal: Unit normal vector (3,). Will be normalized if needed.

    Returns:
        Tuple of (dip_direction_deg, dip_deg).
        Dip direction in [0, 360), dip in [0, 90].

    Example:
        >>> normal_to_dip_dir_dip([0, 0, 1])
        (0.0, 0.0)
    """
    n = normalize(normal)

    # Ensure upper hemisphere
    if n[2] < 0:
        n = -n

    if abs(n[2] - 1.0) < 1e-12:
        # Purely horizontal fracture: dip = 0, dip direction undefined → 0
        return 0.0, 0.0

    beta = math.acos(n[2])  # dip angle in radians
    dip = math.degrees(beta)

    # Dip direction from normal components
    # n = [-sin(α)*sin(β), -cos(α)*sin(β), cos(β)]
    # If sin(β) ≈ 0, fracture is horizontal → α undefined
    sin_beta = math.sin(beta)
    if sin_beta < 1e-12:
        return 0.0, 0.0

    sin_alpha = -n[0] / sin_beta
    cos_alpha = -n[1] / sin_beta

    alpha = math.atan2(sin_alpha, cos_alpha)
    dip_dir = math.degrees(alpha)

    # Normalize to [0, 360)
    dip_dir = dip_dir % 360.0

    return dip_dir, dip


def normal_to_pole(normal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a plane normal to a pole vector for stereographic projection.

    Pole = intersection of the outward normal with the lower hemisphere.

    Args:
        normal: Unit normal vector (upper hemisphere convention).

    Returns:
        Pole vector (3,) pointing downward for stereonet plotting.
    """
    n = normalize(normal)
    # Ensure lower hemisphere for stereonet pole
    if n[2] > 0:
        return -n
    return n


# =============================================================================
# Rotation Matrices
# =============================================================================

def rotation_matrix_x(angle_rad: float) -> NDArray[np.float64]:
    """Rotation matrix about X axis (right-hand rule).

    Args:
        angle_rad: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float64)


def rotation_matrix_y(angle_rad: float) -> NDArray[np.float64]:
    """Rotation matrix about Y axis (right-hand rule).

    Args:
        angle_rad: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def rotation_matrix_z(angle_rad: float) -> NDArray[np.float64]:
    """Rotation matrix about Z axis (right-hand rule).

    Args:
        angle_rad: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)


def rotate_vector(
    v: NDArray[np.float64],
    axis: str,
    angle_rad: float,
) -> NDArray[np.float64]:
    """Rotate a vector about a principal axis.

    Args:
        v: Vector to rotate (3,).
        axis: 'x', 'y', or 'z'.
        angle_rad: Rotation angle in radians.

    Returns:
        Rotated vector (3,).
    """
    matrices = {
        'x': rotation_matrix_x,
        'y': rotation_matrix_y,
        'z': rotation_matrix_z,
    }
    if axis.lower() not in matrices:
        raise ValueError(f"Axis must be 'x', 'y', or 'z', got '{axis}'")
    R = matrices[axis.lower()](angle_rad)
    return R @ v


# =============================================================================
# Strike Conversion
# =============================================================================

def dip_dir_to_strike(dip_direction_deg: float, right_hand_rule: bool = True) -> float:
    """Convert dip direction to strike.

    Args:
        dip_direction_deg: Dip direction (0–360°).
        right_hand_rule: If True, strike = dip_dir - 90° (right-hand rule).
                        If False, strike is measured 90° CCW from dip direction.

    Returns:
        Strike in degrees [0, 360).
    """
    if right_hand_rule:
        strike = dip_direction_deg - 90.0
    else:
        strike = dip_direction_deg + 90.0
    return strike % 360.0


def strike_dip_to_normal(
    strike_deg: float, dip_deg: float, right_hand_rule: bool = True
) -> NDArray[np.float64]:
    """Convert strike and dip to a unit normal vector.

    Args:
        strike_deg: Strike angle (0–360°).
        dip_deg: Dip angle (0–90°).
        right_hand_rule: If True, dip is 90° clockwise from strike.

    Returns:
        Unit normal vector (3,).
    """
    if right_hand_rule:
        dip_dir = strike_deg + 90.0
    else:
        dip_dir = strike_deg - 90.0
    return dip_dir_dip_to_normal(dip_dir % 360.0, dip_deg)


# =============================================================================
# Batch Operations
# =============================================================================

def batch_dip_dir_dip_to_normal(
    dip_directions_deg: NDArray[np.float64],
    dips_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert arrays of dip directions and dips to normal vectors.

    Args:
        dip_directions_deg: (N,) array of dip directions in degrees.
        dips_deg: (N,) array of dips in degrees.

    Returns:
        (N, 3) array of unit normal vectors.
    """
    alpha = np.radians(dip_directions_deg)
    beta = np.radians(dips_deg)

    nx = -np.sin(alpha) * np.sin(beta)
    ny = -np.cos(alpha) * np.sin(beta)
    nz = np.cos(beta)

    normals = np.column_stack([nx, ny, nz])

    # Flip lower-hemisphere vectors
    flip = nz < 0
    normals[flip] = -normals[flip]

    return normals


def batch_normal_to_dip_dir_dip(
    normals: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert arrays of normal vectors to dip directions and dips.

    Args:
        normals: (N, 3) array of normal vectors.

    Returns:
        Tuple of (dip_directions_deg, dips_deg), each (N,).
    """
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    n = normals / norms

    # Upper hemisphere
    flip = n[:, 2] < 0
    n[flip] = -n[flip]

    # Dip from nz
    nz_clipped = np.clip(n[:, 2], -1.0, 1.0)
    beta = np.arccos(nz_clipped)
    dips_deg = np.degrees(beta)

    # Dip direction
    sin_beta = np.sin(beta)
    dip_dirs = np.zeros_like(dips_deg)
    mask = sin_beta > 1e-12
    sin_alpha = -n[mask, 0] / sin_beta[mask]
    cos_alpha = -n[mask, 1] / sin_beta[mask]
    dip_dirs[mask] = np.degrees(np.arctan2(sin_alpha, cos_alpha)) % 360.0

    return dip_dirs, dips_deg
