"""
Fisher distribution sampling for fracture orientations.

Implements the Fisher (1953) spherical distribution sampling algorithm.
Used to generate fracture orientations from mean dip direction/dip and
concentration parameter kappa.

Algorithm: Woodcock (1977) method
  1. Sample colatitude θ from P(θ) ∝ exp(κ * cos θ) * sin θ
  2. Sample azimuth φ uniformly from [0, 2π)
  3. Rotate to the mean orientation using orthonormal basis

References:
  - Fisher, R.A. (1953). Dispersion on a sphere.
  - Woodcock, N.H. (1977). Specification of fabric shapes using a
    Bingham distribution.
  - Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics.
  - SCIENTIFIC_SPEC.md Section 3.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal


def fisher_sample(
    mean_dip_direction_deg: float,
    mean_dip_deg: float,
    kappa: float,
    rng: Optional[np.random.Generator] = None,
    n_samples: int = 1,
) -> NDArray[np.float64]:
    """Sample orientations from a Fisher distribution.

    Generates n_samples unit normal vectors distributed about the mean
    orientation with concentration kappa.

    Algorithm (Woodcock 1977):
      1. Build orthonormal basis {u, v, w} where w = mean normal.
      2. Sample colatitude θ from the Fisher density:
           F(θ) = (1 - exp(-κ*(cos θ + 1))) / (1 - exp(-2κ))
           for cos θ ∈ [-1, 1].
         Solve for cos θ via inverse CDF:
           cos θ = 1 + ln(1 - U*(1 - exp(-2κ))) / κ
         where U ~ Uniform(0, 1).
      3. Sample azimuth φ ~ Uniform(0, 2π).
      4. Build sample = sin θ * (cos φ * u + sin φ * v) + cos θ * w.

    Args:
        mean_dip_direction_deg: Mean dip direction (0–360°).
        mean_dip_deg: Mean dip angle (0–90°).
        kappa: Fisher concentration parameter (>0).
        rng: Seeded numpy random generator. If None, uses np.random.default_rng().
        n_samples: Number of samples to generate.

    Returns:
        (n_samples, 3) array of unit normal vectors from the Fisher distribution.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> normals = fisher_sample(45, 30, 20, rng, n_samples=100)
        >>> normals.shape
        (100, 3)
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if n_samples == 0:
        return np.empty((0, 3), dtype=np.float64)
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if rng is None:
        rng = np.random.default_rng()

    # Mean orientation as unit normal vector
    w = dip_dir_dip_to_normal(mean_dip_direction_deg, mean_dip_deg)

    # Build orthonormal basis {u, v, w}
    u, v = _build_tangent_basis(w)

    # Sample colatitude θ
    # For large kappa, use the approximate method directly
    if kappa > 50:
        # Large-kappa approximation: θ ~ Rayleigh-like
        cos_theta = 1.0 + np.log(1.0 - rng.random(n_samples) * (1.0 - math.exp(-2.0 * kappa))) / kappa
    else:
        # General case
        u_random = rng.random(n_samples)
        cos_theta = 1.0 + np.log(1.0 - u_random * (1.0 - math.exp(-2.0 * kappa))) / kappa

    # Clamp cos θ to [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)

    # Sample azimuth φ uniformly
    phi = rng.uniform(0, 2.0 * math.pi, n_samples)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Combine: n = sin θ * (cos φ * u + sin φ * v) + cos θ * w
    samples = (
        sin_theta[:, np.newaxis] * (cos_phi[:, np.newaxis] * u[np.newaxis, :] + sin_phi[:, np.newaxis] * v[np.newaxis, :])
        + cos_theta[:, np.newaxis] * w[np.newaxis, :]
    )

    # Normalize
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples / norms

    # Enforce upper hemisphere (Z >= 0)
    flip = samples[:, 2] < 0
    if n_samples > 1:
        samples[flip] = -samples[flip]
    elif flip[0]:
        samples = -samples

    return samples


def _build_tangent_basis(
    normal: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build an orthonormal basis {u, v} perpendicular to the given normal.

    Uses the method: pick a vector not parallel to normal, compute u via
    cross product, then v = w × u.

    Args:
        normal: Unit normal vector (3,) — the "w" direction.

    Returns:
        Tuple of (u, v), each (3,) unit vectors, with u·v = u·w = v·w = 0.
    """
    # Pick a reference vector not parallel to normal
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    # u = ref × w (perpendicular to both)
    u = np.cross(ref, normal)
    u = u / np.linalg.norm(u)

    # v = w × u (completes right-handed system)
    v = np.cross(normal, u)

    return u, v


def estimate_kappa(
    normals: NDArray[np.float64],
) -> float:
    """Estimate the Fisher concentration parameter kappa from sample normals.

    Uses the MLE: kappa ≈ (N-1) / (N - R) for N > 3,
    where R is the length of the resultant vector.

    Reference: Mardia & Jupp (2000), Directional Statistics.

    Args:
        normals: (N, 3) array of sample unit normal vectors.

    Returns:
        Estimated kappa value.
    """
    N = normals.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 samples to estimate kappa")

    # Resultant vector
    R_vec = np.sum(normals, axis=0)
    R = float(np.linalg.norm(R_vec))

    if R >= N * (1.0 - 1e-12):
        # All vectors nearly identical → kappa is very large
        return 1e10

    # Mardia & Jupp estimate
    kappa = (N - 1) / (N - R) if R < N else 1e10

    return kappa
