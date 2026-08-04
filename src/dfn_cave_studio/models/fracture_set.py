"""
Fracture set (joint set) data models.

Each fracture set defines a statistical population of fractures:
  - Orientation distribution (Fisher)
  - Size distribution (lognormal, power-law, fixed)
  - Spatial distribution (uniform, clustered)
  - Target intensity (P32)
  - Mechanical property template

References:
  - Dershowitz & Einstein (1988). Characterizing rock joint geometry.
  - SCIENTIFIC_SPEC.md Section 3, 4, 5.
"""

from __future__ import annotations

from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from dfn_cave_studio.models.enums import (
    SizeDistributionType,
    SpatialDistributionType,
)


class OrientationDistribution(BaseModel):
    """Fisher distribution parameters for fracture orientation.

    Probability density:
      f(θ, φ) = (κ / (4π sinh κ)) * exp(κ * cos(angular_deviation))

    where angular_deviation is the angle between the sampled direction
    and the mean direction.

    References:
      - Fisher, R.A. (1953). Dispersion on a sphere.
      - Woodcock, N.H. (1977). Specification of fabric shapes.
    """

    mean_dip_direction: float = Field(default=0.0, ge=0.0, lt=360.0, description="Mean dip direction (°)")
    mean_dip: float = Field(default=0.0, ge=0.0, le=90.0, description="Mean dip angle (°)")
    kappa: float = Field(default=20.0, gt=0.0, description="Fisher concentration parameter (higher = more clustered)")

    @property
    def angular_std_dev_deg(self) -> float:
        """Approximate angular standard deviation from kappa.

        For large kappa (>5), std ≈ 81 / sqrt(kappa) degrees.
        For small kappa, this approximation breaks down.
        """
        import math
        if self.kappa > 5:
            return 81.0 / math.sqrt(self.kappa)
        else:
            return 180.0 / math.sqrt(self.kappa)  # rough estimate


class SizeDistribution(BaseModel):
    """Fracture size (radius) distribution parameters.

    Supported distributions:
      - lognormal: radius ~ logN(μ, σ)
      - power_law: P(R > r) ∝ r^(-D) [truncated]
      - truncated_power_law: as above with lower/upper cutoffs
      - fixed: constant radius
      - exponential: radius ~ Exp(λ)
    """

    distribution_type: SizeDistributionType = SizeDistributionType.LOGNORMAL

    # Lognormal parameters (ln(radius) ~ N(mu, sigma))
    lognormal_mu: float = Field(default=1.0, description="Mean of ln(radius)")
    lognormal_sigma: float = Field(default=0.5, gt=0.0, description="Std of ln(radius)")

    # Power-law parameters
    power_law_exponent: float = Field(default=3.0, gt=0.0, description="Power-law exponent D")

    # Truncation (applies to all types)
    min_radius: float = Field(default=0.5, gt=0.0, description="Minimum fracture radius (m)")
    max_radius: float = Field(default=10.0, gt=0.0, description="Maximum fracture radius (m)")

    @model_validator(mode="after")
    def validate_radius_range(self) -> "SizeDistribution":
        if self.min_radius >= self.max_radius:
            raise ValueError(f"min_radius ({self.min_radius}) must be < max_radius ({self.max_radius})")
        return self

    @property
    def mean_radius(self) -> float:
        """Compute the mean radius for the configured distribution."""
        import math
        if self.distribution_type == SizeDistributionType.LOGNORMAL:
            return math.exp(self.lognormal_mu + self.lognormal_sigma ** 2 / 2)
        elif self.distribution_type == SizeDistributionType.FIXED:
            return self.min_radius
        elif self.distribution_type == SizeDistributionType.POWER_LAW:
            # Truncated power-law mean (approximate)
            D = self.power_law_exponent
            if D == 2.0:
                return self.min_radius * math.log(self.max_radius / self.min_radius)
            elif D == 3.0:
                r_min, r_max = self.min_radius, self.max_radius
                return (r_min * r_max * math.log(r_max / r_min)) / (r_max - r_min)
            else:
                r_min, r_max = self.min_radius, self.max_radius
                return (D - 1) / (D - 2) * (r_max ** (2 - D) - r_min ** (2 - D)) / (r_max ** (1 - D) - r_min ** (1 - D))
        else:
            return (self.min_radius + self.max_radius) / 2.0


class SpatialDistribution(BaseModel):
    """Spatial position distribution for fracture centers."""

    distribution_type: SpatialDistributionType = SpatialDistributionType.UNIFORM

    # Clustering parameters (for future use)
    cluster_centers: Optional[list] = None
    cluster_radius: Optional[float] = None


class JointSetConfig(BaseModel):
    """Configuration for a single fracture (joint) set.

    This is the primary user-facing configuration object for DFN generation.
    It encapsulates all statistical parameters needed to generate one family
    of fractures.
    """

    # Identification
    set_id: int = 0
    name: str = "Joint Set 1"
    color: str = "#1976d2"  # Display color

    # Orientation
    orientation: OrientationDistribution = Field(default_factory=OrientationDistribution)

    # Size
    size: SizeDistribution = Field(default_factory=SizeDistribution)

    # Spatial distribution
    spatial: SpatialDistribution = Field(default_factory=SpatialDistribution)

    # Intensity
    target_p32: float = Field(default=1.0, gt=0.0, description="Target P32 (m²/m³)")
    p32_tolerance: float = Field(default=0.05, ge=0.0, le=1.0, description="Relative tolerance for P32 convergence")

    # Display
    visible: bool = True
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)

    # Mechanical property template
    mechanical_template_id: Optional[str] = None

    def expected_mean_area(self) -> float:
        """Expected mean fracture area for this set (m²).

        Uses the disk model: area = π * r².
        """
        mean_r = self.size.mean_radius
        import math
        return math.pi * mean_r ** 2

    def expected_fracture_count(self, rock_volume: float) -> int:
        """Expected number of fractures to achieve target P32.

        N = P32_target * V_rock / mean_area

        Args:
            rock_volume: Volume of rock mass (m³).

        Returns:
            Expected fracture count (integer).
        """
        mean_area = self.expected_mean_area()
        if mean_area <= 0:
            return 0
        n = self.target_p32 * rock_volume / mean_area
        return max(1, int(round(n)))
