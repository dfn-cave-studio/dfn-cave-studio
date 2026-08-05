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
        """Compute the mean (expected) radius E[R] for the configured distribution."""
        import math
        if self.distribution_type == SizeDistributionType.LOGNORMAL:
            return math.exp(self.lognormal_mu + self.lognormal_sigma ** 2 / 2)
        elif self.distribution_type == SizeDistributionType.FIXED:
            return self.min_radius
        elif self.distribution_type == SizeDistributionType.POWER_LAW:
            D = self.power_law_exponent
            if D == 2.0:
                return self.min_radius * math.log(self.max_radius / self.min_radius)
            elif D == 3.0:
                r_min, r_max = self.min_radius, self.max_radius
                return (r_min * r_max * math.log(r_max / r_min)) / (r_max - r_min)
            else:
                r_min, r_max = self.min_radius, self.max_radius
                return (D - 1) / (D - 2) * (r_max ** (2 - D) - r_min ** (2 - D)) / (r_max ** (1 - D) - r_min ** (1 - D))
        elif self.distribution_type == SizeDistributionType.TRUNCATED_POWER_LAW:
            D = self.power_law_exponent
            r_min, r_max = self.min_radius, self.max_radius
            if abs(D - 2.0) < 1e-10:
                return (r_min * r_max * math.log(r_max / r_min)) / (r_max - r_min)
            elif abs(D - 3.0) < 1e-10:
                return (r_min * r_max * math.log(r_max / r_min)) / (r_max - r_min)
            else:
                return (D - 1) / (D - 2) * (r_max ** (2 - D) - r_min ** (2 - D)) / (r_max ** (1 - D) - r_min ** (1 - D))
        elif self.distribution_type == SizeDistributionType.EXPONENTIAL:
            mean_r = (self.min_radius + self.max_radius) / 2.0
            return mean_r
        else:
            return (self.min_radius + self.max_radius) / 2.0

    @property
    def mean_squared_radius(self) -> float:
        """Compute E[R²] — the expected squared radius.

        CRITICAL for P32 control: fracture area A = πR², so the expected
        fracture area is E[A] = π·E[R²], NOT π·(E[R])².

        The difference is E[R²] = (E[R])² + Var(R), i.e. Jensen's gap.
        For a lognormal with σ=0.5, E[R²]/(E[R])² = exp(σ²) ≈ 1.284,
        meaning the naive π·(E[R])² underestimates area by ~28%.
        """
        import math
        dist_type = self.distribution_type

        if dist_type == SizeDistributionType.LOGNORMAL:
            # ln(R) ~ N(μ, σ) → E[R²] = exp(2μ + 2σ²)
            return math.exp(2.0 * self.lognormal_mu + 2.0 * self.lognormal_sigma ** 2)

        elif dist_type == SizeDistributionType.FIXED:
            r = self.min_radius
            return r * r

        elif dist_type == SizeDistributionType.POWER_LAW:
            # Untruncated power-law: f(r) ∝ r^{-(D+1)}, r ∈ [r_min, ∞)
            D = self.power_law_exponent
            r_min = self.min_radius
            if D <= 2.0:
                # E[R²] diverges for D ≤ 2; use r_max as effective cutoff
                return self.max_radius ** 2
            return (D / (D - 2.0)) * r_min ** 2

        elif dist_type == SizeDistributionType.TRUNCATED_POWER_LAW:
            D = self.power_law_exponent
            r_min, r_max = self.min_radius, self.max_radius
            if abs(D - 3.0) < 1e-10:
                # D=3: f(r) ∝ r^{-4}, exact E[R²]:
                #   E[R²] = 3 / (1/r_min² + 1/(r_min·r_max) + 1/r_max²)
                #         = 3·r_min²·r_max² / (r_max² + r_min·r_max + r_min²)
                return 3.0 * r_min**2 * r_max**2 / (r_max**2 + r_min * r_max + r_min**2)
            if abs(D - 2.0) < 1e-10:
                # D=2: f(r) ∝ r^{-3}, exact E[R²]:
                #   E[R²] = 2·r_min²·r_max²·ln(r_max/r_min) / (r_max² - r_min²)
                if abs(r_max - r_min) < 1e-12:
                    return r_min**2
                return (2.0 * r_min**2 * r_max**2 * math.log(r_max / r_min)) / (r_max**2 - r_min**2)
            if D <= 2.0:
                return self.max_radius ** 2
            # General D: E[R²] = ∫ r^{2}·r^{-(D+1)} / ∫ r^{-(D+1)}
            # = ∫ r^{1-D} / ∫ r^{-D-1}
            # = [r^{2-D}/(2-D)] / [r^{-D}/(-D)]
            # = (D/(D-2)) · (r_min^{2-D} - r_max^{2-D}) / (r_min^{-D} - r_max^{-D})
            num = r_min ** (2.0 - D) - r_max ** (2.0 - D)
            den = r_min ** (-D) - r_max ** (-D)
            if abs(den) < 1e-15:
                return r_min ** 2
            return (D / (D - 2.0)) * num / den

        elif dist_type == SizeDistributionType.EXPONENTIAL:
            mean_r = (self.min_radius + self.max_radius) / 2.0
            # Exp(λ): E[R] = 1/λ, Var(R) = 1/λ², E[R²] = 2/λ² = 2·(E[R])²
            return 2.0 * mean_r ** 2

        else:
            mean_r = (self.min_radius + self.max_radius) / 2.0
            return mean_r ** 2


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

        Uses the disk model: E[A] = π · E[R²].

        IMPORTANT: This is NOT π·(E[R])². Because area ∝ radius²,
        the variance of the radius distribution inflates the expected
        area (Jensen's inequality). Using π·(E[R])² systematically
        underestimates P32.
        """
        import math
        return math.pi * self.size.mean_squared_radius

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
