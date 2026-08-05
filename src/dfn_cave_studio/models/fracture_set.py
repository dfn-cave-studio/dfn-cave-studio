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

from typing import Optional, Dict
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
        """Compute the mean (expected) radius E[R] for the configured distribution.

        All formulas match the actual sampling in DFNGenerator._sample_radii().
        """
        import math
        r_min, r_max = self.min_radius, self.max_radius

        if self.distribution_type == SizeDistributionType.LOGNORMAL:
            return math.exp(self.lognormal_mu + self.lognormal_sigma ** 2 / 2)

        elif self.distribution_type == SizeDistributionType.FIXED:
            # Generator samples (min+max)/2 — must match.
            return (r_min + r_max) / 2.0

        elif self.distribution_type == SizeDistributionType.POWER_LAW:
            # Generator: r = r_min * u^(-1/D), clipped to [r_min, r_max].
            # Equivalent to truncated power-law with clamping.
            D = self.power_law_exponent
            if abs(D - 1.0) < 1e-10:
                # D=1 truncated: E[R] = r_min*r_max*ln(r_max/r_min) / (r_max - r_min)
                return r_min * r_max * math.log(r_max / r_min) / (r_max - r_min)
            elif abs(D - 2.0) < 1e-10:
                # General formula applied at D=2: 2*(r_min⁻¹-r_max⁻¹)/(r_min⁻²-r_max⁻²)
                num2 = r_min ** (-1.0) - r_max ** (-1.0)
                den2 = r_min ** (-2.0) - r_max ** (-2.0)
                if abs(den2) < 1e-15:
                    return r_min
                return 2.0 * num2 / den2
            elif abs(D - 3.0) < 1e-10:
                # D=3 truncated power-law: E[R] = 1.5 * r_min*r_max*(r_max+r_min) / (r_max² + r_min*r_max + r_min²)
                return 1.5 * r_min * r_max * (r_max + r_min) / (r_max**2 + r_min * r_max + r_min**2)
            else:
                # General D truncated power-law: E[R] = D/(D-1) * (r_min^{1-D} - r_max^{1-D}) / (r_min^{-D} - r_max^{-D})
                num = r_min ** (1.0 - D) - r_max ** (1.0 - D)
                den = r_min ** (-D) - r_max ** (-D)
                if abs(den) < 1e-15:
                    return r_min
                return (D / (D - 1.0)) * num / den

        elif self.distribution_type == SizeDistributionType.TRUNCATED_POWER_LAW:
            D = self.power_law_exponent
            if abs(D - 1.0) < 1e-10:
                # D=1 truncated: E[R] = r_min*r_max*ln(r_max/r_min) / (r_max - r_min)
                return r_min * r_max * math.log(r_max / r_min) / (r_max - r_min)
            elif abs(D - 2.0) < 1e-10:
                # General formula applied at D=2: 2*(r_min⁻¹-r_max⁻¹)/(r_min⁻²-r_max⁻²)
                num2 = r_min ** (-1.0) - r_max ** (-1.0)
                den2 = r_min ** (-2.0) - r_max ** (-2.0)
                if abs(den2) < 1e-15:
                    return r_min
                return 2.0 * num2 / den2
            elif abs(D - 3.0) < 1e-10:
                # D=3 truncated power-law: E[R] = 1.5 * r_min*r_max*(r_max+r_min) / (r_max² + r_min*r_max + r_min²)
                return 1.5 * r_min * r_max * (r_max + r_min) / (r_max**2 + r_min * r_max + r_min**2)
            else:
                num = r_min ** (1.0 - D) - r_max ** (1.0 - D)
                den = r_min ** (-D) - r_max ** (-D)
                if abs(den) < 1e-15:
                    return r_min
                return (D / (D - 1.0)) * num / den

        elif self.distribution_type == SizeDistributionType.EXPONENTIAL:
            # Truncated exponential. λ = 2/(min+max).
            # E[R] = ∫ r·λ·e^{-λr} / (e^{-λr_min} - e^{-λr_max}) dr
            # = [B(r_min) - B(r_max)] / [e^{-λr_min} - e^{-λr_max}]
            # where B(r) = (r + 1/λ)·e^{-λr}
            lam = 2.0 / (r_min + r_max) if (r_min + r_max) > 0 else 1.0
            exp_min = math.exp(-lam * r_min)
            exp_max = math.exp(-lam * r_max)
            denom = exp_min - exp_max
            if abs(denom) < 1e-15:
                return (r_min + r_max) / 2.0
            B_min = (r_min + 1.0 / lam) * exp_min
            B_max = (r_max + 1.0 / lam) * exp_max
            return (B_min - B_max) / denom

        else:
            return (r_min + r_max) / 2.0

    @property
    def mean_squared_radius(self) -> float:
        """Compute E[R²] — the expected squared radius.

        CRITICAL for P32 control: fracture area A = πR², so the expected
        fracture area is E[A] = π·E[R²], NOT π·(E[R])².

        The difference is E[R²] = (E[R])² + Var(R), i.e. Jensen's gap.
        For a lognormal with σ=0.5, E[R²]/(E[R])² = exp(σ²) ≈ 1.284,
        meaning the naive π·(E[R])² underestimates area by ~28%.

        All formulas match the actual sampling in DFNGenerator._sample_radii().
        """
        import math
        dist_type = self.distribution_type
        r_min, r_max = self.min_radius, self.max_radius

        if dist_type == SizeDistributionType.LOGNORMAL:
            # ln(R) ~ N(μ, σ) → E[R²] = exp(2μ + 2σ²)
            return math.exp(2.0 * self.lognormal_mu + 2.0 * self.lognormal_sigma ** 2)

        elif dist_type == SizeDistributionType.FIXED:
            # Generator samples (min+max)/2 — must match.
            r = (r_min + r_max) / 2.0
            return r * r

        elif dist_type == SizeDistributionType.POWER_LAW:
            # Generator: r = r_min * u^(-1/D), clipped to [r_min, r_max].
            # Equivalent to truncated power-law — use truncated formula.
            D = self.power_law_exponent
            if abs(D - 2.0) < 1e-10:
                # D=2 truncated: E[R²] = 2*r_min²*r_max²*ln(r_max/r_min) / (r_max² - r_min²)
                if abs(r_max - r_min) < 1e-12:
                    return r_min**2
                return (2.0 * r_min**2 * r_max**2 * math.log(r_max / r_min)) / (r_max**2 - r_min**2)
            elif abs(D - 3.0) < 1e-10:
                # D=3 truncated: E[R²] = 3*r_min²*r_max² / (r_max² + r_min*r_max + r_min²)
                return 3.0 * r_min**2 * r_max**2 / (r_max**2 + r_min * r_max + r_min**2)
            elif D <= 2.0:
                # For D ≤ 2 the untruncated power-law diverges, but the
                # truncated distribution has finite moments.  Use the general
                # formula below instead of a shortcut.
                pass
            # General D truncated: E[R²] = D/(D-2) * (r_min^{2-D} - r_max^{2-D}) / (r_min^{-D} - r_max^{-D})
            num = r_min ** (2.0 - D) - r_max ** (2.0 - D)
            den = r_min ** (-D) - r_max ** (-D)
            if abs(den) < 1e-15:
                return r_min ** 2
            return (D / (D - 2.0)) * num / den

        elif dist_type == SizeDistributionType.TRUNCATED_POWER_LAW:
            D = self.power_law_exponent
            if abs(D - 2.0) < 1e-10:
                if abs(r_max - r_min) < 1e-12:
                    return r_min**2
                return (2.0 * r_min**2 * r_max**2 * math.log(r_max / r_min)) / (r_max**2 - r_min**2)
            elif abs(D - 3.0) < 1e-10:
                return 3.0 * r_min**2 * r_max**2 / (r_max**2 + r_min * r_max + r_min**2)
            elif D <= 2.0:
                pass  # fall through to general formula below
            num = r_min ** (2.0 - D) - r_max ** (2.0 - D)
            den = r_min ** (-D) - r_max ** (-D)
            if abs(den) < 1e-15:
                return r_min ** 2
            return (D / (D - 2.0)) * num / den

        elif dist_type == SizeDistributionType.EXPONENTIAL:
            # Truncated exponential. λ = 2/(min+max).
            # E[R²] = ∫ r²·λ·e^{-λr} / (e^{-λr_min} - e^{-λr_max}) dr from r_min to r_max
            # Closed form:
            #   ∫ r²·λ·e^{-λr} = -(r² + 2r/λ + 2/λ²)·e^{-λr}
            #   E[R²] = [A(r_min) - A(r_max)] / [e^{-λr_min} - e^{-λr_max}]
            #   where A(r) = (r² + 2r/λ + 2/λ²)·e^{-λr}
            lam = 2.0 / (r_min + r_max) if (r_min + r_max) > 0 else 1.0
            exp_min = math.exp(-lam * r_min)
            exp_max = math.exp(-lam * r_max)
            denom = exp_min - exp_max
            if abs(denom) < 1e-15:
                mean_r = (r_min + r_max) / 2.0
                return mean_r ** 2
            A_min = (r_min**2 + 2.0 * r_min / lam + 2.0 / lam**2) * exp_min
            A_max = (r_max**2 + 2.0 * r_max / lam + 2.0 / lam**2) * exp_max
            return (A_min - A_max) / denom

        else:
            mean_r = (r_min + r_max) / 2.0
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

    # Provenance: records whether each parameter came from borehole observations
    # or user input. Keys: "orientation", "size", "p32". Values: "borehole" or "user".
    provenance: Dict[str, str] = Field(default_factory=dict, description="Parameter provenance: 'borehole' or 'user'")

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
