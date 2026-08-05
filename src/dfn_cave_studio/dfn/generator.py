"""
Stochastic DFN generator.

Generates Discrete Fracture Networks from statistical input parameters:
  - User-defined fracture sets (JointSetConfig)
  - Fisher orientation sampling
  - Size distribution sampling
  - Uniform spatial positioning
  - P32 convergence control

The generator creates a DFNRealization containing stochastic fractures
with full geometry metadata for visualization, analysis, and export.

References:
  - Baecher, G.B. et al. (1977). Statistical description of rock properties.
  - Dershowitz, W.S. & Einstein, H.H. (1988). Characterizing rock joint geometry.
  - SCIENTIFIC_SPEC.md Section 4, 5.
"""

from __future__ import annotations

import time
import math
from typing import List, Optional, Callable, Dict, Any

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.models.fracture import (
    StochasticFracture,
    FractureGeometry,
    create_fracture_from_dip,
)
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.dfn_realization import (
    DFNGenerationConfig,
    DFNGenerationResult,
    DFNRealization,
)
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.enums import (
    SizeDistributionType,
    FractureSource,
)
from dfn_cave_studio.dfn.fisher import fisher_sample


class DFNGenerator:
    """Generates stochastic DFN realizations from statistical parameters.

    The generator creates fractures by:
      1. Estimating the required number of fractures from target P32
      2. Sampling orientations from Fisher distributions per joint set
      3. Sampling sizes from the configured distribution
      4. Placing fracture centers uniformly in the model volume
      5. Trimming fractures to the model boundary
      6. Checking P32 convergence
    """

    def __init__(
        self,
        config: DFNGenerationConfig,
        bounds: ModelBounds,
    ):
        """Initialize the DFN generator.

        Args:
            config: Generation configuration (joint sets, seeds, etc.).
            bounds: Model bounding box.
        """
        self.config = config
        self.bounds = bounds
        self._rng: Optional[np.random.Generator] = None
        self._cancelled = False

    # ------------------------------------------------------------------
    # Progress Callback
    # ------------------------------------------------------------------

    def set_progress_callback(
        self, callback: Optional[Callable[[int, int, str], None]]
    ) -> None:
        """Set a callback for generation progress updates.

        Args:
            callback: Function(current, total, message) called during generation.
                     Set to None to disable progress.
        """
        self._progress_callback = callback

    def _report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report generation progress if callback is set."""
        if hasattr(self, '_progress_callback') and self._progress_callback:
            self._progress_callback(current, total, message)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation of the current generation."""
        self._cancelled = True

    def _check_cancelled(self) -> None:
        """Raise if generation has been cancelled."""
        if self._cancelled:
            raise InterruptedError("DFN generation cancelled by user")

    # ------------------------------------------------------------------
    # Main Generation
    # ------------------------------------------------------------------

    def generate(self, realization_number: int = 0) -> DFNRealization:
        """Generate a complete DFN realization.

        Args:
            realization_number: Index of this realization (for multi-realization sets).

        Returns:
            DFNRealization containing all generated fractures and statistics.

        Raises:
            InterruptedError: If generation is cancelled.
        """
        t_start = time.perf_counter()

        # Check for cancellation before starting
        self._check_cancelled()

        # Initialize RNG with master seed + realization number
        seed = self.config.master_seed + realization_number * 1000
        self._rng = np.random.default_rng(seed)

        model_volume = self.bounds.volume
        all_fractures: List[StochasticFracture] = []
        set_statistics: Dict[int, Dict[str, Any]] = {}

        total_sets = len(self.config.joint_sets)
        parameter_provenance: Dict[int, Dict[str, str]] = {}

        for idx, joint_set in enumerate(self.config.joint_sets):
            # Record parameter provenance for this set
            prov = dict(joint_set.provenance) if hasattr(joint_set, 'provenance') else {}
            prov.setdefault("orientation", "user")
            prov.setdefault("size", "user")
            prov.setdefault("p32", "user")
            parameter_provenance[joint_set.set_id] = prov
            self._check_cancelled()
            set_seed = seed + joint_set.set_id * 10000
            set_rng = np.random.default_rng(set_seed)

            self._report_progress(
                idx, total_sets,
                f"Generating {joint_set.name}..."
            )

            # Estimate fracture count
            n_expected = joint_set.expected_fracture_count(model_volume)
            n_expected = min(n_expected, self.config.max_fractures_per_set)

            # Sample orientations
            normals = fisher_sample(
                joint_set.orientation.mean_dip_direction,
                joint_set.orientation.mean_dip,
                joint_set.orientation.kappa,
                rng=set_rng,
                n_samples=n_expected,
            )

            # Sample sizes
            radii = self._sample_radii(
                joint_set, n_expected, set_rng
            )

            # Sample positions uniformly
            centers = self._sample_positions(
                n_expected, set_rng
            )

            # Create fractures
            set_fractures = []
            set_total_area = 0.0

            from dfn_cave_studio.geometry.coordinate import normal_to_dip_dir_dip

            for i in range(n_expected):
                dd, dip = normal_to_dip_dir_dip(normals[i])
                area = math.pi * radii[i] ** 2

                geo = FractureGeometry(
                    geometry_type="disk",
                    center_x=float(centers[i, 0]),
                    center_y=float(centers[i, 1]),
                    center_z=float(centers[i, 2]),
                    radius=float(radii[i]),
                    normal_x=float(normals[i, 0]),
                    normal_y=float(normals[i, 1]),
                    normal_z=float(normals[i, 2]),
                    dip_direction=dd,
                    dip=dip,
                    area=area,
                )

                f = StochasticFracture(
                    geometry=geo,
                    set_id=joint_set.set_id,
                    set_name=joint_set.name,
                    realization_id=realization_number,
                    random_seed=set_seed,
                    radius=float(radii[i]),
                    source=FractureSource.STOCHASTIC,
                )
                set_fractures.append(f)
                set_total_area += area

            all_fractures.extend(set_fractures)

            # Per-set statistics
            set_statistics[joint_set.set_id] = {
                "set_name": joint_set.name,
                "fracture_count": len(set_fractures),
                "total_area": float(set_total_area),
                "achieved_p32": float(set_total_area / model_volume),
                "target_p32": joint_set.target_p32,
                "mean_radius": float(np.mean(radii)) if len(radii) > 0 else 0.0,
            }

        # Global statistics
        total_area = sum(s["total_area"] for s in set_statistics.values())
        total_fractures = len(all_fractures)
        achieved_p32 = total_area / model_volume
        target_p32 = sum(js.target_p32 for js in self.config.joint_sets)
        p32_error = abs(achieved_p32 - target_p32) / target_p32 * 100 if target_p32 > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        # Build generation result
        result = DFNGenerationResult(
            realization_id=realization_number,
            config=self.config,
            stochastic_fractures=all_fractures,
            set_statistics=set_statistics,
            total_fractures=total_fractures,
            total_fracture_area=total_area,
            achieved_p32=achieved_p32,
            target_p32=target_p32,
            p32_error_percent=p32_error,
            convergence_achieved=p32_error <= (self.config.joint_sets[0].p32_tolerance * 100 if self.config.joint_sets else 5.0),
            elapsed_seconds=elapsed,
            cancelled=self._cancelled,
            parameter_provenance=parameter_provenance,
        )

        # Build realization
        realization = DFNRealization(
            realization_number=realization_number,
            name=f"Realization {realization_number + 1}",
            stochastic_fractures=all_fractures,
            generation_config=self.config,
            generation_result=result,
            total_p32=achieved_p32,
        )

        return realization

    # ------------------------------------------------------------------
    # Sampling Helpers
    # ------------------------------------------------------------------

    def _sample_radii(
        self,
        joint_set: JointSetConfig,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample fracture radii from the configured distribution.

        Args:
            joint_set: Fracture set configuration with size distribution.
            n: Number of samples.
            rng: Seeded random generator.

        Returns:
            (n,) array of radii in meters.
        """
        sd = joint_set.size
        dist_type = sd.distribution_type

        if dist_type == SizeDistributionType.FIXED:
            # Fixed radius = midpoint of range
            radii = np.full(n, (sd.min_radius + sd.max_radius) / 2.0)

        elif dist_type == SizeDistributionType.LOGNORMAL:
            # lognormal: ln(radius) ~ N(mu, sigma)
            ln_radii = rng.normal(sd.lognormal_mu, sd.lognormal_sigma, n)
            radii = np.exp(ln_radii)

        elif dist_type == SizeDistributionType.POWER_LAW:
            # Truncated power-law using inverse CDF sampling.
            # Uses the conditional distribution P(R | r_min ≤ R ≤ r_max),
            # matching the analytic truncated moments in fracture_set.py.
            D = sd.power_law_exponent
            r_min, r_max = sd.min_radius, sd.max_radius
            c = r_max ** (-D) - r_min ** (-D)
            u = rng.random(n)
            radii = (-u * c + r_max ** (-D)) ** (-1.0 / D)

        elif dist_type == SizeDistributionType.TRUNCATED_POWER_LAW:
            # Same truncated inverse CDF as POWER_LAW (both use conditional distribution).
            D = sd.power_law_exponent
            r_min, r_max = sd.min_radius, sd.max_radius
            c = r_max ** (-D) - r_min ** (-D)
            u = rng.random(n)
            radii = (-u * c + r_max ** (-D)) ** (-1.0 / D)

        elif dist_type == SizeDistributionType.EXPONENTIAL:
            # Truncated exponential using inverse CDF.
            # F(r) = (1 - exp(-λr)) / (exp(-λ·r_min) - exp(-λ·r_max))
            # where λ = 2/(r_min + r_max).
            r_min, r_max = sd.min_radius, sd.max_radius
            lam = 2.0 / (r_min + r_max) if (r_min + r_max) > 0 else 1.0
            exp_min = math.exp(-lam * r_min)
            exp_max = math.exp(-lam * r_max)
            norm = exp_min - exp_max
            if norm < 1e-15:
                radii = np.full(n, (r_min + r_max) / 2.0)
            else:
                u = rng.random(n)
                radii = -np.log(exp_min - u * norm) / lam

        else:
            # Default fallback
            radii = rng.uniform(sd.min_radius, sd.max_radius, n)

        return radii

    def _sample_positions(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample fracture center positions uniformly within model bounds.

        Positions are sampled in an expanded bounding box to avoid edge effects:
        the box is expanded by boundary_buffer on all sides.

        Args:
            n: Number of points.
            rng: Seeded random generator.

        Returns:
            (n, 3) array of (x, y, z) coordinates in meters.
        """
        buffer = self.config.boundary_buffer
        return np.column_stack([
            rng.uniform(self.bounds.x_min - buffer, self.bounds.x_max + buffer, n),
            rng.uniform(self.bounds.y_min - buffer, self.bounds.y_max + buffer, n),
            rng.uniform(self.bounds.z_min - buffer, self.bounds.z_max + buffer, n),
        ])
