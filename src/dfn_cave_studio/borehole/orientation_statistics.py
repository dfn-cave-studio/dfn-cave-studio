"""
Orientation statistics for fracture observations in boreholes.

Computes Fisher distribution parameters (mean dip direction, mean dip,
concentration kappa) from sets of FractureObservations, enabling the
"Import from Observations" workflow in the Joint Set Manager.

References:
  - Fisher, R.A. (1953). Dispersion on a sphere.
  - Woodcock, N.H. (1977). Specification of fabric shapes.
  - Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.models.borehole import BoreholeCollection, FractureObservation
from dfn_cave_studio.models.fracture_set import OrientationDistribution


@dataclass(frozen=True)
class OrientationSetStatistics:
    """Full-orientation Fisher eligibility plus dip-only-safe summary statistics."""

    set_id: int
    full_orientation_count: int
    dip_only_count: int
    total_count: int
    mean_dip: float
    median_dip: float
    dip_standard_deviation: float
    orientation_fit_eligible: bool
    exclusion_reason: str | None
    fisher_orientation: OrientationDistribution | None


class OrientationStatisticsCalculator:
    """Compute Fisher orientation statistics from fracture observations.

    Groups observations by set_id, converts dip_direction/dip to unit
    normal vectors, computes the resultant vector, and derives Fisher
    mean direction and concentration parameter kappa.

    Usage:
        calc = OrientationStatisticsCalculator()
        stats = calc.compute_by_set(collection)
        for set_id, orient in stats.items():
            joint_set.orientation = orient
            joint_set.provenance["orientation"] = "borehole"
    """

    MIN_OBSERVATIONS = 3  # Minimum observations needed for reliable Fisher stats

    # ── Public API ────────────────────────────────────────────────────────

    def compute_by_set(
        self, collection: BoreholeCollection
    ) -> Dict[int, OrientationDistribution]:
        """Compute Fisher orientation statistics for each set_id.

        Observations without a set_id are skipped.

        Args:
            collection: BoreholeCollection with fracture observations.

        Returns:
            Dict mapping set_id → OrientationDistribution.
            Empty dict if no observations have set_id assigned.
        """
        # Gather all observations with set_id
        by_set: Dict[int, List[FractureObservation]] = {}
        for bh in collection:
            for obs in bh.fracture_observations:
                if obs.set_id is not None:
                    by_set.setdefault(obs.set_id, []).append(obs)

        result: Dict[int, OrientationDistribution] = {}
        for set_id, observations in by_set.items():
            eligible = [observation for observation in observations if observation.has_full_orientation]
            if len(eligible) >= self.MIN_OBSERVATIONS:
                orient = self.compute_fisher_stats(eligible)
                result[set_id] = orient

        return result

    def compute_fisher_stats(
        self, observations: List[FractureObservation]
    ) -> OrientationDistribution:
        """Compute Fisher distribution parameters from a list of observations.

        Converts each observation's dip_direction/dip to a unit normal
        vector, computes the resultant vector R, and derives:
          - mean_dip_direction (°)
          - mean_dip (°)
          - kappa (Fisher concentration)

        Args:
            observations: List of FractureObservations (must have >= 3).

        Returns:
            OrientationDistribution with computed parameters.

        Raises:
            ValueError: If fewer than MIN_OBSERVATIONS provided.
        """
        n = len(observations)
        if n < self.MIN_OBSERVATIONS:
            raise ValueError(
                f"Need at least {self.MIN_OBSERVATIONS} observations, got {n}"
            )
        if any(not observation.has_full_orientation for observation in observations):
            raise ValueError("Fisher statistics require FULL_ORIENTATION records; dip-only records are not eligible")

        # Convert dip_direction/dip to unit normal vectors using the
        # canonical coordinate module (ensures upper-hemisphere convention).
        from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal

        normals = np.array([
            dip_dir_dip_to_normal(o.dip_direction, o.dip)
            for o in observations
        ], dtype=np.float64)

        # Resultant vector
        R_vec = np.sum(normals, axis=0)
        R = float(np.linalg.norm(R_vec))

        if R < 1e-12:
            # All vectors cancel — use first observation as mean
            o = observations[0]
            return OrientationDistribution(
                mean_dip_direction=o.dip_direction,
                mean_dip=o.dip,
                kappa=1.0,  # Nearly uniform
            )

        # Mean direction (unit vector)
        mean_normal = R_vec / R

        # Convert mean normal back to dip_direction/dip
        from dfn_cave_studio.geometry.coordinate import normal_to_dip_dir_dip
        mean_dd, mean_dip = normal_to_dip_dir_dip(mean_normal)

        # Fisher kappa estimate.
        # For n >= 16: kappa ≈ (n-1)/(n-R)  (approximate MLE)
        # For n < 16:  kappa ≈ (n-2)/(n-R) * n/(n-1)  (small-sample correction)
        if n >= 16:
            kappa = (n - 1) / max(n - R, 1e-10)
        else:
            kappa = (n - 2) / max(n - R, 1e-10) * (n / (n - 1))

        # Clamp kappa to reasonable range
        kappa = max(0.1, min(kappa, 999.0))

        return OrientationDistribution(
            mean_dip_direction=round(mean_dd, 1),
            mean_dip=round(mean_dip, 1),
            kappa=round(kappa, 1),
        )

    def compute_detailed_by_set(self, collection: BoreholeCollection) -> Dict[int, OrientationSetStatistics]:
        """Return per-set dip summaries and explicit Fisher eligibility."""
        grouped: Dict[int, List[FractureObservation]] = {}
        for borehole in collection:
            for observation in borehole.fracture_observations:
                if observation.set_id is not None:
                    grouped.setdefault(observation.set_id, []).append(observation)
        output: Dict[int, OrientationSetStatistics] = {}
        for set_id, observations in grouped.items():
            full = [observation for observation in observations if observation.has_full_orientation]
            dips = np.asarray([observation.dip for observation in observations], dtype=float)
            eligible = len(full) >= self.MIN_OBSERVATIONS
            output[set_id] = OrientationSetStatistics(
                set_id=set_id,
                full_orientation_count=len(full),
                dip_only_count=len(observations) - len(full),
                total_count=len(observations),
                mean_dip=float(np.mean(dips)),
                median_dip=float(np.median(dips)),
                dip_standard_deviation=float(np.std(dips)),
                orientation_fit_eligible=eligible,
                exclusion_reason=None if eligible else "INSUFFICIENT_ORIENTATION_DATA",
                fisher_orientation=self.compute_fisher_stats(full) if eligible else None,
            )
        return output

    def locate_all_observations_3d(
        self, collection: BoreholeCollection
    ) -> Dict[str, List[NDArray[np.float64]]]:
        """Compute 3D positions for all fracture observations.

        Args:
            collection: BoreholeCollection with trajectories and observations.

        Returns:
            Dict mapping borehole_id → list of (3,) position arrays.
        """
        positions: Dict[str, List[NDArray[np.float64]]] = {}
        for bh in collection:
            bh_positions = []
            for obs in bh.fracture_observations:
                pos = bh.locate_observation(obs)
                if pos is not None:
                    bh_positions.append(pos)
            if bh_positions:
                positions[bh.borehole_id] = bh_positions
        return positions

    # ── Coordinate Conversion Helpers ─────────────────────────────────────
    # Delegates to dfn_cave_studio.geometry.coordinate for all dip/direction
    # ↔ normal conversions.  No independent transform code is maintained here.
