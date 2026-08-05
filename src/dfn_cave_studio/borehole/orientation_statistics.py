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

import math
from typing import List, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.models.borehole import BoreholeCollection, FractureObservation
from dfn_cave_studio.models.fracture_set import OrientationDistribution


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
            if len(observations) >= self.MIN_OBSERVATIONS:
                orient = self.compute_fisher_stats(observations)
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
