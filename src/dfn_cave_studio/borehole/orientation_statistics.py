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

        # Convert dip_direction/dip to unit normal vectors.
        # Dip direction (0-360° clockwise from North), dip (0-90° from horizontal).
        # Normal vector points DOWNWARD (positive Z is up, so Z component is negative).
        normals = np.array([
            self._dip_to_normal(o.dip_direction, o.dip)
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
        mean_dd, mean_dip = self._normal_to_dip(mean_normal)

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

    @staticmethod
    def _dip_to_normal(dip_direction: float, dip: float) -> NDArray[np.float64]:
        """Convert dip_direction (°) and dip (°) to a unit normal vector.

        Convention:
          - dip_direction = 0° → North (Y+), 90° → East (X+)
          - dip = 0° → horizontal, 90° → vertical (straight down)
          - Normal vector points INTO the rock mass (roughly downward for
            sub-horizontal fractures).

        For a fracture with dip_direction dd and dip angle δ:
          Strike direction = dd - 90° (right-hand rule)
          Normal (pointing down) has:
            nx = -sin(dd) * sin(δ)
            ny = -cos(dd) * sin(δ)
            nz = -cos(δ)
        """
        dd_rad = math.radians(dip_direction)
        dip_rad = math.radians(dip)
        sin_dd = math.sin(dd_rad)
        cos_dd = math.cos(dd_rad)
        sin_dip = math.sin(dip_rad)
        cos_dip = math.cos(dip_rad)

        return np.array([
            -sin_dd * sin_dip,   # X (easting) component
            -cos_dd * sin_dip,   # Y (northing) component
            -cos_dip,             # Z (elevation) component — downward
        ], dtype=np.float64)

    @staticmethod
    def _normal_to_dip(normal: NDArray[np.float64]) -> tuple:
        """Convert a unit normal vector back to (dip_direction, dip) in degrees.

        The normal may point either up or down. We always return the
        downward-pointing hemisphere direction (dip 0-90°).
        """
        nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])

        # Ensure downward-pointing hemisphere
        if nz > 0:
            nx, ny, nz = -nx, -ny, -nz

        # Dip: angle from horizontal plane
        norm_xy = math.sqrt(nx**2 + ny**2)
        dip = math.degrees(math.atan2(abs(nz), norm_xy))

        # Dip direction: direction of steepest descent
        if norm_xy < 1e-10:
            # Vertical — dip direction undefined, default to 0
            dip_direction = 0.0
        else:
            # The dip direction is the azimuth of the projection of the normal
            # onto the horizontal plane. For a downward-pointing normal,
            # dip_direction = atan2(-nx, -ny) converted to [0, 360).
            dip_direction = math.degrees(math.atan2(-nx, -ny)) % 360.0

        return dip_direction, dip
