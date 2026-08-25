"""M7 joint set identification service.

Supports two modes:
  A — Use imported set_id from CSV
  B — Auto-identify via spherical clustering of fracture normals

Auto-identification uses unit normal vectors (not raw dip angles) to
correctly handle spherical geometry and the 0°/360° wrap.  Optional
seeded K-Means++ on the unit sphere produces reproducible results.

Validation boreholes are excluded from clustering but reported separately
for comparison.

DO NOT:
  - Use Euclidean distance on raw (dd, dip) pairs
  - Let validation data affect cluster centroids
  - Treat angular mean as arithmetic mean
"""

from __future__ import annotations

from typing import Optional, List, Dict, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.borehole import BoreholeCollection, FractureObservation
from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal, normal_to_dip_dir_dip


SET_COLORS = ["#1976d2", "#388e3c", "#f57c00", "#d32f2f", "#7b1fa2",
              "#0288d1", "#689f38", "#fbc02d", "#e64a19", "#5c6bc0"]

INSUFFICIENT_ORIENTATIONS_MESSAGE = (
    "Cannot identify K non-empty joint sets:\n"
    "insufficient valid or distinct orientation observations."
)


class JointSetIdentificationResult:
    """Result of joint set identification for one structural domain."""

    def __init__(self, domain_id: int):
        self.domain_id = domain_id
        self.sets: Dict[int, JointSetConfig] = {}  # set_id → config
        self.assignments: Dict[str, int] = {}  # record_id → set_id
        self.mode: str = "imported"  # imported | automatic
        self.calibration_count: int = 0
        self.validation_count: int = 0
        self.random_seed: Optional[int] = None
        self.n_clusters: int = 0
        self.full_orientation_count: int = 0
        self.dip_only_count: int = 0
        self.validation_full_orientation_count: int = 0
        self.validation_dip_only_count: int = 0
        self.set_counts: Dict[int, Dict[str, int]] = {}
        self.created_at: datetime = datetime.now(timezone.utc)


class JointSetService:
    """Identify joint sets from fracture observations."""

    def __init__(self, random_seed: int = 42):
        self._random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        self._results: Dict[int, JointSetIdentificationResult] = {}  # domain_id → result
        self._set_counter = 1  # global set ID counter

    # ── Mode A: Use imported set_id ────────────────────────────────────────

    def identify_from_imported(
        self, collection: BoreholeCollection,
        calibration_holes: Set[str],
        validation_holes: Set[str],
        domain_id: int = 0,
    ) -> JointSetIdentificationResult:
        """Use set_id values already present in the data.

        Args:
            collection: BoreholeCollection with fracture observations.
            calibration_holes: Set of calibration hole IDs.
            validation_holes: Set of validation hole IDs.
            domain_id: Structural domain ID for this analysis.

        Returns:
            JointSetIdentificationResult with per-set statistics.
        """
        result = JointSetIdentificationResult(domain_id)
        result.mode = "imported"

        # Collect statistics per set_id, split by calibration/validation
        cal_normals: Dict[int, List[np.ndarray]] = defaultdict(list)
        val_normals: Dict[int, List[np.ndarray]] = defaultdict(list)
        cal_dip_only: Dict[int, List[FractureObservation]] = defaultdict(list)
        val_dip_only: Dict[int, List[FractureObservation]] = defaultdict(list)

        for bh in collection:
            hole_role = "calibration" if bh.borehole_id in calibration_holes else \
                        "validation" if bh.borehole_id in validation_holes else "calibration"
            for obs_index, obs in enumerate(bh.fracture_observations):
                if obs.set_id is None:
                    continue
                if hole_role == "calibration":
                    result.assignments[f"{bh.borehole_id}:{obs_index}"] = obs.set_id
                    if obs.has_full_orientation:
                        cal_normals[obs.set_id].append(dip_dir_dip_to_normal(obs.dip_direction, obs.dip))
                    else:
                        cal_dip_only[obs.set_id].append(obs)
                else:
                    if obs.has_full_orientation:
                        val_normals[obs.set_id].append(dip_dir_dip_to_normal(obs.dip_direction, obs.dip))
                    else:
                        val_dip_only[obs.set_id].append(obs)

        # Build joint set configs from calibration data
        for set_id in sorted(set(cal_normals) | set(cal_dip_only)):
            normals = np.array(cal_normals[set_id])
            result.set_counts[set_id] = {
                "total": len(cal_normals[set_id]) + len(cal_dip_only[set_id]),
                "full_orientation": len(cal_normals[set_id]),
                "dip_only": len(cal_dip_only[set_id]),
            }
            if len(normals) < 3:
                continue
            dd, dip, kappa = self._fisher_from_normals(normals)
            js = JointSetConfig(
                set_id=set_id,
                name=f"Joint Set {set_id}",
                color=SET_COLORS[(set_id - 1) % len(SET_COLORS)],
                orientation=OrientationDistribution(
                    mean_dip_direction=round(dd, 1),
                    mean_dip=round(dip, 1),
                    kappa=round(kappa, 1),
                ),
                provenance={
                    "orientation": "imported",
                    "orientation_fit_eligible": True,
                    "full_orientation_count": len(normals),
                    "dip_only_count": len(cal_dip_only[set_id]),
                    "size": "user",
                    "p32": "user",
                },
            )
            result.sets[set_id] = js

        result.full_orientation_count = sum(len(v) for v in cal_normals.values())
        result.dip_only_count = sum(len(v) for v in cal_dip_only.values())
        result.validation_full_orientation_count = sum(len(v) for v in val_normals.values())
        result.validation_dip_only_count = sum(len(v) for v in val_dip_only.values())
        result.calibration_count = result.full_orientation_count + result.dip_only_count
        result.validation_count = result.validation_full_orientation_count + result.validation_dip_only_count
        self._results[domain_id] = result
        return result

    # ── Mode B: Auto-identify via spherical clustering ─────────────────────

    def identify_auto(
        self, collection: BoreholeCollection,
        calibration_holes: Set[str],
        validation_holes: Set[str],
        n_clusters: int = 3,
        domain_id: int = 0,
        random_seed: Optional[int] = None,
    ) -> JointSetIdentificationResult:
        """Auto-identify joint sets via spherical K-Means on unit normals.

        Uses unit normal vectors on the unit sphere to avoid:
          - Euclidean distance on (dd, dip) pairs being wrong near 0°/360°
          - Treating angular separations as linear distances

        The algorithm is a simple iterative spherical K-Means:
          1. Initialize K centroids via K-Means++ on sphere
          2. Assign each normal to nearest centroid (cosine distance)
          3. Recompute centroids as mean of assigned normals
          4. Repeat until convergence or max iterations

        Args:
            collection: BoreholeCollection.
            calibration_holes: Calibration hole IDs.
            validation_holes: Validation hole IDs (excluded from fit).
            n_clusters: Number of joint sets to identify.
            domain_id: Domain ID.
            random_seed: Fixed seed for reproducibility.

        Returns:
            JointSetIdentificationResult.
        """
        seed = random_seed if random_seed is not None else self._random_seed
        rng = np.random.default_rng(seed)

        result = JointSetIdentificationResult(domain_id)
        result.mode = "automatic"
        result.random_seed = seed
        result.n_clusters = n_clusters

        # ── Collect calibration normals ────────────────────────────────────
        cal_normals = []
        cal_obs_refs = []  # (borehole_id, obs_index)
        for bh in collection:
            if bh.borehole_id not in calibration_holes:
                continue
            for i, obs in enumerate(bh.fracture_observations):
                if not obs.has_full_orientation:
                    result.dip_only_count += 1
                    continue
                n = dip_dir_dip_to_normal(obs.dip_direction, obs.dip)
                cal_normals.append(n)
                cal_obs_refs.append((bh.borehole_id, i))
        cal_normals = np.array(cal_normals)
        n_cal = len(cal_normals)

        distinct_indices = self._distinct_axial_indices(cal_normals)
        if n_cal < n_clusters or len(distinct_indices) < n_clusters:
            # A smaller result would violate the user's explicit K request.
            raise ValueError(INSUFFICIENT_ORIENTATIONS_MESSAGE)

        # ── Spherical K-Means++ ────────────────────────────────────────────
        # K-Means++ initialization on sphere
        centroids = np.zeros((n_clusters, 3))
        chosen_indices = []
        # First centroid: random
        first_index = int(rng.integers(0, n_cal))
        chosen_indices.append(first_index)
        centroids[0] = cal_normals[first_index]
        for k in range(1, n_clusters):
            # Compute distances to nearest existing centroid
            dists = np.min([1.0 - np.abs(cal_normals @ centroids[j])
                           for j in range(k)], axis=0)
            dists = np.maximum(dists, 1e-10)
            probs = dists / dists.sum()
            chosen_index = int(rng.choice(n_cal, p=probs))
            if any(
                np.isclose(abs(float(np.dot(cal_normals[chosen_index], cal_normals[index]))), 1.0, atol=1e-12)
                for index in chosen_indices
            ):
                for index in chosen_indices:
                    same_axis = np.isclose(np.abs(cal_normals @ cal_normals[index]), 1.0, atol=1e-12)
                    probs[same_axis] = 0.0
                if float(probs.sum()) <= 0.0:
                    raise ValueError(INSUFFICIENT_ORIENTATIONS_MESSAGE)
                probs /= probs.sum()
                chosen_index = int(rng.choice(n_cal, p=probs))
            chosen_indices.append(chosen_index)
            centroids[k] = cal_normals[chosen_index]

        # Iterative refinement
        max_iters = 30
        for _ in range(max_iters):
            # Assign to nearest centroid (cosine similarity)
            labels = self._assign_nonempty(cal_normals, centroids)

            # Recompute centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                mean_vec = cal_normals[mask].mean(axis=0)
                new_centroids[k] = mean_vec / np.linalg.norm(mean_vec)

            converged = bool(np.all(1.0 - np.abs(np.sum(centroids * new_centroids, axis=1)) <= 1e-6))
            centroids = new_centroids
            if converged:
                break

        # Final labels
        final_labels = self._assign_nonempty(cal_normals, centroids)

        # ── Build joint set configs ────────────────────────────────────────
        for k in range(n_clusters):
            mask = final_labels == k
            cluster_normals = cal_normals[mask]
            dd, dip, kappa = self._fisher_from_normals(cluster_normals)
            set_id = k + 1
            js = JointSetConfig(
                set_id=set_id,
                name=f"Auto Set {set_id}",
                color=SET_COLORS[k % len(SET_COLORS)],
                orientation=OrientationDistribution(
                    mean_dip_direction=round(dd % 360.0, 1) % 360.0,
                    mean_dip=round(dip, 1),
                    kappa=round(kappa, 1),
                ),
                provenance={"orientation": "automatic", "size": "user", "p32": "user"},
            )
            result.sets[set_id] = js

            # Assign labels back to observations
            for idx in np.where(mask)[0]:
                bh_id, obs_idx = cal_obs_refs[idx]
                record_id = f"{bh_id}:{obs_idx}"
                result.assignments[record_id] = set_id

        result.calibration_count = n_cal
        result.full_orientation_count = n_cal
        # Count validation observations
        for bh in collection:
            if bh.borehole_id in validation_holes:
                result.validation_count += len(bh.fracture_observations)
                result.validation_full_orientation_count += sum(obs.has_full_orientation for obs in bh.fracture_observations)
                result.validation_dip_only_count += sum(not obs.has_full_orientation for obs in bh.fracture_observations)

        # ── Deterministic reordering ─────────────────────────────────────
        # Assign stable set_ids by sorting clusters by mean dip_direction
        self._stabilize_set_ids(result)

        self._results[domain_id] = result
        return result

    @staticmethod
    def _distinct_axial_indices(normals: np.ndarray) -> List[int]:
        """Return first occurrences of distinct plane-normal axes."""
        distinct: List[int] = []
        for index, normal in enumerate(normals):
            is_new_axis = all(
                not np.isclose(abs(float(np.dot(normal, normals[other]))), 1.0, atol=1e-12)
                for other in distinct
            )
            if is_new_axis:
                distinct.append(index)
        return distinct

    @staticmethod
    def _assign_nonempty(normals: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign every observation while deterministically repairing empty clusters."""
        similarities = np.abs(normals @ centroids.T)
        labels = np.argmax(similarities, axis=1)
        counts = np.bincount(labels, minlength=len(centroids))
        errors = 1.0 - similarities[np.arange(len(normals)), labels]
        claimed: Set[int] = set()

        for empty_cluster in np.flatnonzero(counts == 0):
            candidates = [
                index
                for index, label in enumerate(labels)
                if counts[label] > 1 and index not in claimed
            ]
            if not candidates:
                raise ValueError(INSUFFICIENT_ORIENTATIONS_MESSAGE)
            selected = min(candidates, key=lambda index: (-float(errors[index]), index))
            donor = int(labels[selected])
            labels[selected] = empty_cluster
            counts[donor] -= 1
            counts[empty_cluster] += 1
            claimed.add(selected)

        return labels

    def _stabilize_set_ids(self, result: JointSetIdentificationResult) -> None:
        """Reassign set_ids deterministically by sorting on mean dip_direction.

        This ensures that two runs with the same seed produce identical
        (set_id, assignment) pairs even if K-Means internal cluster indices
        differ.
        """
        if len(result.sets) <= 1:
            return
        # Build ordering: sort by mean_dip_direction
        ordered = sorted(
            result.sets.items(),
            key=lambda item: (item[1].orientation.mean_dip_direction,
                              item[1].orientation.mean_dip)
        )
        # Build old_id → new_id map
        remap = {}
        for new_id, (old_id, _) in enumerate(ordered, start=1):
            remap[old_id] = new_id
        # Rebuild sets dict with new IDs
        new_sets = {}
        for old_id, js in result.sets.items():
            new_id = remap[old_id]
            js.set_id = new_id
            js.name = f"Auto Set {new_id}"
            new_sets[new_id] = js
        result.sets = new_sets
        # Remap assignments
        new_assignments = {}
        for rec_id, old_id in result.assignments.items():
            new_assignments[rec_id] = remap.get(old_id, old_id)
        result.assignments = new_assignments

    # ── Query ─────────────────────────────────────────────────────────────

    def get_result(self, domain_id: int = 0) -> Optional[JointSetIdentificationResult]:
        return self._results.get(domain_id)

    def get_all_joint_sets(self) -> List[JointSetConfig]:
        """Get all identified joint sets across all domains."""
        all_sets: List[JointSetConfig] = []
        for result in self._results.values():
            all_sets.extend(result.sets.values())
        return all_sets

    def assign_manual_set(self, borehole_id: str, obs_index: int, set_id: int) -> None:
        """Manually reassign a single observation to a different set."""
        # Hook for UI manual adjustment
        pass

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _fisher_from_normals(normals: np.ndarray) -> Tuple[float, float, float]:
        """Compute Fisher mean direction and kappa from unit normals.

        Returns (dip_direction, dip, kappa).
        """
        n = len(normals)
        values = normals.copy()
        if n <= 2:
            values[values @ values[0] < 0.0] *= -1.0
        R_vec = np.sum(values, axis=0)
        R = float(np.linalg.norm(R_vec))
        if R < 1e-12:
            return 0.0, 0.0, 1.0
        mean_normal = R_vec / R
        dd, dip = normal_to_dip_dir_dip(mean_normal)
        if n == 1:
            kappa = 999.0
        elif n == 2:
            r_bar = min(R / n, 1.0 - 1e-12)
            if r_bar < 0.53:
                kappa = 2 * r_bar + r_bar**3 + 5 * r_bar**5 / 6
            elif r_bar < 0.85:
                kappa = -0.4 + 1.39 * r_bar + 0.43 / (1 - r_bar)
            else:
                kappa = 1 / max(r_bar**3 - 4 * r_bar**2 + 3 * r_bar, 1e-10)
        elif n >= 16:
            kappa = (n - 1) / max(n - R, 1e-10)
        else:
            kappa = (n - 2) / max(n - R, 1e-10) * (n / (n - 1))
        kappa = max(0.1, min(kappa, 999.0))
        return dd, dip, kappa
