"""Replaceable spatial interpolation and chunked M9 parameter-field builder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from dfn_cave_studio.dfn.intensity import expected_orientation_exposure
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.m9 import (
    DensityMethod,
    DensitySettings,
    DomainOrientationModel,
    P10Interval,
    P32Estimate,
    ParameterFieldMetadata,
    SizeModel,
    VariogramDiagnostics,
)
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.voxel.ordinary_kriging import OrdinaryKrigingInterpolator, fit_variogram
from dfn_cave_studio.voxel.resource_estimate import (
    FieldResourceEstimate,
    estimate_parameter_field_resources,
)


CELL_STATE_CODES = {
    VoxelCellState.OUTSIDE_MODEL: 0,
    VoxelCellState.NO_DATA: 1,
    VoxelCellState.TRUE_ZERO: 2,
    VoxelCellState.MODELED_VALUE: 3,
    VoxelCellState.EXCAVATION: 4,
}
SIZE_TYPE_CODES = {
    "fixed": 0,
    "uniform": 1,
    "truncated_lognormal": 2,
    "truncated_power_law": 3,
    "truncated_exponential": 4,
}
SIZE_SOURCE_CODES = {"fitted": 0, "size_proxy": 1, "assumed": 2, "user_defined": 3, "experimental": 4}
DENSITY_METHOD_CODES = {DensityMethod.GLOBAL_CONSTANT: 0, DensityMethod.IDW: 1, DensityMethod.ORDINARY_KRIGING: 2}
SIZE_PARAMETER_ORDER = {
    "fixed": ["radius"],
    "uniform": [],
    "truncated_lognormal": ["mu", "sigma"],
    "truncated_power_law": ["exponent"],
    "truncated_exponential": ["rate"],
}


def _chunk_points(
    start: int,
    stop: int,
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Create flat indices and XYZ cell centres for one bounded chunk."""
    flat = np.arange(start, stop, dtype=np.int64)
    grid = np.column_stack(np.unravel_index(flat, shape))
    points = np.empty((len(flat), 3), dtype=np.float64)
    for axis in range(3):
        points[:, axis] = origin[axis] + (grid[:, axis] + 0.5) * spacing[axis]
    return flat, points


def _callback_many(callback: Callable | None, points: np.ndarray, default, dtype) -> np.ndarray:
    """Use a callback's batch protocol, retaining point-callback compatibility."""
    if callback is None:
        return np.full(len(points), default, dtype=dtype)
    batch = getattr(callback, "predict_many", None)
    if batch is not None:
        values = np.asarray(batch(points), dtype=dtype)
        if values.shape != (len(points),):
            raise ValueError("batch callback must return one value per point")
        return values
    return np.asarray([callback(tuple(point)) for point in points], dtype=dtype)


def _domain_callback_many(callback: Callable | None, points: np.ndarray) -> np.ndarray:
    """Preserve numeric batch labels while allowing ``None`` from legacy callbacks."""
    if callback is None:
        return np.full(len(points), None, dtype=object)
    batch = getattr(callback, "predict_many", None)
    if batch is not None:
        values = np.asarray(batch(points))
        if values.shape != (len(points),):
            raise ValueError("batch callback must return one value per point")
        return values
    return np.asarray([callback(tuple(point)) for point in points], dtype=object)


def _predict_many_compatible(
    interpolator: SpatialInterpolator, points: np.ndarray, domain_id: int | None
) -> BatchInterpolationResult:
    """Use the batch protocol or adapt a legacy/custom single-point interpolator."""
    batch = getattr(interpolator, "predict_many", None)
    if batch is not None:
        return batch(points, domain_id)
    rows = [interpolator.predict(tuple(point), domain_id) for point in points]
    return BatchInterpolationResult(
        np.asarray([np.nan if row.value is None else row.value for row in rows], dtype=np.float64),
        np.asarray([row.nearest_distance for row in rows], dtype=np.float64),
        np.asarray([row.neighbor_count for row in rows], dtype=np.int16),
        np.asarray([row.confidence for row in rows], dtype=np.float64),
        np.asarray([row.kriging_variance for row in rows], dtype=np.float64),
    )


@dataclass(frozen=True)
class SpatialSample:
    """One calibration-derived spatial observation."""

    x: float
    y: float
    z: float
    value: float
    domain_id: int | None
    effective_length: float = 0.0
    observation_count: int = 0


@dataclass(frozen=True)
class InterpolationResult:
    """A value and its spatial-support diagnostics."""

    value: float | None
    state: VoxelCellState
    nearest_distance: float = float("nan")
    neighbor_count: int = 0
    confidence: float = 0.0
    provenance: str = "no_data"
    kriging_variance: float = float("nan")
    used_regularization: bool = False
    used_pseudoinverse: bool = False


@dataclass(frozen=True)
class BatchInterpolationResult:
    """Bounded NumPy buffers returned by a deterministic batch prediction."""

    values: np.ndarray
    nearest_distances: np.ndarray
    neighbor_counts: np.ndarray
    confidences: np.ndarray
    variances: np.ndarray


class SpatialInterpolator(ABC):
    """Interface reserved for IDW, kriging, GP, and future methods."""

    @abstractmethod
    def predict(self, point: tuple[float, float, float], domain_id: int | None) -> InterpolationResult:
        """Predict at one point without crossing structural domains."""


class IDWInterpolator(SpatialInterpolator):
    """Three-dimensional inverse-distance weighting with domain isolation."""

    def __init__(
        self,
        samples: Iterable[SpatialSample],
        *,
        power: float = 2.0,
        search_radius: float | None = None,
        min_neighbors: int = 1,
        max_neighbors: int = 12,
        anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
        fallback_by_domain: dict[int | None, float] | None = None,
    ) -> None:
        if power <= 0 or min_neighbors < 1 or max_neighbors < min_neighbors or any(item <= 0 for item in anisotropy):
            raise ValueError("invalid IDW settings")
        self.samples = list(samples)
        self.power = power
        self.search_radius = search_radius
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.anisotropy = np.asarray(anisotropy, dtype=float)
        self.fallback_by_domain = fallback_by_domain or {}
        grouped: dict[int | None, list[SpatialSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.domain_id, []).append(sample)
        self._domain_samples = grouped
        self._domain_coordinates = {
            domain_id: np.asarray([(sample.x, sample.y, sample.z) for sample in rows], dtype=float)
            for domain_id, rows in grouped.items()
        }
        self._domain_values = {
            domain_id: np.asarray([sample.value for sample in rows], dtype=float)
            for domain_id, rows in grouped.items()
        }
        self._domain_trees = {
            domain_id: cKDTree(coordinates / self.anisotropy)
            for domain_id, coordinates in self._domain_coordinates.items()
        }

    def predict(self, point: tuple[float, float, float], domain_id: int | None) -> InterpolationResult:
        eligible = self._domain_samples.get(domain_id, [])
        if not eligible:
            return self._fallback(domain_id)
        coordinates = self._domain_coordinates[domain_id]
        distances = np.linalg.norm((coordinates - np.asarray(point, dtype=float)) / self.anisotropy, axis=1)
        exact = np.flatnonzero(distances <= 1e-12)
        if exact.size:
            sample = eligible[int(exact[0])]
            return InterpolationResult(
                sample.value,
                VoxelCellState.TRUE_ZERO if sample.value == 0 else VoxelCellState.MODELED_VALUE,
                0.0,
                1,
                1.0,
                "exact_observation",
            )
        indices = np.lexsort((np.arange(len(distances), dtype=np.int64), distances))
        if self.search_radius is not None:
            indices = indices[distances[indices] <= self.search_radius]
        indices = indices[: self.max_neighbors]
        if len(indices) < self.min_neighbors:
            return self._fallback(domain_id, float(distances.min()))
        weights = distances[indices] ** (-self.power)
        value = float(np.dot(weights, [eligible[int(index)].value for index in indices]) / weights.sum())
        nearest = float(distances[indices[0]])
        confidence = float(min(1.0, len(indices) / self.max_neighbors) / (1.0 + nearest))
        return InterpolationResult(
            value,
            VoxelCellState.TRUE_ZERO if value == 0 else VoxelCellState.MODELED_VALUE,
            nearest,
            len(indices),
            confidence,
            "idw",
        )

    def predict_many(self, points: np.ndarray, domain_id: int | None) -> BatchInterpolationResult:
        """Predict a bounded point batch while reusing domain sample arrays."""
        targets = np.asarray(points, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 3 or np.any(~np.isfinite(targets)):
            raise ValueError("Prediction points must be a finite (N,3) array")
        count = len(targets)
        values = np.full(count, np.nan, dtype=np.float64)
        nearest = np.full(count, np.nan, dtype=np.float64)
        neighbours = np.zeros(count, dtype=np.int16)
        confidence = np.zeros(count, dtype=np.float64)
        variances = np.full(count, np.nan, dtype=np.float64)
        coordinates = self._domain_coordinates.get(domain_id)
        if coordinates is None:
            if domain_id in self.fallback_by_domain:
                values.fill(self.fallback_by_domain[domain_id])
                confidence.fill(0.1)
            return BatchInterpolationResult(values, nearest, neighbours, confidence, variances)
        sample_values = self._domain_values[domain_id]
        selected_count = min(self.max_neighbors, len(coordinates))
        query_count = min(self.max_neighbors + 1, len(coordinates))
        query_distances, candidates = self._domain_trees[domain_id].query(
            targets / self.anisotropy, k=query_count
        )
        query_distances = np.asarray(query_distances, dtype=np.float64).reshape(count, -1)
        candidates = np.asarray(candidates, dtype=np.int64).reshape(count, -1)
        scaled_targets = targets / self.anisotropy
        if query_count > selected_count:
            boundary_distances = query_distances[:, selected_count - 1]
            next_distances = query_distances[:, selected_count]
            boundary_ties = np.abs(boundary_distances - next_distances) <= 1e-12 * np.maximum.reduce(
                (np.ones(count, dtype=np.float64), np.abs(boundary_distances), np.abs(next_distances))
            )
        else:
            boundary_distances = query_distances[:, selected_count - 1]
            boundary_ties = np.zeros(count, dtype=np.bool_)

        # cKDTree does not guarantee original-index ordering for equal distances.
        # Recompute the bounded candidate distances once, then handle the common
        # non-exact/non-boundary-tie rows as dense arrays.  Only exceptional rows
        # need the stable per-row path (and potentially query_ball_point).
        candidate_coordinates = coordinates[candidates]
        exact_distances = np.linalg.norm(
            (candidate_coordinates - targets[:, None, :]) / self.anisotropy,
            axis=2,
        )
        exact_rows = np.any(exact_distances <= 1e-12, axis=1)
        slow_rows = boundary_ties | exact_rows
        fast_rows = np.flatnonzero(~slow_rows)
        if fast_rows.size:
            fast_candidates = candidates[fast_rows]
            fast_distances = exact_distances[fast_rows]
            order = np.lexsort((fast_candidates, fast_distances), axis=1)
            fast_candidates = np.take_along_axis(fast_candidates, order, axis=1)[:, :selected_count]
            fast_distances = np.take_along_axis(fast_distances, order, axis=1)[:, :selected_count]
            nearest_available = fast_distances[:, 0]
            if self.search_radius is None:
                usable = np.ones(fast_distances.shape, dtype=np.bool_)
            else:
                usable = fast_distances <= self.search_radius
            usable_counts = np.sum(usable, axis=1, dtype=np.int64)
            successful = usable_counts >= self.min_neighbors
            failed_rows = fast_rows[~successful]
            if failed_rows.size:
                nearest[failed_rows] = nearest_available[~successful]
                if domain_id in self.fallback_by_domain:
                    values[failed_rows] = self.fallback_by_domain[domain_id]
                    confidence[failed_rows] = 0.1
            successful_rows = fast_rows[successful]
            if successful_rows.size:
                successful_distances = fast_distances[successful]
                successful_candidates = fast_candidates[successful]
                successful_usable = usable[successful]
                weights = np.zeros(successful_distances.shape, dtype=np.float64)
                np.power(
                    successful_distances,
                    -self.power,
                    out=weights,
                    where=successful_usable,
                )
                weighted_values = weights * sample_values[successful_candidates]
                values[successful_rows] = np.sum(weighted_values, axis=1) / np.sum(weights, axis=1)
                nearest[successful_rows] = successful_distances[:, 0]
                neighbours[successful_rows] = usable_counts[successful].astype(np.int16)
                confidence[successful_rows] = (
                    np.minimum(1.0, usable_counts[successful] / self.max_neighbors)
                    / (1.0 + successful_distances[:, 0])
                )

        for row in np.flatnonzero(slow_rows):
            initial_candidates = candidates[row]
            if boundary_ties[row]:
                boundary = float(boundary_distances[row])
                radius = max(boundary, 1e-12) + max(1.0, boundary) * 1e-12
                candidate_indices = np.asarray(
                    self._domain_trees[domain_id].query_ball_point(scaled_targets[row], radius),
                    dtype=np.int64,
                )
            else:
                candidate_indices = initial_candidates[:selected_count]
            row_distances = np.linalg.norm(
                (coordinates[candidate_indices] - targets[row]) / self.anisotropy, axis=1
            )
            exact_indices = candidate_indices[row_distances <= 1e-12]
            if exact_indices.size:
                first_exact = int(np.min(exact_indices))
                values[row] = sample_values[first_exact]
                nearest[row] = 0.0
                neighbours[row] = 1
                confidence[row] = 1.0
                continue
            order = np.lexsort((candidate_indices, row_distances))
            candidate_indices = candidate_indices[order]
            row_distances = row_distances[order]
            nearest_available = float(row_distances[0])
            if self.search_radius is not None:
                within_radius = row_distances <= self.search_radius
                candidate_indices = candidate_indices[within_radius]
                row_distances = row_distances[within_radius]
            candidate_indices = candidate_indices[: self.max_neighbors]
            row_distances = row_distances[: self.max_neighbors]
            if len(candidate_indices) < self.min_neighbors:
                nearest[row] = nearest_available
                if domain_id in self.fallback_by_domain:
                    values[row] = self.fallback_by_domain[domain_id]
                    confidence[row] = 0.1
                continue
            weights = row_distances ** (-self.power)
            values[row] = float(np.dot(weights, sample_values[candidate_indices]) / weights.sum())
            nearest[row] = float(row_distances[0])
            neighbours[row] = len(candidate_indices)
            confidence[row] = float(
                min(1.0, len(candidate_indices) / self.max_neighbors) / (1.0 + nearest[row])
            )
        return BatchInterpolationResult(values, nearest, neighbours, confidence, variances)

    def _fallback(self, domain_id: int | None, distance: float = float("nan")) -> InterpolationResult:
        if domain_id in self.fallback_by_domain:
            value = self.fallback_by_domain[domain_id]
            return InterpolationResult(
                value,
                VoxelCellState.TRUE_ZERO if value == 0 else VoxelCellState.MODELED_VALUE,
                distance,
                0,
                0.1,
                "global_domain_fallback",
            )
        return InterpolationResult(None, VoxelCellState.NO_DATA, distance)


class GlobalConstantInterpolator(SpatialInterpolator):
    """Domain-isolated constant P32 estimator."""

    def __init__(self, values: dict[int | None, float]) -> None:
        self.values = values

    def predict(self, point: tuple[float, float, float], domain_id: int | None) -> InterpolationResult:
        del point
        if domain_id not in self.values:
            return InterpolationResult(None, VoxelCellState.NO_DATA)
        value = self.values[domain_id]
        return InterpolationResult(
            value,
            VoxelCellState.TRUE_ZERO if value == 0 else VoxelCellState.MODELED_VALUE,
            provenance="global_constant",
            confidence=1.0,
        )

    def predict_many(self, points: np.ndarray, domain_id: int | None) -> BatchInterpolationResult:
        """Return constant domain values without allocating Python result objects."""
        count = len(points)
        values = np.full(count, np.nan, dtype=np.float64)
        confidence = np.zeros(count, dtype=np.float64)
        if domain_id in self.values:
            values.fill(self.values[domain_id])
            confidence.fill(1.0)
        return BatchInterpolationResult(
            values, np.full(count, np.nan), np.zeros(count, dtype=np.int16), confidence, np.full(count, np.nan)
        )


class DomainKrigingInterpolator(SpatialInterpolator):
    """Ordinary kriging models fitted independently inside each structural domain."""

    def __init__(self, samples: Iterable[SpatialSample], settings) -> None:
        self.models: dict[int | None, OrdinaryKrigingInterpolator] = {}
        self.diagnostics: dict[int | None, VariogramDiagnostics] = {}
        grouped: dict[int | None, list[SpatialSample]] = {}
        for sample in samples:
            grouped.setdefault(sample.domain_id, []).append(sample)
        self.failures: dict[int | None, str] = {}
        for domain_id, rows in grouped.items():
            coordinates = np.asarray([(row.x, row.y, row.z) for row in rows], dtype=float)
            values = np.asarray([row.value for row in rows], dtype=float)
            try:
                diagnostics = fit_variogram(coordinates, values, settings)
                self.models[domain_id] = OrdinaryKrigingInterpolator(coordinates, values, diagnostics, settings)
                self.diagnostics[domain_id] = diagnostics
            except ValueError as error:
                self.failures[domain_id] = str(error)

    def predict(self, point: tuple[float, float, float], domain_id: int | None) -> InterpolationResult:
        model = self.models.get(domain_id)
        if model is None:
            return InterpolationResult(None, VoxelCellState.NO_DATA, provenance="insufficient_data")
        prediction = model.predict(point)
        if prediction.estimate is None:
            return InterpolationResult(None, VoxelCellState.NO_DATA, neighbor_count=prediction.neighbor_count,
                                       provenance=prediction.status.lower())
        value = prediction.estimate
        return InterpolationResult(
            value,
            VoxelCellState.TRUE_ZERO if value == 0 else VoxelCellState.MODELED_VALUE,
            neighbor_count=prediction.neighbor_count,
            confidence=min(1.0, prediction.neighbor_count / model.settings.maximum_neighbors),
            provenance=prediction.status.lower(),
            kriging_variance=float(prediction.variance) if prediction.variance is not None else float("nan"),
            used_regularization=prediction.used_regularization,
            used_pseudoinverse=prediction.used_pseudoinverse,
        )

    def predict_many(self, points: np.ndarray, domain_id: int | None) -> BatchInterpolationResult:
        """Predict a domain-local bounded batch using the kriging batch solver."""
        count = len(points)
        model = self.models.get(domain_id)
        if model is None:
            return BatchInterpolationResult(
                np.full(count, np.nan), np.full(count, np.nan), np.zeros(count, dtype=np.int16),
                np.zeros(count), np.full(count, np.nan),
            )
        values, variances, neighbours, _status = model.predict_many(points)
        confidence = np.minimum(1.0, neighbours.astype(float) / model.settings.maximum_neighbors)
        return BatchInterpolationResult(values, np.full(count, np.nan), neighbours, confidence, variances)


class ParameterFieldBuilder:
    """Build the first voxelized DFN parameter field without explicit fractures."""

    def __init__(self, memory_warning_bytes: int = 2 * 1024**3) -> None:
        self.memory_warning_bytes = memory_warning_bytes

    @staticmethod
    def estimate_bytes(
        bounds: ModelBounds,
        voxel: VoxelConfig,
        set_count: int,
        method: DensityMethod = DensityMethod.GLOBAL_CONSTANT,
    ) -> int:
        """Estimate exact persisted-array payload before allocating it."""
        return estimate_parameter_field_resources(
            bounds, voxel, range(set_count), method
        ).persistent_bytes

    @staticmethod
    def estimate_resources(
        bounds: ModelBounds,
        voxel: VoxelConfig,
        set_ids: Iterable[int],
        method: DensityMethod,
        *,
        chunk_size: int = 8192,
        budget_bytes: int | None = None,
        system_available_bytes: int | None = None,
    ) -> FieldResourceEstimate:
        """Return the public array-by-array and temporary resource estimate."""
        return estimate_parameter_field_resources(
            bounds, voxel, set_ids, method, chunk_size=chunk_size, budget_bytes=budget_bytes,
            system_available_bytes=system_available_bytes,
        )

    def build(
        self,
        bounds: ModelBounds,
        voxel: VoxelConfig,
        settings: DensitySettings,
        p10_intervals: Iterable[P10Interval],
        p32_estimates: Iterable[P32Estimate],
        joint_sets: Iterable[JointSetConfig],
        size_models: Iterable[SizeModel],
        *,
        random_seed: int,
        domain_at_point: Callable[[tuple[float, float, float]], int | None] | None = None,
        inside_model: Callable[[tuple[float, float, float]], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        chunk_size: int = 8192,
        orientation_models: Iterable[DomainOrientationModel] | None = None,
    ) -> tuple[ParameterFieldMetadata, dict[str, np.ndarray]]:
        """Build arrays in bounded chunks with progress and cancellation hooks."""
        sets = {item.set_id: item for item in joint_sets}
        set_ids = sorted(sets)
        sizes = {(item.domain_id, item.set_id): item for item in size_models}
        explicit_orientation_models = orientation_models is not None
        orientations = {(item.domain_id, item.set_id): item for item in (orientation_models or [])}
        estimates = {(item.domain_id, item.set_id): item for item in p32_estimates if item.p32 is not None}
        p10_intervals = list(p10_intervals)
        nx, ny, nz = voxel.compute_grid_dimensions(bounds)
        shape = (nx, ny, nz)
        total = nx * ny * nz
        arrays: dict[str, np.ndarray] = {
            "cell_state": np.full(shape, CELL_STATE_CODES[VoxelCellState.NO_DATA], dtype=np.uint8),
            "domain_id": np.full(shape, -1, dtype=np.int32),
            "p32_total": np.full(shape, np.nan, dtype=np.float32),
            "nearest_data_distance": np.full(shape, np.nan, dtype=np.float32),
            "neighbour_count": np.zeros(shape, dtype=np.int16),
            "effective_sample_length": np.zeros(shape, dtype=np.float32),
            "observation_count": np.zeros(shape, dtype=np.int32),
            "confidence": np.zeros(shape, dtype=np.float32),
            "density_method": np.full(shape, DENSITY_METHOD_CODES[settings.method], dtype=np.uint8),
        }
        if settings.method == DensityMethod.ORDINARY_KRIGING:
            arrays["kriging_variance_total"] = np.full(shape, np.nan, dtype=np.float32)
        for set_id in set_ids:
            arrays[f"set_{set_id}_p32"] = np.full(shape, np.nan, dtype=np.float32)
            if settings.method == DensityMethod.ORDINARY_KRIGING:
                arrays[f"set_{set_id}_kriging_variance"] = np.full(shape, np.nan, dtype=np.float32)
            unresolved_representative_kappa = sets[set_id].provenance.get("kappa_status") == "UNRESOLVED"
            use_project_orientation = not explicit_orientation_models and not unresolved_representative_kappa
            default_direction = sets[set_id].orientation.mean_dip_direction if use_project_orientation else np.nan
            default_dip = sets[set_id].orientation.mean_dip if use_project_orientation else np.nan
            default_kappa = sets[set_id].orientation.kappa if use_project_orientation else np.nan
            arrays[f"set_{set_id}_dip_direction"] = np.full(shape, default_direction, dtype=np.float32)
            arrays[f"set_{set_id}_dip"] = np.full(shape, default_dip, dtype=np.float32)
            arrays[f"set_{set_id}_kappa"] = np.full(shape, default_kappa, dtype=np.float32)
            arrays[f"set_{set_id}_mean_radius"] = np.full(shape, np.nan, dtype=np.float32)
            arrays[f"set_{set_id}_mean_squared_radius"] = np.full(shape, np.nan, dtype=np.float32)
            arrays[f"set_{set_id}_probability"] = np.full(shape, np.nan, dtype=np.float32)
            arrays[f"set_{set_id}_size_type"] = np.full(shape, 255, dtype=np.uint8)
            arrays[f"set_{set_id}_size_source"] = np.full(shape, 255, dtype=np.uint8)
            arrays[f"set_{set_id}_size_min_radius"] = np.full(shape, np.nan, dtype=np.float32)
            arrays[f"set_{set_id}_size_max_radius"] = np.full(shape, np.nan, dtype=np.float32)
            arrays[f"set_{set_id}_size_parameter_1"] = np.full(shape, np.nan, dtype=np.float32)
            arrays[f"set_{set_id}_size_parameter_2"] = np.full(shape, np.nan, dtype=np.float32)

        domain_values_by_set: dict[int, dict[int | None, float]] = {}
        for set_id in set_ids:
            domain_values_by_set[set_id] = {
                domain_id: estimate.p32 for (domain_id, sid), estimate in estimates.items() if sid == set_id and estimate.p32 is not None
            }
        calibration_rows = [item for item in p10_intervals if item.role == "calibration" and item.p10 is not None]
        interpolators: dict[int, SpatialInterpolator] = {}
        rejected_set_voxel_count = 0
        clipped_set_voxel_count = 0
        rejected_voxel_count = 0
        clipped_voxel_count = 0
        pre_adjustment_minimum: float | None = None
        pre_adjustment_maximum: float | None = None
        negative_clipped_total_change = 0.0
        for set_id in set_ids:
            values = domain_values_by_set[set_id]
            if settings.method == DensityMethod.GLOBAL_CONSTANT:
                interpolators[set_id] = GlobalConstantInterpolator(values)
                continue
            samples = []
            for row_index, row in enumerate(calibration_rows):
                estimate = estimates.get((row.domain_id, set_id))
                if row.set_id == set_id and estimate is not None:
                    orientation = orientations.get((row.domain_id, set_id))
                    exposure_set = sets[set_id]
                    if orientation is not None:
                        exposure_set = exposure_set.model_copy(
                            update={
                                "orientation": exposure_set.orientation.model_copy(
                                    update={
                                        "mean_dip_direction": orientation.mean_dip_direction,
                                        "mean_dip": orientation.mean_dip,
                                        "kappa": orientation.kappa,
                                    }
                                )
                            }
                        )
                    effective_length = sum(
                        length
                        * expected_orientation_exposure(
                            exposure_set,
                            (dx, dy, dz),
                            random_seed=random_seed + set_id * 1_000_003 + row_index,
                            sample_count=settings.monte_carlo_samples,
                        )
                        for dx, dy, dz, length in row.segment_directions
                    )
                    exposure = effective_length / row.sample_length if row.sample_length else 0.0
                    if exposure < settings.low_observability_threshold:
                        continue
                    samples.append(
                        SpatialSample(
                            row.center_x or 0.0,
                            row.center_y or 0.0,
                            row.center_z or 0.0,
                            row.p10 / exposure,
                            row.domain_id,
                            effective_length,
                            row.observation_count,
                        )
                    )
            if settings.method == DensityMethod.IDW:
                interpolators[set_id] = IDWInterpolator(
                    samples,
                    power=settings.power,
                    search_radius=settings.search_radius,
                    min_neighbors=settings.min_neighbors,
                    max_neighbors=settings.max_neighbors,
                    anisotropy=(settings.anisotropy_x, settings.anisotropy_y, settings.anisotropy_z),
                    fallback_by_domain=values if settings.global_fallback else None,
                )
            else:
                interpolators[set_id] = DomainKrigingInterpolator(samples, settings.kriging)

        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        flat_arrays = {name: array.ravel() for name, array in arrays.items()}
        origin = (bounds.x_min, bounds.y_min, bounds.z_min)
        spacing = (voxel.cell_size_x, voxel.cell_size_y, voxel.cell_size_z)
        for flat_start in range(0, total, chunk_size):
            if cancelled and cancelled():
                raise InterruptedError("parameter field generation cancelled")
            flat_end = min(total, flat_start + chunk_size)
            flat_indices, points = _chunk_points(flat_start, flat_end, shape, origin, spacing)
            inside = _callback_many(inside_model, points, True, np.bool_)
            flat_arrays["cell_state"][flat_indices[~inside]] = CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL]
            if np.any(inside):
                inside_rows = np.flatnonzero(inside)
                inside_points = points[inside_rows]
                raw_domains = _domain_callback_many(domain_at_point, inside_points)
                domain_values = (
                    np.unique(raw_domains).tolist()
                    if raw_domains.dtype != object
                    else list(dict.fromkeys(raw_domains.tolist()))
                )
                for domain_id in domain_values:
                    if cancelled and cancelled():
                        raise InterruptedError("parameter field generation cancelled")
                    domain_rows = inside_rows[raw_domains == domain_id]
                    output_flats = flat_indices[domain_rows]
                    target_points = points[domain_rows]
                    flat_arrays["domain_id"][output_flats] = -1 if domain_id is None else domain_id
                    value_rows: list[np.ndarray] = []
                    distance_rows: list[np.ndarray] = []
                    neighbour_rows: list[np.ndarray] = []
                    confidence_rows: list[np.ndarray] = []
                    variance_rows: list[np.ndarray] = []
                    rejected_here = np.zeros(len(output_flats), dtype=np.bool_)
                    clipped_here = np.zeros(len(output_flats), dtype=np.bool_)
                    for set_id in set_ids:
                        if cancelled and cancelled():
                            raise InterruptedError("parameter field generation cancelled")
                        batch = _predict_many_compatible(interpolators[set_id], target_points, domain_id)
                        values = batch.values.copy()
                        raw_finite = np.isfinite(values)
                        if settings.method == DensityMethod.ORDINARY_KRIGING and np.any(raw_finite):
                            batch_minimum = float(np.min(values[raw_finite]))
                            batch_maximum = float(np.max(values[raw_finite]))
                            pre_adjustment_minimum = (
                                batch_minimum if pre_adjustment_minimum is None
                                else min(pre_adjustment_minimum, batch_minimum)
                            )
                            pre_adjustment_maximum = (
                                batch_maximum if pre_adjustment_maximum is None
                                else max(pre_adjustment_maximum, batch_maximum)
                            )
                            negative = raw_finite & (values < 0)
                            if settings.kriging.non_negative_policy.value == "reject":
                                rejected_set_voxel_count += int(np.count_nonzero(negative))
                                rejected_here |= negative
                                values[negative] = np.nan
                            else:
                                clipped_set_voxel_count += int(np.count_nonzero(negative))
                                clipped_here |= negative
                                negative_clipped_total_change += float(np.sum(-values[negative]))
                                values[negative] = 0.0
                        valid = np.isfinite(values)
                        orientation = orientations.get((domain_id, set_id))
                        if orientation is not None:
                            flat_arrays[f"set_{set_id}_dip_direction"][output_flats] = orientation.mean_dip_direction
                            flat_arrays[f"set_{set_id}_dip"][output_flats] = orientation.mean_dip
                            flat_arrays[f"set_{set_id}_kappa"][output_flats] = orientation.kappa
                        selected = output_flats[valid]
                        flat_arrays[f"set_{set_id}_p32"][selected] = values[valid]
                        if settings.method == DensityMethod.ORDINARY_KRIGING:
                            flat_arrays[f"set_{set_id}_kriging_variance"][selected] = batch.variances[valid]
                        size = sizes.get((domain_id, set_id)) or sizes.get((None, set_id))
                        if size is not None and selected.size:
                            flat_arrays[f"set_{set_id}_mean_radius"][selected] = size.mean_radius
                            flat_arrays[f"set_{set_id}_mean_squared_radius"][selected] = size.mean_squared_radius
                            flat_arrays[f"set_{set_id}_size_type"][selected] = SIZE_TYPE_CODES[size.distribution_type]
                            flat_arrays[f"set_{set_id}_size_source"][selected] = SIZE_SOURCE_CODES[size.source.value]
                            flat_arrays[f"set_{set_id}_size_min_radius"][selected] = size.min_radius
                            flat_arrays[f"set_{set_id}_size_max_radius"][selected] = size.max_radius
                            parameter_values = [
                                size.parameters[name] for name in SIZE_PARAMETER_ORDER[size.distribution_type]
                                if name in size.parameters
                            ]
                            if parameter_values:
                                flat_arrays[f"set_{set_id}_size_parameter_1"][selected] = parameter_values[0]
                            if len(parameter_values) > 1:
                                flat_arrays[f"set_{set_id}_size_parameter_2"][selected] = parameter_values[1]
                        value_rows.append(values)
                        distance_rows.append(batch.nearest_distances)
                        neighbour_rows.append(batch.neighbor_counts)
                        confidence_rows.append(batch.confidences)
                        variance_rows.append(batch.variances)
                    rejected_voxel_count += int(np.count_nonzero(rejected_here))
                    clipped_voxel_count += int(np.count_nonzero(clipped_here))
                    if not value_rows:
                        continue
                    valid_rows = [np.isfinite(values) for values in value_rows]
                    any_valid = np.logical_or.reduce(valid_rows)
                    if not np.any(any_valid):
                        continue
                    total_values = np.zeros(len(output_flats), dtype=np.float64)
                    maximum_neighbours = np.zeros(len(output_flats), dtype=np.int16)
                    minimum_confidence = np.full(len(output_flats), np.inf, dtype=np.float64)
                    minimum_distance = np.full(len(output_flats), np.inf, dtype=np.float64)
                    total_variance = np.zeros(len(output_flats), dtype=np.float64)
                    has_variance = np.zeros(len(output_flats), dtype=np.bool_)
                    for set_id, values, valid, distances, neighbours, confidences, variances in zip(
                        set_ids, value_rows, valid_rows, distance_rows, neighbour_rows,
                        confidence_rows, variance_rows, strict=True,
                    ):
                        total_values[valid] += values[valid]
                        maximum_neighbours[valid] = np.maximum(maximum_neighbours[valid], neighbours[valid])
                        minimum_confidence[valid] = np.minimum(minimum_confidence[valid], confidences[valid])
                        finite_distance = valid & np.isfinite(distances)
                        minimum_distance[finite_distance] = np.minimum(
                            minimum_distance[finite_distance], distances[finite_distance]
                        )
                        finite_variance = valid & np.isfinite(variances)
                        total_variance[finite_variance] += variances[finite_variance]
                        has_variance |= finite_variance
                    modeled = output_flats[any_valid]
                    flat_arrays["p32_total"][modeled] = total_values[any_valid]
                    for set_id, values, valid in zip(set_ids, value_rows, valid_rows, strict=True):
                        probability = np.zeros(len(output_flats), dtype=np.float64)
                        positive = valid & (total_values > 0)
                        stored_values = flat_arrays[f"set_{set_id}_p32"][output_flats]
                        probability[positive] = (
                            stored_values[positive] / total_values[positive].astype(np.float32)
                        )
                        flat_arrays[f"set_{set_id}_probability"][output_flats[valid]] = probability[valid]
                    flat_arrays["cell_state"][modeled] = np.where(
                        total_values[any_valid] == 0,
                        CELL_STATE_CODES[VoxelCellState.TRUE_ZERO], CELL_STATE_CODES[VoxelCellState.MODELED_VALUE],
                    )
                    finite_distance = any_valid & np.isfinite(minimum_distance)
                    flat_arrays["nearest_data_distance"][output_flats[finite_distance]] = minimum_distance[finite_distance]
                    flat_arrays["neighbour_count"][modeled] = maximum_neighbours[any_valid]
                    flat_arrays["confidence"][modeled] = minimum_confidence[any_valid]
                    if settings.method == DensityMethod.ORDINARY_KRIGING:
                        flat_arrays["kriging_variance_total"][output_flats[has_variance]] = total_variance[has_variance]
                    supporting = [estimates.get((domain_id, set_id)) for set_id in set_ids]
                    supporting = [item for item in supporting if item is not None]
                    flat_arrays["effective_sample_length"][modeled] = sum(
                        item.effective_sample_length for item in supporting
                    )
                    flat_arrays["observation_count"][modeled] = sum(item.fracture_count for item in supporting)
            if progress:
                progress(flat_end, total)
        metadata = ParameterFieldMetadata(
            shape=shape,
            origin=(bounds.x_min, bounds.y_min, bounds.z_min),
            spacing=(voxel.cell_size_x, voxel.cell_size_y, voxel.cell_size_z),
            field_names=sorted(arrays),
            set_ids=set_ids,
            density_method=settings.method,
            random_seed=random_seed,
            estimated_bytes=sum(array.nbytes for array in arrays.values()),
            provenance={
                "calibration_only": True,
                "explicit_dfn_generated": False,
                "cell_state_codes": {state.value: code for state, code in CELL_STATE_CODES.items()},
                "size_type_codes": SIZE_TYPE_CODES,
                "size_source_codes": SIZE_SOURCE_CODES,
                "size_parameter_order": SIZE_PARAMETER_ORDER,
                "density_method_codes": {method.value: code for method, code in DENSITY_METHOD_CODES.items()},
                "ordinary_kriging": {
                    str(set_id): {
                        "domains": {
                            str(domain_id): diagnostics.model_dump(mode="json")
                            for domain_id, diagnostics in getattr(interpolators[set_id], "diagnostics", {}).items()
                        },
                        "failures": getattr(interpolators[set_id], "failures", {}),
                    }
                    for set_id in set_ids
                } if settings.method == DensityMethod.ORDINARY_KRIGING else {},
                "negative_prediction_audit": {
                    "policy": settings.kriging.non_negative_policy.value,
                    "parameter_bounds": {"minimum": 0.0, "maximum": None},
                    "rejected_set_voxel_count": rejected_set_voxel_count,
                    "clipped_set_voxel_count": clipped_set_voxel_count,
                    "rejected_voxel_count": rejected_voxel_count,
                    "clipped_voxel_count": clipped_voxel_count,
                    "pre_adjustment_minimum": pre_adjustment_minimum,
                    "pre_adjustment_maximum": pre_adjustment_maximum,
                    "pre_clip_minimum": pre_adjustment_minimum,
                    "clipped_total_change": negative_clipped_total_change,
                    "count_units": {
                        "set_voxel_count": "joint-set predictions at spatial voxels",
                        "voxel_count": "unique spatial voxels with at least one adjusted joint-set prediction",
                    },
                    "aggregation_semantics": (
                        "A rejected joint-set prediction is omitted from p32_total; remaining valid joint-set "
                        "predictions are summed. The aggregate cell is NO_DATA only when no valid set remains."
                    ),
                } if settings.method == DensityMethod.ORDINARY_KRIGING else {},
                "kriging_variance_total_semantics": (
                    "Sum of per-joint-set kriging variances under an inter-set independence assumption."
                    if settings.method == DensityMethod.ORDINARY_KRIGING
                    else None
                ),
            },
        )
        return metadata, arrays
