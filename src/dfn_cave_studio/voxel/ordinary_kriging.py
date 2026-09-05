"""Deterministic three-dimensional isotropic ordinary kriging."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree, distance

from dfn_cave_studio.models.m9 import (
    KrigingSettings,
    VariogramDiagnostics,
    VariogramLag,
    VariogramMode,
    VariogramModel,
)


def semivariogram(
    distances: np.ndarray | float,
    model: VariogramModel | str,
    nugget: float,
    sill: float,
    variogram_range: float,
) -> np.ndarray:
    """Evaluate a standard isotropic semivariogram."""
    if nugget < 0 or sill <= 0 or variogram_range <= 0 or sill < nugget:
        raise ValueError("Require nugget >= 0, sill >= nugget, and range > 0")
    h = np.asarray(distances, dtype=np.float64)
    if np.any(~np.isfinite(h)) or np.any(h < 0):
        raise ValueError("Semivariogram distances must be finite and non-negative")
    kind = VariogramModel(model)
    ratio = h / variogram_range
    partial = sill - nugget
    if kind == VariogramModel.SPHERICAL:
        core = np.where(ratio < 1.0, nugget + partial * (1.5 * ratio - 0.5 * ratio**3), sill)
    elif kind == VariogramModel.EXPONENTIAL:
        core = nugget + partial * (1.0 - np.exp(-3.0 * ratio))
    else:
        core = nugget + partial * (1.0 - np.exp(-3.0 * ratio**2))
    return np.where(h == 0.0, 0.0, core)


def experimental_variogram(
    coordinates: np.ndarray,
    values: np.ndarray,
    lag_count: int,
) -> tuple[list[VariogramLag], int]:
    """Calculate pair-count-weighted experimental semivariance bins."""
    xyz, data = _validated_samples(coordinates, values)
    if len(xyz) < 3:
        raise ValueError("INSUFFICIENT_DATA: at least three distinct sample coordinates are required")
    pair_distances = distance.pdist(xyz)
    pair_semivariance = 0.5 * distance.pdist(data[:, None], metric="sqeuclidean")
    positive = pair_distances > 0.0
    pair_distances, pair_semivariance = pair_distances[positive], pair_semivariance[positive]
    if len(pair_distances) < 3:
        raise ValueError("INSUFFICIENT_DATA: too few distinct sample pairs for a variogram")
    edges = np.linspace(0.0, float(pair_distances.max()), lag_count + 1)
    bins = np.minimum(np.digitize(pair_distances, edges[1:], right=True), lag_count - 1)
    lags: list[VariogramLag] = []
    for index in range(lag_count):
        selected = bins == index
        count = int(np.count_nonzero(selected))
        if count:
            lags.append(
                VariogramLag(
                    distance=float(np.mean(pair_distances[selected])),
                    pair_count=count,
                    experimental_semivariance=float(np.mean(pair_semivariance[selected])),
                )
            )
    return lags, len(pair_distances)


def fit_variogram(
    coordinates: np.ndarray,
    values: np.ndarray,
    settings: KrigingSettings,
) -> VariogramDiagnostics:
    """Fit or validate a semivariogram using Calibration samples only."""
    xyz, data = _validated_samples(coordinates, values)
    if settings.mode == VariogramMode.MANUAL:
        return VariogramDiagnostics(
            mode=settings.mode,
            model=settings.model,
            nugget=settings.nugget,
            sill=settings.sill,
            range=settings.range,
            sample_count=len(xyz),
            pair_count=max(0, len(xyz) * (len(xyz) - 1) // 2),
            fit_status="manual",
        )
    lags, pair_count = experimental_variogram(xyz, data, settings.lag_count)
    usable = [item for item in lags if item.pair_count >= 1 and item.experimental_semivariance is not None]
    if len(usable) < 3:
        raise ValueError("INSUFFICIENT_DATA: at least three populated variogram lags are required")
    x = np.asarray([item.distance for item in usable], dtype=float)
    y = np.asarray([item.experimental_semivariance for item in usable], dtype=float)
    weights = np.sqrt(np.asarray([item.pair_count for item in usable], dtype=float))
    variance = float(np.var(data))
    maximum_distance = float(max(x))
    initial = np.asarray(
        [min(settings.nugget, max(variance, 1e-12)), max(variance, settings.sill, 1e-12), max(maximum_distance / 2, 1e-9)]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        nugget, partial_sill, fitted_range = parameters
        fitted_sill = nugget + partial_sill
        return weights * (semivariogram(x, settings.model, nugget, fitted_sill, fitted_range) - y)

    fitted = least_squares(
        residual,
        x0=(initial[0], max(initial[1] - initial[0], 1e-12), initial[2]),
        bounds=((0.0, 1e-12, 1e-9), (np.inf, np.inf, max(maximum_distance * 10.0, 1e-8))),
    )
    if not fitted.success:
        raise ValueError(f"Variogram fitting failed: {fitted.message}")
    nugget, partial_sill, fitted_range = map(float, fitted.x)
    sill = nugget + partial_sill
    for item in lags:
        item.fitted_semivariance = float(
            semivariogram(item.distance, settings.model, nugget, sill, fitted_range)
        )
    warnings = [] if pair_count >= 30 else ["Variogram has fewer than 30 point pairs; reliability is limited"]
    return VariogramDiagnostics(
        mode=settings.mode,
        model=settings.model,
        nugget=nugget,
        sill=sill,
        range=fitted_range,
        lags=lags,
        fit_status="converged" if fitted.success else "failed",
        optimizer_message=str(fitted.message),
        sample_count=len(xyz),
        pair_count=pair_count,
        warnings=warnings,
    )


@dataclass(frozen=True)
class KrigingPrediction:
    """One estimate with conditional model variance and solve diagnostics."""

    estimate: float | None
    variance: float | None
    neighbor_count: int
    status: str
    weights: np.ndarray | None = None
    used_regularization: bool = False
    used_pseudoinverse: bool = False


class OrdinaryKrigingInterpolator:
    """Local-neighbour ordinary kriging backed by a cKDTree."""

    def __init__(self, coordinates: np.ndarray, values: np.ndarray, diagnostics: VariogramDiagnostics, settings: KrigingSettings):
        coordinates, values = _validated_samples(coordinates, values)
        positions: dict[tuple[float, float, float], int] = {}
        coordinate_rows: list[np.ndarray] = []
        value_rows: list[list[float]] = []
        original_indices: list[int] = []
        for original_index, (coordinate, value) in enumerate(zip(coordinates, values, strict=True)):
            key = tuple(float(item) for item in coordinate)
            position = positions.get(key)
            if position is None:
                positions[key] = len(coordinate_rows)
                coordinate_rows.append(coordinate.copy())
                value_rows.append([float(value)])
                original_indices.append(original_index)
            else:
                value_rows[position].append(float(value))
        self.coordinates = np.asarray(coordinate_rows, dtype=np.float64)
        self.values = np.asarray([np.mean(row) for row in value_rows], dtype=np.float64)
        self.original_indices = np.asarray(original_indices, dtype=np.int64)
        self.diagnostics = diagnostics
        self.settings = settings
        self.tree = cKDTree(self.coordinates)

    def _stable_indices(
        self,
        target: np.ndarray,
        queried_distances: np.ndarray | None = None,
        queried_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select neighbours by (distance, original input index), including boundary ties."""
        radius = self.settings.search_radius
        maximum = min(self.settings.maximum_neighbors, len(self.coordinates))
        if queried_distances is None or queried_indices is None:
            query_count = min(maximum + 1, len(self.coordinates))
            queried_distances, queried_indices = self.tree.query(
                target,
                k=query_count,
                distance_upper_bound=np.inf if radius is None else radius,
            )
        distances = np.atleast_1d(np.asarray(queried_distances, dtype=float))
        indices = np.atleast_1d(np.asarray(queried_indices, dtype=np.int64))
        valid = np.isfinite(distances) & (indices < len(self.coordinates))
        distances, indices = distances[valid], indices[valid]
        if len(indices) > maximum:
            boundary_distance = float(np.partition(distances, maximum - 1)[maximum - 1])
            tolerance = max(1.0, abs(boundary_distance)) * 1e-12
            candidates = np.asarray(
                self.tree.query_ball_point(target, boundary_distance + tolerance), dtype=np.int64
            )
            if radius is not None:
                candidate_distances = np.linalg.norm(self.coordinates[candidates] - target, axis=1)
                candidates = candidates[candidate_distances <= radius + tolerance]
        else:
            candidates = indices
        candidate_distances = np.linalg.norm(self.coordinates[candidates] - target, axis=1)
        order = np.lexsort((self.original_indices[candidates], candidate_distances))
        return candidates[order[:maximum]]

    def predict(self, point: tuple[float, float, float] | np.ndarray) -> KrigingPrediction:
        target = np.asarray(point, dtype=float)
        if target.shape != (3,) or np.any(~np.isfinite(target)):
            raise ValueError("Prediction point must contain three finite coordinates")
        indices = self._stable_indices(target)
        if len(indices) < self.settings.minimum_neighbors:
            return KrigingPrediction(None, None, len(indices), "INSUFFICIENT_DATA")
        sample_coordinates = self.coordinates[indices]
        sample_values = self.values[indices]
        if np.any(np.linalg.norm(sample_coordinates - target, axis=1) <= 1e-12):
            exact = int(np.argmin(np.linalg.norm(sample_coordinates - target, axis=1)))
            weights = np.zeros(len(indices)); weights[exact] = 1.0
            return KrigingPrediction(float(sample_values[exact]), 0.0, len(indices), "EXACT_SAMPLE", weights)
        matrix = semivariogram(
            distance.cdist(sample_coordinates, sample_coordinates),
            self.diagnostics.model,
            self.diagnostics.nugget,
            self.diagnostics.sill,
            self.diagnostics.range,
        )
        system = np.empty((len(indices) + 1, len(indices) + 1), dtype=float)
        system[:-1, :-1] = matrix
        system[:-1, -1] = 1.0
        system[-1, :-1] = 1.0
        system[-1, -1] = 0.0
        rhs = np.append(
            semivariogram(
                np.linalg.norm(sample_coordinates - target, axis=1),
                self.diagnostics.model,
                self.diagnostics.nugget,
                self.diagnostics.sill,
                self.diagnostics.range,
            ),
            1.0,
        )
        regularized = False
        pseudoinverse = False
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            regularized = True
            self.diagnostics.regularized_solve_count += 1
            adjusted = system.copy()
            adjusted[:-1, :-1] += np.eye(len(indices)) * self.settings.regularization
            try:
                solution = np.linalg.solve(adjusted, rhs)
            except np.linalg.LinAlgError:
                pseudoinverse = True
                self.diagnostics.pseudoinverse_solve_count += 1
                solution = np.linalg.pinv(adjusted, hermitian=True) @ rhs
        weights, multiplier = solution[:-1], float(solution[-1])
        variance = float(weights @ rhs[:-1] + multiplier)
        tolerance = max(self.diagnostics.sill, 1.0) * 1e-9
        if variance < -tolerance:
            raise ValueError(f"Kriging variance is materially negative: {variance}")
        return KrigingPrediction(
            float(weights @ sample_values),
            max(0.0, variance),
            len(indices),
            "PREDICTED",
            weights.copy(),
            regularized,
            pseudoinverse,
        )

    def predict_many(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict a bounded point batch, reusing each identical local-neighbour system."""
        targets = np.asarray(points, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 3 or np.any(~np.isfinite(targets)):
            raise ValueError("Prediction points must be a finite (N,3) array")
        count = len(targets)
        estimates = np.full(count, np.nan, dtype=np.float64)
        variances = np.full(count, np.nan, dtype=np.float64)
        neighbour_counts = np.zeros(count, dtype=np.int16)
        status = np.full(count, "INSUFFICIENT_DATA", dtype=object)
        if count == 0:
            return estimates, variances, neighbour_counts, status
        radius = self.settings.search_radius if self.settings.search_radius is not None else np.inf
        query_count = min(self.settings.maximum_neighbors + 1, len(self.coordinates))
        distances, indices = self.tree.query(
            targets, k=query_count, distance_upper_bound=radius
        )
        distances = np.asarray(distances).reshape(count, -1)
        indices = np.asarray(indices).reshape(count, -1)
        groups: dict[tuple[int, ...], list[int]] = {}
        for row in range(count):
            selected = tuple(
                int(item)
                for item in self._stable_indices(targets[row], distances[row], indices[row])
            )
            neighbour_counts[row] = len(selected)
            if len(selected) >= self.settings.minimum_neighbors:
                groups.setdefault(selected, []).append(row)
        for selected, rows in groups.items():
            selected_array = np.asarray(selected, dtype=int)
            sample_coordinates = self.coordinates[selected_array]
            sample_values = self.values[selected_array]
            row_array = np.asarray(rows, dtype=int)
            target_group = targets[row_array]
            target_distances = distance.cdist(target_group, sample_coordinates)
            exact_rows = np.any(target_distances <= 1e-12, axis=1)
            if np.any(exact_rows):
                for local_row in np.flatnonzero(exact_rows):
                    output_row = row_array[local_row]
                    exact = int(np.argmin(target_distances[local_row]))
                    estimates[output_row] = sample_values[exact]
                    variances[output_row] = 0.0
                    status[output_row] = "EXACT_SAMPLE"
            solve_local = np.flatnonzero(~exact_rows)
            if not solve_local.size:
                continue
            matrix = semivariogram(
                distance.cdist(sample_coordinates, sample_coordinates), self.diagnostics.model,
                self.diagnostics.nugget, self.diagnostics.sill, self.diagnostics.range,
            )
            system = np.empty((len(selected) + 1, len(selected) + 1), dtype=float)
            system[:-1, :-1] = matrix; system[:-1, -1] = 1.0; system[-1, :-1] = 1.0; system[-1, -1] = 0.0
            rhs_core = semivariogram(
                target_distances[solve_local].T, self.diagnostics.model, self.diagnostics.nugget,
                self.diagnostics.sill, self.diagnostics.range,
            )
            rhs = np.vstack((rhs_core, np.ones((1, len(solve_local)))))
            try:
                solutions = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                self.diagnostics.regularized_solve_count += len(solve_local)
                adjusted = system.copy(); adjusted[:-1, :-1] += np.eye(len(selected)) * self.settings.regularization
                try:
                    solutions = np.linalg.solve(adjusted, rhs)
                except np.linalg.LinAlgError:
                    self.diagnostics.pseudoinverse_solve_count += len(solve_local)
                    solutions = np.linalg.pinv(adjusted, hermitian=True) @ rhs
            weights, multipliers = solutions[:-1], solutions[-1]
            solved_rows = row_array[solve_local]
            estimates[solved_rows] = sample_values @ weights
            solved_variances = np.sum(weights * rhs_core, axis=0) + multipliers
            tolerance = max(self.diagnostics.sill, 1.0) * 1e-9
            if np.any(solved_variances < -tolerance):
                raise ValueError(f"Kriging variance is materially negative: {float(solved_variances.min())}")
            variances[solved_rows] = np.maximum(0.0, solved_variances)
            status[solved_rows] = "PREDICTED"
        return estimates, variances, neighbour_counts, status


def _validated_samples(coordinates: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(coordinates, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or data.shape != (len(xyz),):
        raise ValueError("coordinates must be (N,3) and values must be (N,)")
    if np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(data)):
        raise ValueError("Kriging samples must be finite")
    return xyz, data
