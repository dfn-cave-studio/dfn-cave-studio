"""Replaceable spatial interpolation and chunked M9 parameter-field builder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import numpy as np

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
)
from dfn_cave_studio.models.spatial_grid import VoxelCellState


CELL_STATE_CODES = {
    VoxelCellState.OUTSIDE_MODEL: 0,
    VoxelCellState.NO_DATA: 1,
    VoxelCellState.TRUE_ZERO: 2,
    VoxelCellState.MODELED_VALUE: 3,
}
SIZE_TYPE_CODES = {
    "fixed": 0,
    "uniform": 1,
    "truncated_lognormal": 2,
    "truncated_power_law": 3,
    "truncated_exponential": 4,
}
SIZE_SOURCE_CODES = {"fitted": 0, "size_proxy": 1, "assumed": 2, "user_defined": 3}
DENSITY_METHOD_CODES = {DensityMethod.GLOBAL_CONSTANT: 0, DensityMethod.IDW: 1}
SIZE_PARAMETER_ORDER = {
    "fixed": ["radius"],
    "uniform": [],
    "truncated_lognormal": ["mu", "sigma"],
    "truncated_power_law": ["exponent"],
    "truncated_exponential": ["rate"],
}


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

    def predict(self, point: tuple[float, float, float], domain_id: int | None) -> InterpolationResult:
        eligible = [sample for sample in self.samples if sample.domain_id == domain_id]
        if not eligible:
            return self._fallback(domain_id)
        coordinates = np.asarray([(sample.x, sample.y, sample.z) for sample in eligible], dtype=float)
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
        indices = np.argsort(distances)
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


class ParameterFieldBuilder:
    """Build the first voxelized DFN parameter field without explicit fractures."""

    def __init__(self, memory_warning_bytes: int = 2 * 1024**3) -> None:
        self.memory_warning_bytes = memory_warning_bytes

    @staticmethod
    def estimate_bytes(bounds: ModelBounds, voxel: VoxelConfig, set_count: int) -> int:
        """Estimate dense-array memory before allocating it."""
        nx, ny, nz = voxel.compute_grid_dimensions(bounds)
        base_bytes = 4 * 4 + 1 + 4 + 2 + 4 + 1
        per_set_bytes = 11 * 4 + 2
        return nx * ny * nz * (base_bytes + set_count * per_set_bytes)

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
        for set_id in set_ids:
            arrays[f"set_{set_id}_p32"] = np.full(shape, np.nan, dtype=np.float32)
            arrays[f"set_{set_id}_dip_direction"] = np.full(shape, sets[set_id].orientation.mean_dip_direction, dtype=np.float32)
            arrays[f"set_{set_id}_dip"] = np.full(shape, sets[set_id].orientation.mean_dip, dtype=np.float32)
            arrays[f"set_{set_id}_kappa"] = np.full(shape, sets[set_id].orientation.kappa, dtype=np.float32)
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
        for set_id in set_ids:
            values = domain_values_by_set[set_id]
            if settings.method == DensityMethod.GLOBAL_CONSTANT:
                interpolators[set_id] = GlobalConstantInterpolator(values)
            else:
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
                interpolators[set_id] = IDWInterpolator(
                    samples,
                    power=settings.power,
                    search_radius=settings.search_radius,
                    min_neighbors=settings.min_neighbors,
                    max_neighbors=settings.max_neighbors,
                    anisotropy=(settings.anisotropy_x, settings.anisotropy_y, settings.anisotropy_z),
                    fallback_by_domain=values if settings.global_fallback else None,
                )

        for flat_start in range(0, total, chunk_size):
            if cancelled and cancelled():
                raise InterruptedError("parameter field generation cancelled")
            flat_end = min(total, flat_start + chunk_size)
            for flat in range(flat_start, flat_end):
                i, j, k = np.unravel_index(flat, shape)
                point = (
                    bounds.x_min + (i + 0.5) * voxel.cell_size_x,
                    bounds.y_min + (j + 0.5) * voxel.cell_size_y,
                    bounds.z_min + (k + 0.5) * voxel.cell_size_z,
                )
                if inside_model is not None and not inside_model(point):
                    arrays["cell_state"][i, j, k] = CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL]
                    continue
                domain_id = domain_at_point(point) if domain_at_point else None
                arrays["domain_id"][i, j, k] = -1 if domain_id is None else domain_id
                results = [interpolators[set_id].predict(point, domain_id) for set_id in set_ids]
                valid = [result for result in results if result.value is not None]
                for set_id, result in zip(set_ids, results):
                    orientation = orientations.get((domain_id, set_id))
                    if orientation is not None:
                        arrays[f"set_{set_id}_dip_direction"][i, j, k] = orientation.mean_dip_direction
                        arrays[f"set_{set_id}_dip"][i, j, k] = orientation.mean_dip
                        arrays[f"set_{set_id}_kappa"][i, j, k] = orientation.kappa
                    if result.value is not None:
                        arrays[f"set_{set_id}_p32"][i, j, k] = result.value
                        size = sizes.get((domain_id, set_id)) or sizes.get((None, set_id))
                        if size:
                            arrays[f"set_{set_id}_mean_radius"][i, j, k] = size.mean_radius
                            arrays[f"set_{set_id}_mean_squared_radius"][i, j, k] = size.mean_squared_radius
                            arrays[f"set_{set_id}_size_type"][i, j, k] = SIZE_TYPE_CODES[size.distribution_type]
                            arrays[f"set_{set_id}_size_source"][i, j, k] = SIZE_SOURCE_CODES[size.source.value]
                            arrays[f"set_{set_id}_size_min_radius"][i, j, k] = size.min_radius
                            arrays[f"set_{set_id}_size_max_radius"][i, j, k] = size.max_radius
                            parameter_values = [
                                size.parameters[name]
                                for name in SIZE_PARAMETER_ORDER[size.distribution_type]
                                if name in size.parameters
                            ]
                            if parameter_values:
                                arrays[f"set_{set_id}_size_parameter_1"][i, j, k] = parameter_values[0]
                            if len(parameter_values) > 1:
                                arrays[f"set_{set_id}_size_parameter_2"][i, j, k] = parameter_values[1]
                if not valid:
                    continue
                total_p32 = sum(float(result.value) for result in valid if result.value is not None)
                arrays["p32_total"][i, j, k] = total_p32
                for set_id in set_ids:
                    set_value = arrays[f"set_{set_id}_p32"][i, j, k]
                    if np.isfinite(set_value):
                        arrays[f"set_{set_id}_probability"][i, j, k] = set_value / total_p32 if total_p32 > 0 else 0.0
                arrays["cell_state"][i, j, k] = CELL_STATE_CODES[
                    VoxelCellState.TRUE_ZERO if total_p32 == 0 else VoxelCellState.MODELED_VALUE
                ]
                distances = [result.nearest_distance for result in valid if np.isfinite(result.nearest_distance)]
                arrays["nearest_data_distance"][i, j, k] = min(distances) if distances else np.nan
                arrays["neighbour_count"][i, j, k] = max(result.neighbor_count for result in valid)
                arrays["confidence"][i, j, k] = min(result.confidence for result in valid)
                supporting = [estimates.get((domain_id, set_id)) for set_id in set_ids]
                supporting = [item for item in supporting if item is not None]
                arrays["effective_sample_length"][i, j, k] = sum(item.effective_sample_length for item in supporting)
                arrays["observation_count"][i, j, k] = sum(item.fracture_count for item in supporting)
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
            },
        )
        return metadata, arrays
