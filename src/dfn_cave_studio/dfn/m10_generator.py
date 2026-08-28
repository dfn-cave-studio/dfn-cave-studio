"""Pure M10 conditional explicit-DFN generation from an M9 parameter field."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from dfn_cave_studio.dfn.fisher import fisher_sample
from dfn_cave_studio.dfn.m10_multiscale import (
    SizeClass,
    SizeThresholdResult,
    calculate_thresholds,
    classify_radius,
    sample_truncated_class,
)
from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal, normal_to_dip_dir_dip
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.m10 import (
    DeterministicStructure,
    M10DomainSetSummary,
    M10FractureSource,
    M10GenerationConfig,
    M10QualitySummary,
    M10Realization,
)
from dfn_cave_studio.models.m9 import ParameterFieldMetadata, SizeModel, SizeModelSource
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES
from dfn_cave_studio.models.spatial_grid import VoxelCellState


@dataclass(frozen=True)
class ConditionedObservation:
    """A calibration FULL_ORIENTATION observation ready for conditioning."""

    observation_record_id: str
    position: tuple[float, float, float]
    dip_direction: float
    dip: float
    domain_id: int | None
    set_id: int
    hole_id: str


@dataclass
class _GeneratedFracture:
    fracture_id: str
    center: np.ndarray
    normal: np.ndarray
    radius: float
    original_area: float
    clipped_area: float
    vertices: np.ndarray
    domain_id: int | None
    set_id: int
    source: str
    observation_record_id: str
    distribution: str
    size_source: str
    orientation_source: str
    dip_only_density_evidence: bool
    voxel_index: tuple[int, int, int]
    size_class: int = int(SizeClass.UNKNOWN)


SOURCE_CODES = {
    M10FractureSource.STOCHASTIC.value: np.uint8(0),
    M10FractureSource.CONDITIONED_OBSERVATION.value: np.uint8(1),
    M10FractureSource.DETERMINISTIC_STRUCTURE.value: np.uint8(2),
}
SOURCE_NAMES = {int(code): name for name, code in SOURCE_CODES.items()}
ORIENTATION_CODES = {
    "domain_set_fisher_model": np.uint8(0),
    "observed_full_orientation": np.uint8(1),
    "deterministic_structure": np.uint8(2),
}
SIZE_SOURCE_CODES = {name: np.uint8(index) for index, name in enumerate(("assumed", "user_defined", "experimental"))}
DISTRIBUTION_CODES = {
    name: np.uint8(index)
    for index, name in enumerate(
        ("fixed", "uniform", "truncated_lognormal", "truncated_power_law", "truncated_exponential")
    )
}
GENERATOR_VERSION = "m10-multiscale-1"


class _ColumnarAccumulator:
    """Growable authoritative arrays without one Python object per fracture."""

    _SPECS = {
        "ordinal": (np.uint64, ()),
        "center": (np.float64, (3,)),
        "normal": (np.float64, (3,)),
        "radius": (np.float64, ()),
        "original_area": (np.float64, ()),
        "clipped_area": (np.float64, ()),
        "domain_id": (np.int32, ()),
        "set_id": (np.int16, ()),
        "source_code": (np.uint8, ()),
        "voxel_index": (np.int32, (3,)),
        "clipped": (np.bool_, ()),
        "clipped_offsets": (np.int64, ()),
        "clipped_counts": (np.int16, ()),
        "distribution_code": (np.uint8, ()),
        "size_source_code": (np.uint8, ()),
        "orientation_source_code": (np.uint8, ()),
        "dip_only_density_evidence": (np.bool_, ()),
        "record_ref": (np.int32, ()),
        "size_class": (np.uint8, ()),
    }

    def __init__(self, capacity: int, maximum: int) -> None:
        self.capacity = max(16, min(int(capacity), int(maximum)))
        self.maximum = int(maximum)
        self.count = 0
        self.arrays = {
            name: np.empty((self.capacity, *tail), dtype=dtype) for name, (dtype, tail) in self._SPECS.items()
        }
        self._clipped_parts: list[np.ndarray] = []
        self._clipped_point_count = 0

    def _ensure(self, additional: int) -> None:
        required = self.count + int(additional)
        if required > self.maximum:
            raise MemoryError("Generated fracture count exceeds the configured safety limit")
        if required <= self.capacity:
            return
        new_capacity = min(self.maximum, max(required, int(self.capacity * 1.5)))
        for name, old in self.arrays.items():
            replacement = np.empty((new_capacity, *old.shape[1:]), dtype=old.dtype)
            replacement[: self.count] = old[: self.count]
            self.arrays[name] = replacement
        self.capacity = new_capacity

    def append(
        self,
        *,
        centers: np.ndarray,
        normals: np.ndarray,
        radii: np.ndarray,
        domain_id: int | None,
        set_id: int,
        source_code: int,
        voxel_indices: np.ndarray,
        distribution_code: int,
        size_source_code: int,
        orientation_source_code: int,
        dip_only_density_evidence: bool,
        size_class: int,
        record_ref: int = -1,
        clipped_polygons: dict[int, np.ndarray] | None = None,
    ) -> slice:
        count = int(len(radii))
        self._ensure(count)
        target = slice(self.count, self.count + count)
        original_area = math.pi * np.square(radii, dtype=np.float64)
        self.arrays["ordinal"][target] = np.arange(self.count, self.count + count, dtype=np.uint64)
        self.arrays["center"][target] = centers
        self.arrays["normal"][target] = normals
        self.arrays["radius"][target] = radii
        self.arrays["original_area"][target] = original_area
        self.arrays["clipped_area"][target] = original_area
        self.arrays["domain_id"][target] = -1 if domain_id is None else domain_id
        self.arrays["set_id"][target] = set_id
        self.arrays["source_code"][target] = source_code
        self.arrays["voxel_index"][target] = voxel_indices
        self.arrays["clipped"][target] = False
        self.arrays["clipped_offsets"][target] = -1
        self.arrays["clipped_counts"][target] = 0
        self.arrays["distribution_code"][target] = distribution_code
        self.arrays["size_source_code"][target] = size_source_code
        self.arrays["orientation_source_code"][target] = orientation_source_code
        self.arrays["dip_only_density_evidence"][target] = dip_only_density_evidence
        self.arrays["record_ref"][target] = record_ref
        self.arrays["size_class"][target] = np.uint8(size_class)
        for local_index, polygon in (clipped_polygons or {}).items():
            absolute = self.count + int(local_index)
            polygon = np.asarray(polygon, dtype=np.float64).reshape((-1, 3))
            self.arrays["clipped"][absolute] = True
            self.arrays["clipped_offsets"][absolute] = self._clipped_point_count
            self.arrays["clipped_counts"][absolute] = len(polygon)
            self.arrays["clipped_area"][absolute] = M10ExplicitDFNGenerator._polygon_area(
                polygon, self.arrays["normal"][absolute]
            )
            if len(polygon):
                self._clipped_parts.append(polygon)
                self._clipped_point_count += len(polygon)
        self.count += count
        return target

    def finalize(self) -> dict[str, np.ndarray]:
        result = {name: values[: self.count] for name, values in self.arrays.items()}
        result["clipped_points"] = (
            np.concatenate(self._clipped_parts, axis=0)
            if self._clipped_parts
            else np.empty((0, 3), dtype=np.float64)
        )
        return result


class M10ExplicitDFNGenerator:
    """Generate explicit fracture discs independently for every cell and set."""

    def __init__(
        self,
        *,
        metadata: ParameterFieldMetadata,
        arrays: dict[str, np.ndarray],
        generation_domain: ModelBounds,
        size_models: Iterable[SizeModel],
        config: M10GenerationConfig,
        project_id: str,
        conditioned_observations: Iterable[ConditionedObservation] = (),
        deterministic_structures: Iterable[DeterministicStructure] = (),
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        actual_worker_count: int = 1,
    ) -> None:
        self.metadata = metadata
        self.arrays = arrays
        self.generation_domain = generation_domain
        self.size_models = {(item.domain_id, item.set_id): item for item in size_models}
        self.config = config
        self.project_id = project_id
        self.conditioned_observations = list(conditioned_observations)
        self.deterministic_structures = list(deterministic_structures)
        self.cancelled = cancelled
        self.progress = progress
        self.actual_worker_count = int(actual_worker_count)
        self._validate_inputs()

    @staticmethod
    def expected_fracture_area(size_model: SizeModel) -> float:
        """Return E[A] = pi E[R^2], never pi(E[R])^2."""
        value = math.pi * float(size_model.mean_squared_radius)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Size model must provide a positive finite E[R^2]")
        return value

    def estimate(self) -> tuple[float, int]:
        """Return expected fracture count and compact final storage bytes."""
        details = self.estimate_details()
        return float(details["expected_fractures"]), int(details["final_storage_bytes"])

    def estimate_details(self) -> dict[str, Any]:
        """Estimate count and peak memory components from authoritative array dtypes."""
        expected = 0.0
        expected_total = 0.0
        p32_explicit_area = 0.0
        p32_subgrid_area = 0.0
        cell_volume = float(np.prod(self.metadata.spacing))
        modeled = self.arrays["cell_state"] == CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
        domains = self.arrays["domain_id"]
        targets: list[dict[str, Any]] = []
        for set_id in self.metadata.set_ids:
            p32 = self.arrays[f"set_{set_id}_p32"]
            valid = modeled & np.isfinite(p32) & (p32 > 0.0)
            for raw_domain_id in np.unique(domains[valid]):
                domain_id = self._domain_value(int(raw_domain_id))
                size = self._size_model(domain_id, set_id)
                if size is not None:
                    mask = valid & (domains == raw_domain_id)
                    target_area = float(np.sum(p32[mask], dtype=np.float64) * cell_volume)
                    direction_valid = (
                        np.isfinite(self.arrays[f"set_{set_id}_dip_direction"])
                        & np.isfinite(self.arrays[f"set_{set_id}_dip"])
                        & np.isfinite(self.arrays[f"set_{set_id}_kappa"])
                        & (self.arrays[f"set_{set_id}_kappa"] > 0.0)
                    )
                    modelable_mask = mask & direction_valid
                    modelable_area = float(np.sum(p32[modelable_mask], dtype=np.float64) * cell_volume)
                    unresolved_orientation_area = target_area - modelable_area
                    contribution = modelable_area / self.expected_fracture_area(size)
                    thresholds = self._thresholds(size)
                    enabled = set(self.config.enabled_size_classes)
                    explicit_probability = sum(
                        item.probability for item in thresholds.budgets if item.size_class.name in enabled
                    )
                    explicit_area_share = sum(
                        item.area_share for item in thresholds.budgets if item.size_class.name in enabled
                    )
                    expected_total += contribution
                    expected += contribution * explicit_probability
                    p32_explicit_area += modelable_area * explicit_area_share
                    p32_subgrid_area += modelable_area * (1.0 - explicit_area_share)
                    targets.append(
                        {
                            "domain_id": domain_id,
                            "set_id": int(set_id),
                            "modelled_voxels": int(np.count_nonzero(mask)),
                            "orientation_supported_voxels": int(np.count_nonzero(modelable_mask)),
                            "p32_unresolved_orientation_area": unresolved_orientation_area,
                            "mean_target_p32": float(np.mean(p32[mask], dtype=np.float64)),
                            "voxel_volume": cell_volume,
                            "mean_squared_radius": float(size.mean_squared_radius),
                            "expected_fractures_total": contribution,
                            "expected_fractures": contribution * explicit_probability,
                            "small_medium_radius": thresholds.small_medium_radius,
                            "medium_large_radius": thresholds.medium_large_radius,
                            "small_medium_voxel_ratio": tuple(
                                thresholds.small_medium_radius / float(value) for value in self.metadata.spacing
                            ),
                            "medium_large_voxel_ratio": tuple(
                                thresholds.medium_large_radius / float(value) for value in self.metadata.spacing
                            ),
                            "degenerate_thresholds": thresholds.degenerate,
                            "classes": [
                                {
                                    "size_class": budget.size_class.name,
                                    "radius_range": (budget.lower, budget.upper),
                                    "generate": budget.size_class.name in enabled,
                                    "expected_count": contribution * budget.probability,
                                    "target_p32": (
                                        float(np.mean(p32[modelable_mask], dtype=np.float64)) * budget.area_share
                                        if np.any(modelable_mask)
                                        else 0.0
                                    ),
                                    "p32_share": budget.area_share,
                                }
                                for budget in thresholds.budgets
                            ],
                        }
                    )
        deterministic_count = sum(
            self.config.retain_outside_deterministic
            or self._disk_bounds_relation(
                np.asarray((item.center_x, item.center_y, item.center_z), dtype=float),
                dip_dir_dip_to_normal(item.dip_direction, item.dip), item.radius, self.generation_domain,
            ) != "OUTSIDE"
            for item in self.deterministic_structures
        )
        expected += len(self.conditioned_observations) + deterministic_count
        count = int(math.ceil(expected))
        bytes_per_fracture = sum(np.dtype(dtype).itemsize * (int(np.prod(tail)) if tail else 1) for dtype, tail in _ColumnarAccumulator._SPECS.values())
        fracture_array_bytes = count * bytes_per_fracture
        subgrid_array_bytes = int(len(self.metadata.set_ids) * np.prod(self.metadata.shape) * np.dtype(np.float32).itemsize)
        # Clipped polygons are ragged and input-dependent.  Do not hide omitted
        # fixed arrays behind an arbitrary global clipped-fracture percentage.
        clipped_geometry = 0
        multiscale_metadata_bytes = max(4 * 1024, len(targets) * 512)
        final_storage = fracture_array_bytes + subgrid_array_bytes + clipped_geometry + multiscale_metadata_bytes
        batch = min(max(1, count), self.config.fracture_batch_size)
        temporary_batch = batch * (3 * 8 * 3 + 8 * 4 + 16)
        metadata_bytes = multiscale_metadata_bytes
        # Cold first render includes the VTK runtime plus compact point/normal/radius buffers.
        render_memory = 85 * 1024**2 + count * 32
        # Streaming NPZ-to-ZIP uses bounded compressor buffers rather than a second full geometry copy.
        save_temporary = min(64 * 1024**2, count * 48 + 1 * 1024**2)
        peak_generation = int(
            fracture_array_bytes * 1.12 + subgrid_array_bytes + temporary_batch + metadata_bytes
        )
        parameter_field_bytes = int(sum(value.nbytes for value in self.arrays.values()))
        active_workers = min(self.config.worker_count, self.config.realization_count)
        return {
            "expected_fractures": expected,
            "expected_fractures_total": expected_total + len(self.conditioned_observations) + deterministic_count,
            "p32_explicit_target_area": p32_explicit_area,
            "p32_subgrid_area": p32_subgrid_area,
            "modelled_voxels": int(np.count_nonzero(modeled)),
            "voxel_volume": cell_volume,
            "targets": targets,
            "bytes_per_unclipped_fracture": bytes_per_fracture,
            "fracture_array_bytes": int(fracture_array_bytes),
            "subgrid_array_bytes": subgrid_array_bytes,
            "clipped_geometry_bytes": clipped_geometry,
            "multiscale_metadata_bytes": multiscale_metadata_bytes,
            "final_storage_bytes": int(final_storage),
            "peak_generation_bytes": peak_generation,
            "preview_render_bytes": int(render_memory),
            "project_save_temporary_bytes": int(save_temporary),
            "realization_count": self.config.realization_count,
            "all_realizations_final_bytes": int(final_storage * self.config.realization_count),
            "parameter_field_bytes": parameter_field_bytes,
            "active_worker_count": active_workers,
            "all_realizations_peak_bytes": int(
                final_storage * self.config.realization_count
                + peak_generation * active_workers
                + parameter_field_bytes * max(0, active_workers - 1)
            ),
        }

    def generate(self, realization_index: int) -> M10Realization:
        """Generate one complete realization transactionally in memory."""
        seed = self.config.seed_for(realization_index)
        rng = np.random.default_rng(seed)
        input_hash = self._input_hash()
        realization_id = f"r{realization_index + 1:04d}-s{seed}"
        generated: list[_GeneratedFracture] = []
        budget_deductions: dict[tuple[int, int, int, int], float] = defaultdict(float)
        over_conditioned: list[dict[str, Any]] = []
        warnings: list[str] = []
        skipped: list[dict[str, Any]] = []

        self._check_cancelled()
        self._generate_conditioned(
            rng, realization_id, generated, budget_deductions, warnings
        )
        self._generate_deterministic(
            realization_id, generated, budget_deductions, warnings
        )
        estimate = self.estimate_details()
        initial_capacity = min(
            self.config.maximum_fractures,
            max(len(generated) + 16, int(math.ceil(estimate["expected_fractures"] * 1.02)) + 1024),
        )
        columns = _ColumnarAccumulator(initial_capacity, self.config.maximum_fractures)
        record_ids: list[str] = []
        self._append_generated_objects(columns, generated, record_ids)

        shape = self.metadata.shape
        cell_volume = float(np.prod(self.metadata.spacing))
        states = self.arrays["cell_state"]
        domains = self.arrays["domain_id"]
        modeled = states == CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
        total_work = max(1, int(math.ceil(estimate["expected_fractures"])))
        target_area_total = 0.0
        expected_count_total = 0.0
        summaries: dict[tuple[int | None, int], M10DomainSetSummary] = {}
        subgrid_arrays = {
            int(set_id): np.zeros(shape, dtype=np.float32) for set_id in self.metadata.set_ids
        }

        for set_id in self.metadata.set_ids:
            p32_values = self.arrays[f"set_{set_id}_p32"]
            base_valid = modeled & np.isfinite(p32_values) & (p32_values > 0.0)
            for raw_domain_id in np.unique(domains[base_valid]):
                self._check_cancelled()
                domain_id = self._domain_value(int(raw_domain_id))
                key = (domain_id, set_id)
                summary = summaries.setdefault(key, M10DomainSetSummary(domain_id=domain_id, set_id=set_id))
                domain_mask = base_valid & (domains == raw_domain_id)
                flat_indices = np.flatnonzero(domain_mask)
                target_areas = np.asarray(p32_values.flat[flat_indices], dtype=np.float64) * cell_volume
                target_area = float(target_areas.sum())
                target_area_total += target_area
                summary.target_area += target_area
                size = self._size_model(domain_id, set_id)
                if size is None:
                    summary.status = "INVALID_SIZE_MODEL"
                    summary.skipped_cell_count += len(flat_indices)
                    skipped.append(
                        {"domain_id": domain_id, "set_id": set_id, "cell_count": len(flat_indices), "reason": summary.status}
                    )
                    continue
                deductions = np.zeros(len(flat_indices), dtype=np.float64)
                lookup = {int(flat): position for position, flat in enumerate(flat_indices)}
                for deduction_key, value in budget_deductions.items():
                    if deduction_key[3] != set_id:
                        continue
                    flat = int(np.ravel_multi_index(deduction_key[:3], shape))
                    position = lookup.get(flat)
                    if position is not None:
                        deductions[position] += value
                for fracture in generated:
                    if fracture.set_id != set_id or fracture.domain_id != domain_id:
                        continue
                    if fracture.source == M10FractureSource.CONDITIONED_OBSERVATION.value:
                        summary.conditioned_area_deducted += fracture.original_area
                    elif (
                        fracture.source == M10FractureSource.DETERMINISTIC_STRUCTURE.value
                        and self.config.deterministic_structures_reduce_budget
                    ):
                        summary.deterministic_area_deducted += fracture.original_area
                remaining = target_areas - deductions
                over = remaining <= 0.0
                if np.any(over):
                    summary.over_conditioned_cell_count += int(np.count_nonzero(over))
                    for position in np.flatnonzero(over):
                        index = tuple(int(value) for value in np.unravel_index(flat_indices[position], shape))
                        over_conditioned.append(
                            {
                                "voxel": index,
                                "domain_id": domain_id,
                                "set_id": set_id,
                                "target_area": float(target_areas[position]),
                                "deducted_area": float(deductions[position]),
                            }
                        )
                active = ~over
                if not np.any(active):
                    continue
                active_flat = flat_indices[active]
                active_remaining = np.asarray(remaining[active], dtype=np.float64)
                expected_counts = active_remaining / self.expected_fracture_area(size)
                dd = np.asarray(self.arrays[f"set_{set_id}_dip_direction"].flat[active_flat], dtype=np.float64)
                dip = np.asarray(self.arrays[f"set_{set_id}_dip"].flat[active_flat], dtype=np.float64)
                kappa = np.asarray(self.arrays[f"set_{set_id}_kappa"].flat[active_flat], dtype=np.float64)
                orientation_valid = np.isfinite(dd) & np.isfinite(dip) & np.isfinite(kappa) & (kappa > 0.0)
                if not np.all(orientation_valid):
                    invalid_count = int(np.count_nonzero(~orientation_valid))
                    unresolved_area = float(active_remaining[~orientation_valid].sum(dtype=np.float64))
                    summary.p32_unresolved_orientation += unresolved_area / self.generation_domain.volume
                    summary.status = "INSUFFICIENT_ORIENTATION_DATA"
                    summary.skipped_cell_count += invalid_count
                    skipped.append(
                        {"domain_id": domain_id, "set_id": set_id, "cell_count": invalid_count, "reason": summary.status}
                    )
                    active_flat = active_flat[orientation_valid]
                    active_remaining = active_remaining[orientation_valid]
                    expected_counts = expected_counts[orientation_valid]
                    dd, dip, kappa = dd[orientation_valid], dip[orientation_valid], kappa[orientation_valid]
                if len(active_flat) == 0:
                    continue
                thresholds = self._thresholds(size)
                enabled_classes = set(self.config.enabled_size_classes)
                summary.small_medium_radius = thresholds.small_medium_radius
                summary.medium_large_radius = thresholds.medium_large_radius
                summary.size_threshold_mode = thresholds.mode.value
                summary.size_class_budgets = {
                    budget.size_class.name: {
                        "probability": budget.probability,
                        "area_share": budget.area_share,
                        "conditional_mean_squared_radius": budget.conditional_mean_squared_radius,
                        "generated": budget.size_class.name in enabled_classes,
                    }
                    for budget in thresholds.budgets
                }
                summary.target_mean_dip_direction = float(np.mean(dd))
                summary.target_mean_dip = float(np.mean(dip))
                summary.target_kappa = float(np.mean(kappa))
                parameters = np.column_stack((dd, dip, kappa))
                unique_parameters, inverse = np.unique(parameters, axis=0, return_inverse=True)
                for budget in thresholds.budgets:
                    class_target_areas = active_remaining * budget.area_share
                    class_target_area = float(class_target_areas.sum())
                    class_expected_counts = expected_counts * budget.probability
                    class_expected_total = float(class_expected_counts.sum())
                    generated_class = budget.size_class.name in enabled_classes
                    summary.size_class_budgets[budget.size_class.name].update(
                        {"target_area": class_target_area, "expected_count": class_expected_total}
                    )
                    if not generated_class:
                        flat_target = subgrid_arrays[int(set_id)].ravel()
                        flat_target[active_flat] += np.asarray(
                            class_target_areas / cell_volume, dtype=np.float32
                        )
                        summary.p32_subgrid += class_target_area / self.generation_domain.volume
                        continue
                    summary.p32_explicit_target += class_target_area / self.generation_domain.volume
                    expected_count_total += class_expected_total
                    summary.target_count_expectation += class_expected_total
                    if budget.probability <= 0.0:
                        continue
                    poisson_counts = np.asarray(rng.poisson(class_expected_counts), dtype=np.int64)
                    count = int(poisson_counts.sum())
                    if columns.count + count > self.config.maximum_fractures:
                        raise MemoryError("Generated fracture count exceeds the configured safety limit")
                    if count == 0:
                        continue
                    repeated_flat = np.repeat(active_flat, poisson_counts)
                    repeated_parameters = np.repeat(np.arange(len(active_flat), dtype=np.int64), poisson_counts)
                    normals = np.empty((count, 3), dtype=np.float64)
                    repeated_groups = inverse[repeated_parameters]
                    for group_index, (group_dd, group_dip, group_kappa) in enumerate(unique_parameters):
                        group_positions = np.flatnonzero(repeated_groups == group_index)
                        sampled = fisher_sample(
                            float(group_dd), float(group_dip), float(group_kappa), rng=rng,
                            n_samples=len(group_positions),
                        )
                        normals[group_positions] = np.asarray(sampled, dtype=np.float64).reshape((-1, 3))
                    if not np.all(np.isfinite(normals)) or not np.allclose(
                        np.linalg.norm(normals, axis=1), 1.0, rtol=1e-10, atol=1e-12
                    ):
                        raise ValueError("Fisher sampler returned invalid normal vectors")
                    radii = sample_truncated_class(size, budget, count, rng)
                    voxel_indices = np.column_stack(np.unravel_index(repeated_flat, shape)).astype(np.int32, copy=False)
                    origin = np.asarray(self.metadata.origin, dtype=np.float64)
                    spacing = np.asarray(self.metadata.spacing, dtype=np.float64)
                    centers = origin + (voxel_indices + rng.random((count, 3))) * spacing
                    for start in range(0, count, self.config.fracture_batch_size):
                        self._check_cancelled()
                        stop = min(count, start + self.config.fracture_batch_size)
                        target = self._append_columnar_batch(
                            columns, centers=centers[start:stop], normals=normals[start:stop],
                            radii=radii[start:stop], domain_id=domain_id, set_id=set_id,
                            source=M10FractureSource.STOCHASTIC, voxel_indices=voxel_indices[start:stop],
                            size=size, orientation_source="domain_set_fisher_model",
                            dip_only_density_evidence=self._dip_only_evidence(domain_id, set_id),
                            size_class=budget.size_class,
                        )
                        summary.stochastic_original_area += float(columns.arrays["original_area"][target].sum())
                        summary.stochastic_clipped_area += float(columns.arrays["clipped_area"][target].sum())
                        summary.fracture_count += stop - start
                        if self.progress:
                            self.progress(min(total_work, columns.count), total_work,
                                f"Generating realization {realization_index + 1}: {columns.count:,} fractures")
                summary.p32_conservation_error = (
                    float(np.maximum(remaining[active], 0.0).sum()) / self.generation_domain.volume
                    - summary.p32_explicit_target
                    - summary.p32_subgrid
                    - summary.p32_unresolved_orientation
                )

        self._check_cancelled()
        arrays = columns.finalize()
        for set_id, values in subgrid_arrays.items():
            arrays[f"p32_subgrid_set_{set_id}"] = values
        self._populate_generated_orientation_summaries(summaries, arrays)
        for state_name, state_code in (
            ("OUTSIDE_MODEL", CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL]),
            ("EXCAVATION", CELL_STATE_CODES[VoxelCellState.EXCAVATION]),
            ("NO_DATA", CELL_STATE_CODES[VoxelCellState.NO_DATA]),
            ("TRUE_ZERO", CELL_STATE_CODES[VoxelCellState.TRUE_ZERO]),
        ):
            state_count = int(np.count_nonzero(states == state_code))
            if state_count:
                skipped.append({"reason": state_name, "cell_count": state_count})
        source_values = arrays["source_code"]
        radii = arrays["radius"]
        original_area = float(arrays["original_area"].sum()) if columns.count else 0.0
        clipped_area = float(arrays["clipped_area"].sum()) if columns.count else 0.0
        generation_volume = self.generation_domain.volume
        target_p32 = target_area_total / generation_volume if generation_volume > 0 else 0.0
        p32_subgrid = sum(item.p32_subgrid for item in summaries.values())
        p32_unresolved_orientation = sum(item.p32_unresolved_orientation for item in summaries.values())
        p32_explicit_target = max(0.0, target_p32 - p32_subgrid - p32_unresolved_orientation)
        quality = M10QualitySummary(
            target_fracture_count_expectation=(
                expected_count_total
                + int(np.count_nonzero(source_values == SOURCE_CODES[M10FractureSource.CONDITIONED_OBSERVATION.value]))
                + int(np.count_nonzero(source_values == SOURCE_CODES[M10FractureSource.DETERMINISTIC_STRUCTURE.value]))
            ),
            fracture_count=columns.count,
            stochastic_count=int(np.count_nonzero(source_values == SOURCE_CODES[M10FractureSource.STOCHASTIC.value])),
            conditioned_count=int(np.count_nonzero(source_values == SOURCE_CODES[M10FractureSource.CONDITIONED_OBSERVATION.value])),
            deterministic_count=int(np.count_nonzero(source_values == SOURCE_CODES[M10FractureSource.DETERMINISTIC_STRUCTURE.value])),
            target_p32=target_p32,
            generated_original_p32=original_area / generation_volume if generation_volume > 0 else 0.0,
            generated_clipped_p32=clipped_area / generation_volume if generation_volume > 0 else 0.0,
            p32_explicit_target=p32_explicit_target,
            p32_subgrid=p32_subgrid,
            p32_unresolved_orientation=p32_unresolved_orientation,
            p32_target_conservation_error=(
                target_p32 - p32_explicit_target - p32_subgrid - p32_unresolved_orientation
            ),
            mean_radius=float(np.mean(radii)) if radii.size else None,
            radius_q05=float(np.quantile(radii, 0.05)) if radii.size else None,
            radius_q50=float(np.quantile(radii, 0.50)) if radii.size else None,
            radius_q95=float(np.quantile(radii, 0.95)) if radii.size else None,
            domain_set=sorted(summaries.values(), key=lambda item: (-1 if item.domain_id is None else item.domain_id, item.set_id)),
            warnings=warnings,
            model_sources={
                f"domain_{domain_id}_set_{set_id}": {
                    "density": "m9_parameter_field",
                    "orientation": "domain_set_fisher_model",
                    "size": model.source.value,
                    "distribution": model.distribution_type,
                    "dip_only_density_evidence": bool(model.provenance.get("dip_only_density_evidence", False)),
                }
                for (domain_id, set_id), model in self.size_models.items()
            },
            skipped_regions=skipped,
            over_conditioned_regions=over_conditioned,
        )
        return M10Realization(
            realization_id=realization_id,
            realization_index=realization_index,
            seed=seed,
            config_hash=input_hash,
            quality=quality,
            geometry_arrays=arrays,
            provenance={
                "generator": "m10_parameter_field_poisson",
                "generator_version": GENERATOR_VERSION,
                "p32_count_conversion": "lambda=P32*voxel_volume/(pi*E[R^2])",
                "seed_strategy": self.config.seed_strategy,
                "analysis_field_hash": input_hash,
                "validation_used_for_generation": False,
                "conditioned_area_deducted_from_random_budget": True,
                "deterministic_structures_reduce_budget": self.config.deterministic_structures_reduce_budget,
                "worker_count": self.actual_worker_count,
                "requested_worker_count": self.config.worker_count,
                "parallelism": (
                    "multiprocess_by_realization" if self.actual_worker_count > 1 else "vectorized_single_process"
                ),
                "geometry_format": "columnar-multiscale-v1",
                "size_threshold_method": "area-weighted-cdf-1",
                "size_threshold_mode": self.config.size_threshold_mode,
                "enabled_size_classes": list(self.config.enabled_size_classes),
                "size_class_codes": {"0": "UNKNOWN", "1": "SMALL", "2": "MEDIUM", "3": "LARGE"},
                "p32_contract": (
                    "P32_target=P32_explicit_target+P32_subgrid+P32_unresolved_orientation; "
                    "M11 intersection contract applies only to modelled orientation-supported P32"
                ),
                "source_codes": {str(code): name for code, name in SOURCE_NAMES.items()},
                "distribution_codes": {str(int(code)): name for name, code in DISTRIBUTION_CODES.items()},
                "size_source_codes": {str(int(code)): name for name, code in SIZE_SOURCE_CODES.items()},
                "orientation_source_codes": {str(int(code)): name for name, code in ORIENTATION_CODES.items()},
                "record_ids": record_ids,
            },
        )

    def _validate_inputs(self) -> None:
        required = {"cell_state", "domain_id"}
        for set_id in self.metadata.set_ids:
            required.update(
                {
                    f"set_{set_id}_p32",
                    f"set_{set_id}_dip_direction",
                    f"set_{set_id}_dip",
                    f"set_{set_id}_kappa",
                }
            )
        missing = sorted(required - self.arrays.keys())
        if missing:
            raise ValueError(f"M9 parameter field is missing required arrays: {', '.join(missing)}")
        for size in self.size_models.values():
            if size.source == SizeModelSource.EXPERIMENTAL and not self.config.experimental_size_models_confirmed:
                raise ValueError("EXPERIMENTAL size models require explicit user confirmation")
            if size.converged is False or size.fit_status in {"failed", "invalid"}:
                raise ValueError(f"Size model for set {size.set_id} is not valid for generation")
            self.expected_fracture_area(size)

    def _generate_conditioned(
        self,
        rng: np.random.Generator,
        realization_id: str,
        generated: list[_GeneratedFracture],
        deductions: dict[tuple[int, int, int, int], float],
        warnings: list[str],
    ) -> None:
        if not self.config.condition_calibration_observations:
            return
        seen: set[str] = set()
        for observation in self.conditioned_observations:
            self._check_cancelled()
            if observation.observation_record_id in seen:
                continue
            seen.add(observation.observation_record_id)
            index = self._point_to_cell(observation.position)
            size = self._size_model(observation.domain_id, observation.set_id)
            if index is None or size is None:
                warnings.append(f"Conditioning skipped for {observation.observation_record_id}: outside grid or missing size model")
                continue
            radius = float(self._sample_radii(size, 1, rng)[0])
            normal = dip_dir_dip_to_normal(observation.dip_direction, observation.dip)
            u, v = self._plane_basis(normal)
            offset_radius = radius * math.sqrt(float(rng.random()))
            angle = 2.0 * math.pi * float(rng.random())
            offset = offset_radius * (math.cos(angle) * u + math.sin(angle) * v)
            center = np.asarray(observation.position, dtype=float) - offset
            fracture = self._make_fracture(
                realization_id=realization_id,
                ordinal=len(generated),
                center=center,
                normal=normal,
                radius=radius,
                domain_id=observation.domain_id,
                set_id=observation.set_id,
                source=M10FractureSource.CONDITIONED_OBSERVATION,
                observation_record_id=observation.observation_record_id,
                size=size,
                orientation_source="observed_full_orientation",
                dip_only_density_evidence=self._dip_only_evidence(observation.domain_id, observation.set_id),
                voxel_index=index,
            )
            generated.append(fracture)
            deductions[(*index, observation.set_id)] += fracture.original_area

    def _generate_deterministic(
        self,
        realization_id: str,
        generated: list[_GeneratedFracture],
        deductions: dict[tuple[int, int, int, int], float],
        warnings: list[str],
    ) -> None:
        for structure in self.deterministic_structures:
            center = np.asarray((structure.center_x, structure.center_y, structure.center_z), dtype=float)
            normal = dip_dir_dip_to_normal(structure.dip_direction, structure.dip)
            relation = self._disk_bounds_relation(center, normal, structure.radius, self.generation_domain)
            if relation == "OUTSIDE":
                warning = (
                    f"Deterministic structure {structure.structure_id} (set_id={structure.set_id}) is fully outside "
                    f"the DFN Generation Domain; center={tuple(float(v) for v in center)}, "
                    f"bounds={self.generation_domain.model_dump()}."
                )
                warnings.append(warning)
                if not self.config.retain_outside_deterministic:
                    continue
            index = self._point_to_cell(center) or (-1, -1, -1)
            pseudo_size = SizeModel(
                domain_id=structure.domain_id,
                set_id=structure.set_id or 0,
                distribution_type="fixed",
                parameters={"radius": structure.radius},
                min_radius=structure.radius,
                max_radius=structure.radius,
                mean_radius=structure.radius,
                mean_squared_radius=structure.radius**2,
                source=SizeModelSource.USER_DEFINED,
            )
            fracture = self._make_fracture(
                realization_id=realization_id,
                ordinal=len(generated),
                center=center,
                normal=normal,
                radius=structure.radius,
                domain_id=structure.domain_id,
                set_id=structure.set_id if structure.set_id is not None else -1,
                source=M10FractureSource.DETERMINISTIC_STRUCTURE,
                observation_record_id=structure.structure_id,
                size=pseudo_size,
                orientation_source="deterministic_structure",
                dip_only_density_evidence=False,
                voxel_index=index,
            )
            generated.append(fracture)
            if self.config.deterministic_structures_reduce_budget and index != (-1, -1, -1):
                if structure.set_id is None:
                    warnings.append(
                        f"Deterministic structure {structure.structure_id} was not deducted because set_id is missing"
                    )
                else:
                    deductions[(*index, structure.set_id)] += fracture.original_area

    @classmethod
    def _disk_bounds_relation(
        cls, center: np.ndarray, normal: np.ndarray, radius: float, bounds: ModelBounds
    ) -> str:
        """Classify a circular disc against an axis-aligned box using its plane cross-section."""
        if cls._disk_is_contained(center, normal, radius, bounds):
            return "INSIDE"
        lower = np.asarray((bounds.x_min, bounds.y_min, bounds.z_min), dtype=float)
        upper = np.asarray((bounds.x_max, bounds.y_max, bounds.z_max), dtype=float)
        center = np.asarray(center, dtype=float)
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)
        if np.all(center >= lower) and np.all(center <= upper):
            return "INTERSECTS"
        corners = np.asarray(
            [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])],
            dtype=float,
        )
        signed = (corners - center) @ normal
        edge_pairs = (
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
            (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        )
        points: list[np.ndarray] = [corners[i] for i in range(8) if abs(signed[i]) <= 1e-10]
        for first, second in edge_pairs:
            if signed[first] * signed[second] < 0.0:
                fraction = signed[first] / (signed[first] - signed[second])
                points.append(corners[first] + fraction * (corners[second] - corners[first]))
        if not points:
            return "OUTSIDE"
        u, v = cls._plane_basis(normal)
        projected = np.asarray([[(point - center) @ u, (point - center) @ v] for point in points])
        angles = np.arctan2(projected[:, 1] - projected[:, 1].mean(), projected[:, 0] - projected[:, 0].mean())
        polygon = projected[np.argsort(angles)]
        following = np.roll(polygon, -1, axis=0)
        crosses = polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0]
        if np.all(crosses >= -1e-10) or np.all(crosses <= 1e-10):
            minimum = 0.0
        else:
            minimum = math.inf
            for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
                edge = end - start
                fraction = float(np.clip(-np.dot(start, edge) / max(np.dot(edge, edge), 1e-30), 0.0, 1.0))
                minimum = min(minimum, float(np.linalg.norm(start + fraction * edge)))
        return "INTERSECTS" if minimum <= float(radius) + 1e-10 else "OUTSIDE"

    def _make_fracture(
        self,
        *,
        realization_id: str,
        ordinal: int,
        center: np.ndarray,
        normal: np.ndarray,
        radius: float,
        domain_id: int | None,
        set_id: int,
        source: M10FractureSource,
        observation_record_id: str,
        size: SizeModel,
        orientation_source: str,
        dip_only_density_evidence: bool,
        voxel_index: tuple[int, int, int],
    ) -> _GeneratedFracture:
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)
        vertices = self._disk_vertices(center, normal, radius)
        original_area = math.pi * radius**2
        if self._disk_is_contained(center, normal, radius, self.generation_domain):
            clipped = vertices
            clipped_area = original_area
        else:
            clipped = self._clip_polygon_to_bounds(vertices, self.generation_domain)
            clipped_area = self._polygon_area(clipped, normal)
        fracture_id = f"{realization_id}:f{ordinal:09d}"
        return _GeneratedFracture(
            fracture_id=fracture_id,
            center=np.asarray(center, dtype=float),
            normal=normal,
            radius=radius,
            original_area=original_area,
            clipped_area=clipped_area,
            vertices=clipped,
            domain_id=domain_id,
            set_id=set_id,
            source=source.value,
            observation_record_id=observation_record_id,
            distribution=size.distribution_type,
            size_source=size.source.value,
            orientation_source=orientation_source,
            dip_only_density_evidence=dip_only_density_evidence,
            voxel_index=voxel_index,
            size_class=int(self._classify_radius(size, radius)),
        )

    def _sample_radii(self, model: SizeModel, count: int, rng: np.random.Generator) -> np.ndarray:
        low, high = float(model.min_radius), float(model.max_radius)
        kind = model.distribution_type
        if count <= 0:
            return np.empty(0, dtype=float)
        if kind == "fixed":
            values = np.full(count, float(model.parameters.get("radius", model.mean_radius)))
        elif kind == "uniform":
            values = rng.uniform(low, high, count)
        elif kind == "truncated_lognormal":
            values = self._rejection_sample(count, rng, lambda n: rng.lognormal(model.parameters["mu"], model.parameters["sigma"], n), low, high)
        elif kind == "truncated_power_law":
            exponent = float(model.parameters["exponent"])
            u = rng.random(count)
            if abs(exponent - 1.0) < 1e-12:
                values = low * (high / low) ** u
            else:
                power = 1.0 - exponent
                values = (u * (high**power - low**power) + low**power) ** (1.0 / power)
        elif kind == "truncated_exponential":
            rate = float(model.parameters["rate"])
            u = rng.random(count)
            span_probability = 1.0 - math.exp(-rate * (high - low))
            values = low - np.log1p(-u * span_probability) / rate
        else:
            raise ValueError(f"Unsupported M10 size distribution: {kind}")
        if np.any(~np.isfinite(values)):
            raise ValueError(f"Invalid samples from size distribution {kind}")
        return np.clip(values, low, high)

    @staticmethod
    def _rejection_sample(count: int, rng: np.random.Generator, sampler: Callable[[int], np.ndarray], low: float, high: float) -> np.ndarray:
        accepted: list[float] = []
        attempts = 0
        while len(accepted) < count and attempts < 10_000:
            batch = sampler(max(16, 2 * (count - len(accepted))))
            accepted.extend(float(value) for value in batch if low <= value <= high)
            attempts += len(batch)
        if len(accepted) < count:
            raise ValueError("Truncated size sampling failed to obtain enough valid radii")
        return np.asarray(accepted[:count], dtype=float)

    def _sample_center(self, index: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
        origin = np.asarray(self.metadata.origin, dtype=float)
        spacing = np.asarray(self.metadata.spacing, dtype=float)
        return origin + (np.asarray(index, dtype=float) + rng.random(3)) * spacing

    def _sample_centers(
        self, index: tuple[int, int, int], count: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample one batch of voxel-local centers with the legacy RNG ordering."""
        origin = np.asarray(self.metadata.origin, dtype=float)
        spacing = np.asarray(self.metadata.spacing, dtype=float)
        return origin + (np.asarray(index, dtype=float) + rng.random((count, 3))) * spacing

    def _append_generated_objects(
        self,
        columns: _ColumnarAccumulator,
        generated: list[_GeneratedFracture],
        record_ids: list[str],
    ) -> None:
        """Convert the small conditioned/deterministic prefix into compact columns."""
        for fracture in generated:
            record_ref = -1
            if fracture.observation_record_id:
                record_ref = len(record_ids)
                record_ids.append(fracture.observation_record_id)
            clipped = not math.isclose(fracture.clipped_area, fracture.original_area, rel_tol=1e-12, abs_tol=1e-12)
            columns.append(
                centers=np.asarray(fracture.center, dtype=float).reshape((1, 3)),
                normals=np.asarray(fracture.normal, dtype=float).reshape((1, 3)),
                radii=np.asarray([fracture.radius], dtype=float),
                domain_id=fracture.domain_id,
                set_id=fracture.set_id,
                source_code=int(SOURCE_CODES[fracture.source]),
                voxel_indices=np.asarray([fracture.voxel_index], dtype=np.int32),
                distribution_code=int(DISTRIBUTION_CODES.get(fracture.distribution, np.uint8(255))),
                size_source_code=int(SIZE_SOURCE_CODES.get(fracture.size_source, np.uint8(255))),
                orientation_source_code=int(ORIENTATION_CODES.get(fracture.orientation_source, np.uint8(255))),
                dip_only_density_evidence=fracture.dip_only_density_evidence,
                record_ref=record_ref,
                size_class=fracture.size_class,
                clipped_polygons={0: fracture.vertices} if clipped else None,
            )

    def _append_columnar_batch(
        self,
        columns: _ColumnarAccumulator,
        *,
        centers: np.ndarray,
        normals: np.ndarray,
        radii: np.ndarray,
        domain_id: int | None,
        set_id: int,
        source: M10FractureSource,
        voxel_index: tuple[int, int, int] | None = None,
        voxel_indices: np.ndarray | None = None,
        size: SizeModel,
        orientation_source: str,
        dip_only_density_evidence: bool,
        size_class: SizeClass = SizeClass.UNKNOWN,
    ) -> slice:
        """Append a vectorized fracture batch, materializing polygons only at boundaries."""
        normals = np.asarray(normals, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        lower = np.asarray(
            (self.generation_domain.x_min, self.generation_domain.y_min, self.generation_domain.z_min), dtype=float
        )
        upper = np.asarray(
            (self.generation_domain.x_max, self.generation_domain.y_max, self.generation_domain.z_max), dtype=float
        )
        extents = radii[:, None] * np.sqrt(np.maximum(0.0, 1.0 - np.square(normals)))
        contained = np.all((centers - extents >= lower) & (centers + extents <= upper), axis=1)
        clipped_polygons: dict[int, np.ndarray] = {}
        for local_index in np.flatnonzero(~contained):
            self._check_cancelled()
            vertices = self._disk_vertices(centers[local_index], normals[local_index], float(radii[local_index]))
            clipped_polygons[int(local_index)] = self._clip_polygon_to_bounds(vertices, self.generation_domain)
        if voxel_indices is None:
            if voxel_index is None:
                raise ValueError("voxel_index or voxel_indices is required")
            voxel_indices = np.broadcast_to(np.asarray(voxel_index, dtype=np.int32), (len(radii), 3))
        return columns.append(
            centers=centers,
            normals=normals,
            radii=radii,
            domain_id=domain_id,
            set_id=set_id,
            source_code=int(SOURCE_CODES[source.value]),
            voxel_indices=voxel_indices,
            distribution_code=int(DISTRIBUTION_CODES.get(size.distribution_type, np.uint8(255))),
            size_source_code=int(SIZE_SOURCE_CODES.get(size.source.value, np.uint8(255))),
            orientation_source_code=int(ORIENTATION_CODES.get(orientation_source, np.uint8(255))),
            dip_only_density_evidence=dip_only_density_evidence,
            size_class=int(size_class),
            clipped_polygons=clipped_polygons,
        )

    def _point_to_cell(self, point: tuple[float, float, float] | np.ndarray) -> tuple[int, int, int] | None:
        relative = (np.asarray(point, dtype=float) - np.asarray(self.metadata.origin, dtype=float)) / np.asarray(self.metadata.spacing, dtype=float)
        index = tuple(int(math.floor(value)) for value in relative)
        if any(value < 0 or value >= self.metadata.shape[axis] for axis, value in enumerate(index)):
            return None
        return index

    def _size_model(self, domain_id: int | None, set_id: int) -> SizeModel | None:
        return self.size_models.get((domain_id, set_id)) or self.size_models.get((None, set_id))

    def _thresholds(self, model: SizeModel) -> SizeThresholdResult:
        return calculate_thresholds(
            model,
            mode=self.config.size_threshold_mode,
            small_area_share=self.config.small_area_share,
            medium_large_cumulative_share=self.config.medium_large_cumulative_share,
            manual_small_medium_radius=self.config.manual_small_medium_radius,
            manual_medium_large_radius=self.config.manual_medium_large_radius,
        )

    def _classify_radius(self, model: SizeModel, radius: float) -> SizeClass:
        return classify_radius(radius, self._thresholds(model))

    def _dip_only_evidence(self, domain_id: int | None, set_id: int) -> bool:
        model = self._size_model(domain_id, set_id)
        if model is None:
            return False
        return bool(model.provenance.get("dip_only_density_evidence", False))

    @staticmethod
    def _domain_value(value: int) -> int | None:
        return None if value < 0 else value

    @staticmethod
    def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
        reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(normal, reference)
        u /= np.linalg.norm(u)
        return u, np.cross(normal, u)

    def _disk_vertices(self, center: np.ndarray, normal: np.ndarray, radius: float) -> np.ndarray:
        u, v = self._plane_basis(normal)
        theta = np.linspace(0.0, 2.0 * math.pi, self.config.disk_sides, endpoint=False)
        return center + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)

    @staticmethod
    def _disk_is_contained(center: np.ndarray, normal: np.ndarray, radius: float, bounds: ModelBounds) -> bool:
        """Test exact axis-aligned extents of a planar circular disc."""
        normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
        extents = radius * np.sqrt(np.maximum(0.0, 1.0 - normal**2))
        lower = np.asarray((bounds.x_min, bounds.y_min, bounds.z_min), dtype=float)
        upper = np.asarray((bounds.x_max, bounds.y_max, bounds.z_max), dtype=float)
        return bool(np.all(center - extents >= lower) and np.all(center + extents <= upper))

    @classmethod
    def _clip_polygon_to_bounds(cls, vertices: np.ndarray, bounds: ModelBounds) -> np.ndarray:
        polygon = np.asarray(vertices, dtype=float)
        planes = (
            (0, bounds.x_min, True), (0, bounds.x_max, False),
            (1, bounds.y_min, True), (1, bounds.y_max, False),
            (2, bounds.z_min, True), (2, bounds.z_max, False),
        )
        for axis, value, keep_greater in planes:
            if len(polygon) == 0:
                break
            output: list[np.ndarray] = []
            previous = polygon[-1]
            previous_inside = previous[axis] >= value - 1e-10 if keep_greater else previous[axis] <= value + 1e-10
            for current in polygon:
                current_inside = current[axis] >= value - 1e-10 if keep_greater else current[axis] <= value + 1e-10
                if current_inside != previous_inside:
                    denominator = current[axis] - previous[axis]
                    fraction = 0.0 if abs(denominator) < 1e-15 else (value - previous[axis]) / denominator
                    output.append(previous + fraction * (current - previous))
                if current_inside:
                    output.append(current)
                previous, previous_inside = current, current_inside
            polygon = np.asarray(output, dtype=float).reshape((-1, 3)) if output else np.empty((0, 3), dtype=float)
        return polygon

    @staticmethod
    def _polygon_area(vertices: np.ndarray, normal: np.ndarray) -> float:
        if len(vertices) < 3:
            return 0.0
        cross_sum = np.zeros(3, dtype=float)
        for current, following in zip(vertices, np.roll(vertices, -1, axis=0)):
            cross_sum += np.cross(current, following)
        return abs(float(np.dot(cross_sum, normal))) * 0.5

    @staticmethod
    def _conditioned_deduction(generated: list[_GeneratedFracture], index: tuple[int, int, int], set_id: int) -> float:
        return sum(item.original_area for item in generated if item.source == M10FractureSource.CONDITIONED_OBSERVATION and item.voxel_index == index and item.set_id == set_id)

    @staticmethod
    def _deterministic_deduction(generated: list[_GeneratedFracture], index: tuple[int, int, int], set_id: int) -> float:
        return sum(item.original_area for item in generated if item.source == M10FractureSource.DETERMINISTIC_STRUCTURE and item.voxel_index == index and item.set_id == set_id)

    @staticmethod
    def _populate_generated_orientation_summaries(
        summaries: dict[tuple[int | None, int], M10DomainSetSummary],
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Add axial mean-orientation diagnostics directly from compact columns."""
        domains = arrays["domain_id"]
        sets = arrays["set_id"]
        sources = arrays["source_code"]
        normals_all = arrays["normal"]
        for raw_domain, set_id in sorted(set(zip(domains.tolist(), sets.tolist()))):
            domain_id = None if raw_domain < 0 else int(raw_domain)
            key = (domain_id, int(set_id))
            mask = (domains == raw_domain) & (sets == set_id)
            summary = summaries.setdefault(key, M10DomainSetSummary(domain_id=domain_id, set_id=int(set_id)))
            summary.fracture_count = int(np.count_nonzero(mask))
            stochastic = mask & (sources == SOURCE_CODES[M10FractureSource.STOCHASTIC.value])
            normals = normals_all[stochastic]
            if len(normals) == 0:
                continue
            reference = normals[0]
            aligned = np.where((normals @ reference)[:, None] >= 0.0, normals, -normals)
            resultant = aligned.sum(axis=0)
            length = float(np.linalg.norm(resultant))
            if length <= 1e-12:
                continue
            direction, dip = normal_to_dip_dir_dip(resultant / length)
            summary = summaries[key]
            summary.generated_mean_dip_direction = direction
            summary.generated_mean_dip = dip
            summary.generated_resultant_length = length / len(normals)

    def _input_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.config.model_dump_json().encode("utf-8"))
        digest.update(self.metadata.model_dump_json().encode("utf-8"))
        for name in sorted(self.arrays):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(self.arrays[name]).tobytes())
        digest.update(
            json.dumps(
                [item.model_dump(mode="json") for item in sorted(self.size_models.values(), key=lambda value: str((value.domain_id, value.set_id)))],
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(json.dumps([item.model_dump(mode="json") for item in self.deterministic_structures], sort_keys=True).encode("utf-8"))
        digest.update(json.dumps([item.__dict__ for item in self.conditioned_observations], sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def _check_cancelled(self) -> None:
        if self.cancelled and self.cancelled():
            raise InterruptedError("M10 explicit DFN generation cancelled")
