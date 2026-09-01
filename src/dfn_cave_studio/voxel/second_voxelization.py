"""M11.1 sparse analytic fracture/voxel second voxelization."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

import numpy as np

from dfn_cave_studio.geometry.disk_box import (
    disk_aabb_intersection_area_exact,
    disk_world_aabb,
    geometry_tolerance,
)
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.m9 import ParameterFieldMetadata
from dfn_cave_studio.models.m10 import M10Realization
from dfn_cave_studio.models.m11 import (
    M11ConservationSummary,
    M11DomainSetMetrics,
    M11SecondVoxelizationResult,
)
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]
_PROCESS_CANCEL_EVENT: Any = None


def _initialize_process_cancel_event(cancel_event: Any) -> None:
    """Install one spawn-safe event in each process owned by this calculation."""
    global _PROCESS_CANCEL_EVENT
    _PROCESS_CANCEL_EVENT = cancel_event


class _CallbackCancelEvent:
    """Event-shaped adapter used by the in-process computation path."""

    def __init__(self, callback: CancelCallback) -> None:
        self._callback = callback

    def is_set(self) -> bool:
        return bool(self._callback())


def parameter_field_hash(
    metadata: ParameterFieldMetadata, arrays: dict[str, np.ndarray]
) -> str:
    """Return a stable content hash for the M9 parameter-field input."""
    digest = hashlib.sha256(metadata.model_dump_json().encode("utf-8"))
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        values = np.ascontiguousarray(arrays[name])
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SecondVoxelizationConfig:
    """Execution controls that do not alter the scientific result."""

    worker_count: int = 1
    fracture_batch_size: int = 65_536

    def __post_init__(self) -> None:
        if self.worker_count not in (1, 2, 4, 8):
            raise ValueError("worker_count must be one of 1, 2, 4, or 8")
        if not 1 <= self.fracture_batch_size <= 65_536:
            raise ValueError("fracture_batch_size must be between 1 and 65,536")


@dataclass
class _ChunkResult:
    ordinals: np.ndarray
    voxel_indices: np.ndarray
    areas: np.ndarray
    candidate_count: int
    target_ordinals: np.ndarray
    target_areas: np.ndarray
    analysis_target_areas: np.ndarray
    summed_areas: np.ndarray


def _bounds_arrays(bounds: ModelBounds) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([bounds.x_min, bounds.y_min, bounds.z_min], dtype=np.float64),
        np.asarray([bounds.x_max, bounds.y_max, bounds.z_max], dtype=np.float64),
    )


def _coplanar_owner_axis(
    center: np.ndarray,
    normal: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[int, int] | None:
    tolerance = geometry_tolerance(center, origin, spacing)
    axis = int(np.argmax(np.abs(normal)))
    if abs(abs(float(normal[axis])) - 1.0) > tolerance:
        return None
    if np.max(np.abs(np.delete(normal, axis))) > tolerance:
        return None
    coordinate = (center[axis] - origin[axis]) / spacing[axis]
    boundary = round(float(coordinate))
    if abs(float(coordinate) - boundary) > tolerance / max(float(spacing[axis]), tolerance):
        return None
    if boundary <= 0:
        owner = 0
    elif boundary >= shape[axis]:
        owner = shape[axis] - 1
    else:
        owner = boundary
    return axis, owner


def _candidate_index_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    first = np.floor((lower - origin) / spacing).astype(np.int64)
    last = np.floor((upper - origin) / spacing).astype(np.int64)
    maximum_index = np.asarray(shape, dtype=np.int64) - 1
    first = np.minimum(np.maximum(first, 0), maximum_index)
    last = np.minimum(last, maximum_index)
    if np.any(first > last):
        return None
    return first, last


def _disk_box_area_with_culling(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    disk_min: np.ndarray,
    disk_max: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> float:
    """Return exact area after constant-time containment and separation tests."""
    tolerance = geometry_tolerance(center, radius, box_min, box_max)
    if np.any(disk_max < box_min - tolerance) or np.any(disk_min > box_max + tolerance):
        return 0.0
    if np.all(disk_min >= box_min - tolerance) and np.all(disk_max <= box_max + tolerance):
        return math.pi * radius * radius
    box_center = 0.5 * (box_min + box_max)
    half_extent = 0.5 * (box_max - box_min)
    plane_distance = abs(float(np.dot(normal, box_center - center)))
    projected_half_extent = float(np.dot(np.abs(normal), half_extent))
    if plane_distance > projected_half_extent + tolerance:
        return 0.0
    return disk_aabb_intersection_area_exact(center, normal, radius, box_min, box_max)


def _process_fracture_chunk(payload: dict[str, Any]) -> _ChunkResult:
    cancel_event = payload.get("cancel_event") or _PROCESS_CANCEL_EVENT
    centers = payload["centers"]
    normals = payload["normals"]
    radii = payload["radii"]
    ordinals_input = payload["ordinals"]
    origin = payload["origin"]
    spacing = payload["spacing"]
    shape = tuple(int(value) for value in payload["shape"])
    generation_min = payload["generation_min"]
    generation_max = payload["generation_max"]
    grid_max = origin + spacing * np.asarray(shape, dtype=np.float64)
    analysis_min = origin
    analysis_max = grid_max

    output_ordinals: list[int] = []
    output_voxels: list[int] = []
    output_areas: list[float] = []
    target_areas = np.zeros(len(radii), dtype=np.float64)
    analysis_target_areas = np.zeros(len(radii), dtype=np.float64)
    summed_areas = np.zeros(len(radii), dtype=np.float64)
    candidate_count = 0

    for local_index in range(len(radii)):
        if cancel_event is not None and local_index % 256 == 0 and cancel_event.is_set():
            raise InterruptedError("M11 second voxelization cancelled")
        center = np.asarray(centers[local_index], dtype=np.float64)
        normal = np.asarray(normals[local_index], dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        radius = float(radii[local_index])
        ordinal = int(ordinals_input[local_index])
        disk_min, disk_max = disk_world_aabb(center, normal, radius)
        target_areas[local_index] = _disk_box_area_with_culling(
            center, normal, radius, disk_min, disk_max, generation_min, generation_max
        )
        analysis_box_min = np.maximum(analysis_min, generation_min)
        analysis_box_max = np.minimum(analysis_max, generation_max)
        if np.all(analysis_box_max > analysis_box_min):
            analysis_target_areas[local_index] = _disk_box_area_with_culling(
                center, normal, radius, disk_min, disk_max, analysis_box_min, analysis_box_max
            )
        clipped_min = np.maximum.reduce([disk_min, generation_min, analysis_min])
        clipped_max = np.minimum.reduce([disk_max, generation_max, analysis_max])
        if np.any(clipped_max < clipped_min):
            continue
        index_bounds = _candidate_index_bounds(clipped_min, clipped_max, origin, spacing, shape)
        if index_bounds is None:
            continue
        first, last = index_bounds
        owner = _coplanar_owner_axis(center, normal, origin, spacing, shape)
        candidate_checks = 0
        for i in range(int(first[0]), int(last[0]) + 1):
            for j in range(int(first[1]), int(last[1]) + 1):
                for k in range(int(first[2]), int(last[2]) + 1):
                    candidate_checks += 1
                    if (
                        cancel_event is not None
                        and candidate_checks % 256 == 0
                        and cancel_event.is_set()
                    ):
                        raise InterruptedError("M11 second voxelization cancelled")
                    if owner is not None and (i, j, k)[owner[0]] != owner[1]:
                        continue
                    candidate_count += 1
                    voxel_min = origin + np.asarray([i, j, k], dtype=np.float64) * spacing
                    voxel_max = voxel_min + spacing
                    box_min = np.maximum(voxel_min, generation_min)
                    box_max = np.minimum(voxel_max, generation_max)
                    if np.any(box_max <= box_min):
                        continue
                    area = _disk_box_area_with_culling(
                        center, normal, radius, disk_min, disk_max, box_min, box_max
                    )
                    tolerance = geometry_tolerance(center, radius, box_min, box_max)
                    if area <= tolerance * tolerance:
                        continue
                    output_ordinals.append(ordinal)
                    output_voxels.append(int(np.ravel_multi_index((i, j, k), shape)))
                    output_areas.append(area)
                    summed_areas[local_index] += area
    return _ChunkResult(
        ordinals=np.asarray(output_ordinals, dtype=np.uint64),
        voxel_indices=np.asarray(output_voxels, dtype=np.uint64),
        areas=np.asarray(output_areas, dtype=np.float64),
        candidate_count=candidate_count,
        target_ordinals=np.asarray(ordinals_input, dtype=np.uint64),
        target_areas=target_areas,
        analysis_target_areas=analysis_target_areas,
        summed_areas=summed_areas,
    )


class M11SecondVoxelizer:
    """Compute stable sparse exact-area intersections for one M10 realization."""

    def __init__(
        self,
        metadata: ParameterFieldMetadata,
        parameter_arrays: dict[str, np.ndarray],
        generation_domain: ModelBounds,
        realization: M10Realization,
        config: SecondVoxelizationConfig | None = None,
        *,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> None:
        self.metadata = metadata
        self.parameter_arrays = parameter_arrays
        self.generation_domain = generation_domain
        self.realization = realization
        self.config = config or SecondVoxelizationConfig()
        self.progress = progress
        self.cancelled = cancelled or (lambda: False)
        self._submitted_batch_count = 0
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        required = ("center", "normal", "radius", "ordinal", "set_id")
        missing = [name for name in required if name not in self.realization.geometry_arrays]
        if missing:
            raise ValueError(f"M10 realization is missing geometry arrays: {', '.join(missing)}")
        count = self.realization.fracture_count
        if any(len(self.realization.geometry_arrays[name]) != count for name in required):
            raise ValueError("M10 geometry arrays must have identical fracture counts")
        ordinals = np.asarray(self.realization.geometry_arrays["ordinal"])
        if len(np.unique(ordinals)) != count:
            raise ValueError("M10 fracture ordinals must be unique")
        centers = np.asarray(self.realization.geometry_arrays["center"], dtype=np.float64)
        normals = np.asarray(self.realization.geometry_arrays["normal"], dtype=np.float64)
        radii = np.asarray(self.realization.geometry_arrays["radius"], dtype=np.float64)
        if centers.shape != (count, 3) or normals.shape != (count, 3):
            raise ValueError("M10 center and normal arrays must have shape (N, 3)")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(normals)):
            raise ValueError("M10 centers and normals must be finite")
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("M10 radii must be finite and greater than zero")
        lengths = np.linalg.norm(normals, axis=1)
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
            raise ValueError("M10 normals must have positive finite length")
        if not np.allclose(lengths, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("M10 normals must be unit vectors")
        if "cell_state" not in self.parameter_arrays or "domain_id" not in self.parameter_arrays:
            raise ValueError("M9 parameter field must contain cell_state and domain_id")
        if tuple(self.parameter_arrays["cell_state"].shape) != self.metadata.shape:
            raise ValueError("M9 cell_state shape does not match parameter-field metadata")

    def _payloads(self, cancel_event: Any) -> Iterator[dict[str, Any]]:
        arrays = self.realization.geometry_arrays
        count = self.realization.fracture_count
        generation_min, generation_max = _bounds_arrays(self.generation_domain)
        base = {
            "origin": np.asarray(self.metadata.origin, dtype=np.float64),
            "spacing": np.asarray(self.metadata.spacing, dtype=np.float64),
            "shape": self.metadata.shape,
            "generation_min": generation_min,
            "generation_max": generation_max,
        }
        for start in range(0, count, self.config.fracture_batch_size):
            stop = min(count, start + self.config.fracture_batch_size)
            yield {
                **base,
                "centers": np.asarray(arrays["center"][start:stop], dtype=np.float64),
                "normals": np.asarray(arrays["normal"][start:stop], dtype=np.float64),
                "radii": np.asarray(arrays["radius"][start:stop], dtype=np.float64),
                "ordinals": np.asarray(arrays["ordinal"][start:stop], dtype=np.uint64),
                "cancel_event": cancel_event,
            }

    def compute(self) -> M11SecondVoxelizationResult:
        """Compute one complete result transactionally; cancellation returns no partial state."""
        chunks: list[_ChunkResult] = []
        processed = 0
        total = self.realization.fracture_count
        if self.config.worker_count == 1:
            cancel_event = _CallbackCancelEvent(self.cancelled)
            for payload in self._payloads(cancel_event):
                self._check_cancelled()
                self._submitted_batch_count += 1
                chunk = _process_fracture_chunk(payload)
                chunks.append(chunk)
                processed += len(payload["radii"])
                self._report(processed, total, chunk.candidate_count)
        else:
            context = multiprocessing.get_context("spawn")
            cancel_event = context.Event()
            executor = ProcessPoolExecutor(
                max_workers=self.config.worker_count,
                mp_context=context,
                initializer=_initialize_process_cancel_event,
                initargs=(cancel_event,),
            )
            payloads = iter(self._payloads(None))
            futures: dict[Future[_ChunkResult], int] = {}
            exhausted = False

            def submit_available() -> None:
                nonlocal exhausted
                maximum_in_flight = self.config.worker_count * 2
                while not exhausted and len(futures) < maximum_in_flight:
                    if self.cancelled():
                        cancel_event.set()
                        return
                    try:
                        payload = next(payloads)
                    except StopIteration:
                        exhausted = True
                        return
                    futures[executor.submit(_process_fracture_chunk, payload)] = len(payload["radii"])
                    self._submitted_batch_count += 1

            try:
                submit_available()
                while futures:
                    if self.cancelled():
                        cancel_event.set()
                        raise InterruptedError("M11 second voxelization cancelled")
                    completed, _ = wait(tuple(futures), timeout=0.025, return_when=FIRST_COMPLETED)
                    if not completed:
                        continue
                    for future in completed:
                        batch_count = futures.pop(future)
                        chunk = future.result()
                        chunks.append(chunk)
                        processed += batch_count
                        self._report(processed, total, chunk.candidate_count)
                    submit_available()
            except BaseException:
                cancel_event.set()
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
        self._check_cancelled()
        return self._assemble(chunks)

    def _assemble(self, chunks: list[_ChunkResult]) -> M11SecondVoxelizationResult:
        self._check_cancelled()
        ordinals = np.concatenate([item.ordinals for item in chunks]) if chunks else np.empty(0, dtype=np.uint64)
        voxels = np.concatenate([item.voxel_indices for item in chunks]) if chunks else np.empty(0, dtype=np.uint64)
        areas = np.concatenate([item.areas for item in chunks]) if chunks else np.empty(0, dtype=np.float64)
        if len(areas):
            self._check_cancelled()
            order = np.lexsort((ordinals, voxels))
            ordinals, voxels, areas = ordinals[order], voxels[order], areas[order]
        maximum_ordinal = int(ordinals.max(initial=0))
        maximum_voxel = int(voxels.max(initial=0))
        ordinal_dtype = np.uint32 if maximum_ordinal <= np.iinfo(np.uint32).max else np.uint64
        voxel_dtype = np.uint32 if maximum_voxel <= np.iinfo(np.uint32).max else np.uint64
        sparse_arrays: dict[str, np.ndarray] = {
            "fracture_ordinal": ordinals.astype(ordinal_dtype, copy=False),
            "voxel_flat_index": voxels.astype(voxel_dtype, copy=False),
            "intersection_area": areas,
            "cell_state": np.asarray(self.parameter_arrays["cell_state"], dtype=np.uint8).copy(),
        }
        self._populate_p32_arrays(sparse_arrays)
        self._check_cancelled()
        target_ordinals = np.concatenate([item.target_ordinals for item in chunks]) if chunks else np.empty(0, np.uint64)
        target_areas = np.concatenate([item.target_areas for item in chunks]) if chunks else np.empty(0, np.float64)
        analysis_target_areas = (
            np.concatenate([item.analysis_target_areas for item in chunks]) if chunks else np.empty(0, np.float64)
        )
        summed_areas = np.concatenate([item.summed_areas for item in chunks]) if chunks else np.empty(0, np.float64)
        if len(target_ordinals):
            self._check_cancelled()
            target_order = np.argsort(target_ordinals, kind="stable")
            target_ordinals = target_ordinals[target_order]
            target_areas = target_areas[target_order]
            analysis_target_areas = analysis_target_areas[target_order]
            summed_areas = summed_areas[target_order]
        conservation = self._conservation(target_ordinals, target_areas, analysis_target_areas, summed_areas)
        candidate_count = sum(item.candidate_count for item in chunks)
        sparse_bytes = int(
            sparse_arrays["fracture_ordinal"].nbytes
            + sparse_arrays["voxel_flat_index"].nbytes
            + sparse_arrays["intersection_area"].nbytes
        )
        total_array_bytes = int(sum(array.nbytes for array in sparse_arrays.values()))
        source_field_hash = parameter_field_hash(self.metadata, self.parameter_arrays)
        self._check_cancelled()
        return M11SecondVoxelizationResult(
            realization_id=f"m11-{self.realization.realization_id}",
            source_m10_realization_id=self.realization.realization_id,
            source_m10_config_hash=self.realization.config_hash,
            source_parameter_field_hash=source_field_hash,
            candidate_pair_count=candidate_count,
            positive_intersection_count=len(areas),
            rejected_candidate_count=candidate_count - len(areas),
            sparse_bytes=sparse_bytes,
            total_array_bytes=total_array_bytes,
            p32_unresolved_orientation_reference=self.realization.quality.p32_unresolved_orientation,
            conservation=conservation,
            domain_set_metrics=self._domain_set_metrics(sparse_arrays),
            arrays=sparse_arrays,
            provenance={
                "area_definition": "Area(D_f intersect B_v intersect GenerationDomain)",
                "circle_area_method": "analytic line-and-circular-arc integration",
                "candidate_method": "tight disk AABB to deterministic i/j/k range",
                "candidate_false_positives_allowed": True,
                "candidate_false_negatives_allowed": False,
                "worker_count": self.config.worker_count,
                "fracture_batch_size": self.config.fracture_batch_size,
                "p32_contract": "P32_total=P32_explicit_intersection+P32_subgrid",
                "unresolved_orientation_reference": self.realization.quality.p32_unresolved_orientation,
                "unresolved_orientation_in_total": False,
                "m10_clipped_area_overwritten": False,
            },
        )

    def _populate_p32_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        shape = self.metadata.shape
        voxel_volume = float(np.prod(self.metadata.spacing))
        ordinals = np.asarray(arrays["fracture_ordinal"], dtype=np.int64)
        voxels = np.asarray(arrays["voxel_flat_index"], dtype=np.int64)
        areas = arrays["intersection_area"]
        fracture_sets = np.asarray(self.realization.geometry_arrays["set_id"], dtype=np.int64)
        source_ordinals = np.asarray(self.realization.geometry_arrays["ordinal"], dtype=np.uint64)
        source_order = np.argsort(source_ordinals, kind="stable")
        sorted_source_ordinals = source_ordinals[source_order]
        ordinal_positions = np.searchsorted(sorted_source_ordinals, ordinals.astype(np.uint64, copy=False))
        if len(ordinals) and (
            np.any(ordinal_positions >= len(sorted_source_ordinals))
            or np.any(sorted_source_ordinals[ordinal_positions] != ordinals)
        ):
            raise ValueError("Sparse intersection references an unknown M10 fracture ordinal")
        sparse_sets = fracture_sets[source_order[ordinal_positions]] if len(ordinals) else np.empty(0, np.int64)
        aggregate_explicit = np.zeros(shape, dtype=np.float64)
        aggregate_subgrid = np.zeros(shape, dtype=np.float64)
        aggregate_count = np.zeros(shape, dtype=np.uint32)
        for set_id in self.metadata.set_ids:
            self._check_cancelled()
            explicit_flat = np.zeros(int(np.prod(shape)), dtype=np.float64)
            count_flat = np.zeros(int(np.prod(shape)), dtype=np.uint32)
            if len(areas):
                selected = sparse_sets == int(set_id)
                np.add.at(explicit_flat, voxels[selected], areas[selected] / voxel_volume)
                np.add.at(count_flat, voxels[selected], 1)
            explicit = explicit_flat.reshape(shape)
            count_values = count_flat.reshape(shape)
            subgrid_name = f"p32_subgrid_set_{set_id}"
            subgrid = np.asarray(
                self.realization.geometry_arrays.get(subgrid_name, np.zeros(shape, dtype=np.float32)),
                dtype=np.float64,
            )
            if subgrid.shape != shape:
                raise ValueError(f"{subgrid_name} shape does not match the M9 parameter field")
            total = explicit + subgrid
            arrays[f"p32_explicit_intersection_set_{set_id}"] = explicit
            arrays[subgrid_name] = subgrid.astype(np.float32, copy=True)
            arrays[f"p32_total_set_{set_id}"] = total
            arrays[f"intersecting_fracture_count_set_{set_id}"] = count_values
            aggregate_explicit += explicit
            aggregate_subgrid += subgrid
            aggregate_count += count_values
        arrays["p32_explicit_intersection"] = aggregate_explicit
        arrays["p32_subgrid"] = aggregate_subgrid
        arrays["p32_total"] = aggregate_explicit + aggregate_subgrid
        arrays["intersecting_fracture_count"] = aggregate_count

    def _conservation(
        self,
        ordinals: np.ndarray,
        target_areas: np.ndarray,
        analysis_target_areas: np.ndarray,
        summed_areas: np.ndarray,
    ) -> M11ConservationSummary:
        errors = np.abs(analysis_target_areas - summed_areas)
        absolute_tolerance = 1e-12
        relative_tolerance = 1e-10
        tolerances = np.maximum(
            absolute_tolerance,
            relative_tolerance * np.maximum(analysis_target_areas, 1.0),
        )
        over = errors > tolerances
        total_target = float(target_areas.sum(dtype=np.float64))
        analysis_target = float(analysis_target_areas.sum(dtype=np.float64))
        total_sum = float(summed_areas.sum(dtype=np.float64))
        m10_clipped_source = np.asarray(
            self.realization.geometry_arrays.get("clipped_area", np.zeros_like(target_areas)), dtype=np.float64
        )
        source_ordinals = np.asarray(self.realization.geometry_arrays["ordinal"], dtype=np.uint64)
        if len(m10_clipped_source) != len(source_ordinals) or len(target_areas) != len(ordinals):
            m10_difference = 0.0
        else:
            source_order = np.argsort(source_ordinals, kind="stable")
            positions = np.searchsorted(source_ordinals[source_order], ordinals)
            m10_clipped = m10_clipped_source[source_order[positions]]
            m10_difference = float(np.abs(m10_clipped - target_areas).sum(dtype=np.float64))
        return M11ConservationSummary(
            target_area_total=total_target,
            analysis_domain_target_area_total=analysis_target,
            generation_area_outside_analysis_total=max(0.0, total_target - analysis_target),
            intersection_area_total=total_sum,
            absolute_error_total=abs(analysis_target - total_sum),
            relative_error_total=(
                abs(analysis_target - total_sum) / analysis_target if analysis_target > 0.0 else None
            ),
            maximum_absolute_error=float(errors.max(initial=0.0)),
            error_p50=float(np.quantile(errors, 0.50)) if len(errors) else 0.0,
            error_p95=float(np.quantile(errors, 0.95)) if len(errors) else 0.0,
            error_p99=float(np.quantile(errors, 0.99)) if len(errors) else 0.0,
            over_tolerance_count=int(np.count_nonzero(over)),
            over_tolerance_ordinals=[int(value) for value in ordinals[over][:1000]],
            m10_clipped_area_absolute_difference=m10_difference,
            absolute_area_tolerance=absolute_tolerance,
            relative_area_tolerance=relative_tolerance,
        )

    def _domain_set_metrics(self, arrays: dict[str, np.ndarray]) -> list[M11DomainSetMetrics]:
        domains = np.asarray(self.parameter_arrays["domain_id"])
        states = np.asarray(self.parameter_arrays["cell_state"])
        modeled_code = CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
        shape = self.metadata.shape
        edge = np.zeros(shape, dtype=bool)
        edge[[0, -1], :, :] = True
        edge[:, [0, -1], :] = True
        edge[:, :, [0, -1]] = True
        metrics: list[M11DomainSetMetrics] = []
        for set_id in self.metadata.set_ids:
            target = np.asarray(self.parameter_arrays[f"set_{set_id}_p32"], dtype=np.float64)
            realized = np.asarray(arrays[f"p32_total_set_{set_id}"], dtype=np.float64)
            valid_base = (states == modeled_code) & np.isfinite(target)
            for raw_domain in np.unique(domains[valid_base]):
                valid = valid_base & (domains == raw_domain)
                if not np.any(valid):
                    continue
                errors = realized[valid] - target[valid]
                absolute = np.abs(errors)
                relative = absolute / np.maximum(np.abs(target[valid]), 1e-15)
                boundary_errors = (realized - target)[valid & edge]
                interior_errors = (realized - target)[valid & ~edge]
                domain_id = None if int(raw_domain) < 0 else int(raw_domain)
                metrics.append(
                    M11DomainSetMetrics(
                        domain_id=domain_id,
                        set_id=int(set_id),
                        voxel_count=int(np.count_nonzero(valid)),
                        target_mean_p32=float(np.mean(target[valid])),
                        realized_mean_p32=float(np.mean(realized[valid])),
                        bias=float(np.mean(errors)),
                        mae=float(np.mean(absolute)),
                        rmse=float(math.sqrt(float(np.mean(np.square(errors))))),
                        mean_relative_error=float(np.mean(relative)),
                        error_p50=float(np.quantile(absolute, 0.50)),
                        error_p95=float(np.quantile(absolute, 0.95)),
                        boundary_voxel_count=int(np.count_nonzero(valid & edge)),
                        interior_voxel_count=int(np.count_nonzero(valid & ~edge)),
                        boundary_bias=float(np.mean(boundary_errors)) if len(boundary_errors) else None,
                        boundary_mae=float(np.mean(np.abs(boundary_errors))) if len(boundary_errors) else None,
                        boundary_rmse=(
                            float(np.sqrt(np.mean(np.square(boundary_errors)))) if len(boundary_errors) else None
                        ),
                        interior_bias=float(np.mean(interior_errors)) if len(interior_errors) else None,
                        interior_mae=float(np.mean(np.abs(interior_errors))) if len(interior_errors) else None,
                        interior_rmse=(
                            float(np.sqrt(np.mean(np.square(interior_errors)))) if len(interior_errors) else None
                        ),
                    )
                )
        return metrics

    def _check_cancelled(self) -> None:
        if self.cancelled():
            raise InterruptedError("M11 second voxelization cancelled")

    def _report(self, processed: int, total: int, candidates: int) -> None:
        if self.progress is not None:
            self.progress(
                processed,
                max(total, 1),
                f"Second voxelization: {processed:,}/{total:,} fractures; latest batch {candidates:,} candidates",
            )
