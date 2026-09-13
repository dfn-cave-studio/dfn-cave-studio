"""Allocation-contract based resource estimates for M9 voxel fields."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import math
import os
from typing import Iterable

import numpy as np

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.m9 import DensityMethod


@dataclass(frozen=True)
class ArrayAllocation:
    """One planned NumPy allocation, without allocating its contents."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def nbytes(self) -> int:
        """Return the exact payload bytes implied by shape and dtype."""
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class FieldResourceEstimate:
    """Persistent and bounded-temporary memory required by one field build."""

    grid_shape: tuple[int, int, int]
    voxel_count: int
    arrays: tuple[ArrayAllocation, ...]
    temporary_arrays: tuple[ArrayAllocation, ...]
    worker_extra_bytes: dict[int, int]
    budget_bytes: int | None = None
    system_available_bytes: int | None = None
    safety_margin_bytes: int = 0

    @property
    def persistent_bytes(self) -> int:
        """Return exact final scientific-array payload bytes."""
        return sum(item.nbytes for item in self.arrays)

    @property
    def temporary_bytes(self) -> int:
        """Return the estimated peak payload of named chunk temporaries."""
        return sum(item.nbytes for item in self.temporary_arrays)

    @property
    def exceeds_budget(self) -> bool:
        """Return whether one-worker memory exceeds user or system-safe budget."""
        required = self.peak_bytes
        user_exceeded = self.budget_bytes is not None and required > self.budget_bytes
        system_exceeded = self.system_available_bytes is not None and required > self.system_available_bytes // 2
        return user_exceeded or system_exceeded

    @property
    def peak_bytes(self) -> int:
        """Return persistent payload, live temporaries, and explicit safety margin."""
        return self.persistent_bytes + self.temporary_bytes + self.safety_margin_bytes


def available_system_memory_bytes() -> int | None:
    """Return currently available physical memory without adding a dependency."""
    if os.name != "nt":
        return None

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.available_physical)


def parameter_field_array_contract(
    shape: tuple[int, int, int], set_ids: Iterable[int], method: DensityMethod
) -> tuple[ArrayAllocation, ...]:
    """Return the exact persisted First Parameter Field array contract."""
    specs: list[tuple[str, object]] = [
        ("cell_state", np.uint8), ("domain_id", np.int32), ("p32_total", np.float32),
        ("nearest_data_distance", np.float32), ("neighbour_count", np.int16),
        ("effective_sample_length", np.float32), ("observation_count", np.int32),
        ("confidence", np.float32), ("density_method", np.uint8),
    ]
    if method == DensityMethod.ORDINARY_KRIGING:
        specs.append(("kriging_variance_total", np.float32))
    for set_id in sorted(set(set_ids)):
        specs.extend([
            (f"set_{set_id}_p32", np.float32),
            *(([(f"set_{set_id}_kriging_variance", np.float32)])
              if method == DensityMethod.ORDINARY_KRIGING else []),
            (f"set_{set_id}_dip_direction", np.float32), (f"set_{set_id}_dip", np.float32),
            (f"set_{set_id}_kappa", np.float32), (f"set_{set_id}_mean_radius", np.float32),
            (f"set_{set_id}_mean_squared_radius", np.float32), (f"set_{set_id}_probability", np.float32),
            (f"set_{set_id}_size_type", np.uint8), (f"set_{set_id}_size_source", np.uint8),
            (f"set_{set_id}_size_min_radius", np.float32), (f"set_{set_id}_size_max_radius", np.float32),
            (f"set_{set_id}_size_parameter_1", np.float32), (f"set_{set_id}_size_parameter_2", np.float32),
        ])
    return tuple(ArrayAllocation(name, shape, np.dtype(dtype)) for name, dtype in specs)


def scalar_field_array_contract(shape: tuple[int, int, int]) -> tuple[ArrayAllocation, ...]:
    """Return the exact persisted generic scalar-field array contract."""
    return tuple(ArrayAllocation(name, shape, np.dtype(dtype)) for name, dtype in (
        ("estimate", np.float32), ("kriging_variance", np.float32), ("cell_state", np.uint8),
        ("neighbor_count", np.int16), ("domain_id", np.int32),
    ))


def estimate_borehole_fracture_resources(
    realization_counts: float | Iterable[int | float],
    *,
    interval_count: int = 0,
    chunk_size: int = 65_536,
    budget_bytes: int | None = None,
    system_available_bytes: int | None = None,
    component_count: int = 0,
) -> FieldResourceEstimate:
    """Estimate the real Phase 2A ownership model without allocating fracture rows.

    Every requested realization remains resident in the candidate until it is
    committed/saved.  Only one realization is populated at a time, directly
    into its final arrays; the count/offset plans and one bounded working chunk
    are the only generation-wide temporary arrays.
    """
    if isinstance(realization_counts, (int, float, np.integer, np.floating)):
        raw_counts = [realization_counts]
    else:
        raw_counts = list(realization_counts)
    if not raw_counts:
        raw_counts = [0]
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in raw_counts):
        raise ValueError("realization counts must be finite and non-negative")
    if chunk_size < 1 or interval_count < 0 or component_count < 0:
        raise ValueError("chunk_size must be positive and counts must be non-negative")
    counts = [int(math.ceil(float(value))) for value in raw_counts]
    count = sum(counts)
    chunk = min(max(max(counts, default=0), 1), chunk_size)
    arrays = tuple(
        ArrayAllocation(name, shape, np.dtype(dtype))
        for name, shape, dtype in (
            ("interval_index", (count,), np.int32),
            ("measured_depth", (count,), np.float64),
            ("xyz_offset", (count, 3), np.float32),
            ("global_set_id", (count,), np.int32),
            ("local_component_index", (count,), np.int32),
            ("component_type", (count,), np.uint8),
            ("dip", (count,), np.float32),
            ("dip_direction", (count,), np.float32),
            ("status", (count,), np.uint8),
        )
    )
    temporary = (
        ArrayAllocation("poisson_counts", (len(counts), interval_count), np.dtype(np.int64)),
        ArrayAllocation("realization_offsets", (len(counts), interval_count + 1), np.dtype(np.int64)),
        ArrayAllocation("xyz", (chunk, 3), np.dtype(np.float64)),
        ArrayAllocation("mean_normals", (chunk, 3), np.dtype(np.float64)),
        ArrayAllocation("sampled_normals", (chunk, 3), np.dtype(np.float64)),
        ArrayAllocation("assignment_uniform", (chunk,), np.dtype(np.float64)),
        ArrayAllocation("local_component_codes", (chunk,), np.dtype(np.int32)),
        ArrayAllocation("global_set_codes", (chunk,), np.dtype(np.int32)),
        ArrayAllocation("component_type_codes", (chunk,), np.dtype(np.uint8)),
        ArrayAllocation("dominant_and_random_masks", (2, chunk), np.dtype(np.bool_)),
    )
    temporary_bytes = sum(item.nbytes for item in temporary)
    persistent_bytes = sum(item.nbytes for item in arrays)
    component_metadata_allowance = component_count * 1024
    safety_margin = max(256 * 1024, math.ceil((persistent_bytes + temporary_bytes) * 0.05))
    safety_margin += component_metadata_allowance
    return FieldResourceEstimate(
        (count, 1, 1),
        count,
        arrays,
        temporary,
        {workers: temporary_bytes * workers for workers in (1, 2, 4, 8)},
        budget_bytes,
        system_available_bytes,
        safety_margin,
    )


def estimate_parameter_field_resources(
    bounds: ModelBounds,
    voxel: VoxelConfig,
    set_ids: Iterable[int],
    method: DensityMethod,
    *,
    chunk_size: int = 8192,
    budget_bytes: int | None = None,
    system_available_bytes: int | None = None,
) -> FieldResourceEstimate:
    """Estimate First Parameter Field memory without creating a coordinate grid."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    shape = voxel.compute_grid_dimensions(bounds)
    count = math.prod(shape)
    chunk = min(count, chunk_size)
    set_count = len(set(set_ids))
    temporary = [
        ArrayAllocation("flat_indices", (chunk,), np.dtype(np.int64)),
        ArrayAllocation("grid_indices", (chunk, 3), np.dtype(np.int64)),
        ArrayAllocation("points", (chunk, 3), np.dtype(np.float64)),
        ArrayAllocation("inside_mask", (chunk,), np.dtype(np.bool_)),
        ArrayAllocation("domain_labels", (chunk,), np.dtype(np.int64)),
        ArrayAllocation("domain_kdtree_query_distances", (chunk, 2), np.dtype(np.float64)),
        ArrayAllocation("domain_kdtree_query_indices", (chunk, 2), np.dtype(np.intp)),
        ArrayAllocation("domain_kdtree_tie_rows", (chunk,), np.dtype(np.intp)),
        ArrayAllocation("valid_matrix", (set_count, chunk), np.dtype(np.bool_)),
        ArrayAllocation("value_matrix", (set_count, chunk), np.dtype(np.float64)),
        ArrayAllocation("distance_confidence_variance_matrix", (3 * set_count, chunk), np.dtype(np.float64)),
        ArrayAllocation("neighbor_matrix", (set_count, chunk), np.dtype(np.int16)),
        ArrayAllocation("current_prediction_values", (chunk,), np.dtype(np.float64)),
        ArrayAllocation("aggregate_float_buffers", (4, chunk), np.dtype(np.float64)),
        ArrayAllocation("aggregate_neighbor_buffer", (chunk,), np.dtype(np.int16)),
        ArrayAllocation("aggregate_validity_buffers", (2, chunk), np.dtype(np.bool_)),
    ]
    temporary_bytes = sum(item.nbytes for item in temporary)
    return FieldResourceEstimate(
        shape, count, parameter_field_array_contract(shape, set_ids, method), tuple(temporary),
        {workers: temporary_bytes * workers for workers in (1, 2, 4, 8)}, budget_bytes, system_available_bytes,
    )


def estimate_scalar_field_resources(
    bounds: ModelBounds,
    voxel: VoxelConfig,
    *,
    chunk_size: int = 2048,
    budget_bytes: int | None = None,
    system_available_bytes: int | None = None,
) -> FieldResourceEstimate:
    """Estimate generic scalar field memory without creating voxel coordinates."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    shape = voxel.compute_grid_dimensions(bounds)
    count = math.prod(shape)
    chunk = min(count, chunk_size)
    temporary = (
        ArrayAllocation("flat_indices", (chunk,), np.dtype(np.int64)),
        ArrayAllocation("grid_indices", (chunk, 3), np.dtype(np.int64)),
        ArrayAllocation("points", (chunk, 3), np.dtype(np.float64)),
        ArrayAllocation("inside_mask", (chunk,), np.dtype(np.bool_)),
        ArrayAllocation("domain_labels", (chunk,), np.dtype(np.int64)),
        ArrayAllocation("domain_kdtree_query_distances", (chunk, 2), np.dtype(np.float64)),
        ArrayAllocation("domain_kdtree_query_indices", (chunk, 2), np.dtype(np.intp)),
        ArrayAllocation("domain_kdtree_tie_rows", (chunk,), np.dtype(np.intp)),
        ArrayAllocation("estimate_chunk", (chunk,), np.dtype(np.float64)),
        ArrayAllocation("variance_chunk", (chunk,), np.dtype(np.float64)),
        ArrayAllocation("neighbor_chunk", (chunk,), np.dtype(np.int16)),
    )
    temporary_bytes = sum(item.nbytes for item in temporary)
    return FieldResourceEstimate(
        shape, count, scalar_field_array_contract(shape), temporary,
        {workers: temporary_bytes * workers for workers in (1, 2, 4, 8)}, budget_bytes, system_available_bytes,
    )


def format_resource_estimate(estimate: FieldResourceEstimate) -> str:
    """Format the compact resource summary shown in the main dialog."""
    mib = 1024**2
    status = "OVER BUDGET" if estimate.exceeds_budget else "within budget"
    return (
        f"Shape: {estimate.grid_shape} ({estimate.voxel_count:,} voxels)\n"
        f"Persistent arrays: {estimate.persistent_bytes / mib:,.2f} MiB\n"
        f"Temporary buffers: {estimate.temporary_bytes / mib:,.2f} MiB\n"
        f"Budget status: {status}"
    )


def format_resource_details(estimate: FieldResourceEstimate) -> str:
    """Format the scrollable allocation-by-allocation diagnostic details."""
    mib = 1024**2
    arrays = "\n".join(
        f"  {item.name}: {item.shape}, {item.dtype}, {item.nbytes / mib:,.2f} MiB" for item in estimate.arrays
    )
    temporaries = "\n".join(
        f"  {item.name}: {item.shape}, {item.dtype}, {item.nbytes / mib:,.2f} MiB"
        for item in estimate.temporary_arrays
    )
    system = (
        f"; available physical: {estimate.system_available_bytes / mib:,.2f} MiB; system-safe limit: "
        f"{estimate.system_available_bytes / (2 * mib):,.2f} MiB"
        if estimate.system_available_bytes is not None else ""
    )
    return (
        f"{format_resource_estimate(estimate)}\n"
        f"1/2/4/8-worker extra: "
        + "/".join(f"{estimate.worker_extra_bytes[n] / mib:,.2f}" for n in (1, 2, 4, 8))
        + f" MiB{system}\nPersistent arrays:\n{arrays}\nTemporary buffers:\n{temporaries}"
    )
