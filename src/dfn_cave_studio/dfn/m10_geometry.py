"""Compact M10 geometry accessors and on-demand disc materialization."""

from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Any

import numpy as np

from dfn_cave_studio.dfn.m10_generator import (
    DISTRIBUTION_CODES,
    ORIENTATION_CODES,
    SIZE_SOURCE_CODES,
    SOURCE_CODES,
    SOURCE_NAMES,
)
from dfn_cave_studio.models.m10 import M10Realization


def source_mask(realization: M10Realization, source: str) -> np.ndarray:
    """Return a mask for a user-facing fracture source name."""
    arrays = realization.geometry_arrays
    if "source_code" in arrays:
        return arrays["source_code"] == SOURCE_CODES[source]
    return arrays["source"] == source


def fracture_ids(realization: M10Realization) -> np.ndarray:
    """Construct stable IDs from realization ID and compact ordinals."""
    ordinals = realization.geometry_arrays.get("ordinal", np.arange(realization.fracture_count, dtype=np.uint64))
    return np.asarray([realization.fracture_id(int(value)) for value in ordinals], dtype=object)


def record_id(realization: M10Realization, index: int) -> str:
    """Resolve a sparse conditioned/deterministic source-record reference."""
    arrays = realization.geometry_arrays
    if "record_ref" not in arrays:
        return str(arrays.get("observation_record_id", np.asarray([""] * realization.fracture_count))[index])
    reference = int(arrays["record_ref"][index])
    values = realization.provenance.get("record_ids", [])
    return str(values[reference]) if 0 <= reference < len(values) else ""


def source_name(realization: M10Realization, index: int) -> str:
    """Resolve a compact source code to user-facing text."""
    arrays = realization.geometry_arrays
    if "source_code" not in arrays:
        return str(arrays["source"][index])
    mapping = realization.provenance.get("source_codes", SOURCE_NAMES)
    code = int(arrays["source_code"][index])
    return str(mapping.get(code, mapping.get(str(code), f"UNKNOWN_{code}")))


def disc_polygon(center: np.ndarray, normal: np.ndarray, radius: float, sides: int = 32) -> np.ndarray:
    """Materialize one display/export polygon from its authoritative disc parameters."""
    normal = np.array(normal, dtype=float, copy=True)
    normal /= np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    theta = np.linspace(0.0, 2.0 * math.pi, int(sides), endpoint=False)
    return np.asarray(center, dtype=float) + float(radius) * (
        np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    )


def polygon_for(realization: M10Realization, index: int, sides: int = 32) -> np.ndarray:
    """Return an exact stored clipped polygon or lazily materialize an intact disc."""
    arrays = realization.geometry_arrays
    if "clipped_counts" in arrays and bool(arrays["clipped"][index]):
        offset = int(arrays["clipped_offsets"][index])
        count = int(arrays["clipped_counts"][index])
        return arrays["clipped_points"][offset : offset + count]
    if "vertices" in arrays:
        count = int(arrays["vertex_count"][index])
        return arrays["vertices"][index, :count]
    return disc_polygon(arrays["center"][index], arrays["normal"][index], arrays["radius"][index], sides)


def iter_polygons(
    realization: M10Realization, indices: np.ndarray | None = None, sides: int = 32
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield polygons lazily so callers can stream large exports."""
    selected = range(realization.fracture_count) if indices is None else indices
    for index in selected:
        yield int(index), polygon_for(realization, int(index), sides=sides)


def authoritative_nbytes(realization: M10Realization) -> int:
    """Return exact NumPy payload size for one realization."""
    return int(sum(array.nbytes for array in realization.geometry_arrays.values()))


def code_name(realization: M10Realization, table: str, code: int) -> str:
    """Resolve distribution/orientation/size codes stored in realization metadata."""
    mapping: dict[Any, Any] = realization.provenance.get(table, {})
    return str(mapping.get(int(code), mapping.get(str(int(code)), f"UNKNOWN_{int(code)}")))


def migrate_legacy_geometry(realization: M10Realization) -> bool:
    """Convert the current unreleased object/string geometry layout to columnar-v2."""
    old = realization.geometry_arrays
    if not old:
        return False
    if "source_code" in old:
        if "size_class" in old:
            return False
        count = realization.fracture_count
        old["size_class"] = np.zeros(count, dtype=np.uint8)
        realization.provenance.update(
            {
                "size_class_status": "legacy_unknown_all_explicit",
                "p32_subgrid": 0.0,
                "migrated_from": realization.provenance.get("geometry_format", "columnar-v2"),
                "geometry_format": "columnar-v2-legacy-explicit",
            }
        )
        return True
    count = realization.fracture_count
    clipped = np.zeros(count, dtype=np.bool_)
    clipped_offsets = np.full(count, -1, dtype=np.int64)
    clipped_counts = np.zeros(count, dtype=np.int16)
    clipped_parts: list[np.ndarray] = []
    point_offset = 0
    if "vertices" in old:
        for index in range(count):
            if not math.isclose(
                float(old["clipped_area"][index]), float(old["original_area"][index]), rel_tol=1e-12, abs_tol=1e-12
            ):
                polygon = np.asarray(old["vertices"][index, : int(old["vertex_count"][index])], dtype=np.float64)
                clipped[index] = True
                clipped_offsets[index] = point_offset
                clipped_counts[index] = len(polygon)
                clipped_parts.append(polygon)
                point_offset += len(polygon)
    record_values: list[str] = []
    record_ref = np.full(count, -1, dtype=np.int32)
    for index, value in enumerate(old.get("observation_record_id", np.asarray([""] * count))):
        if str(value):
            record_ref[index] = len(record_values)
            record_values.append(str(value))
    compact = {
        "ordinal": np.arange(count, dtype=np.uint64),
        "center": np.asarray(old["center"], dtype=np.float64),
        "normal": np.asarray(old["normal"], dtype=np.float64),
        "radius": np.asarray(old["radius"], dtype=np.float64),
        "original_area": np.asarray(old["original_area"], dtype=np.float64),
        "clipped_area": np.asarray(old["clipped_area"], dtype=np.float64),
        "domain_id": np.asarray(old["domain_id"], dtype=np.int32),
        "set_id": np.asarray(old["set_id"], dtype=np.int16),
        "source_code": np.asarray([SOURCE_CODES[str(value)] for value in old["source"]], dtype=np.uint8),
        "voxel_index": np.asarray(old["voxel_index"], dtype=np.int32),
        "clipped": clipped,
        "clipped_offsets": clipped_offsets,
        "clipped_counts": clipped_counts,
        "clipped_points": np.concatenate(clipped_parts) if clipped_parts else np.empty((0, 3), dtype=np.float64),
        "distribution_code": np.asarray(
            [DISTRIBUTION_CODES.get(str(value), np.uint8(255)) for value in old["distribution"]], dtype=np.uint8
        ),
        "size_source_code": np.asarray(
            [SIZE_SOURCE_CODES.get(str(value), np.uint8(255)) for value in old["size_source"]], dtype=np.uint8
        ),
        "orientation_source_code": np.asarray(
            [ORIENTATION_CODES.get(str(value), np.uint8(255)) for value in old["orientation_source"]], dtype=np.uint8
        ),
        "dip_only_density_evidence": np.asarray(old["dip_only_density_evidence"], dtype=np.bool_),
        "record_ref": record_ref,
        "size_class": np.zeros(count, dtype=np.uint8),
    }
    realization.geometry_arrays = compact
    realization.provenance.update(
        {
            "geometry_format": "columnar-v2",
            "migrated_from": "m10-object-array-v1",
            "record_ids": record_values,
            "source_codes": {str(code): name for code, name in SOURCE_NAMES.items()},
            "distribution_codes": {str(int(code)): name for name, code in DISTRIBUTION_CODES.items()},
            "size_source_codes": {str(int(code)): name for name, code in SIZE_SOURCE_CODES.items()},
            "orientation_source_codes": {str(int(code)): name for name, code in ORIENTATION_CODES.items()},
            "size_class_status": "legacy_unknown_all_explicit",
            "p32_subgrid": 0.0,
        }
    )
    return True
