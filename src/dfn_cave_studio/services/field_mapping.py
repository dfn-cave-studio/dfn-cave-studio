"""Shared, auditable field-name matching for borehole table imports."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Mapping, Sequence


COLLAR_FIELD_ALIASES: dict[str, list[str]] = {
    "borehole_id": [
        "borehole_id",
        "hole_id",
        "holeid",
        "borehole",
        "HoleName",
        "hole_name",
        "bh_id",
        "hole",
        "id",
    ],
    "collar_x": ["collar_x", "easting", "east", "x", "East", "x_coord"],
    "collar_y": ["collar_y", "northing", "north", "y", "North", "y_coord"],
    "collar_z": ["collar_z", "elevation", "rl", "z", "RL", "elev", "z_coord"],
    "azimuth": ["azimuth", "az", "bearing", "azm"],
    "dip": ["dip", "inclination", "incl", "dip_angle"],
    "final_depth": [
        "final_depth",
        "total_depth",
        "hole_length",
        "depth_total",
        "HoleLength",
        "length",
        "eoh",
    ],
}


class FieldMappingConflictError(ValueError):
    """Raised when automatic or user field mappings are ambiguous."""


@dataclass(frozen=True)
class FieldMappingDetection:
    """Result of matching original source headers to standard fields."""

    mapping: dict[str, str]
    normalized_headers: dict[str, str]
    conflicts: list[str] = field(default_factory=list)

    def require_unambiguous(self) -> dict[str, str]:
        """Return the mapping or raise an explicit conflict error."""
        if self.conflicts:
            raise FieldMappingConflictError("Field mapping conflict: " + "; ".join(self.conflicts))
        return dict(self.mapping)


def normalize_header(header: object) -> str:
    """Normalize a source header without losing its original spelling."""
    value = str(header).strip().lstrip("\ufeff").strip().casefold()
    value = re.sub(r"[\s-]+", "_", value)
    return re.sub(r"_+", "_", value)


def detect_field_mapping(
    columns: Iterable[object],
    field_definitions: Mapping[str, Sequence[str]],
) -> FieldMappingDetection:
    """Detect a one-to-one source-to-standard field mapping.

    Ambiguous matches are deliberately omitted from ``mapping`` and returned
    in ``conflicts`` so a UI can require explicit user confirmation.
    """
    source_columns = [str(column) for column in columns]
    normalized_headers = {column: normalize_header(column) for column in source_columns}
    aliases_by_standard = {
        standard: {normalize_header(standard), *(normalize_header(alias) for alias in aliases)}
        for standard, aliases in field_definitions.items()
    }

    candidates_by_source: dict[str, list[str]] = {}
    candidates_by_standard: dict[str, list[str]] = {standard: [] for standard in field_definitions}
    conflicts: list[str] = []
    for source, normalized in normalized_headers.items():
        candidates = [standard for standard, aliases in aliases_by_standard.items() if normalized in aliases]
        candidates_by_source[source] = candidates
        if len(candidates) > 1:
            conflicts.append(f"source column '{source}' matches multiple fields: {', '.join(candidates)}")
        elif candidates:
            candidates_by_standard[candidates[0]].append(source)

    mapping: dict[str, str] = {}
    for standard, candidates in candidates_by_standard.items():
        if len(candidates) > 1:
            conflicts.append(
                f"standard field '{standard}' has multiple candidate columns: {', '.join(repr(value) for value in candidates)}"
            )
        elif len(candidates) == 1 and len(candidates_by_source[candidates[0]]) == 1:
            mapping[candidates[0]] = standard
    return FieldMappingDetection(mapping=mapping, normalized_headers=normalized_headers, conflicts=conflicts)


def validate_field_mapping(
    columns: Iterable[object],
    mapping: Mapping[str, str],
    standard_fields: Iterable[str],
) -> dict[str, str]:
    """Validate an explicit one-to-one source-to-standard mapping."""
    source_columns = {str(column) for column in columns}
    allowed = set(standard_fields)
    cleaned = {str(source): str(target).strip() for source, target in mapping.items() if str(target).strip()}
    missing_sources = sorted(source for source in cleaned if source not in source_columns)
    invalid_targets = sorted({target for target in cleaned.values() if target not in allowed})
    targets: dict[str, list[str]] = {}
    for source, target in cleaned.items():
        targets.setdefault(target, []).append(source)
    duplicates = {target: sources for target, sources in targets.items() if len(sources) > 1}
    conflicts: list[str] = []
    if missing_sources:
        conflicts.append(f"unknown source columns: {', '.join(missing_sources)}")
    if invalid_targets:
        conflicts.append(f"unknown standard fields: {', '.join(invalid_targets)}")
    conflicts.extend(
        f"standard field '{target}' is mapped from multiple columns: {', '.join(repr(value) for value in sources)}"
        for target, sources in duplicates.items()
    )
    if conflicts:
        raise FieldMappingConflictError("Field mapping conflict: " + "; ".join(conflicts))
    return cleaned
