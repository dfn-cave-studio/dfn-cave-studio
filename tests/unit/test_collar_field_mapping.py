"""Regression coverage for normalized, auditable collar header aliases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter, STANDARD_COLLAR_FIELDS
from dfn_cave_studio.models.data_management import FieldMapping
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.field_mapping import (
    FieldMappingConflictError,
    detect_field_mapping,
    normalize_header,
)
from dfn_cave_studio.services.import_service import UnifiedImportService, read_input_table

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (
            ["borehole_id", "collar_x", "collar_y", "collar_z", "final_depth"],
            {
                "borehole_id": "borehole_id",
                "collar_x": "collar_x",
                "collar_y": "collar_y",
                "collar_z": "collar_z",
                "final_depth": "final_depth",
            },
        ),
        (
            ["hole_id", "easting", "northing", "elevation", "total_depth"],
            {
                "hole_id": "borehole_id",
                "easting": "collar_x",
                "northing": "collar_y",
                "elevation": "collar_z",
                "total_depth": "final_depth",
            },
        ),
        (
            ["HoleName", "East", "North", "RL", "HoleLength"],
            {
                "HoleName": "borehole_id",
                "East": "collar_x",
                "North": "collar_y",
                "RL": "collar_z",
                "HoleLength": "final_depth",
            },
        ),
        (
            ["  HoLe-Id  ", " EASTING ", " NORTHING ", " Elevation", "DEPTH TOTAL "],
            {
                "  HoLe-Id  ": "borehole_id",
                " EASTING ": "collar_x",
                " NORTHING ": "collar_y",
                " Elevation": "collar_z",
                "DEPTH TOTAL ": "final_depth",
            },
        ),
    ],
)
def test_required_collar_alias_groups_are_detected(headers, expected):
    detection = detect_field_mapping(headers, STANDARD_COLLAR_FIELDS)
    assert detection.conflicts == []
    assert {source: target for source, target in detection.mapping.items() if target in expected.values()} == expected


def test_bom_normalization_and_plain_depth_is_not_final_depth():
    assert normalize_header(" \ufeffHole Name ") == "hole_name"
    detection = detect_field_mapping(
        ["\ufeffHoleName", "East", "North", "RL", "depth"],
        STANDARD_COLLAR_FIELDS,
    )
    assert detection.mapping["\ufeffHoleName"] == "borehole_id"
    assert "depth" not in detection.mapping
    assert "final_depth" not in detection.mapping.values()


def test_multiple_candidates_for_one_standard_field_are_an_explicit_conflict():
    detection = detect_field_mapping(
        ["borehole_id", "hole_id", "collar_x", "collar_y", "collar_z", "final_depth"],
        STANDARD_COLLAR_FIELDS,
    )
    assert "borehole_id" not in detection.mapping.values()
    with pytest.raises(FieldMappingConflictError, match="multiple candidate columns"):
        detection.require_unambiguous()


def test_one_source_matching_multiple_standard_fields_is_an_explicit_conflict():
    detection = detect_field_mapping(["Shared"], {"field_a": ["shared"], "field_b": ["SHARED"]})
    assert detection.mapping == {}
    with pytest.raises(FieldMappingConflictError, match="source column 'Shared' matches multiple fields"):
        detection.require_unambiguous()


def test_csv_bom_and_xlsx_preview_detection_are_identical_to_import(tmp_path: Path):
    dataframe = pd.DataFrame(
        [{"HoleName": "BH-X", "East": 11.0, "North": 22.0, "RL": 333.0, "HoleLength": 44.0}]
    )
    csv_path = tmp_path / "collars.csv"
    csv_path.write_text(
        "\ufeffHoleName,East,North,RL,HoleLength\nBH-X,11,22,333,44\n",
        encoding="utf-8",
    )
    xlsx_path = tmp_path / "collars.xlsx"
    dataframe.to_excel(xlsx_path, index=False, sheet_name="Sheet1")

    service = UnifiedImportService()
    expected = {
        "borehole_id": "HoleName",
        "collar_x": "East",
        "collar_y": "North",
        "collar_z": "RL",
        "final_depth": "HoleLength",
    }
    for path in (csv_path, xlsx_path):
        preview = read_input_table(path, nrows=50)
        assert {key: value for key, value in service.detect_fields(preview, "collars").items() if key in expected} == expected
        result = service.import_project(collars=str(path))
        assert result.success
        assert result.collars is not None
        assert result.collars.rows_imported == 1


def test_repository_keeps_original_headers_mapping_and_correct_coordinates():
    source = pd.DataFrame(
        [{"HoleName": "BH-P", "East": 101, "North": 202, "RL": 503, "HoleLength": 154}]
    )
    project = Project()
    repository = BoreholeRepository(project)
    result = repository.import_dataframe("collars", source, "m10-collars.csv")
    assert result["formal"] == 1
    record = repository.query("collars", raw=True)[0]
    assert record.original_values == source.iloc[0].to_dict()
    assert record.source_field_mapping == {
        "HoleName": "borehole_id",
        "East": "collar_x",
        "North": "collar_y",
        "RL": "collar_z",
        "HoleLength": "final_depth",
    }
    assert record.normalized_source_headers["HoleName"] == "holename"
    collar = list(project.borehole_collection)[0].collar
    assert (collar.collar_x, collar.collar_y, collar.collar_z, collar.final_depth) == (101, 202, 503, 154)


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "examples/m7_demo/collars.csv",
        ROOT / "examples/m9_dip_only/collars.csv",
        ROOT / "examples/m10_demo/collars.csv",
    ],
)
def test_published_demo_collar_files_import_without_manual_renaming(path: Path):
    result = BoreholeImporter().import_all(collar_path=str(path))
    assert result.collection is not None
    assert len(result.collection) > 0


def test_explicit_mapping_conflict_and_missing_required_field_are_rejected():
    service = UnifiedImportService()
    conflict = pd.DataFrame(
        [{"hole_id": "A", "borehole_id": "B", "x": 1, "y": 2, "z": 3, "final_depth": 4}]
    )
    with pytest.raises(FieldMappingConflictError, match="multiple candidate columns"):
        service.detect_fields(conflict, "collars")

    incomplete = pd.DataFrame([{"hole_id": "A", "easting": 1, "northing": 2, "elevation": 3}])
    project = Project()
    result = BoreholeRepository(project).import_dataframe("collars", incomplete, "missing.csv")
    assert result["formal"] == 0
    assert result["excluded"] == 1


def test_manual_field_mapping_remains_supported():
    dataframe = pd.DataFrame([{"Name": "BH-M", "CX": 1, "CY": 2, "CZ": 3, "TD": 4}])
    mapping = FieldMapping(
        data_type="collars",
        column_map={"Name": "borehole_id", "CX": "collar_x", "CY": "collar_y", "CZ": "collar_z", "TD": "final_depth"},
    )
    mapped = UnifiedImportService().apply_field_mapping(dataframe, mapping)
    assert list(mapped.columns) == ["borehole_id", "collar_x", "collar_y", "collar_z", "final_depth"]
