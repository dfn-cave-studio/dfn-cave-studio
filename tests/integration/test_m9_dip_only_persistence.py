"""Import and .dfnproj round-trip tests for optional fracture dip direction."""

import json
import zipfile
from pathlib import Path

import pandas as pd

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.import_service import read_input_table


def _repository_with_collar() -> tuple[Project, BoreholeRepository]:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "BH", "collar_x": 0, "collar_y": 0, "collar_z": 100, "final_depth": 100}]),
        "collars.csv",
    )
    return project, repository


def test_csv_without_direction_column_imports_dip_only(tmp_path) -> None:
    path = tmp_path / "fractures-no-direction.csv"
    pd.DataFrame(
        [
            {"hole_id": "BH", "depth": 10, "dip": 40, "set_id": 1},
            {"hole_id": "BH", "depth": 20, "dip": 50, "set_id": 1},
        ]
    ).to_csv(path, index=False)
    project, repository = _repository_with_collar()
    result = repository.import_dataframe("fractures", read_input_table(path), str(path))
    assert result["formal"] == 2
    assert result["dip_only"] == 2
    assert project.borehole_database.orientation_counts() == {"full_orientation": 0, "dip_only": 2}


def test_xlsx_without_direction_and_partial_direction_import(tmp_path) -> None:
    path = tmp_path / "fractures-mixed.xlsx"
    pd.DataFrame(
        [
            {"hole_id": "BH", "depth": 10, "dip": 40, "dip_direction": None},
            {"hole_id": "BH", "depth": 20, "dip": 50, "dip_direction": 0},
            {"hole_id": "BH", "depth": 30, "dip": 60, "dip_direction": "NA"},
        ]
    ).to_excel(path, index=False)
    project, repository = _repository_with_collar()
    result = repository.import_dataframe("fractures", read_input_table(path), str(path))
    assert result["formal"] == 3
    assert result["full_orientation"] == 1
    assert result["dip_only"] == 2
    values = [record.values["dip_direction"] for record in repository.query("fractures", "formal")]
    assert values == [None, 0.0, None]


def test_dfnproj_round_trip_uses_json_null_and_preserves_real_zero(tmp_path) -> None:
    project, repository = _repository_with_collar()
    repository.import_dataframe(
        "fractures",
        pd.DataFrame(
            [
                {"hole_id": "BH", "depth": 10, "dip": 40, "dip_direction": None},
                {"hole_id": "BH", "depth": 20, "dip": 50, "dip_direction": 0},
            ]
        ),
        "fractures.csv",
    )
    path = tmp_path / "dip-only.dfnproj"
    ZipProjectStore().save(project, path)
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read("inputs/borehole_database.json"))
    fracture_values = [row["values"] for row in payload["records"] if row["data_type"] == "fractures"]
    assert fracture_values[0]["dip_direction"] is None
    assert fracture_values[1]["dip_direction"] == 0.0

    reopened = ZipProjectStore().load(path)
    reopened_values = [
        record.values["dip_direction"] for record in reopened.borehole_database.query("fractures", "formal")
    ]
    assert reopened_values == [None, 0.0]
    assert [observation.dip_direction for observation in reopened.borehole_collection["BH"].fracture_observations] == [
        None,
        0.0,
    ]


def test_v090_migration_preserves_ambiguous_zero_with_audit_warning(tmp_path) -> None:
    project, repository = _repository_with_collar()
    repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "BH", "depth": 10, "dip": 40, "dip_direction": 0}]),
        "fractures.csv",
    )
    record = repository.query("fractures", "formal")[0]
    record.values.pop("orientation_completeness")
    project.borehole_database.schema_version = 1
    path = tmp_path / "legacy-v090.dfnproj"
    ZipProjectStore().save(project, path)

    reopened = ZipProjectStore().load(path)
    migrated = reopened.borehole_database.query("fractures", "formal")[0]
    assert migrated.values["dip_direction"] == 0.0
    assert migrated.values["orientation_completeness"] == "full_orientation"
    assert migrated.modification_history[-1].action == "ambiguous_legacy_orientation_preserved_zero"


def test_mixed_orientation_fixture_has_exact_audit_counts() -> None:
    root = Path(__file__).parents[2] / "examples" / "m9_dip_only"
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe("collars", read_input_table(root / "collars.csv"), str(root / "collars.csv"))
    result = repository.import_dataframe(
        "fractures",
        read_input_table(root / "fractures_mixed.csv"),
        str(root / "fractures_mixed.csv"),
    )
    assert result["total_fracture_rows"] == 12
    assert result["formal"] == 10
    assert result["excluded"] == 2
    assert result["full_orientation"] == 4
    assert result["dip_only"] == 6
    assert project.borehole_database.orientation_counts() == {"full_orientation": 4, "dip_only": 6}
