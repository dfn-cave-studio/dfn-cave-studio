from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from dfn_cave_studio.models.borehole_database import BoreholeRecord, RecordState
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.observation_service import ObservationService


def _p(**updates):
    row = {
        "observation_id": "P001-A",
        "point_id": "001",
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "dip": 45.0,
        "dip_direction": 120.0,
        "local_set_id": "A",
        "joint_spacing_m": 0.5,
        "joint_num": 3,
    }
    row.update(updates)
    return row


def _z(**updates):
    row = {
        "observation_id": "Z001-A",
        "point_id": "001",
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "dip": 45.0,
        "dip_direction": 120.0,
        "local_set_id": "A",
        "joint_spacing_m": None,
        "joint_num": None,
    }
    row.update(updates)
    return row


def _import(rows):
    project = Project()
    repository = BoreholeRepository(project)
    result = repository.import_dataframe("orientation_points", pd.DataFrame(rows), "synthetic.csv")
    return project, repository, result


def test_point_cloud_dominant_and_random_contracts() -> None:
    project, _, result = _import([_p(), _p(observation_id="P001-R", local_set_id="RANDOM", dip=None, dip_direction=None)])
    assert result["formal"] == 2
    observations = ObservationService(project).orientation_points()
    assert [item.source_kind for item in observations] == ["POINT_CLOUD", "POINT_CLOUD"]
    assert observations[1].component_type == "RANDOM_BACKGROUND"
    assert observations[1].orientation_status == "MISSING"
    assert all(item.set_id is None and item.calibration_role == "UNASSIGNED" for item in observations)


def test_point_cloud_without_random_and_camera_without_density_are_valid() -> None:
    project, _, result = _import([_p(), _z(joint_spacing_m="", joint_num="")])
    assert result["formal"] == 2
    items = ObservationService(project).orientation_points()
    assert items[0].point_key == "POINT_CLOUD:001"
    assert items[1].point_key == "BOREHOLE_CAMERA:001"
    assert items[1].joint_spacing_m is None and items[1].joint_num is None


def test_spacing_measurement_basis_is_explicit_and_does_not_create_p32() -> None:
    project, _, result = _import([_p(measurement_basis="TRUE_NORMAL"), _p(observation_id="P002-A")])
    assert result["formal"] == 2
    items = ObservationService(project).orientation_points()
    assert items[0].measurement_basis == "TRUE_NORMAL"
    assert items[1].measurement_basis is None
    assert items[0].approximate_p32_from_true_normal_spacing() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="only valid for TRUE_NORMAL"):
        items[1].approximate_p32_from_true_normal_spacing()
    assert project.m9_state.p32_estimates == []
    assert project.m9_state.parameter_field_arrays == {}


@pytest.mark.parametrize(
    "row,field",
    [
        (_z(dip=None), "dip"),
        (_z(joint_spacing_m=0.5), "joint_spacing_m"),
        (_z(joint_num=2), "joint_num"),
        (_p(observation_id="X001-A"), "observation_id"),
        (_p(joint_num=1.5), "joint_num"),
        (_p(joint_spacing_m=0), "joint_spacing_m"),
        (_p(measurement_basis="UNKNOWN"), "measurement_basis"),
        (_p(dip=91), "dip"),
        (_p(local_set_id="RANDOM", dip=None, dip_direction=10), "dip_direction"),
    ],
)
def test_invalid_conditional_fields_are_excluded_with_auditable_error(row, field) -> None:
    _, repository, result = _import([row])
    assert result["excluded"] == 1
    reason = repository.query("orientation_points", RecordState.EXCLUDED)[0].exclusion_reason
    assert "Row 2" in reason and str(row["observation_id"]) in reason and field in reason


def test_duplicate_random_and_same_point_coordinate_conflict_are_rejected() -> None:
    _, repository, result = _import(
        [
            _p(observation_id="P001-R1", local_set_id="RANDOM", dip=None, dip_direction=None),
            _p(observation_id="P001-R2", local_set_id="RANDOM", dip=None, dip_direction=None),
            _p(observation_id="P001-B", local_set_id="B", x=9.0),
        ]
    )
    assert result["formal"] == 1 and result["excluded"] == 2
    reasons = [record.exclusion_reason for record in repository.query("orientation_points", RecordState.EXCLUDED)]
    assert any("duplicate RANDOM" in reason for reason in reasons)
    assert any("x/y/z conflict" in reason for reason in reasons)


def test_observation_id_is_global_and_duplicate_signature_contains_density_contract() -> None:
    project, repository, _ = _import([_p()])
    duplicate = repository.import_dataframe("orientation_points", pd.DataFrame([_p()]), "synthetic.csv")
    changed_density = repository.import_dataframe(
        "orientation_points", pd.DataFrame([_p(joint_spacing_m=0.25)]), "synthetic.csv"
    )
    assert duplicate["duplicates"] == 1
    assert changed_density["duplicates"] == 0 and changed_density["excluded"] == 1
    assert len(project.borehole_database.records) == 2


@pytest.mark.parametrize("spacing,expected_rqd", [(0.25, 100.0), (1 / 4.0001, 110 - 2.5 * 4.0001), (1 / 44.0, 0.0)])
def test_jv_rqd_piecewise_boundaries(spacing, expected_rqd) -> None:
    project, _, _ = _import([_p(joint_spacing_m=spacing)])
    summary = ObservationService(project).orientation_point_summaries()[0]
    assert summary.jv_estimated == pytest.approx(1 / spacing)
    assert summary.rqd_from_jv == pytest.approx(expected_rqd)


def test_jv_uses_spacing_not_joint_num_and_camera_is_excluded() -> None:
    project, repository, _ = _import([_p(joint_num=999), _z()])
    summaries = ObservationService(project).orientation_point_summaries()
    assert len(summaries) == 1 and summaries[0].jv_estimated == pytest.approx(2.0)
    assert project.m9_state.scalar_samples == []
    record = repository.query("orientation_points", RecordState.FORMAL)[0]
    before = project.borehole_database.model_copy(deep=True)
    with pytest.raises(ValueError):
        repository.edit_record(record.record_id, {**record.values, "x": float("nan")})
    assert project.borehole_database == before


def test_edit_delete_and_transaction_rollback_keep_summary_consistent() -> None:
    project, repository, _ = _import([_p()])
    record = repository.query("orientation_points", RecordState.FORMAL)[0]
    repository.edit_record(record.record_id, {**record.values, "joint_spacing_m": 0.25})
    assert project.borehole_database.orientation_point_summaries[0].jv_estimated == pytest.approx(4.0)
    snapshot = project.borehole_database.model_copy(deep=True)
    with pytest.raises(RuntimeError):
        with repository.transaction():
            repository.delete_record(record.record_id)
            raise RuntimeError("synthetic cancellation")
    assert project.borehole_database == snapshot
    repository.delete_record(record.record_id)
    assert project.borehole_database.orientation_point_summaries == []


def test_random_component_is_included_in_jv_audit() -> None:
    project, _, _ = _import(
        [_p(), _p(observation_id="P001-R", local_set_id="RANDOM", dip=None, dip_direction=None, joint_spacing_m=1.0)]
    )
    summary = project.borehole_database.orientation_point_summaries[0]
    assert summary.includes_random
    assert summary.jv_estimated == pytest.approx(3.0)
    assert set(summary.input_observation_ids) == {"P001-A", "P001-R"}


def test_legacy_orientation_point_without_new_fields_remains_readable() -> None:
    project = Project()
    project.borehole_database.records.append(
        BoreholeRecord(
            data_type="orientation_points",
            source_file="legacy.json",
            source_row=0,
            original_values={"point_id": "legacy"},
            values={
                "observation_id": "OP-legacy",
                "point_id": "legacy",
                "x": 1,
                "y": 2,
                "z": 3,
                "dip": 30,
                "dip_direction": 40,
                "set_id": None,
            },
            state=RecordState.FORMAL,
        )
    )
    item = ObservationService(project).orientation_points()[0]
    assert item.source_kind == "POINT_CLOUD"
    assert item.local_set_id == "LEGACY"
    assert item.calibration_role == "UNASSIGNED"


def test_public_synthetic_orientation_example_uses_final_contract() -> None:
    path = Path(__file__).parents[2] / "examples" / "multisource_observation_synthetic" / "orientation_points.csv"
    project = Project()
    result = BoreholeRepository(project).import_dataframe("orientation_points", pd.read_csv(path), str(path))
    assert result["formal"] == 3
    assert {item.source_kind for item in ObservationService(project).orientation_points()} == {
        "POINT_CLOUD",
        "BOREHOLE_CAMERA",
    }
