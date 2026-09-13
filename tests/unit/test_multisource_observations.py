"""Synthetic contracts for multi-source observation imports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfn_cave_studio.models.borehole_database import BoreholeDataType, FractureObservationMode, RecordState
from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, ScalarFieldMetadata, ScalarFieldResult
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.services.observation_service import ObservationService
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.services.m9_service import M9Service


def _project() -> tuple[Project, BoreholeRepository]:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        BoreholeDataType.COLLARS,
        pd.DataFrame(
            [
                {
                    "borehole_id": "SYN-1",
                    "collar_x": 10.0,
                    "collar_y": 20.0,
                    "collar_z": 100.0,
                    "final_depth": 20.0,
                    "azimuth": 90.0,
                    "dip": -45.0,
                    "domain_id": None,
                }
            ]
        ),
        "synthetic-collars.csv",
    )
    repository.import_dataframe(
        BoreholeDataType.SURVEYS,
        pd.DataFrame(
            [
                {"hole_id": "SYN-1", "measured_depth": 0.0, "azimuth": 90.0, "dip": -45.0},
                {"hole_id": "SYN-1", "measured_depth": 20.0, "azimuth": 90.0, "dip": -45.0},
            ]
        ),
        "synthetic-surveys.csv",
    )
    return project, repository


def test_optional_collar_domain_and_blank_rows_are_preserved_or_skipped() -> None:
    project = Project()
    result = BoreholeRepository(project).import_dataframe(
        BoreholeDataType.COLLARS,
        pd.DataFrame(
            [
                {"borehole_id": "SYN-1", "collar_x": 0, "collar_y": 0, "collar_z": 10, "final_depth": 5},
                {"borehole_id": None, "collar_x": None, "collar_y": None, "collar_z": None, "final_depth": None},
            ]
        ),
        "synthetic.csv",
    )
    assert result == {
        "raw": 1,
        "formal": 1,
        "excluded": 0,
        "pending": 0,
        "duplicates": 0,
        "blank_rows_skipped": 1,
        "cancelled": False,
        "data_type": "collars",
    }
    values = project.borehole_database.records[0].values
    assert values.get("domain_id") is None


def test_partial_collar_row_is_excluded_not_silently_skipped() -> None:
    project = Project()
    result = BoreholeRepository(project).import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "SYN-1", "collar_x": 0, "collar_y": None, "collar_z": 1, "final_depth": 5}]),
        "synthetic.csv",
    )
    assert result.get("blank_rows_skipped", 0) == 0
    assert result["excluded"] == 1


@pytest.mark.parametrize(
    ("mode", "row"),
    [
        (
            FractureObservationMode.FULL_ORIENTATION,
            {"hole_id": "SYN-1", "depth": 5, "dip_direction": 120, "dip": 40},
        ),
        (
            FractureObservationMode.INTERVAL_SPACING,
            {"hole_id": "SYN-1", "from_depth": 2, "to_depth": 8, "fracture_spacing": 10},
        ),
        (
            FractureObservationMode.AXIS_PLANE_ANGLE,
            {"hole_id": "SYN-1", "depth": 6, "axis_plane_angle": 35},
        ),
    ],
)
def test_three_fracture_modes_import_without_synthesizing_missing_orientation(mode, row) -> None:
    project, repository = _project()
    metadata = {"observation_mode": mode}
    if mode == FractureObservationMode.INTERVAL_SPACING:
        metadata["spacing_unit"] = "cm"
    result = repository.import_dataframe(
        "fractures", pd.DataFrame([row]), "synthetic.csv", import_metadata=metadata
    )
    assert result["formal"] == 1
    record = repository.query("fractures", RecordState.FORMAL)[0]
    assert record.values["observation_mode"] == mode
    if mode != FractureObservationMode.FULL_ORIENTATION:
        assert len(project.borehole_collection["SYN-1"].fracture_observations) == 0


def test_spacing_centimetres_derives_p10_without_creating_fractures() -> None:
    project, repository = _project()
    repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 10, "fracture_spacing": 10}]),
        "synthetic-spacing.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "cm"},
    )
    observation = ObservationService(project).spacing_observations()[0]
    assert observation.derived_p10 == pytest.approx(10.0)
    assert observation.spacing_unit == "cm"
    assert observation.measurement_basis == "BOREHOLE_ALONG_HOLE"
    assert project.borehole_collection["SYN-1"].fracture_observations == []


@pytest.mark.parametrize("measurement_basis", ["BOREHOLE_ALONG_HOLE", "TRUE_NORMAL", "SCANLINE_APPARENT"])
def test_spacing_measurement_basis_is_preserved_without_implicit_p32(
    measurement_basis: str,
) -> None:
    project, repository = _project()
    result = repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 10, "fracture_spacing": 2}]),
        "synthetic-spacing.csv",
        import_metadata={
            "observation_mode": "interval_spacing",
            "spacing_unit": "m",
            "measurement_basis": measurement_basis,
        },
    )
    assert result["formal"] == 1
    observation = ObservationService(project).spacing_observations()[0]
    assert observation.measurement_basis == measurement_basis
    assert observation.derived_p10 == pytest.approx(0.5)
    assert not hasattr(observation, "derived_p32")


def test_new_fracture_modes_are_idempotent_but_mode_and_unit_are_part_of_identity() -> None:
    _project_value, repository = _project()
    row = pd.DataFrame(
        [{"hole_id": "SYN-1", "depth": 5, "dip": 40, "axis_plane_angle": 40}]
    )
    first = repository.import_dataframe(
        "fractures", row, "synthetic.csv", import_metadata={"observation_mode": "global_dip_only"}
    )
    duplicate = repository.import_dataframe(
        "fractures", row, "synthetic.csv", import_metadata={"observation_mode": "global_dip_only"}
    )
    different_mode = repository.import_dataframe(
        "fractures", row, "synthetic.csv", import_metadata={"observation_mode": "axis_plane_angle"}
    )
    spacing = pd.DataFrame(
        [{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "fracture_spacing": 10}]
    )
    centimetres = repository.import_dataframe(
        "fractures",
        spacing,
        "spacing.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "cm"},
    )
    metres = repository.import_dataframe(
        "fractures",
        spacing,
        "spacing.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "m"},
    )
    assert (first["raw"], duplicate["raw"], duplicate["duplicates"], different_mode["raw"]) == (1, 0, 1, 1)
    assert (centimetres["raw"], metres["raw"]) == (1, 1)
    derived = sorted(
        record.values["derived_p10"]
        for record in repository.query("fractures", RecordState.FORMAL)
        if record.values.get("observation_mode") == "interval_spacing"
    )
    assert derived == [0.1, 10.0]


@pytest.mark.parametrize(
    ("mode", "row"),
    [
        ("interval_spacing", {"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "fracture_spacing": 10}),
        ("axis_plane_angle", {"hole_id": "SYN-1", "depth": 5, "axis_plane_angle": 30}),
    ],
)
def test_m9_rejects_projects_with_only_non_global_orientation_records(mode, row) -> None:
    project, repository = _project()
    metadata = {"observation_mode": mode}
    if mode == "interval_spacing":
        metadata["spacing_unit"] = "cm"
    repository.import_dataframe("fractures", pd.DataFrame([row]), "synthetic.csv", import_metadata=metadata)
    with pytest.raises(ValueError, match="No global fracture orientations.*direction completion is not implemented"):
        M9Service(project).calculate_density(DensitySettings(monte_carlo_samples=100))


def test_mixed_orientation_modes_use_complete_path_and_report_excluded_records() -> None:
    project, repository = _project()
    repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "depth": 4, "dip_direction": 90, "dip": 45}]),
        "full.csv",
        import_metadata={"observation_mode": "full_orientation"},
    )
    repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "fracture_spacing": 10}]),
        "spacing.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "cm"},
    )
    repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "depth": 6, "axis_plane_angle": 30}]),
        "relative.csv",
        import_metadata={"observation_mode": "axis_plane_angle"},
    )
    holdout = HoldoutService()
    holdout.select_manual(["SYN-1"], [])
    holdout.lock()
    set_holdout(project, holdout)
    M9Service(project).calculate_density(DensitySettings(monte_carlo_samples=100))
    assert len(project.borehole_collection["SYN-1"].fracture_observations) == 1
    assert project.m9_state.provenance["non_global_orientation_records_excluded_from_density_fit"] == {
        "interval_spacing": 1,
        "axis_plane_angle": 1,
        "reason": "direction completion is not implemented in this phase",
    }


@pytest.mark.parametrize("spacing", [0, -1])
def test_nonpositive_spacing_is_excluded(spacing: float) -> None:
    _project_value, repository = _project()
    result = repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "fracture_spacing": spacing}]),
        "synthetic.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "cm"},
    )
    assert result["excluded"] == 1


def test_unknown_hole_and_out_of_range_relative_angle_do_not_enter_formal_projection() -> None:
    project, repository = _project()
    pending = repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "UNKNOWN", "depth": 1, "axis_plane_angle": 30}]),
        "synthetic.csv",
        import_metadata={"observation_mode": "axis_plane_angle"},
    )
    invalid = repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "depth": 1, "axis_plane_angle": 91}]),
        "synthetic-invalid.csv",
        import_metadata={"observation_mode": "axis_plane_angle"},
    )
    assert pending["pending"] == 1
    assert invalid["excluded"] == 1
    assert project.borehole_collection["SYN-1"].fracture_observations == []


def test_explicit_full_orientation_requires_dip_direction_but_legacy_dip_only_remains_supported() -> None:
    project, repository = _project()
    row = pd.DataFrame([{"hole_id": "SYN-1", "depth": 1, "dip": 45}])
    full = repository.import_dataframe(
        "fractures", row, "synthetic-full.csv", import_metadata={"observation_mode": "full_orientation"}
    )
    legacy = repository.import_dataframe(
        "fractures", row, "synthetic-legacy.csv", import_metadata={"observation_mode": "global_dip_only"}
    )
    assert full["excluded"] == 1
    assert legacy["formal"] == 1
    assert len(project.borehole_collection["SYN-1"].fracture_observations) == 1


def test_axis_angle_is_trajectory_located_and_not_written_as_global_dip() -> None:
    project, repository = _project()
    repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "depth": 10, "axis_plane_angle": 30}]),
        "synthetic.csv",
        import_metadata={"observation_mode": "axis_plane_angle"},
    )
    observation = ObservationService(project).axis_plane_angle_observations()[0]
    assert observation.reference_frame == "BOREHOLE_RELATIVE"
    assert observation.x > 10.0 and observation.z < 100.0
    assert observation.z != pytest.approx(10.0)
    record = repository.query("fractures", RecordState.FORMAL)[0]
    assert record.values.get("dip") is None
    assert record.values.get("dip_direction") is None
    issues = BoreholeQualityService(project).run_checks()
    assert any(issue.code == "borehole_relative_orientation" and issue.severity == "info" for issue in issues)


def test_cross_domain_associations_preserve_one_original_observation() -> None:
    project, repository = _project()
    repository.import_dataframe(
        "domain_intervals",
        pd.DataFrame(
            [
                {"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "domain_id": 2},
                {"hole_id": "SYN-1", "from_depth": 5, "to_depth": 10, "domain_id": 4},
            ]
        ),
        "synthetic-domains.csv",
    )
    repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 2, "to_depth": 8, "fracture_spacing": 20}]),
        "synthetic-spacing.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "cm"},
    )
    observations = ObservationService(project).spacing_observations()
    assert len(observations) == 1
    assert [(item.from_depth, item.to_depth, item.domain_id) for item in observations[0].domain_segments] == [
        (2.0, 5.0, 2),
        (5.0, 8.0, 4),
    ]
    assert sum(item.to_depth - item.from_depth for item in observations[0].domain_segments) == pytest.approx(6.0)


def test_overlapping_or_uncovered_domain_intervals_still_partition_length_once() -> None:
    project, repository = _project()
    repository.import_dataframe(
        "domain_intervals",
        pd.DataFrame(
            [
                {"hole_id": "SYN-1", "from_depth": 1, "to_depth": 6, "domain_id": 2},
                {"hole_id": "SYN-1", "from_depth": 4, "to_depth": 8, "domain_id": 4},
            ]
        ),
        "synthetic-overlap.csv",
    )
    segments = ObservationService(project).domain_associations("SYN-1", 0, 10)
    assert sum(item.to_depth - item.from_depth for item in segments) == pytest.approx(10.0)
    assert [(item.from_depth, item.to_depth, item.assignment_source) for item in segments] == [
        (0.0, 1.0, "unassigned"),
        (1.0, 4.0, "domain_intervals"),
        (4.0, 6.0, "overlap_conflict"),
        (6.0, 8.0, "domain_intervals"),
        (8.0, 10.0, "unassigned"),
    ]


def test_rqd_and_rmr_zero_are_valid_and_use_existing_scalar_sample_contract() -> None:
    project, repository = _project()
    for data_type, field in (("rqd", "rqd"), ("rmr", "rmr")):
        result = repository.import_dataframe(
            data_type,
            pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, field: 0}]),
            f"synthetic-{data_type}.csv",
        )
        assert result["formal"] == 1
    count = ObservationService(project).synchronize_scalar_samples()
    assert count == 2
    assert {(sample.parameter_name, sample.value) for sample in project.m9_state.scalar_samples} == {
        ("rqd", 0.0),
        ("rmr", 0.0),
    }


def test_rmr_edit_delete_and_repeated_sync_invalidate_only_changed_parameter_field() -> None:
    project, repository = _project()
    repository.import_dataframe(
        "domain_intervals",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 20, "domain_id": 2}]),
        "synthetic-domains.csv",
    )
    repository.import_dataframe(
        "rmr",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "rmr": 50}]),
        "synthetic-rmr.csv",
    )
    rmr_field = ScalarFieldResult(
        metadata=ScalarFieldMetadata(
            field_id="rmr-field",
            parameter_name="rmr",
            unit="score",
            method=DensityMethod.GLOBAL_CONSTANT,
            shape=(1, 1, 1),
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            array_names=["estimate"],
            config_hash="rmr",
        ),
        settings=DensitySettings(),
        arrays={"estimate": np.asarray([[[50.0]]], dtype=np.float32)},
    )
    ucs_field = rmr_field.model_copy(deep=True)
    ucs_field.metadata.field_id = "ucs-field"
    ucs_field.metadata.parameter_name = "ucs"
    project.m9_state.scalar_fields = [rmr_field, ucs_field]

    ObservationService(project).synchronize_scalar_samples()
    assert {field.metadata.field_id for field in project.m9_state.scalar_fields} == {"rmr-field", "ucs-field"}
    record = repository.query("rmr", RecordState.FORMAL)[0]
    repository.edit_record(record.record_id, {**record.values, "rmr": 60})
    assert [(sample.parameter_name, sample.value) for sample in project.m9_state.scalar_samples] == [("rmr", 60.0)]
    assert [field.metadata.field_id for field in project.m9_state.scalar_fields] == ["ucs-field"]

    project.m9_state.scalar_fields.insert(0, rmr_field)
    ObservationService(project).synchronize_scalar_samples()
    assert {field.metadata.field_id for field in project.m9_state.scalar_fields} == {"rmr-field", "ucs-field"}
    repository.delete_record(record.record_id)
    assert project.m9_state.scalar_samples == []
    assert [field.metadata.field_id for field in project.m9_state.scalar_fields] == ["ucs-field"]


def test_rmr_interval_position_and_domain_changes_each_invalidate_its_old_field() -> None:
    project, repository = _project()
    repository.import_dataframe(
        "domain_intervals",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 20, "domain_id": 2}]),
        "domains.csv",
    )
    repository.import_dataframe(
        "rmr",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "rmr": 50}]),
        "rmr.csv",
    )

    def attach_field() -> None:
        project.m9_state.scalar_fields = [
            ScalarFieldResult(
                metadata=ScalarFieldMetadata(
                    field_id="rmr-field",
                    parameter_name="rmr",
                    unit="score",
                    method=DensityMethod.GLOBAL_CONSTANT,
                    shape=(1, 1, 1),
                    origin=(0, 0, 0),
                    spacing=(1, 1, 1),
                    array_names=["estimate"],
                    config_hash="old",
                ),
                settings=DensitySettings(),
                arrays={"estimate": np.asarray([[[50.0]]], dtype=np.float32)},
            )
        ]

    rmr = repository.query("rmr", RecordState.FORMAL)[0]
    attach_field()
    repository.edit_record(rmr.record_id, {**rmr.values, "from_depth": 1, "to_depth": 6})
    assert project.m9_state.scalar_fields == []

    attach_field()
    survey = repository.query("surveys", RecordState.FORMAL)[-1]
    repository.edit_record(survey.record_id, {**survey.values, "azimuth": 180})
    assert project.m9_state.scalar_fields == []

    attach_field()
    domain = repository.query("domain_intervals", RecordState.FORMAL)[0]
    repository.edit_record(domain.record_id, {**domain.values, "domain_id": 4})
    assert project.m9_state.scalar_fields == []

def test_rmr_overlap_is_reported_by_common_quality_service() -> None:
    project, repository = _project()
    repository.import_dataframe(
        "rmr",
        pd.DataFrame(
            [
                {"hole_id": "SYN-1", "from_depth": 0, "to_depth": 6, "rmr": 50},
                {"hole_id": "SYN-1", "from_depth": 5, "to_depth": 10, "rmr": 60},
            ]
        ),
        "synthetic-rmr.csv",
    )
    issues = BoreholeQualityService(project).run_checks()
    assert any(issue.code.startswith("rmr_overlap:") for issue in issues)


def test_same_coordinate_orientation_points_remain_distinct_with_stable_ids() -> None:
    project = Project()
    repository = BoreholeRepository(project)
    frame = pd.DataFrame(
        [
            {"observation_id": "P001-A", "point_id": "001", "x": 1, "y": 2, "z": 3, "dip": 40, "dip_direction": 100, "local_set_id": "A", "joint_spacing_m": 0.5, "joint_num": 2},
            {"observation_id": "P001-B", "point_id": "001", "x": 1, "y": 2, "z": 3, "dip": 50, "dip_direction": 120, "local_set_id": "B", "joint_spacing_m": 1.0, "joint_num": 1},
        ]
    )
    repository.import_dataframe("orientation_points", frame, "synthetic-points.csv")
    observations = ObservationService(project).orientation_points()
    assert len(observations) == 2
    assert len({item.observation_id for item in observations}) == 2
    assert {item.calibration_role for item in observations} == {"UNASSIGNED"}
    assert project.borehole_collection.boreholes == []


def test_input_arrays_are_not_modified() -> None:
    project, repository = _project()
    frame = pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "fracture_spacing": 10}])
    before = frame.copy(deep=True)
    repository.import_dataframe(
        "fractures",
        frame,
        "synthetic.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "cm"},
    )
    assert np.array_equal(frame.to_numpy(), before.to_numpy())
