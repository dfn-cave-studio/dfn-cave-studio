"""End-to-end synthetic tests for generic M9 scalar fields."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from dfn_cave_studio.models.borehole import Borehole, BoreholeCollection, BoreholeSurvey, Collar, SurveyStation
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, KrigingSettings, ScalarParameterSample, VariogramMode
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.services.m7_state import set_workflow
from dfn_cave_studio.services.scalar_field_service import ScalarParameterFieldService, scalar_field_id
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController


def _project() -> Project:
    project = Project()
    holes = [
        Borehole(collar=Collar(borehole_id=name, collar_x=x, collar_y=y, collar_z=10.0, final_depth=10.0),
                 survey=BoreholeSurvey())
        for name, x, y in [("A", 0.0, 0.0), ("B", 10.0, 0.0), ("C", 0.0, 10.0), ("D", 10.0, 10.0), ("V", 5.0, 5.0)]
    ]
    project.borehole_collection = BoreholeCollection(boreholes=holes)
    bounds = ModelBounds(x_min=0, x_max=12, y_min=0, y_max=12, z_min=0, z_max=10)
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=bounds, generation_domain=bounds)
    project.voxel_config = VoxelConfig(cell_size_x=4, cell_size_y=4, cell_size_z=5)
    holdout = HoldoutService(); holdout.select_manual([hole.borehole_id for hole in holes], ["V"]); holdout.lock()
    set_holdout(project, holdout)
    return project


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "borehole_id": ["A", "B", "C", "D", "V"],
            "from_depth": [0.0] * 5,
            "to_depth": [10.0] * 5,
            "parameter_name": ["UCS"] * 5,
            "value": [10.0, 20.0, 30.0, 40.0, 25.0],
            "unit": ["MPa"] * 5,
            "quality_flag": ["accepted"] * 5,
        }
    )


def test_scalar_import_uses_real_trajectory_midpoint_and_validation_isolation() -> None:
    project = _project()
    samples = ScalarParameterFieldService(project).import_frame(_frame(), source_dataset="synthetic.csv")
    assert len(samples) == 5
    assert samples[0].midpoint_z == 5.0
    assert samples[-1].role == "validation"
    assert samples[-1].sample_id not in [item.sample_id for item in samples[:-1]]


def test_deviated_borehole_midpoint_uses_minimum_curvature_xyz() -> None:
    project = _project()
    hole = project.borehole_collection.boreholes[0]
    hole.survey = BoreholeSurvey(
        stations=[
            SurveyStation(measured_depth=0.0, azimuth=0.0, dip=-90.0),
            SurveyStation(measured_depth=10.0, azimuth=90.0, dip=0.0),
        ]
    )
    frame = _frame().iloc[:1].copy()
    sample = ScalarParameterFieldService(project).import_frame(frame, source_dataset="synthetic.csv")[0]
    radius = 20.0 / np.pi
    assert sample.midpoint_x == pytest.approx(radius * (1.0 - np.sqrt(0.5)))
    assert sample.midpoint_y == pytest.approx(0.0, abs=1e-12)
    assert sample.midpoint_z == pytest.approx(10.0 - radius * np.sqrt(0.5))
    assert sample.midpoint_z != pytest.approx(hole.collar.collar_z - 5.0)


def test_kriging_field_save_reopen_and_export(tmp_path) -> None:
    project = _project(); service = ScalarParameterFieldService(project)
    service.import_frame(_frame(), source_dataset="synthetic.csv")
    settings = DensitySettings(
        method=DensityMethod.ORDINARY_KRIGING,
        kriging=KrigingSettings(mode=VariogramMode.MANUAL, nugget=0.0, sill=100.0, range=20.0,
                                minimum_neighbors=2, maximum_neighbors=4),
    )
    result = service.build("UCS", settings); service.commit(result)
    assert result.metadata.method == DensityMethod.ORDINARY_KRIGING
    assert result.metadata.provenance["validation_sample_ids_excluded_from_fit"]
    assert np.isfinite(result.arrays["estimate"]).any()
    assert np.nanmin(result.arrays["kriging_variance"]) >= 0.0
    archive = tmp_path / "scalar.dfnproj"; ZipProjectStore().save(project, archive)
    reopened = ZipProjectStore().load(archive)
    restored = reopened.m9_state.scalar_fields[0]
    for name in result.arrays:
        np.testing.assert_array_equal(restored.arrays[name], result.arrays[name])
    exported = ScalarParameterFieldService(reopened).export(restored, tmp_path / "export")
    assert {item.suffix for item in exported} == {".json", ".csv", ".npz", ".vti"}


def test_conflicting_units_and_duplicate_rows_are_rejected_atomically() -> None:
    project = _project(); service = ScalarParameterFieldService(project)
    database_before = project.borehole_database.model_dump(mode="python")
    frame = _frame(); frame.loc[1, "unit"] = "Pa"
    try:
        service.import_frame(frame, source_dataset="synthetic.csv")
    except ValueError as error:
        assert "conflicting units" in str(error)
    else:
        raise AssertionError("conflicting units were accepted")
    assert project.m9_state.scalar_samples == []
    assert project.borehole_database.model_dump(mode="python") == database_before


def test_auxiliary_scalar_import_does_not_invalidate_dfn_workflow() -> None:
    project = _project(); workflow = WorkflowController(); set_workflow(project, workflow)
    for step in ("density", "size", "parameter_field", "validation", "explicit_dfn", "second_voxelization"):
        workflow.complete_step(step)
    before = workflow.to_dict()
    ScalarParameterFieldService(project).import_frame(_frame(), source_dataset="synthetic.csv")
    assert workflow.to_dict() == before


def test_database_cross_domain_samples_are_reported_and_excluded_without_changing_legacy_none_domain() -> None:
    project = _project()
    project.m9_state.scalar_samples = [
        ScalarParameterSample(
            sample_id="db:eligible",
            borehole_id="A",
            from_depth=0,
            to_depth=10,
            parameter_name="RMR",
            value=60,
            unit="score",
            midpoint_x=0,
            midpoint_y=0,
            midpoint_z=5,
            domain_id=2,
            source_record_id="eligible",
            domain_segments=[{"from_depth": 0, "to_depth": 10, "domain_id": 2}],
            domain_assignment_method="explicit_segments",
        ),
        ScalarParameterSample(
            sample_id="db:cross-domain",
            borehole_id="B",
            from_depth=0,
            to_depth=10,
            parameter_name="RMR",
            value=70,
            unit="score",
            midpoint_x=10,
            midpoint_y=0,
            midpoint_z=5,
            domain_id=None,
            source_record_id="cross-domain",
            domain_segments=[
                {"from_depth": 0, "to_depth": 4, "domain_id": 2},
                {"from_depth": 4, "to_depth": 10, "domain_id": 4},
            ],
            domain_assignment_method="explicit_segments",
        ),
        ScalarParameterSample(
            sample_id="db:unassigned",
            borehole_id="C",
            from_depth=0,
            to_depth=10,
            parameter_name="RMR",
            value=80,
            unit="score",
            midpoint_x=0,
            midpoint_y=10,
            midpoint_z=5,
            domain_id=None,
            source_record_id="unassigned",
            domain_segments=[
                {"from_depth": 0, "to_depth": 5, "domain_id": 2},
                {"from_depth": 5, "to_depth": 10, "domain_id": None},
            ],
            domain_assignment_method="explicit_segments",
        ),
        ScalarParameterSample(
            sample_id="legacy:none-domain",
            borehole_id="D",
            from_depth=0,
            to_depth=10,
            parameter_name="RMR",
            value=50,
            unit="score",
            midpoint_x=10,
            midpoint_y=10,
            midpoint_z=5,
            domain_id=None,
        ),
        ScalarParameterSample(
            sample_id="db:validation",
            borehole_id="V",
            from_depth=0,
            to_depth=10,
            parameter_name="RMR",
            value=65,
            unit="score",
            midpoint_x=5,
            midpoint_y=5,
            midpoint_z=5,
            domain_id=2,
            source_record_id="validation",
            domain_segments=[{"from_depth": 0, "to_depth": 10, "domain_id": 2}],
            domain_assignment_method="explicit_segments",
        ),
    ]
    result = ScalarParameterFieldService(project).build(
        "RMR", DensitySettings(method=DensityMethod.GLOBAL_CONSTANT)
    )
    excluded = result.metadata.provenance["database_samples_excluded_from_fit_and_validation"]
    assert excluded == [
        {"sample_id": "db:cross-domain", "reason": "crosses_multiple_domains"},
        {"sample_id": "db:unassigned", "reason": "partially_unassigned_domain"},
    ]
    assert result.metadata.provenance["calibration_sample_ids"] == ["db:eligible", "legacy:none-domain"]
    assert result.metadata.provenance["validation_sample_ids_excluded_from_fit"] == ["db:validation"]
    assert result.validation_summary.sample_count == 1


def test_only_ambiguous_database_domain_samples_fail_with_auditable_reason() -> None:
    project = _project()
    project.m9_state.scalar_samples = [
        ScalarParameterSample(
            sample_id="db:cross-domain",
            borehole_id="A",
            from_depth=0,
            to_depth=10,
            parameter_name="RQD",
            value=75,
            unit="%",
            midpoint_x=0,
            midpoint_y=0,
            midpoint_z=5,
            domain_id=None,
            source_record_id="cross-domain",
            domain_segments=[
                {"from_depth": 0, "to_depth": 5, "domain_id": 2},
                {"from_depth": 5, "to_depth": 10, "domain_id": 4},
            ],
            domain_assignment_method="explicit_segments",
        )
    ]
    with pytest.raises(ValueError, match="excluded database samples: crosses_multiple_domains"):
        ScalarParameterFieldService(project).build("RQD", DensitySettings(method="global_constant"))


def test_only_p32_field_commit_invalidates_m10_and_m11() -> None:
    project = _project()
    workflow = WorkflowController()
    set_workflow(project, workflow)
    for step in ("density", "size", "parameter_field", "validation", "explicit_dfn", "second_voxelization"):
        workflow.complete_step(step)
    service = ScalarParameterFieldService(project)
    service.import_frame(_frame(), source_dataset="synthetic.csv")
    service.commit(service.build("UCS", DensitySettings(method="global_constant")))
    assert workflow.get_step("explicit_dfn").status == StepStatus.COMPLETED
    assert workflow.get_step("second_voxelization").status == StepStatus.COMPLETED
    p32 = _frame().copy()
    p32["parameter_name"] = "P32"
    p32["unit"] = "m^-1"
    p32["value"] = [0.1, 0.2, 0.3, 0.4, 0.25]
    service.import_frame(p32, source_dataset="synthetic-p32.csv")
    service.commit(service.build("P32", DensitySettings(method="global_constant")))
    assert workflow.get_step("explicit_dfn").status == StepStatus.STALE
    assert workflow.get_step("second_voxelization").status == StepStatus.STALE


def test_scalar_csv_import_uses_same_long_table_contract(tmp_path) -> None:
    project = _project(); path = tmp_path / "synthetic_scalar.csv"; _frame().to_csv(path, index=False)
    imported = ScalarParameterFieldService(project).import_table(path)
    assert len(imported) == 5
    assert project.borehole_database.counts("scalar_parameters") == {"raw": 5, "formal": 5, "excluded": 0, "pending": 0}


def test_negative_prediction_policy_rejects_or_clips_with_audit() -> None:
    coordinates = np.asarray([
        [0.8025662504, 0.3957048590, 1.6810347667], [1.5436928907, 0.1426744999, 1.0196045744],
        [1.8575246436, 0.4696493671, 1.3012862086], [0.8860408773, 1.6175430478, 0.4499217102],
        [1.2840687489, 1.1536178579, 0.1364901646], [1.8608036851, 0.1657631769, 1.3273655112],
    ])
    values = [0.6203434192, 0.0979508128, 0.0361064494, 0.0805884798, 0.0146559873, 0.6837179537]
    target = np.asarray([1.7532164540, 1.2232780340, 0.7990488412])
    project = Project(); bounds = ModelBounds(x_min=target[0]-.5, x_max=target[0]+.5,
        y_min=target[1]-.5, y_max=target[1]+.5, z_min=target[2]-.5, z_max=target[2]+.5)
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=bounds, generation_domain=bounds)
    project.voxel_config = VoxelConfig()
    project.m9_state.scalar_samples = [ScalarParameterSample(sample_id=f"p{i}", borehole_id=f"H{i}", from_depth=0,
        to_depth=1, parameter_name="P32", value=value, unit="m^-1", midpoint_x=point[0], midpoint_y=point[1],
        midpoint_z=point[2], role="calibration") for i, (point, value) in enumerate(zip(coordinates, values))]
    holdout = HoldoutService()
    holdout.select_manual([f"H{i}" for i in range(len(coordinates))], [])
    holdout.lock()
    set_holdout(project, holdout)
    base = {"mode": "manual", "nugget": 0, "sill": 1, "range": 2, "minimum_neighbors": 2, "maximum_neighbors": 6}
    rejected = ScalarParameterFieldService(project).build("P32", DensitySettings(method="ordinary_kriging", kriging={**base, "non_negative_policy": "reject"}))
    assert not np.isfinite(rejected.arrays["estimate"][0, 0, 0])
    assert rejected.metadata.rejected_voxel_count == 1
    assert rejected.metadata.clipped_voxel_count == 0
    clipped = ScalarParameterFieldService(project).build("P32", DensitySettings(method="ordinary_kriging", kriging={**base, "non_negative_policy": "clip_with_audit"}))
    assert clipped.arrays["estimate"][0, 0, 0] == 0
    assert clipped.metadata.clipped_voxel_count == 1
    assert clipped.metadata.pre_clip_minimum < 0
    assert clipped.metadata.clipped_total_change > 0


def test_locked_current_holdout_overrides_import_role_snapshot() -> None:
    project = _project()
    service = ScalarParameterFieldService(project)
    samples = service.import_frame(_frame(), source_dataset="synthetic.csv")
    for sample in samples:
        sample.role = "validation" if sample.borehole_id != "V" else "calibration"
    settings = DensitySettings(method="global_constant")
    result = service.build("UCS", settings)
    calibration_ids = set(result.metadata.provenance["calibration_sample_ids"])
    validation_ids = set(result.metadata.provenance["validation_sample_ids_excluded_from_fit"])
    assert calibration_ids == {sample.sample_id for sample in samples if sample.borehole_id != "V"}
    assert validation_ids == {sample.sample_id for sample in samples if sample.borehole_id == "V"}


def test_validation_extreme_value_does_not_change_fitted_voxel_arrays() -> None:
    project = _project()
    service = ScalarParameterFieldService(project)
    samples = service.import_frame(_frame(), source_dataset="synthetic.csv")
    settings = DensitySettings(
        method="ordinary_kriging",
        kriging={"mode": "manual", "nugget": 0, "sill": 100, "range": 20,
                 "minimum_neighbors": 2, "maximum_neighbors": 4},
    )
    first = service.build("UCS", settings)
    next(sample for sample in samples if sample.borehole_id == "V").value = 1_000_000.0
    second = service.build("UCS", settings)
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name], second.arrays[name])
    assert first.metadata.variograms == second.metadata.variograms


def test_leave_one_borehole_out_never_splits_intervals_within_a_hole() -> None:
    project = _project()
    service = ScalarParameterFieldService(project)
    service.import_frame(_frame(), source_dataset="synthetic.csv")
    results, summary = service.leave_one_borehole_out("UCS", DensitySettings(method="global_constant"))
    assert summary.sample_count == 4
    assert {item.borehole_id for item in results} == {"A", "B", "C", "D"}
    assert all(item.status == "predicted" for item in results)


def test_missing_or_unlocked_holdout_blocks_build() -> None:
    project = _project()
    service = ScalarParameterFieldService(project)
    service.import_frame(_frame(), source_dataset="synthetic.csv")
    holdout = HoldoutService()
    holdout.select_manual([hole.borehole_id for hole in project.borehole_collection], ["V"])
    set_holdout(project, holdout)
    with pytest.raises(RuntimeError, match="locked Validation Holdout"):
        service.build("UCS", DensitySettings(method="global_constant"))


def test_insufficient_domain_samples_produce_no_data_with_reason_not_idw_fallback() -> None:
    project = _project()
    service = ScalarParameterFieldService(project)
    service.import_frame(_frame().iloc[:1].copy(), source_dataset="synthetic.csv")
    result = service.build(
        "UCS",
        DensitySettings(method="ordinary_kriging", kriging={"mode": "manual", "nugget": 0,
                        "sill": 1, "range": 10, "minimum_neighbors": 2, "maximum_neighbors": 4}),
    )
    assert not np.isfinite(result.arrays["estimate"]).any()
    assert result.metadata.provenance["insufficient_data_by_domain"]
    assert "INSUFFICIENT_DATA" in next(iter(result.metadata.provenance["insufficient_data_by_domain"].values()))


def test_unicode_parameter_ids_are_stable_and_collision_resistant() -> None:
    first = scalar_field_id("岩体强度")
    assert first == scalar_field_id(" 岩体强度 ")
    assert first != scalar_field_id("节理密度")
    assert first.startswith("parameter-")


def test_two_unicode_parameter_fields_do_not_overwrite_and_ids_survive_reopen(tmp_path) -> None:
    project = _project()
    service = ScalarParameterFieldService(project)
    expected_ids = []
    for name, unit in (("岩体强度", "MPa"), ("节理密度", "m^-1")):
        frame = _frame().copy()
        frame["parameter_name"] = name
        frame["unit"] = unit
        service.import_frame(frame, source_dataset=f"synthetic-{name}.csv")
        result = service.build(name, DensitySettings(method="global_constant"))
        service.commit(result)
        expected_ids.append(result.metadata.field_id)
    assert len(set(expected_ids)) == 2
    path = tmp_path / "unicode-fields.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)
    assert {item.metadata.field_id for item in reopened.m9_state.scalar_fields} == set(expected_ids)


@pytest.mark.parametrize(
    ("parameter", "invalid"),
    (("P32", -0.1), ("RQD", -0.1), ("RQD", 100.1), ("RMR", -0.1), ("RMR", 100.1),
     ("UCS", -0.1), ("joint spacing", -0.1), ("joint-density", -0.1)),
)
def test_parameter_input_bounds_are_enforced(parameter: str, invalid: float) -> None:
    project = _project()
    frame = _frame().iloc[:1].copy()
    frame.loc[0, "parameter_name"] = parameter
    frame.loc[0, "value"] = invalid
    with pytest.raises(ValueError, match="value must be"):
        ScalarParameterFieldService(project).import_frame(frame, source_dataset="synthetic.csv")


def test_rqd_upper_prediction_policy_rejects_and_clips() -> None:
    coordinates = np.asarray([
        [0.8025662504, 0.3957048590, 1.6810347667], [1.5436928907, 0.1426744999, 1.0196045744],
        [1.8575246436, 0.4696493671, 1.3012862086], [0.8860408773, 1.6175430478, 0.4499217102],
        [1.2840687489, 1.1536178579, 0.1364901646], [1.8608036851, 0.1657631769, 1.3273655112],
    ])
    base_values = np.asarray([0.6203434192, 0.0979508128, 0.0361064494, 0.0805884798, 0.0146559873, 0.6837179537])
    values = 100.0 - 100.0 * base_values
    target = np.asarray([1.7532164540, 1.2232780340, 0.7990488412])
    project = Project()
    bounds = ModelBounds(x_min=target[0] - 0.5, x_max=target[0] + 0.5,
                         y_min=target[1] - 0.5, y_max=target[1] + 0.5,
                         z_min=target[2] - 0.5, z_max=target[2] + 0.5)
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=bounds, generation_domain=bounds)
    project.voxel_config = VoxelConfig()
    project.m9_state.scalar_samples = [
        ScalarParameterSample(sample_id=f"r{i}", borehole_id=f"H{i}", from_depth=0, to_depth=1,
                              parameter_name="RQD", value=float(value), unit="%", midpoint_x=point[0],
                              midpoint_y=point[1], midpoint_z=point[2])
        for i, (point, value) in enumerate(zip(coordinates, values, strict=True))
    ]
    holdout = HoldoutService()
    holdout.select_manual([f"H{i}" for i in range(len(coordinates))], [])
    holdout.lock()
    set_holdout(project, holdout)
    base = {"mode": "manual", "nugget": 0, "sill": 10_000, "range": 2,
            "minimum_neighbors": 2, "maximum_neighbors": 6}
    rejected = ScalarParameterFieldService(project).build(
        "RQD", DensitySettings(method="ordinary_kriging", kriging={**base, "non_negative_policy": "reject"})
    )
    assert rejected.metadata.rejected_voxel_count == 1
    assert not np.isfinite(rejected.arrays["estimate"][0, 0, 0])
    clipped = ScalarParameterFieldService(project).build(
        "RQD", DensitySettings(method="ordinary_kriging", kriging={**base, "non_negative_policy": "clip_with_audit"})
    )
    assert clipped.arrays["estimate"][0, 0, 0] == 100.0
    assert clipped.metadata.clipped_voxel_count == 1
    assert clipped.metadata.pre_adjustment_maximum > 100.0
    assert clipped.metadata.parameter_bounds == (0.0, 100.0)


def test_cancelled_build_does_not_commit_partial_field() -> None:
    project = _project(); service = ScalarParameterFieldService(project); service.import_frame(_frame(), source_dataset="synthetic.csv")
    before = list(project.m9_state.scalar_fields)
    settings = DensitySettings(method="ordinary_kriging", kriging={"mode": "manual", "nugget": 0,
        "sill": 100, "range": 20, "minimum_neighbors": 2, "maximum_neighbors": 4})
    with pytest.raises(InterruptedError):
        service.build("UCS", settings, cancelled=lambda: True)
    assert project.m9_state.scalar_fields == before


@pytest.mark.parametrize("corruption", ["name", "shape", "dtype"])
def test_scalar_npz_load_rejects_corrupt_array_contract(tmp_path, corruption: str) -> None:
    project = _project()
    service = ScalarParameterFieldService(project)
    service.import_frame(_frame(), source_dataset="synthetic.csv")
    result = service.build("UCS", DensitySettings(method="global_constant"))
    service.commit(result)
    valid_path = tmp_path / "valid.dfnproj"
    corrupt_path = tmp_path / f"corrupt-{corruption}.dfnproj"
    ZipProjectStore().save(project, valid_path)
    member = f"results/m9_scalar/{result.metadata.field_id}.npz"
    arrays = {name: values.copy() for name, values in result.arrays.items()}
    if corruption == "name":
        arrays["unexpected"] = arrays.pop("estimate")
    elif corruption == "shape":
        arrays["estimate"] = arrays["estimate"].reshape(-1)
    else:
        arrays["neighbor_count"] = arrays["neighbor_count"].astype(np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    with zipfile.ZipFile(valid_path, "r") as source, zipfile.ZipFile(corrupt_path, "w") as target:
        for info in source.infolist():
            target.writestr(info, buffer.getvalue() if info.filename == member else source.read(info.filename))
    with pytest.raises(ValueError, match="M9 scalar array"):
        ZipProjectStore().load(corrupt_path)
