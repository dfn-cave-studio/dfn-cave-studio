"""M8 repository, migration, spatial-domain, and grid-definition tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dfn_cave_studio.models.borehole import (
    Borehole,
    BoreholeCollection,
    BoreholeSurvey,
    Collar,
    FractureObservation,
    SurveyStation,
)
from dfn_cave_studio.models.borehole_database import BoreholeDatabase, RecordState
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.rock_mask import RockMask
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.persistence.project_store import ProjectStore
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.spatial_domain_service import SpatialDomainService
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController
from dfn_cave_studio.visualization.m8_spatial_preview import M8SpatialPreviewRenderer

DEMO = Path(__file__).parents[2] / "examples" / "m7_demo"


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(DEMO / f"{name}.csv")


def _import_all(repository: BoreholeRepository, order: list[str]) -> None:
    for name in order:
        repository.import_dataframe(name, _table(name), str(DEMO / f"{name}.csv"))


def test_collars_only_can_be_queried_and_round_trip(tmp_path: Path) -> None:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe("collars", _table("collars"), "collars.csv")
    assert repository.database.counts("collars") == {"raw": 10, "formal": 8, "excluded": 2, "pending": 0}
    path = tmp_path / "collars-only.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)
    assert len(reopened.borehole_collection) == 8
    assert reopened.borehole_database.counts("collars")["raw"] == 10


def test_surveys_pending_then_automatically_link_to_later_collars() -> None:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe("surveys", _table("surveys"), "surveys.csv")
    assert repository.database.counts("surveys")["pending"] == 153
    repository.import_dataframe("collars", _table("collars"), "collars.csv")
    assert repository.database.counts("surveys") == {"raw": 153, "formal": 148, "excluded": 5, "pending": 0}
    assert sum(len(borehole.survey.stations) for borehole in project.borehole_collection) < 148
    service = BoreholeQualityService(project)
    service.run_checks()
    assert service.apply_auto_fixes() == 155
    assert sum(len(borehole.survey.stations) for borehole in project.borehole_collection) == 148


@pytest.mark.parametrize(
    "order",
    [
        ["collars", "surveys", "fractures", "rqd", "domain_intervals"],
        ["domain_intervals", "rqd", "fractures", "surveys", "collars"],
        ["fractures", "collars", "domain_intervals", "surveys", "rqd"],
    ],
)
def test_five_tables_import_in_arbitrary_order(order: list[str]) -> None:
    project = Project()
    repository = BoreholeRepository(project)
    _import_all(repository, order)
    assert len(project.borehole_collection) == 8
    assert repository.database.counts("fractures") == {"raw": 83, "formal": 80, "excluded": 3, "pending": 0}
    assert repository.database.counts("domain_intervals")["formal"] == 11


def test_duplicate_append_is_reported_and_not_silently_added() -> None:
    repository = BoreholeRepository(Project())
    first = repository.import_dataframe("collars", _table("collars"), "collars.csv")
    second = repository.import_dataframe("collars", _table("collars"), "collars.csv")
    assert first["raw"] == 10
    assert second["raw"] == 0
    assert second["duplicates"] == 10
    assert repository.database.counts("collars")["raw"] == 10


def test_append_replace_and_cancel_retain_audit_rows() -> None:
    repository = BoreholeRepository(Project())
    source = pd.DataFrame([{"borehole_id": "A", "collar_x": 0, "collar_y": 0, "collar_z": 10, "final_depth": 50}])
    repository.import_dataframe("collars", source, "one.csv")
    cancelled = repository.import_dataframe("collars", source, "one.csv", mode="cancel")
    assert cancelled["cancelled"]
    replacement = source.assign(collar_x=5)
    repository.import_dataframe("collars", replacement, "two.csv", mode="replace")
    assert repository.database.counts("collars") == {"raw": 2, "formal": 1, "excluded": 1, "pending": 0}
    assert repository.query("collars", RecordState.FORMAL)[0].values["collar_x"] == 5


def test_raw_formal_excluded_pending_round_trip(tmp_path: Path) -> None:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe("fractures", _table("fractures"), "fractures.csv")
    repository.import_dataframe("collars", _table("collars"), "collars.csv")
    expected = repository.database.counts("fractures")
    path = tmp_path / "states.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)
    assert reopened.borehole_database.counts("fractures") == expected
    excluded = reopened.borehole_database.query("fractures", RecordState.EXCLUDED)
    assert {record.source_row for record in excluded} == {80, 81, 82}
    assert all(record.exclusion_reason for record in excluded)


def test_m7_archive_migrates_all_demo_rows_without_loss(tmp_path: Path) -> None:
    project = Project()
    repository = BoreholeRepository(project)
    _import_all(repository, ["collars", "surveys", "fractures", "rqd", "domain_intervals"])
    quality = BoreholeQualityService(project)
    quality.run_checks()
    quality.apply_auto_fixes()
    legacy_collection = project.borehole_collection.model_copy(deep=True)
    project._m7_data = {
        "raw_surveys": _table("surveys"),
        "raw_fractures": _table("fractures"),
        "raw_rqd": _table("rqd"),
        "raw_domain_intervals": _table("domain_intervals"),
    }
    project.borehole_database = BoreholeDatabase()
    project.borehole_collection = legacy_collection
    path = tmp_path / "legacy-v0.7.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)
    assert len(reopened.borehole_collection) == 8
    assert reopened.borehole_database.counts("surveys") == {
        "raw": 153,
        "formal": 148,
        "excluded": 5,
        "pending": 0,
    }
    assert reopened.borehole_database.counts("fractures") == {
        "raw": 83,
        "formal": 80,
        "excluded": 3,
        "pending": 0,
    }
    assert reopened.borehole_database.counts("domain_intervals")["formal"] == 11
    assert sum(len(borehole.survey.stations) for borehole in reopened.borehole_collection) == 148


def test_table_changes_invalidate_only_real_workflow_dependencies() -> None:
    project = Project()
    workflow = WorkflowController()
    for step in workflow.get_steps():
        workflow.complete_step(step.step_id)
    project._m7_data = {"workflow": workflow}
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "rqd",
        pd.DataFrame([{"hole_id": "UNKNOWN", "from_depth": 0, "to_depth": 1, "rqd": 50}]),
        "rqd.csv",
    )
    assert workflow.get_step("clean").status == StepStatus.STALE
    assert workflow.get_step("joint_sets").status == StepStatus.COMPLETED
    assert workflow.get_step("bounds").status == StepStatus.COMPLETED


def test_repeated_save_autosave_and_reopen_preserve_database(tmp_path: Path) -> None:
    store = ProjectStore()
    project = store.new_project("M8 autosave")
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "A", "collar_x": 0, "collar_y": 0, "collar_z": 10, "final_depth": 50}]),
        "collars.csv",
    )
    path = tmp_path / "autosave.dfnproj"
    store.save_as(path)
    store.save()
    record = repository.query("collars", RecordState.FORMAL)[0]
    repository.edit_record(record.record_id, dict(record.values, collar_x=7))
    store.mark_dirty()
    store.configure_auto_save(enabled=True, interval_seconds=0)
    assert store.tick_auto_save()
    reopened = ZipProjectStore().load(path)
    formal = reopened.borehole_database.query("collars", RecordState.FORMAL)
    assert len(formal) == 1
    assert formal[0].values["collar_x"] == 7
    assert formal[0].original_values["collar_x"] == 0


def test_edit_and_delete_keep_original_values_and_history() -> None:
    repository = BoreholeRepository(Project())
    repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "A", "collar_x": 0, "collar_y": 0, "collar_z": 0, "final_depth": 10}]),
        "a.csv",
    )
    record = repository.query("collars", RecordState.FORMAL)[0]
    original = record.original_values.copy()
    edited = dict(record.values, collar_x=2)
    repository.edit_record(record.record_id, edited)
    assert record.original_values == original
    assert record.values["collar_x"] == 2
    assert record.modification_history[-1].action == "edit"
    repository.delete_record(record.record_id)
    assert record.state == RecordState.EXCLUDED
    assert record.original_values == original


def _deviated_collection() -> BoreholeCollection:
    collar = Collar(
        borehole_id="D1",
        collar_x=0,
        collar_y=0,
        collar_z=100,
        azimuth=90,
        dip=-45,
        final_depth=100,
    )
    survey = BoreholeSurvey(
        stations=[
            SurveyStation(measured_depth=0, azimuth=90, dip=-45),
            SurveyStation(measured_depth=50, azimuth=90, dip=-45),
        ]
    )
    borehole = Borehole(
        borehole_id="D1",
        collar=collar,
        survey=survey,
        fracture_observations=[FractureObservation(measured_depth=90, dip_direction=0, dip=45)],
    )
    return BoreholeCollection(boreholes=[borehole])


def test_automatic_bounds_include_complete_trajectory_and_observations() -> None:
    collection = _deviated_collection()
    bounds = SpatialDomainService.automatic_bounds(collection, outward_margin=2)
    points = collection["D1"].compute_trajectory()[0]
    assert all(bounds.contains_point(*point) for point in points)
    assert bounds.x_max >= points[-1, 0] + 2
    assert points[-1, 2] < collection["D1"].collar.collar_z


def test_collar_inside_but_trajectory_outside_is_reported() -> None:
    collection = _deviated_collection()
    bounds = ModelBounds(x_min=-1, x_max=10, y_min=-1, y_max=1, z_min=90, z_max=101)
    assert bounds.contains_point(*collection["D1"].collar.position)
    report = SpatialDomainService.check_bounds(bounds, collection)
    assert report.outside_borehole_count == 1
    assert report.outside_trajectory_point_count > 0
    assert report.maximum_exceedance["x_max"] > 0
    assert report.affected_holes == ["D1"]


def test_manual_insufficient_boundary_has_recommended_expansion() -> None:
    collection = _deviated_collection()
    bounds = ModelBounds(x_min=-1, x_max=5, y_min=-1, y_max=1, z_min=95, z_max=101)
    report = SpatialDomainService.check_bounds(bounds, collection)
    full = SpatialDomainService.automatic_bounds(collection)
    assert report.recommended_domain.x_max >= full.x_max
    assert report.recommended_domain.z_min <= full.z_min


def test_anisotropic_voxel_dimensions_use_ceil_and_memory_is_exact() -> None:
    bounds = ModelBounds(x_min=0, x_max=10.1, y_min=0, y_max=9.1, z_min=0, z_max=8.1)
    voxel = VoxelConfig(cell_size_x=3, cell_size_y=4, cell_size_z=5)
    summary = SpatialDomainService.grid_summary(bounds, voxel)
    assert (summary.nx, summary.ny, summary.nz) == (4, 3, 2)
    assert summary.total_voxels == 24
    assert summary.estimated_bytes == 24 * (1 + 1 + 4 + 4)
    assert set(summary.state_semantics) == set(VoxelCellState)
    mask = RockMask(x_min=0, x_max=6, y_min=0, y_max=8, z_min=0, z_max=5)
    assert SpatialDomainService.box_mask_active_voxels(bounds, voxel, mask) == 2 * 2 * 1


def test_generation_domain_contains_analysis_and_uses_axis_buffers() -> None:
    analysis = ModelBounds(x_min=0, x_max=10, y_min=0, y_max=20, z_min=-5, z_max=5)
    voxel = VoxelConfig(cell_size_x=2, cell_size_y=4, cell_size_z=8)
    generation = SpatialDomainService.generation_domain(analysis, voxel, maximum_fracture_radius=6, buffer_layers=2)
    assert generation.x_min == -6
    assert generation.y_min == -8
    assert generation.z_min == -21
    config = SpatialDomainService.build_config(analysis, voxel, "manual", 0, 2, 6)
    assert config.generation_domain == generation


def test_spatial_domains_round_trip_in_dfnproj(tmp_path: Path) -> None:
    project = Project()
    analysis = ModelBounds(x_min=0, x_max=10, y_min=0, y_max=20, z_min=-5, z_max=5)
    voxel = VoxelConfig(cell_size_x=2, cell_size_y=4, cell_size_z=8)
    project.voxel_config = voxel
    project.spatial_grid_config = SpatialDomainService.build_config(analysis, voxel, "manual", 0, 2, 6)
    path = tmp_path / "spatial.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)
    assert reopened.spatial_grid_config is not None
    assert reopened.spatial_grid_config.analysis_domain == analysis
    assert reopened.spatial_grid_config.generation_domain.x_min == -6


def test_grid_summary_does_not_create_dense_voxel_array(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense allocation attempted")

    monkeypatch.setattr("numpy.zeros", forbidden)
    bounds = ModelBounds(x_min=0, x_max=10_000, y_min=0, y_max=10_000, z_min=0, z_max=10_000)
    summary = SpatialDomainService.grid_summary(bounds, VoxelConfig(), warning_limit_bytes=1)
    assert summary.total_voxels == 10**12
    assert summary.warning is not None


def test_preview_grid_lines_are_lod_bounded_for_huge_grid() -> None:
    bounds = ModelBounds(x_min=0, x_max=1_000_000, y_min=0, y_max=1_000_000, z_min=0, z_max=1_000_000)
    lines = M8SpatialPreviewRenderer._sampled_grid_lines(
        bounds,
        VoxelConfig(cell_size_x=0.1, cell_size_y=0.1, cell_size_z=0.1),
        maximum=12,
        slice_axis="z",
        slice_fraction=0.5,
    )
    assert lines.n_cells <= 24
    assert lines.n_points <= 48
