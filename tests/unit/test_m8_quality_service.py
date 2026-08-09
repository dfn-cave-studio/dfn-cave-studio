"""Regression tests for the canonical M8 data-quality workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dfn_cave_studio.models.borehole_database import QualityIssueStatus, QualitySeverity, RecordState
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController

DEMO = Path(__file__).parents[2] / "examples" / "m7_demo"


def _demo_project(order: tuple[str, ...] = ("collars", "surveys", "fractures", "rqd", "domain_intervals")):
    project = Project()
    repository = BoreholeRepository(project)
    for name in order:
        repository.import_dataframe(name, pd.read_csv(DEMO / f"{name}.csv"), str(DEMO / f"{name}.csv"))
    return project, repository


def test_demo_quality_baseline_and_audited_survey_corrections() -> None:
    project, repository = _demo_project(("surveys", "collars", "fractures", "rqd", "domain_intervals"))
    assert repository.database.counts() == {"raw": 284, "formal": 272, "excluded": 12, "pending": 0}
    assert repository.database.counts("fractures") == {"raw": 83, "formal": 80, "excluded": 3, "pending": 0}

    service = BoreholeQualityService(project)
    service.run_checks()
    assert service.confirm_exclusions() == 12
    assert service.apply_auto_fixes() == 155
    assert service.unresolved_error_count == 0

    auto_fix_events = [
        event
        for record in repository.query("surveys", raw=True)
        for event in record.modification_history
        if event.source.startswith("m8_quality:auto_fix")
    ]
    corrected_records = {
        record.record_id
        for record in repository.query("surveys", raw=True)
        if any(event.source.startswith("m8_quality:auto_fix") for event in record.modification_history)
    }
    assert len(corrected_records) == 112
    assert len(auto_fix_events) == 155
    assert all(event.before is not None and event.after is not None for event in auto_fix_events)
    assert any(event.before["azimuth"] != event.after["azimuth"] for event in auto_fix_events)

    formal = repository.query("surveys", RecordState.FORMAL)
    assert len(formal) == 148
    assert all(0 <= float(record.values["azimuth"]) < 360 for record in formal)
    assert all(-90 <= float(record.values["dip"]) <= 90 for record in formal)
    projected = {
        (borehole.borehole_id, station.measured_depth): (station.azimuth, station.dip)
        for borehole in project.borehole_collection
        for station in borehole.survey.stations
    }
    assert len(projected) == 148
    for record in formal:
        key = (record.hole_id, float(record.values["measured_depth"]))
        assert projected[key] == (float(record.values["azimuth"]), float(record.values["dip"]))


def test_rebuild_does_not_silently_change_database_values() -> None:
    project, repository = _demo_project(("collars", "surveys"))
    record = next(
        record
        for record in repository.query("surveys", RecordState.FORMAL)
        if float(record.values["azimuth"]) < 0 or float(record.values["azimuth"]) >= 360
    )
    before = dict(record.values)
    history_before = list(record.modification_history)
    repository.rebuild_formal_collection()
    assert record.values == before
    assert record.modification_history == history_before


def test_confirmed_exclusions_do_not_block_but_pending_and_open_errors_do() -> None:
    project, repository = _demo_project()
    service = BoreholeQualityService(project)
    service.run_checks()
    service.confirm_exclusions()
    service.apply_auto_fixes()
    workflow = WorkflowController()
    service.confirm_complete(workflow)
    assert workflow.get_step("import").status == StepStatus.COMPLETED
    assert workflow.get_step("clean").status == StepStatus.COMPLETED
    assert workflow.get_step("holdout").status == StepStatus.READY
    assert repository.database.counts()["excluded"] == 12

    pending_project = Project()
    pending_repository = BoreholeRepository(pending_project)
    pending_repository.import_dataframe(
        "surveys",
        pd.DataFrame([{"hole_id": "UNKNOWN", "measured_depth": 0, "azimuth": 0, "dip": -90}]),
        "surveys.csv",
    )
    pending_service = BoreholeQualityService(pending_project)
    pending_service.run_checks()
    with pytest.raises(ValueError, match="Pending"):
        pending_service.confirm_complete(WorkflowController())

    error_project = Project()
    error_repository = BoreholeRepository(error_project)
    error_repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "A", "collar_x": 0, "collar_y": 0, "collar_z": 0, "final_depth": 20}]),
        "collars.csv",
    )
    error_repository.import_dataframe(
        "surveys",
        pd.DataFrame([{"hole_id": "A", "measured_depth": 0, "azimuth": -1, "dip": -90}]),
        "surveys.csv",
    )
    error_service = BoreholeQualityService(error_project)
    error_service.run_checks()
    with pytest.raises(ValueError, match="unresolved ERROR"):
        error_service.confirm_complete(WorkflowController())


def test_rqd_and_domain_overlaps_are_detected_without_altering_records() -> None:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "A", "collar_x": 0, "collar_y": 0, "collar_z": 0, "final_depth": 100}]),
        "collars.csv",
    )
    repository.import_dataframe(
        "rqd",
        pd.DataFrame(
            [
                {"hole_id": "A", "from_depth": 0, "to_depth": 60, "rqd": 70},
                {"hole_id": "A", "from_depth": 50, "to_depth": 80, "rqd": 80},
            ]
        ),
        "rqd.csv",
    )
    repository.import_dataframe(
        "domain_intervals",
        pd.DataFrame(
            [
                {"hole_id": "A", "from_depth": 0, "to_depth": 70, "domain_id": 1},
                {"hole_id": "A", "from_depth": 60, "to_depth": 90, "domain_id": 2},
            ]
        ),
        "domains.csv",
    )
    service = BoreholeQualityService(project)
    issues = service.run_checks()
    overlap_types = {issue.data_type for issue in issues if "_overlap:" in issue.code}
    assert overlap_types == {"rqd", "domain_intervals"}
    assert all(issue.severity == QualitySeverity.WARNING for issue in issues if "_overlap:" in issue.code)
    assert repository.database.counts() == {"raw": 5, "formal": 5, "excluded": 0, "pending": 0}


def test_checks_and_auto_fixes_are_idempotent() -> None:
    project, _ = _demo_project()
    service = BoreholeQualityService(project)
    service.run_checks()
    issue_ids = [issue.issue_id for issue in service.issues]
    service.confirm_exclusions()
    assert service.apply_auto_fixes() == 155
    count_after = len(service.issues)
    assert service.apply_auto_fixes() == 0
    service.run_checks()
    assert len(service.issues) == count_after
    assert set(issue_ids).issubset({issue.issue_id for issue in service.issues})
    assert service.confirm_exclusions() == 0


def test_quality_issues_confirmation_and_history_round_trip(tmp_path: Path) -> None:
    project, repository = _demo_project()
    workflow = WorkflowController()
    project._m7_data = {"workflow": workflow}
    service = BoreholeQualityService(project)
    service.run_checks()
    service.confirm_exclusions()
    service.apply_auto_fixes()
    service.confirm_complete(workflow, note="reviewed")
    json_report = service.export_json(tmp_path / "quality.json")
    csv_report = service.export_csv(tmp_path / "quality.csv")
    assert json.loads(json_report.read_text(encoding="utf-8"))["counts"]["raw"] == 284
    assert "issue_id,record_id,data_type" in csv_report.read_text(encoding="utf-8-sig")
    path = tmp_path / "quality.dfnproj"
    ZipProjectStore().save(project, path)

    reopened = ZipProjectStore().load(path)
    database = reopened.borehole_database
    assert database.counts() == {"raw": 284, "formal": 272, "excluded": 12, "pending": 0}
    assert database.quality_confirmed_at is not None
    assert database.quality_confirmation_note == "reviewed"
    assert database.unresolved_error_count == 0
    assert any(issue.status == QualityIssueStatus.RESOLVED for issue in database.quality_issues)
    assert all(
        record.exclusion_confirmed_at is not None for record in database.records if record.state == RecordState.EXCLUDED
    )
    assert (
        sum(
            event.source.startswith("m8_quality:auto_fix")
            for record in database.query("surveys", raw=True)
            for event in record.modification_history
        )
        == 155
    )
