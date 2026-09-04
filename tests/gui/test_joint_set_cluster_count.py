"""Real-button GUI regression tests for Mode B K=4, 5, and 6."""

from pathlib import Path

import pandas as pd
import pytest

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.ui.dialogs.m7_joint_set_dialog import M7JointSetDialog
from dfn_cave_studio.ui.i18n import LANGUAGE_CHINESE, LANGUAGE_ENGLISH, language_manager
from dfn_cave_studio.ui.qt_adapter import QApplication, QDialogButtonBox, Qt
from dfn_cave_studio.models.borehole_database import BoreholeDataType, RecordState
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore


DEMO = Path(__file__).parents[2] / "examples" / "m7_demo"


@pytest.fixture(scope="module")
def joint_set_project() -> Project:
    """Build a real cleaned demo project with a locked 60/20 holdout."""
    project = Project()
    repository = BoreholeRepository(project)
    for name in ("surveys", "collars", "fractures", "rqd", "domain_intervals"):
        repository.import_dataframe(name, pd.read_csv(DEMO / f"{name}.csv"), str(DEMO / f"{name}.csv"))
    quality = BoreholeQualityService(project)
    quality.run_checks()
    quality.apply_auto_fixes()
    quality.confirm_exclusions()

    hole_ids = sorted(hole.borehole_id for hole in project.borehole_collection)
    holdout = HoldoutService()
    holdout.select_manual(hole_ids, ["BH-07", "BH-08"])
    holdout.lock()
    set_holdout(project, holdout)
    return project


@pytest.mark.parametrize("requested_k", [4, 5, 6])
def test_identify_button_displays_exact_requested_rows(
    joint_set_project: Project, qtbot, requested_k: int
) -> None:
    """The production button path renders K non-empty rows with all 60 assignments."""
    dialog = M7JointSetDialog(joint_set_project, WorkflowController())
    qtbot.addWidget(dialog)
    dialog.show()
    dialog._mode_combo.setCurrentIndex(dialog._mode_combo.findData("automatic"))
    dialog._n_clusters_spin.setValue(requested_k)
    dialog._seed_spin.setValue(42)

    qtbot.mouseClick(dialog._identify_btn, Qt.MouseButton.LeftButton)

    counts = [int(dialog._result_table.item(row, 5).text()) for row in range(dialog._result_table.rowCount())]
    assert dialog._result_table.rowCount() == requested_k
    assert all(count > 0 for count in counts)
    assert sum(counts) == 60
    assert "Calibration fractures used: 60" in dialog._stats_label.text()
    assert "Validation fractures excluded: 20" in dialog._stats_label.text()


def test_k7_ok_commits_assignments_to_canonical_database(qtbot, tmp_path) -> None:
    """OK persists the fitted labels used by M9, not only seven display models."""
    project = Project()
    repository = BoreholeRepository(project)
    for name in ("surveys", "collars", "fractures", "rqd", "domain_intervals"):
        repository.import_dataframe(name, pd.read_csv(DEMO / f"{name}.csv"), str(DEMO / f"{name}.csv"))
    quality = BoreholeQualityService(project)
    quality.run_checks()
    quality.apply_auto_fixes()
    quality.confirm_exclusions()
    hole_ids = sorted(hole.borehole_id for hole in project.borehole_collection)
    holdout = HoldoutService()
    holdout.select_manual(hole_ids, ["BH-07", "BH-08"])
    holdout.lock()
    set_holdout(project, holdout)
    workflow = WorkflowController()
    dialog = M7JointSetDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog._mode_combo.setCurrentIndex(dialog._mode_combo.findData("automatic"))
    dialog._n_clusters_spin.setValue(7)
    dialog._seed_spin.setValue(42)

    qtbot.mouseClick(dialog._identify_btn, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(dialog._button_box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    formal = repository.query(BoreholeDataType.FRACTURES, RecordState.FORMAL)
    assert len(project.joint_sets) == 7
    assert {item.provenance["requested_number_of_sets"] for item in project.joint_sets} == {7}
    assert set(record.values["set_id"] for record in formal) == set(range(1, 8))
    assert all(
        any(event.action == "assign_joint_set" for event in record.modification_history)
        for record in formal
        if record.modification_source == "m8_joint_set:auto_assignment"
    )
    assert {
        observation.set_id
        for hole in project.borehole_collection
        for observation in hole.fracture_observations
    } == set(range(1, 8))
    path = tmp_path / "m8-k7.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)
    assert {item.set_id for item in restored.joint_sets} == set(range(1, 8))
    assert {
        observation.set_id
        for hole in restored.borehole_collection
        for observation in hole.fracture_observations
    } == set(range(1, 8))


def test_joint_set_headers_switch_language_without_project_mutation(qtbot) -> None:
    """A-only translations update presentation while retaining project state."""
    manager = language_manager()
    original_language = manager.language
    manager.set_language(LANGUAGE_ENGLISH, persist=False, force=True)
    project = Project()
    workflow = WorkflowController()
    dialog = M7JointSetDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.show()
    project_before = project.model_dump()
    workflow_before = workflow.to_dict()
    try:
        assert dialog._result_table.horizontalHeaderItem(5).text() == "Calibration Count"
        assert manager.set_language(LANGUAGE_CHINESE, persist=False, force=True)
        QApplication.processEvents()
        assert dialog._result_table.horizontalHeaderItem(5).text() == "校准集数量"
        assert dialog._result_table.horizontalHeaderItem(9).text() == "验证集数量"
        assert project.model_dump() == project_before
        assert workflow.to_dict() == workflow_before
        assert manager.set_language(LANGUAGE_ENGLISH, persist=False, force=True)
        QApplication.processEvents()
        assert dialog._result_table.horizontalHeaderItem(5).text() == "Calibration Count"
    finally:
        manager.set_language(original_language, persist=False, force=True)
