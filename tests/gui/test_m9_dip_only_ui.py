"""GUI button-path tests for optional fracture dip direction."""

import pandas as pd

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.ui.dialogs.m8_import_dialog import M8ImportDialog
from dfn_cave_studio.ui.dialogs.m7_joint_set_dialog import M7JointSetDialog
from dfn_cave_studio.services.joint_set_service import JointSetIdentificationResult
from dfn_cave_studio.ui.qt_adapter import QComboBox, QDialogButtonBox, QPushButton, Qt


def _button(dialog, text: str) -> QPushButton:
    return next(button for button in dialog.findChildren(QPushButton) if button.text() == text)


def _project_with_collar() -> Project:
    project = Project()
    BoreholeRepository(project).import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "BH", "collar_x": 0, "collar_y": 0, "collar_z": 100, "final_depth": 100}]),
        "collars.csv",
    )
    return project


def test_import_dialog_without_direction_column_marks_dip_only_and_required_fields(qtbot, tmp_path) -> None:
    path = tmp_path / "dip-only.csv"
    pd.DataFrame([{"hole_id": "BH", "depth": 10, "dip": 45, "set_id": 1}]).to_csv(path, index=False)
    project = _project_with_collar()
    dialog = M8ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._type_combo.setCurrentText("fractures")
    dialog._path_edit.setText(str(path))
    dialog.show()
    qtbot.mouseClick(_button(dialog, "Preview"), Qt.MouseButton.LeftButton)

    mappings = {
        dialog._mapping_table.cellWidget(row, 1).currentText(): dialog._mapping_table.item(row, 2).text()
        for row in range(dialog._mapping_table.rowCount())
        if isinstance(dialog._mapping_table.cellWidget(row, 1), QComboBox)
    }
    assert mappings["hole_id"] == "Yes"
    assert mappings["depth"] == "Yes"
    assert mappings["dip"] == "Yes"
    assert "dip_direction" not in mappings

    qtbot.mouseClick(_button(dialog, "Stage This Import"), Qt.MouseButton.LeftButton)
    assert "dip only=1" in dialog._status.text()
    ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
    qtbot.mouseClick(ok, Qt.MouseButton.LeftButton)
    assert dialog.committed_changes is True
    record = project.borehole_database.query("fractures", "formal")[0]
    assert record.values["dip_direction"] is None
    assert record.values["orientation_completeness"] == "dip_only"


def test_import_dialog_cancel_rolls_back_dip_only_rows(qtbot, tmp_path) -> None:
    path = tmp_path / "dip-only-cancel.csv"
    pd.DataFrame([{"hole_id": "BH", "depth": 10, "dip": 45}]).to_csv(path, index=False)
    project = _project_with_collar()
    dialog = M8ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._type_combo.setCurrentText("fractures")
    dialog._path_edit.setText(str(path))
    dialog.show()
    qtbot.mouseClick(_button(dialog, "Preview"), Qt.MouseButton.LeftButton)
    qtbot.mouseClick(_button(dialog, "Stage This Import"), Qt.MouseButton.LeftButton)
    assert project.borehole_database.query("fractures", "formal")
    cancel = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Cancel)
    qtbot.mouseClick(cancel, Qt.MouseButton.LeftButton)
    assert project.borehole_database.query("fractures", "formal") == []


def test_mode_a_table_shows_dip_only_only_set_without_fake_orientation(qtbot) -> None:
    project = Project()
    workflow = WorkflowController()
    dialog = M7JointSetDialog(project, workflow)
    qtbot.addWidget(dialog)
    result = JointSetIdentificationResult(0)
    result.mode = "imported"
    result.set_counts[9] = {"total": 2, "full_orientation": 0, "dip_only": 2}
    result.calibration_count = 2
    result.dip_only_count = 2
    dialog._populate_results(result)
    assert dialog._result_table.rowCount() == 1
    assert dialog._result_table.item(0, 2).text() == "—"
    assert dialog._result_table.item(0, 5).text() == "2"
    assert dialog._result_table.item(0, 7).text() == "2"
    assert dialog._result_table.item(0, 8).text() == "INSUFFICIENT_ORIENTATION_DATA"
