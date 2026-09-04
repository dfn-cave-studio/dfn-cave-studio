"""Real-button GUI regression tests for M10 generation and layer management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController
from dfn_cave_studio.ui.dialogs.m10_dialog import M10ExplicitDFNDialog
from dfn_cave_studio.ui.i18n import LANGUAGE_CHINESE, LANGUAGE_ENGLISH, language_manager
from dfn_cave_studio.ui.qt_adapter import QApplication, QFileDialog, QDialogButtonBox, QMessageBox, Qt, QThreadPool
from dfn_cave_studio.visualization.dfn_layer_manager import DFNLayerManager
from tests.integration.test_m10_persistence_export import make_m10_project
from tests.integration.test_joint_set_seven_propagation import _generate, _seven_set_project


class FakePlotter:
    def __init__(self):
        self.actors = {}
        self.remove_actor = MagicMock(side_effect=self._remove)
        self.render = MagicMock()

    def add_mesh(self, mesh, *, name, **kwargs):
        del mesh, kwargs
        actor = MagicMock()
        self.actors[name] = actor
        return actor

    def _remove(self, actor, render=False):
        del render
        for name, candidate in list(self.actors.items()):
            if candidate is actor:
                self.actors.pop(name)


def _workflow():
    workflow = WorkflowController()
    for step in ("bounds", "voxel_grid", "density", "size", "parameter_field"):
        workflow.complete_step(step)
    workflow.mark_ready("explicit_dfn")
    return workflow


def test_generate_button_then_ok_commits_complete_realization(qtbot, monkeypatch):
    project = make_m10_project()
    workflow = _workflow()
    manager = DFNLayerManager(FakePlotter())
    dialog = M10ExplicitDFNDialog(project, workflow, manager)
    qtbot.addWidget(dialog)
    dialog.condition_observations.setChecked(False)
    monkeypatch.setattr(QThreadPool, "start", lambda self, worker: worker.run())
    qtbot.mouseClick(dialog.generate_button, Qt.MouseButton.LeftButton)
    assert dialog._pending_realizations is not None
    assert project.m10_state.realizations == []
    assert dialog.realizations_table.rowCount() == 1
    qtbot.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
    assert dialog.committed_changes
    assert len(project.m10_state.realizations) == 1
    assert workflow.get_step("explicit_dfn").status == StepStatus.COMPLETED


def test_cancel_after_generation_rolls_back_project(qtbot, monkeypatch):
    project = make_m10_project()
    before = project.m10_state.model_copy(deep=True)
    dialog = M10ExplicitDFNDialog(project, _workflow(), DFNLayerManager(FakePlotter()))
    qtbot.addWidget(dialog)
    dialog.condition_observations.setChecked(False)
    monkeypatch.setattr(QThreadPool, "start", lambda self, worker: worker.run())
    qtbot.mouseClick(dialog.generate_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel), Qt.MouseButton.LeftButton)
    assert project.m10_state == before


def test_render_buttons_replace_actor_and_display_does_not_change_science(qtbot, monkeypatch):
    project = make_m10_project()
    workflow = _workflow()
    plotter = FakePlotter()
    manager = DFNLayerManager(plotter)
    dialog = M10ExplicitDFNDialog(project, workflow, manager)
    qtbot.addWidget(dialog)
    dialog.condition_observations.setChecked(False)
    monkeypatch.setattr(QThreadPool, "start", lambda self, worker: worker.run())
    qtbot.mouseClick(dialog.generate_button, Qt.MouseButton.LeftButton)
    dialog.realizations_table.selectRow(0)
    workflow_before = workflow.to_dict()
    m9_before = project.m9_state.model_copy(deep=True)
    m10_before = project.m10_state.model_copy(deep=True)
    for _ in range(10):
        qtbot.mouseClick(dialog.render_all_button, Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == 1
    assert len(plotter.actors) == 1
    dialog.layers_table.selectRow(0)
    qtbot.mouseClick(dialog.toggle_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(dialog.opacity_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(dialog.clear_all_button, Qt.MouseButton.LeftButton)
    assert manager.list_layers() == []
    assert project.m9_state.model_dump(exclude={"parameter_field_arrays"}) == m9_before.model_dump(
        exclude={"parameter_field_arrays"}
    )
    for name in m9_before.parameter_field_arrays:
        assert (project.m9_state.parameter_field_arrays[name] == m9_before.parameter_field_arrays[name]).all()
    assert project.m10_state == m10_before
    assert workflow.to_dict() == workflow_before


def test_layer_buttons_are_safe_without_selection(qtbot):
    dialog = M10ExplicitDFNDialog(make_m10_project(), _workflow(), DFNLayerManager(FakePlotter()))
    qtbot.addWidget(dialog)
    for button in (dialog.toggle_button, dialog.opacity_button, dialog.remove_button, dialog.clear_current_button):
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def test_multiscale_controls_defaults_manual_switch_estimate_and_color_modes(qtbot, monkeypatch):
    project = make_m10_project()
    manager = DFNLayerManager(FakePlotter())
    dialog = M10ExplicitDFNDialog(project, _workflow(), manager)
    qtbot.addWidget(dialog)
    assert dialog.threshold_mode.currentText() == "Auto"
    assert not dialog.generate_small.isChecked()
    assert dialog.generate_medium.isChecked()
    assert dialog.generate_large.isChecked()
    dialog.threshold_mode.setCurrentText("Manual")
    dialog.manual_sm.setValue(1.2)
    dialog.manual_ml.setValue(2.4)
    dialog._refresh_estimate()
    assert dialog.manual_sm.isEnabled()
    assert dialog.size_estimate_table.rowCount() == 3
    dialog.condition_observations.setChecked(False)
    monkeypatch.setattr(QThreadPool, "start", lambda self, worker: worker.run())
    qtbot.mouseClick(dialog.generate_button, Qt.MouseButton.LeftButton)
    dialog.realizations_table.selectRow(0)
    for color_mode in ("Joint Set", "Domain", "Source", "Size Class"):
        dialog.color_by.setCurrentText(color_mode)
        qtbot.mouseClick(dialog.render_all_button, Qt.MouseButton.LeftButton)
        layers = manager.list_layers()
        assert len(layers) == 1
        assert layers[0].color_by == color_mode
        assert "explicit" in dialog.visible_count_label.text()
    before = manager.list_layers()[0].fracture_count
    dialog.display_size_class.setCurrentText("MEDIUM")
    qtbot.mouseClick(dialog.toggle_size_class_button, Qt.MouseButton.LeftButton)
    assert "MEDIUM (hidden)" in dialog.legend_label.text()
    assert manager.list_layers()[0].fracture_count <= before


def test_outside_deterministic_import_warns_through_real_button(qtbot, monkeypatch):
    dialog = M10ExplicitDFNDialog(make_m10_project(), _workflow(), DFNLayerManager(FakePlotter()))
    qtbot.addWidget(dialog)
    path = Path(__file__).parents[2] / "examples/m10_demo/deterministic_structures.csv"
    warnings = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), "CSV (*.csv)"))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args[2]))
    qtbot.mouseClick(dialog.import_structures_button, Qt.MouseButton.LeftButton)
    assert warnings
    assert "FAULT-DEMO-01" in warnings[0]
    assert "center=" in warnings[0]


def test_render_modes_rebuild_without_duplicate_actors_or_visible_counts(qtbot, monkeypatch):
    project = make_m10_project()
    plotter = FakePlotter()
    manager = DFNLayerManager(plotter)
    dialog = M10ExplicitDFNDialog(project, _workflow(), manager)
    qtbot.addWidget(dialog)
    dialog.condition_observations.setChecked(False)
    monkeypatch.setattr(QThreadPool, "start", lambda self, worker: worker.run())
    qtbot.mouseClick(dialog.generate_button, Qt.MouseButton.LeftButton)
    dialog.realizations_table.selectRow(0)
    realization = dialog._selected_realization()

    qtbot.mouseClick(dialog.render_all_button, Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == len(plotter.actors) == 1
    qtbot.mouseClick(dialog.render_set_button, Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == len(plotter.actors) == 1
    qtbot.mouseClick(dialog.render_deterministic_button, Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == len(plotter.actors) == 1
    qtbot.mouseClick(dialog.render_all_button, Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == len(plotter.actors) == 1
    visible = int(dialog.visible_count_label.text().split()[1].replace(",", ""))
    assert visible <= realization.fracture_count


def test_seven_set_summary_and_real_render_button_show_all_groups(qtbot):
    """The production summary and Render by Set button expose all seven groups."""
    project = _seven_set_project()
    realization = _generate(project)
    plotter = FakePlotter()
    manager = DFNLayerManager(plotter)
    dialog = M10ExplicitDFNDialog(project, _workflow(), manager)
    qtbot.addWidget(dialog)
    dialog.realizations_table.selectRow(0)

    assert dialog.group_summary_table.rowCount() == 7
    assert [int(dialog.group_summary_table.item(row, 1).text()) for row in range(7)] == list(range(1, 8))
    qtbot.mouseClick(dialog.render_set_button, Qt.MouseButton.LeftButton)

    assert len(manager.list_layers()) == len(plotter.actors) == 7
    assert all(layer.fracture_count > 0 for layer in manager.list_layers())
    assert all(f"Joint Set {set_id}" in dialog.legend_label.text() for set_id in range(1, 8))
    assert realization.fracture_count >= sum(layer.fracture_count for layer in manager.list_layers())


def test_seven_set_summary_translation_preserves_codes_and_science(qtbot):
    """Chinese presentation never changes stable statuses, set IDs, or generated arrays."""
    manager = language_manager()
    original_language = manager.language
    manager.set_language(LANGUAGE_ENGLISH, persist=False, force=True)
    project = _seven_set_project()
    realization = _generate(project)
    workflow = _workflow()
    dialog = M10ExplicitDFNDialog(project, workflow, DFNLayerManager(FakePlotter()))
    qtbot.addWidget(dialog)
    dialog.show()
    arrays_before = {name: value.copy() for name, value in realization.geometry_arrays.items()}
    workflow_before = workflow.to_dict()
    try:
        assert dialog.group_summary_table.horizontalHeaderItem(3).text() == "Orientation Status"
        assert manager.set_language(LANGUAGE_CHINESE, persist=False, force=True)
        QApplication.processEvents()
        assert dialog.group_summary_table.horizontalHeaderItem(3).text() == "方向状态"
        assert dialog.group_summary_table.item(0, 3).text() == "有效"
        assert dialog.group_summary_table.item(0, 3).data(Qt.ItemDataRole.UserRole) == "valid"
        assert [dialog.group_summary_table.item(row, 1).text() for row in range(7)] == [str(value) for value in range(1, 8)]
        assert workflow.to_dict() == workflow_before
        for name, expected in arrays_before.items():
            assert (realization.geometry_arrays[name] == expected).all()
        assert manager.set_language(LANGUAGE_ENGLISH, persist=False, force=True)
        QApplication.processEvents()
        assert dialog.group_summary_table.horizontalHeaderItem(3).text() == "Orientation Status"
        assert dialog.group_summary_table.item(0, 3).text() == "Valid"
    finally:
        manager.set_language(original_language, persist=False, force=True)
