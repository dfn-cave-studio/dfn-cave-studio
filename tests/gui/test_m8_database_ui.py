"""GUI tests that verify M8 table contents and transaction lifecycle."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import numpy as np

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, ScalarFieldMetadata, ScalarFieldResult
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.persistence.project_store import ProjectStore
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.spatial_domain_service import SpatialDomainService
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController
from dfn_cave_studio.ui.dialogs.m8_import_dialog import M8ImportDialog
from dfn_cave_studio.ui.dialogs.m8_quality_dialog import M8QualityDialog
from dfn_cave_studio.ui.panels.borehole_database_panel import BoreholeDatabasePanel
from dfn_cave_studio.ui.qt_adapter import QAction, QDialog, QDialogButtonBox, QMessageBox, QPoint, QTimer, Qt

DEMO = Path(__file__).parents[2] / "examples" / "m7_demo"


def _demo_quality_dialog(qtbot):
    project = Project()
    workflow = WorkflowController()
    project._m7_data = {"workflow": workflow}
    repository = BoreholeRepository(project)
    for name in ("surveys", "collars", "fractures", "rqd", "domain_intervals"):
        repository.import_dataframe(name, pd.read_csv(DEMO / f"{name}.csv"), str(DEMO / f"{name}.csv"))
    dialog = M8QualityDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.show()
    return project, repository, workflow, dialog


def _issue_table_row(dialog: M8QualityDialog, field: str, original_value: str) -> int:
    headers = {
        dialog._table.horizontalHeaderItem(column).text(): column for column in range(dialog._table.columnCount())
    }
    for row in range(dialog._table.rowCount()):
        if (
            dialog._table.item(row, headers["field"]).text() == field
            and dialog._table.item(row, headers["original_value"]).text() == original_value
        ):
            return row
    raise AssertionError(f"Issue row not found: {field}={original_value}")


def _main_window_with_demo(qtbot):
    from dfn_cave_studio.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window._on_new_project()
    project = window._project_store.current_project
    repository = BoreholeRepository(project)
    for name in ("surveys", "collars", "fractures", "rqd", "domain_intervals"):
        repository.import_dataframe(name, pd.read_csv(DEMO / f"{name}.csv"), str(DEMO / f"{name}.csv"))
    window._database_panel.refresh()
    window.show()
    return window, project


def _run_quality_ok_path(window, qtbot, *, confirm_exclusions: bool, confirm_quality: bool) -> dict:
    result = {}

    def drive_dialog() -> None:
        dialog = window.findChild(M8QualityDialog)
        if dialog is None:
            result["error"] = AssertionError("M8QualityDialog was not opened")
            return
        try:
            started = time.perf_counter()
            qtbot.mouseClick(dialog._all_fix_button, Qt.MouseButton.LeftButton)
            result["after_fix"] = dialog._service.unresolved_error_count
            if confirm_exclusions:
                qtbot.mouseClick(dialog._confirm_all_button, Qt.MouseButton.LeftButton)
                result["after_exclusions"] = dialog._service.unresolved_error_count
            if confirm_quality:
                qtbot.mouseClick(dialog._complete_button, Qt.MouseButton.LeftButton)
            result["dialog_work_s"] = time.perf_counter() - started
            ok_button = dialog._dialog_buttons.button(QDialogButtonBox.StandardButton.Ok)
            qtbot.mouseClick(ok_button, Qt.MouseButton.LeftButton)
            result["dialog_result"] = dialog.result()
        except Exception as error:
            result["error"] = error
            dialog.reject()

    QTimer.singleShot(0, drive_dialog)
    started = time.perf_counter()
    window._m7_clean()
    result["total_s"] = time.perf_counter() - started
    assert "error" not in result, result.get("error")
    assert result["total_s"] < 5.0
    return result


def test_database_panel_displays_real_rows_fields_and_counts(qtbot) -> None:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame(
            [
                {"borehole_id": "A", "collar_x": 1, "collar_y": 2, "collar_z": 3, "final_depth": 40},
                {"borehole_id": "B", "collar_x": 4, "collar_y": 5, "collar_z": 6, "final_depth": 50},
            ]
        ),
        "collars.csv",
    )
    panel = BoreholeDatabasePanel(project)
    qtbot.addWidget(panel)
    assert panel._table.rowCount() == 2
    headers = [panel._table.horizontalHeaderItem(index).text() for index in range(panel._table.columnCount())]
    assert "collar_x" in headers
    assert "final_depth" in headers
    assert "Raw 2" in panel._counts.text()
    visible_values = {
        panel._table.item(row, column).text()
        for row in range(panel._table.rowCount())
        for column in range(panel._table.columnCount())
    }
    assert {"A", "B", "40", "50"}.issubset(visible_values)


def test_import_dialog_cancel_restores_project(qtbot) -> None:
    project = Project()
    repository = BoreholeRepository(project)
    dialog = M8ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "A", "collar_x": 0, "collar_y": 0, "collar_z": 0, "final_depth": 10}]),
        "a.csv",
    )
    dialog._staged_changes = True
    dialog.reject()
    assert repository.database.counts("collars")["raw"] == 0
    assert not dialog.committed_changes


def test_staged_import_cancel_does_not_stale_workflow(qtbot) -> None:
    project = Project()
    workflow = WorkflowController()
    for step in workflow.get_steps():
        workflow.complete_step(step.step_id)
    project._m7_data = {"workflow": workflow}
    dialog = M8ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._repository.import_dataframe(
        "rqd",
        pd.DataFrame([{"hole_id": "UNKNOWN", "from_depth": 0, "to_depth": 1, "rqd": 50}]),
        "rqd.csv",
        modification_source="staged_import",
    )
    dialog._staged_changes = True
    dialog.reject()
    assert workflow.get_step("clean").status == StepStatus.COMPLETED
    assert project.borehole_database.counts("rqd")["raw"] == 0


def test_import_dialog_accept_reports_committed_change(qtbot) -> None:
    project = Project()
    dialog = M8ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "A", "collar_x": 0, "collar_y": 0, "collar_z": 0, "final_depth": 10}]),
        "a.csv",
    )
    dialog._staged_changes = True
    dialog._accept()
    assert dialog.committed_changes
    assert project.borehole_database.counts("collars")["formal"] == 1


def test_panel_cancel_keeps_clean_and_accepted_change_marks_dirty(qtbot) -> None:
    project = Project()
    store = ProjectStore()
    store.adopt_project(project)
    panel = BoreholeDatabasePanel(project, on_changed=store.mark_dirty)
    qtbot.addWidget(panel)
    cancelled = M8ImportDialog(project)
    qtbot.addWidget(cancelled)
    cancelled.reject()
    assert not store.is_dirty
    panel._repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "A", "collar_x": 0, "collar_y": 0, "collar_z": 0, "final_depth": 10}]),
        "a.csv",
    )
    panel._notify_changed()
    assert store.is_dirty


def test_quality_dialog_is_real_transactional_interface(qtbot) -> None:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "A", "collar_x": 0, "collar_y": 0, "collar_z": 0, "final_depth": 20}]),
        "collars.csv",
    )
    repository.import_dataframe(
        "surveys",
        pd.DataFrame([{"hole_id": "A", "measured_depth": 0, "azimuth": -1, "dip": -91}]),
        "surveys.csv",
    )
    workflow = WorkflowController()
    dialog = M8QualityDialog(project, workflow)
    qtbot.addWidget(dialog)
    assert dialog._table.rowCount() == 2
    assert "Unresolved ERROR 2" in dialog._summary.text()
    dialog._accept_all_fixes()
    assert repository.query("surveys", "formal")[0].values["azimuth"] == 359
    assert repository.query("surveys", "formal")[0].values["dip"] == -90
    assert not dialog.committed_changes
    dialog.reject()
    restored = BoreholeRepository(project).query("surveys", "formal")[0]
    assert restored.values["azimuth"] == -1
    assert restored.values["dip"] == -91
    assert not dialog.committed_changes


def test_quality_buttons_apply_demo_fixes_once_and_refresh_real_table(monkeypatch, qtbot) -> None:
    project, repository, workflow, dialog = _demo_quality_dialog(qtbot)
    unexpected_errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args: unexpected_errors.append(_args[-1]))
    button_states_during_apply = []
    original_apply = dialog._service.apply_auto_fixes

    def observed_apply(issue_ids=None):
        button_states_during_apply.append(
            tuple(
                button.isEnabled()
                for button in (dialog._rerun_button, dialog._selected_fix_button, dialog._all_fix_button)
            )
        )
        return original_apply(issue_ids)

    monkeypatch.setattr(dialog._service, "apply_auto_fixes", observed_apply)
    assert project.borehole_database.counts() == {"raw": 284, "formal": 272, "excluded": 12, "pending": 0}
    assert dialog._service.unresolved_error_count == 167
    assert "Unresolved ERROR 167" in dialog._summary.text()
    for button in (dialog._rerun_button, dialog._selected_fix_button, dialog._all_fix_button):
        assert not button.isCheckable()
        assert not button.autoRepeat()

    qtbot.mouseClick(dialog._all_fix_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._service.unresolved_error_count == 12)
    assert unexpected_errors == []
    assert button_states_during_apply[0] == (False, False, False)
    assert dialog._service.last_auto_fix_record_count == 112
    assert dialog._service.last_auto_fix_field_count == 155
    assert project.borehole_database.counts() == {"raw": 284, "formal": 272, "excluded": 12, "pending": 0}
    assert sum(len(borehole.survey.stations) for borehole in project.borehole_collection) == 148

    headers = {
        dialog._table.horizontalHeaderItem(column).text(): column for column in range(dialog._table.columnCount())
    }
    azimuth_row = _issue_table_row(dialog, "azimuth", "-0.74")
    dip_row = _issue_table_row(dialog, "dip", "-90.05")
    assert dialog._table.item(azimuth_row, headers["current_value"]).text() == "359.26"
    assert dialog._table.item(azimuth_row, headers["status"]).text() == "resolved"
    assert dialog._table.item(dip_row, headers["current_value"]).text() == "-90.0"
    assert dialog._table.item(dip_row, headers["status"]).text() == "resolved"

    auto_events = [
        event
        for record in repository.query("surveys", raw=True)
        for event in record.modification_history
        if event.action == "auto_fix"
    ]
    assert len(auto_events) == 155
    assert all(event.issue_id and event.before is not None and event.after is not None for event in auto_events)

    issue_count = len(project.borehole_database.quality_issues)
    qtbot.mouseClick(dialog._rerun_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._rerun_button.isEnabled())
    assert dialog._service.unresolved_error_count == 12
    assert len(project.borehole_database.quality_issues) == issue_count

    for _ in range(4):
        qtbot.mouseClick(dialog._all_fix_button, Qt.MouseButton.LeftButton)
    assert dialog._service.unresolved_error_count == 12
    assert (
        len(
            [
                event
                for record in repository.query("surveys", raw=True)
                for event in record.modification_history
                if event.action == "auto_fix"
            ]
        )
        == 155
    )

    qtbot.mouseClick(dialog._confirm_all_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._service.unresolved_error_count == 0)
    qtbot.mouseClick(dialog._complete_button, Qt.MouseButton.LeftButton)
    assert workflow.get_step("clean").status == StepStatus.COMPLETED
    assert workflow.get_step("holdout").status == StepStatus.READY
    assert project.borehole_database.counts() == {"raw": 284, "formal": 272, "excluded": 12, "pending": 0}


def test_quality_button_batch_failure_rolls_back_database_issues_and_workflow(monkeypatch, qtbot) -> None:
    project, _repository, workflow, dialog = _demo_quality_dialog(qtbot)
    database_before = project.borehole_database.model_dump_json()
    collection_before = project.borehole_collection.model_dump_json()
    workflow_before = workflow.to_dict()
    messages = []

    def fail_checks():
        raise RuntimeError("injected batch failure")

    monkeypatch.setattr(dialog._service, "run_checks", fail_checks)
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args: messages.append(_args[-1]))
    qtbot.mouseClick(dialog._all_fix_button, Qt.MouseButton.LeftButton)

    assert messages == ["injected batch failure"]
    assert project.borehole_database.model_dump_json() == database_before
    assert project.borehole_collection.model_dump_json() == collection_before
    assert workflow.to_dict() == workflow_before
    assert dialog._service.unresolved_error_count == 167
    assert all(
        button.isEnabled() for button in (dialog._rerun_button, dialog._selected_fix_button, dialog._all_fix_button)
    )


def test_quality_path_a_auto_fix_then_ok_returns_to_main_window(qtbot) -> None:
    window, project = _main_window_with_demo(qtbot)
    result = _run_quality_ok_path(window, qtbot, confirm_exclusions=False, confirm_quality=False)
    assert result["after_fix"] == 12
    assert result["dialog_result"] == QDialog.DialogCode.Accepted
    assert project.borehole_database.unresolved_error_count == 12
    assert window.isEnabled()
    window._project_store._dirty = False
    window.close()


def test_quality_path_b_auto_fix_confirm_exclusions_then_ok_returns(qtbot) -> None:
    window, project = _main_window_with_demo(qtbot)
    result = _run_quality_ok_path(window, qtbot, confirm_exclusions=True, confirm_quality=False)
    assert result["after_fix"] == 12
    assert result["after_exclusions"] == 0
    assert result["dialog_result"] == QDialog.DialogCode.Accepted
    assert project.borehole_database.unresolved_error_count == 0
    assert window.isEnabled()
    window._project_store._dirty = False
    window.close()


def test_quality_path_c_complete_then_ok_keeps_main_interactive_without_auto_holdout(monkeypatch, qtbot) -> None:
    window, project = _main_window_with_demo(qtbot)
    holdout_calls = []
    monkeypatch.setattr(window, "_m7_holdout", lambda: holdout_calls.append(True))
    result = _run_quality_ok_path(window, qtbot, confirm_exclusions=True, confirm_quality=True)
    assert result["after_fix"] == 12
    assert result["after_exclusions"] == 0
    assert result["dialog_result"] == QDialog.DialogCode.Accepted
    assert window._workflow.get_step("holdout").status == StepStatus.READY
    assert project.borehole_database.quality_confirmed_at is not None
    assert holdout_calls == []
    qtbot.mouseClick(window.menuBar(), Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert window.isEnabled()
    assert window.isVisible()
    window._project_store._dirty = False
    window.close()


def test_workflow_data_quality_entry_opens_quality_dialog(monkeypatch, qtbot) -> None:
    from dfn_cave_studio.ui.main_window import MainWindow
    import dfn_cave_studio.ui.dialogs.m8_quality_dialog as quality_module

    opened = {}

    class FakeQualityDialog:
        def __init__(self, project, workflow, parent):
            opened["project"] = project
            opened["workflow"] = workflow
            opened["parent"] = parent
            self.committed_changes = False

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(quality_module, "M8QualityDialog", FakeQualityDialog)
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_new_project()
    window._m7_clean()
    assert opened["project"] is window._project_store.current_project
    assert opened["workflow"] is window._workflow
    assert opened["parent"] is window
    window._project_store._dirty = False
    window.close()


def test_new_project_save_routes_to_save_as(monkeypatch, qtbot) -> None:
    from dfn_cave_studio.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window._on_new_project()
    calls = []
    monkeypatch.setattr(window, "_on_save_project_as", lambda: calls.append(True))
    window._on_save_project()
    assert calls == [True]
    window._project_store._dirty = False
    window.close()


def test_auxiliary_database_callback_preserves_dfn_state_and_removes_only_orphaned_scalar_layer(qtbot) -> None:
    from types import SimpleNamespace

    from dfn_cave_studio.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window._on_new_project()
    project = window._project_store.current_project
    field = ScalarFieldResult(
        metadata=ScalarFieldMetadata(
            field_id="ucs-field",
            parameter_name="ucs",
            unit="MPa",
            method=DensityMethod.GLOBAL_CONSTANT,
            shape=(1, 1, 1),
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            array_names=["estimate"],
            config_hash="unchanged",
        ),
        settings=DensitySettings(),
        arrays={"estimate": np.asarray([[[1.0]]], dtype=np.float32)},
    )
    project.m9_state.scalar_fields = [field]
    for step in ("explicit_dfn", "second_voxelization"):
        window._workflow.complete_step(step)

    class LayerManager:
        def __init__(self):
            self.layers = [
                SimpleNamespace(layer_id="m9_slice:scalar_rmr-field_estimate:z:0:exact"),
                SimpleNamespace(layer_id="m9_slice:p32_total:z:0"),
            ]
            self.removed = []

        def list_layers(self):
            return list(self.layers)

        def remove(self, layer_id):
            self.removed.append(layer_id)
            self.layers = [item for item in self.layers if item.layer_id != layer_id]

    manager = LayerManager()
    window._m9_layer_manager = manager
    window._on_database_changed({"rmr"})
    assert [item.metadata.field_id for item in project.m9_state.scalar_fields] == ["ucs-field"]
    assert manager.removed == ["m9_slice:scalar_rmr-field_estimate:z:0:exact"]
    assert [item.layer_id for item in manager.layers] == ["m9_slice:p32_total:z:0"]
    assert window._workflow.get_step("explicit_dfn").status == StepStatus.COMPLETED
    assert window._workflow.get_step("second_voxelization").status == StepStatus.COMPLETED
    window._project_store._dirty = False
    window.close()


def test_recent_dfnproj_uses_unified_load_and_restores_state(tmp_path: Path, monkeypatch, qtbot) -> None:
    from dfn_cave_studio.ui.main_window import MainWindow

    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "A", "collar_x": 1, "collar_y": 2, "collar_z": 3, "final_depth": 20}]),
        "collars.csv",
    )
    workflow = WorkflowController()
    workflow.complete_step("import")
    project._m7_data = {"workflow": workflow}
    bounds = ModelBounds(x_min=0, x_max=10, y_min=0, y_max=10, z_min=0, z_max=10)
    project.model_bounds = bounds
    project.voxel_config = VoxelConfig(cell_size_x=2, cell_size_y=3, cell_size_z=4)
    project.spatial_grid_config = SpatialDomainService.build_config(bounds, project.voxel_config, "manual", 0, 2, 5)
    path = tmp_path / "recent.dfnproj"
    ZipProjectStore().save(project, path)

    window = MainWindow()
    qtbot.addWidget(window)
    window._on_new_project()
    action = QAction("recent", window)
    action.setData(str(path))
    monkeypatch.setattr(window, "sender", lambda: action)
    window._on_open_recent()
    reopened = window._project_store.current_project
    assert window._project_store.current_path == path
    assert reopened.borehole_database.counts()["formal"] == 1
    assert reopened.voxel_config.cell_size_y == 3
    generation = reopened.spatial_grid_config.generation_domain
    assert generation.x_min <= bounds.x_min and generation.x_max >= bounds.x_max
    assert generation.y_min <= bounds.y_min and generation.y_max >= bounds.y_max
    assert generation.z_min <= bounds.z_min and generation.z_max >= bounds.z_max
    assert window._workflow.get_step("import").status == StepStatus.COMPLETED
    assert window._database_panel._project is reopened
    window._project_store._dirty = False
    window.close()


def test_bounds_and_voxel_workflow_confirm_independently(monkeypatch, qtbot) -> None:
    from dfn_cave_studio.ui.main_window import MainWindow
    import dfn_cave_studio.ui.dialogs.m8_spatial_grid_dialog as spatial_module

    modes = []
    analysis = ModelBounds(x_min=0, x_max=12, y_min=0, y_max=15, z_min=-5, z_max=5)
    voxel = VoxelConfig(cell_size_x=2, cell_size_y=3, cell_size_z=4)
    config = SpatialDomainService.build_config(analysis, voxel, "manual", 0, 2, 5)

    class FakeSpatialDialog:
        def __init__(self, project, plotter, mode, borehole_display_manager=None, parent=None):
            del project, plotter, borehole_display_manager, parent
            modes.append(mode)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def get_config(self):
            return config

        def get_voxel_config(self):
            return voxel

    monkeypatch.setattr(spatial_module, "M8SpatialGridDialog", FakeSpatialDialog)
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_new_project()
    window._m7_bounds()
    assert window._workflow.get_step("bounds").status == StepStatus.COMPLETED
    assert window._workflow.get_step("voxel_grid").status == StepStatus.READY
    window._m7_voxel_grid()
    assert window._workflow.get_step("bounds").status == StepStatus.COMPLETED
    assert window._workflow.get_step("voxel_grid").status == StepStatus.COMPLETED
    assert modes == ["bounds", "voxel"]
    window._project_store._dirty = False
    window.close()
