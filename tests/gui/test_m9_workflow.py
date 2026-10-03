"""Real-button GUI regression tests for the M9 workflow."""

from copy import deepcopy

import pytest

from dfn_cave_studio.dfn.size_models import assumed_size_model
from dfn_cave_studio.models.borehole import Borehole, BoreholeCollection, Collar, FractureObservation
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.m9 import SizeModelSource
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.services.m9_service import M9Service
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.ui.dialogs.m9_dialogs import M9DensityDialog, M9ParameterFieldDialog, M9SizeDialog
from dfn_cave_studio.ui.qt_adapter import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QSettings,
    Qt,
    QThreadPool,
)


def _project() -> Project:
    project = Project()
    project.borehole_collection = BoreholeCollection(
        boreholes=[
            Borehole(
                borehole_id="CAL",
                collar=Collar(borehole_id="CAL", collar_x=0, collar_y=0, collar_z=10, dip=-90, final_depth=10),
                fracture_observations=[
                    FractureObservation(borehole_id="CAL", measured_depth=2, dip_direction=0, dip=0, set_id=1)
                ],
            ),
            Borehole(
                borehole_id="VAL",
                collar=Collar(borehole_id="VAL", collar_x=1, collar_y=0, collar_z=10, dip=-90, final_depth=10),
                fracture_observations=[
                    FractureObservation(borehole_id="VAL", measured_depth=3, dip_direction=0, dip=0, set_id=1)
                ],
            ),
        ]
    )
    holdout = HoldoutService()
    holdout.select_manual(["CAL", "VAL"], ["VAL"])
    holdout.lock()
    set_holdout(project, holdout)
    project.joint_sets = [
        JointSetConfig(set_id=1, orientation=OrientationDistribution(mean_dip_direction=0, mean_dip=0, kappa=1000))
    ]
    bounds = ModelBounds(x_min=-1, x_max=2, y_min=-1, y_max=1, z_min=-1, z_max=11)
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=bounds, generation_domain=bounds)
    project.voxel_config = VoxelConfig(cell_size_x=1, cell_size_y=1, cell_size_z=1)
    return project


def _ok(dialog):
    return dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)


def _project_with_density() -> Project:
    project = _project()
    M9Service(project).calculate_density()
    return project


@pytest.fixture
def budget_settings(tmp_path):
    path = tmp_path / "preferences.ini"
    settings = QSettings(str(path), QSettings.Format.IniFormat)
    settings.clear()
    settings.sync()
    yield path, settings
    settings.clear()
    settings.sync()


def _apply_size_dialog(qtbot, dialog: M9SizeDialog, distribution: str, lower: float, upper: float) -> None:
    dialog.distribution.setCurrentIndex(dialog.distribution.findData(distribution))
    dialog.lower.setValue(lower)
    dialog.upper.setValue(upper)
    dialog.manual_source.setCurrentIndex(dialog.manual_source.findData("user_defined"))
    dialog.confirm_size.setChecked(True)
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)


def test_density_and_size_buttons_commit_real_m9_state(qtbot):
    project, workflow = _project(), WorkflowController()
    density = M9DensityDialog(project, workflow)
    qtbot.addWidget(density)
    density.show()
    qtbot.mouseClick(density.calculate_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: density._worker is None, timeout=5000)
    assert len(project.m9_state.p10_intervals) == 2
    assert {row.role for row in project.m9_state.p10_intervals} == {"calibration", "validation"}
    qtbot.mouseClick(_ok(density), Qt.MouseButton.LeftButton)
    assert workflow.get_step("density").status == StepStatus.COMPLETED

    size = M9SizeDialog(project, workflow)
    qtbot.addWidget(size)
    size.show()
    size.confirm_size.setChecked(True)
    qtbot.mouseClick(size.apply_button, Qt.MouseButton.LeftButton)
    assert project.m9_state.size_models[0].source.value == "manual_fixed"
    qtbot.mouseClick(_ok(size), Qt.MouseButton.LeftButton)
    assert workflow.get_step("size").status == StepStatus.COMPLETED


def test_size_dialog_does_not_commit_default_one_metre_without_confirmation(qtbot, monkeypatch):
    project, workflow = _project_with_density(), WorkflowController()
    dialog = M9SizeDialog(project, workflow)
    qtbot.addWidget(dialog)
    warnings = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args: warnings.append(_args[2]))

    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)

    assert warnings == ["Confirm that the circular-disc radius model and metre unit are intentional"]
    assert project.m9_state.size_models == []


@pytest.mark.parametrize("distribution", ["truncated_lognormal", "uniform"])
def test_size_dialog_restores_saved_nonfixed_model(qtbot, distribution):
    project, workflow = _project_with_density(), WorkflowController()
    M9Service(project).set_assumed_sizes(distribution, {"mu": 1.0, "sigma": 0.5} if distribution == "truncated_lognormal" else {}, 1.0, 10.0)

    dialog = M9SizeDialog(project, workflow)
    qtbot.addWidget(dialog)

    assert dialog.distribution.currentData() == distribution
    assert dialog.lower.value() == pytest.approx(1.0)
    assert dialog.upper.value() == pytest.approx(10.0)
    assert dialog.manual_source.currentData() == "user_defined"
    assert not dialog.confirm_size.isChecked()
    assert not dialog.committed_changes


def test_size_dialog_restores_fixed_radius_and_assumed_source(qtbot):
    project, workflow = _project_with_density(), WorkflowController()
    M9Service(project).set_assumed_sizes("fixed", {"radius": 2.5}, 2.5, 9.0, user_defined=False)

    dialog = M9SizeDialog(project, workflow)
    qtbot.addWidget(dialog)

    assert dialog.distribution.currentData() == "fixed"
    assert dialog.lower.value() == pytest.approx(2.5)
    assert dialog.upper.value() == pytest.approx(2.5)
    assert dialog.manual_source.currentData() == "assumed"
    assert not dialog.confirm_size.isChecked()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SizeModelSource.USER_DEFINED, "user_defined"),
        (SizeModelSource.MANUAL_FIXED, "user_defined"),
        (SizeModelSource.MANUAL_DISTRIBUTION, "user_defined"),
        (SizeModelSource.ASSUMED, "assumed"),
        (SizeModelSource.ASSUMED_SCENARIO, "assumed"),
    ],
)
def test_size_dialog_restores_legacy_and_current_manual_source(qtbot, source, expected):
    project = _project_with_density()
    M9Service(project).set_assumed_sizes("uniform", {}, 1.0, 10.0)
    project.m9_state.size_models[0].source = source
    dialog = M9SizeDialog(project, WorkflowController())
    qtbot.addWidget(dialog)
    assert dialog.manual_source.currentData() == expected


def test_size_dialog_open_cancel_and_no_change_ok_are_transactionally_inert(qtbot):
    project, workflow = _project_with_density(), WorkflowController()
    M9Service(project).set_assumed_sizes("uniform", {}, 1.0, 10.0)
    models_before = deepcopy(project.m9_state.size_models)
    workflow_before = workflow.to_dict()

    cancelled = M9SizeDialog(project, workflow)
    qtbot.addWidget(cancelled)
    cancelled.reject()
    assert project.m9_state.size_models == models_before
    assert workflow.to_dict() == workflow_before

    accepted = M9SizeDialog(project, workflow)
    qtbot.addWidget(accepted)
    accepted.accept()
    assert not accepted.committed_changes
    assert project.m9_state.size_models == models_before
    assert workflow.to_dict() == workflow_before


def test_size_dialog_reapplying_identical_values_is_not_a_change(qtbot):
    project, workflow = _project_with_density(), WorkflowController()
    M9Service(project).set_assumed_sizes("uniform", {}, 1.0, 10.0)
    models_before = deepcopy(project.m9_state.size_models)
    dialog = M9SizeDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.confirm_size.setChecked(True)

    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)

    assert not dialog.committed_changes
    assert project.m9_state.size_models == models_before


def test_size_dialog_restores_after_project_save_reopen(qtbot, tmp_path):
    project = _project_with_density()
    M9Service(project).set_assumed_sizes(
        "truncated_lognormal", {"mu": 1.0, "sigma": 0.5}, 1.0, 10.0
    )
    path = tmp_path / "synthetic-size-dialog.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)

    dialog = M9SizeDialog(restored, WorkflowController())
    qtbot.addWidget(dialog)
    assert dialog.distribution.currentData() == "truncated_lognormal"
    assert dialog.lower.value() == pytest.approx(1.0)
    assert dialog.upper.value() == pytest.approx(10.0)
    assert dialog.manual_source.currentData() == "user_defined"


def test_size_dialog_real_apply_ok_reopen_and_persist_lifecycle(qtbot, tmp_path):
    project, workflow = _project_with_density(), WorkflowController()
    dialog = M9SizeDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.show()
    _apply_size_dialog(qtbot, dialog, "truncated_lognormal", 1.0, 10.0)
    assert "Pending size model: 1–10 m" in dialog.pending_status.text()

    qtbot.mouseClick(dialog.button_box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert project.m9_state.size_models[0].min_radius == pytest.approx(1.0)
    assert project.m9_state.size_models[0].max_radius == pytest.approx(10.0)
    assert workflow.get_step("size").status == StepStatus.COMPLETED

    reopened_dialog = M9SizeDialog(project, workflow)
    qtbot.addWidget(reopened_dialog)
    assert reopened_dialog.distribution.currentData() == "truncated_lognormal"
    assert reopened_dialog.lower.value() == pytest.approx(1.0)
    assert reopened_dialog.upper.value() == pytest.approx(10.0)

    path = tmp_path / "synthetic-real-dialog-lifecycle.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)
    persisted_dialog = M9SizeDialog(restored, WorkflowController())
    qtbot.addWidget(persisted_dialog)
    assert persisted_dialog.distribution.currentData() == "truncated_lognormal"
    assert persisted_dialog.lower.value() == pytest.approx(1.0)
    assert persisted_dialog.upper.value() == pytest.approx(10.0)


@pytest.mark.parametrize("close_mode", ["cancel", "window_x", "escape"])
def test_size_dialog_all_cancel_paths_restore_entry_models(qtbot, close_mode):
    project, workflow = _project_with_density(), WorkflowController()
    M9Service(project).set_assumed_sizes("uniform", {}, 2.0, 8.0)
    models_before = deepcopy(project.m9_state.size_models)
    workflow_before = workflow.to_dict()
    dialog = M9SizeDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.show()
    _apply_size_dialog(qtbot, dialog, "truncated_lognormal", 1.0, 10.0)
    assert dialog.committed_changes

    if close_mode == "cancel":
        qtbot.mouseClick(dialog.button_box.button(QDialogButtonBox.StandardButton.Cancel), Qt.MouseButton.LeftButton)
    elif close_mode == "window_x":
        dialog.close()
    else:
        qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    QApplication.processEvents()

    assert dialog.result() == QDialog.DialogCode.Rejected
    assert project.m9_state.size_models == models_before
    assert workflow.to_dict() == workflow_before


def test_size_dialog_small_window_keeps_buttons_visible_and_details_scrollable(qtbot):
    project = _project()
    project.m9_state.size_models = [
        assumed_size_model(
            domain_id=index,
            set_id=index,
            distribution_type="uniform",
            parameters={},
            lower=1.0,
            upper=10.0,
        )
        for index in range(1, 21)
    ]
    dialog = M9SizeDialog(project, WorkflowController())
    qtbot.addWidget(dialog)
    dialog.resize(520, 360)
    dialog.show()
    QApplication.processEvents()

    assert dialog.button_box.isVisible()
    assert dialog.button_box.geometry().bottom() <= dialog.contentsRect().bottom()
    assert dialog.scroll_area.geometry().bottom() <= dialog.button_box.geometry().top()
    assert dialog.details.verticalScrollBar().maximum() > 0


def test_size_dialog_empty_models_keep_new_model_defaults(qtbot):
    dialog = M9SizeDialog(_project(), WorkflowController())
    qtbot.addWidget(dialog)
    assert dialog.lower.value() == pytest.approx(1.0)
    assert dialog.upper.value() == pytest.approx(5.0)
    assert dialog.apply_button.isEnabled()
    assert "new-model defaults" in dialog.editor_status.text()


def test_size_dialog_does_not_misrepresent_heterogeneous_models(qtbot):
    project = _project()
    project.m9_state.size_models = [
        assumed_size_model(
            domain_id=1, set_id=1, distribution_type="uniform", parameters={}, lower=1.0, upper=10.0
        ),
        assumed_size_model(
            domain_id=2, set_id=1, distribution_type="uniform", parameters={}, lower=2.0, upper=8.0
        ),
    ]
    dialog = M9SizeDialog(project, WorkflowController())
    qtbot.addWidget(dialog)

    assert not dialog.apply_button.isEnabled()
    assert "Multiple existing size models" in dialog.summary.text()
    assert "not project values" in dialog.editor_status.text()


def test_parameter_field_button_runs_in_worker_and_can_close(qtbot):
    project, workflow = _project(), WorkflowController()
    from dfn_cave_studio.services.m9_service import M9Service

    service = M9Service(project)
    service.calculate_density()
    service.set_assumed_sizes("fixed", {"radius": 1.0}, 1.0, 1.000001)
    dialog = M9ParameterFieldDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.mouseClick(dialog.generate_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=5000)
    assert project.m9_state.parameter_field_metadata is not None
    assert dialog.field.count() > 0
    qtbot.mouseClick(_ok(dialog), Qt.MouseButton.LeftButton)
    assert workflow.get_step("parameter_field").status == StepStatus.COMPLETED


def test_parameter_field_memory_budget_defaults_to_two_gib(qtbot, budget_settings):
    _path, settings = budget_settings
    dialog = M9ParameterFieldDialog(_project(), WorkflowController(), settings=settings)
    qtbot.addWidget(dialog)
    assert dialog.memory_budget_gib.value() == pytest.approx(2.0)


def test_parameter_field_memory_budget_persists_after_edit_and_cancel(qtbot, budget_settings):
    path, settings = budget_settings
    project, workflow = _project(), WorkflowController()
    state_before = deepcopy(project.m9_state)
    workflow_before = workflow.to_dict()
    dialog = M9ParameterFieldDialog(project, workflow, settings=settings)
    qtbot.addWidget(dialog)
    dialog.memory_budget_gib.setValue(12.5)
    dialog.memory_budget_gib.editingFinished.emit()
    dialog.reject()

    restarted_settings = QSettings(str(path), QSettings.Format.IniFormat)
    reopened = M9ParameterFieldDialog(project, workflow, settings=restarted_settings)
    qtbot.addWidget(reopened)
    assert reopened.memory_budget_gib.value() == pytest.approx(12.5)
    assert project.m9_state == state_before
    assert workflow.to_dict() == workflow_before
    assert not reopened.committed_changes


@pytest.mark.parametrize(
    ("stored", "expected"),
    [("not-a-number", 2.0), ("nan", 2.0), ("inf", 2.0), (-10.0, 0.25), (900.0, 512.0)],
)
def test_parameter_field_memory_budget_validates_stored_values(qtbot, budget_settings, stored, expected):
    _path, settings = budget_settings
    settings.setValue(M9ParameterFieldDialog.MEMORY_BUDGET_KEY, stored)
    settings.sync()
    dialog = M9ParameterFieldDialog(_project(), WorkflowController(), settings=settings)
    qtbot.addWidget(dialog)
    assert dialog.memory_budget_gib.value() == pytest.approx(expected)


def test_parameter_field_generate_uses_restored_memory_budget(qtbot, budget_settings, monkeypatch):
    _path, settings = budget_settings
    settings.setValue(M9ParameterFieldDialog.MEMORY_BUDGET_KEY, 12.5)
    settings.sync()
    from dfn_cave_studio.voxel.parameter_field import ParameterFieldBuilder

    original = ParameterFieldBuilder.estimate_resources
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(ParameterFieldBuilder, "estimate_resources", staticmethod(capture))
    monkeypatch.setattr(QThreadPool, "start", lambda *_args: None)
    dialog = M9ParameterFieldDialog(_project(), WorkflowController(), settings=settings)
    qtbot.addWidget(dialog)
    calls.clear()
    dialog._generate()

    assert calls
    assert calls[-1]["budget_bytes"] == int(12.5 * 1024**3)
    assert float(settings.value(M9ParameterFieldDialog.MEMORY_BUDGET_KEY)) == pytest.approx(12.5)


def test_parameter_field_high_saved_budget_keeps_system_memory_guard(qtbot, budget_settings, monkeypatch):
    _path, settings = budget_settings
    settings.setValue(M9ParameterFieldDialog.MEMORY_BUDGET_KEY, 512.0)
    settings.sync()
    monkeypatch.setattr(
        "dfn_cave_studio.ui.dialogs.m9_dialogs.available_system_memory_bytes",
        lambda: 1024,
    )
    dialog = M9ParameterFieldDialog(_project(), WorkflowController(), settings=settings)
    qtbot.addWidget(dialog)

    assert dialog.memory_budget_gib.value() == pytest.approx(512.0)
    assert dialog._last_resource_estimate.system_available_bytes == 1024
    assert dialog._last_resource_estimate.exceeds_budget


def test_parameter_field_ok_saves_budget_without_project_side_effects(qtbot, budget_settings):
    _path, settings = budget_settings
    project, workflow = _project(), WorkflowController()
    state_before = deepcopy(project.m9_state)
    workflow_before = workflow.to_dict()
    dialog = M9ParameterFieldDialog(project, workflow, settings=settings)
    qtbot.addWidget(dialog)
    dialog.memory_budget_gib.setValue(7.25)
    dialog.accept()

    assert float(settings.value(M9ParameterFieldDialog.MEMORY_BUDGET_KEY)) == pytest.approx(7.25)
    assert project.m9_state == state_before
    assert workflow.to_dict() == workflow_before
    assert not dialog.committed_changes


def test_parameter_field_over_budget_requires_confirmation_before_worker(
    qtbot, monkeypatch
):
    project, workflow = _project(), WorkflowController()
    project.spatial_grid_config = SpatialGridConfig(
        analysis_domain=ModelBounds(x_min=0, x_max=200, y_min=0, y_max=200, z_min=0, z_max=200),
        generation_domain=ModelBounds(x_min=0, x_max=200, y_min=0, y_max=200, z_min=0, z_max=200),
    )
    dialog = M9ParameterFieldDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.memory_budget_gib.setValue(0.25)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: warnings.append(_args) or QMessageBox.StandardButton.No)
    dialog._generate()
    assert warnings
    assert dialog._worker is None
    assert "8,000,000 voxels" in dialog.resource_summary.text()
    assert dialog.resource_summary.text().count("\n") == 3
    assert "cell_state" not in dialog.resource_summary.text()
    assert "cell_state" not in warnings[0][2]
    assert len(warnings[0][2].splitlines()) <= 7


def test_parameter_field_compact_resource_summary_keeps_scrollable_details(qtbot):
    project, workflow = _project(), WorkflowController()
    dialog = M9ParameterFieldDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.resize(640, 480)
    dialog.show()
    dialog._update_resource_summary()

    assert dialog.resource_summary.text().count("\n") == 3
    assert "Shape:" in dialog.resource_summary.text()
    assert "Persistent arrays:" in dialog.resource_summary.text()
    assert "Temporary buffers:" in dialog.resource_summary.text()
    assert dialog.resource_details_button.isVisible()
    assert dialog.resource_details_button.text() == "Details..."
    assert dialog._last_resource_estimate is not None


def test_parameter_field_running_disables_ok_and_reject_discards_late_result(qtbot):
    project, workflow = _project(), WorkflowController()
    dialog = M9ParameterFieldDialog(project, workflow)
    qtbot.addWidget(dialog)

    class _LateWorker:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    worker = _LateWorker()
    dialog._worker = worker
    dialog._set_generation_running(True)
    ok = dialog.dialog_buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok.isEnabled() is False
    dialog.accept()
    assert dialog.result() == 0

    dialog.reject()
    dialog._done(object(), worker)
    assert worker.cancelled
    assert project.m9_state.parameter_field_metadata is None
    assert project.m9_state.parameter_field_arrays == {}


def test_parameter_field_window_close_discards_late_worker_result(qtbot):
    project, workflow = _project(), WorkflowController()
    dialog = M9ParameterFieldDialog(project, workflow)
    qtbot.addWidget(dialog)

    class _LateWorker:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    worker = _LateWorker()
    dialog._worker = worker
    dialog._set_generation_running(True)
    dialog.show()
    dialog.close()
    dialog._done(object(), worker)

    assert worker.cancelled
    assert project.m9_state.parameter_field_metadata is None
    assert project.m9_state.parameter_field_arrays == {}
    assert workflow.get_step("parameter_field").status == StepStatus.NOT_STARTED


def test_cancelled_parameter_worker_cannot_commit_late_result(qtbot):
    project, workflow = _project(), WorkflowController()
    dialog = M9ParameterFieldDialog(project, workflow)
    qtbot.addWidget(dialog)

    class _LateWorker:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    worker = _LateWorker()
    dialog._worker = worker
    dialog._cancel()
    dialog._done(object(), worker)
    assert worker.cancelled
    assert project.m9_state.parameter_field_metadata is None
    assert project.m9_state.parameter_field_arrays == {}


def test_cancel_restores_project_and_workflow(qtbot):
    project, workflow = _project(), WorkflowController()
    dialog = M9DensityDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.mouseClick(dialog.calculate_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=5000)
    assert project.m9_state.p10_intervals
    cancel = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Cancel)
    qtbot.mouseClick(cancel, Qt.MouseButton.LeftButton)
    assert project.m9_state.p10_intervals == []
    assert workflow.get_step("density").status == StepStatus.NOT_STARTED
