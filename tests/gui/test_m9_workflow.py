"""Real-button GUI regression tests for the M9 workflow."""

from dfn_cave_studio.models.borehole import Borehole, BoreholeCollection, Collar, FractureObservation
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController
from dfn_cave_studio.ui.dialogs.m9_dialogs import M9DensityDialog, M9ParameterFieldDialog, M9SizeDialog
from dfn_cave_studio.ui.qt_adapter import QDialogButtonBox, QMessageBox, Qt


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
    qtbot.mouseClick(size.apply_button, Qt.MouseButton.LeftButton)
    assert project.m9_state.size_models[0].source.value == "user_defined"
    qtbot.mouseClick(_ok(size), Qt.MouseButton.LeftButton)
    assert workflow.get_step("size").status == StepStatus.COMPLETED


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
