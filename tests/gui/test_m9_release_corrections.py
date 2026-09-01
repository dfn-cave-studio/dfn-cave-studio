"""GUI lifecycle regressions for M9 settings and spatial invalidation."""

from copy import deepcopy

import numpy as np

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.m9 import (
    DensityMethod,
    DensitySettings,
    P10Interval,
    P32Estimate,
    ParameterFieldMetadata,
    SizeModel,
    ValidationIntervalResult,
    ValidationState,
    ValidationSummary,
)
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.m7_state import get_workflow, set_workflow
from dfn_cave_studio.services.spatial_domain_service import SpatialDomainService
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController
from dfn_cave_studio.ui.dialogs.m9_dialogs import M9DensityDialog
from dfn_cave_studio.ui.qt_adapter import QDialog, QDialogButtonBox, Qt


def _completed_project() -> Project:
    project = Project()
    bounds = ModelBounds(x_min=0, x_max=20, y_min=0, y_max=20, z_min=0, z_max=20)
    voxel = VoxelConfig(cell_size_x=5, cell_size_y=5, cell_size_z=5)
    project.model_bounds = bounds
    project.voxel_config = voxel
    project.spatial_grid_config = SpatialDomainService.build_config(bounds, voxel, "manual", 0, 2, 5)
    project.m9_state.density_settings = DensitySettings(
        interval_mode="domain",
        interval_length=25,
        method=DensityMethod.IDW,
        power=3.5,
        search_radius=123.0,
        min_neighbors=2,
        max_neighbors=9,
        anisotropy_x=1.5,
        anisotropy_y=2.5,
        anisotropy_z=3.5,
        global_fallback=True,
        monte_carlo_samples=321,
        low_observability_threshold=0.12,
    )
    project.m9_state.random_seed = 77
    project.m9_state.p10_intervals = [
        P10Interval(
            hole_id="BH-1",
            from_depth=0,
            to_depth=10,
            domain_id=1,
            set_id=1,
            observation_count=1,
            sample_length=10,
            p10=0.1,
        )
    ]
    project.m9_state.p32_estimates = [
        P32Estimate(
            domain_id=1,
            set_id=1,
            fracture_count=1,
            raw_sample_length=10,
            effective_sample_length=5,
            mean_exposure=0.5,
            p32=0.2,
            random_seed=77,
        )
    ]
    project.m9_state.size_models = [SizeModel(domain_id=1, set_id=1)]
    project.m9_state.parameter_field_metadata = ParameterFieldMetadata(
        shape=(4, 4, 4),
        origin=(0, 0, 0),
        spacing=(5, 5, 5),
        field_names=["p32_total"],
        set_ids=[1],
        density_method=DensityMethod.IDW,
        random_seed=77,
        estimated_bytes=256,
    )
    project.m9_state.parameter_field_arrays = {"p32_total": np.ones((4, 4, 4), dtype=np.float32)}
    project.m9_state.validation_results = [
        ValidationIntervalResult(
            hole_id="BH-V",
            from_depth=0,
            to_depth=10,
            domain_id=1,
            set_id=1,
            observed_count=1,
            observed_p10=0.1,
            predicted_p10=0.1,
            predicted_p32=0.2,
            absolute_error=0,
        )
    ]
    project.m9_state.validation_summary = ValidationSummary(
        state=ValidationState.COMPLETE,
        mae=0,
        rmse=0,
        bias=0,
        valid_interval_count=1,
    )
    workflow = WorkflowController()
    for step in workflow.get_steps():
        workflow.complete_step(step.step_id)
    set_workflow(project, workflow)
    return project


def _state_without_arrays(project: Project) -> dict:
    return project.m9_state.model_dump(mode="json", exclude={"parameter_field_arrays"})


def _install_spatial_dialog(monkeypatch, config, voxel, result=QDialog.DialogCode.Accepted) -> None:
    import dfn_cave_studio.ui.dialogs.m8_spatial_grid_dialog as spatial_module

    class FakeSpatialDialog:
        def __init__(self, project, plotter, mode, parent):
            del project, plotter, mode, parent

        def exec(self):
            return result

        def get_config(self):
            return config

        def get_voxel_config(self):
            return voxel

    monkeypatch.setattr(spatial_module, "M8SpatialGridDialog", FakeSpatialDialog)


def _install_real_ok_spatial_dialog(monkeypatch, qtbot) -> None:
    """Run the production dialog's real OK button without a nested test event loop."""
    import dfn_cave_studio.ui.dialogs.m8_spatial_grid_dialog as spatial_module

    real_dialog = spatial_module.M8SpatialGridDialog

    class ClickOkSpatialDialog(real_dialog):
        def exec(self):
            self.show()
            ok = self.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
            qtbot.mouseClick(ok, Qt.MouseButton.LeftButton)
            return self.result()

    monkeypatch.setattr(spatial_module, "M8SpatialGridDialog", ClickOkSpatialDialog)


def _window_with_project(project: Project, qtbot):
    from dfn_cave_studio.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window._project_store.adopt_project(project)
    window._workflow = get_workflow(project)
    window._project_store._dirty = False
    return window


def test_reopened_project_voxel_ok_without_changes_preserves_all_m9(
    monkeypatch, qtbot, tmp_path
) -> None:
    project = _completed_project()
    path = tmp_path / "complete-m9.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)
    before = _state_without_arrays(reopened)
    arrays_before = {name: array.copy() for name, array in reopened.m9_state.parameter_field_arrays.items()}
    workflow_before = get_workflow(reopened).to_dict()
    _install_real_ok_spatial_dialog(monkeypatch, qtbot)
    window = _window_with_project(reopened, qtbot)

    window._m7_voxel_grid()

    assert _state_without_arrays(reopened) == before
    assert get_workflow(reopened).to_dict() == workflow_before
    assert window._project_store.is_dirty is False
    for name, expected in arrays_before.items():
        np.testing.assert_array_equal(reopened.m9_state.parameter_field_arrays[name], expected)


def test_preview_only_change_does_not_invalidate_m9(monkeypatch, qtbot) -> None:
    project = _completed_project()
    config = project.spatial_grid_config.model_copy(
        deep=True,
        update={"preview_opacity": 0.75, "preview_slice_axis": "x", "preview_slice_fraction": 0.25},
    )
    before = _state_without_arrays(project)
    _install_spatial_dialog(monkeypatch, config, project.voxel_config.model_copy())
    window = _window_with_project(project, qtbot)

    window._m7_voxel_grid()

    assert _state_without_arrays(project) == before
    assert window._workflow.get_step("parameter_field").status == StepStatus.COMPLETED
    assert window._workflow.get_step("validation").status == StepStatus.COMPLETED
    assert window._project_store.is_dirty is True


def test_generation_domain_only_change_does_not_invalidate_m9(monkeypatch, qtbot) -> None:
    project = _completed_project()
    config = SpatialDomainService.build_config(project.model_bounds, project.voxel_config, "manual", 0, 5, 30)
    before = _state_without_arrays(project)
    _install_spatial_dialog(monkeypatch, config, project.voxel_config.model_copy())
    window = _window_with_project(project, qtbot)

    window._m7_voxel_grid()

    assert _state_without_arrays(project) == before
    assert window._workflow.get_step("density").status == StepStatus.COMPLETED
    assert window._workflow.get_step("parameter_field").status == StepStatus.COMPLETED
    assert window._workflow.get_step("validation").status == StepStatus.COMPLETED


def test_analysis_domain_change_stales_only_spatial_m9_results(monkeypatch, qtbot) -> None:
    project = _completed_project()
    bounds = ModelBounds(x_min=0, x_max=25, y_min=0, y_max=20, z_min=0, z_max=20)
    config = SpatialDomainService.build_config(bounds, project.voxel_config, "manual", 0, 2, 5)
    before = _state_without_arrays(project)
    _install_spatial_dialog(monkeypatch, config, project.voxel_config.model_copy())
    window = _window_with_project(project, qtbot)

    window._m7_bounds()

    assert _state_without_arrays(project) == before
    assert window._workflow.get_step("density").status == StepStatus.COMPLETED
    assert window._workflow.get_step("size").status == StepStatus.COMPLETED
    assert window._workflow.get_step("parameter_field").status == StepStatus.STALE
    assert window._workflow.get_step("validation").status == StepStatus.STALE
    assert window._workflow.get_step("voxel_grid").status == StepStatus.READY


def test_voxel_spacing_change_stales_only_spatial_m9_steps(monkeypatch, qtbot, tmp_path) -> None:
    project = _completed_project()
    voxel = VoxelConfig(cell_size_x=2.5, cell_size_y=5, cell_size_z=5)
    config = SpatialDomainService.build_config(project.model_bounds, voxel, "manual", 0, 2, 5)
    before = _state_without_arrays(project)
    arrays_before = {name: array.copy() for name, array in project.m9_state.parameter_field_arrays.items()}
    _install_spatial_dialog(monkeypatch, config, voxel)
    window = _window_with_project(project, qtbot)

    window._m7_voxel_grid()

    assert window._workflow.get_step("density").status == StepStatus.COMPLETED
    assert window._workflow.get_step("size").status == StepStatus.COMPLETED
    assert window._workflow.get_step("parameter_field").status == StepStatus.STALE
    assert window._workflow.get_step("validation").status == StepStatus.STALE
    assert _state_without_arrays(project) == before
    for name, expected in arrays_before.items():
        np.testing.assert_array_equal(project.m9_state.parameter_field_arrays[name], expected)

    path = tmp_path / "stale-spatial-results.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)
    assert get_workflow(reopened).get_step("density").status == StepStatus.COMPLETED
    assert get_workflow(reopened).get_step("size").status == StepStatus.COMPLETED
    assert get_workflow(reopened).get_step("parameter_field").status == StepStatus.STALE
    assert get_workflow(reopened).get_step("validation").status == StepStatus.STALE
    assert _state_without_arrays(reopened) == before
    for name, expected in arrays_before.items():
        np.testing.assert_array_equal(reopened.m9_state.parameter_field_arrays[name], expected)


def test_voxel_dialog_cancel_rolls_back_project_workflow_dirty_and_m9(monkeypatch, qtbot) -> None:
    project = _completed_project()
    before = project.model_dump(mode="json", exclude={"m9_state": {"parameter_field_arrays"}})
    workflow_before = get_workflow(project).to_dict()
    changed_voxel = VoxelConfig(cell_size_x=1, cell_size_y=1, cell_size_z=1)
    changed_config = SpatialDomainService.build_config(project.model_bounds, changed_voxel, "manual", 0, 2, 5)
    _install_spatial_dialog(monkeypatch, changed_config, changed_voxel, QDialog.DialogCode.Rejected)
    window = _window_with_project(project, qtbot)

    window._m7_voxel_grid()

    assert project.model_dump(mode="json", exclude={"m9_state": {"parameter_field_arrays"}}) == before
    assert get_workflow(project).to_dict() == workflow_before
    assert window._project_store.is_dirty is False


def test_density_dialog_restores_all_saved_controls_and_noop_ok(qtbot) -> None:
    project = _completed_project()
    workflow = get_workflow(project)
    before = deepcopy(project.m9_state)
    workflow_before = workflow.to_dict()
    dialog = M9DensityDialog(project, workflow)
    qtbot.addWidget(dialog)

    assert dialog.interval_mode.currentData() == "domain"
    assert dialog.method.currentData() == DensityMethod.IDW.value
    assert dialog.interval_length.value() == 25
    assert dialog.power.value() == 3.5
    assert dialog.radius.value() == 123
    assert dialog.min_neighbors.value() == 2
    assert dialog.max_neighbors.value() == 9
    assert (dialog.anisotropy_x.value(), dialog.anisotropy_y.value(), dialog.anisotropy_z.value()) == (1.5, 2.5, 3.5)
    assert dialog.global_fallback.isChecked()
    assert dialog.seed.value() == 77
    assert dialog.monte_carlo_samples.value() == 321
    assert dialog.low_observability_threshold.value() == 0.12

    ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
    qtbot.mouseClick(ok, Qt.MouseButton.LeftButton)

    assert dialog.committed_changes is False
    assert project.m9_state.density_settings == before.density_settings
    assert project.m9_state.p10_intervals == before.p10_intervals
    assert workflow.to_dict() == workflow_before
