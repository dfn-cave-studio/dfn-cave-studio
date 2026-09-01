"""Real-button GUI regressions for M9 rendered slice layer management."""

from copy import deepcopy

import numpy as np

from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.m7_state import set_workflow
from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.ui.dialogs.m9_dialogs import M9ParameterFieldDialog
from dfn_cave_studio.ui.main_window import MainWindow
from dfn_cave_studio.ui.qt_adapter import Qt


def _completed_field_project() -> tuple[Project, WorkflowController]:
    project = Project()
    names = ["p32_total", *[f"field_{index:02d}" for index in range(1, 48)]]
    project.m9_state.parameter_field_metadata = ParameterFieldMetadata(
        shape=(4, 4, 4),
        origin=(100.0, 200.0, 300.0),
        spacing=(2.0, 3.0, 4.0),
        field_names=names,
        set_ids=[1],
        density_method=DensityMethod.IDW,
        random_seed=42,
        estimated_bytes=48 * 4 * 4 * 4 * 4,
    )
    project.m9_state.parameter_field_arrays = {
        name: np.full((4, 4, 4), index, dtype=np.float32) for index, name in enumerate(names)
    }
    workflow = WorkflowController()
    for step_id in ("density", "size", "parameter_field", "validation"):
        workflow.complete_step(step_id)
    set_workflow(project, workflow)
    return project, workflow


def _window(project: Project, workflow: WorkflowController, qtbot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window._project_store.adopt_project(project)
    window._workflow = workflow
    return window


def _install_fake_slice_render(monkeypatch) -> None:
    from dfn_cave_studio.visualization.parameter_field_renderer import ParameterFieldRenderer

    def render_slice(
        self,
        plotter,
        metadata,
        arrays,
        field_name,
        axis,
        fraction,
        *,
        opacity=1.0,
        cmap="viridis",
        actor_name=None,
        show_scalar_bar=True,
    ):
        del self, metadata, arrays, field_name, axis, fraction, cmap, show_scalar_bar
        return plotter.add_mesh(object(), name=actor_name, opacity=opacity)

    monkeypatch.setattr(ParameterFieldRenderer, "render_slice", render_slice)


def _m9_actor_count(window: MainWindow) -> int:
    return sum(actor.name.startswith("m9_slice:") for actor in window._plotter._actors)


def test_render_replace_and_all_layer_buttons_preserve_science(monkeypatch, qtbot) -> None:
    _install_fake_slice_render(monkeypatch)
    project, workflow = _completed_field_project()
    state_before = project.m9_state.model_dump(mode="python", exclude={"parameter_field_arrays"})
    arrays_before = {name: value.copy() for name, value in project.m9_state.parameter_field_arrays.items()}
    workflow_before = deepcopy(workflow.to_dict())
    window = _window(project, workflow, qtbot)
    for name in ("BH-01", "voxel_analysis_domain", "voxel_grid_preview", "coordinate_axes"):
        window._plotter.add_mesh(object(), name=name)
    dialog = M9ParameterFieldDialog(project, workflow, window)
    qtbot.addWidget(dialog)
    dialog.show()

    for _ in range(10):
        qtbot.mouseClick(dialog.render_button, Qt.MouseButton.LeftButton)
    assert _m9_actor_count(window) == 1
    assert len(window._get_m9_layer_manager().list_layers()) == 1
    assert dialog.layers_table.rowCount() == 1

    dialog.slice_fraction.setValue(0.0)
    qtbot.mouseClick(dialog.render_button, Qt.MouseButton.LeftButton)
    assert _m9_actor_count(window) == 2
    dialog.field.setCurrentIndex(1)
    qtbot.mouseClick(dialog.render_button, Qt.MouseButton.LeftButton)
    assert _m9_actor_count(window) == 3
    assert dialog.layers_table.rowCount() == 3

    selected_id = dialog._current_layer_id()
    record = window._get_m9_layer_manager().get(selected_id)
    assert record is not None
    qtbot.mouseClick(dialog.toggle_layer_button, Qt.MouseButton.LeftButton)
    assert record.visible is False
    qtbot.mouseClick(dialog.toggle_layer_button, Qt.MouseButton.LeftButton)
    assert record.visible is True
    dialog.opacity.setValue(0.35)
    assert record.opacity == 0.35
    assert record.actor.GetProperty().opacity == 0.35

    qtbot.mouseClick(dialog.clear_current_button, Qt.MouseButton.LeftButton)
    assert not window._get_m9_layer_manager().contains(selected_id)
    assert _m9_actor_count(window) == 2
    dialog.layers_table.selectRow(0)
    removed_id = dialog._selected_layer_id()
    qtbot.mouseClick(dialog.remove_layer_button, Qt.MouseButton.LeftButton)
    assert not window._get_m9_layer_manager().contains(removed_id)
    assert _m9_actor_count(window) == 1
    qtbot.mouseClick(dialog.clear_all_button, Qt.MouseButton.LeftButton)
    assert _m9_actor_count(window) == 0
    assert window._get_m9_layer_manager().list_layers() == []
    assert {actor.name for actor in window._plotter._actors} == {
        "BH-01",
        "voxel_analysis_domain",
        "voxel_grid_preview",
        "coordinate_axes",
    }

    assert project.m9_state.model_dump(mode="python", exclude={"parameter_field_arrays"}) == state_before
    assert workflow.to_dict() == workflow_before
    assert len(project.m9_state.parameter_field_arrays) == 48
    for name, expected in arrays_before.items():
        np.testing.assert_array_equal(project.m9_state.parameter_field_arrays[name], expected)


def test_visible_checkbox_and_no_selection_buttons_are_safe(monkeypatch, qtbot) -> None:
    _install_fake_slice_render(monkeypatch)
    project, workflow = _completed_field_project()
    window = _window(project, workflow, qtbot)
    dialog = M9ParameterFieldDialog(project, workflow, window)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.mouseClick(dialog.render_button, Qt.MouseButton.LeftButton)
    visible_item = dialog.layers_table.item(0, 0)
    visible_item.setCheckState(Qt.CheckState.Unchecked)
    record = window._get_m9_layer_manager().list_layers()[0]
    assert record.visible is False
    visible_item.setCheckState(Qt.CheckState.Checked)
    assert record.visible is True

    dialog.layers_table.clearSelection()
    dialog.layers_table.setCurrentCell(-1, -1)
    qtbot.mouseClick(dialog.remove_layer_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(dialog.toggle_layer_button, Qt.MouseButton.LeftButton)
    assert len(window._get_m9_layer_manager().list_layers()) == 1


def test_render_failure_leaves_no_layer_entry(monkeypatch, qtbot) -> None:
    from dfn_cave_studio.visualization.parameter_field_renderer import ParameterFieldRenderer

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("render failed")

    monkeypatch.setattr(ParameterFieldRenderer, "render_slice", fail)
    project, workflow = _completed_field_project()
    window = _window(project, workflow, qtbot)
    dialog = M9ParameterFieldDialog(project, workflow, window)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.mouseClick(dialog.render_button, Qt.MouseButton.LeftButton)
    assert window._get_m9_layer_manager().list_layers() == []
    assert _m9_actor_count(window) == 0


def test_new_and_open_project_clear_session_layers(monkeypatch, qtbot, tmp_path) -> None:
    _install_fake_slice_render(monkeypatch)
    project, workflow = _completed_field_project()
    window = _window(project, workflow, qtbot)
    dialog = M9ParameterFieldDialog(project, workflow, window)
    qtbot.addWidget(dialog)
    qtbot.mouseClick(dialog.render_button, Qt.MouseButton.LeftButton)
    assert _m9_actor_count(window) == 1
    window._on_new_project()
    assert window._get_m9_layer_manager().list_layers() == []
    assert _m9_actor_count(window) == 0

    fresh, fresh_workflow = _completed_field_project()
    window._project_store.adopt_project(fresh)
    window._workflow = fresh_workflow
    second = M9ParameterFieldDialog(fresh, fresh_workflow, window)
    qtbot.addWidget(second)
    qtbot.mouseClick(second.render_button, Qt.MouseButton.LeftButton)
    assert _m9_actor_count(window) == 1
    legacy_path = tmp_path / "next.dfncs"
    legacy_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(window._project_store, "open", lambda path: Project())
    window._load_project(legacy_path)
    assert window._get_m9_layer_manager().list_layers() == []
    assert _m9_actor_count(window) == 0
