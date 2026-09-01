"""GUI regressions for the non-modal committed-result M11 visualization dock."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv

from dfn_cave_studio.persistence.project_store import ProjectStore
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.m11_service import M11SecondVoxelizationService
from dfn_cave_studio.ui.main_window import MainWindow
from dfn_cave_studio.ui.panels.m11_visualization_panel import M11VisualizationPanel
from dfn_cave_studio.ui.qt_adapter import Qt
from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager
from dfn_cave_studio.visualization.m11_voxel_renderer import M11VoxelRenderer
from tests.gui.test_m11_dialog import FakeActor, FakePlaneWidget, FakePlotter, FakeScalarBarActor, _project_and_workflow


class InteractiveFakePlotter(FakePlotter):
    """Fake plotter that exposes plane and box callbacks without OpenGL."""

    def __init__(self) -> None:
        super().__init__()
        self.renderer.actors.update(
            {
                "m9_slice:keep": FakeActor(),
                "m10_dfn:keep": FakeActor(),
                "borehole:keep": FakeActor(),
                "model_bounds:keep": FakeActor(),
                "axes:keep": FakeActor(),
            }
        )
        self.scalar_bars["m9_scalar_bar:keep"] = FakeScalarBarActor()
        self.plane_callbacks = []
        self.plane_options = []
        self.box_callbacks = []

    def add_plane_widget(self, callback, **kwargs):
        widget = super().add_plane_widget(callback, **kwargs)
        self.plane_callbacks.append(callback)
        self.plane_options.append(dict(kwargs))
        return widget

    def add_box_widget(self, callback, **kwargs):
        widget = FakePlaneWidget(bounds=kwargs.get("bounds"))
        self.widgets.append(widget)
        self.box_callbacks.append(callback)
        return widget


def _panel(qtbot):
    project, workflow = _project_and_workflow()
    result = M11SecondVoxelizationService(project).compute(project.m10_state.realizations[0].realization_id)
    workflow.complete_step("second_voxelization")
    plotter = InteractiveFakePlotter()
    manager = M11LayerManager(plotter)
    panel = M11VisualizationPanel(manager)
    qtbot.addWidget(panel)
    panel.set_context(project, workflow)
    panel.show()
    return project, workflow, result, plotter, manager, panel


def _m11_actor_count(plotter) -> int:
    return sum(name.startswith("m11_voxel:") for name in plotter.renderer.actors)


def _m11_scalar_bar_count(plotter) -> int:
    return sum(name.startswith("m11_scalar_bar:") for name in plotter.scalar_bars)


def test_dock_is_non_modal_scrollable_and_reopenable(qtbot) -> None:
    _, _, _, _, manager, panel = _panel(qtbot)
    assert panel.isModal() is False
    assert panel.widget().widgetResizable()
    assert panel.features() & panel.DockWidgetFeature.DockWidgetFloatable
    panel.close()
    assert manager.control_widget_count() == 0
    panel.show()
    assert panel.isVisible()


def test_xyz_sections_sync_and_same_slots_do_not_accumulate(qtbot) -> None:
    _, _, _, plotter, manager, panel = _panel(qtbot)
    for axis in "xyz":
        qtbot.mouseClick(panel._axis_controls[axis]["enabled"], Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == 3
    assert _m11_actor_count(plotter) == 3

    x = panel._axis_controls["x"]
    for value in range(10, 20):
        x["slider"].setValue(value * 40)
        panel._axis_timers["x"].stop()
        panel._render_axis("x")
    assert len(manager.list_layers()) == 3
    assert _m11_actor_count(plotter) == 3
    lower, upper = panel._axis_bounds("x")
    expected = lower + (upper - lower) * x["slider"].value() / 1000.0
    assert x["coordinate"].value() == expected

    callback = plotter.plane_callbacks[0]
    callback((1.0, 0.0, 0.0), (upper, 0.0, 0.0))
    panel._axis_timers["x"].stop()
    panel._render_axis("x")
    assert x["coordinate"].value() == upper
    assert len(manager.list_layers()) == 3


def test_xyz_slider_numeric_and_widget_remain_bidirectional_for_50_alternations(qtbot) -> None:
    _, _, _, plotter, manager, panel = _panel(qtbot)
    for axis in "xyz":
        panel._axis_controls[axis]["enabled"].setChecked(True)
    assert [options["normal_rotation"] for options in plotter.plane_options] == [False, False, False]
    assert [options["assign_to_axis"] for options in plotter.plane_options] == ["x", "y", "z"]
    y_before = panel._axis_controls["y"]["coordinate"].value()
    z_before = panel._axis_controls["z"]["coordinate"].value()
    lower, upper = panel._axis_bounds("x")

    for index in range(50):
        if index % 2:
            plotter.plane_callbacks[0]((0.0, 1.0, 0.0), (lower + 0.2 * (upper - lower), 0.0, 0.0))
        else:
            panel._axis_controls["x"]["slider"].setValue(250 + index)
        panel._axis_timers["x"].stop()
        panel._render_axis("x")

    numeric_position = lower + 0.73 * (upper - lower)
    panel._axis_controls["x"]["coordinate"].setValue(numeric_position)
    qtbot.wait(100)
    record = panel._live_record("xyz_x", "orthogonal_section")
    assert record is not None
    assert record.coordinate == pytest.approx(panel._axis_controls["x"]["coordinate"].value())
    assert plotter.widgets[0].origin[0] == pytest.approx(record.coordinate)
    assert plotter.widgets[0].normal == (1.0, 0.0, 0.0)
    assert plotter.meshes[record.layer_id].bounds.x_min == pytest.approx(record.coordinate)
    assert plotter.meshes[record.layer_id].bounds.x_max == pytest.approx(record.coordinate)
    assert panel._axis_controls["y"]["coordinate"].value() == y_before
    assert panel._axis_controls["z"]["coordinate"].value() == z_before
    assert len(manager.list_layers()) == 3
    assert _m11_actor_count(plotter) == 3
    assert _m11_scalar_bar_count(plotter) == 1
    assert len(plotter.plane_callbacks) == 3
    assert sum(widget.observer_count for widget in plotter.widgets if widget.enabled) == 3


def test_arbitrary_inputs_widget_reset_and_axis_presets_are_bidirectional(qtbot) -> None:
    _, _, _, plotter, manager, panel = _panel(qtbot)
    panel.arbitrary_enabled.setChecked(True)
    widget = plotter.widgets[-1]
    for control, value in zip(panel.arbitrary_origin, (102.0, 204.0, 306.0)):
        control.setValue(value)
    for control, value in zip(panel.arbitrary_normal, (0.0, 2.0, 0.0)):
        control.setValue(value)
    panel._arbitrary_timer.stop()
    panel._render_arbitrary(False, False)
    assert widget.origin == (102.0, 204.0, 306.0)
    assert widget.normal == (0.0, 1.0, 0.0)

    plotter.plane_callbacks[-1]((1.0, 1.0, 0.0), (103.0, 203.0, 304.0))
    panel._arbitrary_timer.stop()
    panel._render_arbitrary(False, False)
    assert tuple(control.value() for control in panel.arbitrary_origin) == (103.0, 203.0, 304.0)
    assert tuple(control.value() for control in panel.arbitrary_normal) == (1.0, 1.0, 0.0)
    assert len(manager.list_layers()) == 1
    assert len(plotter.plane_callbacks) == 1

    qtbot.mouseClick(panel.arbitrary_orientations["z"], Qt.MouseButton.LeftButton)
    assert widget.normal == (0.0, 0.0, 1.0)
    qtbot.mouseClick(panel.reset_arbitrary, Qt.MouseButton.LeftButton)
    assert tuple(control.value() for control in panel.arbitrary_normal) == (0.0, 0.0, 1.0)


def test_cutaway_flip_exposes_internal_surface_and_uses_one_live_slot(qtbot) -> None:
    project, workflow, result, plotter, manager, panel = _panel(qtbot)
    arrays_before = {name: values.copy() for name, values in result.arrays.items()}
    workflow_before = workflow.to_dict()
    panel.cutaway_enabled.setChecked(True)
    live = panel._live_record("cutaway", "cutaway")
    assert live is not None
    first_bounds = tuple(plotter.meshes[live.layer_id].bounds)
    assert live.display_mode == "cutaway"
    assert len(live.widgets) == 1

    panel.cutaway_flip.setChecked(True)
    panel._cutaway_timer.stop()
    panel._render_cutaway(False, False)
    flipped = panel._live_record("cutaway", "cutaway")
    second_bounds = tuple(plotter.meshes[flipped.layer_id].bounds)
    assert flipped.layer_id == live.layer_id
    assert first_bounds != second_bounds
    assert len(manager.list_layers()) == 1
    assert _m11_actor_count(plotter) == 1
    assert _m11_scalar_bar_count(plotter) == 1

    qtbot.mouseClick(panel.snapshot_cutaway, Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == 2
    assert manager.set_visible(flipped.layer_id, False)
    assert not flipped.widgets[0].enabled
    assert manager.set_visible(flipped.layer_id, True)
    assert flipped.widgets[0].enabled
    assert manager.remove(flipped.layer_id)
    assert not flipped.widgets[0].enabled
    assert sum(record.display_mode == "cutaway" for record in manager.list_layers()) == 1
    manager.clear_m11_layers()
    assert _m11_actor_count(plotter) == 0
    assert _m11_scalar_bar_count(plotter) == 0
    assert {"m9_slice:keep", "m10_dfn:keep", "borehole:keep", "model_bounds:keep", "axes:keep"}.issubset(
        plotter.renderer.actors
    )
    for name, expected in arrays_before.items():
        np.testing.assert_array_equal(result.arrays[name], expected)
    assert workflow.to_dict() == workflow_before
    assert project.m11_state.results[0] is result


def test_snapshot_is_explicit_and_arbitrary_zero_normal_is_rejected(qtbot, monkeypatch) -> None:
    _, _, _, _, manager, panel = _panel(qtbot)
    qtbot.mouseClick(panel._axis_controls["z"]["enabled"], Qt.MouseButton.LeftButton)
    qtbot.mouseClick(panel._axis_controls["z"]["snapshot"], Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == 2

    calls = []
    monkeypatch.setattr("dfn_cave_studio.ui.panels.m11_visualization_panel.QMessageBox.critical", lambda *args: calls.append(args))
    for value in panel.arbitrary_normal:
        value.setValue(0.0)
    panel._render_arbitrary(False, False)
    assert calls
    assert len(manager.list_layers()) == 2


def test_box_cutaway_numeric_widget_sync_and_50_moves_preserve_other_namespaces(qtbot) -> None:
    _, workflow, result, plotter, manager, panel = _panel(qtbot)
    arrays_before = {name: values.copy() for name, values in result.arrays.items()}
    workflow_before = workflow.to_dict()
    qtbot.mouseClick(panel.surface_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(panel._axis_controls["x"]["enabled"], Qt.MouseButton.LeftButton)
    baseline_layers = len(manager.list_layers())
    panel.box_snap.setChecked(False)
    panel.box_enabled.setChecked(True)
    live = panel._box_live_record()
    assert live is not None and len(live.widgets) == 1
    assert manager.control_widget_count() == 0
    bounds = panel._analysis_bounds()
    for index in range(50):
        width = bounds[1] - bounds[0]
        moved = (
            bounds[0] + width * (0.10 + index / 1000.0),
            bounds[1] - width * 0.10,
            bounds[2],
            bounds[3],
            bounds[4],
            bounds[5],
        )
        if index % 2:
            widget = panel._box_live_record().widgets[0]
            widget.PlaceWidget(moved)
            plotter.box_callbacks[-1](SimpleNamespace(bounds=moved), widget)
            widget.InvokeEvent("EndInteractionEvent")
        else:
            panel.box_bounds[0].setValue(moved[0])
        panel._box_timer.stop()
        panel._render_box_cutaway(False, False)
    current = tuple(control.value() for control in panel.box_bounds)
    assert panel._box_live_record().widgets[0].bounds == pytest.approx(current)
    assert len(manager.list_layers()) == baseline_layers + 1
    assert _m11_actor_count(plotter) == baseline_layers + 1
    assert len(plotter.box_callbacks) == 1
    assert sum(widget.observer_count for widget in plotter.widgets if widget.enabled) == 3

    qtbot.mouseClick(panel.snapshot_box, Qt.MouseButton.LeftButton)
    snapshot = next(record for record in manager.list_layers() if record.display_mode == "box_cutaway" and not record.plane_id.startswith("slot_"))
    snapshot_bounds = tuple(plotter.meshes[snapshot.layer_id].bounds)
    panel.box_bounds[0].setValue(current[0] + (bounds[1] - bounds[0]) * 0.05)
    panel._box_timer.stop()
    panel._render_box_cutaway(False, False)
    assert tuple(plotter.meshes[snapshot.layer_id].bounds) == snapshot_bounds
    assert {"m9_slice:keep", "m10_dfn:keep", "borehole:keep", "model_bounds:keep", "axes:keep"}.issubset(
        plotter.renderer.actors
    )
    for name, expected in arrays_before.items():
        np.testing.assert_array_equal(result.arrays[name], expected)
    assert workflow.to_dict() == workflow_before


def test_box_cutaway_snap_reset_handle_and_snapshot_lifecycle(qtbot) -> None:
    _, _, _, plotter, manager, panel = _panel(qtbot)
    assert panel.box_snap.isChecked()
    qtbot.mouseClick(panel.reset_box_model, Qt.MouseButton.LeftButton)
    assert tuple(control.value() for control in panel.box_bounds) == pytest.approx(panel._analysis_bounds())
    qtbot.mouseClick(panel.reset_box_centre, Qt.MouseButton.LeftButton)
    metadata = panel._metadata()
    centred = tuple(control.value() for control in panel.box_bounds)
    for axis in range(3):
        for value in centred[2 * axis : 2 * axis + 2]:
            face = (value - metadata.origin[axis]) / metadata.spacing[axis]
            assert face == pytest.approx(round(face))

    panel.box_enabled.setChecked(True)
    live = panel._box_live_record()
    assert live is not None and live.plane_id.endswith("_snap")
    assert len(live.widgets) == 1
    panel.box_handle.setChecked(False)
    assert not live.widgets[0].enabled
    panel.box_handle.setChecked(True)
    assert live.widgets[0].enabled
    panel.box_snap.setChecked(False)
    continuous = panel._box_live_record()
    assert continuous is not None and continuous.plane_id.endswith("_continuous")
    assert len(manager.list_layers()) == 1
    assert _m11_actor_count(plotter) == 1
    assert _m11_scalar_bar_count(plotter) == 1
    assert sum(widget.observer_count for widget in plotter.widgets) == 2
    assert manager.remove(continuous.layer_id)
    assert _m11_actor_count(plotter) == 0
    assert _m11_scalar_bar_count(plotter) == 0
    assert sum(widget.observer_count for widget in plotter.widgets) == 0


def test_reset_box_to_authoritative_domain_ten_times_keeps_one_synchronized_live_slot(qtbot) -> None:
    project, workflow, result, plotter, manager, panel = _panel(qtbot)
    store = ProjectStore()
    store.adopt_project(project)
    arrays_before = {name: values.copy() for name, values in result.arrays.items()}
    workflow_before = workflow.to_dict()
    dirty_before = store.is_dirty
    panel.box_enabled.setChecked(True)
    expected = M11VoxelRenderer.analysis_bounds(panel._metadata())

    for _ in range(10):
        qtbot.mouseClick(panel.reset_box_model, Qt.MouseButton.LeftButton)
        assert tuple(control.value() for control in panel.box_bounds) == expected
        record = panel._box_live_record()
        assert record is not None and len(record.widgets) == 1
        assert record.widgets[0].bounds == pytest.approx(expected)
        assert tuple(plotter.meshes[record.layer_id].bounds) == pytest.approx(expected)

    assert len([record for record in manager.list_layers() if record.display_mode == "box_cutaway"]) == 1
    assert _m11_actor_count(plotter) == 1
    assert _m11_scalar_bar_count(plotter) == 1
    assert sum(widget.observer_count for widget in plotter.widgets if widget.enabled) == 2
    for name, expected_array in arrays_before.items():
        np.testing.assert_array_equal(result.arrays[name], expected_array)
    assert workflow.to_dict() == workflow_before
    assert store.is_dirty is dirty_before


@pytest.mark.real_vtk_render
def test_real_vtk_box_events_preview_continuously_then_snap_on_end(qtbot) -> None:
    """Exercise real vtkBoxWidget events; mouse picking remains manual QA."""
    project, workflow = _project_and_workflow()
    M11SecondVoxelizationService(project).compute(project.m10_state.realizations[0].realization_id)
    workflow.complete_step("second_voxelization")
    plotter = pv.Plotter(off_screen=True)
    manager = M11LayerManager(plotter)
    panel = M11VisualizationPanel(manager)
    qtbot.addWidget(panel)
    panel.set_context(project, workflow)
    panel.show()

    panel.box_enabled.setChecked(True)
    record = panel._box_live_record()
    assert record is not None and len(record.widgets) == 1
    widget = record.widgets[0]
    assert widget.GetEnabled() == 1
    assert widget.GetInteractor() is plotter.iren.interactor
    assert widget.GetRotationEnabled() == 0
    assert widget.GetTranslationEnabled() == 1
    assert widget.HasObserver("InteractionEvent")
    assert widget.HasObserver("EndInteractionEvent")

    analysis = panel._analysis_bounds()
    continuous = tuple(
        value
        for axis in range(3)
        for value in (
            analysis[2 * axis] + 0.23 * (analysis[2 * axis + 1] - analysis[2 * axis]),
            analysis[2 * axis + 1] - 0.17 * (analysis[2 * axis + 1] - analysis[2 * axis]),
        )
    )
    widget.PlaceWidget(continuous)
    widget.InvokeEvent("InteractionEvent")
    qtbot.wait(10)
    assert tuple(control.value() for control in panel.box_bounds) == pytest.approx(continuous)
    during_drag = pv.PolyData()
    widget.GetPolyData(during_drag)
    # Snap is enabled, but InteractionEvent must not PlaceWidget back.
    assert tuple(during_drag.bounds) == pytest.approx(continuous)
    assert panel._box_timer.isActive()
    panel._box_timer.stop()
    panel._render_box_cutaway(False, False)
    preview_record = panel._box_live_record()
    assert preview_record is not None and preview_record.widgets == (widget,)
    assert tuple(preview_record.actor.mapper.dataset.bounds) == pytest.approx(continuous)

    widget.InvokeEvent("EndInteractionEvent")
    qtbot.wait(10)
    snapped = M11VoxelRenderer.snap_box_bounds(panel._metadata(), continuous)
    assert not panel._box_timer.isActive()
    assert tuple(control.value() for control in panel.box_bounds) == pytest.approx(snapped)
    final_widget = pv.PolyData()
    widget.GetPolyData(final_widget)
    assert tuple(final_widget.bounds) == pytest.approx(snapped)
    final_record = panel._box_live_record()
    assert final_record is not None and final_record.widgets == (widget,)
    assert tuple(final_record.actor.mapper.dataset.bounds) == pytest.approx(snapped)

    panel.box_snap.setChecked(False)
    unsnapped_record = panel._box_live_record()
    unsnapped_widget = unsnapped_record.widgets[0]
    unsnapped = tuple(
        value
        for axis in range(3)
        for value in (
            analysis[2 * axis] + 0.19 * (analysis[2 * axis + 1] - analysis[2 * axis]),
            analysis[2 * axis + 1] - 0.21 * (analysis[2 * axis + 1] - analysis[2 * axis]),
        )
    )
    for face_index in range(6):
        moved_face = list(unsnapped)
        span = analysis[2 * (face_index // 2) + 1] - analysis[2 * (face_index // 2)]
        moved_face[face_index] += (0.01 if face_index % 2 == 0 else -0.01) * span
        unsnapped_widget.PlaceWidget(tuple(moved_face))
        unsnapped_widget.InvokeEvent("InteractionEvent")
        unsnapped_widget.InvokeEvent("EndInteractionEvent")
        qtbot.wait(2)
        assert tuple(control.value() for control in panel.box_bounds) == pytest.approx(moved_face)
        assert tuple(panel._box_live_record().actor.mapper.dataset.bounds) == pytest.approx(moved_face)

    translated = tuple(
        value + 0.01 * (analysis[2 * (index // 2) + 1] - analysis[2 * (index // 2)])
        for index, value in enumerate(unsnapped)
    )
    unsnapped_widget.PlaceWidget(translated)
    unsnapped_widget.InvokeEvent("InteractionEvent")
    unsnapped_widget.InvokeEvent("EndInteractionEvent")
    qtbot.wait(10)
    assert tuple(control.value() for control in panel.box_bounds) == pytest.approx(translated)
    assert tuple(panel._box_live_record().actor.mapper.dataset.bounds) == pytest.approx(translated)

    panel.close()
    assert widget.GetEnabled() == 0
    assert unsnapped_widget.GetEnabled() == 0
    panel.show()
    qtbot.wait(10)
    restored = panel._box_live_record().widgets[0]
    assert restored is not unsnapped_widget
    assert restored.GetEnabled() == 1
    assert restored.HasObserver("InteractionEvent")
    assert restored.HasObserver("EndInteractionEvent")
    panel.close()
    plotter.close()


def test_layer_hide_remove_clear_and_opacity_are_display_only(qtbot) -> None:
    project, workflow, result, plotter, manager, panel = _panel(qtbot)
    store = ProjectStore()
    store.adopt_project(project)
    arrays_before = {name: value.copy() for name, value in result.arrays.items()}
    m10_before = {
        realization.realization_id: {
            name: values.copy() for name, values in realization.geometry_arrays.items()
        }
        for realization in project.m10_state.realizations
    }
    workflow_before = workflow.to_dict()
    qtbot.mouseClick(panel.surface_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(panel._axis_controls["z"]["enabled"], Qt.MouseButton.LeftButton)
    panel.layers_table.selectRow(0)
    selected = panel._selected_layer_id()
    qtbot.mouseClick(panel.toggle_layer, Qt.MouseButton.LeftButton)
    assert manager.get(selected).visible is False
    qtbot.mouseClick(panel.toggle_layer, Qt.MouseButton.LeftButton)
    panel.layer_opacity.setValue(0.35)
    assert manager.get(selected).opacity == 0.35
    qtbot.mouseClick(panel.remove_layer, Qt.MouseButton.LeftButton)
    assert manager.get(selected) is None
    qtbot.mouseClick(panel.clear_layers, Qt.MouseButton.LeftButton)
    assert manager.list_layers() == []
    assert _m11_scalar_bar_count(plotter) == 0
    assert "m9_scalar_bar:keep" in plotter.scalar_bars
    assert store.is_dirty is False
    assert workflow.to_dict() == workflow_before
    for realization in project.m10_state.realizations:
        for name, expected in m10_before[realization.realization_id].items():
            np.testing.assert_array_equal(realization.geometry_arrays[name], expected)
    for name, expected in arrays_before.items():
        np.testing.assert_array_equal(result.arrays[name], expected)


def test_invalid_or_stale_result_disables_controls(qtbot) -> None:
    project, workflow, _, _, _, panel = _panel(qtbot)
    workflow.invalidate_steps(["second_voxelization"])
    panel.refresh()
    assert not panel.controls.isEnabled()
    assert "No valid committed" in panel.status_label.text()
    workflow.complete_step("second_voxelization")
    project.m10_state.realizations[0].config_hash = "stale"
    panel.refresh()
    assert not panel.controls.isEnabled()


def test_close_reopen_restores_one_handle_without_duplicate_registration(qtbot) -> None:
    _, _, _, _, manager, panel = _panel(qtbot)
    panel._axis_controls["x"]["enabled"].setChecked(True)
    assert sum(widget.enabled for widget in manager.plotter.widgets) == 1
    for _ in range(3):
        panel.hide()
        assert sum(widget.enabled for widget in manager.plotter.widgets) == 0
        assert sum(widget.observer_count for widget in manager.plotter.widgets) == 0
        panel.show()
        assert sum(widget.enabled for widget in manager.plotter.widgets) == 1
        assert sum(widget.observer_count for widget in manager.plotter.widgets) == 1
    assert len(manager.list_layers()) == 1
    assert len(manager.list_layers()[0].widgets) == 1


def test_m10_invalidation_clears_only_m11_resources(qtbot) -> None:
    _, workflow, _, plotter, manager, panel = _panel(qtbot)
    panel._axis_controls["z"]["enabled"].setChecked(True)
    panel.box_enabled.setChecked(True)
    window = MainWindow()
    qtbot.addWidget(window)
    window._workflow = workflow
    window._m11_layer_manager = manager
    window._m11_visualization_panel.layer_manager = manager
    window._invalidate_m11_after_m10_change()
    assert manager.list_layers() == []
    assert manager.control_widget_count() == 0
    assert not any(name.startswith("m11_scalar_bar:") for name in plotter.scalar_bars)
    assert {"m9_slice:keep", "m10_dfn:keep", "borehole:keep", "model_bounds:keep", "axes:keep"}.issubset(
        plotter.renderer.actors
    )
    window._auto_save_timer.stop()
    window._project_store._dirty = False
    window.close()


def test_main_window_exposes_m11_visualization_menu_and_lifecycle(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window._m11_visualization_panel is not None
    assert window._m11_visualization_action in window._vis_menu.actions()
    assert window._m11_visualization_panel.widget().widgetResizable()
    window._m11_visualization_action.setChecked(False)
    window._m11_visualization_action.trigger()
    qtbot.wait(20)
    assert window._m11_visualization_panel.isVisible()
    window._auto_save_timer.stop()
    window._project_store._dirty = False
    window.close()


def test_new_and_open_project_clear_prior_m11_session_layers(qtbot, tmp_path) -> None:
    project, workflow, _, _, manager, panel = _panel(qtbot)
    panel._axis_controls["x"]["enabled"].setChecked(True)
    assert manager.list_layers()
    path = tmp_path / "committed-m11.dfnproj"
    ZipProjectStore().save(project, path)

    window = MainWindow()
    qtbot.addWidget(window)
    window._workflow = workflow
    window._m11_layer_manager = manager
    window._m11_visualization_panel.layer_manager = manager
    window._load_project(path)
    assert manager.list_layers() == []

    panel.set_context(project, workflow)
    panel._render_axis("x", create_widget=True)
    assert manager.list_layers()
    window._on_new_project()
    assert manager.list_layers() == []
    window._auto_save_timer.stop()
    window._project_store._dirty = False
    window.close()
