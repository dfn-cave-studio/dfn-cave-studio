"""Real-button GUI regression tests for M11.1 slice-layer management."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dfn_cave_studio.models.m10 import M10GenerationConfig
from dfn_cave_studio.persistence.project_store import ProjectStore
from dfn_cave_studio.services.m10_service import M10Service
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController
from dfn_cave_studio.ui.dialogs.m11_dialog import M11SecondVoxelizationDialog
from dfn_cave_studio.ui.main_window import MainWindow
from dfn_cave_studio.ui.qt_adapter import QDialogButtonBox, Qt, QThreadPool
from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager
from tests.integration.test_m10_persistence_export import make_m10_project


class FakeProperty:
    def __init__(self) -> None:
        self.opacity = 1.0

    def SetOpacity(self, opacity: float) -> None:
        self.opacity = float(opacity)


class FakeActor:
    def __init__(self) -> None:
        self.visible = True
        self.actor_property = FakeProperty()

    def SetVisibility(self, visible: bool) -> None:
        self.visible = bool(visible)

    def GetProperty(self) -> FakeProperty:
        return self.actor_property

    def GetMapper(self):
        return object()


class FakeScalarBarActor:
    def __init__(self) -> None:
        self.visible = True
        self.title = ""

    def SetVisibility(self, visible: bool) -> None:
        self.visible = bool(visible)

    def SetTitle(self, title: str) -> None:
        self.title = str(title)


class FakePlaneWidget:
    def __init__(self, *, origin=None, normal=None, bounds=None) -> None:
        self.enabled = True
        self.origin = None if origin is None else tuple(float(value) for value in origin)
        self.normal = None if normal is None else tuple(float(value) for value in normal)
        self.bounds = None if bounds is None else tuple(float(value) for value in bounds)
        self.modified_count = 0
        self.observer_count = 1
        self._observers = {}
        self._next_observer_id = 1
        self.rotation_enabled = True
        self.translation_enabled = True

    def Off(self) -> None:
        self.enabled = False

    def SetEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def On(self) -> None:
        self.enabled = True

    def SetRotationEnabled(self, enabled: bool) -> None:
        self.rotation_enabled = bool(enabled)

    def SetTranslationEnabled(self, enabled: bool) -> None:
        self.translation_enabled = bool(enabled)

    def SetOrigin(self, *origin) -> None:
        self.origin = tuple(float(value) for value in origin)

    def SetNormal(self, *normal) -> None:
        self.normal = tuple(float(value) for value in normal)

    def PlaceWidget(self, *bounds) -> None:
        values = bounds[0] if len(bounds) == 1 and not isinstance(bounds[0], (int, float)) else bounds
        self.bounds = tuple(float(value) for value in values)

    def GetPolyData(self, poly_data) -> None:
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        poly_data.points = np.asarray(
            [
                (x, y, z)
                for x in (xmin, xmax)
                for y in (ymin, ymax)
                for z in (zmin, zmax)
            ],
            dtype=float,
        )

    def Modified(self) -> None:
        self.modified_count += 1

    def AddObserver(self, event: str, callback) -> int:
        observer_id = self._next_observer_id
        self._next_observer_id += 1
        self._observers[observer_id] = (str(event), callback)
        self.observer_count += 1
        return observer_id

    def RemoveObserver(self, observer_id: int) -> None:
        if self._observers.pop(int(observer_id), None) is not None:
            self.observer_count -= 1

    def InvokeEvent(self, event: str) -> None:
        for registered_event, callback in tuple(self._observers.values()):
            if registered_event == str(event):
                callback(self, event)

    def RemoveAllObservers(self) -> None:
        self.observer_count = 0
        self._observers.clear()


class FakePlotter:
    def __init__(self) -> None:
        self.renderer = SimpleNamespace(actors={})
        self.scalar_bars = {"m9_scalar_bar:keep": FakeScalarBarActor()}
        self.meshes = {}
        self.mesh_kwargs = {}
        self.widgets = []
        self.render_count = 0
        self.fail_render = False

    def add_mesh(self, mesh, *, name, **kwargs):
        if self.fail_render:
            raise RuntimeError("synthetic render failure")
        actor = FakeActor()
        self.renderer.actors[name] = actor
        self.meshes[name] = mesh
        self.mesh_kwargs[name] = kwargs
        return actor

    def remove_actor(self, actor, *, render=True) -> None:
        if isinstance(actor, str):
            self.renderer.actors.pop(actor, None)
        else:
            for name, candidate in list(self.renderer.actors.items()):
                if candidate is actor:
                    self.renderer.actors.pop(name)
        if render:
            self.render()

    def render(self) -> None:
        self.render_count += 1

    def add_scalar_bar(self, *, title, mapper, render=False):
        del mapper, render
        if self.fail_render:
            raise RuntimeError("synthetic scalar-bar failure")
        if title in self.scalar_bars:
            return None
        actor = FakeScalarBarActor()
        self.scalar_bars[title] = actor
        return actor

    def remove_scalar_bar(self, *, title, render=False) -> None:
        del render
        self.scalar_bars.pop(title)

    def add_plane_widget(self, callback, **kwargs):
        del callback
        widget = FakePlaneWidget(origin=kwargs.get("origin"), normal=kwargs.get("normal"))
        self.widgets.append(widget)
        return widget


def _project_and_workflow():
    project = make_m10_project()
    M10Service(project).generate_batch(
        M10GenerationConfig(base_seed=42, condition_calibration_observations=False)
    )
    workflow = WorkflowController()
    for step in ("voxel_grid", "density", "size", "parameter_field", "explicit_dfn"):
        workflow.complete_step(step)
    workflow.mark_ready("second_voxelization")
    return project, workflow


def _dialog(qtbot, monkeypatch):
    project, workflow = _project_and_workflow()
    plotter = FakePlotter()
    manager = M11LayerManager(plotter)
    dialog = M11SecondVoxelizationDialog(project, workflow, manager)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QThreadPool, "start", lambda self, worker: worker.run())
    return project, workflow, plotter, manager, dialog


def _compute(dialog, qtbot) -> None:
    qtbot.mouseClick(dialog.compute_button, Qt.MouseButton.LeftButton)
    assert dialog._pending_result is not None


def _render(dialog, qtbot) -> None:
    qtbot.mouseClick(dialog.render_button, Qt.MouseButton.LeftButton)


def test_compute_button_then_ok_commits_complete_result(qtbot, monkeypatch) -> None:
    project, workflow, _, manager, dialog = _dialog(qtbot, monkeypatch)
    _compute(dialog, qtbot)
    assert project.m11_state.results == []
    _render(dialog, qtbot)
    assert manager.list_layers()[0].pending

    qtbot.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    assert dialog.committed_changes
    assert len(project.m11_state.results) == 1
    assert workflow.get_step("second_voxelization").status == StepStatus.COMPLETED
    assert not manager.list_layers()[0].pending


def test_cancel_after_compute_removes_only_pending_layers(qtbot, monkeypatch) -> None:
    project, workflow, plotter, manager, dialog = _dialog(qtbot, monkeypatch)
    before = project.m11_state.model_copy(deep=True)
    workflow_before = workflow.to_dict()
    plotter.renderer.actors["m9_slice:keep"] = FakeActor()
    plotter.renderer.actors["m10_dfn:keep"] = FakeActor()
    _compute(dialog, qtbot)
    _render(dialog, qtbot)

    qtbot.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel), Qt.MouseButton.LeftButton)

    assert project.m11_state == before
    assert workflow.to_dict() == workflow_before
    assert manager.list_layers() == []
    assert set(plotter.renderer.actors) == {"m9_slice:keep", "m10_dfn:keep"}
    assert set(plotter.scalar_bars) == {"m9_scalar_bar:keep"}


def test_running_cancel_reports_request_stopping_and_stopped_states(qtbot, monkeypatch) -> None:
    _, _, _, _, dialog = _dialog(qtbot, monkeypatch)

    class FakeRunningWorker:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    worker = FakeRunningWorker()
    dialog._worker = worker
    dialog._set_running(True)
    qtbot.mouseClick(dialog.cancel_button, Qt.MouseButton.LeftButton)
    assert worker.cancelled
    assert "stopping" in dialog.status.text().lower()
    assert not dialog.cancel_button.isEnabled()
    assert not dialog.compute_button.isEnabled()
    dialog._on_cancelled()
    assert "no partial result" in dialog.status.text().lower()
    assert dialog.compute_button.isEnabled()


def test_same_slice_rendered_ten_times_has_one_actor_and_record(qtbot, monkeypatch) -> None:
    _, _, plotter, manager, dialog = _dialog(qtbot, monkeypatch)
    _compute(dialog, qtbot)
    for _ in range(10):
        _render(dialog, qtbot)
    assert len(manager.list_layers()) == 1
    assert len(plotter.renderer.actors) == 1
    assert len(plotter.scalar_bars) == 2
    assert dialog.layer_table.rowCount() == 1


def test_three_fields_coexist_and_layer_operations_preserve_science(qtbot, monkeypatch) -> None:
    project, workflow, plotter, manager, dialog = _dialog(qtbot, monkeypatch)
    store = ProjectStore()
    store.adopt_project(project)
    assert not store.is_dirty
    _compute(dialog, qtbot)
    pending_before = dialog._pending_result.model_copy(deep=True)
    m10_before = project.m10_state.model_copy(deep=True)
    workflow_before = workflow.to_dict()

    for field in ("P32 explicit intersection", "P32 subgrid", "P32 total"):
        dialog.field_combo.setCurrentText(field)
        _render(dialog, qtbot)
    assert len(manager.list_layers()) == 3
    assert len(plotter.renderer.actors) == 3

    dialog.layer_table.selectRow(0)
    selected_id = manager.list_layers()[0].layer_id
    qtbot.mouseClick(dialog.toggle_layer_button, Qt.MouseButton.LeftButton)
    assert not manager.get(selected_id).visible
    assert not manager.get(selected_id).scalar_bar_actor.visible
    qtbot.mouseClick(dialog.toggle_layer_button, Qt.MouseButton.LeftButton)
    assert manager.get(selected_id).visible
    assert manager.get(selected_id).scalar_bar_actor.visible
    dialog.opacity.setValue(0.35)
    assert manager.get(selected_id).opacity == 0.35
    assert manager.get(selected_id).actor.actor_property.opacity == 0.35
    removed_scalar_bar_id = manager.get(selected_id).scalar_bar_id
    qtbot.mouseClick(dialog.remove_layer_button, Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == 2
    assert removed_scalar_bar_id not in plotter.scalar_bars

    assert dialog._pending_result.model_dump(exclude={"arrays"}) == pending_before.model_dump(exclude={"arrays"})
    for name, expected in pending_before.arrays.items():
        np.testing.assert_array_equal(dialog._pending_result.arrays[name], expected)
    assert project.m10_state.model_dump(exclude={"realizations"}) == m10_before.model_dump(exclude={"realizations"})
    assert len(project.m10_state.realizations) == len(m10_before.realizations)
    for actual, expected in zip(project.m10_state.realizations, m10_before.realizations):
        assert actual.model_dump(exclude={"geometry_arrays"}) == expected.model_dump(exclude={"geometry_arrays"})
        for name, expected_array in expected.geometry_arrays.items():
            np.testing.assert_array_equal(actual.geometry_arrays[name], expected_array)
    assert workflow.to_dict() == workflow_before
    assert not store.is_dirty


def test_clear_current_and_clear_all_preserve_other_namespaces(qtbot, monkeypatch) -> None:
    _, _, plotter, manager, dialog = _dialog(qtbot, monkeypatch)
    _compute(dialog, qtbot)
    dialog.display_mode_combo.setCurrentText("Orthogonal Section")
    plotter.renderer.actors.update(
        {
            "m9_slice:keep": FakeActor(),
            "m10_dfn:keep": FakeActor(),
            "borehole:keep": FakeActor(),
            "voxel_analysis_domain": FakeActor(),
            "dfn_generation_domain": FakeActor(),
            "axes": FakeActor(),
        }
    )
    _render(dialog, qtbot)
    dialog.slice_fraction.setValue(0.9)
    _render(dialog, qtbot)
    assert len(manager.list_layers()) == 2
    shared_scalar_bar_id = manager.list_layers()[0].scalar_bar_id
    assert shared_scalar_bar_id == manager.list_layers()[1].scalar_bar_id

    qtbot.mouseClick(dialog.clear_current_button, Qt.MouseButton.LeftButton)
    assert len(manager.list_layers()) == 1
    assert shared_scalar_bar_id in plotter.scalar_bars
    qtbot.mouseClick(dialog.clear_all_button, Qt.MouseButton.LeftButton)
    assert manager.list_layers() == []
    assert shared_scalar_bar_id not in plotter.scalar_bars
    assert set(plotter.scalar_bars) == {"m9_scalar_bar:keep"}
    assert set(plotter.renderer.actors) == {
        "m9_slice:keep",
        "m10_dfn:keep",
        "borehole:keep",
        "voxel_analysis_domain",
        "dfn_generation_domain",
        "axes",
    }


def test_no_selection_buttons_are_safe_and_render_failure_is_not_registered(qtbot, monkeypatch) -> None:
    _, _, plotter, manager, dialog = _dialog(qtbot, monkeypatch)
    _compute(dialog, qtbot)
    qtbot.mouseClick(dialog.toggle_layer_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(dialog.remove_layer_button, Qt.MouseButton.LeftButton)
    plotter.fail_render = True
    monkeypatch.setattr("dfn_cave_studio.ui.dialogs.m11_dialog.QMessageBox.critical", lambda *args: None)
    _render(dialog, qtbot)
    assert manager.list_layers() == []
    assert plotter.renderer.actors == {}
    assert set(plotter.scalar_bars) == {"m9_scalar_bar:keep"}


def test_four_display_modes_controls_and_cancel_preserve_science(qtbot, monkeypatch) -> None:
    project, workflow, plotter, manager, dialog = _dialog(qtbot, monkeypatch)
    _compute(dialog, qtbot)
    pending_result = dialog._pending_result
    arrays_before = {name: array.copy() for name, array in pending_result.arrays.items()}
    workflow_before = workflow.to_dict()
    store = ProjectStore()
    store.adopt_project(project)
    assert not store.is_dirty

    dialog.show_grid_lines.setChecked(True)
    for mode in ("Voxel Cells", "Outer Surface Cloud", "Orthogonal Section"):
        dialog.display_mode_combo.setCurrentText(mode)
        _render(dialog, qtbot)
    dialog.display_mode_combo.setCurrentText("Arbitrary Plane")
    dialog.interpolation_combo.setCurrentText("Smooth Display")
    dialog.interactive_plane.setChecked(True)
    _render(dialog, qtbot)

    assert len(manager.list_layers()) == 4
    assert dialog.layer_table.columnCount() == 11
    assert {record.display_mode for record in manager.list_layers()} == {
        "voxel_cells",
        "outer_surface",
        "orthogonal_section",
        "arbitrary_plane",
    }
    assert any(record.interpolation_mode == "smooth" for record in manager.list_layers())
    assert all(plotter.mesh_kwargs[record.layer_id]["show_edges"] for record in manager.list_layers())
    assert sum(widget.enabled for widget in plotter.widgets) == 1
    assert "scientific voxel values unchanged" in dialog.interpolation_notice.text()

    qtbot.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel), Qt.MouseButton.LeftButton)
    assert manager.list_layers() == []
    assert not any(widget.enabled for widget in plotter.widgets)
    for name, expected in arrays_before.items():
        np.testing.assert_array_equal(pending_result.arrays[name], expected)
    assert workflow.to_dict() == workflow_before
    assert not store.is_dirty


def test_m10_invalidation_clears_stale_m11_layers(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    plotter = FakePlotter()
    manager = M11LayerManager(plotter)
    window._m11_layer_manager = manager
    actor = plotter.add_mesh(object(), name="m11_voxel:stale")
    scalar_bar_id = "m11_scalar_bar:r1:p32_total:set_all"
    scalar_bar_actor = plotter.add_scalar_bar(
        title=scalar_bar_id, mapper=actor.GetMapper(), render=False
    )
    manager.add_or_replace(
        "m11_voxel:r1:p32_total:set_all:z:1",
        actor,
        scalar_bar_id=scalar_bar_id,
        scalar_bar_actor=scalar_bar_actor,
        scalar_bar_title="p32_total (m^-1)",
        realization_id="r1",
        field_name="p32_total",
        joint_set_id=None,
        axis="z",
        slice_index=1,
        coordinate=1.5,
        opacity=1.0,
    )
    window._workflow.complete_step("second_voxelization")

    window._invalidate_m11_after_m10_change()

    assert manager.list_layers() == []
    assert window._workflow.get_step("second_voxelization").status == StepStatus.STALE
