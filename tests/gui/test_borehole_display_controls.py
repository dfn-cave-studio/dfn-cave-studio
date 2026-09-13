from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.borehole_fracture_realization import BoreholeFractureComponent
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.ui.dialogs.m8_spatial_grid_dialog import M8SpatialGridDialog
from dfn_cave_studio.ui.main_window import MainWindow
from dfn_cave_studio.ui.panels.borehole_database_panel import BoreholeDatabasePanel
from dfn_cave_studio.ui.qt_adapter import Qt
from dfn_cave_studio.visualization.borehole_display_manager import BoreholeActorMetadata, BoreholeDisplayManager
from dfn_cave_studio.visualization.borehole_fracture_renderer import BoreholeFractureRenderer
from dfn_cave_studio.visualization.m8_spatial_preview import M8SpatialPreviewRenderer


def _window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_new_project()
    repository = BoreholeRepository(window._project_store.current_project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame(
            [
                {
                    "borehole_id": "SYN-1",
                    "collar_x": 10,
                    "collar_y": 20,
                    "collar_z": 30,
                    "final_depth": 50,
                    "azimuth": 0,
                    "dip": -90,
                    "domain_id": 2,
                }
            ]
        ),
        "synthetic.csv",
    )
    return window


def _click_preview(window, qtbot) -> M8SpatialGridDialog:
    dialog = M8SpatialGridDialog(
        window._project_store.current_project,
        plotter=window._plotter,
        mode="voxel",
        borehole_display_manager=window._get_borehole_display_manager(),
        parent=window,
    )
    qtbot.addWidget(dialog)
    qtbot.mouseClick(dialog._preview_button, Qt.MouseButton.LeftButton)
    return dialog


def test_borehole_labels_projection_and_views_are_session_only(qtbot) -> None:
    window = _window(qtbot)
    project = window._project_store.current_project
    before = project.model_dump(mode="json")
    workflow_before = window._workflow.to_dict()
    dirty_before = window._project_store.is_dirty
    assert "borehole:collar_labels" not in window._plotter._actors_by_name
    assert not window._borehole_labels_action.isEnabled()
    assert window._plotter._observers == {}
    dialog = _click_preview(window, qtbot)
    manager = window._borehole_display_manager
    assert window._borehole_labels_action.isEnabled()
    assert "borehole:collar_labels" in window._plotter._actors_by_name
    label = window._plotter._actors_by_name["borehole:collar_labels"]
    borehole_actor = window._plotter._actors_by_name["trajectory:SYN-1"]
    assert label.labels == ["SYN-1"]
    assert label.points[0][:2] == [10.0, 20.0]
    assert 30.0 < label.points[0][2] < 31.0
    assert not label.GetPickable() and borehole_actor.GetPickable()

    window._borehole_labels_action.setChecked(False)
    assert not label.GetVisibility()
    assert borehole_actor.GetVisibility()
    qtbot.mouseClick(dialog._preview_button, Qt.MouseButton.LeftButton)
    label = window._plotter._actors_by_name["borehole:collar_labels"]
    assert not label.GetVisibility()
    window._borehole_labels_action.setChecked(True)
    assert label.GetVisibility()

    class Camera:
        position = (10.0, 10.0, 10.0)
        focal = (0.0, 0.0, 0.0)
        up = (0.0, 0.0, 1.0)
        angle = 30.0
        scale = 1.0

        def GetPosition(self):
            return self.position

        def GetFocalPoint(self):
            return self.focal

        def GetViewUp(self):
            return self.up

        def GetViewAngle(self):
            return self.angle

        def GetParallelScale(self):
            return self.scale

        def SetPosition(self, value):
            self.position = tuple(value)

        def SetFocalPoint(self, value):
            self.focal = tuple(value)

        def SetViewUp(self, value):
            self.up = tuple(value)

        def SetViewAngle(self, value):
            self.angle = float(value)

        def SetParallelScale(self, value):
            self.scale = float(value)

    vtk_camera = Camera()
    window._plotter.camera = vtk_camera
    pose = (vtk_camera.position, vtk_camera.focal, vtk_camera.up)
    window._orthographic_action.trigger()
    assert window._plotter.parallel_projection
    assert (vtk_camera.position, vtk_camera.focal, vtk_camera.up) == pose
    assert vtk_camera.scale > 0
    window._perspective_action.trigger()
    assert not window._plotter.parallel_projection
    assert (vtk_camera.position, vtk_camera.focal, vtk_camera.up) == pose
    assert vtk_camera.angle == pytest.approx(30.0)
    current_pose = list(window._plotter.camera_position)
    window._on_current_view()
    assert window._plotter.camera_position == current_pose
    window._on_top_view()
    assert window._plotter.camera_position[0][2] > window._plotter.camera_position[1][2]
    assert window._plotter.camera_position[2] == (0.0, 1.0, 0.0)
    window._on_front_view()
    assert window._plotter.camera_position[0][1] > window._plotter.camera_position[1][1]
    assert window._plotter.camera_position[2] == (0.0, 0.0, 1.0)
    window._on_left_view()
    assert window._plotter.camera_position[0][0] > window._plotter.camera_position[1][0]
    window._on_isometric_view()
    position, focal, view_up = window._plotter.camera_position
    assert all(position[index] > focal[index] for index in range(3))
    assert view_up == (0.0, 0.0, 1.0)
    assert project.model_dump(mode="json") == before
    assert window._workflow.to_dict() == workflow_before
    assert window._project_store.is_dirty == dirty_before
    assert manager is window._borehole_display_manager
    assert len(window._plotter._observers) == 1
    for _ in range(10):
        qtbot.mouseClick(dialog._preview_button, Qt.MouseButton.LeftButton)
    assert len(window._plotter._observers) == 1
    assert len([name for name in window._plotter._actors_by_name if name.startswith("borehole:")]) == 1
    info = next(iter(manager.actor_metadata.values()))
    hover = manager._format_hover(info)
    assert "SYN-1" in hover and "E/N/R: 10.00, 20.00, 30.00" in hover and "Domain: 2" in hover


def test_project_switch_clears_old_borehole_display_resources(qtbot) -> None:
    window = _window(qtbot)
    _click_preview(window, qtbot)
    old_manager = window._borehole_display_manager
    assert old_manager.actor_metadata
    window._on_new_project()
    assert old_manager.actor_metadata == {}
    assert window._plotter._observers == {}
    assert "borehole:collar_labels" not in window._plotter._actors_by_name


def test_empty_collection_keeps_borehole_display_controls_safe(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_new_project()
    _click_preview(window, qtbot)
    manager = window._borehole_display_manager
    assert manager.actor_metadata == {}
    assert window._plotter._observers == {}
    assert "borehole:collar_labels" not in window._plotter._actors_by_name
    window._borehole_labels_action.setChecked(False)
    window._orthographic_action.trigger()
    window._perspective_action.trigger()


def test_m8_preview_reuses_shared_borehole_display_controller(qtbot) -> None:
    window = _window(qtbot)
    project = window._project_store.current_project
    manager = window._get_borehole_display_manager()
    bounds = ModelBounds(x_min=0, x_max=20, y_min=10, y_max=30, z_min=-30, z_max=40)
    before = project.model_dump(mode="json")
    workflow_before = window._workflow.to_dict()
    M8SpatialPreviewRenderer.render(
        window._plotter,
        bounds,
        bounds,
        VoxelConfig(cell_size_x=5, cell_size_y=5, cell_size_z=5),
        project.borehole_collection,
        borehole_display_manager=manager,
        domain_by_hole={"SYN-1": 2},
    )
    assert "trajectory:SYN-1" in window._plotter._actors_by_name
    assert "borehole:collar_labels" in window._plotter._actors_by_name
    assert len(manager.actor_metadata) == 1
    assert len(window._plotter._observers) == 1
    assert project.model_dump(mode="json") == before
    assert window._workflow.to_dict() == workflow_before


def test_preview_keeps_all_overlapping_labels_without_decluttering(qtbot) -> None:
    window = _window(qtbot)
    project = window._project_store.current_project
    BoreholeRepository(project).import_dataframe(
        "collars",
        pd.DataFrame(
            [
                {
                    "borehole_id": "SYN-2",
                    "collar_x": 10,
                    "collar_y": 20,
                    "collar_z": 30.01,
                    "final_depth": 40,
                    "azimuth": 0,
                    "dip": -90,
                }
            ]
        ),
        "synthetic-overlap.csv",
    )
    _click_preview(window, qtbot)
    label = window._plotter._actors_by_name["borehole:collar_labels"]
    assert label.labels == ["SYN-1", "SYN-2"]
    assert len(label.points) == 2
    assert label.declutter_disabled
    assert len(window._borehole_display_manager.actor_metadata) == 2


def test_preview_adds_one_independent_label_per_pz_point(qtbot) -> None:
    window = _window(qtbot)
    project = window._project_store.current_project
    BoreholeRepository(project).import_dataframe(
        "orientation_points",
        pd.DataFrame(
            [
                {
                    "observation_id": "P-SITE-A",
                    "point_id": "P-SITE",
                    "x": 101.0,
                    "y": 202.0,
                    "z": 303.0,
                    "dip": 45.0,
                    "dip_direction": 120.0,
                    "local_set_id": "A",
                    "joint_spacing_m": 0.5,
                    "joint_num": 3,
                },
                {
                    "observation_id": "P-SITE-B",
                    "point_id": "P-SITE",
                    "x": 101.0,
                    "y": 202.0,
                    "z": 303.0,
                    "dip": 60.0,
                    "dip_direction": 220.0,
                    "local_set_id": "B",
                    "joint_spacing_m": 0.8,
                    "joint_num": 2,
                },
                {
                    "observation_id": "Z-SITE-A",
                    "point_id": "Z-SITE",
                    "x": 111.0,
                    "y": 212.0,
                    "z": 313.0,
                    "dip": 35.0,
                    "dip_direction": 80.0,
                    "local_set_id": "A",
                },
                {
                    "observation_id": "Z-SITE-B",
                    "point_id": "Z-SITE",
                    "x": 111.0,
                    "y": 212.0,
                    "z": 313.0,
                    "dip": 75.0,
                    "dip_direction": 310.0,
                    "local_set_id": "B",
                },
            ]
        ),
        "synthetic-pz-points.csv",
    )
    assert "observation_point:labels" not in window._plotter._actors_by_name
    assert not window._point_labels_action.isEnabled()

    dialog = _click_preview(window, qtbot)
    manager = window._borehole_display_manager
    point_labels = window._plotter._actors_by_name[manager.POINT_LABEL_ACTOR]
    assert point_labels.labels == ["P-SITE", "Z-SITE"]
    assert len(point_labels.points) == 2
    assert point_labels.points[0][:2] == [101.0, 202.0]
    assert 303.0 < point_labels.points[0][2] < 304.0
    assert point_labels.points[1][:2] == [111.0, 212.0]
    assert 313.0 < point_labels.points[1][2] < 314.0
    assert not point_labels.GetPickable()
    assert window._point_labels_action.isEnabled()
    collar_label = window._plotter._actors_by_name[manager.LABEL_ACTOR]
    window._point_labels_action.setChecked(False)
    assert not point_labels.GetVisibility()
    assert collar_label.GetVisibility()
    qtbot.mouseClick(dialog._preview_button, Qt.MouseButton.LeftButton)
    replacement = window._plotter._actors_by_name[manager.POINT_LABEL_ACTOR]
    assert replacement.labels == ["P-SITE", "Z-SITE"]
    assert not replacement.GetVisibility()
    assert len(window._plotter._observers) == 1
    window._on_new_project()
    assert manager.POINT_LABEL_ACTOR not in window._plotter._actors_by_name
    assert not window._point_labels_action.isEnabled()


def test_native_label_mapper_draws_every_label_without_depth_or_collision_filter(qtbot) -> None:
    from vtkmodules.vtkRenderingCore import vtkActor

    window = _window(qtbot)
    project = window._project_store.current_project
    BoreholeRepository(project).import_dataframe(
        "collars",
        pd.DataFrame(
            [
                {
                    "borehole_id": "SYN-2",
                    "collar_x": 10,
                    "collar_y": 20,
                    "collar_z": 30.01,
                    "final_depth": 40,
                    "azimuth": 0,
                    "dip": -90,
                }
            ]
        ),
        "synthetic-overlap-native.csv",
    )

    class Renderer:
        def __init__(self):
            self.actors = {"trajectory:SYN-1": vtkActor(), "trajectory:SYN-2": vtkActor()}

        def add_actor(self, actor, *, name, **_kwargs):
            self.actors[name] = actor

    class Plotter:
        def __init__(self):
            self.renderer = Renderer()
            self._observer = None
            self.iren = SimpleNamespace(interactor=self)

        def AddObserver(self, _event, callback):
            self._observer = callback
            return 1

        def RemoveObserver(self, _observer_id):
            self._observer = None

        def remove_actor(self, name, **_kwargs):
            self.renderer.actors.pop(name, None)

        def render(self):
            return None

    manager = BoreholeDisplayManager(Plotter())
    manager.attach(
        project.borehole_collection,
        {"SYN-1": "trajectory:SYN-1", "SYN-2": "trajectory:SYN-2"},
    )
    label_actor = manager.plotter.renderer.actors[manager.LABEL_ACTOR]
    assert label_actor.GetMapper().GetClassName() == "vtkLabeledDataMapper"
    assert label_actor.GetMapper().GetInput().GetNumberOfPoints() == 2
    assert not label_actor.GetPickable()
    assert manager._picker.GetPickList().GetNumberOfItems() == 2


def test_database_change_discards_preview_until_next_user_preview(qtbot) -> None:
    window = _window(qtbot)
    project = window._project_store.current_project
    _click_preview(window, qtbot)
    assert "borehole:collar_labels" in window._plotter._actors_by_name
    BoreholeRepository(project).import_dataframe(
        "collars",
        pd.DataFrame(
            [
                {
                    "borehole_id": "SYN-2",
                    "collar_x": 11,
                    "collar_y": 21,
                    "collar_z": 31,
                    "final_depth": 40,
                    "azimuth": 0,
                    "dip": -90,
                }
            ]
        ),
        "synthetic-update.csv",
    )
    window._on_database_changed({"collars"})
    assert "borehole:collar_labels" not in window._plotter._actors_by_name
    assert "trajectory:SYN-1" not in window._plotter._actors_by_name
    assert window._plotter._observers == {}
    assert not window._borehole_labels_action.isEnabled()
    before = project.model_dump(mode="json")
    workflow_before = window._workflow.to_dict()
    _click_preview(window, qtbot)
    label = window._plotter._actors_by_name["borehole:collar_labels"]
    assert label.labels == ["SYN-1", "SYN-2"]
    assert len(window._plotter._observers) == 1
    assert project.model_dump(mode="json") == before
    assert window._workflow.to_dict() == workflow_before


def test_failed_preview_leaves_no_labels_picker_or_observer(monkeypatch, qtbot) -> None:
    window = _window(qtbot)
    _click_preview(window, qtbot)

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("synthetic preview failure")

    monkeypatch.setattr(window._plotter, "add_mesh", fail_render)
    project = window._project_store.current_project
    bounds = ModelBounds(x_min=0, x_max=20, y_min=10, y_max=30, z_min=-30, z_max=40)
    with pytest.raises(RuntimeError, match="synthetic preview failure"):
        M8SpatialPreviewRenderer.render(
            window._plotter,
            bounds,
            bounds,
            VoxelConfig(cell_size_x=5, cell_size_y=5, cell_size_z=5),
            project.borehole_collection,
            borehole_display_manager=window._borehole_display_manager,
        )
    assert "borehole:collar_labels" not in window._plotter._actors_by_name
    assert window._plotter._observers == {}
    assert not window._borehole_display_manager.preview_active
    assert not window._borehole_labels_action.isEnabled()


def test_hover_overlay_is_model_derived_throttled_and_hides_on_leave(qtbot) -> None:
    window = _window(qtbot)
    _click_preview(window, qtbot)
    manager = window._borehole_display_manager
    actor = window._plotter._actors_by_name["trajectory:SYN-1"]
    realization_renderer = BoreholeFractureRenderer(window._plotter)
    realization_renderer.render(
        SimpleNamespace(
            realization_id="synthetic-hover",
            arrays={
                "xyz_offset": np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32),
                "component_type": np.asarray([BoreholeFractureComponent.DOMINANT_SET], dtype=np.uint8),
            },
            provenance={"xyz_origin": (0.0, 0.0, 0.0)},
        )
    )
    fracture_actor = next(iter(realization_renderer._actors.values()))
    assert not fracture_actor.GetPickable()
    assert manager._pick_actors == [actor]

    class Picker:
        selected = actor

        def Pick(self, *_args):
            return 1

        def GetActor(self):
            return self.selected

    picker = Picker()
    manager._picker = picker
    caller = SimpleNamespace(GetEventPosition=lambda: (10, 20))
    unassigned = BoreholeActorMetadata("SYN-X", 1, 2, 3, 4, 5, -90, None)
    assert "Domain: Unassigned" in manager._format_hover(unassigned)
    assert "Inclination: -90.00 degree" in manager._format_hover(unassigned)
    manager._last_pick_at = 0.0
    manager._on_mouse_move(caller, "MouseMoveEvent")
    assert "borehole:hover_info" in window._plotter._actors_by_name
    picker.selected = None
    manager._last_pick_at = 0.0
    manager._on_mouse_move(caller, "MouseMoveEvent")
    assert "borehole:hover_info" not in window._plotter._actors_by_name


def test_orientation_point_and_derived_summary_are_visible_in_database_panel(qtbot) -> None:
    window = _window(qtbot)
    project = window._project_store.current_project
    BoreholeRepository(project).import_dataframe(
        "orientation_points",
        pd.DataFrame(
            [
                {
                    "observation_id": "P001-A",
                    "point_id": "001",
                    "x": 1,
                    "y": 2,
                    "z": 3,
                    "dip": 45,
                    "dip_direction": 120,
                    "local_set_id": "A",
                    "joint_spacing_m": 0.5,
                    "joint_num": 2,
                }
            ]
        ),
        "synthetic-points.csv",
    )
    panel = BoreholeDatabasePanel(project)
    qtbot.addWidget(panel)
    panel._type.setCurrentIndex(panel._type.findData("orientation_points"))
    panel.refresh()
    assert panel._table.rowCount() == 1
    assert not panel._orientation_summary.isHidden()
    assert panel._orientation_summary.rowCount() == 1
    assert panel._orientation_summary.item(0, 0).text() == "POINT_CLOUD:001"
