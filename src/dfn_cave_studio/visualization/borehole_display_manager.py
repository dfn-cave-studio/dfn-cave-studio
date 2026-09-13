"""Shared session-only controls layered onto existing borehole actors."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoreholeActorMetadata:
    """Stable, model-derived hover payload for one rendered borehole actor."""

    hole_id: str
    easting: float
    northing: float
    elevation: float
    total_depth: float
    azimuth: float
    inclination: float
    domain_id: int | None


class BoreholeDisplayManager:
    """Add labels, picking, and camera controls to existing borehole geometry."""

    LABEL_ACTOR = "borehole:collar_labels"
    POINT_LABEL_ACTOR = "observation_point:labels"
    HOVER_ACTOR = "borehole:hover_info"
    HOVER_INTERVAL_SECONDS = 0.05

    def __init__(self, plotter: Any, on_preview_state_changed: Callable[[bool], None] | None = None):
        self.plotter = plotter
        self.labels_visible = True
        self.point_labels_visible = True
        self.preview_active = False
        self._on_preview_state_changed = on_preview_state_changed
        self._label_actor: Any | None = None
        self._label_pipeline: tuple[Any, ...] | None = None
        self._point_label_actor: Any | None = None
        self._point_label_pipeline: tuple[Any, ...] | None = None
        self._point_label_anchors: dict[str, tuple[float, float, float]] = {}
        self._hover_actor: Any | None = None
        self._actor_metadata: dict[str, BoreholeActorMetadata] = {}
        self._pick_actors: list[Any] = []
        self._label_anchors: dict[str, tuple[float, float, float]] = {}
        self._picker: Any | None = None
        self._interactor: Any | None = None
        self._observer_id: int | None = None
        self._hovered_actor_id: str | None = None
        self._last_pick_at = 0.0
        self._model_bounds: tuple[float, float, float, float, float, float] | None = None

    @property
    def actor_metadata(self) -> dict[str, BoreholeActorMetadata]:
        """Return a copy of the stable actor metadata mapping."""
        return dict(self._actor_metadata)

    @property
    def has_point_labels(self) -> bool:
        """Return whether the active preview owns point-observation labels."""
        return self._point_label_actor is not None

    def attach(
        self,
        collection: Any,
        actor_names: dict[str, str],
        domain_by_hole: dict[str, int | None] | None = None,
        model_bounds: Any | None = None,
        point_labels: list[tuple[str, str, tuple[float, float, float]]] | None = None,
    ) -> None:
        """Attach controls to borehole actors already created by an existing renderer."""
        self.clear()
        collar_points: list[list[float]] = []
        collar_labels: list[str] = []
        extent_points: list[np.ndarray] = []
        domains = domain_by_hole or {}
        actors = getattr(getattr(self.plotter, "renderer", None), "actors", {})
        offset = self._label_offset(collection)
        for borehole in collection:
            hole_id = str(borehole.borehole_id)
            collar = borehole.collar
            metadata = BoreholeActorMetadata(
                hole_id=hole_id,
                easting=float(collar.collar_x),
                northing=float(collar.collar_y),
                elevation=float(collar.collar_z),
                total_depth=float(collar.final_depth),
                azimuth=float(collar.azimuth),
                inclination=float(collar.dip),
                domain_id=domains.get(hole_id),
            )
            actor = actors.get(actor_names.get(hole_id, "")) if hasattr(actors, "get") else None
            if actor is not None:
                if hasattr(actor, "SetPickable"):
                    actor.SetPickable(True)
                self._actor_metadata[self._actor_key(actor)] = metadata
                self._pick_actors.append(actor)
                anchor = (metadata.easting, metadata.northing, metadata.elevation + offset)
                self._label_anchors[hole_id] = anchor
                collar_points.append(list(anchor))
                collar_labels.append(hole_id)
            points, _ = borehole.compute_trajectory(step_length=5.0)
            if len(points):
                extent_points.append(np.asarray(points, dtype=float))
        self._model_bounds = (
            tuple(float(value) for value in model_bounds.to_array())
            if model_bounds is not None
            else self._bounds_from_points(extent_points)
        )
        if collar_points:
            self._label_actor, self._label_pipeline = self._add_all_labels(
                collar_points, collar_labels, self.LABEL_ACTOR, "borehole_name"
            )
            if hasattr(self._label_actor, "SetPickable"):
                self._label_actor.SetPickable(False)
            self._label_actor.SetVisibility(self.labels_visible)
        unique_point_labels: dict[str, tuple[str, tuple[float, float, float]]] = {}
        for point_key, point_id, coordinates in point_labels or []:
            unique_point_labels.setdefault(
                str(point_key),
                (str(point_id), tuple(float(value) for value in coordinates)),
            )
        if unique_point_labels:
            point_offset = self._point_label_offset(unique_point_labels.values())
            points: list[list[float]] = []
            labels: list[str] = []
            for point_key, (point_id, coordinates) in unique_point_labels.items():
                anchor = (coordinates[0], coordinates[1], coordinates[2] + point_offset)
                self._point_label_anchors[point_key] = anchor
                points.append(list(anchor))
                labels.append(point_id)
            self._point_label_actor, self._point_label_pipeline = self._add_all_labels(
                points, labels, self.POINT_LABEL_ACTOR, "point_name"
            )
            if hasattr(self._point_label_actor, "SetPickable"):
                self._point_label_actor.SetPickable(False)
            self._point_label_actor.SetVisibility(self.point_labels_visible)
        if self._actor_metadata:
            self._install_hover_observer()
        self.preview_active = bool(self._actor_metadata or self._point_label_actor is not None)
        self._notify_preview_state()
        self._render_once()

    def set_labels_visible(self, visible: bool) -> None:
        """Hide labels only; existing borehole geometry remains visible."""
        self.labels_visible = bool(visible)
        if self._label_actor is not None:
            self._label_actor.SetVisibility(self.labels_visible)
            self._render_once()

    def set_point_labels_visible(self, visible: bool) -> None:
        """Hide point labels only; observation points and boreholes remain visible."""
        self.point_labels_visible = bool(visible)
        if self._point_label_actor is not None:
            self._point_label_actor.SetVisibility(self.point_labels_visible)
            self._render_once()

    def set_parallel_projection(self, enabled: bool) -> None:
        """Toggle projection while preserving pose and approximate screen framing."""
        camera = getattr(self.plotter, "camera", None)
        camera_position = getattr(self.plotter, "camera_position", None)
        methods = ("GetPosition", "GetFocalPoint", "GetViewUp", "GetViewAngle", "GetParallelScale")
        if camera is not None and all(hasattr(camera, method) for method in methods):
            position = tuple(camera.GetPosition())
            focal = tuple(camera.GetFocalPoint())
            view_up = tuple(camera.GetViewUp())
            distance = max(math.dist(position, focal), 1e-12)
            if enabled:
                camera.SetParallelScale(max(distance * math.tan(math.radians(camera.GetViewAngle()) / 2.0), 1e-12))
                self.plotter.enable_parallel_projection()
            else:
                angle = math.degrees(2.0 * math.atan(camera.GetParallelScale() / distance))
                self.plotter.disable_parallel_projection()
                camera.SetViewAngle(min(179.0, max(1e-6, angle)))
            camera.SetPosition(position)
            camera.SetFocalPoint(focal)
            camera.SetViewUp(view_up)
        else:
            method = "enable_parallel_projection" if enabled else "disable_parallel_projection"
            getattr(self.plotter, method)()
            if camera_position is not None:
                self.plotter.camera_position = camera_position
        self._render_once()

    def set_view(self, view: str) -> None:
        """Fit one explicit E/N/R view without changing projection mode."""
        bounds = self._model_bounds
        if bounds is None:
            return
        center = np.asarray(
            ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2),
            dtype=float,
        )
        if view == "current":
            self._fit(bounds)
            return
        directions = {
            "top": (np.asarray((0.0, 0.0, -1.0)), np.asarray((0.0, 1.0, 0.0))),
            "front": (np.asarray((0.0, -1.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
            "side": (np.asarray((-1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
            "isometric": (np.asarray((-1.0, -1.0, -1.0)) / math.sqrt(3.0), np.asarray((0.0, 0.0, 1.0))),
        }
        if view not in directions:
            raise ValueError(f"Unknown borehole camera view: {view}")
        look_direction, view_up = directions[view]
        extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1.0)
        position = center - look_direction * (2.5 * extent)
        self.plotter.camera_position = [tuple(position), tuple(center), tuple(view_up)]
        self._fit(bounds)

    def clear(self) -> None:
        """Remove only owned labels, tooltip, metadata, and hover observer."""
        if self._interactor is not None and self._observer_id is not None:
            try:
                self._interactor.RemoveObserver(self._observer_id)
            except (AttributeError, RuntimeError, TypeError):
                pass
        self._observer_id = None
        self._interactor = None
        self._picker = None
        for name in (self.LABEL_ACTOR, self.POINT_LABEL_ACTOR, self.HOVER_ACTOR):
            try:
                self.plotter.remove_actor(name, render=False)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                continue
        self._actor_metadata.clear()
        self._pick_actors.clear()
        self._label_anchors.clear()
        self._point_label_anchors.clear()
        self._label_actor = None
        self._label_pipeline = None
        self._point_label_actor = None
        self._point_label_pipeline = None
        self._hover_actor = None
        self._hovered_actor_id = None
        self._model_bounds = None
        self.preview_active = False
        self._notify_preview_state()
        self._render_once()

    def reset_for_project(self) -> None:
        """Clear the prior preview and restore the next preview's default visibility."""
        self.clear()
        self.labels_visible = True
        self.point_labels_visible = True

    def _install_hover_observer(self) -> None:
        try:
            from vtkmodules.vtkRenderingCore import vtkPropPicker

            wrapper = getattr(self.plotter, "iren", None)
            interactor = getattr(wrapper, "interactor", wrapper)
            if interactor is None or not hasattr(interactor, "AddObserver"):
                return
            self._picker = vtkPropPicker()
            if hasattr(self._picker, "PickFromListOn"):
                self._picker.PickFromListOn()
            if hasattr(self._picker, "AddPickList"):
                for actor in self._pick_actors:
                    try:
                        self._picker.AddPickList(actor)
                    except TypeError:
                        # Headless GUI tests use lightweight actor doubles. Real
                        # VTK/PyVista borehole actors are vtkProp instances.
                        continue
            self._interactor = interactor
            self._observer_id = interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
        except (AttributeError, ImportError, RuntimeError):
            self._picker = None
            self._interactor = None
            self._observer_id = None

    def _add_all_labels(
        self,
        points: list[list[float]],
        labels: list[str],
        actor_name: str,
        field_name: str,
    ) -> tuple[Any, tuple[Any, ...] | None]:
        """Create fixed-size overlay labels without hierarchy, occlusion, or decluttering."""
        renderer = getattr(self.plotter, "renderer", None)
        if renderer is not None and hasattr(renderer, "add_actor"):
            from vtkmodules.vtkCommonCore import vtkPoints, vtkStringArray
            from vtkmodules.vtkCommonDataModel import vtkPolyData
            from vtkmodules.vtkRenderingCore import vtkActor2D
            from vtkmodules.vtkRenderingLabel import vtkLabeledDataMapper

            vtk_points = vtkPoints()
            label_values = vtkStringArray()
            label_values.SetName(field_name)
            for point, label in zip(points, labels, strict=True):
                vtk_points.InsertNextPoint(*point)
                label_values.InsertNextValue(label)
            data = vtkPolyData()
            data.SetPoints(vtk_points)
            data.GetPointData().AddArray(label_values)
            mapper = vtkLabeledDataMapper()
            mapper.SetInputData(data)
            mapper.SetLabelModeToLabelFieldData()
            mapper.SetFieldDataName(field_name)
            text = mapper.GetLabelTextProperty()
            text.SetFontSize(12)
            text.SetBold(True)
            actor = vtkActor2D()
            actor.SetMapper(mapper)
            actor.SetPickable(False)
            renderer.add_actor(actor, name=actor_name, pickable=False, render=False)
            return actor, (vtk_points, label_values, data, mapper)
        actor = self.plotter.add_point_labels(
            points,
            labels,
            name=actor_name,
            font_size=12,
            point_size=0,
            always_visible=True,
            shape=None,
            pickable=False,
            render=False,
        )
        actor.declutter_disabled = True
        return actor, None

    def _notify_preview_state(self) -> None:
        if self._on_preview_state_changed is not None:
            self._on_preview_state_changed(self.preview_active)

    def _on_mouse_move(self, caller: Any, _event: str) -> None:
        now = time.monotonic()
        if self._picker is None or now - self._last_pick_at < self.HOVER_INTERVAL_SECONDS:
            return
        self._last_pick_at = now
        x, y = caller.GetEventPosition()
        self._picker.Pick(x, y, 0, getattr(self.plotter, "renderer", None))
        actor_key = self._actor_key(self._picker.GetActor())
        metadata = self._actor_metadata.get(actor_key)
        if actor_key == self._hovered_actor_id and metadata is not None:
            return
        self._hovered_actor_id = actor_key if metadata is not None else None
        self._hide_hover()
        if metadata is not None and hasattr(self.plotter, "add_text"):
            self._hover_actor = self.plotter.add_text(
                self._format_hover(metadata), name=self.HOVER_ACTOR, position="upper_right", font_size=10
            )
        self._render_once()

    def _hide_hover(self) -> None:
        try:
            self.plotter.remove_actor(self.HOVER_ACTOR, render=False)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            pass
        self._hover_actor = None

    @staticmethod
    def _format_hover(metadata: BoreholeActorMetadata) -> str:
        domain = str(metadata.domain_id) if metadata.domain_id is not None else "Unassigned"
        return (
            f"{metadata.hole_id}\n"
            f"E/N/R: {metadata.easting:.2f}, {metadata.northing:.2f}, {metadata.elevation:.2f}\n"
            f"Total depth: {metadata.total_depth:.2f} m\n"
            f"Azimuth: {metadata.azimuth:.2f} degree\n"
            f"Inclination: {metadata.inclination:.2f} degree\n"
            f"Domain: {domain}"
        )

    @staticmethod
    def _actor_key(actor: Any | None) -> str:
        """Return a stable VTK identity even when Python creates a new wrapper."""
        if actor is None:
            return ""
        if hasattr(actor, "GetAddressAsString"):
            try:
                return str(actor.GetAddressAsString(""))
            except (RuntimeError, TypeError):
                pass
        return f"python:{id(actor)}"

    @staticmethod
    def _label_offset(collection: Any) -> float:
        depths = [float(item.collar.final_depth) for item in collection]
        return min(0.5, max(0.05, max(depths, default=1.0) * 0.002))

    @staticmethod
    def _point_label_offset(points: Any) -> float:
        coordinates = [np.asarray(value[1], dtype=float) for value in points]
        if not coordinates:
            return 0.05
        cloud = np.vstack(coordinates)
        span = float(np.max(np.ptp(cloud, axis=0))) if len(cloud) > 1 else 1.0
        return min(0.5, max(0.05, span * 0.002))

    @staticmethod
    def _bounds_from_points(groups: list[np.ndarray]) -> tuple[float, float, float, float, float, float] | None:
        if not groups:
            return None
        points = np.vstack(groups)
        low = np.min(points, axis=0)
        high = np.max(points, axis=0)
        return float(low[0]), float(high[0]), float(low[1]), float(high[1]), float(low[2]), float(high[2])

    def _fit(self, bounds: tuple[float, float, float, float, float, float]) -> None:
        try:
            self.plotter.reset_camera(bounds=bounds)
        except TypeError:
            self.plotter.reset_camera()
        self._render_once()

    def _render_once(self) -> None:
        try:
            self.plotter.render()
        except (AttributeError, RuntimeError):
            pass
