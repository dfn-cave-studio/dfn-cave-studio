"""Session-only registry for M11 second-voxelization slice actors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

M11_VOXEL_PREFIX = "m11_voxel:"
M11_SCALAR_BAR_PREFIX = "m11_scalar_bar:"


@dataclass(slots=True)
class M11LayerRecord:
    """Actor and display metadata for one M11 voxel slice."""

    layer_id: str
    actor_name: str
    actor: Any
    scalar_bar_id: str
    scalar_bar_actor: Any
    scalar_bar_title: str
    realization_id: str
    display_mode: str
    interpolation_mode: str
    field_name: str
    joint_set_id: int | None
    axis: str
    slice_index: int
    coordinate: float
    plane_id: str
    show_grid_lines: bool
    color_range: tuple[float, float]
    widgets: tuple[Any, ...]
    widget_observer_ids: tuple[tuple[Any, int], ...]
    visible: bool
    opacity: float
    created_at: datetime
    pending: bool = False


class M11LayerManager:
    """Own only actors in the ``m11_voxel:`` namespace."""

    def __init__(self, plotter: Any) -> None:
        self.plotter = plotter
        self._layers: dict[str, M11LayerRecord] = {}
        self._control_widgets: dict[str, Any] = {}

    @staticmethod
    def layer_id(
        realization_id: str,
        field_name: str,
        joint_set_id: int | None,
        axis: str,
        slice_index: int,
        *,
        display_mode: str = "orthogonal_section",
        interpolation_mode: str = "exact",
        plane_id: str = "",
    ) -> str:
        """Build the stable identity for one scientific result slice."""
        set_token = "all" if joint_set_id is None else str(int(joint_set_id))
        location = plane_id or f"{axis.lower()}:{int(slice_index)}"
        return f"{M11_VOXEL_PREFIX}{realization_id}:{display_mode}:{field_name}:set_{set_token}:{location}:{interpolation_mode}"

    @staticmethod
    def scalar_bar_id(
        realization_id: str,
        field_name: str,
        joint_set_id: int | None,
        color_signature: str = "global:continuous",
    ) -> str:
        """Build an internal M11-only scalar-bar registry key."""
        set_token = "all" if joint_set_id is None else str(int(joint_set_id))
        return f"{M11_SCALAR_BAR_PREFIX}{realization_id}:{field_name}:set_{set_token}:{color_signature}"

    def create_or_get_scalar_bar(
        self,
        scalar_bar_id: str,
        scalar_bar_title: str,
        mesh_actor: Any,
        *,
        attach_mapper: bool,
    ) -> Any:
        """Create or share one M11-owned scalar bar using its private key."""
        if not scalar_bar_id.startswith(M11_SCALAR_BAR_PREFIX):
            raise ValueError("M11 scalar-bar IDs must use the m11_scalar_bar namespace")
        existing = self._get_scalar_bar_actor(scalar_bar_id)
        if existing is not None and not attach_mapper:
            return existing
        mapper = getattr(mesh_actor, "mapper", None)
        if mapper is None:
            getter = getattr(mesh_actor, "GetMapper", None)
            mapper = getter() if getter is not None else None
        if mapper is None:
            raise ValueError("Cannot create an M11 scalar bar without a mesh mapper")
        creator = getattr(self.plotter, "add_scalar_bar", None)
        if creator is None:
            raise ValueError("The active plotter does not support scalar bars")
        created = creator(title=scalar_bar_id, mapper=mapper, render=False)
        scalar_bar_actor = created if created is not None else self._get_scalar_bar_actor(scalar_bar_id)
        if scalar_bar_actor is None:
            raise RuntimeError("The M11 scalar bar was not registered by the plotter")
        title_setter = getattr(scalar_bar_actor, "SetTitle", None)
        if title_setter is not None:
            title_setter(str(scalar_bar_title))
        return scalar_bar_actor

    def add_or_replace(
        self,
        layer_id: str,
        actor: Any,
        *,
        scalar_bar_id: str,
        scalar_bar_actor: Any,
        scalar_bar_title: str,
        realization_id: str,
        display_mode: str = "orthogonal_section",
        interpolation_mode: str = "exact",
        field_name: str,
        joint_set_id: int | None,
        axis: str,
        slice_index: int,
        coordinate: float,
        plane_id: str = "",
        show_grid_lines: bool = False,
        color_range: tuple[float, float] = (0.0, 1.0),
        widgets: tuple[Any, ...] = (),
        widget_observer_ids: tuple[tuple[Any, int], ...] = (),
        opacity: float,
        pending: bool = False,
    ) -> M11LayerRecord:
        """Register a successful render and replace the same logical slice."""
        if not layer_id.startswith(M11_VOXEL_PREFIX):
            raise ValueError("M11 layer IDs must use the m11_voxel namespace")
        if actor is None:
            raise ValueError("Cannot register a missing M11 slice actor")
        if not scalar_bar_id.startswith(M11_SCALAR_BAR_PREFIX) or scalar_bar_actor is None:
            raise ValueError("Cannot register an M11 slice without its owned scalar bar")
        value = self._validate_opacity(opacity)
        previous = self._layers.get(layer_id)
        effective_widgets = tuple(widgets)
        effective_observer_ids = tuple(widget_observer_ids)
        if previous is not None and previous.actor is not actor:
            self._remove_actor(previous.actor)
            if not effective_widgets and previous.widgets and plane_id.startswith("slot_"):
                effective_widgets = previous.widgets
                effective_observer_ids = previous.widget_observer_ids
            else:
                self._remove_widgets(previous.widgets, previous.widget_observer_ids)
        record = M11LayerRecord(
            layer_id=layer_id,
            actor_name=layer_id,
            actor=actor,
            scalar_bar_id=scalar_bar_id,
            scalar_bar_actor=scalar_bar_actor,
            scalar_bar_title=str(scalar_bar_title),
            realization_id=str(realization_id),
            display_mode=str(display_mode),
            interpolation_mode=str(interpolation_mode),
            field_name=str(field_name),
            joint_set_id=None if joint_set_id is None else int(joint_set_id),
            axis=axis.lower(),
            slice_index=int(slice_index),
            coordinate=float(coordinate),
            plane_id=str(plane_id),
            show_grid_lines=bool(show_grid_lines),
            color_range=(float(color_range[0]), float(color_range[1])),
            widgets=effective_widgets,
            widget_observer_ids=effective_observer_ids,
            visible=True,
            opacity=value,
            created_at=datetime.now(UTC),
            pending=bool(pending),
        )
        self._layers[layer_id] = record
        if previous is not None and previous.scalar_bar_id != scalar_bar_id:
            self._remove_scalar_bar_if_unused(previous.scalar_bar_id)
        self._set_actor_visibility(actor, True)
        self._set_actor_opacity(actor, value)
        self._sync_scalar_bar_visibility(scalar_bar_id)
        return record

    def contains(self, layer_id: str) -> bool:
        """Return whether a logical slice is registered."""
        return layer_id in self._layers

    def get(self, layer_id: str) -> M11LayerRecord | None:
        """Return one layer record, if present."""
        return self._layers.get(layer_id)

    def list_layers(self) -> list[M11LayerRecord]:
        """List layers in stable creation/name order."""
        return sorted(self._layers.values(), key=lambda item: (item.created_at, item.layer_id))

    def set_visible(self, layer_id: str, visible: bool) -> bool:
        """Hide or show one actor without discarding its metadata."""
        record = self._layers.get(layer_id)
        if record is None:
            return False
        record.visible = bool(visible)
        self._set_actor_visibility(record.actor, record.visible)
        self._set_widgets_visible(record.widgets, record.visible)
        self._sync_scalar_bar_visibility(record.scalar_bar_id)
        self._render()
        return True

    def set_opacity(self, layer_id: str, opacity: float) -> bool:
        """Change one actor opacity without touching scientific state."""
        record = self._layers.get(layer_id)
        if record is None:
            return False
        record.opacity = self._validate_opacity(opacity)
        self._set_actor_opacity(record.actor, record.opacity)
        self._render()
        return True

    def remove(self, layer_id: str) -> bool:
        """Remove exactly one registered M11 actor."""
        record = self._layers.pop(layer_id, None)
        if record is None:
            return False
        self._remove_actor(record.actor)
        self._remove_widgets(record.widgets, record.widget_observer_ids)
        self._remove_scalar_bar_if_unused(record.scalar_bar_id)
        self._render()
        return True

    def clear_current(
        self,
        realization_id: str,
        field_name: str,
        joint_set_id: int | None,
        axis: str,
        slice_index: int,
    ) -> bool:
        """Remove the exact slice described by the current UI controls."""
        return self.remove(self.layer_id(realization_id, field_name, joint_set_id, axis, slice_index))

    def clear_pending(self, realization_id: str | None = None) -> int:
        """Remove only actors rendered from uncommitted calculation results."""
        ids = [
            record.layer_id
            for record in self._layers.values()
            if record.pending and (realization_id is None or record.realization_id == realization_id)
        ]
        return self._remove_many(ids)

    def mark_committed(self, realization_id: str) -> None:
        """Retain rendered actors after their result is committed."""
        for record in self._layers.values():
            if record.realization_id == realization_id:
                record.pending = False

    def clear_m11_layers(self) -> int:
        """Remove registered M11 actors only; never clear the plotter."""
        removed = self._remove_many(list(self._layers))
        self.clear_control_widgets()
        return removed

    def set_widgets_visible(self, layer_id: str, visible: bool) -> bool:
        """Show or hide only the interaction handles owned by one M11 layer."""
        record = self._layers.get(layer_id)
        if record is None:
            return False
        self._set_widgets_visible(record.widgets, visible)
        self._render()
        return True

    def set_widget_plane(
        self,
        layer_id: str,
        origin: tuple[float, float, float],
        normal: tuple[float, float, float] | None = None,
    ) -> bool:
        """Move an existing live plane widget without replacing its observers."""
        record = self._layers.get(layer_id)
        if record is None or not record.widgets:
            return False
        for widget in record.widgets:
            set_origin = getattr(widget, "SetOrigin", None)
            if set_origin is not None:
                set_origin(*origin)
            if normal is not None:
                set_normal = getattr(widget, "SetNormal", None)
                if set_normal is not None:
                    set_normal(*normal)
            modified = getattr(widget, "Modified", None)
            if modified is not None:
                modified()
        self._render()
        return True

    def set_widget_bounds(
        self,
        layer_id: str,
        bounds: tuple[float, float, float, float, float, float],
    ) -> bool:
        """Move an existing live box widget without replacing its observers."""
        record = self._layers.get(layer_id)
        if record is None or not record.widgets:
            return False
        for widget in record.widgets:
            place = getattr(widget, "PlaceWidget", None)
            if place is not None:
                try:
                    place(*bounds)
                except TypeError:
                    place(bounds)
            modified = getattr(widget, "Modified", None)
            if modified is not None:
                modified()
        self._render()
        return True

    def add_or_replace_control_widget(self, widget_id: str, widget: Any) -> None:
        """Register a dock-level M11 widget without touching other module widgets."""
        previous = self._control_widgets.get(widget_id)
        if previous is not None and previous is not widget:
            self._remove_widgets((previous,))
        self._control_widgets[widget_id] = widget

    def remove_control_widget(self, widget_id: str) -> bool:
        """Remove one M11 dock control widget."""
        widget = self._control_widgets.pop(widget_id, None)
        if widget is None:
            return False
        self._remove_widgets((widget,))
        self._render()
        return True

    def clear_control_widgets(self) -> int:
        """Remove all dock-level widgets owned by M11 only."""
        widgets = tuple(self._control_widgets.values())
        self._control_widgets.clear()
        self._remove_widgets(widgets)
        if widgets:
            self._render()
        return len(widgets)

    def control_widget_count(self) -> int:
        """Return the number of M11 dock-level interaction widgets."""
        return len(self._control_widgets)

    def clear_widgets(self) -> int:
        """Disable only M11-owned plane widgets while retaining rendered actors."""
        count = 0
        for record in self._layers.values():
            count += len(record.widgets)
            self._remove_widgets(record.widgets, record.widget_observer_ids)
            record.widgets = ()
            record.widget_observer_ids = ()
        if count:
            self._render()
        return count

    def _remove_many(self, layer_ids: list[str]) -> int:
        records = [self._layers.pop(layer_id) for layer_id in layer_ids if layer_id in self._layers]
        scalar_bar_ids = {record.scalar_bar_id for record in records}
        for record in records:
            self._remove_actor(record.actor)
            self._remove_widgets(record.widgets, record.widget_observer_ids)
        for scalar_bar_id in scalar_bar_ids:
            self._remove_scalar_bar_if_unused(scalar_bar_id)
        if records:
            self._render()
        return len(records)

    def discard_unregistered(
        self,
        mesh_actor: Any,
        scalar_bar_id: str,
        widgets: tuple[Any, ...] = (),
        widget_observer_ids: tuple[tuple[Any, int], ...] = (),
    ) -> None:
        """Roll back display objects created before a failed registration."""
        self._remove_actor(mesh_actor)
        self._remove_widgets(widgets, widget_observer_ids)
        self._remove_scalar_bar_if_unused(scalar_bar_id)
        self._render()

    def _remove_scalar_bar_if_unused(self, scalar_bar_id: str) -> bool:
        if any(record.scalar_bar_id == scalar_bar_id for record in self._layers.values()):
            self._sync_scalar_bar_visibility(scalar_bar_id)
            return False
        if not scalar_bar_id.startswith(M11_SCALAR_BAR_PREFIX):
            return False
        actor = self._get_scalar_bar_actor(scalar_bar_id)
        if actor is None:
            return False
        remover = getattr(self.plotter, "remove_scalar_bar", None)
        if remover is not None:
            remover(title=scalar_bar_id, render=False)
        else:
            self._remove_actor(actor)
        return True

    def _sync_scalar_bar_visibility(self, scalar_bar_id: str) -> None:
        actor = self._get_scalar_bar_actor(scalar_bar_id)
        if actor is None:
            return
        visible = any(
            record.visible for record in self._layers.values() if record.scalar_bar_id == scalar_bar_id
        )
        self._set_actor_visibility(actor, visible)

    def _get_scalar_bar_actor(self, scalar_bar_id: str) -> Any | None:
        scalar_bars = getattr(self.plotter, "scalar_bars", None)
        if scalar_bars is None:
            return None
        try:
            return scalar_bars[scalar_bar_id]
        except (KeyError, TypeError):
            return None

    def _remove_actor(self, actor: Any) -> None:
        remover = getattr(self.plotter, "remove_actor", None)
        if remover is not None:
            remover(actor, render=False)

    @staticmethod
    def _remove_widgets(
        widgets: tuple[Any, ...],
        widget_observer_ids: tuple[tuple[Any, int], ...] = (),
    ) -> None:
        for widget, observer_id in widget_observer_ids:
            remove_observer = getattr(widget, "RemoveObserver", None)
            if remove_observer is not None:
                remove_observer(observer_id)
        for widget in widgets:
            off = getattr(widget, "Off", None)
            if off is not None:
                off()
            enabled = getattr(widget, "SetEnabled", None)
            if enabled is not None:
                enabled(False)
            remove_observers = getattr(widget, "RemoveAllObservers", None)
            if remove_observers is not None:
                remove_observers()

    @staticmethod
    def _set_widgets_visible(widgets: tuple[Any, ...], visible: bool) -> None:
        for widget in widgets:
            enabled = getattr(widget, "SetEnabled", None)
            if enabled is not None:
                enabled(bool(visible))

    def _render(self) -> None:
        render = getattr(self.plotter, "render", None)
        if render is not None:
            render()

    @staticmethod
    def _validate_opacity(opacity: float) -> float:
        value = float(opacity)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Layer opacity must be between 0 and 1")
        return value

    @staticmethod
    def _set_actor_visibility(actor: Any, visible: bool) -> None:
        setter = getattr(actor, "SetVisibility", None)
        if setter is not None:
            setter(bool(visible))

    @staticmethod
    def _set_actor_opacity(actor: Any, opacity: float) -> None:
        get_property = getattr(actor, "GetProperty", None)
        actor_property = get_property() if get_property is not None else getattr(actor, "prop", None)
        setter = getattr(actor_property, "SetOpacity", None)
        if setter is not None:
            setter(float(opacity))
