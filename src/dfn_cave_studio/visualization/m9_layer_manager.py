"""Session-only registry for rendered M9 parameter-field slices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

M9_SLICE_PREFIX = "m9_slice:"
M9_SCALAR_BAR_PREFIX = "m9_scalar_bar:"


@dataclass(slots=True)
class M9LayerRecord:
    """Metadata and actor reference for one rendered M9 slice."""

    layer_id: str
    actor_name: str
    actor: Any
    scalar_bar_id: str
    scalar_bar_actor: Any
    field_name: str
    axis: str
    slice_index: int
    coordinate: float
    visible: bool
    opacity: float
    created_at: datetime


class M9LayerManager:
    """Own only M9 slice actors without affecting other scene objects."""

    def __init__(self, plotter: Any) -> None:
        self.plotter = plotter
        self._layers: dict[str, M9LayerRecord] = {}

    def add_or_replace(
        self,
        layer_id: str,
        actor: Any,
        *,
        field_name: str,
        axis: str,
        slice_index: int,
        coordinate: float,
        opacity: float,
        scalar_bar_id: str = "",
        scalar_bar_actor: Any = None,
    ) -> M9LayerRecord:
        """Register a successfully rendered actor, replacing the same layer."""
        if not layer_id.startswith(M9_SLICE_PREFIX):
            raise ValueError("M9 slice layer IDs must use the m9_slice namespace")
        if actor is None:
            raise ValueError("Cannot register a missing slice actor")
        opacity = self._validate_opacity(opacity)
        old = self._layers.get(layer_id)
        if old is not None and old.actor is not actor:
            self._remove_actor(old.actor)
        record = M9LayerRecord(
            layer_id=layer_id,
            actor_name=layer_id,
            actor=actor,
            scalar_bar_id=str(scalar_bar_id),
            scalar_bar_actor=scalar_bar_actor,
            field_name=field_name,
            axis=axis.lower(),
            slice_index=int(slice_index),
            coordinate=float(coordinate),
            visible=True,
            opacity=opacity,
            created_at=datetime.now(UTC),
        )
        self._layers[layer_id] = record
        self._set_actor_visibility(actor, True)
        self._set_actor_opacity(actor, opacity)
        if scalar_bar_id:
            self._sync_scalar_bar_visibility(scalar_bar_id)
        return record

    @staticmethod
    def scalar_bar_id(field_name: str) -> str:
        """Return a stable internal M9-only scalar-bar identity."""
        return f"{M9_SCALAR_BAR_PREFIX}{field_name}"

    def create_or_get_scalar_bar(self, scalar_bar_id: str, visible_title: str, mesh_actor: Any) -> Any:
        """Create or reuse a private M9 scalar bar when supported by the plotter."""
        existing = self._get_scalar_bar_actor(scalar_bar_id)
        if existing is not None:
            return existing
        creator = getattr(self.plotter, "add_scalar_bar", None)
        if creator is None:
            return None
        mapper = getattr(mesh_actor, "mapper", None)
        if mapper is None:
            getter = getattr(mesh_actor, "GetMapper", None)
            mapper = getter() if getter is not None else None
        if mapper is None:
            return None
        created = creator(title=scalar_bar_id, mapper=mapper, render=False)
        actor = created if created is not None else self._get_scalar_bar_actor(scalar_bar_id)
        setter = getattr(actor, "SetTitle", None)
        if setter is not None:
            setter(str(visible_title))
        return actor

    def contains(self, layer_id: str) -> bool:
        """Return whether a layer is registered."""
        return layer_id in self._layers

    def get(self, layer_id: str) -> M9LayerRecord | None:
        """Return one registered layer."""
        return self._layers.get(layer_id)

    def list_layers(self) -> list[M9LayerRecord]:
        """Return layers in stable creation/name order."""
        return sorted(self._layers.values(), key=lambda item: (item.created_at, item.layer_id))

    def set_visible(self, layer_id: str, visible: bool) -> bool:
        """Change actor visibility while keeping its registry entry."""
        record = self._layers.get(layer_id)
        if record is None:
            return False
        record.visible = bool(visible)
        self._set_actor_visibility(record.actor, record.visible)
        if record.scalar_bar_id:
            self._sync_scalar_bar_visibility(record.scalar_bar_id)
        self._render()
        return True

    def set_opacity(self, layer_id: str, opacity: float) -> bool:
        """Set an actor opacity in the inclusive range zero to one."""
        record = self._layers.get(layer_id)
        if record is None:
            return False
        record.opacity = self._validate_opacity(opacity)
        self._set_actor_opacity(record.actor, record.opacity)
        self._render()
        return True

    def remove(self, layer_id: str) -> bool:
        """Remove exactly one M9 actor and registry entry."""
        record = self._layers.pop(layer_id, None)
        if record is None:
            return False
        self._remove_actor(record.actor)
        self._remove_scalar_bar_if_unused(record.scalar_bar_id)
        self._render()
        return True

    def clear_m9_layers(self) -> int:
        """Remove registered M9 slices only; never clear the renderer."""
        records = list(self._layers.values())
        self._layers.clear()
        for record in records:
            self._remove_actor(record.actor)
        for scalar_bar_id in {record.scalar_bar_id for record in records if record.scalar_bar_id}:
            self._remove_scalar_bar_if_unused(scalar_bar_id)
        if records:
            self._render()
        return len(records)

    def discard_unregistered(self, actor: Any, scalar_bar_id: str = "") -> None:
        """Remove display objects created before a failed registry operation."""
        self._remove_actor(actor)
        self._remove_scalar_bar_if_unused(scalar_bar_id)
        self._render()

    def _get_scalar_bar_actor(self, scalar_bar_id: str) -> Any:
        scalar_bars = getattr(self.plotter, "scalar_bars", None)
        if scalar_bars is None:
            return None
        try:
            return scalar_bars[scalar_bar_id]
        except (KeyError, TypeError):
            return None

    def _remove_scalar_bar_if_unused(self, scalar_bar_id: str) -> bool:
        if not scalar_bar_id or any(
            record.scalar_bar_id == scalar_bar_id for record in self._layers.values()
        ):
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

    def _remove_actor(self, actor: Any) -> None:
        remover = getattr(self.plotter, "remove_actor", None)
        if remover is not None:
            remover(actor, render=False)

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
