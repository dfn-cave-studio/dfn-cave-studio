"""Session-only point preview for Phase 2A borehole realizations."""

from __future__ import annotations

from typing import Any

import numpy as np

from dfn_cave_studio.models.borehole_fracture_realization import BoreholeFractureComponent
from dfn_cave_studio.services.joint_set_service import joint_set_color


class BoreholeFractureRenderer:
    """Render bounded point previews without altering realization arrays."""

    PREFIX = "borehole_realization:"
    LAYER_NAME = "Borehole Fracture Realization"
    RANDOM_COLOR = "#9e9e9e"
    LEGEND_NAME = "borehole_realization_legend"

    def __init__(self, plotter: Any, maximum_points: int = 100_000):
        self.plotter = plotter
        self.maximum_points = maximum_points
        self._actor_names: set[str] = set()
        self._actors: dict[str, Any] = {}
        self.current_realization_id: str | None = None
        self.visible = False
        self._legend_actor: Any | None = None

    @property
    def actor_names(self) -> set[str]:
        """Return the owned Phase 2A actor names."""
        return set(self._actor_names)

    def render(
        self,
        realization,
        joint_sets: list[Any] | None = None,
        *,
        visible_set_ids: set[int] | None = None,
        show_random: bool = True,
        visible_component_indices: set[int] | None = None,
    ) -> int:
        """Replace the named realization layer and return the displayed point count."""
        self.clear()
        offsets = realization.arrays.get("xyz_offset")
        if offsets is None:
            return 0
        origin = np.asarray(realization.provenance.get("xyz_origin", (0, 0, 0)), dtype=np.float64)
        points = np.asarray(offsets, dtype=np.float32) + origin
        components = np.asarray(realization.arrays["component_type"], dtype=np.uint8)
        set_ids = np.asarray(realization.arrays.get("global_set_id", np.ones(len(points))), dtype=np.int32)
        local_indices = np.asarray(
            realization.arrays.get("local_component_index", np.full(len(points), -1)), dtype=np.int32
        )
        colours = {int(item.set_id): str(item.color) for item in (joint_sets or [])}
        available_sets = sorted(int(value) for value in np.unique(set_ids[components == int(BoreholeFractureComponent.DOMINANT_SET)]))
        selected_sets = set(available_sets) if visible_set_ids is None else set(visible_set_ids)
        eligible = np.zeros(len(points), dtype=bool)
        for set_id in selected_sets:
            eligible |= (components == int(BoreholeFractureComponent.DOMINANT_SET)) & (set_ids == set_id)
        if show_random:
            eligible |= components == int(BoreholeFractureComponent.RANDOM_BACKGROUND)
        if visible_component_indices is not None:
            eligible &= np.isin(local_indices, np.asarray(sorted(visible_component_indices), dtype=np.int32))
        eligible_count = int(np.count_nonzero(eligible))
        displayed_count = 0

        def sampled(mask: np.ndarray) -> np.ndarray:
            nonlocal displayed_count
            indices = np.flatnonzero(eligible & mask)
            if not len(indices):
                return indices
            quota = max(1, int(self.maximum_points * len(indices) / max(eligible_count, 1)))
            stride = max(1, int(np.ceil(len(indices) / quota)))
            result = indices[::stride]
            displayed_count += len(result)
            return result

        try:
            legend_entries: list[tuple[str, str]] = []
            for set_id in sorted(selected_sets):
                indices = sampled(
                    (components == int(BoreholeFractureComponent.DOMINANT_SET)) & (set_ids == set_id)
                )
                if not len(indices):
                    continue
                name = f"{self.PREFIX}{realization.realization_id}:global_set:{set_id}"
                color = colours.get(set_id, joint_set_color(set_id))
                actor = self.plotter.add_points(
                    points[indices],
                    name=name,
                    color=color,
                    point_size=6,
                    render_points_as_spheres=True,
                    pickable=False,
                )
                if hasattr(actor, "SetPickable"):
                    actor.SetPickable(False)
                self._actor_names.add(name)
                self._actors[name] = actor
                legend_entries.append((f"Global Set {set_id}", color))
            if show_random:
                indices = sampled(components == int(BoreholeFractureComponent.RANDOM_BACKGROUND))
                if len(indices):
                    name = f"{self.PREFIX}{realization.realization_id}:random_background"
                    actor = self.plotter.add_points(
                        points[indices],
                        name=name,
                        color=self.RANDOM_COLOR,
                        point_size=6,
                        render_points_as_spheres=True,
                        pickable=False,
                    )
                    if hasattr(actor, "SetPickable"):
                        actor.SetPickable(False)
                    self._actor_names.add(name)
                    self._actors[name] = actor
                    legend_entries.append(("Random Background", self.RANDOM_COLOR))
            if legend_entries and hasattr(self.plotter, "add_legend"):
                self._legend_actor = self.plotter.add_legend(
                    legend_entries, bcolor="white", loc="upper right", name=self.LEGEND_NAME
                )
        except Exception:
            self.clear()
            raise
        self.current_realization_id = realization.realization_id
        self.visible = bool(self._actors)
        self.plotter.render()
        return displayed_count

    def set_visible(self, visible: bool) -> None:
        """Toggle the current session layer without changing realization data."""
        self.visible = bool(visible and self._actors)
        for actor in self._actors.values():
            if hasattr(actor, "SetVisibility"):
                actor.SetVisibility(self.visible)
        if self._legend_actor is not None and hasattr(self._legend_actor, "SetVisibility"):
            self._legend_actor.SetVisibility(self.visible)
        try:
            self.plotter.render()
        except (AttributeError, RuntimeError):
            pass

    def clear(self) -> None:
        """Remove only Phase 2A preview actors."""
        for name in list(self._actor_names):
            try:
                self.plotter.remove_actor(name, render=False)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass
        if self._legend_actor is not None:
            try:
                self.plotter.remove_actor(self._legend_actor, render=False)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass
            self._legend_actor = None
        self._actor_names.clear()
        self._actors.clear()
        self.current_realization_id = None
        self.visible = False
        try:
            self.plotter.render()
        except (AttributeError, RuntimeError):
            pass
