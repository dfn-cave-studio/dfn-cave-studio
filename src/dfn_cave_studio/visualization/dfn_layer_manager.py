"""Session-only registry and merged rendering for M10 explicit DFN layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from dfn_cave_studio.dfn.m10_geometry import iter_polygons, source_mask


DFN_LAYER_PREFIX = "dfn:"


@dataclass(slots=True)
class DFNLayerRecord:
    """Display state and actor reference for one merged DFN layer."""

    layer_id: str
    actor: Any
    realization_id: str
    set_id: int | None
    source: str | None
    domain_id: int | None
    size_class: int | None
    color_by: str
    category: str
    visible: bool
    opacity: float
    color: str
    fracture_count: int
    created_at: datetime


class DFNLayerManager:
    """Manage only actors in the ``dfn:`` namespace."""

    def __init__(self, plotter: Any) -> None:
        self.plotter = plotter
        self._layers: dict[str, DFNLayerRecord] = {}

    def render_realization(
        self,
        realization: Any,
        *,
        set_id: int | None = None,
        source: str | None = None,
        domain_id: int | None = None,
        size_class: int | None = None,
        color_by: str = "Joint Set",
        category: str = "all",
        excluded_size_classes: set[int] | None = None,
        opacity: float = 0.7,
        color: str = "#1976d2",
        mode: str = "lod",
        exact_limit: int = 100_000,
    ) -> DFNLayerRecord:
        """Render all selected fractures using GPU LOD or bounded exact geometry."""
        layer_id = self.layer_id(
            realization.realization_id, set_id=set_id, source=source,
            domain_id=domain_id, size_class=size_class, color_by=color_by,
        )
        arrays = realization.geometry_arrays
        mask = np.ones(realization.fracture_count, dtype=bool)
        if set_id is not None:
            mask &= arrays["set_id"] == set_id
        if source is not None:
            mask &= source_mask(realization, source)
        if domain_id is not None:
            mask &= arrays["domain_id"] == domain_id
        if size_class is not None:
            mask &= arrays.get("size_class", np.zeros(realization.fracture_count, dtype=np.uint8)) == size_class
        if excluded_size_classes:
            class_values = arrays.get("size_class", np.zeros(realization.fracture_count, dtype=np.uint8))
            mask &= ~np.isin(class_values, list(excluded_size_classes))
        indices = np.flatnonzero(mask)
        if mode == "exact":
            if len(indices) > exact_limit:
                raise ValueError(f"Exact Geometry is limited to {exact_limit:,} selected fractures")
            mesh = self._polydata(realization, indices)
            actor = self.plotter.add_mesh(
                mesh, name=layer_id, color=color, opacity=float(opacity), show_edges=False
            )
        elif mode == "lod":
            actor = self._glyph_actor(realization, indices, layer_id, color, float(opacity), color_by)
        else:
            raise ValueError("mode must be 'lod' or 'exact'")
        if actor is None:
            raise RuntimeError("Renderer did not return an actor")
        return self.add_or_replace(
            layer_id,
            actor,
            realization_id=realization.realization_id,
            set_id=set_id,
            source=source,
            domain_id=domain_id,
            size_class=size_class,
            color_by=color_by,
            category=category,
            opacity=opacity,
            color=color,
            fracture_count=len(indices),
        )

    @staticmethod
    def layer_id(
        realization_id: str, *, set_id: int | None = None, source: str | None = None,
        domain_id: int | None = None, size_class: int | None = None, color_by: str = "Joint Set",
    ) -> str:
        """Build a stable namespace ID for one display grouping."""
        if source == "CONDITIONED_OBSERVATION":
            suffix = "conditioned"
        elif source == "DETERMINISTIC_STRUCTURE":
            suffix = "deterministic"
        elif set_id is not None:
            suffix = f"set:{set_id}"
        elif domain_id is not None:
            suffix = f"domain:{domain_id}"
        elif size_class is not None:
            suffix = f"size:{size_class}"
        else:
            suffix = "all"
        return f"dfn:realization:{realization_id}:{color_by.lower().replace(' ', '_')}:{suffix}"

    def add_or_replace(self, layer_id: str, actor: Any, **metadata: Any) -> DFNLayerRecord:
        """Register a valid actor, replacing only the same DFN layer."""
        if not layer_id.startswith(DFN_LAYER_PREFIX):
            raise ValueError("DFN layer IDs must use the dfn namespace")
        if actor is None:
            raise ValueError("Cannot register a missing DFN actor")
        old = self._layers.get(layer_id)
        if old is not None and old.actor is not actor:
            self._remove_actor(old.actor)
        metadata.setdefault("domain_id", None)
        metadata.setdefault("size_class", None)
        metadata.setdefault("color_by", "Joint Set")
        metadata.setdefault("category", "all")
        record = DFNLayerRecord(layer_id=layer_id, actor=actor, visible=True, created_at=datetime.now(UTC), **metadata)
        self._layers[layer_id] = record
        return record

    def list_layers(self) -> list[DFNLayerRecord]:
        """Return registered layers in stable order."""
        return sorted(self._layers.values(), key=lambda item: (item.created_at, item.layer_id))

    def contains(self, layer_id: str) -> bool:
        """Return whether a layer ID is registered."""
        return layer_id in self._layers

    def set_visible(self, layer_id: str, visible: bool) -> bool:
        """Set actor visibility without touching project state."""
        record = self._layers.get(layer_id)
        if record is None:
            return False
        record.visible = bool(visible)
        setter = getattr(record.actor, "SetVisibility", None)
        if setter:
            setter(record.visible)
        self._render()
        return True

    def set_opacity(self, layer_id: str, opacity: float) -> bool:
        """Set actor opacity in the inclusive range zero to one."""
        opacity = float(opacity)
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("opacity must be between 0 and 1")
        record = self._layers.get(layer_id)
        if record is None:
            return False
        record.opacity = opacity
        get_property = getattr(record.actor, "GetProperty", None)
        actor_property = get_property() if get_property else getattr(record.actor, "prop", None)
        setter = getattr(actor_property, "SetOpacity", None)
        if setter:
            setter(opacity)
        self._render()
        return True

    def remove(self, layer_id: str) -> bool:
        """Remove one selected DFN actor only."""
        record = self._layers.pop(layer_id, None)
        if record is None:
            return False
        self._remove_actor(record.actor)
        self._render()
        return True

    def clear_realization(self, realization_id: str) -> int:
        """Remove all session layers belonging to one realization."""
        ids = [item.layer_id for item in self._layers.values() if item.realization_id == realization_id]
        for layer_id in ids:
            record = self._layers.pop(layer_id)
            self._remove_actor(record.actor)
        if ids:
            self._render()
        return len(ids)

    def clear_dfn_layers(self) -> int:
        """Remove only registered ``dfn:`` actors; never clear the renderer."""
        records = list(self._layers.values())
        self._layers.clear()
        for record in records:
            self._remove_actor(record.actor)
        if records:
            self._render()
        return len(records)

    @staticmethod
    def _polydata(realization: Any, indices: np.ndarray):
        import pyvista as pv

        arrays = realization.geometry_arrays
        points: list[np.ndarray] = []
        faces: list[int] = []
        offset = 0
        valid_indices: list[int] = []
        for index, polygon in iter_polygons(realization, indices):
            count = len(polygon)
            if count < 3:
                continue
            points.extend(polygon)
            faces.extend([count, *range(offset, offset + count)])
            offset += count
            valid_indices.append(int(index))
        mesh = pv.PolyData(np.asarray(points, dtype=float).reshape((-1, 3)), np.asarray(faces, dtype=np.int64))
        if valid_indices:
            mesh.cell_data["set_id"] = arrays["set_id"][valid_indices]
            mesh.cell_data["radius"] = arrays["radius"][valid_indices]
        return mesh

    def _glyph_actor(
        self,
        realization: Any,
        indices: np.ndarray,
        layer_id: str,
        color: str,
        opacity: float,
        color_by: str,
    ) -> Any:
        """Create one vtkGlyph3DMapper actor without expanding disc vertices in Python."""
        arrays = realization.geometry_arrays
        try:
            import pyvista as pv
            from vtkmodules.vtkFiltersSources import vtkDiskSource
            from vtkmodules.vtkRenderingCore import vtkActor, vtkGlyph3DMapper

            points = pv.PolyData(np.asarray(arrays["center"][indices], dtype=np.float32))
            points.point_data["normal"] = np.asarray(arrays["normal"][indices], dtype=np.float32)
            points.point_data["radius"] = np.asarray(arrays["radius"][indices], dtype=np.float32)
            category_values = self._color_categories(realization, indices, color_by)
            palette = np.asarray(
                ((31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
                 (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127)),
                dtype=np.uint8,
            )
            if color_by == "Size Class":
                stable = {0: (127, 127, 127), 1: (158, 202, 225), 2: (253, 174, 107), 3: (214, 39, 40)}
                rgb = [stable.get(int(value), stable[0]) for value in category_values]
            elif color_by == "Source":
                stable = {0: (78, 121, 167), 1: (242, 142, 43), 2: (225, 87, 89)}
                rgb = [stable.get(int(value), (127, 127, 127)) for value in category_values]
            else:
                rgb = [palette[abs(int(value)) % len(palette)] for value in category_values]
            points.point_data["category_rgb"] = np.asarray(rgb, dtype=np.uint8)
            source = vtkDiskSource()
            source.SetInnerRadius(0.0)
            source.SetOuterRadius(1.0)
            source.SetCircumferentialResolution(8)
            mapper = vtkGlyph3DMapper()
            mapper.SetInputData(points)
            mapper.SetSourceConnection(source.GetOutputPort())
            mapper.SetOrientationArray("normal")
            mapper.OrientOn()
            mapper.SetScaleArray("radius")
            mapper.SetScaleModeToScaleByMagnitude()
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray("category_rgb")
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
            actor = vtkActor()
            actor.SetMapper(mapper)
            rgb = pv.Color(color).float_rgb
            actor.GetProperty().SetColor(*rgb)
            actor.GetProperty().SetOpacity(opacity)
            add_actor = getattr(self.plotter, "add_actor", None)
            if add_actor is not None:
                add_actor(actor, name=layer_id, render=True)
                return actor
        except (ImportError, AttributeError, TypeError):
            pass
        import pyvista as pv

        points = pv.PolyData(np.asarray(arrays["center"][indices], dtype=np.float32))
        return self.plotter.add_mesh(
            points,
            name=layer_id,
            color=color,
            opacity=opacity,
            style="points",
            point_size=2,
            render_points_as_spheres=False,
        )

    @staticmethod
    def _color_categories(realization: Any, indices: np.ndarray, color_by: str) -> np.ndarray:
        arrays = realization.geometry_arrays
        if color_by == "Domain":
            return np.asarray(arrays["domain_id"][indices], dtype=np.int32)
        if color_by == "Source":
            return np.asarray(arrays["source_code"][indices], dtype=np.int32)
        if color_by == "Size Class":
            values = arrays.get("size_class", np.zeros(realization.fracture_count, dtype=np.uint8))
            return np.asarray(values[indices], dtype=np.int32)
        return np.asarray(arrays["set_id"][indices], dtype=np.int32)

    def _remove_actor(self, actor: Any) -> None:
        remover = getattr(self.plotter, "remove_actor", None)
        if remover:
            remover(actor, render=False)

    def _render(self) -> None:
        render = getattr(self.plotter, "render", None)
        if render:
            render()
