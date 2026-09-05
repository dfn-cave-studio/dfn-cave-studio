"""Generic M9 scalar cloud rendering by reuse of the validated M11 VTK preparation path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from dfn_cave_studio.visualization.m11_voxel_renderer import M11DisplayConfig, M11VoxelRenderer
from dfn_cave_studio.visualization.scalar_lut import make_lookup_table_opaque


class ScalarFieldRenderer:
    """Render session-only M9 scalar fields without copying scientific arrays long-term."""

    def __init__(self) -> None:
        self._preparer = M11VoxelRenderer()

    def render(self, layer_manager: Any, metadata: Any, arrays: dict, field_id: str, field_name: str,
               unit: str, config: M11DisplayConfig) -> Any:
        """Prepare a native VTK dataset and register it in the existing M9 namespace."""
        result = SimpleNamespace(realization_id=f"m9_scalar_{field_id}", arrays=arrays)
        prepared = self._preparer.prepare_display(metadata, result, field_name, None, config)
        mode_token = prepared.display_mode
        location = prepared.plane_id or f"{prepared.axis}_{prepared.slice_index}"
        layer_id = f"m9_slice:scalar_{field_id}_{field_name}:{mode_token}:{location}:{prepared.interpolation_mode}"
        actor = layer_manager.plotter.add_mesh(
            prepared.mesh, scalars=field_name, preference=prepared.scalar_preference, cmap="viridis",
            clim=prepared.color_range, opacity=config.opacity, nan_opacity=1.0,
            show_edges=config.show_grid_lines, edge_color="#303030", line_width=0.5,
            show_scalar_bar=False, name=layer_id,
        )
        make_lookup_table_opaque(actor)
        scalar_bar_id = layer_manager.scalar_bar_id(f"scalar_{field_id}_{field_name}")
        try:
            scalar_bar = layer_manager.create_or_get_scalar_bar(scalar_bar_id, f"{field_name} [{unit}]", actor)
            return layer_manager.add_or_replace(
                layer_id, actor, field_name=field_name, axis=prepared.axis,
                slice_index=prepared.slice_index, coordinate=prepared.coordinate,
                opacity=config.opacity, scalar_bar_id=scalar_bar_id if scalar_bar is not None else "",
                scalar_bar_actor=scalar_bar,
            )
        except Exception:
            layer_manager.discard_unregistered(actor, scalar_bar_id)
            raise
