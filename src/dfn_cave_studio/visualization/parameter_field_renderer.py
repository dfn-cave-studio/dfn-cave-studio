"""PyVista rendering adapter for M9 parameter fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from dfn_cave_studio.models.m9 import ParameterFieldMetadata
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES
from dfn_cave_studio.models.spatial_grid import VoxelCellState


class ParameterFieldRenderer:
    """Render slices only; never builds one geometry object per voxel."""

    def create_grid(self, metadata: ParameterFieldMetadata, arrays: dict[str, np.ndarray]) -> pv.ImageData:
        """Create one image grid with cell arrays attached by reference/copy."""
        grid = pv.ImageData(
            dimensions=tuple(item + 1 for item in metadata.shape),
            spacing=metadata.spacing,
            origin=metadata.origin,
        )
        for name, array in arrays.items():
            grid.cell_data[name] = array.ravel(order="F")
        return grid

    def render_slice(
        self,
        plotter: Any,
        metadata: ParameterFieldMetadata,
        arrays: dict[str, np.ndarray],
        field_name: str,
        axis: str,
        fraction: float,
        *,
        opacity: float = 1.0,
        cmap: str = "viridis",
        actor_name: str | None = None,
    ) -> Any:
        """Show one X/Y/Z slice, keeping NO_DATA transparent and TRUE_ZERO visible."""
        if field_name not in arrays or axis not in {"x", "y", "z"} or not 0 <= fraction <= 1:
            raise ValueError("invalid parameter-field slice request")
        grid = self.create_grid(metadata, arrays)
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        normal = [0.0, 0.0, 0.0]
        normal[axis_index] = 1.0
        slice_index, coordinate = self.slice_location(metadata, axis, fraction)
        origin = list(grid.center)
        origin[axis_index] = coordinate
        sliced = grid.slice(normal=normal, origin=origin)
        name = actor_name or self.layer_id(field_name, axis, slice_index)
        actor = plotter.add_mesh(
            sliced,
            scalars=field_name,
            cmap=cmap,
            opacity=opacity,
            nan_opacity=0.0,
            show_scalar_bar=True,
            scalar_bar_args={"title": field_name},
            name=name,
        )
        return actor

    @staticmethod
    def slice_location(
        metadata: ParameterFieldMetadata,
        axis: str,
        fraction: float,
    ) -> tuple[int, float]:
        """Map a UI fraction to a stable voxel index and cell-centre coordinate."""
        if axis not in {"x", "y", "z"} or not 0 <= fraction <= 1:
            raise ValueError("invalid parameter-field slice request")
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        index = min(metadata.shape[axis_index] - 1, round(fraction * (metadata.shape[axis_index] - 1)))
        coordinate = metadata.origin[axis_index] + (index + 0.5) * metadata.spacing[axis_index]
        return index, coordinate

    @staticmethod
    def layer_id(field_name: str, axis: str, slice_index: int) -> str:
        """Return the stable actor/registry name for one M9 slice."""
        return f"m9_slice:{field_name}:{axis.lower()}:{int(slice_index)}"

    @staticmethod
    def render_borehole_roles(plotter: Any, collection: Any, holdout: Any) -> None:
        """Overlay calibration and validation trajectories with distinct colors."""
        for borehole in collection:
            points, _ = borehole.compute_trajectory(step_length=2.0)
            if len(points) < 2:
                continue
            polyline = pv.lines_from_points(points)
            is_validation = holdout is not None and holdout.is_validation(borehole.borehole_id)
            plotter.add_mesh(
                polyline,
                color="#d81b60" if is_validation else "#00acc1",
                line_width=4,
                name=f"m9-borehole:{borehole.borehole_id}",
            )

    @staticmethod
    def state_counts(arrays: dict[str, np.ndarray]) -> dict[str, int]:
        """Return explicit counts for all four semantic cell states."""
        states = arrays.get("cell_state")
        if states is None:
            return {state.value: 0 for state in VoxelCellState}
        return {state.value: int(np.count_nonzero(states == code)) for state, code in CELL_STATE_CODES.items()}

    @staticmethod
    def screenshot(plotter: Any, path: Path) -> Path:
        """Save the current parameter-field view as PNG."""
        output = Path(path)
        plotter.screenshot(str(output))
        return output
