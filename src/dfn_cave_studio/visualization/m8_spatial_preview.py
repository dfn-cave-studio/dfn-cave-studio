"""Lightweight M8 spatial preview renderer.

The renderer creates only domain boxes, trajectory polylines, observation
points, and a bounded number of sampled grid lines. It never allocates a
dense voxel array or renders every voxel.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from dfn_cave_studio.models.borehole import BoreholeCollection
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.services.spatial_domain_service import SpatialDomainService


class M8SpatialPreviewRenderer:
    """Render analysis/generation domains and borehole spatial evidence."""

    @staticmethod
    def render(
        plotter,
        analysis: ModelBounds,
        generation: ModelBounds,
        voxel: VoxelConfig,
        collection: BoreholeCollection,
        opacity: float = 0.35,
        maximum_grid_lines: int = 12,
        show_sampled_wireframe: bool = True,
        slice_axis: str = "z",
        slice_fraction: float = 0.5,
    ) -> dict[str, int]:
        """Render a bounded-complexity preview and return primitive counts."""
        plotter.clear()
        generation_box = pv.Box(bounds=generation.to_array()).extract_all_edges()
        analysis_box = pv.Box(bounds=analysis.to_array()).extract_all_edges()
        plotter.add_mesh(generation_box, color="#ff9800", line_width=2, name="dfn_generation_domain")
        plotter.add_mesh(analysis_box, color="#00bcd4", line_width=3, name="voxel_analysis_domain")

        report = SpatialDomainService.check_bounds(analysis, collection)
        trajectory_points = 0
        observation_points = 0
        for borehole in collection:
            points, _ = borehole.compute_trajectory(step_length=max(1.0, min(voxel.cell_size_x, voxel.cell_size_y)))
            trajectory_points += len(points)
            polyline = pv.lines_from_points(points, close=False)
            color = "#ff1744" if borehole.borehole_id in report.affected_holes else "#66bb6a"
            plotter.add_mesh(polyline, color=color, line_width=3, name=f"trajectory:{borehole.borehole_id}")
            inside = []
            outside = []
            for observation in borehole.fracture_observations:
                point = borehole.locate_observation(observation)
                if point is None:
                    continue
                observation_points += 1
                (inside if analysis.contains_point(*point) else outside).append(point)
            if inside:
                plotter.add_points(
                    np.asarray(inside), color="#ffee58", point_size=7, name=f"observations:{borehole.borehole_id}"
                )
            if outside:
                plotter.add_points(
                    np.asarray(outside),
                    color="#ff1744",
                    point_size=11,
                    render_points_as_spheres=True,
                    name=f"outside:{borehole.borehole_id}",
                )

        grid_lines = M8SpatialPreviewRenderer._sampled_grid_lines(
            analysis, voxel, maximum_grid_lines, slice_axis, slice_fraction
        )
        if show_sampled_wireframe and grid_lines.n_cells:
            plotter.add_mesh(grid_lines, color="#90a4ae", opacity=opacity, line_width=1, name="sampled_grid")
        plotter.show_axes()
        plotter.reset_camera()
        return {
            "trajectory_points": trajectory_points,
            "observation_points": observation_points,
            "grid_lines": int(grid_lines.n_cells),
        }

    @staticmethod
    def _sampled_grid_lines(
        bounds: ModelBounds,
        voxel: VoxelConfig,
        maximum: int,
        slice_axis: str,
        slice_fraction: float,
    ) -> pv.PolyData:
        nx, ny, nz = voxel.compute_grid_dimensions(bounds)
        indices = {
            "x": np.unique(np.linspace(0, nx, min(maximum, nx + 1), dtype=int)),
            "y": np.unique(np.linspace(0, ny, min(maximum, ny + 1), dtype=int)),
            "z": np.unique(np.linspace(0, nz, min(maximum, nz + 1), dtype=int)),
        }
        coordinates = {
            "x": lambda index: min(bounds.x_max, bounds.x_min + index * voxel.cell_size_x),
            "y": lambda index: min(bounds.y_max, bounds.y_min + index * voxel.cell_size_y),
            "z": lambda index: min(bounds.z_max, bounds.z_min + index * voxel.cell_size_z),
        }
        fixed = {
            "x": bounds.x_min + bounds.width * slice_fraction,
            "y": bounds.y_min + bounds.depth * slice_fraction,
            "z": bounds.z_min + bounds.height * slice_fraction,
        }
        segments: list[pv.PolyData] = []
        varying = [axis for axis in ("x", "y", "z") if axis != slice_axis]
        for index in indices[varying[0]]:
            point_a = {"x": bounds.x_min, "y": bounds.y_min, "z": bounds.z_min}
            point_b = {"x": bounds.x_min, "y": bounds.y_min, "z": bounds.z_min}
            point_a[slice_axis] = point_b[slice_axis] = fixed[slice_axis]
            point_a[varying[0]] = point_b[varying[0]] = coordinates[varying[0]](index)
            point_a[varying[1]] = getattr(bounds, f"{varying[1]}_min")
            point_b[varying[1]] = getattr(bounds, f"{varying[1]}_max")
            segments.append(pv.Line(tuple(point_a.values()), tuple(point_b.values())))
        for index in indices[varying[1]]:
            point_a = {"x": bounds.x_min, "y": bounds.y_min, "z": bounds.z_min}
            point_b = {"x": bounds.x_min, "y": bounds.y_min, "z": bounds.z_min}
            point_a[slice_axis] = point_b[slice_axis] = fixed[slice_axis]
            point_a[varying[1]] = point_b[varying[1]] = coordinates[varying[1]](index)
            point_a[varying[0]] = getattr(bounds, f"{varying[0]}_min")
            point_b[varying[0]] = getattr(bounds, f"{varying[0]}_max")
            segments.append(pv.Line(tuple(point_a.values()), tuple(point_b.values())))
        if not segments:
            return pv.PolyData()
        merged = segments[0]
        for segment in segments[1:]:
            merged = merged.merge(segment)
        return merged
