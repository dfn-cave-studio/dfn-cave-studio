"""Lightweight M8 spatial preview renderer.

The renderer creates only domain boxes, trajectory polylines, observation
points, and a bounded number of sampled grid lines. It never allocates a
dense voxel array or renders every voxel.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyvista as pv

from dfn_cave_studio.models.borehole import BoreholeCollection
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.services.spatial_domain_service import SpatialDomainService


class M8SpatialPreviewRenderer:
    """Render analysis/generation domains and borehole spatial evidence."""

    _FIXED_ACTORS = {
        "dfn_generation_domain",
        "voxel_analysis_domain",
        "multisource_observations",
        "outside:multisource_observations",
        "sampled_grid",
    }
    _DYNAMIC_PREFIXES = ("trajectory:", "observations:", "outside:")

    @staticmethod
    def clear(plotter) -> None:
        """Remove only actors created by the current M8 preview."""
        if plotter is None:
            return
        actors = getattr(getattr(plotter, "renderer", None), "actors", {})
        names = list(actors) if hasattr(actors, "__iter__") else []
        for name in names:
            if name in M8SpatialPreviewRenderer._FIXED_ACTORS or name.startswith(
                M8SpatialPreviewRenderer._DYNAMIC_PREFIXES
            ):
                try:
                    plotter.remove_actor(name, render=False)
                except (AttributeError, KeyError, RuntimeError, TypeError):
                    continue

    @staticmethod
    def render(
        plotter,
        analysis: ModelBounds,
        generation: ModelBounds,
        voxel: VoxelConfig,
        collection: BoreholeCollection,
        extra_points: Iterable[tuple[str, tuple[float, float, float]]] = (),
        point_labels: Iterable[tuple[str, str, tuple[float, float, float]]] = (),
        opacity: float = 0.35,
        maximum_grid_lines: int = 12,
        show_sampled_wireframe: bool = True,
        slice_axis: str = "z",
        slice_fraction: float = 0.5,
        borehole_display_manager=None,
        domain_by_hole: dict[str, int | None] | None = None,
    ) -> dict[str, int]:
        """Render a bounded-complexity preview and return primitive counts."""
        if borehole_display_manager is not None:
            borehole_display_manager.clear()
        plotter.clear()
        generation_box = pv.Box(bounds=generation.to_array()).extract_all_edges()
        analysis_box = pv.Box(bounds=analysis.to_array()).extract_all_edges()
        plotter.add_mesh(generation_box, color="#ff9800", line_width=2, name="dfn_generation_domain")
        plotter.add_mesh(analysis_box, color="#00bcd4", line_width=3, name="voxel_analysis_domain")

        extra_points = list(extra_points)
        report = SpatialDomainService.check_bounds(analysis, collection, extra_points)
        trajectory_points = 0
        observation_points = 0
        borehole_actor_names: dict[str, str] = {}
        for borehole in collection:
            points, _ = borehole.compute_trajectory(step_length=max(1.0, min(voxel.cell_size_x, voxel.cell_size_y)))
            trajectory_points += len(points)
            polyline = pv.lines_from_points(points, close=False)
            color = "#ff1744" if borehole.borehole_id in report.affected_holes else "#66bb6a"
            actor_name = f"trajectory:{borehole.borehole_id}"
            plotter.add_mesh(polyline, color=color, line_width=3, name=actor_name, pickable=True)
            borehole_actor_names[str(borehole.borehole_id)] = actor_name
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

        extra_inside = [point for _, point in extra_points if analysis.contains_point(*point)]
        extra_outside = [point for _, point in extra_points if not analysis.contains_point(*point)]
        observation_points += len(extra_points)
        if extra_inside:
            plotter.add_points(
                np.asarray(extra_inside), color="#ab47bc", point_size=8, name="multisource_observations"
            )
        if extra_outside:
            plotter.add_points(
                np.asarray(extra_outside),
                color="#ff1744",
                point_size=11,
                render_points_as_spheres=True,
                name="outside:multisource_observations",
            )

        grid_lines = M8SpatialPreviewRenderer._sampled_grid_lines(
            analysis, voxel, maximum_grid_lines, slice_axis, slice_fraction
        )
        if show_sampled_wireframe and grid_lines.n_cells:
            plotter.add_mesh(grid_lines, color="#90a4ae", opacity=opacity, line_width=1, name="sampled_grid")
        plotter.show_axes()
        if borehole_display_manager is not None:
            try:
                borehole_display_manager.attach(
                    collection,
                    borehole_actor_names,
                    domain_by_hole,
                    analysis,
                    point_labels=list(point_labels),
                )
            except Exception:
                borehole_display_manager.clear()
                raise
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
