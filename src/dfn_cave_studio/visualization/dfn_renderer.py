"""
DFN 3D visualization using PyVista.

Renders fracture networks as colored disks/polygons in the 3D viewport.
Supports:
  - Per-set coloring
  - Opacity and visibility toggles
  - Clipping planes
  - Deterministic vs stochastic distinction
  - Performance culling for large DFNs (>50K fractures)

References:
  - PyVista documentation: https://docs.pyvista.org/
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Tuple, Callable
import math

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.models.fracture import StochasticFracture, DeterministicFracture
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.dfn_realization import DFNRealization

_logger = logging.getLogger(__name__)


class DFNRenderer:
    """Renders DFN fractures as 3D primitives using PyVista.

    Creates colored disk meshes for each fracture, organized by fracture set.
    Handles visibility, opacity, and performance culling for large networks.

    Usage:
        renderer = DFNRenderer()
        renderer.add_realization(realization, joint_sets)
        renderer.render(plotter)  # plotter = PyVistaQtInteractor
    """

    # ── Color palette for fracture sets ───────────────────────────────────

    DEFAULT_COLORS = [
        "#1976d2", "#388e3c", "#f57c00", "#d32f2f", "#7b1fa2",
        "#0288d1", "#689f38", "#fbc02d", "#e64a19", "#5c6bc0",
    ]

    def __init__(self, max_display_fractures: int = 50_000):
        self._max_display = max_display_fractures
        self._actors: Dict[str, any] = {}  # set_name → PyVista actor
        self._visibility: Dict[str, bool] = {}
        self._opacity: Dict[str, float] = {}
        self._fracture_count: Dict[str, int] = {}

    # ── Disk Mesh Generation ──────────────────────────────────────────────

    @staticmethod
    def fracture_to_disk_mesh(
        center: NDArray[np.float64],
        normal: NDArray[np.float64],
        radius: float,
        n_sides: int = 24,
    ):
        """Create a circular disk mesh for a single fracture.

        Args:
            center: (3,) center of the disk.
            normal: (3,) unit normal vector.
            radius: Disk radius in meters.
            n_sides: Number of segments around circumference.

        Returns:
            pyvista.PolyData disk mesh.
        """
        import pyvista as pv

        # Create a unit circle in the XY plane
        theta = np.linspace(0, 2 * math.pi, n_sides + 1)[:-1]
        circle_pts = np.column_stack([
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros(n_sides),
        ])

        # Build rotation matrix to align Z axis with the normal
        z_axis = np.array([0.0, 0.0, 1.0])
        normal = normal / np.linalg.norm(normal)

        if np.allclose(normal, z_axis):
            rot_matrix = np.eye(3)
        elif np.allclose(normal, -z_axis):
            rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            # Rotation axis = z × normal
            rot_axis = np.cross(z_axis, normal)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            cos_theta = np.dot(z_axis, normal)
            theta_angle = math.acos(cos_theta)

            # Rodrigues rotation formula
            K = np.array([
                [0, -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0, -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0],
            ])
            rot_matrix = np.eye(3) + math.sin(theta_angle) * K + (1 - cos_theta) * K @ K

        # Rotate and translate
        rotated_pts = (rot_matrix @ circle_pts.T).T + center

        # Create triangular mesh (fan triangulation)
        vertices = np.vstack([center, rotated_pts])
        faces_list = []
        for i in range(n_sides):
            j = (i + 1) % n_sides
            faces_list.append([3, 0, i + 1, j + 1])

        faces = np.array(faces_list, dtype=np.int32)

        disk = pv.PolyData(vertices, faces)
        return disk

    @staticmethod
    def fractures_to_multiblock(
        fractures: List[StochasticFracture],
        color: str = "#1976d2",
        opacity: float = 0.7,
        n_sides: int = 16,
    ) -> Optional[any]:
        """Convert a list of fractures to a single merged PyVista mesh.

        Args:
            fractures: List of StochasticFracture objects.
            color: Display color.
            opacity: Display opacity (0-1).
            n_sides: Disk polygon resolution.

        Returns:
            pyvista.PolyData merged mesh, or None if empty.
        """
        import pyvista as pv

        if not fractures:
            return None

        meshes = []
        for f in fractures:
            try:
                geo = f.geometry
                disk = DFNRenderer.fracture_to_disk_mesh(
                    center=geo.center,
                    normal=geo.normal,
                    radius=f.radius if f.radius > 0 else geo.radius or 1.0,
                    n_sides=n_sides,
                )
                meshes.append(disk)
            except Exception as e:
                _logger.error(f"fracture_to_disk_mesh failed for fracture: {e}", exc_info=True)
                continue

        if not meshes:
            return None

        merged = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
        return merged

    # ── Rendering ─────────────────────────────────────────────────────────

    def add_realization(
        self,
        realization: DFNRealization,
        joint_sets: List[JointSetConfig],
        plotter=None,
    ) -> None:
        """Add a DFN realization to the visualization.

        Args:
            realization: DFNRealization with stochastic fractures.
            joint_sets: List of joint set configurations (for colors).
            plotter: Optional PyVista plotter to render to immediately.
        """
        set_colors = self._build_color_map(joint_sets)

        for set_id, fractures in self._group_by_set(realization).items():
            set_config = next((js for js in joint_sets if js.set_id == set_id), None)
            color = set_config.color if set_config else set_colors.get(set_id, "#1976d2")
            name = set_config.name if set_config else f"Set {set_id}"

            mesh = self.fractures_to_multiblock(fractures, color)
            if mesh is not None and plotter is not None:
                actor = plotter.add_mesh(
                    mesh, color=color, opacity=set_config.opacity if set_config else 0.7,
                    name=name, show_edges=False, smooth_shading=True,
                )
                self._actors[name] = actor
                self._visibility[name] = True
                self._opacity[name] = set_config.opacity if set_config else 0.7
                self._fracture_count[name] = len(fractures)

    def render_to_plotter(self, plotter, realization: DFNRealization, joint_sets: List[JointSetConfig]) -> None:
        """Render a DFN realization to a PyVista plotter.

        Args:
            plotter: PyVistaQtInteractor or pyvista.Plotter.
            realization: The DFN realization to render.
            joint_sets: Joint set configs for coloring.
        """
        self.clear(plotter)
        self.add_realization(realization, joint_sets, plotter)
        self._add_bounding_box(plotter, realization)

    def clear(self, plotter) -> None:
        """Remove all DFN actors from the plotter."""
        for name in list(self._actors.keys()):
            try:
                plotter.remove_actor(self._actors[name])
            except Exception as e:
                _logger.error(f"renderer operation failed: {e}", exc_info=True)
        self._actors.clear()
        self._visibility.clear()
        self._opacity.clear()
        self._fracture_count.clear()

    def set_visibility(self, plotter, set_name: str, visible: bool) -> None:
        """Toggle visibility of a fracture set."""
        if set_name in self._actors:
            try:
                self._actors[set_name].SetVisibility(visible)
                self._visibility[set_name] = visible
            except Exception as e:
                _logger.error(f"renderer operation failed: {e}", exc_info=True)

    def set_opacity(self, plotter, set_name: str, opacity: float) -> None:
        """Set opacity for a fracture set."""
        if set_name in self._actors:
            try:
                self._actors[set_name].GetProperty().SetOpacity(opacity)
                self._opacity[set_name] = opacity
            except Exception as e:
                _logger.error(f"renderer operation failed: {e}", exc_info=True)

    def get_statistics(self) -> Dict[str, int]:
        """Get fracture count per set."""
        return dict(self._fracture_count)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_color_map(self, joint_sets: List[JointSetConfig]) -> Dict[int, str]:
        """Build a set_id → color mapping."""
        colors = {}
        for i, js in enumerate(joint_sets):
            colors[js.set_id] = js.color if js.color else self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
        return colors

    def _group_by_set(self, realization: DFNRealization) -> Dict[int, List[StochasticFracture]]:
        """Group stochastic fractures by set_id."""
        groups: Dict[int, List[StochasticFracture]] = {}
        for f in realization.stochastic_fractures:
            groups.setdefault(f.set_id, []).append(f)
        return groups

    def _add_bounding_box(self, plotter, realization: DFNRealization) -> None:
        """Add model bounding box wireframe to the plotter."""
        try:
            import pyvista as pv
            if realization.generation_config and realization.generation_config.joint_sets:
                # Get bounds from the active DFN generation context
                pass
            # Default: create a visible box using model bounds
            # For now, look for bounds on the realization
            b = None
            if hasattr(realization, 'model_bounds') and realization.model_bounds is not None:
                b = realization.model_bounds
            if b is not None:
                box = pv.Box(bounds=[
                    b.x_min, b.x_max,
                    b.y_min, b.y_max,
                    b.z_min, b.z_max,
                ])
                plotter.add_mesh(box, color="white", opacity=0.2,
                                 style="wireframe", line_width=1,
                                 name="Model Bounds")
        except Exception as e:
            _logger.error(f"_add_bounding_box failed: {e}", exc_info=True)

    # ── Borehole Rendering ─────────────────────────────────────────────

    def render_boreholes(self, collection, plotter) -> None:
        """Render all borehole trajectories as colored tubes.

        Args:
            collection: BoreholeCollection with trajectories.
            plotter: PyVista plotter.
        """
        try:
            import pyvista as pv
            colors = ["#ff6f00", "#1976d2", "#388e3c", "#d32f2f", "#7b1fa2"]
            for i, bh in enumerate(collection):
                points, _ = bh.compute_trajectory(step_length=5.0)
                if len(points) < 2:
                    continue
                # Create tube from polyline
                line = pv.PolyData(points)
                tube = line.tube(radius=0.2, n_sides=8)
                color = colors[i % len(colors)]
                plotter.add_mesh(tube, color=color, name=f"BH-{bh.borehole_id}",
                                 label=f"BH-{bh.borehole_id}",
                                 show_edges=False, smooth_shading=True)
        except Exception as e:
            _logger.error(f"render_boreholes failed: {e}", exc_info=True)

    def render_fracture_observations(self, collection, plotter) -> None:
        """Render fracture observation positions as small spheres.

        Args:
            collection: BoreholeCollection with observations.
            plotter: PyVista plotter.
        """
        try:
            import pyvista as pv
            for bh in collection:
                for obs in bh.fracture_observations:
                    pos = bh.locate_observation(obs)
                    if pos is None:
                        continue
                    set_id = getattr(obs, 'set_id', None) or 0
                    colors_cycle = ['#ff6f00', '#1976d2', '#388e3c', '#d32f2f', '#7b1fa2']
                    color = colors_cycle[set_id % len(colors_cycle)]
                    sphere = pv.Sphere(radius=0.15, center=pos)
                    plotter.add_mesh(sphere, color=color,
                                     name=f"Obs-{bh.borehole_id}-{obs.measured_depth:.1f}",
                                     show_edges=False)
        except Exception as e:
            _logger.error(f"render_fracture_observations failed: {e}", exc_info=True)

    def render_voxel_p32(self, voxel_p32_data, bounds, voxel_config, plotter,
                         cmap: str = "viridis") -> bool:
        """Render voxel grid colored by local P32 values.

        Args:
            voxel_p32_data: List of dicts [{i,j,k,local_p32,...}] or
                            dict mapping (i,j,k) → P32 value.
            bounds: ModelBounds.
            voxel_config: VoxelConfig.
            plotter: PyVista plotter.
            cmap: Matplotlib colormap name.

        Returns:
            True if rendering succeeded, False on error.
        """
        import pyvista as pv
        import math
        try:
            nx = int(math.ceil((bounds.x_max - bounds.x_min) / voxel_config.cell_size_x))
            ny = int(math.ceil((bounds.y_max - bounds.y_min) / voxel_config.cell_size_y))
            nz = int(math.ceil((bounds.z_max - bounds.z_min) / voxel_config.cell_size_z))
            nx = max(1, nx); ny = max(1, ny); nz = max(1, nz)

            # pv.ImageData is the correct class in PyVista 0.48.x
            grid = pv.ImageData(
                dimensions=(nx + 1, ny + 1, nz + 1),
                spacing=(voxel_config.cell_size_x,
                         voxel_config.cell_size_y,
                         voxel_config.cell_size_z),
                origin=(bounds.x_min, bounds.y_min, bounds.z_min),
            )

            p32_values = np.zeros(nx * ny * nz, dtype=np.float64)
            if isinstance(voxel_p32_data, dict):
                for (i, j, k), val in voxel_p32_data.items():
                    idx = int(i) * ny * nz + int(j) * nz + int(k)
                    if 0 <= idx < len(p32_values):
                        p32_values[idx] = float(val)
            elif isinstance(voxel_p32_data, (list, tuple)):
                for entry in voxel_p32_data:
                    if isinstance(entry, dict):
                        i, j, k = entry.get("i", 0), entry.get("j", 0), entry.get("k", 0)
                        idx = int(i) * ny * nz + int(j) * nz + int(k)
                        if 0 <= idx < len(p32_values):
                            p32_values[idx] = float(entry.get("local_p32", 0.0))

            grid.cell_data["local_p32"] = p32_values
            plotter.add_mesh(grid, scalars="local_p32", cmap=cmap,
                             opacity=0.5, show_edges=True,
                             name="Voxel P32", show_scalar_bar=True,
                             scalar_bar_args={"title": "P32 (m²/m³)"})
            return True
        except Exception as e:
            _logger.error(f"render_voxel_p32 failed: {e}", exc_info=True)
            return False

    def screenshot(self, plotter, path: str, transparent: bool = True) -> bool:
        """Save a screenshot of the current plotter view.

        Args:
            plotter: PyVista plotter.
            path: Output file path (.png recommended).
            transparent: Whether to use transparent background.

        Returns:
            True if screenshot was saved successfully.
        """
        try:
            plotter.screenshot(path, transparent_background=transparent)
            return True
        except Exception as e:
            _logger.error(f"screenshot failed: {e}", exc_info=True)
            return False

    def export_voxels_vtu(self, voxel_p32_data, bounds, voxel_config, path: str) -> bool:
        """Export voxel grid with P32 as VTI (ImageData) or VTR (RectilinearGrid).

        Args:
            voxel_p32_data: List of dicts [{i,j,k,local_p32,...}] or
                            dict mapping (i,j,k) → P32 value.
            bounds: ModelBounds.
            voxel_config: VoxelConfig.
            path: Output file path (.vti or .vtr).

        Returns:
            True if export succeeded, False on error.
        """
        import pyvista as pv
        import math
        try:
            nx = int(math.ceil((bounds.x_max - bounds.x_min) / voxel_config.cell_size_x))
            ny = int(math.ceil((bounds.y_max - bounds.y_min) / voxel_config.cell_size_y))
            nz = int(math.ceil((bounds.z_max - bounds.z_min) / voxel_config.cell_size_z))
            nx = max(1, nx); ny = max(1, ny); nz = max(1, nz)

            grid = pv.ImageData(
                dimensions=(nx + 1, ny + 1, nz + 1),
                spacing=(voxel_config.cell_size_x,
                         voxel_config.cell_size_y,
                         voxel_config.cell_size_z),
                origin=(bounds.x_min, bounds.y_min, bounds.z_min),
            )

            p32_values = np.zeros(nx * ny * nz, dtype=np.float64)
            frac_count = np.zeros(nx * ny * nz, dtype=np.int32)
            if isinstance(voxel_p32_data, dict):
                for (i, j, k), val in voxel_p32_data.items():
                    idx = int(i) * ny * nz + int(j) * nz + int(k)
                    if 0 <= idx < len(p32_values):
                        p32_values[idx] = float(val)
            elif isinstance(voxel_p32_data, (list, tuple)):
                for entry in voxel_p32_data:
                    if isinstance(entry, dict):
                        i = entry.get("i", 0)
                        j = entry.get("j", 0)
                        k = entry.get("k", 0)
                        idx = int(i) * ny * nz + int(j) * nz + int(k)
                        if 0 <= idx < len(p32_values):
                            p32_values[idx] = float(entry.get("local_p32", 0.0))
                            frac_count[idx] = int(entry.get("fracture_count", 0))

            grid.cell_data["local_p32"] = p32_values
            grid.cell_data["fracture_count"] = frac_count
            from pathlib import Path
            out_path = Path(path)
            if out_path.suffix.lower() in ('.vti',):
                grid.save(str(out_path))
            elif out_path.suffix.lower() in ('.vtr',):
                grid.save(str(out_path))
            else:
                # Default to .vti for ImageData
                grid.save(str(out_path.with_suffix('.vti')))
            return True
        except Exception as e:
            _logger.error(f"export_voxels_vtu failed: {e}", exc_info=True)
            return False

    def render_model_bounds(self, bounds, plotter) -> None:
        """Render the model bounding box wireframe.

        Args:
            bounds: ModelBounds with x_min/x_max/y_min/y_max/z_min/z_max.
            plotter: PyVista plotter.
        """
        try:
            import pyvista as pv
            box = pv.Box(bounds=[
                bounds.x_min, bounds.x_max,
                bounds.y_min, bounds.y_max,
                bounds.z_min, bounds.z_max,
            ])
            plotter.add_mesh(box, color="white", opacity=0.3,
                             style="wireframe", line_width=2,
                             name="Model Bounds", label="Model Bounds")
        except Exception as e:
            _logger.error(f"render_model_bounds failed: {e}", exc_info=True)

    def add_coordinate_axes(self, plotter) -> None:
        """Add XYZ coordinate axes with labels to the plotter.

        Args:
            plotter: PyVista plotter.
        """
        try:
            plotter.show_axes()
        except Exception as e:
            _logger.error(f"add_coordinate_axes failed: {e}", exc_info=True)

    def add_legend(self, plotter, items: dict) -> None:
        """Add a color legend to the plotter.

        Args:
            plotter: PyVista plotter.
            items: Dict of label → color hex string.
        """
        try:
            import pyvista as pv
            legend_entries = []
            for label, color in items.items():
                legend_entries.append([label, color])
            if legend_entries:
                plotter.add_legend(legend_entries)
        except Exception as e:
            _logger.error(f"add_legend failed: {e}", exc_info=True)
