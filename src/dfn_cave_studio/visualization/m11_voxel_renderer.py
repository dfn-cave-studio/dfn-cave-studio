"""Session-only interactive M11 P32 cloud rendering built on native VTK datasets."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyvista as pv

from dfn_cave_studio.models.m9 import ParameterFieldMetadata
from dfn_cave_studio.models.m11 import M11SecondVoxelizationResult
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager, M11LayerRecord
from dfn_cave_studio.visualization.parameter_field_renderer import ParameterFieldRenderer
from dfn_cave_studio.visualization.scalar_lut import make_lookup_table_opaque
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES

DisplayMode = Literal[
    "voxel_cells",
    "outer_surface",
    "orthogonal_section",
    "arbitrary_plane",
    "cutaway",
    "box_cutaway",
]
InterpolationMode = Literal["exact", "smooth"]
ColorMode = Literal["continuous", "discrete"]
RangeMode = Literal["global", "manual"]


@dataclass(frozen=True, slots=True)
class M11DisplayConfig:
    """Immutable display request; never part of the project schema."""

    display_mode: DisplayMode = "orthogonal_section"
    interpolation_mode: InterpolationMode = "exact"
    axis: str = "z"
    fraction: float = 0.5
    section_coordinate: float | None = None
    plane_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    flip_side: bool = False
    interactive_plane: bool = False
    show_grid_lines: bool = False
    range_mode: RangeMode = "global"
    manual_min: float | None = None
    manual_max: float | None = None
    color_mode: ColorMode = "continuous"
    contour_bands: int = 10
    opacity: float = 1.0
    interactive_slot: str = ""
    interaction_callback: Callable[..., None] | None = None
    interaction_end_callback: Callable[..., None] | None = None
    box_bounds: tuple[float, float, float, float, float, float] | None = None
    snap_box_to_voxel_faces: bool = True
    box_preview_continuous: bool = False


@dataclass(frozen=True, slots=True)
class M11PreparedDisplay:
    """Prepared VTK dataset and auditable display metadata."""

    mesh: pv.DataSet
    scalar_preference: Literal["cell", "point"]
    layer_id: str
    display_mode: str
    interpolation_mode: str
    axis: str
    slice_index: int
    coordinate: float
    plane_id: str
    color_range: tuple[float, float]


class M11VoxelRenderer:
    """Render exact or display-smoothed M11 clouds without mutating arrays."""

    ACTOR_PREFIX = "m11_voxel:"
    _CACHE_LIMIT = 4

    def __init__(self) -> None:
        self._location_helper = ParameterFieldRenderer()
        self._valid_grid_cache: OrderedDict[tuple[object, ...], pv.UnstructuredGrid] = OrderedDict()

    def clear_cache(self) -> None:
        """Release session preparation caches without touching scientific state."""
        self._valid_grid_cache.clear()

    def slice_location(
        self, metadata: ParameterFieldMetadata, axis: str, fraction: float
    ) -> tuple[int, float]:
        """Return the stable orthogonal slice index and cell-centre coordinate."""
        return self._location_helper.slice_location(metadata, axis.lower(), fraction)

    @staticmethod
    def normalize_plane_normal(normal: tuple[float, float, float]) -> tuple[float, float, float]:
        """Validate and normalize an arbitrary-plane normal."""
        vector = np.asarray(normal, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("Plane normal must contain three finite components")
        length = float(np.linalg.norm(vector))
        if length <= np.finfo(np.float64).eps:
            raise ValueError("Plane normal must be non-zero")
        normalized = vector / length
        return tuple(float(item) for item in normalized)

    @staticmethod
    def validate_manual_range(minimum: float, maximum: float) -> tuple[float, float]:
        """Validate a finite, increasing manual scalar range."""
        low, high = float(minimum), float(maximum)
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("Manual colour range requires finite min < max")
        return low, high

    def prepare_display(
        self,
        metadata: ParameterFieldMetadata,
        result: M11SecondVoxelizationResult,
        field_name: str,
        joint_set_id: int | None,
        config: M11DisplayConfig,
    ) -> M11PreparedDisplay:
        """Prepare native VTK geometry while preserving the source arrays."""
        self._validate_config(config)
        valid_grid = self._valid_grid(metadata, result, field_name)
        source: pv.DataSet = valid_grid
        scalar_preference: Literal["cell", "point"] = "cell"
        if config.interpolation_mode == "smooth":
            source = valid_grid.cell_data_to_point_data(pass_cell_data=False)
            scalar_preference = "point"

        axis = config.axis.lower()
        slice_index = -1
        coordinate = float("nan")
        plane_id = ""
        if config.display_mode == "voxel_cells":
            mesh = source
            axis = "volume"
        elif config.display_mode == "outer_surface":
            mesh = source.extract_surface(algorithm="dataset_surface")
            axis = "surface"
        elif config.display_mode == "orthogonal_section":
            if config.section_coordinate is None:
                slice_index, coordinate = self.slice_location(metadata, axis, config.fraction)
            else:
                coordinate = float(config.section_coordinate)
                axis_index = {"x": 0, "y": 1, "z": 2}[axis]
                lower = metadata.origin[axis_index]
                upper = lower + metadata.shape[axis_index] * metadata.spacing[axis_index]
                if not math.isfinite(coordinate) or not lower <= coordinate <= upper:
                    raise ValueError("Orthogonal section coordinate is outside the Analysis Domain")
                floating_index = (coordinate - lower) / metadata.spacing[axis_index] - 0.5
                nearest = round(floating_index)
                slice_index = nearest if 0 <= nearest < metadata.shape[axis_index] and abs(floating_index - nearest) < 1e-8 else -1
            normal = tuple(1.0 if index == {"x": 0, "y": 1, "z": 2}[axis] else 0.0 for index in range(3))
            origin = list(source.center)
            origin[{"x": 0, "y": 1, "z": 2}[axis]] = coordinate
            mesh = source.slice(normal=normal, origin=origin)
            if slice_index < 0:
                plane_id = self._plane_hash(tuple(origin), normal)
        elif config.display_mode == "arbitrary_plane":
            normal = self.normalize_plane_normal(config.plane_normal)
            origin = self._validate_plane_origin(config.plane_origin)
            plane_id = self._plane_hash(origin, normal)
            coordinate = float(np.dot(origin, normal))
            axis = "plane"
            mesh = source.slice(normal=normal, origin=origin)
        elif config.display_mode == "cutaway":
            normal = self.normalize_plane_normal(config.plane_normal)
            origin = self._validate_plane_origin(config.plane_origin)
            side_token = "negative" if config.flip_side else "positive"
            plane_id = f"{self._plane_hash(origin, normal)}_{side_token}"
            coordinate = float(np.dot(origin, normal))
            axis = "cutaway"
            # Clip the valid volume first and only then extract its surface.  This
            # retains the newly exposed, scalar-valued internal face; clipping an
            # already extracted outer shell would leave an unauditable open hole.
            clipped_volume = source.clip(normal=normal, origin=origin, invert=config.flip_side)
            mesh = clipped_volume.extract_surface(algorithm="dataset_surface")
        else:
            analysis_bounds = self.analysis_bounds(metadata)
            bounds = self.validate_box_bounds(config.box_bounds, analysis_bounds)
            effective_snap = config.snap_box_to_voxel_faces and not config.box_preview_continuous
            if effective_snap:
                bounds = self.snap_box_bounds(metadata, bounds)
            clipped_volume = source.clip_box(
                bounds=bounds,
                invert=False,
                crinkle=effective_snap,
            )
            if clipped_volume.n_cells == 0:
                raise ValueError("Box Cutaway does not overlap any valid cells for the selected field")
            mesh = clipped_volume.extract_surface(algorithm="dataset_surface")
            axis = "box"
            coordinate = float("nan")
            snap_token = "snap" if config.snap_box_to_voxel_faces else "continuous"
            plane_id = f"box_{self._bounds_hash(bounds)}_{snap_token}"

        color_range = self._color_range(result, field_name, config)
        if config.interactive_slot:
            plane_id = f"slot_{config.interactive_slot}"
            if config.display_mode == "box_cutaway":
                plane_id += "_snap" if config.snap_box_to_voxel_faces else "_continuous"
        layer_id = M11LayerManager.layer_id(
            result.realization_id,
            field_name,
            joint_set_id,
            axis,
            slice_index,
            display_mode=config.display_mode,
            interpolation_mode=config.interpolation_mode,
            plane_id=plane_id,
        )
        return M11PreparedDisplay(
            mesh=mesh,
            scalar_preference=scalar_preference,
            layer_id=layer_id,
            display_mode=config.display_mode,
            interpolation_mode=config.interpolation_mode,
            axis=axis,
            slice_index=slice_index,
            coordinate=coordinate,
            plane_id=plane_id,
            color_range=color_range,
        )

    def render(
        self,
        layer_manager: M11LayerManager,
        metadata: ParameterFieldMetadata,
        result: M11SecondVoxelizationResult,
        field_name: str,
        joint_set_id: int | None,
        config: M11DisplayConfig,
        *,
        pending: bool = False,
    ) -> M11LayerRecord:
        """Render one cloud and atomically register its actor, colour bar, and widget."""
        prepared = self.prepare_display(metadata, result, field_name, joint_set_id, config)
        color_signature = self._color_signature(config, prepared.color_range)
        scalar_bar_id = layer_manager.scalar_bar_id(
            result.realization_id, field_name, joint_set_id, color_signature
        )
        scalar_bar_title = f"{field_name} (m^-1)"
        actor = layer_manager.plotter.add_mesh(
            prepared.mesh,
            scalars=field_name,
            preference=prepared.scalar_preference,
            cmap="viridis",
            clim=prepared.color_range,
            n_colors=256 if config.color_mode == "continuous" else config.contour_bands,
            interpolate_before_map=config.interpolation_mode == "smooth" and config.color_mode == "continuous",
            opacity=config.opacity,
            nan_opacity=1.0,
            show_edges=config.show_grid_lines
            and not (
                config.display_mode == "box_cutaway"
                and (not config.snap_box_to_voxel_faces or config.box_preview_continuous)
            ),
            edge_color="#303030",
            line_width=0.5,
            show_scalar_bar=False,
            name=prepared.layer_id,
        )
        make_lookup_table_opaque(actor)
        widgets: tuple[object, ...] = ()
        widget_observer_ids: tuple[tuple[object, int], ...] = ()
        try:
            if config.display_mode in {"orthogonal_section", "arbitrary_plane", "cutaway"} and config.interactive_plane:
                widgets = self._add_plane_widget(layer_manager, metadata, config)
            elif config.display_mode == "box_cutaway" and config.interactive_plane:
                widgets, widget_observer_ids = self._add_box_widget(layer_manager, metadata, config)
            scalar_bar_actor = layer_manager.create_or_get_scalar_bar(
                scalar_bar_id,
                scalar_bar_title,
                actor,
                attach_mapper=not layer_manager.contains(prepared.layer_id),
            )
            return layer_manager.add_or_replace(
                prepared.layer_id,
                actor,
                scalar_bar_id=scalar_bar_id,
                scalar_bar_actor=scalar_bar_actor,
                scalar_bar_title=scalar_bar_title,
                realization_id=result.realization_id,
                display_mode=prepared.display_mode,
                interpolation_mode=prepared.interpolation_mode,
                field_name=field_name,
                joint_set_id=joint_set_id,
                axis=prepared.axis,
                slice_index=prepared.slice_index,
                coordinate=prepared.coordinate,
                plane_id=prepared.plane_id,
                show_grid_lines=config.show_grid_lines
                and not (
                    config.display_mode == "box_cutaway"
                    and (not config.snap_box_to_voxel_faces or config.box_preview_continuous)
                ),
                color_range=prepared.color_range,
                widgets=widgets,
                widget_observer_ids=widget_observer_ids,
                opacity=config.opacity,
                pending=pending,
            )
        except Exception:
            layer_manager.discard_unregistered(actor, scalar_bar_id, widgets, widget_observer_ids)
            raise

    def render_slice(
        self,
        layer_manager: M11LayerManager,
        metadata: ParameterFieldMetadata,
        result: M11SecondVoxelizationResult,
        field_name: str,
        axis: str,
        fraction: float,
        *,
        opacity: float = 1.0,
        joint_set_id: int | None = None,
        pending: bool = False,
    ) -> M11LayerRecord:
        """Backward-compatible exact orthogonal-section rendering entry point."""
        return self.render(
            layer_manager,
            metadata,
            result,
            field_name,
            joint_set_id,
            M11DisplayConfig(axis=axis.lower(), fraction=fraction, opacity=opacity),
            pending=pending,
        )

    def _valid_grid(
        self,
        metadata: ParameterFieldMetadata,
        result: M11SecondVoxelizationResult,
        field_name: str,
    ) -> pv.UnstructuredGrid:
        if field_name not in result.arrays:
            raise ValueError(f"M11 result does not contain field: {field_name}")
        values = np.asarray(result.arrays[field_name])
        states = np.asarray(result.arrays.get("cell_state"))
        if values.shape != metadata.shape or states.shape != metadata.shape:
            raise ValueError("M11 field and cell-state shapes must match the voxel metadata")
        key = (
            id(result),
            result.realization_id,
            field_name,
            metadata.shape,
            metadata.origin,
            metadata.spacing,
        )
        cached = self._valid_grid_cache.get(key)
        if cached is not None:
            self._valid_grid_cache.move_to_end(key)
            return cached
        hidden = np.isin(
            states,
            [
                CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL],
                CELL_STATE_CODES[VoxelCellState.NO_DATA],
                CELL_STATE_CODES[VoxelCellState.EXCAVATION],
            ],
        )
        display_values = np.asarray(values, dtype=np.float32).copy()
        finite = np.isfinite(display_values)
        valid = ~hidden & finite
        grid = pv.ImageData(
            dimensions=tuple(size + 1 for size in metadata.shape),
            spacing=metadata.spacing,
            origin=metadata.origin,
        )
        grid.cell_data[field_name] = display_values.ravel(order="F")
        grid.cell_data["__m11_valid__"] = valid.astype(np.uint8).ravel(order="F")
        valid_grid = grid.threshold((1, 1), scalars="__m11_valid__", preference="cell")
        del valid_grid.cell_data["__m11_valid__"]
        self._valid_grid_cache[key] = valid_grid
        while len(self._valid_grid_cache) > self._CACHE_LIMIT:
            self._valid_grid_cache.popitem(last=False)
        return valid_grid

    def _color_range(
        self, result: M11SecondVoxelizationResult, field_name: str, config: M11DisplayConfig
    ) -> tuple[float, float]:
        if config.range_mode == "manual":
            if config.manual_min is None or config.manual_max is None:
                raise ValueError("Manual colour range requires both min and max")
            return self.validate_manual_range(config.manual_min, config.manual_max)
        values = np.asarray(result.arrays[field_name], dtype=np.float64)
        states = np.asarray(result.arrays["cell_state"])
        valid = ~np.isin(
            states,
            [
                CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL],
                CELL_STATE_CODES[VoxelCellState.NO_DATA],
                CELL_STATE_CODES[VoxelCellState.EXCAVATION],
            ],
        )
        finite = values[valid & np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("The selected M11 field has no finite modelled values")
        low, high = float(np.min(finite)), float(np.max(finite))
        if low == high:
            high = float(np.nextafter(high, math.inf))
        return low, high

    @staticmethod
    def _validate_config(config: M11DisplayConfig) -> None:
        if config.display_mode not in {
            "voxel_cells",
            "outer_surface",
            "orthogonal_section",
            "arbitrary_plane",
            "cutaway",
            "box_cutaway",
        }:
            raise ValueError("Unsupported M11 display mode")
        if config.interpolation_mode not in {"exact", "smooth"}:
            raise ValueError("Unsupported display interpolation mode")
        if config.axis.lower() not in {"x", "y", "z"}:
            raise ValueError("Orthogonal section axis must be X, Y, or Z")
        if not 0.0 <= config.fraction <= 1.0:
            raise ValueError("Section position must be between zero and one")
        if config.color_mode not in {"continuous", "discrete"}:
            raise ValueError("Unsupported colour-map mode")
        if not 5 <= config.contour_bands <= 30:
            raise ValueError("Discrete contour bands must be between 5 and 30")
        if not 0.0 <= config.opacity <= 1.0:
            raise ValueError("Opacity must be between zero and one")
        if config.display_mode == "box_cutaway":
            M11VoxelRenderer.validate_box_bounds(config.box_bounds)

    @staticmethod
    def validate_box_bounds(
        bounds: tuple[float, float, float, float, float, float] | None,
        analysis_bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """Validate an axis-aligned display cutaway box that retains its interior."""
        if bounds is None:
            raise ValueError("Box Cutaway requires six bounds")
        values = np.asarray(bounds, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("Box Cutaway bounds must contain six finite values")
        if values[0] >= values[1] or values[2] >= values[3] or values[4] >= values[5]:
            raise ValueError("Box Cutaway requires min < max on X, Y, and Z")
        if analysis_bounds is not None:
            domain = np.asarray(analysis_bounds, dtype=np.float64)
            if domain.shape != (6,) or not np.all(np.isfinite(domain)):
                raise ValueError("Analysis Domain bounds must contain six finite values")
            domain_min = domain[[0, 2, 4]]
            domain_max = domain[[1, 3, 5]]
            extent = domain_max - domain_min
            if np.any(extent <= 0.0):
                raise ValueError("Analysis Domain requires min < max on X, Y, and Z")
            tolerance = max(float(np.max(extent)) * 1e-9, np.finfo(np.float64).eps)
            value_min = values[[0, 2, 4]]
            value_max = values[[1, 3, 5]]
            if np.any(value_min < domain_min - tolerance) or np.any(value_max > domain_max + tolerance):
                raise ValueError("Box Cutaway bounds must remain inside the Analysis Domain")
            # VTK and UI round-trips may move an exact outer face by a few ULPs.
            # Only values within the scale-dependent tolerance are normalized;
            # no inward epsilon is introduced and no boundary voxel is removed.
            for minimum_index, maximum_index in ((0, 1), (2, 3), (4, 5)):
                if abs(values[minimum_index] - domain[minimum_index]) <= tolerance:
                    values[minimum_index] = domain[minimum_index]
                if abs(values[maximum_index] - domain[maximum_index]) <= tolerance:
                    values[maximum_index] = domain[maximum_index]
            if values[0] >= values[1] or values[2] >= values[3] or values[4] >= values[5]:
                raise ValueError("Box Cutaway requires min < max on X, Y, and Z")
        return tuple(float(item) for item in values)

    @staticmethod
    def analysis_bounds(
        metadata: ParameterFieldMetadata,
    ) -> tuple[float, float, float, float, float, float]:
        """Return first/last authoritative voxel edges of the Analysis Domain."""
        edges = [M11VoxelRenderer.voxel_axis_edges(metadata, axis) for axis in range(3)]
        return (
            float(edges[0][0]),
            float(edges[0][-1]),
            float(edges[1][0]),
            float(edges[1][-1]),
            float(edges[2][0]),
            float(edges[2][-1]),
        )

    @staticmethod
    def voxel_axis_edges(metadata: ParameterFieldMetadata, axis: int) -> np.ndarray:
        """Build one authoritative float64 voxel-edge array from grid metadata."""
        if axis not in (0, 1, 2):
            raise ValueError("Voxel edge axis must be 0, 1, or 2")
        origin = float(metadata.origin[axis])
        spacing = float(metadata.spacing[axis])
        count = int(metadata.shape[axis])
        return origin + np.arange(count + 1, dtype=np.float64) * spacing

    @staticmethod
    def snap_box_bounds(
        metadata: ParameterFieldMetadata,
        bounds: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        """Snap display bounds to actual voxel faces inside the Analysis Domain."""
        validated = M11VoxelRenderer.validate_box_bounds(
            bounds, M11VoxelRenderer.analysis_bounds(metadata)
        )
        snapped: list[float] = []
        for axis in range(3):
            edges = M11VoxelRenderer.voxel_axis_edges(metadata, axis)
            origin = float(edges[0])
            spacing = float(metadata.spacing[axis])
            count = int(metadata.shape[axis])
            low = round((validated[2 * axis] - origin) / spacing)
            high = round((validated[2 * axis + 1] - origin) / spacing)
            low = min(max(low, 0), count)
            high = min(max(high, 0), count)
            if high <= low:
                if high < count:
                    high = low + 1
                elif low > 0:
                    low = high - 1
                else:
                    raise ValueError("Box Cutaway is too narrow to snap to a voxel interval")
            snapped.extend((float(edges[low]), float(edges[high])))
        return tuple(snapped)

    @staticmethod
    def _validate_plane_origin(origin: tuple[float, float, float]) -> tuple[float, float, float]:
        values = np.asarray(origin, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("Plane origin must contain three finite coordinates")
        return tuple(float(item) for item in values)

    @staticmethod
    def _plane_hash(
        origin: tuple[float, float, float], normal: tuple[float, float, float]
    ) -> str:
        stable = ",".join(f"{item:.9g}" for item in (*origin, *normal))
        return f"plane_{hashlib.sha256(stable.encode('ascii')).hexdigest()[:16]}"

    @staticmethod
    def _bounds_hash(bounds: tuple[float, ...] | None) -> str:
        stable = ",".join(f"{item:.9g}" for item in (bounds or ()))
        return f"clip_{hashlib.sha256(stable.encode('ascii')).hexdigest()[:12]}"

    @staticmethod
    def _color_signature(config: M11DisplayConfig, color_range: tuple[float, float]) -> str:
        range_token = "global" if config.range_mode == "global" else f"manual_{color_range[0]:.9g}_{color_range[1]:.9g}"
        mode_token = "continuous" if config.color_mode == "continuous" else f"discrete_{config.contour_bands}"
        return f"{range_token}:{mode_token}"

    @staticmethod
    def _add_plane_widget(
        layer_manager: M11LayerManager,
        metadata: ParameterFieldMetadata,
        config: M11DisplayConfig,
    ) -> tuple[object, ...]:
        creator = getattr(layer_manager.plotter, "add_plane_widget", None)
        if creator is None:
            raise ValueError("The active plotter does not support interactive plane widgets")
        if config.display_mode == "orthogonal_section":
            axis_index = {"x": 0, "y": 1, "z": 2}[config.axis.lower()]
            normal = tuple(1.0 if index == axis_index else 0.0 for index in range(3))
            if config.section_coordinate is None:
                _, coordinate = ParameterFieldRenderer.slice_location(metadata, config.axis.lower(), config.fraction)
            else:
                coordinate = float(config.section_coordinate)
            origin_values = [
                metadata.origin[index] + metadata.shape[index] * metadata.spacing[index] / 2.0
                for index in range(3)
            ]
            origin_values[axis_index] = coordinate
            origin = tuple(origin_values)
        else:
            normal = M11VoxelRenderer.normalize_plane_normal(config.plane_normal)
            origin = M11VoxelRenderer._validate_plane_origin(config.plane_origin)
        maximum = tuple(
            metadata.origin[index] + metadata.shape[index] * metadata.spacing[index]
            for index in range(3)
        )
        bounds = (
            metadata.origin[0], maximum[0], metadata.origin[1], maximum[1], metadata.origin[2], maximum[2]
        )
        callback = config.interaction_callback or (lambda *_: None)
        widget_options = {
            "normal": normal,
            "origin": origin,
            "bounds": bounds,
            "interaction_event": "always",
        }
        if config.display_mode == "orthogonal_section":
            widget_options.update(
                normal_rotation=False,
                assign_to_axis=config.axis.lower(),
                origin_translation=True,
            )
        widget = creator(callback, **widget_options)
        return () if widget is None else (widget,)

    @staticmethod
    def _add_box_widget(
        layer_manager: M11LayerManager,
        metadata: ParameterFieldMetadata,
        config: M11DisplayConfig,
    ) -> tuple[tuple[object, ...], tuple[tuple[object, int], ...]]:
        creator = getattr(layer_manager.plotter, "add_box_widget", None)
        if creator is None:
            raise ValueError("The active plotter does not support interactive box widgets")
        bounds = M11VoxelRenderer.validate_box_bounds(
            config.box_bounds,
            M11VoxelRenderer.analysis_bounds(metadata),
        )
        if config.snap_box_to_voxel_faces:
            bounds = M11VoxelRenderer.snap_box_bounds(metadata, bounds)
        callback = config.interaction_callback or (lambda *_: None)
        creating = True

        def interaction_callback(box: object, widget: object | None = None) -> None:
            # PyVista invokes the callback once from inside add_box_widget,
            # before the widget can be registered.  That is initialization,
            # not a user interaction.
            if not creating:
                callback(box, widget)

        widget = creator(
            interaction_callback,
            bounds=bounds,
            factor=1.0,
            rotation_enabled=False,
            outline_translation=True,
            pass_widget=True,
            interaction_event="always",
        )
        creating = False
        if widget is None:
            return (), ()

        set_rotation = getattr(widget, "SetRotationEnabled", None)
        if set_rotation is not None:
            set_rotation(False)
        set_translation = getattr(widget, "SetTranslationEnabled", None)
        if set_translation is not None:
            set_translation(True)
        enabled = getattr(widget, "SetEnabled", None)
        if enabled is not None:
            enabled(True)
        turn_on = getattr(widget, "On", None)
        if turn_on is not None:
            turn_on()

        observer_ids: tuple[tuple[object, int], ...] = ()
        end_callback = config.interaction_end_callback
        add_observer = getattr(widget, "AddObserver", None)
        if end_callback is not None and add_observer is not None:

            def interaction_ended(caller: object, _event: object) -> None:
                box = pv.PolyData()
                get_poly_data = getattr(caller, "GetPolyData", None)
                if get_poly_data is None:
                    return
                get_poly_data(box)
                end_callback(box, caller)

            observer_id = int(add_observer("EndInteractionEvent", interaction_ended))
            observer_ids = ((widget, observer_id),)
        return (widget,), observer_ids

    @staticmethod
    def estimated_display_bytes(prepared: M11PreparedDisplay) -> int:
        """Return the native mesh array bytes retained for one display object."""
        total = 0
        for mapping in (prepared.mesh.point_data, prepared.mesh.cell_data):
            total += sum(np.asarray(array).nbytes for array in mapping.values())
        points = getattr(prepared.mesh, "points", None)
        if points is not None:
            total += np.asarray(points).nbytes
        return int(total)
