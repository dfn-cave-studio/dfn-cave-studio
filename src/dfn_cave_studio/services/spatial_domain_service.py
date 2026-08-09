"""Pure M8 calculations for domains, coverage checks, and grid summaries."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from dfn_cave_studio.models.borehole import BoreholeCollection
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.spatial_grid import (
    BoundaryViolationReport,
    GridMemoryField,
    SpatialGridConfig,
    VoxelGridSummary,
)

DEFAULT_MEMORY_FIELDS = [
    GridMemoryField(name="cell_state", bytes_per_voxel=1),
    GridMemoryField(name="active_mask", bytes_per_voxel=1),
    GridMemoryField(name="value", bytes_per_voxel=4),
    GridMemoryField(name="uncertainty", bytes_per_voxel=4),
]


class SpatialDomainService:
    """Compute M8 spatial definitions without allocating voxel arrays."""

    @staticmethod
    def trajectory_points(collection: BoreholeCollection, step_length: float = 1.0) -> dict[str, np.ndarray]:
        """Return full collar-to-toe trajectories for every borehole."""
        return {
            borehole.borehole_id: borehole.compute_trajectory(step_length=step_length)[0] for borehole in collection
        }

    @staticmethod
    def observation_points(collection: BoreholeCollection) -> dict[str, list[np.ndarray]]:
        """Return located formal fracture observation points by borehole."""
        result: dict[str, list[np.ndarray]] = {}
        for borehole in collection:
            points = []
            for observation in borehole.fracture_observations:
                point = borehole.locate_observation(observation)
                if point is not None:
                    points.append(point)
            result[borehole.borehole_id] = points
        return result

    @classmethod
    def automatic_bounds(
        cls,
        collection: BoreholeCollection,
        outward_margin: float = 0.0,
        extra_points: Iterable[tuple[float, float, float]] = (),
    ) -> ModelBounds:
        """Bound complete trajectories and valid observation/extension points."""
        if outward_margin < 0:
            raise ValueError("outward_margin must be non-negative")
        arrays = list(cls.trajectory_points(collection).values())
        observations = [point for points in cls.observation_points(collection).values() for point in points]
        points = [point for array in arrays for point in array]
        points.extend(observations)
        points.extend(np.asarray(point, dtype=float) for point in extra_points)
        if not points:
            raise ValueError("At least one spatial point is required for automatic bounds")
        matrix = np.asarray(points, dtype=float)
        minima = matrix.min(axis=0) - outward_margin
        maxima = matrix.max(axis=0) + outward_margin
        # ModelBounds requires positive extents even for a vertical or single-point hole.
        for axis in range(3):
            if maxima[axis] - minima[axis] < 1e-9:
                minima[axis] -= 0.5
                maxima[axis] += 0.5
        return ModelBounds(
            x_min=float(minima[0]),
            x_max=float(maxima[0]),
            y_min=float(minima[1]),
            y_max=float(maxima[1]),
            z_min=float(minima[2]),
            z_max=float(maxima[2]),
        )

    @classmethod
    def check_bounds(
        cls,
        bounds: ModelBounds,
        collection: BoreholeCollection,
        extra_points: Iterable[tuple[str, tuple[float, float, float]]] = (),
    ) -> BoundaryViolationReport:
        """Check trajectories and observations and recommend an expanded domain."""
        trajectories = cls.trajectory_points(collection)
        observations = cls.observation_points(collection)
        exceedance = {key: 0.0 for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
        affected: set[str] = set()
        trajectory_count = 0
        observation_count = 0
        all_points: list[np.ndarray] = []

        for hole_id, points in trajectories.items():
            all_points.extend(points)
            outside = [point for point in points if not bounds.contains_point(*point)]
            trajectory_count += len(outside)
            if outside:
                affected.add(hole_id)
            for point in outside:
                cls._accumulate_exceedance(exceedance, bounds, point)
        for hole_id, points in observations.items():
            all_points.extend(points)
            outside = [point for point in points if not bounds.contains_point(*point)]
            observation_count += len(outside)
            if outside:
                affected.add(hole_id)
            for point in outside:
                cls._accumulate_exceedance(exceedance, bounds, point)
        for hole_id, point_tuple in extra_points:
            point = np.asarray(point_tuple, dtype=float)
            all_points.append(point)
            if not bounds.contains_point(*point):
                observation_count += 1
                affected.add(hole_id)
                cls._accumulate_exceedance(exceedance, bounds, point)

        if all_points:
            matrix = np.asarray(all_points)
            recommended = ModelBounds(
                x_min=min(bounds.x_min, float(matrix[:, 0].min())),
                x_max=max(bounds.x_max, float(matrix[:, 0].max())),
                y_min=min(bounds.y_min, float(matrix[:, 1].min())),
                y_max=max(bounds.y_max, float(matrix[:, 1].max())),
                z_min=min(bounds.z_min, float(matrix[:, 2].min())),
                z_max=max(bounds.z_max, float(matrix[:, 2].max())),
            )
        else:
            recommended = bounds.model_copy()
        return BoundaryViolationReport(
            outside_borehole_count=len(affected),
            outside_trajectory_point_count=trajectory_count,
            outside_observation_point_count=observation_count,
            maximum_exceedance=exceedance,
            affected_holes=sorted(affected),
            recommended_domain=recommended,
        )

    @staticmethod
    def generation_domain(
        analysis_domain: ModelBounds,
        voxel: VoxelConfig,
        maximum_fracture_radius: float,
        buffer_layers: int,
    ) -> ModelBounds:
        """Expand analysis bounds using max(radius, layers * axis cell size)."""
        if maximum_fracture_radius < 0 or buffer_layers < 0:
            raise ValueError("Generation-domain buffers must be non-negative")
        bx = max(maximum_fracture_radius, buffer_layers * voxel.cell_size_x)
        by = max(maximum_fracture_radius, buffer_layers * voxel.cell_size_y)
        bz = max(maximum_fracture_radius, buffer_layers * voxel.cell_size_z)
        return ModelBounds(
            x_min=analysis_domain.x_min - bx,
            x_max=analysis_domain.x_max + bx,
            y_min=analysis_domain.y_min - by,
            y_max=analysis_domain.y_max + by,
            z_min=analysis_domain.z_min - bz,
            z_max=analysis_domain.z_max + bz,
        )

    @staticmethod
    def grid_summary(
        bounds: ModelBounds,
        voxel: VoxelConfig,
        fields: list[GridMemoryField] | None = None,
        active_voxels: int | None = None,
        warning_limit_bytes: int = 2 * 1024**3,
    ) -> VoxelGridSummary:
        """Compute ceil-covered dimensions and byte estimate without arrays."""
        selected_fields = fields or DEFAULT_MEMORY_FIELDS
        nx, ny, nz = voxel.compute_grid_dimensions(bounds)
        total = nx * ny * nz
        if active_voxels is not None and active_voxels > total:
            raise ValueError("active_voxels cannot exceed total_voxels")
        estimated = total * sum(field.bytes_per_voxel for field in selected_fields)
        warning = None
        if estimated > warning_limit_bytes:
            warning = (
                f"Estimated dense allocation is {estimated / 1024**3:.2f} GiB, "
                f"above the {warning_limit_bytes / 1024**3:.2f} GiB warning limit"
            )
        return VoxelGridSummary(
            nx=nx,
            ny=ny,
            nz=nz,
            total_voxels=total,
            active_voxels=active_voxels,
            estimated_bytes=estimated,
            fields=selected_fields,
            warning=warning,
        )

    @staticmethod
    def box_mask_active_voxels(bounds: ModelBounds, voxel: VoxelConfig, rock_mask) -> int | None:
        """Count cells intersecting an enabled box mask without allocating it."""
        required = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        if rock_mask is None or not getattr(rock_mask, "enabled", False):
            return None
        mask_type = getattr(rock_mask, "mask_type", "")
        mask_value = mask_type.value if hasattr(mask_type, "value") else str(mask_type)
        if mask_value != "box" or any(getattr(rock_mask, name, None) is None for name in required):
            return None
        dimensions = voxel.compute_grid_dimensions(bounds)
        counts = []
        for axis, cell_size, total in zip(
            ("x", "y", "z"),
            (voxel.cell_size_x, voxel.cell_size_y, voxel.cell_size_z),
            dimensions,
            strict=True,
        ):
            domain_min = getattr(bounds, f"{axis}_min")
            domain_max = getattr(bounds, f"{axis}_max")
            mask_min = max(domain_min, float(getattr(rock_mask, f"{axis}_min")))
            mask_max = min(domain_max, float(getattr(rock_mask, f"{axis}_max")))
            if mask_min >= mask_max:
                return 0
            first = max(0, math.floor((mask_min - domain_min) / cell_size))
            stop = min(total, math.ceil((mask_max - domain_min) / cell_size))
            counts.append(max(0, stop - first))
        return counts[0] * counts[1] * counts[2]

    @classmethod
    def build_config(
        cls,
        analysis_domain: ModelBounds,
        voxel: VoxelConfig,
        boundary_mode: str,
        outward_margin: float,
        buffer_layers: int,
        maximum_fracture_radius: float,
        clipping_acknowledged: bool = False,
    ) -> SpatialGridConfig:
        """Create a valid persisted analysis/generation domain pair."""
        generation = cls.generation_domain(
            analysis_domain,
            voxel,
            maximum_fracture_radius=maximum_fracture_radius,
            buffer_layers=buffer_layers,
        )
        return SpatialGridConfig(
            analysis_domain=analysis_domain,
            generation_domain=generation,
            boundary_mode=boundary_mode,
            outward_margin=outward_margin,
            buffer_layers=buffer_layers,
            maximum_fracture_radius=maximum_fracture_radius,
            clipping_acknowledged=clipping_acknowledged,
        )

    @staticmethod
    def _accumulate_exceedance(
        result: dict[str, float],
        bounds: ModelBounds,
        point: np.ndarray,
    ) -> None:
        result["x_min"] = max(result["x_min"], bounds.x_min - float(point[0]))
        result["x_max"] = max(result["x_max"], float(point[0]) - bounds.x_max)
        result["y_min"] = max(result["y_min"], bounds.y_min - float(point[1]))
        result["y_max"] = max(result["y_max"], float(point[1]) - bounds.y_max)
        result["z_min"] = max(result["z_min"], bounds.z_min - float(point[2]))
        result["z_max"] = max(result["z_max"], float(point[2]) - bounds.z_max)
