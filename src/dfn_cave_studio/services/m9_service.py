"""Project-level orchestration for the M9 parameter-field workflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from dfn_cave_studio.dfn.intensity import build_p10_intervals, domain_at_depth, estimate_p32, expected_orientation_exposure
from dfn_cave_studio.dfn.size_models import MLESizeModelFitter, assumed_size_model
from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal, normal_to_dip_dir_dip
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.m9 import (
    DensitySettings,
    DomainOrientationModel,
    ParameterFieldMetadata,
    ValidationIntervalResult,
    ValidationState,
    ValidationSummary,
)
from dfn_cave_studio.services.m7_state import get_domain_intervals, get_holdout
from dfn_cave_studio.voxel.parameter_field import ParameterFieldBuilder


@dataclass(frozen=True)
class _NearestDomainClassifier:
    coordinates: np.ndarray
    labels: tuple[int, ...]
    _tree: cKDTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) != len(self.labels):
            raise ValueError("Domain centres must be a finite (N,3) array matching labels")
        if len(coordinates) == 0 or np.any(~np.isfinite(coordinates)):
            raise ValueError("Domain centres must be a non-empty finite array")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "_tree", cKDTree(coordinates))

    def __call__(self, point: tuple[float, float, float]) -> int:
        return int(self.predict_many(np.asarray([point], dtype=np.float64))[0])

    def predict_many(self, points: np.ndarray) -> np.ndarray:
        """Classify a bounded batch without a point-by-all-centres matrix."""
        targets = np.asarray(points, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 3 or np.any(~np.isfinite(targets)):
            raise ValueError("Domain query points must be a finite (N,3) array")
        if len(targets) == 0:
            return np.empty(0, dtype=np.int64)
        query_count = min(2, len(self.coordinates))
        distances, indices = self._tree.query(targets, k=query_count)
        distances = np.asarray(distances, dtype=np.float64).reshape(len(targets), query_count)
        indices = np.asarray(indices, dtype=np.int64).reshape(len(targets), query_count)
        selected = indices[:, 0].copy()
        if query_count == 2:
            tie_rows = np.flatnonzero(
                np.isclose(distances[:, 0], distances[:, 1], rtol=1e-12, atol=1e-12)
            )
            for row in tie_rows:
                radius = distances[row, 0] + max(1.0, distances[row, 0]) * 1e-12
                candidates = np.asarray(self._tree.query_ball_point(targets[row], radius), dtype=np.int64)
                exact_distances = np.linalg.norm(self.coordinates[candidates] - targets[row], axis=1)
                order = np.lexsort((candidates, exact_distances))
                selected[row] = candidates[order[0]]
        return np.asarray(self.labels, dtype=np.int64)[selected]


@dataclass(frozen=True)
class _ActiveRockClassifier:
    rock_mask: Any
    excavation_mask: Any

    def __call__(self, point: tuple[float, float, float]) -> bool:
        return self.rock_mask.contains_point(*point) and not self.excavation_mask.is_excavated(*point)

    def predict_many(self, points: np.ndarray) -> np.ndarray:
        """Classify rock/excavation state through mask batch protocols."""
        return self.rock_mask.contains_points(points) & ~self.excavation_mask.contains_points(points)


class M9Service:
    """Keep M9 scientific work out of Qt and mutate only ``project.m9_state``."""

    def __init__(self, project: Any) -> None:
        self.project = project

    def _roles(self) -> dict[str, str]:
        holdout = get_holdout(self.project)
        if holdout is None or not holdout.is_locked:
            raise RuntimeError("Validation Holdout must be locked before M9 fitting")
        return {
            hole.borehole_id: "validation" if holdout.is_validation(hole.borehole_id) else "calibration"
            for hole in self.project.borehole_collection.boreholes
        }

    def calculate_density(self, settings: DensitySettings | None = None, *, progress=None, cancelled=None) -> None:
        """Compute P10 and direction-corrected P32 with no validation leakage."""
        compatible_count = sum(
            len(hole.fracture_observations) for hole in self.project.borehole_collection.boreholes
        )
        database = getattr(self.project, "borehole_database", None)
        counts = database.observation_mode_counts() if database is not None else {}
        spacing = int(counts.get("interval_spacing", 0))
        relative = int(counts.get("axis_plane_angle", 0))
        if compatible_count == 0:
            raise ValueError(
                "No global fracture orientations are available for the existing P32 workflow. "
                f"Imported interval-spacing records={spacing}; borehole-relative angle records={relative}. "
                "Their import and derived metrics are retained, but direction completion is not implemented in this phase."
            )
        settings = settings or self.project.m9_state.density_settings
        roles = self._roles()
        intervals = build_p10_intervals(
            self.project.borehole_collection,
            roles,
            get_domain_intervals(self.project),
            interval_length=settings.interval_length,
            interval_mode=settings.interval_mode,
            set_ids=[item.set_id for item in self.project.joint_sets],
        )
        orientation_models, domain_sets = self._fit_domain_orientations(roles)
        estimates = estimate_p32(
            intervals,
            self.project.joint_sets,
            random_seed=self.project.m9_state.random_seed,
            sample_count=settings.monte_carlo_samples,
            low_observability_threshold=settings.low_observability_threshold,
            joint_sets_by_domain=domain_sets,
            progress=progress,
            cancelled=cancelled,
        )
        if cancelled and cancelled():
            raise InterruptedError("density estimation cancelled")
        self.project.m9_state.density_settings = settings
        self.project.m9_state.p10_intervals = intervals
        self.project.m9_state.p32_estimates = estimates
        self.project.m9_state.orientation_models = orientation_models
        self.project.m9_state.parameter_field_metadata = None
        self.project.m9_state.parameter_field_arrays = {}
        self.project.m9_state.validation_results = []
        self.project.m9_state.validation_summary = ValidationSummary()
        self.project.m9_state.provenance["density_fit_holes"] = sorted(
            hole_id for hole_id, role in roles.items() if role == "calibration"
        )
        self.project.m9_state.provenance["validation_holes_excluded"] = sorted(
            hole_id for hole_id, role in roles.items() if role == "validation"
        )
        self.project.m9_state.provenance["non_global_orientation_records_excluded_from_density_fit"] = {
            "interval_spacing": spacing,
            "axis_plane_angle": relative,
            "reason": "direction completion is not implemented in this phase",
        }

    def _fit_domain_orientations(self, roles: dict[str, str]):
        """Fit Fisher orientation per domain/set from Calibration observations."""
        domain_rows = get_domain_intervals(self.project)
        groups: dict[tuple[int | None, int], list[np.ndarray]] = {}
        dip_only_counts: dict[tuple[int | None, int], int] = {}
        for hole in self.project.borehole_collection.boreholes:
            if roles.get(hole.borehole_id) != "calibration":
                continue
            for observation in hole.fracture_observations:
                if observation.set_id is None:
                    continue
                domain_id = domain_at_depth(hole.borehole_id, observation.measured_depth, domain_rows)
                key = (domain_id, observation.set_id)
                if not observation.has_full_orientation:
                    dip_only_counts[key] = dip_only_counts.get(key, 0) + 1
                    continue
                groups.setdefault(key, []).append(
                    dip_dir_dip_to_normal(observation.dip_direction, observation.dip)
                )
        models: list[DomainOrientationModel] = []
        configs: dict[tuple[int | None, int], JointSetConfig] = {}
        base_sets = {item.set_id: item for item in self.project.joint_sets}
        for (domain_id, set_id), normals in sorted(groups.items(), key=str):
            if len(normals) < 3:
                continue
            array = np.asarray(normals)
            resultant = array.sum(axis=0)
            length = float(np.linalg.norm(resultant))
            if length <= 1e-12:
                continue
            mean_normal = resultant / length
            dip_direction, dip = normal_to_dip_dir_dip(mean_normal)
            count = len(array)
            if count == 1:
                kappa = base_sets.get(set_id, JointSetConfig(set_id=set_id)).orientation.kappa
            elif count >= 16:
                kappa = (count - 1) / max(count - length, 1e-10)
            else:
                kappa = (count - 2) / max(count - length, 1e-10) * count / max(count - 1, 1)
            kappa = max(0.1, min(float(kappa), 999.0))
            models.append(
                DomainOrientationModel(
                    domain_id=domain_id,
                    set_id=set_id,
                    mean_dip_direction=dip_direction,
                    mean_dip=dip,
                    kappa=kappa,
                    observation_count=count,
                    full_orientation_count=count,
                    dip_only_count=dip_only_counts.get((domain_id, set_id), 0),
                    orientation_fit_eligible=True,
                )
            )
            base = base_sets.get(set_id, JointSetConfig(set_id=set_id))
            configs[(domain_id, set_id)] = base.model_copy(
                update={"orientation": OrientationDistribution(mean_dip_direction=dip_direction, mean_dip=dip, kappa=kappa)}
            )
        return models, configs

    def set_assumed_sizes(
        self,
        distribution_type: str,
        parameters: dict[str, float],
        lower: float,
        upper: float,
        *,
        user_defined: bool = True,
    ) -> None:
        """Set manual size models for every available domain/set combination."""
        combinations = {(item.domain_id, item.set_id) for item in self.project.m9_state.p32_estimates}
        if not combinations:
            raise RuntimeError("Calculate P10/P32 before defining size models")
        self.project.m9_state.size_models = [
            assumed_size_model(
                domain_id=domain_id,
                set_id=set_id,
                distribution_type=distribution_type,
                parameters=parameters,
                lower=lower,
                upper=upper,
                user_defined=user_defined,
            )
            for domain_id, set_id in sorted(combinations, key=str)
        ]

    def fit_sizes(self, source_field: str) -> None:
        """Fit real calibration size measurements; never substitutes aperture or RQD."""
        allowed = {"radius", "diameter", "trace_length", "mapped_length"}
        if source_field not in allowed:
            raise ValueError(f"size fitting is only allowed for {sorted(allowed)}")
        roles = self._roles()
        domains = get_domain_intervals(self.project)
        samples: dict[tuple[int | None, int], list[float]] = {}
        for record in self.project.borehole_database.query("fractures", "formal"):
            if roles.get(record.hole_id) != "calibration" or source_field not in record.values:
                continue
            set_id = record.values.get("set_id")
            depth = record.values.get("depth", record.values.get("measured_depth"))
            if set_id is None or depth is None:
                continue
            domain_id = domain_at_depth(record.hole_id or "", float(depth), domains)
            value = float(record.values[source_field])
            if source_field == "diameter":
                value /= 2.0
            samples.setdefault((domain_id, int(set_id)), []).append(value)
        if not samples:
            raise ValueError("No real calibration size measurements are available; use an ASSUMED model")
        fitter = MLESizeModelFitter()
        self.project.m9_state.size_models = [
            fitter.fit(values, domain_id=domain_id, set_id=set_id, source_field=source_field)
            for (domain_id, set_id), values in sorted(samples.items(), key=str)
        ]

    def _domain_classifier(self):
        """Classify voxels by nearest calibration domain observation (explicit provenance)."""
        roles = self._roles()
        domain_rows = [
            item
            for item in get_domain_intervals(self.project)
            if roles.get(item.hole_id) == "calibration"
        ]
        centers: list[tuple[np.ndarray, int]] = []
        holes = {hole.borehole_id: hole for hole in self.project.borehole_collection.boreholes}
        for row in domain_rows:
            hole = holes.get(row.hole_id)
            if hole is None:
                continue
            points, mds = hole.compute_trajectory(step_length=0.5)
            md = (row.from_depth + row.to_depth) / 2.0
            point = np.asarray([np.interp(md, mds, points[:, axis]) for axis in range(3)])
            centers.append((point, row.domain_id))
        if not centers:
            return lambda point: None
        coordinates = np.asarray([item[0] for item in centers])
        return _NearestDomainClassifier(coordinates, tuple(item[1] for item in centers))

    def build_parameter_field(
        self, *, progress=None, cancelled=None, commit: bool = True
    ) -> tuple[ParameterFieldMetadata, dict[str, np.ndarray]]:
        """Build M9's first voxel field and optionally commit the complete candidate."""
        if self.project.spatial_grid_config is None:
            raise RuntimeError("A confirmed M8 voxel analysis domain is required")
        if not self.project.m9_state.p32_estimates:
            raise RuntimeError("Calculate the density model first")
        if not self.project.m9_state.size_models:
            raise RuntimeError("Set fracture-size models first")
        builder = ParameterFieldBuilder()
        metadata, arrays = builder.build(
            self.project.spatial_grid_config.analysis_domain,
            self.project.voxel_config,
            self.project.m9_state.density_settings,
            self.project.m9_state.p10_intervals,
            self.project.m9_state.p32_estimates,
            self.project.joint_sets,
            self.project.m9_state.size_models,
            random_seed=self.project.m9_state.random_seed,
            domain_at_point=self._domain_classifier(),
            inside_model=_ActiveRockClassifier(self.project.rock_mask, self.project.excavation_mask),
            progress=progress,
            cancelled=cancelled,
            orientation_models=self.project.m9_state.orientation_models,
        )
        metadata.provenance["domain_assignment"] = "nearest_calibration_domain_interval_center"
        if cancelled is not None and cancelled():
            raise InterruptedError("parameter field generation cancelled")
        if commit:
            self.project.m9_state.parameter_field_metadata = metadata
            self.project.m9_state.parameter_field_arrays = arrays
        return metadata, arrays

    def validate(self, *, progress=None, cancelled=None) -> ValidationSummary:
        """Evaluate held-out P10 only after all calibration fitting is complete."""
        state = self.project.m9_state
        if state.parameter_field_metadata is None:
            raise RuntimeError("Build the parameter field before validation")
        domain_orientations = {
            (item.domain_id, item.set_id): JointSetConfig(
                set_id=item.set_id,
                orientation=OrientationDistribution(
                    mean_dip_direction=item.mean_dip_direction,
                    mean_dip=item.mean_dip,
                    kappa=item.kappa,
                ),
            )
            for item in state.orientation_models
        }
        results: list[ValidationIntervalResult] = []
        validation_rows = [item for item in state.p10_intervals if item.role == "validation"]
        for row_index, row in enumerate(validation_rows):
            if cancelled and cancelled():
                raise InterruptedError("validation cancelled")
            predicted_p32 = self._field_value(row, row.set_id or 0)
            exposure = 0.0
            selected_set = domain_orientations.get((row.domain_id, row.set_id))
            if predicted_p32 is not None and selected_set is not None and row.sample_length > 0:
                exposure = sum(
                    length
                    * expected_orientation_exposure(
                        selected_set,
                        (dx, dy, dz),
                        random_seed=state.random_seed + row_index,
                        sample_count=state.density_settings.monte_carlo_samples,
                    )
                    for dx, dy, dz, length in row.segment_directions
                ) / row.sample_length
            predicted_p10 = predicted_p32 * exposure if predicted_p32 is not None else None
            error = abs(predicted_p10 - row.p10) if predicted_p10 is not None and row.p10 is not None else None
            relative = error / row.p10 if error is not None and row.p10 else None
            results.append(
                ValidationIntervalResult(
                    hole_id=row.hole_id,
                    from_depth=row.from_depth,
                    to_depth=row.to_depth,
                    domain_id=row.domain_id,
                    set_id=row.set_id or 0,
                    observed_count=row.observation_count,
                    observed_p10=row.p10,
                    predicted_p10=predicted_p10,
                    predicted_p32=predicted_p32,
                    absolute_error=error,
                    relative_error=relative,
                    data_state="no_data" if predicted_p10 is None else row.data_state,
                )
            )
            if progress:
                progress(row_index + 1, len(validation_rows))
        valid = [item for item in results if item.predicted_p10 is not None and item.observed_p10 is not None]
        no_data = len(results) - len(valid)
        if len(valid) < 2:
            summary = ValidationSummary(
                state=ValidationState.INSUFFICIENT_VALIDATION,
                valid_interval_count=len(valid),
                no_data_interval_count=no_data,
            )
        else:
            observed = np.asarray([item.observed_p10 for item in valid], dtype=float)
            predicted = np.asarray([item.predicted_p10 for item in valid], dtype=float)
            residual = predicted - observed
            correlation = float(np.corrcoef(observed, predicted)[0, 1]) if np.std(observed) and np.std(predicted) else None
            denominator = float(np.sum((observed - observed.mean()) ** 2))
            summary = ValidationSummary(
                state=ValidationState.COMPLETE,
                mae=float(np.mean(np.abs(residual))),
                rmse=float(np.sqrt(np.mean(residual**2))),
                bias=float(np.mean(residual)),
                r_squared=1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else None,
                correlation=correlation,
                valid_interval_count=len(valid),
                no_data_interval_count=no_data,
            )
        if cancelled and cancelled():
            raise InterruptedError("validation cancelled")
        state.validation_results = results
        state.validation_summary = summary
        return summary

    def _field_value(self, interval, set_id: int) -> float | None:
        """Read a validation prediction from the generated voxel field."""
        metadata = self.project.m9_state.parameter_field_metadata
        array = self.project.m9_state.parameter_field_arrays.get(f"set_{set_id}_p32")
        if metadata is None or array is None or interval.center_x is None:
            return None
        point = (interval.center_x, interval.center_y, interval.center_z)
        indices = tuple(
            int(math.floor((coordinate - origin) / spacing))
            for coordinate, origin, spacing in zip(point, metadata.origin, metadata.spacing)
        )
        if any(index < 0 or index >= metadata.shape[axis] for axis, index in enumerate(indices)):
            return None
        value = float(array[indices])
        return value if np.isfinite(value) else None

    def export(self, directory: Path) -> list[Path]:
        """Export all required M9 tabular, JSON, NPZ, and VTI artifacts."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = self.project.m9_state
        export_metadata = self._export_metadata()
        paths: list[Path] = []
        paths.append(self._write_models_csv(directory / "p10_intervals.csv", state.p10_intervals))
        paths.append(self._write_models_csv(directory / "p32_estimates.csv", state.p32_estimates))
        paths.append(
            self._write_json(
                directory / "density_model.json",
                {"metadata": export_metadata, "settings": state.density_settings.model_dump(mode="json"), "provenance": state.provenance},
            )
        )
        paths.append(
            self._write_json(
                directory / "fracture_size_models.json",
                {"metadata": export_metadata, "models": [item.model_dump(mode="json") for item in state.size_models]},
            )
        )
        paths.append(self._write_models_csv(directory / "validation_results.csv", state.validation_results))
        paths.append(
            self._write_json(
                directory / "validation_summary.json",
                {"metadata": export_metadata, "summary": state.validation_summary.model_dump(mode="json")},
            )
        )
        if state.parameter_field_arrays:
            npz_path = directory / "voxel_parameter_field.npz"
            np.savez_compressed(
                npz_path,
                **state.parameter_field_arrays,
                _metadata_json=np.asarray(json.dumps(export_metadata)),
            )
            paths.append(npz_path)
            paths.append(self._write_vti(directory / "voxel_parameter_field.vti"))
        return paths

    def _write_models_csv(self, path: Path, models: list[Any]) -> Path:
        metadata = self._export_metadata()
        rows = [
            {
                "project_version": metadata["project_version"],
                "coordinate_system": metadata["coordinate_system"],
                "length_unit": metadata["length_unit"],
                **item.model_dump(mode="json"),
            }
            for item in models
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return path

    def _export_metadata(self) -> dict[str, Any]:
        metadata = self.project.m9_state.parameter_field_metadata
        return {
            "project_version": self.project.metadata.software_version,
            "coordinate_system": "X=Easting,Y=Northing,Z=Elevation",
            "length_unit": "m",
            "p10_unit": "m^-1",
            "p32_unit": "m^-1",
            "random_seed": self.project.m9_state.random_seed,
            "grid_origin": metadata.origin if metadata else None,
            "grid_shape": metadata.shape if metadata else None,
            "grid_spacing": metadata.spacing if metadata else None,
        }

    @staticmethod
    def _write_json(path: Path, data: Any) -> Path:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_vti(self, path: Path) -> Path:
        import pyvista as pv

        metadata = self.project.m9_state.parameter_field_metadata
        assert metadata is not None
        grid = pv.ImageData(dimensions=tuple(item + 1 for item in metadata.shape), spacing=metadata.spacing, origin=metadata.origin)
        for name, values in self.project.m9_state.parameter_field_arrays.items():
            grid.cell_data[name] = values.ravel(order="F")
        grid.save(path)
        return path
