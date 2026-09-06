"""Import, interpolate, validate, and export continuous M9 scalar parameters."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dfn_cave_studio.dfn.intensity import domain_at_depth
from dfn_cave_studio.models.m9 import (
    DensityMethod,
    DensitySettings,
    NonNegativePolicy,
    ScalarFieldMetadata,
    ScalarFieldResult,
    ScalarParameterSample,
    ScalarValidationResult,
    ScalarValidationSummary,
)
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.services.m7_state import get_domain_intervals, get_holdout, get_workflow
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.voxel.ordinary_kriging import OrdinaryKrigingInterpolator, fit_variogram
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES, IDWInterpolator, SpatialSample
from dfn_cave_studio.voxel.resource_estimate import FieldResourceEstimate, estimate_scalar_field_resources


REQUIRED_COLUMNS = {"borehole_id", "from_depth", "to_depth", "parameter_name", "value", "unit"}
PARAMETER_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "p32": (0.0, None),
    "rqd": (0.0, 100.0),
    "rmr": (0.0, 100.0),
    "joint_spacing": (0.0, None),
    "joint_density": (0.0, None),
    "ucs": (0.0, None),
}
CATEGORICAL_PARAMETERS = {"rock_class", "rock_mass_class", "lithology", "weathering_class"}


def canonical_parameter_name(name: str) -> str:
    """Return a stable comparison key without discarding Unicode text."""
    return re.sub(r"[\s\-]+", "_", unicodedata.normalize("NFKC", name).strip().casefold())


def scalar_field_id(parameter_name: str) -> str:
    """Return a deterministic, file-safe ID with a Unicode-safe collision suffix."""
    canonical = canonical_parameter_name(parameter_name)
    ascii_text = unicodedata.normalize("NFKD", canonical).encode("ascii", "ignore").decode("ascii")
    slug = (re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_") or "parameter")[:48].rstrip("_")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


class ScalarParameterFieldService:
    """Project-level service for generic continuous physical parameter fields."""

    def __init__(self, project: Any) -> None:
        self.project = project

    def estimate_resources(
        self, *, chunk_size: int = 2048, budget_bytes: int | None = None,
        system_available_bytes: int | None = None,
    ) -> FieldResourceEstimate:
        """Estimate the scalar field allocation contract before building it."""
        if self.project.spatial_grid_config is None:
            raise RuntimeError("A confirmed voxel analysis domain is required")
        return estimate_scalar_field_resources(
            self.project.spatial_grid_config.analysis_domain,
            self.project.voxel_config,
            chunk_size=chunk_size,
            budget_bytes=budget_bytes,
            system_available_bytes=system_available_bytes,
        )

    def import_table(self, path: Path) -> list[ScalarParameterSample]:
        """Validate and append a CSV/XLSX long table as trajectory-located samples."""
        source = Path(path)
        frame = pd.read_excel(source) if source.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(source)
        return self.import_frame(frame, source_dataset=str(source))

    def import_frame(self, frame: pd.DataFrame, *, source_dataset: str = "") -> list[ScalarParameterSample]:
        """Validate a long-format dataframe atomically, without storing user data in logs."""
        repository = BoreholeRepository(self.project)
        normalized = {str(column).strip().lstrip("\ufeff").lower().replace("-", "_").replace(" ", "_"): column for column in frame}
        missing = sorted(REQUIRED_COLUMNS - normalized.keys())
        if missing:
            raise ValueError(f"Missing required scalar-parameter columns: {', '.join(missing)}")
        holes = {hole.borehole_id: hole for hole in self.project.borehole_collection}
        holdout = get_holdout(self.project)
        domains = get_domain_intervals(self.project)
        existing_units: dict[str, set[str]] = {}
        existing_keys: set[tuple[str, float, float, str]] = set()
        for item in self.project.m9_state.scalar_samples:
            canonical = canonical_parameter_name(item.parameter_name)
            existing_units.setdefault(canonical, set()).add(item.unit)
            existing_keys.add((item.borehole_id, item.from_depth, item.to_depth, canonical))
        imported: list[ScalarParameterSample] = []
        batch_units: dict[str, set[str]] = {}
        batch_keys: set[tuple[str, float, float, str]] = set()
        for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
            hole_id = str(row[normalized["borehole_id"]]).strip()
            hole = holes.get(hole_id)
            if hole is None:
                raise ValueError(f"Row {row_number}: unknown borehole_id {hole_id!r}")
            try:
                from_depth = float(row[normalized["from_depth"]])
                to_depth = float(row[normalized["to_depth"]])
                value = float(row[normalized["value"]])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {row_number}: depths and value must be numeric") from error
            if not all(math.isfinite(item) for item in (from_depth, to_depth, value)):
                raise ValueError(f"Row {row_number}: depths and value must be finite")
            if from_depth < 0 or to_depth <= from_depth or to_depth > hole.collar.final_depth + 1e-9:
                raise ValueError(f"Row {row_number}: interval is outside borehole depth or is reversed")
            raw_parameter, raw_unit = row[normalized["parameter_name"]], row[normalized["unit"]]
            if pd.isna(raw_parameter) or pd.isna(raw_unit):
                raise ValueError(f"Row {row_number}: parameter_name and unit are required")
            parameter = str(raw_parameter).strip()
            unit = str(raw_unit).strip()
            if not parameter or not unit:
                raise ValueError(f"Row {row_number}: parameter_name and unit are required")
            canonical = canonical_parameter_name(parameter)
            if canonical in CATEGORICAL_PARAMETERS:
                raise ValueError(f"Row {row_number}: categorical parameter {parameter!r} cannot use continuous interpolation")
            lower, upper = PARAMETER_BOUNDS.get(canonical, (None, None))
            if lower is not None and value < lower:
                raise ValueError(f"Row {row_number}: {parameter} value must be >= {lower:g}")
            if upper is not None and value > upper:
                raise ValueError(f"Row {row_number}: {parameter} value must be <= {upper:g}")
            batch_units.setdefault(canonical, set()).add(unit)
            if len(existing_units.get(canonical, set()) | batch_units[canonical]) != 1:
                raise ValueError(f"Row {row_number}: parameter {parameter!r} has conflicting units")
            key = (hole_id, from_depth, to_depth, canonical)
            if key in existing_keys or key in batch_keys:
                raise ValueError(f"Row {row_number}: duplicate scalar-parameter interval")
            batch_keys.add(key)
            midpoint = (from_depth + to_depth) / 2.0
            points, measured_depths = hole.compute_trajectory(step_length=0.5)
            xyz = [float(np.interp(midpoint, measured_depths, points[:, axis])) for axis in range(3)]
            source_column = normalized.get("source_dataset")
            quality_column = normalized.get("quality_flag")
            row_source = str(row[source_column]).strip() if source_column and pd.notna(row[source_column]) else source_dataset
            quality = str(row[quality_column]).strip() if quality_column and pd.notna(row[quality_column]) else ""
            digest = hashlib.sha256(
                json.dumps([row_source, row_number, hole_id, from_depth, to_depth, parameter, value, unit]).encode()
            ).hexdigest()[:24]
            imported.append(
                ScalarParameterSample(
                    sample_id=f"scalar-{digest}",
                    borehole_id=hole_id,
                    from_depth=from_depth,
                    to_depth=to_depth,
                    parameter_name=parameter,
                    value=value,
                    unit=unit,
                    source_dataset=row_source,
                    quality_flag=quality,
                    midpoint_x=xyz[0],
                    midpoint_y=xyz[1],
                    midpoint_z=xyz[2],
                    domain_id=domain_at_depth(hole_id, midpoint, domains),
                    role="validation" if holdout is not None and holdout.is_validation(hole_id) else "calibration",
                )
            )
        # Raw/audit rows remain in the canonical M8 database; M9 samples are its
        # trajectory-located Formal projection used for scientific computation.
        with repository.transaction():
            if not repository.database.records and len(self.project.borehole_collection):
                repository.migrate_m7()
            repository.import_dataframe(
                "scalar_parameters", frame, source_dataset or "scalar_parameter_import", modification_source="m9_scalar_import"
            )
        self.project.m9_state.scalar_samples.extend(imported)
        return imported

    def build(
        self,
        parameter_name: str,
        settings: DensitySettings,
        *,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        chunk_size: int = 2048,
    ) -> ScalarFieldResult:
        """Build one field transactionally; callers commit the returned result."""
        if self.project.spatial_grid_config is None:
            raise RuntimeError("A confirmed voxel analysis domain is required")
        canonical_name = canonical_parameter_name(parameter_name)
        selected = [
            item
            for item in self.project.m9_state.scalar_samples
            if canonical_parameter_name(item.parameter_name) == canonical_name
        ]
        if not selected:
            raise ValueError(f"No samples exist for parameter {parameter_name!r}")
        units = {item.unit for item in selected}
        if len(units) != 1:
            raise ValueError("A scalar parameter must have exactly one unit")
        calibration, validation = self._partition_by_current_holdout(selected)
        if not calibration:
            raise ValueError("INSUFFICIENT_DATA: no Calibration samples are available")
        groups: dict[int | None, list[ScalarParameterSample]] = {}
        for item in calibration:
            groups.setdefault(item.domain_id, []).append(item)
        predictors: dict[int | None, Any] = {}
        variograms = {}
        insufficient_data_by_domain: dict[str, str] = {}
        for domain_id, samples in groups.items():
            coordinates = np.asarray([(item.midpoint_x, item.midpoint_y, item.midpoint_z) for item in samples])
            values = np.asarray([item.value for item in samples])
            if settings.method == DensityMethod.ORDINARY_KRIGING:
                distinct_count = len(np.unique(coordinates, axis=0))
                if distinct_count < settings.kriging.minimum_neighbors:
                    insufficient_data_by_domain[str(domain_id)] = (
                        "INSUFFICIENT_DATA: fewer distinct Calibration locations than minimum_neighbors"
                    )
                    continue
                try:
                    diagnostics = fit_variogram(coordinates, values, settings.kriging)
                except ValueError as error:
                    if "INSUFFICIENT_DATA" not in str(error):
                        raise
                    insufficient_data_by_domain[str(domain_id)] = str(error)
                    continue
                predictors[domain_id] = OrdinaryKrigingInterpolator(coordinates, values, diagnostics, settings.kriging)
                variograms[str(domain_id)] = diagnostics
            elif settings.method == DensityMethod.IDW:
                predictors[domain_id] = IDWInterpolator(
                    [SpatialSample(*coordinate, float(value), domain_id) for coordinate, value in zip(coordinates, values)],
                    power=settings.power,
                    search_radius=settings.search_radius,
                    min_neighbors=settings.min_neighbors,
                    max_neighbors=settings.max_neighbors,
                    anisotropy=(settings.anisotropy_x, settings.anisotropy_y, settings.anisotropy_z),
                )
            else:
                predictors[domain_id] = float(np.mean(values))
        bounds = self.project.spatial_grid_config.analysis_domain
        voxel = self.project.voxel_config
        spacing = (voxel.cell_size_x, voxel.cell_size_y, voxel.cell_size_z)
        shape = voxel.compute_grid_dimensions(bounds)
        total = math.prod(shape)
        arrays = {
            "estimate": np.full(shape, np.nan, dtype=np.float32),
            "kriging_variance": np.full(shape, np.nan, dtype=np.float32),
            "cell_state": np.full(shape, CELL_STATE_CODES[VoxelCellState.NO_DATA], dtype=np.uint8),
            "neighbor_count": np.zeros(shape, dtype=np.int16),
            "domain_id": np.full(shape, -1, dtype=np.int32),
        }
        domain_centers = np.asarray([(item.midpoint_x, item.midpoint_y, item.midpoint_z) for item in calibration])
        domain_labels = np.asarray(
            [-1 if item.domain_id is None else item.domain_id for item in calibration], dtype=np.int64
        )
        domain_tree = cKDTree(domain_centers)
        rejected_count = 0
        clipped_count = 0
        pre_adjustment_minimum: float | None = None
        pre_adjustment_maximum: float | None = None
        clipped_change = 0.0
        lower_bound, upper_bound = PARAMETER_BOUNDS.get(canonical_name, (None, None))
        for start in range(0, total, chunk_size):
            if cancelled and cancelled():
                raise InterruptedError("scalar parameter field generation cancelled")
            end = min(total, start + chunk_size)
            flat_indices = np.arange(start, end)
            grid_indices = np.column_stack(np.unravel_index(flat_indices, shape))
            points = np.column_stack([
                lower + (grid_indices[:, axis] + 0.5) * spacing[axis]
                for axis, lower in enumerate((bounds.x_min, bounds.y_min, bounds.z_min))
            ])
            inside = self.project.rock_mask.contains_points(points) & ~self.project.excavation_mask.contains_points(points)
            state_flat = arrays["cell_state"].ravel()
            state_flat[flat_indices[~inside]] = CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL]
            if np.any(inside):
                inside_rows = np.flatnonzero(inside)
                _, nearest = domain_tree.query(points[inside_rows], k=1)
                labels = domain_labels[np.atleast_1d(nearest).astype(np.int64)]
                for encoded_domain in np.unique(labels):
                    if cancelled and cancelled():
                        raise InterruptedError("scalar parameter field generation cancelled")
                    domain_id = None if encoded_domain == -1 else int(encoded_domain)
                    local_rows = inside_rows[labels == encoded_domain]
                    output_flats = flat_indices[local_rows]
                    arrays["domain_id"].ravel()[output_flats] = -1 if domain_id is None else domain_id
                    predictor = predictors.get(domain_id)
                    if predictor is None:
                        continue
                    target_points = points[local_rows]
                    values = np.full(len(local_rows), np.nan)
                    variances = np.full(len(local_rows), np.nan)
                    neighbours = np.zeros(len(local_rows), dtype=np.int16)
                    if settings.method == DensityMethod.ORDINARY_KRIGING:
                        values, variances, neighbours, _ = predictor.predict_many(target_points)
                    elif settings.method == DensityMethod.IDW:
                        prediction = predictor.predict_many(target_points, domain_id)
                        values = prediction.values
                        neighbours = prediction.neighbor_counts
                    else:
                        values.fill(float(predictor))
                    finite = np.isfinite(values)
                    if np.any(finite):
                        chunk_minimum = float(np.min(values[finite]))
                        chunk_maximum = float(np.max(values[finite]))
                        pre_adjustment_minimum = (
                            chunk_minimum
                            if pre_adjustment_minimum is None
                            else min(pre_adjustment_minimum, chunk_minimum)
                        )
                        pre_adjustment_maximum = (
                            chunk_maximum
                            if pre_adjustment_maximum is None
                            else max(pre_adjustment_maximum, chunk_maximum)
                        )
                    outside = np.zeros(len(values), dtype=bool)
                    if lower_bound is not None:
                        outside |= finite & (values < lower_bound)
                    if upper_bound is not None:
                        outside |= finite & (values > upper_bound)
                    if np.any(outside):
                        if settings.kriging.non_negative_policy == NonNegativePolicy.REJECT:
                            rejected_count += int(np.count_nonzero(outside))
                            finite[outside] = False
                        else:
                            original = values[outside].copy()
                            values[outside] = np.clip(values[outside], lower_bound, upper_bound)
                            clipped_count += int(np.count_nonzero(outside))
                            clipped_change += float(np.sum(np.abs(values[outside] - original)))
                    selected_flats = output_flats[finite]
                    arrays["estimate"].ravel()[selected_flats] = values[finite]
                    arrays["kriging_variance"].ravel()[selected_flats] = variances[finite]
                    arrays["neighbor_count"].ravel()[selected_flats] = neighbours[finite]
                    state_flat[selected_flats] = np.where(
                        values[finite] == 0, CELL_STATE_CODES[VoxelCellState.TRUE_ZERO],
                        CELL_STATE_CODES[VoxelCellState.MODELED_VALUE],
                    )
            if progress:
                progress(end, total)
        config = settings.model_dump(mode="json")
        config_hash = hashlib.sha256(
            json.dumps({"parameter": canonical_name, "unit": next(iter(units)), "settings": config,
                        "samples": [item.sample_id for item in calibration]}, sort_keys=True).encode()
        ).hexdigest()
        field_id = scalar_field_id(parameter_name)
        metadata = ScalarFieldMetadata(
            field_id=field_id,
            parameter_name=selected[0].parameter_name,
            unit=next(iter(units)),
            method=settings.method,
            shape=shape,
            origin=(bounds.x_min, bounds.y_min, bounds.z_min),
            spacing=spacing,
            array_names=sorted(arrays),
            config_hash=config_hash,
            variograms=variograms,
            rejected_voxel_count=rejected_count,
            clipped_voxel_count=clipped_count,
            pre_clip_minimum=pre_adjustment_minimum,
            pre_adjustment_minimum=pre_adjustment_minimum,
            pre_adjustment_maximum=pre_adjustment_maximum,
            clipped_total_change=clipped_change,
            parameter_bounds=(lower_bound, upper_bound),
            adjustment_policy=settings.kriging.non_negative_policy,
            provenance={
                "calibration_sample_ids": [item.sample_id for item in calibration],
                "validation_sample_ids_excluded_from_fit": [item.sample_id for item in validation],
                "import_role_is_audit_snapshot": True,
                "holdout_locked_at_build": True,
                "domain_isolation": True,
                "domain_assignment": "nearest_calibration_domain_interval_midpoint",
                "isotropic_3d": True,
                "insufficient_data_by_domain": insufficient_data_by_domain,
                "parameter_bounds": {"minimum": lower_bound, "maximum": upper_bound},
                "bound_policy": settings.kriging.non_negative_policy.value,
                "rmr_system_version_responsibility": (
                    "The importer preserves the supplied RMR parameter name, unit, source dataset, and quality flag; "
                    "the user is responsible for recording the applicable RMR system/version in project provenance."
                    if canonical_name == "rmr"
                    else None
                ),
            },
        )
        result = ScalarFieldResult(metadata=metadata, settings=settings, arrays=arrays)
        result.validation_results, result.validation_summary = self._validate(validation, predictors, settings)
        return result

    def commit(self, result: ScalarFieldResult) -> None:
        """Replace one field and invalidate DFN outputs only when P32 changed."""
        previous = next(
            (item for item in self.project.m9_state.scalar_fields if item.metadata.field_id == result.metadata.field_id),
            None,
        )
        fields = [item for item in self.project.m9_state.scalar_fields if item.metadata.field_id != result.metadata.field_id]
        fields.append(result)
        self.project.m9_state.scalar_fields = fields
        changed = previous is None or previous.metadata.config_hash != result.metadata.config_hash
        if changed and canonical_parameter_name(result.metadata.parameter_name) == "p32":
            workflow = get_workflow(self.project)
            if workflow is not None:
                workflow.invalidate_steps(["explicit_dfn", "second_voxelization"])

    def leave_one_borehole_out(
        self, parameter_name: str, settings: DensitySettings
    ) -> tuple[list[ScalarValidationResult], ScalarValidationSummary]:
        """Cross-validate by whole borehole; intervals from one hole never leak into its fit."""
        selected = [
            item
            for item in self.project.m9_state.scalar_samples
            if canonical_parameter_name(item.parameter_name) == canonical_parameter_name(parameter_name)
        ]
        samples, _ = self._partition_by_current_holdout(selected)
        results: list[ScalarValidationResult] = []
        for hole_id in sorted({item.borehole_id for item in samples}):
            training = [item for item in samples if item.borehole_id != hole_id]
            testing = [item for item in samples if item.borehole_id == hole_id]
            grouped: dict[int | None, list[ScalarParameterSample]] = {}
            for item in training:
                grouped.setdefault(item.domain_id, []).append(item)
            predictors: dict[int | None, Any] = {}
            for domain_id, rows in grouped.items():
                xyz = np.asarray([(item.midpoint_x, item.midpoint_y, item.midpoint_z) for item in rows])
                values = np.asarray([item.value for item in rows])
                if settings.method == DensityMethod.ORDINARY_KRIGING:
                    try:
                        diagnostics = fit_variogram(xyz, values, settings.kriging)
                        predictors[domain_id] = OrdinaryKrigingInterpolator(xyz, values, diagnostics, settings.kriging)
                    except ValueError:
                        continue
                elif settings.method == DensityMethod.IDW:
                    predictors[domain_id] = IDWInterpolator(
                        [SpatialSample(*point, float(value), domain_id) for point, value in zip(xyz, values)],
                        power=settings.power, search_radius=settings.search_radius,
                        min_neighbors=settings.min_neighbors, max_neighbors=settings.max_neighbors,
                        anisotropy=(settings.anisotropy_x, settings.anisotropy_y, settings.anisotropy_z),
                    )
                else:
                    predictors[domain_id] = float(np.mean(values))
            fold, _ = self._validate(testing, predictors, settings)
            results.extend(fold)
        valid = [item for item in results if item.predicted is not None and item.residual is not None]
        if not valid:
            return results, ScalarValidationSummary(sample_count=len(results))
        residuals = np.asarray([item.residual for item in valid], dtype=float)
        observed = np.asarray([item.observed for item in valid], dtype=float)
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        return results, ScalarValidationSummary(
            sample_count=len(results), predicted_count=len(valid), mean_error=float(residuals.mean()),
            mae=float(np.mean(np.abs(residuals))), rmse=float(np.sqrt(np.mean(residuals**2))),
            r_squared=1.0 - float(np.sum(residuals**2)) / denominator if len(valid) >= 3 and denominator > 0 else None,
        )

    def export(self, result: ScalarFieldResult, directory: Path) -> list[Path]:
        """Export metadata, validation, arrays, and VTI for one scalar field."""
        directory.mkdir(parents=True, exist_ok=True)
        stem = result.metadata.field_id
        metadata_path = directory / f"{stem}.json"
        metadata_path.write_text(
            json.dumps({"metadata": result.metadata.model_dump(mode="json"), "settings": result.settings.model_dump(mode="json"),
                        "validation": result.validation_summary.model_dump(mode="json")}, indent=2), encoding="utf-8"
        )
        npz_path = directory / f"{stem}.npz"
        np.savez_compressed(npz_path, **result.arrays)
        csv_path = directory / f"{stem}_validation.csv"
        rows = [item.model_dump(mode="json") for item in result.validation_results]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        import pyvista as pv
        grid = pv.ImageData(dimensions=tuple(item + 1 for item in result.metadata.shape), spacing=result.metadata.spacing,
                            origin=result.metadata.origin)
        for name, values in result.arrays.items():
            grid.cell_data[name] = values.ravel(order="F")
        vti_path = directory / f"{stem}.vti"; grid.save(vti_path)
        return [metadata_path, csv_path, npz_path, vti_path]

    def _partition_by_current_holdout(
        self, samples: list[ScalarParameterSample]
    ) -> tuple[list[ScalarParameterSample], list[ScalarParameterSample]]:
        """Partition using the current locked holdout; stored sample roles are audit-only."""
        holdout = get_holdout(self.project)
        if holdout is None or not holdout.is_locked:
            raise RuntimeError("A locked Validation Holdout is required before building or validating a scalar field")
        calibration_ids = set(holdout.calibration_holes)
        validation_ids = set(holdout.validation_holes)
        unknown = sorted({item.borehole_id for item in samples} - calibration_ids - validation_ids)
        if unknown:
            raise RuntimeError(f"Current Validation Holdout does not classify borehole(s): {', '.join(unknown)}")
        return (
            [item for item in samples if item.borehole_id in calibration_ids],
            [item for item in samples if item.borehole_id in validation_ids],
        )

    @staticmethod
    def _validate(samples: list[ScalarParameterSample], predictors: dict[int | None, Any], settings: DensitySettings):
        results: list[ScalarValidationResult] = []
        for sample in samples:
            predictor = predictors.get(sample.domain_id)
            predicted = variance = None
            status = "insufficient_data"
            if predictor is not None:
                point = (sample.midpoint_x, sample.midpoint_y, sample.midpoint_z)
                if settings.method == DensityMethod.ORDINARY_KRIGING:
                    item = predictor.predict(point); predicted, variance, status = item.estimate, item.variance, item.status.lower()
                elif settings.method == DensityMethod.IDW:
                    item = predictor.predict(point, sample.domain_id); predicted = item.value; status = item.provenance
                else:
                    predicted, status = float(predictor), "predicted"
            residual = None if predicted is None else predicted - sample.value
            standard = None if residual is None or variance is None or variance <= 0 else residual / math.sqrt(variance)
            delta = None if variance is None else 1.96 * math.sqrt(variance)
            results.append(ScalarValidationResult(sample_id=sample.sample_id, borehole_id=sample.borehole_id,
                from_depth=sample.from_depth, to_depth=sample.to_depth, domain_id=sample.domain_id,
                observed=sample.value, predicted=predicted, residual=residual, kriging_variance=variance,
                standardized_residual=standard, prediction_interval_low=None if delta is None or predicted is None else predicted-delta,
                prediction_interval_high=None if delta is None or predicted is None else predicted+delta, status=status))
        valid = [item for item in results if item.predicted is not None and item.residual is not None]
        if not valid:
            return results, ScalarValidationSummary(sample_count=len(results))
        residuals = np.asarray([item.residual for item in valid], dtype=float)
        observed = np.asarray([item.observed for item in valid], dtype=float)
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        summary = ScalarValidationSummary(sample_count=len(results), predicted_count=len(valid), mean_error=float(residuals.mean()),
            mae=float(np.mean(np.abs(residuals))), rmse=float(np.sqrt(np.mean(residuals**2))),
            r_squared=1.0-float(np.sum(residuals**2))/denominator if len(valid) >= 3 and denominator > 0 else None)
        return results, summary
