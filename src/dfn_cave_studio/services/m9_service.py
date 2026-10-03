"""Project-level orchestration for the M9 parameter-field workflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
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
from dfn_cave_studio.models.borehole_database import BoreholeDataType, FractureObservationMode
from dfn_cave_studio.models.m9 import (
    DensityMethod,
    DensitySettings,
    DomainOrientationModel,
    M9DensityInputMode,
    M9State,
    ParameterFieldMetadata,
    ValidationIntervalResult,
    ValidationState,
    ValidationSummary,
)
from dfn_cave_studio.services.m7_state import get_domain_intervals, get_holdout
from dfn_cave_studio.services.observation_service import ObservationService
from dfn_cave_studio.voxel.ordinary_kriging import OrdinaryKrigingInterpolator, fit_variogram
from dfn_cave_studio.voxel.parameter_field import IDWInterpolator, ParameterFieldBuilder, SpatialSample


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

    def calculate_density(
        self,
        settings: DensitySettings | None = None,
        *,
        input_mode: M9DensityInputMode | str | None = None,
        realization_id: str | None = None,
        random_seed: int | None = None,
        progress=None,
        cancelled=None,
        commit: bool = True,
    ) -> M9State:
        """Compute P10/P32 from exactly one selected source without validation leakage."""
        mode = M9DensityInputMode(input_mode or self.project.m9_state.density_input_mode)
        settings = settings or self.project.m9_state.density_settings
        seed = self.project.m9_state.random_seed if random_seed is None else int(random_seed)
        compatible_count = sum(
            len(hole.fracture_observations) for hole in self.project.borehole_collection.boreholes
        )
        database = getattr(self.project, "borehole_database", None)
        counts = database.observation_mode_counts() if database is not None else {}
        spacing = int(counts.get("interval_spacing", 0))
        relative = int(counts.get("axis_plane_angle", 0))
        if mode == M9DensityInputMode.FORMAL_OBSERVATIONS and compatible_count == 0:
            raise ValueError(
                "No global fracture orientations are available for the existing P32 workflow. "
                f"Imported interval-spacing records={spacing}; borehole-relative angle records={relative}. "
                "Their import and derived metrics are retained, but direction completion is not implemented in this phase."
            )
        roles = self._roles()
        if mode == M9DensityInputMode.FORMAL_OBSERVATIONS:
            intervals = build_p10_intervals(
                self.project.borehole_collection,
                roles,
                get_domain_intervals(self.project),
                interval_length=settings.interval_length,
                interval_mode=settings.interval_mode,
                set_ids=[item.set_id for item in self.project.joint_sets],
            )
            orientation_models, domain_sets = self._fit_domain_orientations(roles)
            input_provenance = {
                "source_mode": mode.value,
                "realization_id": None,
                "validation_semantics": "independent whole-borehole holdout",
            }
            selected_realization_id = None
        else:
            if not realization_id:
                raise ValueError("Select one saved Phase 2A borehole-fracture realization for M9")
            from dfn_cave_studio.services.m9_phase2a_adapter import Phase2AM9Adapter

            adapted = Phase2AM9Adapter(self.project).build(
                realization_id,
                roles,
                interval_mode=settings.interval_mode,
                interval_length=settings.interval_length,
                cancelled=cancelled,
            )
            intervals = adapted.intervals
            orientation_models = adapted.orientation_models
            domain_sets = adapted.domain_sets
            input_provenance = adapted.provenance
            selected_realization_id = realization_id
        eligible_intervals = [item for item in intervals if item.p10 is not None]
        estimates = estimate_p32(
            eligible_intervals,
            self.project.joint_sets,
            random_seed=seed,
            sample_count=settings.monte_carlo_samples,
            low_observability_threshold=settings.low_observability_threshold,
            joint_sets_by_domain=domain_sets,
            progress=progress,
            cancelled=cancelled,
        )
        if cancelled and cancelled():
            raise InterruptedError("density estimation cancelled")
        state = self.project.m9_state.model_copy(deep=False)
        state.density_settings = settings
        state.density_input_mode = mode
        state.density_input_realization_id = selected_realization_id
        state.random_seed = seed
        state.p10_intervals = intervals
        state.p32_estimates = estimates
        state.orientation_models = orientation_models
        state.parameter_field_metadata = None
        state.parameter_field_arrays = {}
        state.validation_results = []
        state.validation_summary = ValidationSummary()
        state.provenance = dict(self.project.m9_state.provenance)
        state.provenance["density_input"] = input_provenance
        state.provenance["density_fit_holes"] = sorted(
            hole_id for hole_id, role in roles.items() if role == "calibration"
        )
        state.provenance["validation_holes_excluded"] = sorted(
            hole_id for hole_id, role in roles.items() if role == "validation"
        )
        state.provenance["non_global_orientation_records_excluded_from_density_fit"] = {
            "interval_spacing": spacing,
            "axis_plane_angle": relative,
            "reason": (
                "direction completion is not implemented in this phase"
                if mode == M9DensityInputMode.FORMAL_OBSERVATIONS
                else "Phase 2A source selected exclusively; raw non-global observations are not mixed"
            ),
        }
        if commit:
            self.project.m9_state = state
        return state

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
    ) -> bool:
        """Set manual size models for every available domain/set combination."""
        combinations = {(item.domain_id, item.set_id) for item in self.project.m9_state.p32_estimates}
        if not combinations:
            raise RuntimeError("Calculate P10/P32 before defining size models")
        if distribution_type == "fixed":
            radius = float(parameters["radius"])
            lower = radius
            upper = radius
        models = [
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
        if models == self.project.m9_state.size_models:
            return False
        self.project.m9_state.size_models = models
        return True

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
        self._validate_selected_density_input()
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
        metadata.provenance["density_input"] = dict(
            self.project.m9_state.provenance.get("density_input", {})
        )
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
        self._validate_selected_density_input()
        if state.density_input_mode == M9DensityInputMode.PHASE2A_REALIZATION:
            return self._validate_phase2a_holdout_p10(progress=progress, cancelled=cancelled)
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
                    calibration_or_validation=(
                        "internal_consistency_not_independent"
                        if state.density_input_mode == M9DensityInputMode.PHASE2A_REALIZATION
                        else "validation"
                    ),
                    data_state=(
                        "internal_consistency_only"
                        if state.density_input_mode == M9DensityInputMode.PHASE2A_REALIZATION
                        and predicted_p10 is not None
                        else ("no_data" if predicted_p10 is None else row.data_state)
                    ),
                )
            )
            if progress:
                progress(row_index + 1, len(validation_rows))
        valid = [item for item in results if item.predicted_p10 is not None and item.observed_p10 is not None]
        no_data = len(results) - len(valid)
        if state.density_input_mode == M9DensityInputMode.PHASE2A_REALIZATION:
            summary = ValidationSummary(
                state=ValidationState.NOT_VALIDATED,
                valid_interval_count=len(valid),
                no_data_interval_count=no_data,
            )
            state.provenance["validation_interpretation"] = {
                "classification": "internal_consistency_not_independent",
                "reason": (
                    "Each held-out borehole realization was generated from that borehole's own spacing interval; "
                    "it is excluded from fitting but is not independent validation ground truth."
                ),
                "raw_spacing_may_validate": "P10/count only under a separately demonstrated independence design",
                "p32_ground_truth": False,
            }
        elif len(valid) < 2:
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

    def _validate_phase2a_holdout_p10(self, *, progress=None, cancelled=None) -> ValidationSummary:
        """Compare calibration-only field predictions with held-out along-hole spacing observations."""
        state = self.project.m9_state
        holdout = get_holdout(self.project)
        if holdout is None or not holdout.is_locked:
            raise RuntimeError("A locked whole-borehole Validation Holdout is required")
        validation_holes = set(holdout.validation_holes)
        calibration_holes = set(holdout.calibration_holes)
        validation_keys = {self._hole_key(item) for item in validation_holes}
        realization = next(
            item
            for item in self.project.borehole_fracture_state.realizations
            if item.realization_id == state.density_input_realization_id
        )
        fit = self.project.borehole_fracture_state.global_fit
        isolation = {} if fit is None else dict(fit.provenance.get("validation_isolation", {}))
        uncertain_z = list(isolation.get("unassociated_z_observations", []))
        mapped_ids = {item.observation_id for item in fit.mappings} if fit is not None else set()
        leaked_uncertain_z = sorted(set(uncertain_z) & mapped_ids)
        leaked_intervals = sorted(
            {
                item.hole_id
                for item in realization.interval_diagnostics
                if self._hole_key(item.hole_id) in validation_keys
            }
        )
        blocked_reason = None
        if leaked_uncertain_z:
            blocked_reason = (
                "stale_input_hash: an older global fit used BOREHOLE_CAMERA representatives without an explicit "
                "borehole_id; recompute the mapping and realization before independent validation"
            )
        elif leaked_intervals:
            blocked_reason = (
                "The selected realization contains intervals generated from current Validation-hole spacing; "
                "recompute it with the locked holdout before independent validation"
            )

        all_spacing = ObservationService(self.project).spacing_observations()
        spacing_rows = [item for item in all_spacing if self._hole_key(item.hole_id) in validation_keys]
        predictors, calibration_input_hash, calibration_exclusions, predictor_failures = self._calibration_spacing_p10_predictors(
            calibration_holes
        )
        results: list[ValidationIntervalResult] = []
        holes = {self._hole_key(item.borehole_id): item for item in self.project.borehole_collection.boreholes}
        reason_counts: dict[str, int] = {}
        validation_segment_count = 0
        for row_index, observation in enumerate(spacing_rows):
            if cancelled and cancelled():
                raise InterruptedError("validation cancelled")
            exclusion_reason = blocked_reason
            predicted_p10 = None
            segments, segment_reason = self._spacing_domain_segments(observation)
            validation_segment_count += len(segments)
            if observation.measurement_basis != "BOREHOLE_ALONG_HOLE":
                exclusion_reason = "unknown_or_invalid_basis"
            elif self._hole_key(observation.hole_id) not in holes:
                exclusion_reason = "invalid_trajectory"
            elif segment_reason is not None:
                exclusion_reason = segment_reason
            elif exclusion_reason is None:
                predicted_p10, exclusion_reason = self._integrated_total_p10_prediction(
                    holes[self._hole_key(observation.hole_id)], segments, predictors, predictor_failures
                )
            if predicted_p10 is None:
                category = exclusion_reason or "other_no_data"
                reason_counts[category] = reason_counts.get(category, 0) + 1
            observed_p10 = observation.derived_p10
            signed_error = predicted_p10 - observed_p10 if predicted_p10 is not None else None
            results.append(
                ValidationIntervalResult(
                    hole_id=observation.hole_id,
                    from_depth=observation.from_depth,
                    to_depth=observation.to_depth,
                    domain_id=None,
                    set_id=0,
                    observed_count=None,
                    observed_p10=observed_p10,
                    predicted_p10=predicted_p10,
                    predicted_p32=None,
                    absolute_error=abs(signed_error) if signed_error is not None else None,
                    signed_error=signed_error,
                    relative_error=(abs(signed_error) / observed_p10 if signed_error is not None else None),
                    calibration_or_validation="independent_holdout_along_hole_p10",
                    data_state="modeled_value" if predicted_p10 is not None else "no_data",
                    observation_source="validation_interval_spacing_reciprocal",
                    prediction_source="calibration_only_along_hole_p10_spatial_predictor_trajectory_integral",
                    exclusion_reason=exclusion_reason,
                )
            )
            if progress:
                progress(row_index + 1, len(spacing_rows))

        valid = [item for item in results if item.predicted_p10 is not None]
        no_data = len(results) - len(valid)
        if valid:
            residual = np.asarray([item.signed_error for item in valid], dtype=np.float64)
            summary = ValidationSummary(
                state=ValidationState.COMPLETE,
                mae=float(np.mean(np.abs(residual))),
                rmse=float(np.sqrt(np.mean(residual**2))),
                bias=float(np.mean(residual)),
                valid_interval_count=len(valid),
                no_data_interval_count=no_data,
            )
        else:
            summary = ValidationSummary(
                state=ValidationState.INSUFFICIENT_VALIDATION,
                valid_interval_count=0,
                no_data_interval_count=no_data,
            )
        config_payload = {
            "density_settings": state.density_settings.model_dump(mode="json"),
            "density_input": state.provenance.get("density_input", {}),
            "parameter_field": (
                None
                if state.parameter_field_metadata is None
                else {
                    "shape": state.parameter_field_metadata.shape,
                    "origin": state.parameter_field_metadata.origin,
                    "spacing": state.parameter_field_metadata.spacing,
                    "fields": state.parameter_field_metadata.field_names,
                }
            ),
        }
        if cancelled and cancelled():
            raise InterruptedError("validation cancelled")
        pipeline = self._phase2a_validation_pipeline_diagnostics(
            all_spacing,
            calibration_holes,
            validation_holes,
            spacing_rows,
            validation_segment_count,
            selected_valid=len(valid),
            selected_no_data=no_data,
            selected_reasons=reason_counts,
            blocked_reason=blocked_reason,
        )
        state.provenance["validation_interpretation"] = {
            "classification": "independent_holdout_along_hole_frequency",
            "scope": "P10 along-hole frequency only; not P32, set orientation, block size, or complete DFN validation",
            "locked_calibration_holes": sorted(calibration_holes),
            "locked_validation_holes": sorted(validation_holes),
            "model_config_id": hashlib.sha256(
                json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "realization_id": realization.realization_id,
            "realization_input_hash": realization.input_hash,
            "calibration_spacing_input_hash": calibration_input_hash,
            "calibration_spacing_exclusions": calibration_exclusions,
            "excluded_validation_z_observations": isolation.get("excluded_validation_z_observations", []),
            "unassociated_z_observations": uncertain_z,
            "excluded_unassociated_z_observations": isolation.get("excluded_unassociated_z_observations", []),
            "blocked_reason": blocked_reason,
            "pipeline_diagnostics": pipeline,
            "observed_count_semantics": "not reported; L/S is an expectation and is not stored as an exact count",
        }
        state.validation_results = results
        state.validation_summary = summary
        return summary

    def _calibration_spacing_p10_predictors(
        self, calibration_holes: set[str], settings: DensitySettings | None = None
    ) -> tuple[dict[int | None, Any], str, list[dict[str, str]], dict[int | None, str]]:
        """Fit along-hole P10 predictors exclusively from current Calibration spacing."""
        state = self.project.m9_state
        observations = ObservationService(self.project).spacing_observations()
        holes = {self._hole_key(item.borehole_id): item for item in self.project.borehole_collection.boreholes}
        calibration_keys = {self._hole_key(item) for item in calibration_holes}
        grouped: dict[int | None, list[SpatialSample]] = {}
        audit_rows: list[dict[str, Any]] = []
        exclusions: list[dict[str, str]] = []
        failures: dict[int | None, str] = {}
        for observation in observations:
            if self._hole_key(observation.hole_id) not in calibration_keys:
                continue
            if observation.measurement_basis != "BOREHOLE_ALONG_HOLE":
                exclusions.append(
                    {"observation_id": observation.observation_id, "reason": "not_borehole_along_hole"}
                )
                continue
            hole = holes.get(self._hole_key(observation.hole_id))
            if hole is None:
                exclusions.append({"observation_id": observation.observation_id, "reason": "missing_trajectory"})
                continue
            segments, reason = self._spacing_domain_segments(observation)
            if reason is not None:
                exclusions.append({"observation_id": observation.observation_id, "reason": reason})
                continue
            points, measured_depths = hole.compute_trajectory(step_length=0.5)
            for segment in segments:
                midpoint = (segment.from_depth + segment.to_depth) / 2.0
                xyz = tuple(float(np.interp(midpoint, measured_depths, points[:, axis])) for axis in range(3))
                grouped.setdefault(segment.domain_id, []).append(
                    SpatialSample(*xyz, observation.derived_p10, segment.domain_id)
                )
                audit_rows.append(
                    {
                        "observation_id": observation.observation_id,
                        "hole_id": observation.hole_id,
                        "from_depth": segment.from_depth,
                        "to_depth": segment.to_depth,
                        "p10": observation.derived_p10,
                        "domain_id": segment.domain_id,
                        "xyz": xyz,
                    }
                )

        predictors: dict[int | None, Any] = {}
        settings = settings or state.density_settings
        for domain_id, samples in grouped.items():
            coordinates = np.asarray([(item.x, item.y, item.z) for item in samples], dtype=np.float64)
            values = np.asarray([item.value for item in samples], dtype=np.float64)
            if settings.method.value == "global_constant":
                predictors[domain_id] = float(np.mean(values))
            elif settings.method.value == "idw":
                fallback = {domain_id: float(np.mean(values))} if settings.global_fallback else None
                predictors[domain_id] = IDWInterpolator(
                    samples,
                    power=settings.power,
                    search_radius=settings.search_radius,
                    min_neighbors=settings.min_neighbors,
                    max_neighbors=settings.max_neighbors,
                    anisotropy=(settings.anisotropy_x, settings.anisotropy_y, settings.anisotropy_z),
                    fallback_by_domain=fallback,
                )
            else:
                if len(np.unique(coordinates, axis=0)) < settings.kriging.minimum_neighbors:
                    failures[domain_id] = "insufficient_neighbors"
                    exclusions.append(
                        {"observation_id": f"domain:{domain_id}", "reason": "insufficient_kriging_locations"}
                    )
                    continue
                try:
                    diagnostics = fit_variogram(coordinates, values, settings.kriging)
                except ValueError as error:
                    failures[domain_id] = "variogram_fit_failed"
                    exclusions.append({"observation_id": f"domain:{domain_id}", "reason": str(error)})
                    continue
                predictors[domain_id] = OrdinaryKrigingInterpolator(
                    coordinates, values, diagnostics, settings.kriging
                )
        input_hash = hashlib.sha256(
            json.dumps(audit_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return predictors, input_hash, exclusions, failures

    def _spacing_domain_segments(self, observation) -> tuple[list[Any], str | None]:
        """Return authoritative assigned subsegments; a cross-domain source row remains one observation."""
        rows_for_hole = [
            item
            for item in get_domain_intervals(self.project)
            if self._hole_key(item.hole_id) == self._hole_key(observation.hole_id)
        ]
        segments = list(observation.domain_segments)
        expected = observation.to_depth - observation.from_depth
        covered = sum(item.to_depth - item.from_depth for item in segments)
        tolerance = max(1e-9, expected * 1e-9)
        if abs(covered - expected) > tolerance:
            return segments, "unassigned_domain"
        if rows_for_hole and any(item.domain_id is None for item in segments):
            return segments, "unassigned_domain"
        return segments, None

    def _integrated_total_p10_prediction(
        self,
        hole,
        domain_segments: list[Any],
        predictors: dict[int | None, Any],
        predictor_failures: dict[int | None, str] | None = None,
    ) -> tuple[float | None, str | None]:
        """Integrate a Calibration-only along-hole P10 predictor on the real survey trajectory."""
        state = self.project.m9_state
        metadata = state.parameter_field_metadata
        if metadata is None or not domain_segments:
            return None, "invalid_trajectory"
        predictor_failures = predictor_failures or {}
        if not predictors:
            reason = next(
                (predictor_failures[item.domain_id] for item in domain_segments if item.domain_id in predictor_failures),
                "no_calibration_spacing",
            )
            return None, reason
        points, measured_depths = hole.compute_trajectory(step_length=0.5)

        def point_at(depth: float) -> np.ndarray:
            return np.asarray(
                [np.interp(depth, measured_depths, points[:, axis]) for axis in range(3)],
                dtype=np.float64,
            )

        weighted_prediction = 0.0
        total_length = 0.0
        analysis_domain = self.project.spatial_grid_config.analysis_domain
        step = max(min(metadata.spacing) * 0.5, 1e-6)
        for segment in domain_segments:
            if segment.domain_id not in predictors:
                return None, predictor_failures.get(segment.domain_id, "no_same_domain_support")
            cuts = np.linspace(
                segment.from_depth,
                segment.to_depth,
                max(1, int(math.ceil((segment.to_depth - segment.from_depth) / step))) + 1,
            )
            predictor = predictors[segment.domain_id]
            for left, right in zip(cuts[:-1], cuts[1:]):
                midpoint = float((left + right) / 2.0)
                first = point_at(float(left))
                last = point_at(float(right))
                length = float(np.linalg.norm(last - first))
                if length <= 1e-12:
                    continue
                point = point_at(midpoint)
                if not analysis_domain.contains_point(*point):
                    return None, "outside_analysis_domain"
                if not self.project.rock_mask.contains_point(*point) or self.project.excavation_mask.is_excavated(*point):
                    return None, "outside_active_model"
                if isinstance(predictor, float):
                    predicted = predictor
                elif isinstance(predictor, OrdinaryKrigingInterpolator):
                    result = predictor.predict(point)
                    predicted = result.estimate
                    if predicted is None:
                        reason = (
                            "outside_search_radius"
                            if predictor.settings.search_radius is not None
                            else "insufficient_neighbors"
                        )
                        return None, reason
                else:
                    result = predictor.predict(point, segment.domain_id)
                    predicted = result.value
                    if predicted is None:
                        reason = (
                            "outside_search_radius"
                            if predictor.search_radius is not None and result.nearest_distance > predictor.search_radius
                            else "insufficient_neighbors"
                        )
                        return None, reason
                if predicted is None or not math.isfinite(predicted):
                    return None, "other_non_finite_prediction"
                weighted_prediction += float(predicted) * length
                total_length += length
        return (weighted_prediction / total_length, None) if total_length > 0 else (None, "invalid_trajectory")

    @staticmethod
    def _hole_key(value: Any) -> str:
        """Match imported and holdout hole identifiers without changing their persisted spelling."""
        return str(value).strip().casefold()

    def _phase2a_validation_pipeline_diagnostics(
        self,
        all_spacing: list[Any],
        calibration_holes: set[str],
        validation_holes: set[str],
        validation_rows: list[Any],
        validation_segment_count: int,
        *,
        selected_valid: int,
        selected_no_data: int,
        selected_reasons: dict[str, int],
        blocked_reason: str | None,
    ) -> dict[str, Any]:
        """Persist anonymous stage counts and method outcomes for validation troubleshooting."""
        calibration_keys = {self._hole_key(item) for item in calibration_holes}
        validation_keys = {self._hole_key(item) for item in validation_holes}
        calibration_rows = [item for item in all_spacing if self._hole_key(item.hole_id) in calibration_keys]
        qualified_calibration = [item for item in calibration_rows if item.measurement_basis == "BOREHOLE_ALONG_HOLE"]
        qualified_validation = [item for item in validation_rows if item.measurement_basis == "BOREHOLE_ALONG_HOLE"]
        calibration_segments = sum(
            len(self._spacing_domain_segments(item)[0])
            for item in qualified_calibration
            if self._spacing_domain_segments(item)[1] is None
        )
        selected_method = self.project.m9_state.density_settings.method.value
        method_results: dict[str, dict[str, Any]] = {
            selected_method: {
                "valid": selected_valid,
                "no_data": selected_no_data,
                "no_data_reasons": dict(sorted(selected_reasons.items())),
            }
        }
        holes = {self._hole_key(item.borehole_id): item for item in self.project.borehole_collection.boreholes}
        for method in DensityMethod:
            if method.value == selected_method:
                continue
            candidate = DensitySettings.model_validate(
                {**self.project.m9_state.density_settings.model_dump(mode="python"), "method": method}
            )
            predictors, _input_hash, _exclusions, predictor_failures = self._calibration_spacing_p10_predictors(
                calibration_holes, candidate
            )
            valid_count = 0
            reasons: dict[str, int] = {}
            for observation in validation_rows:
                reason = blocked_reason
                predicted = None
                segments, segment_reason = self._spacing_domain_segments(observation)
                if observation.measurement_basis != "BOREHOLE_ALONG_HOLE":
                    reason = "unknown_or_invalid_basis"
                elif self._hole_key(observation.hole_id) not in holes:
                    reason = "invalid_trajectory"
                elif segment_reason is not None:
                    reason = segment_reason
                elif reason is None:
                    predicted, reason = self._integrated_total_p10_prediction(
                        holes[self._hole_key(observation.hole_id)], segments, predictors, predictor_failures
                    )
                if predicted is not None:
                    valid_count += 1
                else:
                    category = reason or "other_no_data"
                    reasons[category] = reasons.get(category, 0) + 1
            method_results[method.value] = {
                "valid": valid_count,
                "no_data": len(validation_rows) - valid_count,
                "no_data_reasons": dict(sorted(reasons.items())),
            }
        return {
            "raw_spacing_records": sum(
                1
                for item in ObservationService(self.project).repository.query(
                    BoreholeDataType.FRACTURES, raw=True
                )
                if item.values.get("observation_mode") == FractureObservationMode.INTERVAL_SPACING
            ),
            "formal_spacing_records": len(all_spacing),
            "holdout_calibration_records": len(calibration_rows),
            "holdout_validation_records": sum(
                1 for item in all_spacing if self._hole_key(item.hole_id) in validation_keys
            ),
            "qualified_calibration_records": len(qualified_calibration),
            "qualified_validation_records": len(qualified_validation),
            "calibration_support_segments": calibration_segments,
            "validation_target_segments": validation_segment_count,
            "selected_method": selected_method,
            "methods": method_results,
        }

    def _validate_selected_density_input(self) -> None:
        """Reject stale or missing Phase 2A lineage before downstream M9 work."""
        state = self.project.m9_state
        if state.density_input_mode != M9DensityInputMode.PHASE2A_REALIZATION:
            return
        realization_id = state.density_input_realization_id
        realization = next(
            (
                item
                for item in self.project.borehole_fracture_state.realizations
                if item.realization_id == realization_id
            ),
            None,
        )
        if realization is None:
            raise RuntimeError("The selected Phase 2A realization no longer exists")
        from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService

        if realization.input_hash != BoreholeFractureService(self.project).authoritative_confirmed_fit().input_hash:
            raise RuntimeError("The selected Phase 2A realization is stale and cannot be reused by M9")

    def _field_value(self, interval, set_id: int) -> float | None:
        """Read a validation prediction from the generated voxel field."""
        if interval.center_x is None:
            return None
        return self._field_value_at((interval.center_x, interval.center_y, interval.center_z), set_id)

    def _field_value_at(self, point, set_id: int) -> float | None:
        """Read one finite set-specific P32 value at an XYZ point."""
        metadata = self.project.m9_state.parameter_field_metadata
        array = self.project.m9_state.parameter_field_arrays.get(f"set_{set_id}_p32")
        if metadata is None or array is None:
            return None
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
