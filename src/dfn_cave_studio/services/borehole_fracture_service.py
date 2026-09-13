"""Phase 2A constrained stochastic fracture generation along boreholes."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal, normal_to_dip_dir_dip
from dfn_cave_studio.models.borehole_fracture_realization import (
    BoreholeFractureComponent,
    BoreholeFractureGenerationConfig,
    BoreholeFractureRealization,
    BoreholeFractureState,
    BoreholeIntervalDiagnostic,
    GlobalJointSetFit,
    GlobalJointSetModel,
    GlobalSetMapping,
    LocalOrientationComponent,
    RandomComponentStatus,
)
from dfn_cave_studio.services.joint_set_service import INSUFFICIENT_ORIENTATIONS_MESSAGE, JointSetService
from dfn_cave_studio.services.observation_service import ObservationService
from dfn_cave_studio.voxel.resource_estimate import (
    available_system_memory_bytes,
    estimate_borehole_fracture_resources,
)


class BoreholeGenerationCancelled(RuntimeError):
    """Raised internally when a Phase 2A calculation is cancelled."""


class FracturePositionSampler(Protocol):
    """Extensible measured-depth sampling strategy contract."""

    name: str

    def sample_count(
        self,
        rng: np.random.Generator,
        from_depth: float,
        to_depth: float,
        spacing_m: float,
    ) -> int:
        """Sample only the event count without allocating fracture rows."""

    def fill_depths(
        self,
        output: np.ndarray,
        rng: np.random.Generator,
        from_depth: float,
        to_depth: float,
        *,
        chunk_size: int,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        """Fill a final-array view with sorted half-open measured depths."""


class HomogeneousPoissonSampler:
    """Homogeneous Poisson process conditioned on a Poisson event count."""

    name = "HOMOGENEOUS_POISSON"

    def sample_count(
        self,
        rng: np.random.Generator,
        from_depth: float,
        to_depth: float,
        spacing_m: float,
    ) -> int:
        if not 0 <= from_depth < to_depth or spacing_m <= 0:
            raise ValueError("A valid half-open interval and positive spacing are required")
        return int(rng.poisson((to_depth - from_depth) / spacing_m))

    def fill_depths(
        self,
        output: np.ndarray,
        rng: np.random.Generator,
        from_depth: float,
        to_depth: float,
        *,
        chunk_size: int,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        if not 0 <= from_depth < to_depth:
            raise ValueError("A valid half-open interval is required")
        for start in range(0, len(output), chunk_size):
            if cancelled is not None and cancelled():
                raise BoreholeGenerationCancelled("Borehole fracture generation cancelled")
            stop = min(len(output), start + chunk_size)
            output[start:stop] = rng.uniform(from_depth, to_depth, size=stop - start)
        np.minimum(output, np.nextafter(float(to_depth), float(from_depth)), out=output)
        output.sort()


POSITION_SAMPLERS: dict[str, FracturePositionSampler] = {
    HomogeneousPoissonSampler.name: HomogeneousPoissonSampler(),
}


class StableIDW:
    """Small deterministic 3-D IDW query preserving original-index tie order."""

    def __init__(self, power: float, search_radius: float, max_neighbors: int, min_neighbors: int):
        if power <= 0 or search_radius <= 0 or max_neighbors < 1 or min_neighbors < 1:
            raise ValueError("IDW parameters must be positive")
        if min_neighbors > max_neighbors:
            raise ValueError("min_neighbors must be <= max_neighbors")
        self.power = float(power)
        self.search_radius = float(search_radius)
        self.max_neighbors = int(max_neighbors)
        self.min_neighbors = int(min_neighbors)

    def predict(self, coordinates: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        """Predict a scalar or vector, or return None when local support is insufficient."""
        coordinates = np.asarray(coordinates, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) != len(values):
            raise ValueError("IDW coordinates must have shape (N, 3) and align with values")
        distances = np.linalg.norm(coordinates - target, axis=1)
        candidates = np.flatnonzero(distances <= self.search_radius)
        if len(candidates) < self.min_neighbors:
            return None
        order = np.lexsort((candidates, distances[candidates]))
        selected = candidates[order[: self.max_neighbors]]
        exact = selected[distances[selected] <= 1e-12]
        if len(exact):
            return np.asarray(values[int(exact[0])], dtype=np.float64)
        weights = distances[selected] ** (-self.power)
        return np.asarray(np.tensordot(weights / weights.sum(), values[selected], axes=(0, 0)), dtype=np.float64)

    def predict_many(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict bounded target batches without a target-by-sample distance matrix."""
        targets = np.asarray(targets, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("IDW targets must have shape (N, 3)")
        output_shape = (len(targets), *values.shape[1:])
        output = np.zeros(output_shape, dtype=np.float64)
        valid = np.zeros(len(targets), dtype=bool)
        for index, target in enumerate(targets):
            predicted = self.predict(coordinates, values, target)
            if predicted is None:
                continue
            output[index] = predicted
            valid[index] = True
        return output, valid


class BoreholeFractureService:
    """Fit global sets and build complete columnar borehole realizations transactionally."""

    REQUIRED_DTYPES = {
        "interval_index": np.dtype("int32"),
        "measured_depth": np.dtype("float64"),
        "xyz_offset": np.dtype("float32"),
        "global_set_id": np.dtype("int32"),
        "local_component_index": np.dtype("int32"),
        "component_type": np.dtype("uint8"),
        "dip": np.dtype("float32"),
        "dip_direction": np.dtype("float32"),
        "status": np.dtype("uint8"),
    }

    def __init__(self, project: Any):
        self.project = project
        self.observations = ObservationService(project)

    def fit_global_sets(self, number_of_sets: int, random_seed: int) -> GlobalJointSetFit:
        """Cluster complete P/Z representative orientations with explicit weights."""
        rows = [
            item
            for item in self.observations.orientation_points()
            if item.component_type == "LOCAL_DOMINANT_SET"
            and item.orientation_status == "COMPLETE"
            and item.dip is not None
            and item.dip_direction is not None
        ]
        rows.sort(key=lambda item: item.observation_id)
        if number_of_sets < 1 or len(rows) < number_of_sets:
            raise ValueError(INSUFFICIENT_ORIENTATIONS_MESSAGE)
        normals = np.asarray([dip_dir_dip_to_normal(item.dip_direction, item.dip) for item in rows])
        weights = np.asarray(
            [item.joint_num if item.source_kind == "POINT_CLOUD" and item.joint_num is not None else 1 for item in rows],
            dtype=np.int64,
        )
        distinct = JointSetService._distinct_axial_indices(normals)
        if len(distinct) < number_of_sets:
            raise ValueError(INSUFFICIENT_ORIENTATIONS_MESSAGE)
        labels, centroids = self._weighted_axial_kmeans(normals, weights, number_of_sets, random_seed)
        descriptors: list[tuple[int, float, float, float]] = []
        for cluster in range(number_of_sets):
            mask = labels == cluster
            dd, dip, kappa = self._weighted_fisher(normals[mask], weights[mask])
            descriptors.append((cluster, dd % 360.0, dip, kappa))
        ordered = sorted(descriptors, key=lambda item: (item[1], item[2], item[0]))
        remap = {cluster: index for index, (cluster, *_rest) in enumerate(ordered, start=1)}
        sets: list[GlobalJointSetModel] = []
        for cluster, dd, dip, kappa in ordered:
            mask = labels == cluster
            sets.append(
                GlobalJointSetModel(
                    global_set_id=remap[cluster],
                    mean_dip_direction=dd,
                    mean_dip=dip,
                    kappa=kappa,
                    kappa_status=("UNRESOLVED" if int(np.count_nonzero(mask)) < 2 else "SITE_MEAN_DISPERSION"),
                    representative_count=int(np.count_nonzero(mask)),
                    weighted_count=int(weights[mask].sum()),
                )
            )
        mappings = [
            GlobalSetMapping(
                observation_id=item.observation_id,
                component_id=self._component_id(item),
                point_key=item.point_key,
                local_set_id=item.local_set_id,
                source_kind=item.source_kind,
                global_set_id=remap[int(labels[index])],
                sample_weight=int(weights[index]),
            )
            for index, item in enumerate(rows)
        ]
        input_hash = self.input_hash()
        return GlobalJointSetFit(
            number_of_sets=number_of_sets,
            random_seed=random_seed,
            sets=sets,
            mappings=mappings,
            local_components=self._local_components(mappings),
            input_hash=input_hash,
            provenance={
                "weighting": "P representative joint_num; Z representative weight 1",
                "meaning": "clustered representative orientations, not raw individual fracture orientations",
                "random_and_missing_directions_excluded": True,
                "axial_plane_normals": True,
                "kappa_semantics": "UNRESOLVED for one representative; otherwise SITE_MEAN_DISPERSION",
                "intensity_semantics": (
                    "P-site reciprocal spacing supplies relative group/random allocation only; borehole interval "
                    "spacing alone controls Poisson total count; neither is automatically converted to P32"
                ),
            },
        )

    def use_confirmed_project_sets(self, joint_sets: list[Any] | None = None) -> GlobalJointSetFit:
        """Use confirmed project sets and map P/Z representatives by axial proximity."""
        project_sets = sorted(self.project.joint_sets if joint_sets is None else joint_sets, key=lambda item: item.set_id)
        if not project_sets:
            raise ValueError(
                "No confirmed project joint sets are available. Fit global groups from P/Z representative "
                "orientations instead."
            )
        rows = [
            item
            for item in self.observations.orientation_points()
            if item.component_type == "LOCAL_DOMINANT_SET"
            and item.orientation_status == "COMPLETE"
            and item.dip is not None
            and item.dip_direction is not None
        ]
        rows.sort(key=lambda item: item.observation_id)
        weights = np.asarray(
            [item.joint_num if item.source_kind == "POINT_CLOUD" and item.joint_num is not None else 1 for item in rows],
            dtype=np.int64,
        )
        set_normals = np.asarray(
            [dip_dir_dip_to_normal(item.orientation.mean_dip_direction, item.orientation.mean_dip) for item in project_sets]
        )
        if rows:
            row_normals = np.asarray([dip_dir_dip_to_normal(item.dip_direction, item.dip) for item in rows])
            labels = np.argmax(np.abs(row_normals @ set_normals.T), axis=1)
        else:
            labels = np.empty(0, dtype=np.int64)
        mappings = [
            GlobalSetMapping(
                observation_id=item.observation_id,
                component_id=self._component_id(item),
                point_key=item.point_key,
                local_set_id=item.local_set_id,
                source_kind=item.source_kind,
                global_set_id=project_sets[int(labels[index])].set_id,
                sample_weight=int(weights[index]),
            )
            for index, item in enumerate(rows)
        ]
        models = []
        for set_index, item in enumerate(project_sets):
            mask = labels == set_index
            kappa_status = str(item.provenance.get("kappa_status", "LEGACY_UNSPECIFIED"))
            if kappa_status not in {
                "UNRESOLVED",
                "SITE_MEAN_DISPERSION",
                "MEASURED_WITHIN_SET",
                "MANUAL",
                "ASSUMED",
                "LEGACY_UNSPECIFIED",
            }:
                kappa_status = "LEGACY_UNSPECIFIED"
            models.append(
                GlobalJointSetModel(
                    global_set_id=item.set_id,
                    mean_dip_direction=item.orientation.mean_dip_direction,
                    mean_dip=item.orientation.mean_dip,
                    kappa=item.orientation.kappa,
                    kappa_status=kappa_status,
                    representative_count=int(np.count_nonzero(mask)),
                    weighted_count=int(weights[mask].sum()) if np.any(mask) else 0,
                )
            )
        payload = {
            "phase2a_input_hash": self.input_hash(),
            "project_joint_sets": [
                {
                    "set_id": item.set_id,
                    "dip_direction": item.orientation.mean_dip_direction,
                    "dip": item.orientation.mean_dip,
                    "kappa": item.orientation.kappa,
                }
                for item in project_sets
            ],
        }
        confirmed_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return GlobalJointSetFit(
            number_of_sets=len(project_sets),
            random_seed=0,
            sets=models,
            mappings=mappings,
            local_components=self._local_components(mappings),
            input_hash=confirmed_hash,
            algorithm="CONFIRMED_PROJECT_JOINT_SETS_WITH_AXIAL_NEAREST_MAPPING",
            provenance={
                "source": "confirmed_project_joint_sets",
                "mapping": "nearest absolute axial-normal dot product; ties use ascending project set_id",
                "local_set_ids_are_point_local_only": True,
                "random_and_missing_directions_excluded": True,
                "no_orientation_was_invented": True,
                "kappa_semantics": "read from each confirmed project joint-set provenance",
            },
        )

    def authoritative_confirmed_fit(self, joint_sets: list[Any] | None = None) -> GlobalJointSetFit:
        """Return the saved authoritative fit when it still matches current inputs and sets.

        A previously confirmed P/Z fit contains both the global orientation models and
        the local-to-global mapping reviewed by the user.  Reopening a dialog must not
        silently replace that fit with a newly derived preview.  If the confirmed sets
        or imported representative rows changed, a deterministic axial-nearest mapping
        is built instead and can be explicitly confirmed by the caller.
        """
        project_sets = sorted(self.project.joint_sets if joint_sets is None else joint_sets, key=lambda item: item.set_id)
        if not project_sets:
            return self.use_confirmed_project_sets(project_sets)
        saved = self.project.borehole_fracture_state.global_fit
        rows = [
            item
            for item in self.observations.orientation_points()
            if item.component_type == "LOCAL_DOMINANT_SET"
            and item.orientation_status == "COMPLETE"
            and item.dip is not None
            and item.dip_direction is not None
        ]
        expected_observation_ids = {item.observation_id for item in rows}
        if saved is not None and len(saved.sets) == len(project_sets):
            saved_models = {item.global_set_id: item for item in saved.sets}
            set_models_match = all(
                item.set_id in saved_models
                and abs(item.orientation.mean_dip_direction - saved_models[item.set_id].mean_dip_direction) <= 1e-10
                and abs(item.orientation.mean_dip - saved_models[item.set_id].mean_dip) <= 1e-10
                and abs(item.orientation.kappa - saved_models[item.set_id].kappa) <= 1e-10
                for item in project_sets
            )
            mapped_observation_ids = {item.observation_id for item in saved.mappings}
            mapped_set_ids = {item.global_set_id for item in saved.mappings}
            confirmed_set_ids = {item.set_id for item in project_sets}
            if (
                set_models_match
                and mapped_observation_ids == expected_observation_ids
                and mapped_set_ids <= confirmed_set_ids
            ):
                current_components = self._local_components(saved.mappings)
                saved_component_ids = [item.component_id for item in saved.local_components]
                if saved_component_ids == [item.component_id for item in current_components]:
                    return saved.model_copy(deep=True)
        return self.use_confirmed_project_sets(project_sets)

    @staticmethod
    def _component_id(item: Any) -> str:
        """Return a stable identity without treating point-local set labels as global."""
        payload = "\x1f".join(
            (item.source_kind, item.point_id, item.local_set_id, item.observation_id)
        ).encode("utf-8")
        return f"lc-{hashlib.sha256(payload).hexdigest()[:24]}"

    def _local_components(self, mappings: list[GlobalSetMapping]) -> list[LocalOrientationComponent]:
        """Build the immutable component table once for dominant and random rows."""
        mapped = {item.observation_id: item.global_set_id for item in mappings}
        components: list[LocalOrientationComponent] = []
        for item in sorted(self.observations.orientation_points(), key=lambda value: value.observation_id):
            if item.component_type == "LOCAL_DOMINANT_SET" and item.observation_id not in mapped:
                continue
            if item.component_type == "RANDOM_BACKGROUND" and item.source_kind != "POINT_CLOUD":
                continue
            components.append(
                LocalOrientationComponent(
                    component_id=self._component_id(item),
                    observation_id=item.observation_id,
                    point_key=item.point_key,
                    point_id=item.point_id,
                    local_set_id=item.local_set_id,
                    source_kind=item.source_kind,
                    x=item.x,
                    y=item.y,
                    z=item.z,
                    dip=item.dip,
                    dip_direction=item.dip_direction,
                    joint_spacing_m=item.joint_spacing_m,
                    joint_num=item.joint_num,
                    component_type=item.component_type,
                    global_set_id=mapped.get(item.observation_id),
                    source_file=item.source_file,
                    source_row=item.source_row,
                )
            )
        return components

    def build_candidate(
        self,
        config: BoreholeFractureGenerationConfig,
        *,
        fit_override: GlobalJointSetFit | None = None,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> BoreholeFractureState:
        """Build all requested realizations without mutating the project."""
        if fit_override is not None:
            fit = fit_override.model_copy(deep=True)
        elif config.use_confirmed_global_fit:
            fit = self.authoritative_confirmed_fit()
        else:
            fit = self.fit_global_sets(config.number_of_sets, config.master_seed)
        if not fit.local_components:
            fit.local_components = self._local_components(fit.mappings)
        intervals = self.observations.spacing_observations()
        effective_radius = np.inf if config.idw_search_mode == "ALL_WITHIN_DOMAIN" else config.search_radius
        idw = StableIDW(config.idw_power, effective_radius, config.max_neighbors, config.min_neighbors)
        constraints = self._intensity_constraints(fit)
        constraints["search_mode"] = config.idw_search_mode
        supports = self._interval_supports(intervals, fit, idw, constraints)
        counts, offsets = self._sample_realization_counts(config, intervals, supports, cancelled, progress)
        expected_total = sum((item.to_depth - item.from_depth) * item.derived_p10 for item in intervals)
        estimate = self._resource_details(
            config,
            [int(value) for value in offsets[:, -1]],
            interval_count=len(intervals),
            expected_fractures=expected_total * config.realization_count,
            estimate_basis="actual_poisson_counts",
            component_count=len(fit.local_components),
        )
        if estimate["exceeds_budget"]:
            raise MemoryError(
                f"Actual-count peak memory {estimate['peak_bytes'] / 1024**2:.1f} MiB exceeds "
                f"the configured budget {config.memory_budget_bytes / 1024**2:.1f} MiB"
            )
        realizations = []
        for index in range(config.realization_count):
            self._check_cancel(cancelled)
            realizations.append(
                self._generate_one(
                    config,
                    fit,
                    index,
                    intervals,
                    supports,
                    counts[index],
                    offsets[index],
                    idw,
                    constraints,
                    cancelled,
                    progress,
                )
            )
        return BoreholeFractureState(
            config=config,
            global_fit=fit,
            random_component_status={
                item.point_key: item.random_component_status
                for item in self.observations.orientation_point_summaries()
            },
            realizations=realizations,
            selected_realization_id=realizations[0].realization_id if realizations else None,
        )

    def commit(self, candidate: BoreholeFractureState) -> None:
        """Commit only a complete candidate state."""
        if any(not item.complete for item in candidate.realizations):
            raise ValueError("Partial borehole-fracture realizations cannot be committed")
        self.project.borehole_fracture_state = candidate

    def estimate(self, config: BoreholeFractureGenerationConfig) -> dict[str, int | float]:
        """Preview expected ownership before stochastic counts are available."""
        intervals = self.observations.spacing_observations()
        expected_per_realization = sum((item.to_depth - item.from_depth) * item.derived_p10 for item in intervals)
        return self._resource_details(
            config,
            [int(np.ceil(expected_per_realization))] * config.realization_count,
            interval_count=len(intervals),
            expected_fractures=expected_per_realization * config.realization_count,
            estimate_basis="expected_count_ceiling",
            component_count=len(self.observations.orientation_points()),
        )

    @staticmethod
    def _resource_details(
        config: BoreholeFractureGenerationConfig,
        realization_counts: list[int],
        *,
        interval_count: int,
        expected_fractures: float,
        estimate_basis: str,
        component_count: int = 0,
    ) -> dict[str, int | float | str | bool]:
        resources = estimate_borehole_fracture_resources(
            realization_counts,
            interval_count=interval_count,
            chunk_size=config.generation_chunk_size,
            budget_bytes=config.memory_budget_bytes,
            system_available_bytes=available_system_memory_bytes(),
            component_count=component_count,
        )
        return {
            "expected_fractures": expected_fractures,
            "actual_fractures": sum(realization_counts) if estimate_basis == "actual_poisson_counts" else None,
            "persistent_bytes": resources.persistent_bytes,
            "temporary_bytes": resources.temporary_bytes,
            "safety_margin_bytes": resources.safety_margin_bytes,
            "peak_bytes": resources.peak_bytes,
            "exceeds_budget": resources.exceeds_budget,
            "simultaneously_resident_realizations": len(realization_counts),
            "estimate_basis": estimate_basis,
        }

    def input_hash(self) -> str:
        """Hash only formal Phase 2A inputs in stable scientific order."""
        orientation = [item.model_dump(mode="json") for item in self.observations.orientation_points()]
        spacing = [item.model_dump(mode="json") for item in self.observations.spacing_observations()]
        statuses = [item.model_dump(mode="json") for item in self.observations.orientation_point_summaries()]
        payload = {"orientation_points": orientation, "spacing_intervals": spacing, "random_status": statuses}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def validate_arrays(cls, realization: BoreholeFractureRealization) -> None:
        """Validate the compact column contract and aligned row count."""
        arrays = realization.arrays
        missing = cls.REQUIRED_DTYPES.keys() - arrays.keys()
        if missing:
            raise ValueError(f"Missing borehole realization arrays: {sorted(missing)}")
        count = realization.fracture_count
        for name, dtype in cls.REQUIRED_DTYPES.items():
            array = np.asarray(arrays[name])
            if array.dtype != dtype:
                raise ValueError(f"{name} must use {dtype}, not {array.dtype}")
            if len(array) != count:
                raise ValueError(f"{name} row count does not match fracture_count")
        if arrays["xyz_offset"].shape != (count, 3):
            raise ValueError("xyz_offset must have shape (N, 3)")
        if np.any(arrays["local_component_index"] < 0):
            raise ValueError("local_component_index must reference the fit-level component table")
        random_mask = arrays["component_type"] == int(BoreholeFractureComponent.RANDOM_BACKGROUND)
        dominant_mask = arrays["component_type"] == int(BoreholeFractureComponent.DOMINANT_SET)
        if np.any(arrays["global_set_id"][random_mask] != -1):
            raise ValueError("Random background rows must use the -1 global-set sentinel")
        if np.any(arrays["global_set_id"][dominant_mask] <= 0):
            raise ValueError("Dominant local components must reference a positive global set")

    def _interval_supports(
        self,
        intervals: list[Any],
        fit: GlobalJointSetFit,
        idw: StableIDW,
        constraints: dict[str, Any],
    ) -> list[dict[str, Any]]:
        supports: list[dict[str, Any]] = []
        for interval in intervals:
            midpoint = self.observations.position_at_depth(
                interval.hole_id, (interval.from_depth + interval.to_depth) / 2.0
            )
            supports.append(self._local_probabilities(idw, constraints, midpoint, fit))
        return supports

    def _sample_realization_counts(
        self,
        config: BoreholeFractureGenerationConfig,
        intervals: list[Any],
        supports: list[dict[str, Any]],
        cancelled: Callable[[], bool] | None,
        progress: Callable[[int, int, str], None] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Phase one: sample only small count/offset plans for every realization."""
        counts = np.zeros((config.realization_count, len(intervals)), dtype=np.int64)
        sampler = POSITION_SAMPLERS[config.position_sampler]
        total_steps = max(config.realization_count * len(intervals), 1)
        completed = 0
        for realization_index in range(config.realization_count):
            for interval_index, interval in enumerate(intervals):
                self._check_cancel(cancelled)
                sampled = sampler.sample_count(
                    config.rng_for(realization_index, interval_index, 1),
                    interval.from_depth,
                    interval.to_depth,
                    1.0 / interval.derived_p10,
                )
                if supports[interval_index]["local_probabilities"]:
                    counts[realization_index, interval_index] = sampled
                completed += 1
                if progress is not None:
                    progress(completed, total_steps, f"Sampling Poisson count {completed}/{total_steps}")
        offsets = np.zeros((config.realization_count, len(intervals) + 1), dtype=np.int64)
        if intervals:
            offsets[:, 1:] = np.cumsum(counts, axis=1)
        return counts, offsets

    def _generate_one(
        self,
        config: BoreholeFractureGenerationConfig,
        fit: GlobalJointSetFit,
        realization_index: int,
        intervals: list[Any],
        supports: list[dict[str, Any]],
        counts: np.ndarray,
        offsets: np.ndarray,
        idw: StableIDW,
        constraints: dict[str, Any],
        cancelled: Callable[[], bool] | None,
        progress: Callable[[int, int, str], None] | None,
    ) -> BoreholeFractureRealization:
        seed = config.seed_for(realization_index)
        origin = np.asarray(
            (self.project.model_bounds.origin_x, self.project.model_bounds.origin_y, self.project.model_bounds.origin_z),
            dtype=np.float64,
        )
        arrays = self._allocate_arrays(int(offsets[-1]))
        diagnostics: list[BoreholeIntervalDiagnostic] = []
        set_counts: dict[int, int] = defaultdict(int)
        random_count = 0
        random_diagnostic = self._random_component_diagnostic(constraints)
        for interval_index, interval in enumerate(intervals):
            self._check_cancel(cancelled)
            support = supports[interval_index]
            probabilities = support["global_set_probabilities"]
            random_probability = support["random_probability"]
            start, stop = int(offsets[interval_index]), int(offsets[interval_index + 1])
            count = int(counts[interval_index])
            spacing_m = 1.0 / interval.derived_p10
            supported = bool(support["local_probabilities"])
            diagnostics.append(
                BoreholeIntervalDiagnostic(
                    interval_index=interval_index,
                    observation_id=interval.observation_id,
                    hole_id=interval.hole_id,
                    from_depth=interval.from_depth,
                    to_depth=interval.to_depth,
                    spacing_m=spacing_m,
                    expected_count=(interval.to_depth - interval.from_depth) / spacing_m,
                    generated_count=count,
                    status="GENERATED" if supported else "BLOCKED_NO_LOCAL_INTENSITY",
                    global_set_probabilities=probabilities,
                    local_component_probabilities=support["local_probabilities"],
                    random_probability=random_probability,
                    random_component_diagnostic=random_diagnostic,
                    selected_point_count=support["selected_point_count"],
                    nearest_constraint_distance=support["nearest_distance"],
                    farthest_constraint_distance=support["farthest_distance"],
                    is_long_range_extrapolation=support["is_long_range_extrapolation"],
                )
            )
            if count:
                interval_set_counts, interval_random = self._fill_interval(
                    arrays,
                    slice(start, stop),
                    interval_index,
                    interval,
                    support,
                    fit,
                    idw,
                    constraints,
                    config,
                    realization_index,
                    origin,
                    cancelled,
                )
                for set_id, value in interval_set_counts.items():
                    set_counts[set_id] += value
                random_count += interval_random
            if progress is not None:
                progress(interval_index + 1, len(intervals), f"Generating interval {interval_index + 1}/{len(intervals)}")
        input_hash = fit.input_hash
        realization_id = f"bhf-{realization_index:03d}-{hashlib.sha256(f'{input_hash}:{seed}'.encode()).hexdigest()[:16]}"
        realization = BoreholeFractureRealization(
            realization_id=realization_id,
            realization_index=realization_index,
            master_seed=config.master_seed,
            derived_seed=seed,
            input_hash=input_hash,
            fracture_count=len(arrays["measured_depth"]),
            interval_diagnostics=diagnostics,
            global_set_counts=dict(sorted(set_counts.items())),
            random_background_count=random_count,
            array_member=f"results/borehole_fractures/{realization_id}.npz",
            provenance={
                "xyz_origin": origin.tolist(),
                "half_open_intervals": True,
                "seed_strategy": (
                    "SeedSequence(master_seed, realization_index, interval_index, stream_id, 0x2A); "
                    "independent count/depth/group/random/set-direction streams"
                ),
                "memory_ownership": (
                    "all requested final realization arrays resident in candidate; one bounded generation chunk live"
                ),
                "direction_dispersion_notice": (
                    "SPATIALLY_FITTED_WITHIN_GLOBAL_SET reflects differences among representative site means, "
                    "not raw within-set orientation dispersion."
                ),
                "local_component_table": "GlobalJointSetFit.local_components; array rows store compact indices",
                "spatial_probability": (
                    "select P sites by (distance, stable point index), include every valid component at each selected "
                    "site, and normalize reciprocal-spacing times one shared site distance kernel"
                ),
            },
            arrays=arrays,
        )
        self.validate_arrays(realization)
        if realization.fracture_count and int(np.max(arrays["local_component_index"])) >= len(fit.local_components):
            raise ValueError("local_component_index exceeds the fit-level component table")
        return realization

    def _fill_interval(
        self,
        arrays: dict[str, np.ndarray],
        output_slice: slice,
        interval_index: int,
        interval: Any,
        support: dict[str, Any],
        fit: GlobalJointSetFit,
        idw: StableIDW,
        constraints: dict[str, Any],
        config: BoreholeFractureGenerationConfig,
        realization_index: int,
        origin: np.ndarray,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[dict[int, int], int]:
        start = int(output_slice.start or 0)
        stop = int(output_slice.stop or start)
        count = stop - start
        depths = arrays["measured_depth"][output_slice]
        POSITION_SAMPLERS[config.position_sampler].fill_depths(
            depths,
            config.rng_for(realization_index, interval_index, 2),
            interval.from_depth,
            interval.to_depth,
            chunk_size=config.generation_chunk_size,
            cancelled=cancelled,
        )
        arrays["interval_index"][output_slice] = interval_index
        arrays["status"][output_slice] = 1
        component_probabilities = support["local_probabilities"]
        choices = np.asarray(sorted(component_probabilities), dtype=np.int32)
        choice_probabilities = np.asarray([component_probabilities[key] for key in choices], dtype=np.float64)
        choice_probabilities /= choice_probabilities.sum()
        cumulative = np.cumsum(choice_probabilities)
        cumulative[-1] = 1.0
        assignment_rng = config.rng_for(realization_index, interval_index, 3)
        random_polar_rng = config.rng_for(realization_index, interval_index, 4)
        random_azimuth_rng = config.rng_for(realization_index, interval_index, 5)
        models = {item.global_set_id: item for item in fit.sets}
        component_table = fit.local_components
        component_global_ids = np.asarray(
            [item.global_set_id or -1 for item in component_table], dtype=np.int32
        )
        component_types = np.asarray(
            [
                BoreholeFractureComponent.RANDOM_BACKGROUND
                if item.component_type == "RANDOM_BACKGROUND"
                else BoreholeFractureComponent.DOMINANT_SET
                for item in component_table
            ],
            dtype=np.uint8,
        )
        component_dips = np.asarray(
            [np.nan if item.dip is None else item.dip for item in component_table], dtype=np.float32
        )
        component_directions = np.asarray(
            [np.nan if item.dip_direction is None else item.dip_direction for item in component_table],
            dtype=np.float32,
        )
        set_counts: dict[int, int] = defaultdict(int)
        random_count = 0
        for local_start in range(0, count, config.generation_chunk_size):
            self._check_cancel(cancelled)
            local_stop = min(count, local_start + config.generation_chunk_size)
            target = slice(start + local_start, start + local_stop)
            chunk_count = local_stop - local_start
            chunk_component_indices = choices[
                np.searchsorted(cumulative, assignment_rng.random(chunk_count), side="right")
            ]
            arrays["local_component_index"][target] = chunk_component_indices
            chunk_ids = component_global_ids[chunk_component_indices]
            arrays["global_set_id"][target] = chunk_ids
            chunk_component_types = component_types[chunk_component_indices]
            arrays["component_type"][target] = chunk_component_types
            random_mask = chunk_component_types == int(BoreholeFractureComponent.RANDOM_BACKGROUND)
            xyz = self._positions_at_depths(interval.hole_id, depths[local_start:local_stop])
            arrays["xyz_offset"][target] = (xyz - origin).astype(np.float32)
            chunk_direction = arrays["dip_direction"][target]
            chunk_dip = arrays["dip"][target]
            if np.any(random_mask):
                amount = int(np.count_nonzero(random_mask))
                normals = self._sample_isotropic_axial(amount, random_polar_rng, random_azimuth_rng)
                dd, dips = self._normals_to_orientations(normals)
                chunk_direction[random_mask] = dd
                chunk_dip[random_mask] = dips
                random_count += amount
            for set_id in sorted(models):
                mask = chunk_ids == set_id
                amount = int(np.count_nonzero(mask))
                if not amount:
                    continue
                model = models[set_id]
                if config.direction_mode == "SPATIALLY_FITTED_WITHIN_GLOBAL_SET":
                    mean_normals, valid = self._directions_at(idw, constraints, set_id, xyz[mask], fit)
                    if not np.all(valid):
                        raise ValueError(
                            f"Missing valid axial direction support for interval {interval.observation_id}, "
                            f"global set {set_id}, at {int(np.count_nonzero(~valid))} generated position(s)"
                        )
                    dd, dips = self._normals_to_orientations(mean_normals)
                    chunk_direction[mask] = dd
                    chunk_dip[mask] = dips
                elif config.direction_mode == "LOCAL_REPRESENTATIVE":
                    local_indices = chunk_component_indices[mask]
                    local_directions = component_directions[local_indices]
                    local_dips = component_dips[local_indices]
                    if not np.isfinite(local_directions).all() or not np.isfinite(local_dips).all():
                        raise ValueError(f"Global set {set_id} contains a selected local component with no direction")
                    chunk_direction[mask] = local_directions
                    chunk_dip[mask] = local_dips
                else:
                    chunk_direction[mask] = model.mean_dip_direction
                    chunk_dip[mask] = model.mean_dip
                set_counts[set_id] += amount
        return dict(set_counts), random_count

    @classmethod
    def _allocate_arrays(cls, count: int) -> dict[str, np.ndarray]:
        """Allocate each final scientific column exactly once."""
        return {
            name: np.empty((count, 3), dtype=dtype) if name == "xyz_offset" else np.empty(count, dtype=dtype)
            for name, dtype in cls.REQUIRED_DTYPES.items()
        }

    def _intensity_constraints(self, fit: GlobalJointSetFit) -> dict[str, Any]:
        """Create one stable component table and point-grouped P-site constraints."""
        observations = {item.observation_id: item for item in self.observations.orientation_points()}
        component_indices = {item.component_id: index for index, item in enumerate(fit.local_components)}
        p_sites: dict[str, dict[str, Any]] = {}
        directions: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        for mapping in fit.mappings:
            item = observations[mapping.observation_id]
            coordinate = np.asarray((item.x, item.y, item.z), dtype=np.float64)
            normal = dip_dir_dip_to_normal(item.dip_direction, item.dip)
            directions[mapping.global_set_id].append((coordinate, normal))
            if item.source_kind != "POINT_CLOUD" or item.joint_spacing_m is None:
                continue
            site = p_sites.setdefault(item.point_key, {"coordinate": coordinate, "components": []})
            site["components"].append(
                {
                    "component_index": component_indices[mapping.component_id],
                    "global_set_id": mapping.global_set_id,
                    "component_type": "LOCAL_DOMINANT_SET",
                    "intensity": 1.0 / item.joint_spacing_m,
                }
            )
        statuses = {item.point_key: item.random_component_status for item in self.observations.orientation_point_summaries()}
        for item in observations.values():
            if (
                item.source_kind == "POINT_CLOUD"
                and item.component_type == "RANDOM_BACKGROUND"
                and item.joint_spacing_m is not None
                and statuses.get(item.point_key) == RandomComponentStatus.REPORTED_PRESENT
            ):
                coordinate = np.asarray((item.x, item.y, item.z), dtype=np.float64)
                site = p_sites.setdefault(item.point_key, {"coordinate": coordinate, "components": []})
                site["components"].append(
                    {
                        "component_index": component_indices[self._component_id(item)],
                        "global_set_id": None,
                        "component_type": "RANDOM_BACKGROUND",
                        "intensity": 1.0 / item.joint_spacing_m,
                    }
                )
        ordered_sites = []
        for stable_index, point_key in enumerate(sorted(p_sites)):
            site = p_sites[point_key]
            site["point_key"] = point_key
            site["stable_index"] = stable_index
            site["components"].sort(key=lambda value: value["component_index"])
            ordered_sites.append(site)
        return {"p_sites": ordered_sites, "directions": directions, "random_present": any(
            component["component_type"] == "RANDOM_BACKGROUND"
            for site in ordered_sites
            for component in site["components"]
        )}

    def _random_component_diagnostic(self, constraints: dict[str, Any]) -> str:
        if constraints["random_present"]:
            return "INTERPOLATED_REPORTED_PRESENT"
        statuses = [item.random_component_status for item in self.observations.orientation_point_summaries()]
        if RandomComponentStatus.NOT_REPORTED in statuses:
            return "NOT_REPORTED_NO_RANDOM_ESTIMATE"
        if statuses and all(item == RandomComponentStatus.REPORTED_ABSENT for item in statuses):
            return "REPORTED_ABSENT_NO_RANDOM_COMPONENT"
        return "NO_POINT_CLOUD_RANDOM_CONSTRAINT"

    @staticmethod
    def _local_probabilities(
        idw: StableIDW, constraints: dict[str, Any], target: np.ndarray, fit: GlobalJointSetFit
    ) -> dict[str, Any]:
        """Select P sites once, then normalize every component contribution."""
        del fit  # Mapping is already materialized in the immutable component table.
        sites = constraints["p_sites"]
        empty = {
            "local_probabilities": {},
            "global_set_probabilities": {},
            "random_probability": 0.0,
            "selected_point_count": 0,
            "nearest_distance": None,
            "farthest_distance": None,
            "is_long_range_extrapolation": False,
        }
        if not sites:
            return empty
        coordinates = np.asarray([site["coordinate"] for site in sites], dtype=np.float64)
        distances = np.linalg.norm(coordinates - np.asarray(target, dtype=np.float64), axis=1)
        exact = np.flatnonzero(distances <= 1e-12)
        if len(exact):
            selected = exact
            kernels = np.ones(len(selected), dtype=np.float64)
        else:
            candidates = np.flatnonzero(distances <= idw.search_radius)
            if len(candidates) < idw.min_neighbors:
                return empty
            order = np.lexsort((np.asarray([sites[index]["stable_index"] for index in candidates]), distances[candidates]))
            ordered = candidates[order]
            selected = ordered if constraints.get("search_mode") == "ALL_WITHIN_DOMAIN" else ordered[: idw.max_neighbors]
            kernels = distances[selected] ** (-idw.power)
        if len(selected) < idw.min_neighbors:
            return empty
        contributions: dict[int, float] = {}
        global_contributions: dict[int, float] = defaultdict(float)
        random_contribution = 0.0
        for site_index, kernel in zip(selected, kernels, strict=True):
            for component in sites[int(site_index)]["components"]:
                contribution = float(component["intensity"] * kernel)
                contributions[int(component["component_index"])] = contribution
                if component["component_type"] == "RANDOM_BACKGROUND":
                    random_contribution += contribution
                else:
                    global_contributions[int(component["global_set_id"])] += contribution
        total = sum(contributions.values())
        if total <= 0:
            return empty
        nearest = float(np.min(distances[selected]))
        farthest = float(np.max(distances[selected]))
        return {
            "local_probabilities": {key: value / total for key, value in sorted(contributions.items())},
            "global_set_probabilities": {
                key: value / total for key, value in sorted(global_contributions.items())
            },
            "random_probability": random_contribution / total,
            "selected_point_count": int(len(selected)),
            "nearest_distance": nearest,
            "farthest_distance": farthest,
            "is_long_range_extrapolation": (
                constraints.get("search_mode") == "ALL_WITHIN_DOMAIN" and nearest > 0.0
            ),
        }

    @staticmethod
    def _directions_at(
        idw: StableIDW,
        constraints: dict[str, Any],
        set_id: int,
        targets: np.ndarray,
        fit: GlobalJointSetFit,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate one axial mean normal at every generated position."""
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("Direction targets must have shape (N, 3)")
        rows = list(constraints["directions"].get(set_id, []))
        if not rows:
            return np.empty((len(targets), 3), dtype=np.float64), np.zeros(len(targets), dtype=bool)
        reference = dip_dir_dip_to_normal(
            next(item.mean_dip_direction for item in fit.sets if item.global_set_id == set_id),
            next(item.mean_dip for item in fit.sets if item.global_set_id == set_id),
        )
        normals = np.asarray([row[1] * (1 if np.dot(row[1], reference) >= 0 else -1) for row in rows])
        values, valid = idw.predict_many(np.asarray([row[0] for row in rows]), normals, targets)
        lengths = np.linalg.norm(values, axis=1)
        valid &= np.isfinite(values).all(axis=1) & (lengths > 1e-12)
        if np.any(valid):
            values[valid] /= lengths[valid, None]
            values[valid & (values[:, 2] < 0)] *= -1
        return values, valid

    @staticmethod
    def _sample_isotropic_axial(
        count: int,
        polar_rng: np.random.Generator,
        azimuth_rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample unoriented plane normals uniformly on the upper hemisphere."""
        if count < 0:
            raise ValueError("count must be non-negative")
        z = polar_rng.random(count)
        azimuth = azimuth_rng.uniform(0.0, 2.0 * np.pi, size=count)
        radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack((radial * np.sin(azimuth), radial * np.cos(azimuth), z))

    @staticmethod
    def _sample_fisher_varying(
        mean_normals: np.ndarray,
        kappa: float,
        polar_rng: np.random.Generator,
        azimuth_rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample Fisher normals around row-specific axial means with stable substreams."""
        means = np.asarray(mean_normals, dtype=np.float64).copy()
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError("mean_normals must have shape (N, 3)")
        if not np.isfinite(means).all() or not np.isfinite(kappa) or kappa <= 0:
            raise ValueError("Finite mean normals and positive kappa are required")
        lengths = np.linalg.norm(means, axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("Mean normals must be non-zero")
        means /= lengths[:, None]
        means[means[:, 2] < 0] *= -1
        count = len(means)
        uniform = polar_rng.random(count)
        exp_neg_2k = np.exp(-2.0 * kappa)
        cos_theta = 1.0 + np.log(uniform + (1.0 - uniform) * exp_neg_2k) / kappa
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
        phi = azimuth_rng.uniform(0.0, 2.0 * np.pi, size=count)

        reference = np.zeros_like(means)
        use_x = np.abs(means[:, 0]) < 0.9
        reference[use_x, 0] = 1.0
        reference[~use_x, 1] = 1.0
        tangent_1 = np.cross(means, reference)
        tangent_1 /= np.linalg.norm(tangent_1, axis=1)[:, None]
        tangent_2 = np.cross(means, tangent_1)
        samples = (
            cos_theta[:, None] * means
            + sin_theta[:, None]
            * (np.cos(phi)[:, None] * tangent_1 + np.sin(phi)[:, None] * tangent_2)
        )
        samples /= np.linalg.norm(samples, axis=1)[:, None]
        samples[samples[:, 2] < 0] *= -1
        return samples

    @staticmethod
    def _normals_to_orientations(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert upper-hemisphere plane normals to UI/IO degree conventions."""
        values = np.asarray(normals, dtype=np.float64).copy()
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("Normals must be a finite array with shape (N, 3)")
        lengths = np.linalg.norm(values, axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("Normals must be non-zero")
        values /= lengths[:, None]
        values[values[:, 2] < 0] *= -1
        dip = np.degrees(np.arccos(np.clip(values[:, 2], -1.0, 1.0)))
        dip_direction = np.degrees(np.arctan2(-values[:, 0], -values[:, 1])) % 360.0
        dip_direction[dip <= 1e-12] = 0.0
        return dip_direction.astype(np.float32), dip.astype(np.float32)

    def _positions_at_depths(self, hole_id: str, depths: np.ndarray) -> np.ndarray:
        hole = self.project.borehole_collection[hole_id]
        points, measured_depths = hole.compute_trajectory()
        return np.column_stack([np.interp(depths, measured_depths, points[:, axis]) for axis in range(3)])

    @staticmethod
    def _weighted_axial_kmeans(
        normals: np.ndarray, weights: np.ndarray, k: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed & ((1 << 64) - 1))
        first = int(rng.choice(len(normals), p=weights / weights.sum()))
        centroids = [normals[first].copy()]
        chosen = [first]
        for _cluster in range(1, k):
            distances = 1.0 - np.max(np.abs(normals @ np.asarray(centroids).T), axis=1)
            probabilities = distances * weights
            for index in chosen:
                probabilities[np.isclose(np.abs(normals @ normals[index]), 1.0, atol=1e-12)] = 0.0
            if probabilities.sum() <= 0:
                raise ValueError(INSUFFICIENT_ORIENTATIONS_MESSAGE)
            selected = int(rng.choice(len(normals), p=probabilities / probabilities.sum()))
            chosen.append(selected)
            centroids.append(normals[selected].copy())
        centroid_array = np.asarray(centroids)
        for _iteration in range(30):
            labels = JointSetService._assign_nonempty(normals, centroid_array)
            updated = np.empty_like(centroid_array)
            for cluster in range(k):
                values = normals[labels == cluster].copy()
                local_weights = weights[labels == cluster]
                reference = centroid_array[cluster]
                values[values @ reference < 0] *= -1
                mean = np.average(values, axis=0, weights=local_weights)
                updated[cluster] = mean / np.linalg.norm(mean)
            if np.all(1.0 - np.abs(np.sum(centroid_array * updated, axis=1)) <= 1e-6):
                centroid_array = updated
                break
            centroid_array = updated
        return JointSetService._assign_nonempty(normals, centroid_array), centroid_array

    @staticmethod
    def _weighted_fisher(normals: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
        reference = normals[0]
        aligned = normals.copy()
        aligned[aligned @ reference < 0] *= -1
        vector = np.sum(aligned * weights[:, None], axis=0)
        resultant = float(np.linalg.norm(vector))
        count = int(weights.sum())
        if resultant <= 1e-12:
            raise ValueError("Degenerate weighted orientation cluster")
        dd, dip = normal_to_dip_dir_dip(vector / resultant)
        if count == 1:
            return dd, dip, 999.0
        r_bar = min(resultant / count, 1.0 - 1e-12)
        if r_bar < 0.53:
            kappa = 2 * r_bar + r_bar**3 + 5 * r_bar**5 / 6
        elif r_bar < 0.85:
            kappa = -0.4 + 1.39 * r_bar + 0.43 / (1 - r_bar)
        else:
            kappa = 1 / max(r_bar**3 - 4 * r_bar**2 + 3 * r_bar, 1e-10)
        return dd, dip, max(0.1, min(float(kappa), 999.0))

    @staticmethod
    def _check_cancel(cancelled: Callable[[], bool] | None) -> None:
        if cancelled is not None and cancelled():
            raise BoreholeGenerationCancelled("Borehole fracture generation cancelled")
