"""Array-based adapter from one saved Phase 2A realization to M9 density inputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dfn_cave_studio.dfn.intensity import _point_at_depth, _trajectory_segments, domain_at_depth
from dfn_cave_studio.geometry.coordinate import normal_to_dip_dir_dip
from dfn_cave_studio.models.borehole_fracture_realization import BoreholeFractureComponent
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.m9 import DomainOrientationModel, P10Interval
from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService
from dfn_cave_studio.services.m7_state import get_domain_intervals


@dataclass(frozen=True)
class Phase2ADensityInputs:
    """Complete non-owning scientific inputs derived from one realization."""

    intervals: list[P10Interval]
    orientation_models: list[DomainOrientationModel]
    domain_sets: dict[tuple[int | None, int], JointSetConfig]
    provenance: dict[str, Any]


class Phase2AM9Adapter:
    """Build M9 P10 rows without materialising per-fracture Python objects."""

    DEFAULT_XYZ_VALIDATION_CHUNK_SIZE = 65_536

    def __init__(self, project: Any, *, xyz_validation_chunk_size: int = DEFAULT_XYZ_VALIDATION_CHUNK_SIZE) -> None:
        if xyz_validation_chunk_size <= 0:
            raise ValueError("xyz_validation_chunk_size must be positive")
        self.project = project
        self.xyz_validation_chunk_size = int(xyz_validation_chunk_size)

    def build(
        self,
        realization_id: str,
        roles: dict[str, str],
        *,
        interval_mode: str,
        interval_length: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> Phase2ADensityInputs:
        realization, arrays = self._resolve(realization_id)
        self._validate_source(realization, arrays)
        diagnostics = sorted(realization.interval_diagnostics, key=lambda item: item.interval_index)
        if [item.interval_index for item in diagnostics] != list(range(len(diagnostics))):
            raise ValueError("Phase 2A interval diagnostics are not contiguous and cannot be traced safely")
        holes = {hole.borehole_id: hole for hole in self.project.borehole_collection.boreholes}
        domains = list(get_domain_intervals(self.project))
        set_ids = sorted(item.set_id for item in self.project.joint_sets)
        if not set_ids:
            raise ValueError("Confirmed global joint sets are required before using a Phase 2A realization in M9")
        unknown = set(np.unique(arrays["global_set_id"][arrays["global_set_id"] > 0]).tolist()) - set(set_ids)
        if unknown:
            raise ValueError(f"Phase 2A realization references unconfirmed global set IDs: {sorted(unknown)}")
        self._validate_interval_coverage(diagnostics)
        intervals: list[P10Interval] = []
        set_states: dict[int, set[str]] = {set_id: set() for set_id in set_ids}
        random_excluded = 0
        for diagnostic in diagnostics:
            self._check_cancel(cancelled)
            hole = holes.get(diagnostic.hole_id)
            if hole is None:
                raise ValueError(f"Phase 2A source interval references unknown borehole {diagnostic.hole_id}")
            row_mask = arrays["interval_index"] == diagnostic.interval_index
            random_excluded += int(
                np.count_nonzero(row_mask & (arrays["component_type"] == int(BoreholeFractureComponent.RANDOM_BACKGROUND)))
            )
            for start, end, domain_id, split_provenance in self._split_source_interval(
                diagnostic.hole_id,
                diagnostic.from_depth,
                diagnostic.to_depth,
                domains,
                interval_mode,
                interval_length,
            ):
                segment_directions = _trajectory_segments(hole, start, end)
                sample_length = sum(item[3] for item in segment_directions)
                center = _point_at_depth(hole, (start + end) / 2.0)
                depth_mask = row_mask & (arrays["measured_depth"] >= start) & (arrays["measured_depth"] < end)
                for set_id in set_ids:
                    supported = diagnostic.status == "GENERATED" and diagnostic.global_set_probabilities.get(set_id, 0.0) > 0.0
                    selected = depth_mask & (arrays["component_type"] == int(BoreholeFractureComponent.DOMINANT_SET)) & (
                        arrays["global_set_id"] == set_id
                    )
                    count = int(np.count_nonzero(selected))
                    if supported:
                        state = "true_zero" if count == 0 and sample_length > 0 else "modeled_value"
                        p10 = count / sample_length if sample_length > 0 else None
                        set_states[set_id].add("OBSERVATION_SUPPORTED")
                    else:
                        state = "no_data"
                        p10 = None
                        set_states[set_id].add("NO_P_INTENSITY_SUPPORT")
                    role = roles.get(hole.borehole_id, "calibration")
                    intervals.append(
                        P10Interval(
                            hole_id=hole.borehole_id,
                            from_depth=start,
                            to_depth=end,
                            domain_id=domain_id,
                            set_id=set_id,
                            observation_count=count,
                            sample_length=sample_length,
                            p10=p10,
                            role=role,
                            data_state=state,
                            center_x=float(center[0]),
                            center_y=float(center[1]),
                            center_z=float(center[2]),
                            segment_directions=segment_directions,
                            source="phase2a_borehole_realization",
                            provenance={
                                **split_provenance,
                                "realization_id": realization.realization_id,
                                "source_interval_index": diagnostic.interval_index,
                                "source_observation_id": diagnostic.observation_id,
                                "support_probability": diagnostic.global_set_probabilities.get(set_id),
                                "validation_interpretation": (
                                    "internal_consistency_not_independent" if role == "validation" else "calibration_fit"
                                ),
                            },
                            full_orientation_count=count,
                        )
                    )
        orientation_models, domain_sets = self._orientation_inputs(
            arrays, intervals, roles, realization.interval_diagnostics
        )
        provenance = {
            "source_mode": "phase2a_realization",
            "realization_id": realization.realization_id,
            "generator_version": realization.generator_version,
            "master_seed": realization.master_seed,
            "derived_seed": realization.derived_seed,
            "input_hash": realization.input_hash,
            "excluded_random_background_count": random_excluded,
            "set_states": {str(key): sorted(value) for key, value in set_states.items()},
            "p10_semantics": "counts from one saved stochastic realization divided by real trajectory length",
            "p32_semantics": "direction-corrected internal realization estimate; not observed P32 ground truth",
            "validation_semantics": "holdout holes are excluded from fitting; self-generated rows are internal consistency only",
        }
        return Phase2ADensityInputs(intervals, orientation_models, domain_sets, provenance)

    def _resolve(self, realization_id: str):
        state = self.project.borehole_fracture_state
        realization = next((item for item in state.realizations if item.realization_id == realization_id), None)
        if realization is None:
            raise ValueError(f"Unknown Phase 2A realization: {realization_id}")
        if not realization.complete:
            raise ValueError("Incomplete Phase 2A realizations cannot be used by M9")
        arrays = realization.arrays
        if not arrays:
            if not state.archive_source:
                raise ValueError("The selected Phase 2A realization arrays are not loaded and have no archive source")
            from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore

            arrays = ZipProjectStore().read_borehole_fracture_arrays(
                self.project, state.archive_source, realization.realization_id
            )
        BoreholeFractureService.validate_arrays(realization.model_copy(update={"arrays": arrays}))
        return realization, arrays

    def _validate_source(self, realization, arrays: dict[str, np.ndarray]) -> None:
        current_hash = BoreholeFractureService(self.project).authoritative_confirmed_fit().input_hash
        if current_hash != realization.input_hash:
            raise ValueError(
                "The selected Phase 2A realization is stale: its input hash does not match the current observations"
            )
        if np.any(~np.isfinite(arrays["measured_depth"])) or np.any(~np.isfinite(arrays["xyz_offset"])):
            raise ValueError("Phase 2A realization contains non-finite measured depth or XYZ values")
        if np.any(~np.isfinite(arrays["dip"])) or np.any(~np.isfinite(arrays["dip_direction"])):
            raise ValueError("Phase 2A realization contains non-finite generated orientations")
        origin = np.asarray(realization.provenance.get("xyz_origin", ()), dtype=np.float64)
        if origin.shape != (3,) or np.any(~np.isfinite(origin)):
            raise ValueError("Phase 2A realization is missing its finite float64 XYZ origin")
        interval_by_index = {item.interval_index: item for item in realization.interval_diagnostics}
        holes = {hole.borehole_id: hole for hole in self.project.borehole_collection.boreholes}
        trajectory_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for start in range(0, realization.fracture_count, self.xyz_validation_chunk_size):
            stop = min(start + self.xyz_validation_chunk_size, realization.fracture_count)
            interval_indices = arrays["interval_index"][start:stop]
            measured_depths = arrays["measured_depth"][start:stop]
            stored_offsets = arrays["xyz_offset"][start:stop]
            expected = np.empty((stop - start, 3), dtype=np.float64)
            for interval_index in np.unique(interval_indices):
                diagnostic = interval_by_index.get(int(interval_index))
                if diagnostic is None or diagnostic.hole_id not in holes:
                    raise ValueError("Phase 2A realization contains an invalid interval reference")
                local = interval_indices == interval_index
                if diagnostic.hole_id not in trajectory_cache:
                    # Generation uses BoreholeFractureService._positions_at_depths(),
                    # which calls compute_trajectory() with its 1 m default.
                    trajectory_cache[diagnostic.hole_id] = holes[diagnostic.hole_id].compute_trajectory()
                points, mds = trajectory_cache[diagnostic.hole_id]
                depths = measured_depths[local]
                expected[local] = np.column_stack(
                    [np.interp(depths, mds, points[:, axis]) for axis in range(3)]
                )
            expected_relative = expected - origin
            expected_encoding = expected_relative.astype(np.float32)
            if np.any(~np.isfinite(expected_encoding)):
                raise ValueError(
                    "Current measured-depth survey trajectories exceed the finite origin-relative float32 "
                    "XYZ storage range used by the selected Phase 2A realization"
                )
            stored_relative = np.asarray(stored_offsets, dtype=np.float64)
            component_error = np.abs(stored_relative - expected_relative)
            roundoff_limits = self._float32_roundoff_limits_m(expected_encoding)
            mismatch = np.any(component_error > np.nextafter(roundoff_limits, np.inf), axis=1)
            if np.any(mismatch):
                local_row = int(np.flatnonzero(mismatch)[0])
                global_row = start + local_row
                interval_index = int(interval_indices[local_row])
                diagnostic = interval_by_index[interval_index]
                stored_row = stored_relative[local_row]
                delta = stored_row - expected_relative[local_row]
                roundoff_ceiling = float(np.max(roundoff_limits[local_row], initial=0.0))
                stored_xyz = origin + stored_row
                raise ValueError(
                    "Phase 2A XYZ does not match the current measured-depth survey trajectory: "
                    f"array row {global_row}, hole {diagnostic.hole_id}, interval {interval_index}, "
                    f"MD {float(measured_depths[local_row]):.9g} m; maximum component mismatch "
                    f"{float(np.max(np.abs(delta))):.9g} m exceeds the origin-relative float32 encoding "
                    f"contract (rounding ceiling {roundoff_ceiling:.9g} m); stored XYZ "
                    f"{tuple(float(value) for value in stored_xyz)}, recomputed XYZ "
                    f"{tuple(float(value) for value in expected[local_row])}"
                )

    @staticmethod
    def _float32_roundoff_limits_m(encoded_offset: np.ndarray) -> np.ndarray:
        """Return per-component IEEE-754 round-to-nearest limits, in metres.

        Phase 2A stores float32 offsets from a float64 project origin.  Half of
        the wider adjacent float32 representable gap is therefore a conservative
        upper bound on storage roundoff and is independent of absolute E/N/R.
        """
        values = np.asarray(encoded_offset, dtype=np.float32)
        lower = np.nextafter(values, np.float32(-np.inf)).astype(np.float64)
        upper = np.nextafter(values, np.float32(np.inf)).astype(np.float64)
        values64 = values.astype(np.float64)
        half_ulp = 0.5 * np.maximum(np.abs(values64 - lower), np.abs(upper - values64))
        return half_ulp

    @staticmethod
    def _validate_interval_coverage(diagnostics: list[Any]) -> None:
        by_hole: dict[str, list[Any]] = defaultdict(list)
        for item in diagnostics:
            by_hole[item.hole_id].append(item)
        for hole_id, rows in by_hole.items():
            ordered = sorted(rows, key=lambda item: (item.from_depth, item.to_depth))
            if any(right.from_depth < left.to_depth - 1e-10 for left, right in zip(ordered[:-1], ordered[1:])):
                raise ValueError(f"Overlapping Phase 2A source intervals would double count sampling length in {hole_id}")

    @staticmethod
    def _split_source_interval(hole_id, start, end, domains, mode, interval_length):
        cuts = {float(start), float(end)}
        if mode == "fixed":
            first = int(np.floor(start / interval_length)) + 1
            cuts.update(
                value for value in np.arange(first * interval_length, end, interval_length) if start < value < end
            )
        cuts.update(
            float(value)
            for item in domains
            if item.hole_id == hole_id
            for value in (item.from_depth, item.to_depth)
            if start < value < end
        )
        ordered = sorted(cuts)
        split = len(ordered) > 2
        return [
            (
                left,
                right,
                domain_at_depth(hole_id, (left + right) / 2.0, domains),
                {
                    "interval_mode": mode,
                    "split_by_domain_or_m9_boundary": split,
                    "parent_from_depth": float(start),
                    "parent_to_depth": float(end),
                },
            )
            for left, right in zip(ordered[:-1], ordered[1:])
        ]

    def _orientation_inputs(self, arrays, intervals, roles, diagnostics):
        domains = list(get_domain_intervals(self.project))
        interval_holes = {item.interval_index: item.hole_id for item in diagnostics}
        statistics: dict[tuple[int | None, int], tuple[np.ndarray, int]] = {}
        dominant = arrays["component_type"] == int(BoreholeFractureComponent.DOMINANT_SET)
        for start in range(0, len(arrays["measured_depth"]), 65_536):
            stop = min(start + 65_536, len(arrays["measured_depth"]))
            chunk_rows = np.flatnonzero(dominant[start:stop]) + start
            if not len(chunk_rows):
                continue
            chunk_domains = np.full(len(chunk_rows), -1, dtype=np.int64)
            keep = np.zeros(len(chunk_rows), dtype=bool)
            interval_values = arrays["interval_index"][chunk_rows]
            for interval_index in np.unique(interval_values):
                local = interval_values == interval_index
                hole_id = interval_holes[int(interval_index)]
                if roles.get(hole_id) != "calibration":
                    continue
                keep[local] = True
                depths = arrays["measured_depth"][chunk_rows[local]]
                for domain in (item for item in domains if item.hole_id == hole_id):
                    in_domain = (depths >= domain.from_depth) & (depths < domain.to_depth)
                    chunk_domains[np.flatnonzero(local)[in_domain]] = domain.domain_id
            chunk_rows = chunk_rows[keep]
            chunk_domains = chunk_domains[keep]
            if not len(chunk_rows):
                continue
            dd = np.radians(np.asarray(arrays["dip_direction"][chunk_rows], dtype=np.float64))
            dip = np.radians(np.asarray(arrays["dip"][chunk_rows], dtype=np.float64))
            sin_dip = np.sin(dip)
            normals = np.column_stack((-np.sin(dd) * sin_dip, -np.cos(dd) * sin_dip, np.cos(dip)))
            set_values = np.asarray(arrays["global_set_id"][chunk_rows], dtype=np.int64)
            for domain_id, set_id in set(zip(chunk_domains.tolist(), set_values.tolist())):
                selected = (chunk_domains == domain_id) & (set_values == set_id)
                vector_sum = normals[selected].sum(axis=0)
                previous_sum, previous_count = statistics.get(
                    (None if domain_id == -1 else domain_id, set_id),
                    (np.zeros(3, dtype=np.float64), 0),
                )
                statistics[(None if domain_id == -1 else domain_id, set_id)] = (
                    previous_sum + vector_sum,
                    previous_count + int(np.count_nonzero(selected)),
                )
        base_sets = {item.set_id: item for item in self.project.joint_sets}
        models: list[DomainOrientationModel] = []
        configs: dict[tuple[int | None, int], JointSetConfig] = {}
        supported_keys = {(item.domain_id, item.set_id) for item in intervals if item.role == "calibration" and item.p10 is not None}
        for key in sorted(supported_keys, key=str):
            domain_id, set_id = key
            resultant, count = statistics.get(key, (np.zeros(3, dtype=np.float64), 0))
            if count < 3:
                continue
            length = float(np.linalg.norm(resultant))
            if length <= 1e-12:
                continue
            mean = resultant / length
            dip_direction, dip = normal_to_dip_dir_dip(mean)
            kappa = (count - 1) / max(count - length, 1e-10) if count >= 16 else (
                (count - 2) / max(count - length, 1e-10) * count / max(count - 1, 1)
            )
            kappa = max(0.1, min(float(kappa), 999.0))
            model = DomainOrientationModel(
                domain_id=domain_id,
                set_id=set_id,
                mean_dip_direction=dip_direction,
                mean_dip=dip,
                kappa=kappa,
                observation_count=count,
                full_orientation_count=count,
                source="phase2a_realization_calibration_orientations",
            )
            models.append(model)
            base = base_sets[set_id]
            configs[key] = base.model_copy(
                update={"orientation": OrientationDistribution(mean_dip_direction=dip_direction, mean_dip=dip, kappa=kappa)}
            )
        return models, configs

    @staticmethod
    def _check_cancel(cancelled) -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Phase 2A to M9 adaptation cancelled")
