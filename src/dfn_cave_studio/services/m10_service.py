"""Project orchestration for M10 explicit DFN generation and audit exports."""

from __future__ import annotations

import csv
import ctypes
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from dfn_cave_studio.dfn.intensity import domain_at_depth
from dfn_cave_studio.dfn.m10_generator import (
    SOURCE_CODES,
    ConditionedObservation,
    M10ExplicitDFNGenerator,
)
from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal
from dfn_cave_studio.models.borehole import FractureObservation, OrientationCompleteness
from dfn_cave_studio.models.borehole_database import BoreholeDataType, RecordState
from dfn_cave_studio.models.m9 import ValidationState
from dfn_cave_studio.models.m10 import DeterministicStructure, M10GenerationConfig, M10Realization
from dfn_cave_studio.services.m7_state import get_domain_intervals, get_holdout
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES
from dfn_cave_studio.models.spatial_grid import VoxelCellState


def _generate_realization_process(
    generator: M10ExplicitDFNGenerator, realization_index: int
) -> M10Realization:
    """Top-level spawn-safe process entry point."""
    return generator.generate(realization_index)


class M10Service:
    """Generate and export M10 state without placing scientific logic in Qt."""

    def __init__(self, project: Any) -> None:
        self.project = project

    @staticmethod
    def validated_config(config: M10GenerationConfig) -> M10GenerationConfig:
        """Re-parse a complete M10 config so unchecked model copies cannot be committed."""
        return M10GenerationConfig.model_validate(config.model_dump(mode="python"))

    def validate_readiness(self, config: M10GenerationConfig | None = None) -> list[str]:
        """Return blocking input errors; an empty list means generation is allowed."""
        config = self.validated_config(config or self.project.m10_state.config)
        errors: list[str] = []
        state = self.project.m9_state
        if state.parameter_field_metadata is None or not state.parameter_field_arrays:
            errors.append("M9 parameter field is not available")
        if not state.size_models:
            errors.append("M9 fracture size models are not available")
        elif state.parameter_field_metadata is not None and state.parameter_field_arrays:
            missing_size = self._missing_active_size_models()
            if missing_size:
                labels = ", ".join(
                    f"Domain {domain_id if domain_id is not None else 'None'} / Set {set_id}"
                    for domain_id, set_id in missing_size
                )
                errors.append(f"M9 fracture size model is missing for active targets: {labels}")
        if self.project.spatial_grid_config is None:
            errors.append("DFN Generation Domain is not configured")
        if any(model.source.value == "experimental" for model in state.size_models) and not config.experimental_size_models_confirmed:
            errors.append("EXPERIMENTAL size models require explicit confirmation")
        invalid = [model for model in state.size_models if model.converged is False or model.fit_status in {"failed", "invalid"}]
        if invalid:
            errors.append("One or more size models failed or contain invalid parameters")
        if (
            state.validation_summary.state != ValidationState.COMPLETE
            and not config.validation_warning_acknowledged
        ):
            errors.append("Validation is incomplete or insufficient; acknowledge the warning to continue")
        holdout = get_holdout(self.project)
        if config.condition_calibration_observations and (holdout is None or not holdout.is_locked):
            errors.append("Validation Holdout must be locked before conditioning")
        return errors

    def _missing_active_size_models(self) -> list[tuple[int | None, int]]:
        """Return positive-P32 Domain/Set targets without a usable size model."""
        state = self.project.m9_state
        metadata = state.parameter_field_metadata
        if metadata is None:
            return []
        arrays = state.parameter_field_arrays
        modeled = arrays["cell_state"] == CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
        domains = arrays["domain_id"]
        available = {(item.domain_id, item.set_id) for item in state.size_models}
        missing: list[tuple[int | None, int]] = []
        for set_id in metadata.set_ids:
            values = arrays[f"set_{set_id}_p32"]
            active = modeled & np.isfinite(values) & (values > 0.0)
            for raw_domain_id in np.unique(domains[active]):
                domain_id = None if int(raw_domain_id) < 0 else int(raw_domain_id)
                if (domain_id, int(set_id)) not in available and (None, int(set_id)) not in available:
                    missing.append((domain_id, int(set_id)))
        return missing

    def joint_set_diagnostics(
        self,
        config: M10GenerationConfig | None = None,
        realization: M10Realization | None = None,
        *,
        estimate_details: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Summarize every parameter-field Domain/Set, including zero-output groups."""
        state = self.project.m9_state
        metadata = state.parameter_field_metadata
        if metadata is None or not state.parameter_field_arrays:
            return []
        arrays = state.parameter_field_arrays
        modeled = arrays["cell_state"] == CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
        true_zero = arrays["cell_state"] == CELL_STATE_CODES[VoxelCellState.TRUE_ZERO]
        domains = arrays["domain_id"]
        details = estimate_details
        if details is None:
            try:
                details = self.estimate_details(config)
            except (KeyError, RuntimeError, TypeError, ValueError):
                details = {"targets": []}
        estimates = {
            (item.get("domain_id"), int(item["set_id"])): item
            for item in details.get("targets", [])
        }
        observation_counts: dict[tuple[int | None, int], dict[str, int]] = {}
        holdout = get_holdout(self.project)
        domain_intervals = get_domain_intervals(self.project)
        for hole in self.project.borehole_collection.boreholes:
            role = "validation" if holdout is not None and holdout.is_validation(hole.borehole_id) else "calibration"
            for observation in hole.fracture_observations:
                if observation.set_id is None:
                    continue
                domain_id = domain_at_depth(hole.borehole_id, observation.measured_depth, domain_intervals)
                key = (domain_id, int(observation.set_id))
                counts = observation_counts.setdefault(
                    key,
                    {"calibration": 0, "validation": 0, "full_orientation": 0, "dip_only": 0},
                )
                counts[role] += 1
                counts["full_orientation" if observation.has_full_orientation else "dip_only"] += 1

        size_models = {(item.domain_id, item.set_id) for item in state.size_models}
        quality = {
            (item.domain_id, item.set_id): item
            for item in (realization.quality.domain_set if realization is not None else [])
        }
        actual_counts: dict[tuple[int | None, int], dict[str, int]] = {}
        if realization is not None:
            geometry = realization.geometry_arrays
            source_values = np.asarray(geometry.get("source_code", []), dtype=np.uint8)
            for raw_domain_id, set_id in zip(
                np.asarray(geometry.get("domain_id", []), dtype=np.int32),
                np.asarray(geometry.get("set_id", []), dtype=np.int16),
                strict=True,
            ):
                domain_id = None if int(raw_domain_id) < 0 else int(raw_domain_id)
                counts = actual_counts.setdefault(
                    (domain_id, int(set_id)),
                    {"actual": 0, "random": 0, "conditioned": 0, "deterministic": 0},
                )
                counts["actual"] += 1
            for source_name, source_code in SOURCE_CODES.items():
                mask = source_values == source_code
                for raw_domain_id, set_id in zip(
                    np.asarray(geometry.get("domain_id", []), dtype=np.int32)[mask],
                    np.asarray(geometry.get("set_id", []), dtype=np.int16)[mask],
                    strict=True,
                ):
                    domain_id = None if int(raw_domain_id) < 0 else int(raw_domain_id)
                    label = {
                        "STOCHASTIC": "random",
                        "CONDITIONED_OBSERVATION": "conditioned",
                        "DETERMINISTIC_STRUCTURE": "deterministic",
                    }[source_name]
                    actual_counts[(domain_id, int(set_id))][label] += 1

        domain_values = sorted(
            {None if int(value) < 0 else int(value) for value in np.unique(domains[modeled | true_zero])},
            key=lambda value: (-1 if value is None else value),
        )
        rows: list[dict[str, Any]] = []
        for domain_id in domain_values:
            domain_mask = domains == (-1 if domain_id is None else domain_id)
            for set_id in metadata.set_ids:
                values = arrays[f"set_{set_id}_p32"]
                finite = domain_mask & (modeled | true_zero) & np.isfinite(values)
                positive = finite & (values > 0.0)
                target = float(np.mean(values[finite], dtype=np.float64)) if np.any(finite) else float("nan")
                direction_valid = positive
                for field in ("dip_direction", "dip", "kappa"):
                    direction_valid = direction_valid & np.isfinite(arrays[f"set_{set_id}_{field}"])
                direction_valid = direction_valid & (arrays[f"set_{set_id}_kappa"] > 0.0)
                has_size = (domain_id, int(set_id)) in size_models or (None, int(set_id)) in size_models
                estimate = estimates.get((domain_id, int(set_id)), {})
                generated = quality.get((domain_id, int(set_id)))
                counts = observation_counts.get(
                    (domain_id, int(set_id)),
                    {"calibration": 0, "validation": 0, "full_orientation": 0, "dip_only": 0},
                )
                actual = actual_counts.get(
                    (domain_id, int(set_id)),
                    {"actual": 0, "random": 0, "conditioned": 0, "deterministic": 0},
                )
                reason = ""
                if not np.any(finite):
                    reason = "NO_DATA"
                elif not np.any(positive):
                    reason = "TARGET_P32_ZERO"
                elif not has_size:
                    reason = "MISSING_SIZE_MODEL"
                elif not np.all(direction_valid[positive]):
                    reason = "INSUFFICIENT_ORIENTATION_DATA"
                if generated is not None and generated.status != "generated":
                    reason = generated.status
                rows.append(
                    {
                        "domain_id": domain_id,
                        "set_id": int(set_id),
                        "observations": counts["calibration"] + counts["validation"],
                        "calibration_observations": counts["calibration"],
                        "validation_observations": counts["validation"],
                        "full_orientation": counts["full_orientation"],
                        "dip_only": counts["dip_only"],
                        "orientation_status": (
                            "valid"
                            if np.any(positive) and np.all(direction_valid[positive])
                            else "invalid" if np.any(positive) else "not_applicable"
                        ),
                        "effective_voxels": int(np.count_nonzero(positive)),
                        "target_p32": target,
                        "expected": float(estimate.get("expected_fractures", 0.0)),
                        **actual,
                        "p32_unresolved_orientation": (
                            float(generated.p32_unresolved_orientation) if generated is not None else 0.0
                        ),
                        "unresolved_reason": reason,
                    }
                )
        return rows

    def estimate(self, config: M10GenerationConfig | None = None) -> tuple[float, int]:
        """Estimate fracture count and storage before allocation."""
        generator = self._generator(
            self.validated_config(config or self.project.m10_state.config), realization_index=0
        )
        return generator.estimate()

    def estimate_details(self, config: M10GenerationConfig | None = None) -> dict[str, Any]:
        """Return count, storage, generation, render, and save memory estimates."""
        generator = self._generator(
            self.validated_config(config or self.project.m10_state.config), realization_index=0
        )
        details = generator.estimate_details()
        details["available_memory_bytes"] = self.available_memory_bytes()
        return details

    @staticmethod
    def available_memory_bytes() -> int:
        """Return currently available physical memory without an optional dependency."""
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        pages = getattr(os, "sysconf", lambda _: 0)("SC_AVPHYS_PAGES")
        page_size = getattr(os, "sysconf", lambda _: 0)("SC_PAGE_SIZE")
        return int(pages * page_size)

    def generate_batch(
        self,
        config: M10GenerationConfig,
        *,
        progress=None,
        cancelled=None,
        replace: bool = True,
        commit: bool = True,
        deterministic_structures: list[DeterministicStructure] | None = None,
    ) -> list[M10Realization]:
        """Generate a complete small batch and commit only after every run succeeds."""
        config = self.validated_config(config)
        errors = self.validate_readiness(config)
        if errors:
            raise RuntimeError("; ".join(errors))
        details = self.estimate_details(config)
        available = int(details["available_memory_bytes"])
        required = int(details["all_realizations_peak_bytes"])
        if available > 0 and required > available * config.memory_safety_fraction:
            raise MemoryError(
                f"Estimated peak memory {required / 1024**3:.2f} GiB exceeds "
                f"{config.memory_safety_fraction:.0%} of available memory ({available / 1024**3:.2f} GiB); "
                "reduce the realization count or fracture density"
            )
        state = self.project.m9_state
        completed: list[M10Realization] = []
        worker_count = min(config.worker_count, config.realization_count)
        if worker_count == 1:
            for index in range(config.realization_count):
                if cancelled and cancelled():
                    raise InterruptedError("M10 explicit DFN generation cancelled")
                generator = self._generator(
                    config,
                    realization_index=index,
                    cancelled=cancelled,
                    deterministic_structures=deterministic_structures,
                    progress=(
                        (lambda current, total, message, batch_index=index: progress(
                            batch_index * total + current, config.realization_count * total, message
                        ))
                        if progress
                        else None
                    ),
                    actual_worker_count=1,
                )
                completed.append(generator.generate(index))
        else:
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(processes=worker_count)
            jobs = []
            try:
                for index in range(config.realization_count):
                    generator = self._generator(
                        config,
                        realization_index=index,
                        cancelled=None,
                        progress=None,
                        deterministic_structures=deterministic_structures,
                        actual_worker_count=worker_count,
                    )
                    jobs.append(pool.apply_async(_generate_realization_process, (generator, index)))
                while not all(job.ready() for job in jobs):
                    if cancelled and cancelled():
                        pool.terminate()
                        pool.join()
                        raise InterruptedError("M10 explicit DFN generation cancelled")
                    if progress:
                        finished = sum(job.ready() for job in jobs)
                        progress(finished, len(jobs), f"Generated {finished}/{len(jobs)} realizations")
                    time.sleep(0.05)
                completed = [job.get() for job in jobs]
                pool.close()
                pool.join()
            except BaseException:
                pool.terminate()
                pool.join()
                raise
            completed.sort(key=lambda item: item.realization_index)
        for item in completed:
            if state.validation_summary.state != ValidationState.COMPLETE:
                item.quality.warnings.append(
                    f"Generation continued after explicit acknowledgement of Validation state: "
                    f"{state.validation_summary.state.value}"
                )
        if cancelled and cancelled():
            raise InterruptedError("M10 explicit DFN generation cancelled")
        if commit:
            self.commit_realizations(config, completed, replace=replace)
        return completed

    def commit_realizations(
        self, config: M10GenerationConfig, realizations: list[M10Realization], *, replace: bool = True
    ) -> None:
        """Commit an already complete batch as one project mutation."""
        config = self.validated_config(config)
        if any(not item.complete for item in realizations):
            raise ValueError("Cannot commit an incomplete M10 realization")
        self.project.m11_state.results = []
        self.project.m11_state.provenance["invalidated_by"] = "M10 realization regeneration"
        self.project.m10_state.config = config
        if replace:
            self.project.m10_state.realizations = list(realizations)
        else:
            existing_ids = {item.realization_id for item in self.project.m10_state.realizations}
            self.project.m10_state.realizations.extend(item for item in realizations if item.realization_id not in existing_ids)
        self.project.m10_state.last_input_hash = realizations[0].config_hash if realizations else None
        self.project.m10_state.provenance.update(
            {
                "source": "M9 first voxel parameter field",
                "second_voxelization_completed": False,
                "validation_used_for_generation": False,
            }
        )

    def commit_config(self, config: M10GenerationConfig) -> M10GenerationConfig:
        """Validate and atomically replace only the persisted M10 configuration."""
        validated = self.validated_config(config)
        self.project.m10_state.config = validated
        return validated

    def import_deterministic_csv(self, path: Path, *, commit: bool = True) -> list[DeterministicStructure]:
        """Read parameterized deterministic discs from CSV transactionally."""
        path = Path(path)
        required = {
            "structure_id", "center_x", "center_y", "center_z", "dip_direction", "dip", "radius", "structure_type", "domain_id"
        }
        imported: list[DeterministicStructure] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Deterministic structure CSV is missing: {', '.join(sorted(missing))}")
            for source_row, row in enumerate(reader, start=2):
                imported.append(
                    DeterministicStructure(
                        structure_id=row["structure_id"].strip(),
                        center_x=float(row["center_x"]),
                        center_y=float(row["center_y"]),
                        center_z=float(row["center_z"]),
                        dip_direction=float(row["dip_direction"]),
                        dip=float(row["dip"]),
                        radius=float(row["radius"]),
                        structure_type=row["structure_type"].strip(),
                        domain_id=int(row["domain_id"]) if row["domain_id"].strip() else None,
                        set_id=int(row["set_id"]) if row.get("set_id", "").strip() else None,
                        source_file=str(path),
                        source_row=source_row,
                        provenance={"format": "parameterized_csv_disc", "complex_mesh_supported": False},
                    )
                )
        if len({item.structure_id for item in imported}) != len(imported):
            raise ValueError("Deterministic structure_id values must be unique")
        if commit:
            self.project.m10_state.deterministic_structures = imported
        return imported

    def deterministic_structure_relations(
        self, structures: list[DeterministicStructure]
    ) -> list[dict[str, Any]]:
        """Classify imported deterministic discs against the generation domain without mutating them."""
        if self.project.spatial_grid_config is None:
            return []
        bounds = self.project.spatial_grid_config.generation_domain
        result: list[dict[str, Any]] = []
        for structure in structures:
            center = (structure.center_x, structure.center_y, structure.center_z)
            relation = M10ExplicitDFNGenerator._disk_bounds_relation(
                center, dip_dir_dip_to_normal(structure.dip_direction, structure.dip), structure.radius, bounds
            )
            result.append(
                {
                    "structure_id": structure.structure_id,
                    "set_id": structure.set_id,
                    "center": center,
                    "relation": relation,
                    "generation_domain": bounds.model_dump(),
                }
            )
        return result

    def export_realization(self, realization_id: str, directory: Path) -> list[Path]:
        """Export CSV, JSON, NPZ, VTP, and separated source tables."""
        realization = next(
            (item for item in self.project.m10_state.realizations if item.realization_id == realization_id), None
        )
        if realization is None:
            raise KeyError(f"Unknown M10 realization: {realization_id}")
        from dfn_cave_studio.simulation_export.m10_exporter import M10Exporter

        return M10Exporter().export_all(realization, Path(directory))

    def _generator(
        self,
        config: M10GenerationConfig,
        realization_index: int,
        cancelled=None,
        progress=None,
        deterministic_structures: list[DeterministicStructure] | None = None,
        actual_worker_count: int = 1,
    ):
        del realization_index
        state = self.project.m9_state
        if state.parameter_field_metadata is None or self.project.spatial_grid_config is None:
            raise RuntimeError("M9 parameter field and DFN Generation Domain are required")
        dip_only = {
            (item.domain_id, item.set_id)
            for item in state.p32_estimates
            if item.dip_only_count > 0
        }
        size_models = []
        for model in state.size_models:
            provenance = dict(model.provenance)
            provenance["dip_only_density_evidence"] = (model.domain_id, model.set_id) in dip_only
            size_models.append(model.model_copy(update={"provenance": provenance}))
        return M10ExplicitDFNGenerator(
            metadata=state.parameter_field_metadata,
            arrays=state.parameter_field_arrays,
            generation_domain=self.project.spatial_grid_config.generation_domain,
            size_models=size_models,
            config=config,
            project_id=str(self.project.project_id),
            conditioned_observations=self._conditioned_observations(),
            deterministic_structures=(
                self.project.m10_state.deterministic_structures
                if deterministic_structures is None
                else deterministic_structures
            ),
            cancelled=cancelled,
            progress=progress,
            actual_worker_count=actual_worker_count,
        )

    def _conditioned_observations(self) -> list[ConditionedObservation]:
        holdout = get_holdout(self.project)
        if holdout is None or not holdout.is_locked:
            return []
        holes = {hole.borehole_id: hole for hole in self.project.borehole_collection.boreholes}
        domains = get_domain_intervals(self.project)
        result: list[ConditionedObservation] = []
        for record in self.project.borehole_database.query(BoreholeDataType.FRACTURES, RecordState.FORMAL):
            if holdout.is_validation(record.hole_id):
                continue
            values = record.values
            direction = values.get("dip_direction")
            set_id = values.get("set_id")
            hole = holes.get(record.hole_id)
            if direction is None or set_id in (None, "") or hole is None:
                continue
            observation = FractureObservation(
                borehole_id=record.hole_id,
                measured_depth=float(values["depth"]),
                dip_direction=float(direction),
                dip=float(values["dip"]),
                set_id=int(set_id),
            )
            if observation.orientation_completeness != OrientationCompleteness.FULL_ORIENTATION:
                continue
            position = hole.locate_observation(observation)
            if position is None:
                continue
            result.append(
                ConditionedObservation(
                    observation_record_id=record.record_id,
                    position=tuple(float(value) for value in position),
                    dip_direction=float(observation.dip_direction),
                    dip=observation.dip,
                    domain_id=domain_at_depth(record.hole_id, observation.measured_depth, domains),
                    set_id=int(observation.set_id),
                    hole_id=record.hole_id,
                )
            )
        return result
