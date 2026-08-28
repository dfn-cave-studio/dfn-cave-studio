"""Project orchestration for M10 explicit DFN generation and audit exports."""

from __future__ import annotations

import csv
import ctypes
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

from dfn_cave_studio.dfn.intensity import domain_at_depth
from dfn_cave_studio.dfn.m10_generator import ConditionedObservation, M10ExplicitDFNGenerator
from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal
from dfn_cave_studio.models.borehole import FractureObservation, OrientationCompleteness
from dfn_cave_studio.models.borehole_database import BoreholeDataType, RecordState
from dfn_cave_studio.models.m10 import DeterministicStructure, M10GenerationConfig, M10Realization
from dfn_cave_studio.models.m9 import ValidationState
from dfn_cave_studio.services.m7_state import get_domain_intervals, get_holdout


def _generate_realization_process(
    generator: M10ExplicitDFNGenerator, realization_index: int
) -> M10Realization:
    """Top-level spawn-safe process entry point."""
    return generator.generate(realization_index)


class M10Service:
    """Generate and export M10 state without placing scientific logic in Qt."""

    def __init__(self, project: Any) -> None:
        self.project = project

    def validate_readiness(self, config: M10GenerationConfig | None = None) -> list[str]:
        """Return blocking input errors; an empty list means generation is allowed."""
        config = config or self.project.m10_state.config
        errors: list[str] = []
        state = self.project.m9_state
        if state.parameter_field_metadata is None or not state.parameter_field_arrays:
            errors.append("M9 parameter field is not available")
        if not state.size_models:
            errors.append("M9 fracture size models are not available")
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

    def estimate(self, config: M10GenerationConfig | None = None) -> tuple[float, int]:
        """Estimate fracture count and storage before allocation."""
        generator = self._generator(config or self.project.m10_state.config, realization_index=0)
        return generator.estimate()

    def estimate_details(self, config: M10GenerationConfig | None = None) -> dict[str, Any]:
        """Return count, storage, generation, render, and save memory estimates."""
        generator = self._generator(config or self.project.m10_state.config, realization_index=0)
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
        if any(not item.complete for item in realizations):
            raise ValueError("Cannot commit an incomplete M10 realization")
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
