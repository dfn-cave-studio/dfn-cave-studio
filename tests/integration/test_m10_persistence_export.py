"""M10 generation, persistence, migration, and generic-export integration tests."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pyvista as pv
import pytest

from dfn_cave_studio.dfn.m10_geometry import fracture_ids, source_mask
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.borehole import Borehole, BoreholeCollection, Collar
from dfn_cave_studio.models.borehole_database import BoreholeDataType, BoreholeRecord, RecordState
from dfn_cave_studio.models.m10 import DeterministicStructure, M10GenerationConfig, M10Realization
from dfn_cave_studio.models.m9 import (
    DensityMethod,
    ObservabilityState,
    P32Estimate,
    ParameterFieldMetadata,
    SizeModel,
    SizeModelSource,
    ValidationState,
)
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig, VoxelCellState
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.persistence.project_store import ProjectStore
from dfn_cave_studio.services.m10_service import M10Service
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES


def make_m10_project() -> Project:
    project = Project()
    analysis = ModelBounds(x_min=0, x_max=20, y_min=0, y_max=10, z_min=0, z_max=10)
    generation = ModelBounds(x_min=-2, x_max=22, y_min=-2, y_max=12, z_min=-2, z_max=12)
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=analysis, generation_domain=generation)
    project.m9_state.parameter_field_metadata = ParameterFieldMetadata(
        shape=(2, 1, 1),
        origin=(0, 0, 0),
        spacing=(10, 10, 10),
        field_names=[],
        set_ids=[1],
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=42,
        estimated_bytes=0,
    )
    shape = (2, 1, 1)
    project.m9_state.parameter_field_arrays = {
        "cell_state": np.full(shape, CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], dtype=np.uint8),
        "domain_id": np.ones(shape, dtype=np.int32),
        "set_1_p32": np.asarray([[[0.03]], [[0.09]]], dtype=np.float32),
        "set_1_dip_direction": np.full(shape, 45.0, dtype=np.float32),
        "set_1_dip": np.full(shape, 60.0, dtype=np.float32),
        "set_1_kappa": np.full(shape, 40.0, dtype=np.float32),
    }
    project.m9_state.size_models = [
        SizeModel(
            domain_id=1,
            set_id=1,
            distribution_type="fixed",
            parameters={"radius": 1.0},
            min_radius=1.0,
            max_radius=1.0,
            mean_radius=1.0,
            mean_squared_radius=1.0,
            source=SizeModelSource.ASSUMED,
        )
    ]
    project.m9_state.validation_summary.state = ValidationState.COMPLETE
    project.m10_state.deterministic_structures = [
        DeterministicStructure(
            structure_id="FAULT-01",
            center_x=10,
            center_y=5,
            center_z=5,
            dip_direction=90,
            dip=75,
            radius=3,
            structure_type="fault",
            domain_id=1,
        )
    ]
    return project


def test_m10_batch_save_reopen_arrays_and_seeds_are_identical(tmp_path: Path):
    project = make_m10_project()
    config = M10GenerationConfig(base_seed=100, realization_count=3, condition_calibration_observations=False)
    generated = M10Service(project).generate_batch(config)
    assert [item.seed for item in generated] == [100, 101, 102]
    assert all(item.complete for item in generated)
    all_ids = [fracture_id for item in generated for fracture_id in fracture_ids(item)]
    assert len(all_ids) == len(set(all_ids))
    path = tmp_path / "m10.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)
    assert restored.m10_state.config == project.m10_state.config
    assert len(restored.m10_state.realizations) == 3
    for expected, actual in zip(project.m10_state.realizations, restored.m10_state.realizations):
        assert actual.model_dump(exclude={"geometry_arrays"}) == expected.model_dump(exclude={"geometry_arrays"})
        assert actual.geometry_arrays.keys() == expected.geometry_arrays.keys()
        for name in expected.geometry_arrays:
            np.testing.assert_equal(actual.geometry_arrays[name], expected.geometry_arrays[name])
    second_path = tmp_path / "m10-second-save.dfnproj"
    ZipProjectStore().save(restored, second_path)
    second = ZipProjectStore().load(second_path)
    np.testing.assert_equal(
        second.m10_state.realizations[0].geometry_arrays["center"],
        project.m10_state.realizations[0].geometry_arrays["center"],
    )


def test_multiscale_settings_subgrid_and_size_class_round_trip(tmp_path: Path):
    project = make_m10_project()
    project.m9_state.parameter_field_arrays["set_1_dip_direction"][1, 0, 0] = np.nan
    project.m9_state.size_models = [
        model.model_copy(
            update={
                "distribution_type": "uniform", "parameters": {}, "min_radius": 0.5,
                "max_radius": 2.5, "mean_radius": 1.5, "mean_squared_radius": 3.25,
                "source": SizeModelSource.USER_DEFINED,
            }
        )
        for model in project.m9_state.size_models
    ]
    config = M10GenerationConfig(
        base_seed=77, condition_calibration_observations=False, size_threshold_mode="auto",
        small_area_share=0.1, medium_large_cumulative_share=0.7,
        enabled_size_classes=["MEDIUM", "LARGE"],
    )
    realization = M10Service(project).generate_batch(config)[0]
    assert realization.quality.p32_subgrid > 0.0
    assert realization.quality.p32_unresolved_orientation > 0.0
    path = tmp_path / "multiscale.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)
    assert restored.m10_state.config == config
    actual = restored.m10_state.realizations[0]
    assert actual.quality == realization.quality
    for name, expected in realization.geometry_arrays.items():
        np.testing.assert_array_equal(actual.geometry_arrays[name], expected)


def test_multiprocess_realizations_are_stably_sorted_and_reproducible():
    first_project = make_m10_project()
    second_project = make_m10_project()
    config = M10GenerationConfig(
        base_seed=123,
        realization_count=2,
        worker_count=2,
        condition_calibration_observations=False,
    )
    first = M10Service(first_project).generate_batch(config)
    second = M10Service(second_project).generate_batch(config)
    assert [item.seed for item in first] == [123, 124]
    assert [item.provenance["worker_count"] for item in first] == [2, 2]
    for expected, actual in zip(first, second):
        for name in expected.geometry_arrays:
            np.testing.assert_equal(actual.geometry_arrays[name], expected.geometry_arrays[name])


@pytest.mark.parametrize("worker_count", [2, 4, 8])
def test_worker_count_does_not_change_seeded_geometry(worker_count):
    serial_project = make_m10_project()
    parallel_project = make_m10_project()
    serial = M10Service(serial_project).generate_batch(
        M10GenerationConfig(
            base_seed=50, realization_count=worker_count, worker_count=1,
            condition_calibration_observations=False,
        )
    )
    parallel = M10Service(parallel_project).generate_batch(
        M10GenerationConfig(
            base_seed=50, realization_count=worker_count, worker_count=worker_count,
            condition_calibration_observations=False,
        )
    )
    for expected, actual in zip(serial, parallel):
        for name in expected.geometry_arrays:
            np.testing.assert_equal(actual.geometry_arrays[name], expected.geometry_arrays[name])


def test_m10_exports_are_readable_and_complete(tmp_path: Path):
    project = make_m10_project()
    realization = M10Service(project).generate_batch(
        M10GenerationConfig(base_seed=7, condition_calibration_observations=False)
    )[0]
    paths = M10Service(project).export_realization(realization.realization_id, tmp_path)
    assert len(paths) == 7
    with (tmp_path / "fractures.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == realization.fracture_count
    summary = json.loads((tmp_path / "realization_summary.json").read_text(encoding="utf-8"))
    assert summary["quality"]["local_p32_status"] == "CENTER_ASSIGNED_PRELIMINARY"
    with np.load(tmp_path / "fractures.npz", allow_pickle=False) as archive:
        for name in realization.geometry_arrays:
            np.testing.assert_equal(archive[name], realization.geometry_arrays[name])
    mesh = pv.read(tmp_path / "fractures.vtp")
    assert mesh.n_cells <= realization.fracture_count
    assert mesh.n_cells > 0


def test_project_m10_npz_is_compressed_once_and_outer_zip_entry_is_stored(tmp_path: Path):
    project = make_m10_project()
    M10Service(project).generate_batch(M10GenerationConfig(condition_calibration_observations=False))
    path = tmp_path / "single-compression.dfnproj"
    ZipProjectStore().save(project, path)
    with zipfile.ZipFile(path) as archive:
        entry = next(item for item in archive.infolist() if item.filename.startswith("results/m10/") and item.filename.endswith(".npz"))
        assert entry.compress_type == zipfile.ZIP_STORED


def test_v091_project_schema_migrates_to_empty_m10_state():
    old = Project().model_dump(mode="python")
    old["schema_version"] = 3
    old.pop("m10_state", None)
    migrated = Project.from_dict(old)
    assert migrated.schema_version == 4
    assert migrated.m10_state.realizations == []


def test_saved_unreleased_m10_geometry_is_migrated_on_real_project_open(tmp_path: Path):
    project = make_m10_project()
    project.m10_state.realizations = [
        M10Realization(
            realization_id="legacy", realization_index=0, seed=1, config_hash="hash",
            geometry_arrays={
                "center": np.asarray([[1.0, 2.0, 3.0]]), "normal": np.asarray([[0.0, 0.0, 1.0]]),
                "radius": np.asarray([1.0]), "original_area": np.asarray([math.pi]),
                "clipped_area": np.asarray([math.pi]), "domain_id": np.asarray([1]), "set_id": np.asarray([2]),
                "source": np.asarray(["STOCHASTIC"]), "voxel_index": np.asarray([[0, 0, 0]]),
                "vertices": np.zeros((1, 32, 3)), "vertex_count": np.asarray([32]),
                "distribution": np.asarray(["fixed"]), "size_source": np.asarray(["assumed"]),
                "orientation_source": np.asarray(["domain_set_fisher_model"]),
                "dip_only_density_evidence": np.asarray([False]), "observation_record_id": np.asarray([""]),
            },
        )
    ]
    path = tmp_path / "legacy-m10.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)
    realization = restored.m10_state.realizations[0]
    assert realization.provenance["migrated_from"] == "m10-object-array-v1"
    assert "source_code" in realization.geometry_arrays
    assert "vertices" not in realization.geometry_arrays


def test_cancelled_batch_leaves_existing_project_state_untouched():
    project = make_m10_project()
    service = M10Service(project)
    service.generate_batch(M10GenerationConfig(base_seed=1, condition_calibration_observations=False))
    before = project.m10_state.model_copy(deep=True)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 2

    try:
        service.generate_batch(
            M10GenerationConfig(base_seed=2, realization_count=5, condition_calibration_observations=False),
            cancelled=cancelled,
        )
    except InterruptedError:
        pass
    assert project.m10_state.model_dump(exclude={"realizations"}) == before.model_dump(exclude={"realizations"})
    assert project.m10_state.realizations[0].seed == before.realizations[0].seed


def test_conditioning_uses_only_calibration_full_orientation_records():
    project = make_m10_project()
    project.borehole_collection = BoreholeCollection(
        boreholes=[
            Borehole(
                borehole_id="CAL",
                collar=Collar(borehole_id="CAL", collar_x=5, collar_y=5, collar_z=10, final_depth=20),
            ),
            Borehole(
                borehole_id="VAL",
                collar=Collar(borehole_id="VAL", collar_x=15, collar_y=5, collar_z=10, final_depth=20),
            ),
        ]
    )
    rows = [
        ("cal-full", "CAL", 5.0, 120.0),
        ("cal-dip", "CAL", 6.0, None),
        ("val-full", "VAL", 5.0, 220.0),
    ]
    project.borehole_database.records = [
        BoreholeRecord(
            record_id=record_id,
            data_type=BoreholeDataType.FRACTURES,
            hole_id=hole_id,
            source_file="mixed.csv",
            source_row=index + 2,
            original_values={"hole_id": hole_id, "depth": depth, "dip_direction": direction, "dip": 60, "set_id": 1},
            values={"hole_id": hole_id, "depth": depth, "dip_direction": direction, "dip": 60, "set_id": 1},
            state=RecordState.FORMAL,
        )
        for index, (record_id, hole_id, depth, direction) in enumerate(rows)
    ]
    holdout = HoldoutService()
    holdout.select_manual(["CAL", "VAL"], ["VAL"])
    holdout.lock()
    set_holdout(project, holdout)
    conditioned = M10Service(project)._conditioned_observations()
    assert [item.observation_record_id for item in conditioned] == ["cal-full"]
    project.m9_state.size_models[0] = project.m9_state.size_models[0].model_copy(update={"domain_id": None})
    service = M10Service(project)
    deterministic = service.import_deterministic_csv(
        Path(__file__).parents[2] / "examples/m10_demo/deterministic_structures.csv",
        commit=False,
    )
    realization = service.generate_batch(
        M10GenerationConfig(
            base_seed=42, realization_count=1, condition_calibration_observations=True,
            retain_outside_deterministic=True,
        ),
        deterministic_structures=deterministic,
    )[0]
    assert realization.complete
    assert realization.quality.conditioned_count == 1
    assert realization.quality.deterministic_count == 1
    assert realization.geometry_arrays["normal"].shape == (realization.fracture_count, 3)


def test_imported_m10_deterministic_csv_and_seed_42_single_sample_generate_successfully():
    project = make_m10_project()
    project.m9_state.parameter_field_arrays["set_1_p32"][:] = 0.0
    project.m9_state.parameter_field_arrays["set_1_p32"][0, 0, 0] = math.pi / 1000.0
    path = Path(__file__).parents[2] / "examples/m10_demo/deterministic_structures.csv"
    service = M10Service(project)
    imported = service.import_deterministic_csv(path, commit=False)
    realization = service.generate_batch(
        M10GenerationConfig(
            base_seed=42, realization_count=1, condition_calibration_observations=False,
            retain_outside_deterministic=True,
        ),
        deterministic_structures=imported,
    )[0]
    assert realization.complete
    assert realization.quality.stochastic_count == 1
    assert realization.quality.deterministic_count == 1
    assert realization.geometry_arrays["normal"].shape == (2, 3)
    assert len(set(fracture_ids(realization))) == realization.fracture_count


def test_seed_42_single_sample_normals_are_identical_after_project_reopen(tmp_path: Path):
    project = make_m10_project()
    project.m10_state.deterministic_structures = []
    project.m9_state.parameter_field_arrays["set_1_p32"][:] = 0.0
    project.m9_state.parameter_field_arrays["set_1_p32"][0, 0, 0] = math.pi / 1000.0
    realization = M10Service(project).generate_batch(
        M10GenerationConfig(base_seed=42, realization_count=1, condition_calibration_observations=False)
    )[0]
    assert realization.quality.stochastic_count == 1
    path = tmp_path / "single-sample.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)
    np.testing.assert_equal(
        restored.m10_state.realizations[0].geometry_arrays["normal"],
        realization.geometry_arrays["normal"],
    )


def test_dip_only_density_evidence_is_provenance_not_a_fabricated_direction():
    project = make_m10_project()
    project.m9_state.p32_estimates = [
        P32Estimate(
            domain_id=1,
            set_id=1,
            fracture_count=10,
            raw_sample_length=100,
            effective_sample_length=80,
            mean_exposure=0.8,
            p32=0.1,
            observability=ObservabilityState.ADEQUATE,
            random_seed=42,
            full_orientation_count=6,
            dip_only_count=4,
        )
    ]
    realization = M10Service(project).generate_batch(
        M10GenerationConfig(base_seed=4, condition_calibration_observations=False)
    )[0]
    stochastic = source_mask(realization, "STOCHASTIC")
    assert stochastic.any()
    assert realization.geometry_arrays["dip_only_density_evidence"][stochastic].all()
    assert set(realization.geometry_arrays["orientation_source_code"][stochastic]) == {0}


def test_m10_project_store_consecutive_save_and_autosave(tmp_path: Path):
    project = make_m10_project()
    M10Service(project).generate_batch(
        M10GenerationConfig(base_seed=18, condition_calibration_observations=False)
    )
    store = ProjectStore()
    store.adopt_project(project)
    path = tmp_path / "autosave-m10.dfnproj"
    store.save_as(path)
    store.mark_dirty()
    store.configure_auto_save(enabled=True, interval_seconds=0)
    assert store.tick_auto_save()
    restored = ZipProjectStore().load(path)
    np.testing.assert_equal(
        fracture_ids(restored.m10_state.realizations[0]),
        fracture_ids(project.m10_state.realizations[0]),
    )


def test_large_m10_state_is_explicitly_skipped_by_synchronous_autosave(tmp_path: Path):
    project = make_m10_project()
    M10Service(project).generate_batch(M10GenerationConfig(condition_calibration_observations=False))
    store = ProjectStore()
    store.adopt_project(project)
    path = tmp_path / "large-autosave.dfnproj"
    store.save_as(path)
    store._large_autosave_limit_bytes = 1
    store.mark_dirty()
    store.configure_auto_save(enabled=True, interval_seconds=0)
    assert store.tick_auto_save() is False
    assert store.is_dirty


def test_deterministic_csv_import_is_transactional_and_audited(tmp_path: Path):
    path = tmp_path / "structures.csv"
    path.write_text(
        "structure_id,center_x,center_y,center_z,dip_direction,dip,radius,structure_type,domain_id,set_id\n"
        "F-1,1,2,3,90,70,5,fault,1,2\n",
        encoding="utf-8",
    )
    project = make_m10_project()
    original = list(project.m10_state.deterministic_structures)
    imported = M10Service(project).import_deterministic_csv(path, commit=False)
    assert project.m10_state.deterministic_structures == original
    assert imported[0].dip_direction == 90.0
    assert imported[0].source_file == str(path)
    assert imported[0].source_row == 2
    assert imported[0].set_id == 2


def test_legacy_json_save_is_rejected_instead_of_silently_losing_m10_arrays(tmp_path: Path):
    project = make_m10_project()
    M10Service(project).generate_batch(
        M10GenerationConfig(base_seed=3, condition_calibration_observations=False)
    )
    store = ProjectStore()
    store.adopt_project(project)
    with pytest.raises(ValueError, match="requires the compressed .dfnproj"):
        store.save_as(tmp_path / "unsafe.dfncs")
