"""M11.1 project persistence and invalidation regression tests."""

from __future__ import annotations

import numpy as np

from dfn_cave_studio.models.m10 import M10GenerationConfig, M10Realization
from dfn_cave_studio.models.m11 import M11SecondVoxelizationResult
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.persistence.project_store import ProjectStore
from dfn_cave_studio.services.m10_service import M10Service


def test_m11_sparse_and_dense_arrays_round_trip(tmp_path) -> None:
    project = Project()
    project.m10_state.realizations = [
        M10Realization(
            realization_id="m10-r0",
            realization_index=0,
            seed=42,
            config_hash="source-hash",
            geometry_arrays={
                "center": np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
                "normal": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
                "radius": np.asarray([0.2], dtype=np.float32),
                "ordinal": np.asarray([0], dtype=np.uint32),
                "set_id": np.asarray([1], dtype=np.int16),
                "source_code": np.asarray([0], dtype=np.uint8),
                "size_class": np.asarray([3], dtype=np.uint8),
            },
        )
    ]
    arrays = {
        "fracture_ordinal": np.asarray([0], dtype=np.uint32),
        "voxel_flat_index": np.asarray([7], dtype=np.uint32),
        "intersection_area": np.asarray([0.125], dtype=np.float64),
        "p32_total": np.arange(8, dtype=np.float64).reshape(2, 2, 2),
        "cell_state": np.full((2, 2, 2), 3, dtype=np.uint8),
    }
    project.m11_state.results = [
        M11SecondVoxelizationResult(
            realization_id="m11-m10-r0",
            source_m10_realization_id="m10-r0",
            source_m10_config_hash="source-hash",
            source_parameter_field_hash="field-hash",
            candidate_pair_count=2,
            positive_intersection_count=1,
            rejected_candidate_count=1,
            arrays=arrays,
        )
    ]
    path = tmp_path / "m11-roundtrip.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)

    assert restored.schema_version == 5
    assert len(restored.m11_state.results) == 1
    loaded = restored.m11_state.results[0]
    assert loaded.algorithm_version == "m11-second-voxelization-1"
    assert loaded.arrays.keys() == arrays.keys()
    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], expected)


def test_old_m10_project_defaults_to_empty_m11_state() -> None:
    raw = Project().model_dump(mode="json", exclude={"m11_state"})
    raw["schema_version"] = 4
    restored = Project.from_dict(raw)
    assert restored.schema_version == 5
    assert restored.m11_state.results == []


def test_m10_regeneration_invalidates_all_m11_results() -> None:
    project = Project()
    project.m11_state.results = [
        M11SecondVoxelizationResult(
            realization_id="old-m11",
            source_m10_realization_id="old-m10",
            source_m10_config_hash="old",
            source_parameter_field_hash="old-field",
        )
    ]
    replacement = M10Realization(
        realization_id="new-m10", realization_index=0, seed=42, config_hash="new", geometry_arrays={}
    )
    M10Service(project).commit_realizations(M10GenerationConfig(), [replacement], replace=True)
    assert project.m11_state.results == []
    assert project.m11_state.provenance["invalidated_by"] == "M10 realization regeneration"


def test_legacy_json_save_rejects_m11_arrays(tmp_path) -> None:
    project = Project()
    project.m11_state.results = [
        M11SecondVoxelizationResult(
            realization_id="m11",
            source_m10_realization_id="m10",
            source_m10_config_hash="source",
            source_parameter_field_hash="field",
            arrays={"intersection_area": np.asarray([1.0])},
        )
    ]
    store = ProjectStore()
    store.adopt_project(project)
    with np.testing.assert_raises_regex(ValueError, "M10/M11 geometry requires"):
        store.save_as(tmp_path / "unsafe.dfncs")


def test_large_m11_state_is_skipped_by_synchronous_autosave(tmp_path) -> None:
    project = Project()
    project.m11_state.results = [
        M11SecondVoxelizationResult(
            realization_id="m11",
            source_m10_realization_id="m10",
            source_m10_config_hash="source",
            source_parameter_field_hash="field",
            arrays={"intersection_area": np.ones(4, dtype=np.float64)},
        )
    ]
    store = ProjectStore()
    store.adopt_project(project)
    store.save_as(tmp_path / "large-m11.dfnproj")
    store._large_autosave_limit_bytes = 1
    store.mark_dirty()
    store.configure_auto_save(enabled=True, interval_seconds=0)
    assert store.tick_auto_save() is False
    assert store.is_dirty
