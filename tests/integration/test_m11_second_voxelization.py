"""End-to-end M10-to-M11.1 computation and project round-trip."""

from __future__ import annotations

import numpy as np

from dfn_cave_studio.models.m10 import M10GenerationConfig
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.m10_service import M10Service
from dfn_cave_studio.services.m11_service import M11SecondVoxelizationService
from tests.integration.test_m10_persistence_export import make_m10_project


def test_generated_m10_second_voxelization_save_reopen_is_identical(tmp_path) -> None:
    project = make_m10_project()
    realization = M10Service(project).generate_batch(
        M10GenerationConfig(base_seed=42, condition_calibration_observations=False)
    )[0]
    m10_before = {name: values.copy() for name, values in realization.geometry_arrays.items()}
    result = M11SecondVoxelizationService(project).compute(realization.realization_id)
    assert result.complete
    assert result.positive_intersection_count == result.pair_count
    assert result.conservation.absolute_error_total < 1e-9
    assert result.provenance["unresolved_orientation_in_total"] is False
    for name, expected in m10_before.items():
        np.testing.assert_array_equal(realization.geometry_arrays[name], expected)

    path = tmp_path / "complete-m11.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)
    loaded = restored.m11_state.result_for(realization.realization_id)
    assert loaded is not None
    assert loaded.model_dump(exclude={"arrays"}) == result.model_dump(exclude={"arrays"})
    assert loaded.arrays.keys() == result.arrays.keys()
    for name, expected in result.arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], expected)


def test_m11_result_is_invalid_after_source_hash_changes() -> None:
    project = make_m10_project()
    realization = M10Service(project).generate_batch(
        M10GenerationConfig(base_seed=7, condition_calibration_observations=False)
    )[0]
    service = M11SecondVoxelizationService(project)
    result = service.compute(realization.realization_id)
    assert service.is_valid(result)
    original_p32 = project.m9_state.parameter_field_arrays["set_1_p32"].copy()
    project.m9_state.parameter_field_arrays["set_1_p32"][0, 0, 0] += 0.01
    assert not service.is_valid(result)
    project.m9_state.parameter_field_arrays["set_1_p32"][:] = original_p32
    assert service.is_valid(result)
    realization.config_hash = "changed"
    assert not service.is_valid(result)
