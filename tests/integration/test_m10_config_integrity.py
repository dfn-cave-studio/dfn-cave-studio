"""Persistence and copy-on-recovery regressions for invalid M10 metadata."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from dfn_cave_studio.models.m10 import M10GenerationConfig
from dfn_cave_studio.persistence.project_store import ProjectStore
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.m7_state import get_workflow, set_workflow
from dfn_cave_studio.services.m10_config_recovery import M10ConfigRecoveryService
from dfn_cave_studio.services.m10_service import M10Service
from dfn_cave_studio.services.m11_service import M11SecondVoxelizationService
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController
from tests.integration.test_m10_persistence_export import make_m10_project


def _complete_project():
    project = make_m10_project()
    config = M10GenerationConfig(base_seed=42, condition_calibration_observations=False)
    realization = M10Service(project).generate_batch(config)[0]
    M11SecondVoxelizationService(project).compute(realization.realization_id)
    workflow = WorkflowController()
    for step in ("explicit_dfn", "second_voxelization"):
        workflow.complete_step(step)
    set_workflow(project, workflow)
    return project


def _replace_m10_config(path: Path, updates: dict[str, object]) -> None:
    temporary = path.with_suffix(".rewrite")
    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(temporary, "w", allowZip64=True) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == M10ConfigRecoveryService.M10_STATE_PATH:
                state = json.loads(payload.decode("utf-8"))
                state["config"].update(updates)
                payload = json.dumps(state, indent=2).encode("utf-8")
            target.writestr(info, payload)
    os.replace(temporary, path)


def test_save_validation_preserves_existing_file_and_dirty_state(tmp_path: Path) -> None:
    project = _complete_project()
    project.m10_state.config = project.m10_state.config.model_copy(
        update={"small_area_share": 0.998, "medium_large_cumulative_share": 0.7}
    )
    target = tmp_path / "existing.dfnproj"
    original = b"original project bytes"
    target.write_bytes(original)
    store = ProjectStore()
    store.adopt_project(project, target)
    store.mark_dirty()
    with pytest.raises(ValueError, match="Auto shares"):
        store.save()
    assert target.read_bytes() == original
    assert store.is_dirty
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_direct_zip_save_rejects_invalid_nested_config_before_opening_target(tmp_path: Path) -> None:
    project = _complete_project()
    project.m10_state.config = project.m10_state.config.model_copy(
        update={"small_area_share": 0.998, "medium_large_cumulative_share": 0.7}
    )
    target = tmp_path / "direct.dfnproj"
    original = b"do not replace"
    target.write_bytes(original)
    with pytest.raises(ValueError, match="small_area_share"):
        ZipProjectStore().save(project, target)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_service_revalidates_unchecked_config_before_any_state_invalidation() -> None:
    project = _complete_project()
    before_m10 = project.m10_state.model_copy(deep=True)
    before_m11 = project.m11_state.model_copy(deep=True)
    invalid = project.m10_state.config.model_copy(
        update={"small_area_share": 0.998, "medium_large_cumulative_share": 0.7}
    )
    with pytest.raises(ValueError, match="small_area_share"):
        M10Service(project).commit_realizations(invalid, list(project.m10_state.realizations))
    assert project.m10_state.model_dump(mode="python", exclude={"realizations"}) == before_m10.model_dump(
        mode="python", exclude={"realizations"}
    )
    assert project.m11_state.model_dump(mode="python", exclude={"results"}) == before_m11.model_dump(
        mode="python", exclude={"results"}
    )
    for current, expected in zip(project.m10_state.realizations, before_m10.realizations):
        for name in current.geometry_arrays:
            np.testing.assert_array_equal(current.geometry_arrays[name], expected.geometry_arrays[name])
    for current, expected in zip(project.m11_state.results, before_m11.results):
        for name in current.arrays:
            np.testing.assert_array_equal(current.arrays[name], expected.arrays[name])


def test_recovery_creates_only_new_file_and_preserves_m10_m11_arrays(tmp_path: Path) -> None:
    project = _complete_project()
    valid_source = tmp_path / "valid.dfnproj"
    ZipProjectStore().save(project, valid_source)
    _replace_m10_config(
        valid_source,
        {"small_area_share": 0.998, "medium_large_cumulative_share": 0.7},
    )
    invalid_source_bytes = valid_source.read_bytes()
    inspection = M10ConfigRecoveryService.inspect(valid_source)
    assert not inspection.valid
    assert inspection.raw_config["small_area_share"] == 0.998
    assert any("Auto shares" in error for error in inspection.validation_errors)

    recovered_path = tmp_path / "recovered.dfnproj"
    replacement = M10GenerationConfig.model_validate(
        {**inspection.raw_config, "small_area_share": 0.1, "medium_large_cumulative_share": 0.7}
    )
    M10ConfigRecoveryService.recover_to_new_file(
        valid_source, recovered_path, replacement, confirmed=True
    )
    assert valid_source.read_bytes() == invalid_source_bytes
    restored = ZipProjectStore().load(recovered_path)
    assert restored.m10_state.config == replacement
    assert restored.m10_state.provenance["config_recovery"]["result_config_consistency_confirmed"] is False
    assert restored.m11_state.provenance["m10_config_consistency"] == "unconfirmed_after_metadata_recovery"
    assert get_workflow(restored).get_step("explicit_dfn").status == StepStatus.STALE
    assert get_workflow(restored).get_step("second_voxelization").status == StepStatus.STALE
    for original, recovered in zip(project.m10_state.realizations, restored.m10_state.realizations):
        assert original.geometry_arrays.keys() == recovered.geometry_arrays.keys()
        for name in original.geometry_arrays:
            np.testing.assert_array_equal(original.geometry_arrays[name], recovered.geometry_arrays[name])
    for original, recovered in zip(project.m11_state.results, restored.m11_state.results):
        assert original.arrays.keys() == recovered.arrays.keys()
        for name in original.arrays:
            np.testing.assert_array_equal(original.arrays[name], recovered.arrays[name])


def test_recovery_requires_confirmation_and_different_new_path(tmp_path: Path) -> None:
    project = _complete_project()
    source = tmp_path / "source.dfnproj"
    ZipProjectStore().save(project, source)
    with pytest.raises(ValueError, match="confirmation"):
        M10ConfigRecoveryService.recover_to_new_file(
            source, tmp_path / "copy.dfnproj", project.m10_state.config, confirmed=False
        )
    with pytest.raises(ValueError, match="new file"):
        M10ConfigRecoveryService.recover_to_new_file(
            source, source, project.m10_state.config, confirmed=True
        )
