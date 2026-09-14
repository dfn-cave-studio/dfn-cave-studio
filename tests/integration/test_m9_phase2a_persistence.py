"""Save/reopen lineage and arrays for the Phase 2A-backed M9 field."""

from __future__ import annotations

import numpy as np

from dfn_cave_studio.models.m9 import DensitySettings, M9DensityInputMode
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.m9_service import M9Service
from tests.unit.test_m9_phase2a_adapter import _project_with_realizations


def test_phase2a_backed_m9_field_save_reopen_is_traceable(tmp_path) -> None:
    project = _project_with_realizations(2)
    selected = project.borehole_fracture_state.realizations[1]
    service = M9Service(project)
    project.m9_state = service.calculate_density(
        DensitySettings(monte_carlo_samples=100),
        input_mode=M9DensityInputMode.PHASE2A_REALIZATION,
        realization_id=selected.realization_id,
        commit=False,
    )
    service.set_assumed_sizes("fixed", {"radius": 2.0}, 2.0, 2.000001)
    service.build_parameter_field()
    expected = {name: value.copy() for name, value in project.m9_state.parameter_field_arrays.items()}
    target = tmp_path / "synthetic-phase2a-m9.dfnproj"
    ZipProjectStore().save(project, target)
    reopened = ZipProjectStore().load(target)
    assert reopened.m9_state.density_input_mode == M9DensityInputMode.PHASE2A_REALIZATION
    assert reopened.m9_state.density_input_realization_id == selected.realization_id
    assert reopened.m9_state.provenance["density_input"]["input_hash"] == selected.input_hash
    assert set(reopened.m9_state.parameter_field_arrays) == set(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(reopened.m9_state.parameter_field_arrays[name], values)
    loaded = [item for item in reopened.borehole_fracture_state.realizations if item.arrays]
    assert len(loaded) == 1
    resident_id = loaded[0].realization_id
    candidate = M9Service(reopened).calculate_density(
        DensitySettings(monte_carlo_samples=100),
        input_mode=M9DensityInputMode.PHASE2A_REALIZATION,
        realization_id=selected.realization_id,
        commit=False,
    )
    assert candidate.density_input_realization_id == selected.realization_id
    assert reopened.borehole_fracture_state.selected_realization_id == resident_id
    assert [item.realization_id for item in reopened.borehole_fracture_state.realizations if item.arrays] == [resident_id]
