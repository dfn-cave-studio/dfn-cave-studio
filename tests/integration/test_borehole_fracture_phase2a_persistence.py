"""Persistence contracts for columnar Phase 2A borehole realizations."""

from __future__ import annotations

import numpy as np

from dfn_cave_studio.models.borehole_fracture_realization import BoreholeFractureGenerationConfig
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService
from tests.unit.test_borehole_fracture_phase2a import _synthetic_project


def test_save_reopen_loads_only_selected_realization_and_preserves_arrays(tmp_path) -> None:
    project = _synthetic_project()
    candidate = BoreholeFractureService(project).build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=2, realization_count=2, master_seed=17)
    )
    original = {
        realization.realization_id: {name: values.copy() for name, values in realization.arrays.items()}
        for realization in candidate.realizations
    }
    BoreholeFractureService(project).commit(candidate)
    target = tmp_path / "synthetic-phase2a.dfnproj"
    store = ZipProjectStore()
    store.save(project, target)

    reopened = store.load(target)
    state = reopened.borehole_fracture_state
    assert len(state.realizations) == 2
    assert state.archive_source == str(target)
    selected = state.selected_realization_id
    assert selected is not None
    assert [bool(item.arrays) for item in state.realizations].count(True) == 1
    for item in state.realizations:
        if item.realization_id == selected:
            for name, values in original[selected].items():
                assert values.dtype == item.arrays[name].dtype
                np.testing.assert_array_equal(values, item.arrays[name])

    other = next(item.realization_id for item in state.realizations if item.realization_id != selected)
    store.load_borehole_fracture_arrays(reopened, target, other)
    assert state.selected_realization_id == other
    assert [bool(item.arrays) for item in state.realizations].count(True) == 1
    loaded = next(item for item in state.realizations if item.realization_id == other)
    for name, values in original[other].items():
        np.testing.assert_array_equal(values, loaded.arrays[name])
    second = tmp_path / "synthetic-phase2a-resaved.dfnproj"
    store.save(reopened, second)
    twice = store.load(second)
    assert len(twice.borehole_fracture_state.realizations) == 2
    assert twice.borehole_fracture_state.global_fit == state.global_fit
    assert state.global_fit is not None
    assert len(state.global_fit.local_components) == 4
    selected_realization = next(item for item in state.realizations if item.realization_id == other)
    assert selected_realization.arrays["local_component_index"].dtype == np.int32
    assert all(item.local_component_probabilities for item in selected_realization.interval_diagnostics)


def test_old_project_without_phase2a_member_gets_empty_compatible_state(tmp_path) -> None:
    project = _synthetic_project()
    target = tmp_path / "synthetic-old-compatible.dfnproj"
    ZipProjectStore().save(project, target)
    reopened = ZipProjectStore().load(target)
    assert reopened.borehole_fracture_state.realizations == []
    assert reopened.borehole_fracture_state.global_fit is None
