"""Synthetic scientific contracts for the exclusive Phase 2A to M9 adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dfn_cave_studio.models.borehole_database import BoreholeDataType
from dfn_cave_studio.models.borehole_fracture_realization import BoreholeFractureGenerationConfig
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.m9 import (
    DensityMethod,
    DensitySettings,
    KrigingSettings,
    M9DensityInputMode,
    ValidationState,
    VariogramMode,
)
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig
from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.services.m9_service import M9Service
from dfn_cave_studio.services.m9_phase2a_adapter import Phase2AM9Adapter
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from tests.unit.test_borehole_fracture_phase2a import _synthetic_project


def _project_with_realizations(count: int = 2, *, include_validation: bool = False):
    project = _synthetic_project()
    if include_validation:
        repository = BoreholeRepository(project)
        repository.import_dataframe(
            BoreholeDataType.COLLARS,
            pd.DataFrame(
                [
                    {
                        "borehole_id": "SYN-V",
                        "collar_x": 20.0,
                        "collar_y": 10.0,
                        "collar_z": 90.0,
                        "final_depth": 20.0,
                        "azimuth": 90.0,
                        "dip": -45.0,
                    }
                ]
            ),
            "synthetic-validation-collar.csv",
        )
        repository.import_dataframe(
            BoreholeDataType.SURVEYS,
            pd.DataFrame(
                [
                    {"hole_id": "SYN-V", "measured_depth": 0.0, "azimuth": 90.0, "dip": -45.0},
                    {"hole_id": "SYN-V", "measured_depth": 20.0, "azimuth": 90.0, "dip": -45.0},
                ]
            ),
            "synthetic-validation-survey.csv",
        )
        repository.import_dataframe(
            BoreholeDataType.FRACTURES,
            pd.DataFrame(
                [{"hole_id": "SYN-V", "from_depth": 0.0, "to_depth": 20.0, "fracture_spacing": 0.5}]
            ),
            "synthetic-validation-spacing.csv",
            import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "m"},
        )
    project.joint_sets = [
        JointSetConfig(
            set_id=4,
            orientation=OrientationDistribution(mean_dip_direction=20.0, mean_dip=35.0, kappa=25.0),
        ),
        JointSetConfig(
            set_id=9,
            orientation=OrientationDistribution(mean_dip_direction=210.0, mean_dip=65.0, kappa=25.0),
        ),
        JointSetConfig(
            set_id=20,
            orientation=OrientationDistribution(mean_dip_direction=110.0, mean_dip=80.0, kappa=25.0),
        ),
    ]
    generator = BoreholeFractureService(project)
    fit = generator.use_confirmed_project_sets()
    project.borehole_fracture_state = project.borehole_fracture_state.model_copy(update={"global_fit": fit})
    project.borehole_fracture_state = generator.build_candidate(
        BoreholeFractureGenerationConfig(
            use_confirmed_global_fit=True,
            realization_count=count,
            master_seed=42,
        )
    )
    holdout = HoldoutService()
    holdout.select_manual(
        ["SYN-A", *( ["SYN-V"] if include_validation else [])],
        ["SYN-V"] if include_validation else [],
    )
    holdout.lock()
    set_holdout(project, holdout)
    bounds = ModelBounds(x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=100)
    project.voxel_config = VoxelConfig(cell_size_x=50, cell_size_y=50, cell_size_z=50)
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=bounds, generation_domain=bounds)
    return project


def test_selected_realization_is_exclusive_traceable_and_preserves_no_data() -> None:
    project = _project_with_realizations()
    realization = project.borehole_fracture_state.realizations[0]
    formal_before = project.borehole_collection.model_copy(deep=True)
    candidate = M9Service(project).calculate_density(
        DensitySettings(
            interval_length=10,
            method=DensityMethod.IDW,
            min_neighbors=1,
            max_neighbors=8,
            global_fallback=True,
            monte_carlo_samples=100,
        ),
        input_mode=M9DensityInputMode.PHASE2A_REALIZATION,
        realization_id=realization.realization_id,
        commit=False,
    )
    assert project.borehole_collection == formal_before
    assert project.m9_state.p10_intervals == []
    assert candidate.density_input_realization_id == realization.realization_id
    lineage = candidate.provenance["density_input"]
    assert lineage["generator_version"] == realization.generator_version
    assert lineage["master_seed"] == realization.master_seed
    assert lineage["input_hash"] == realization.input_hash
    assert lineage["excluded_random_background_count"] == realization.random_background_count
    set_20 = [item for item in candidate.p10_intervals if item.set_id == 20]
    assert set_20 and all(item.p10 is None and item.data_state == "no_data" for item in set_20)
    assert not any(item.set_id == -1 for item in candidate.p10_intervals)
    assert {item.set_id for item in candidate.p32_estimates} <= {4, 9}


def test_two_realizations_are_never_auto_mixed_and_switch_deterministically() -> None:
    project = _project_with_realizations()
    service = M9Service(project)
    first, second = project.borehole_fracture_state.realizations
    first_state = service.calculate_density(
        DensitySettings(monte_carlo_samples=100),
        input_mode="phase2a_realization",
        realization_id=first.realization_id,
        commit=False,
    )
    second_state = service.calculate_density(
        DensitySettings(monte_carlo_samples=100),
        input_mode="phase2a_realization",
        realization_id=second.realization_id,
        commit=False,
    )
    assert first_state.density_input_realization_id == first.realization_id
    assert second_state.density_input_realization_id == second.realization_id
    assert sum(item.observation_count for item in first_state.p10_intervals) == sum(first.global_set_counts.values())
    assert sum(item.observation_count for item in second_state.p10_intervals) == sum(second.global_set_counts.values())


def test_cancelled_candidate_does_not_commit_and_stale_hash_is_rejected() -> None:
    project = _project_with_realizations(1)
    service = M9Service(project)
    original = project.m9_state
    realization = project.borehole_fracture_state.realizations[0]
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    try:
        service.calculate_density(
            DensitySettings(monte_carlo_samples=100),
            input_mode="phase2a_realization",
            realization_id=realization.realization_id,
            cancelled=cancelled,
            commit=False,
        )
    except InterruptedError:
        pass
    else:
        raise AssertionError("cancellation must stop adaptation")
    assert project.m9_state is original
    realization.input_hash = "0" * 64
    try:
        service.calculate_density(
            DensitySettings(monte_carlo_samples=100),
            input_mode="phase2a_realization",
            realization_id=realization.realization_id,
            commit=False,
        )
    except ValueError as exc:
        assert "stale" in str(exc).lower()
    else:
        raise AssertionError("stale realization must be rejected")


def test_parameter_field_and_validation_keep_phase2a_lineage() -> None:
    project = _project_with_realizations(1, include_validation=True)
    realization = project.borehole_fracture_state.realizations[0]
    service = M9Service(project)
    project.m9_state = service.calculate_density(
        DensitySettings(
            method=DensityMethod.GLOBAL_CONSTANT,
            monte_carlo_samples=100,
        ),
        input_mode="phase2a_realization",
        realization_id=realization.realization_id,
        commit=False,
    )
    service.set_assumed_sizes("fixed", {"radius": 2.0}, 2.0, 2.000001)
    metadata, arrays = service.build_parameter_field()
    assert metadata.provenance["density_input"]["realization_id"] == realization.realization_id
    assert arrays and all(isinstance(value, np.ndarray) for value in arrays.values())
    summary = service.validate()
    assert summary.state == ValidationState.NOT_VALIDATED
    assert project.m9_state.validation_results
    assert {
        item.calibration_or_validation for item in project.m9_state.validation_results
    } == {"internal_consistency_not_independent"}
    assert all("SYN-V" not in item.calibration_holes for item in project.m9_state.p32_estimates)
    assert project.m9_state.provenance["validation_interpretation"]["p32_ground_truth"] is False


def test_phase2a_input_supports_existing_ordinary_kriging_without_new_conversion() -> None:
    project = _project_with_realizations(1)
    realization = project.borehole_fracture_state.realizations[0]
    service = M9Service(project)
    project.m9_state = service.calculate_density(
        DensitySettings(
            method=DensityMethod.ORDINARY_KRIGING,
            monte_carlo_samples=100,
            kriging=KrigingSettings(
                mode=VariogramMode.MANUAL,
                nugget=0.0,
                sill=1.0,
                range=50.0,
                minimum_neighbors=2,
                maximum_neighbors=8,
            ),
        ),
        input_mode="phase2a_realization",
        realization_id=realization.realization_id,
        commit=False,
    )
    service.set_assumed_sizes("fixed", {"radius": 2.0}, 2.0, 2.000001)
    metadata, arrays = service.build_parameter_field()
    assert metadata.density_method == DensityMethod.ORDINARY_KRIGING
    assert "kriging_variance_total" in arrays
    assert metadata.provenance["density_input"]["p32_semantics"].startswith("direction-corrected")


def _translate_realization_frame(project, translation: np.ndarray) -> None:
    """Translate synthetic trajectory and float64 origin without changing local offsets."""
    for hole in project.borehole_collection.boreholes:
        hole.collar.collar_x += float(translation[0])
        hole.collar.collar_y += float(translation[1])
        hole.collar.collar_z += float(translation[2])
    for realization in project.borehole_fracture_state.realizations:
        origin = np.asarray(realization.provenance["xyz_origin"], dtype=np.float64)
        realization.provenance["xyz_origin"] = (origin + translation).tolist()


def test_xyz_validation_uses_origin_relative_float32_contract_at_large_coordinates() -> None:
    project = _project_with_realizations(1)
    realization = project.borehole_fracture_state.realizations[0]
    offsets_before = realization.arrays["xyz_offset"].copy()
    _translate_realization_frame(project, np.asarray([1.25e9, -2.5e9, 7.5e8]))

    large = Phase2AM9Adapter(project, xyz_validation_chunk_size=3).build(
        realization.realization_id,
        {"SYN-A": "calibration"},
        interval_mode="fixed",
        interval_length=10.0,
    )
    assert large.intervals
    np.testing.assert_array_equal(realization.arrays["xyz_offset"], offsets_before)

    realization.arrays["xyz_offset"][0, 0] += np.float32(2.0)
    with np.testing.assert_raises_regex(ValueError, r"maximum component mismatch 2.*rounding ceiling.*stored XYZ"):
        Phase2AM9Adapter(project).build(
            realization.realization_id,
            {"SYN-A": "calibration"},
            interval_mode="fixed",
            interval_length=10.0,
        )


def test_xyz_validation_is_translation_invariant_and_chunk_invariant() -> None:
    local_project = _project_with_realizations(1)
    translated_project = local_project.model_copy(deep=True)
    _translate_realization_frame(translated_project, np.asarray([8.0e8, 9.0e8, -6.0e8]))
    local_realization = local_project.borehole_fracture_state.realizations[0]
    translated_realization = translated_project.borehole_fracture_state.realizations[0]

    local = Phase2AM9Adapter(local_project, xyz_validation_chunk_size=1).build(
        local_realization.realization_id,
        {"SYN-A": "calibration"},
        interval_mode="fixed",
        interval_length=10.0,
    )
    translated_small_chunks = Phase2AM9Adapter(translated_project, xyz_validation_chunk_size=1).build(
        translated_realization.realization_id,
        {"SYN-A": "calibration"},
        interval_mode="fixed",
        interval_length=10.0,
    )
    translated = Phase2AM9Adapter(translated_project, xyz_validation_chunk_size=7).build(
        translated_realization.realization_id,
        {"SYN-A": "calibration"},
        interval_mode="fixed",
        interval_length=10.0,
    )
    assert local.intervals
    assert [item.model_dump() for item in translated_small_chunks.intervals] == [
        item.model_dump() for item in translated.intervals
    ]
    assert translated_small_chunks.orientation_models == translated.orientation_models
    assert translated_small_chunks.provenance == translated.provenance


def test_xyz_validation_caches_each_hole_trajectory_across_chunks(monkeypatch) -> None:
    project = _project_with_realizations(1)
    realization = project.borehole_fracture_state.realizations[0]
    hole_type = type(project.borehole_collection.boreholes[0])
    original = hole_type.compute_trajectory
    calls: dict[str, int] = {}

    def counted(hole, *args, **kwargs):
        calls[hole.borehole_id] = calls.get(hole.borehole_id, 0) + 1
        return original(hole, *args, **kwargs)

    monkeypatch.setattr(hole_type, "compute_trajectory", counted)
    Phase2AM9Adapter(project, xyz_validation_chunk_size=1)._validate_source(realization, realization.arrays)
    assert calls == {"SYN-A": 1}
