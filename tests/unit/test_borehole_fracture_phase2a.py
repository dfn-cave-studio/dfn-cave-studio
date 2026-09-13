"""Synthetic scientific contracts for Phase 2A along-hole realizations."""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest

from dfn_cave_studio.models.borehole_database import BoreholeDataType, RecordState
from dfn_cave_studio.models.borehole_fracture_realization import (
    BoreholeFractureComponent,
    BoreholeFractureGenerationConfig,
    RandomComponentStatus,
)
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_fracture_service import (
    BoreholeFractureService,
    BoreholeGenerationCancelled,
    HomogeneousPoissonSampler,
    StableIDW,
)
from dfn_cave_studio.services.joint_set_service import joint_set_color
from dfn_cave_studio.services.borehole_repository import BoreholeRepository


def _synthetic_project(*, include_random: bool = True) -> Project:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        BoreholeDataType.COLLARS,
        pd.DataFrame(
            [
                {
                    "borehole_id": "SYN-A",
                    "collar_x": 10.0,
                    "collar_y": 10.0,
                    "collar_z": 90.0,
                    "final_depth": 40.0,
                    "azimuth": 90.0,
                    "dip": -45.0,
                }
            ]
        ),
        "synthetic-collars.csv",
    )
    repository.import_dataframe(
        BoreholeDataType.SURVEYS,
        pd.DataFrame(
            [
                {"hole_id": "SYN-A", "measured_depth": 0.0, "azimuth": 90.0, "dip": -45.0},
                {"hole_id": "SYN-A", "measured_depth": 40.0, "azimuth": 90.0, "dip": -45.0},
            ]
        ),
        "synthetic-surveys.csv",
    )
    rows = [
        {
            "observation_id": "P-A-1",
            "point_id": "A",
            "x": 10.0,
            "y": 10.0,
            "z": 90.0,
            "dip": 35.0,
            "dip_direction": 20.0,
            "local_set_id": "LOCAL-RED",
            "joint_spacing_m": 0.8,
            "joint_num": 8,
        },
        {
            "observation_id": "P-B-1",
            "point_id": "B",
            "x": 35.0,
            "y": 10.0,
            "z": 70.0,
            "dip": 65.0,
            "dip_direction": 210.0,
            "local_set_id": "LOCAL-RED",
            "joint_spacing_m": 1.2,
            "joint_num": 5,
        },
        {
            "observation_id": "Z-C-1",
            "point_id": "C",
            "x": 20.0,
            "y": 10.0,
            "z": 80.0,
            "dip": 36.0,
            "dip_direction": 21.0,
            "local_set_id": "CAMERA-A",
            "joint_spacing_m": None,
            "joint_num": None,
        },
    ]
    if include_random:
        rows.append(
            {
                "observation_id": "P-A-R",
                "point_id": "A",
                "x": 10.0,
                "y": 10.0,
                "z": 90.0,
                "dip": None,
                "dip_direction": None,
                "local_set_id": "RANDOM",
                "joint_spacing_m": 2.0,
                "joint_num": 2,
            }
        )
    result = repository.import_dataframe(
        BoreholeDataType.ORIENTATION_POINTS,
        pd.DataFrame(rows),
        "synthetic-orientations.csv",
    )
    assert result["excluded"] == 0
    repository.import_dataframe(
        BoreholeDataType.FRACTURES,
        pd.DataFrame(
            [
                {
                    "hole_id": "SYN-A",
                    "from_depth": 0.0,
                    "to_depth": 20.0,
                    "fracture_spacing": 0.5,
                },
                {
                    "hole_id": "SYN-A",
                    "from_depth": 20.0,
                    "to_depth": 40.0,
                    "fracture_spacing": 0.8,
                },
            ]
        ),
        "synthetic-spacing.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "m"},
    )
    return project


def _array_hash(state) -> str:
    digest = hashlib.sha256()
    for realization in state.realizations:
        for name in sorted(realization.arrays):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(realization.arrays[name]).tobytes())
    return digest.hexdigest()


def test_global_mapping_is_stable_weighted_and_excludes_random() -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    first = service.fit_global_sets(2, 42)
    second = service.fit_global_sets(2, 42)
    assert first == second
    assert {item.observation_id for item in first.mappings} == {"P-A-1", "P-B-1", "Z-C-1"}
    assert next(item for item in first.mappings if item.observation_id == "P-A-1").sample_weight == 8
    assert next(item for item in first.mappings if item.observation_id == "Z-C-1").sample_weight == 1
    assert first.mappings[0].local_set_id == first.mappings[1].local_set_id
    assert first.mappings[0].point_key != first.mappings[1].point_key
    assert first.mappings[0].global_set_id != first.mappings[1].global_set_id
    assert len(first.local_components) == 4
    assert len({item.component_id for item in first.local_components}) == 4
    assert sum(item.component_type == "LOCAL_DOMINANT_SET" for item in first.local_components) == 3
    assert sum(item.component_type == "RANDOM_BACKGROUND" for item in first.local_components) == 1


def test_confirmed_project_sets_are_used_without_treating_local_ids_as_global(monkeypatch) -> None:
    project = _synthetic_project()
    project.joint_sets = [
        JointSetConfig(
            set_id=4,
            name="Synthetic north-east",
            orientation=OrientationDistribution(mean_dip_direction=20.0, mean_dip=35.0, kappa=25.0),
        ),
        JointSetConfig(
            set_id=9,
            name="Synthetic south-west",
            orientation=OrientationDistribution(mean_dip_direction=210.0, mean_dip=65.0, kappa=25.0),
        ),
    ]
    service = BoreholeFractureService(project)
    fit = service.use_confirmed_project_sets()
    assert fit.algorithm == "CONFIRMED_PROJECT_JOINT_SETS_WITH_AXIAL_NEAREST_MAPPING"
    assert {item.global_set_id for item in fit.sets} == {4, 9}
    same_local_id = [item for item in fit.mappings if item.local_set_id == "LOCAL-RED"]
    assert len(same_local_id) == 2
    assert {item.global_set_id for item in same_local_id} == {4, 9}
    assert fit.provenance["local_set_ids_are_point_local_only"] is True
    project.borehole_fracture_state = project.borehole_fracture_state.model_copy(update={"global_fit": fit})
    observation_ids_before = {
        item.observation_id for item in service.observations.orientation_points()
    }

    def fail_hidden_refit(*_args, **_kwargs):
        raise AssertionError("confirmed mode must not run a hidden global-set clustering")

    monkeypatch.setattr(service, "fit_global_sets", fail_hidden_refit)
    state = service.build_candidate(
        BoreholeFractureGenerationConfig(
            number_of_sets=2,
            use_confirmed_global_fit=True,
            master_seed=42,
        )
    )
    assert state.global_fit is not None
    assert state.global_fit.algorithm == "CONFIRMED_PROJECT_JOINT_SETS_WITH_AXIAL_NEAREST_MAPPING"
    dominant = state.realizations[0].arrays["component_type"] == int(BoreholeFractureComponent.DOMINANT_SET)
    assert set(np.unique(state.realizations[0].arrays["global_set_id"][dominant])) <= {4, 9}
    assert {item.observation_id for item in service.observations.orientation_points()} == observation_ids_before
    assert state.global_fit == fit


def test_random_status_defaults_to_not_reported_and_requires_explicit_absence() -> None:
    project = _synthetic_project(include_random=False)
    repository = BoreholeRepository(project)
    summaries = project.borehole_database.orientation_point_summaries
    assert {item.random_component_status for item in summaries} == {RandomComponentStatus.NOT_REPORTED}
    repository.set_random_component_status("POINT_CLOUD:A", RandomComponentStatus.REPORTED_ABSENT)
    summary = next(item for item in project.borehole_database.orientation_point_summaries if item.point_key.endswith(":A"))
    assert summary.random_component_status == RandomComponentStatus.REPORTED_ABSENT


def test_stable_idw_exact_radius_insufficient_and_tie_order() -> None:
    coordinates = np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=float)
    values = np.asarray([9.0, 1.0, 2.0, 3.0])
    assert float(StableIDW(2, 2, 2, 1).predict(coordinates, values, coordinates[0])) == 9.0
    assert StableIDW(2, 0.5, 3, 2).predict(coordinates, values, np.asarray([5, 0, 0])) is None
    tied = StableIDW(2, 2, 2, 1).predict(coordinates[1:], values[1:], np.zeros(3))
    assert float(tied) == pytest.approx(1.5)


def test_generation_is_columnar_half_open_trajectory_based_and_reproducible() -> None:
    project = _synthetic_project()
    database_before = project.borehole_database.model_copy(deep=True)
    config = BoreholeFractureGenerationConfig(number_of_sets=2, realization_count=2, master_seed=42)
    first = BoreholeFractureService(project).build_candidate(config)
    second = BoreholeFractureService(project).build_candidate(config)
    assert _array_hash(first) == _array_hash(second)
    assert _array_hash(first) == "35e763637b6af6f3d0f77a31d0e4ef45b1ba99af5dd71b3a522356af106bfda1"
    assert _array_hash(first) != _array_hash(
        BoreholeFractureService(project).build_candidate(config.model_copy(update={"master_seed": 43}))
    )
    assert len(first.realizations) == 2
    assert project.borehole_database == database_before
    for realization in first.realizations:
        assert not any(isinstance(value, list) for value in realization.arrays.values())
        assert realization.arrays["xyz_offset"].dtype == np.float32
        assert realization.arrays["measured_depth"].dtype == np.float64
        assert np.all(realization.arrays["measured_depth"] >= 0)
        assert np.all(realization.arrays["measured_depth"] < 40)
        assert np.unique(realization.arrays["measured_depth"]).size == realization.fracture_count
        dominant = realization.arrays["component_type"] == int(BoreholeFractureComponent.DOMINANT_SET)
        assert np.all(realization.arrays["global_set_id"][dominant] > 0)
        # The 45-degree inclined trajectory changes X and Z; MD is never copied into Z.
        xyz = realization.arrays["xyz_offset"].astype(float)
        assert np.ptp(xyz[:, 0]) > 0 and np.ptp(xyz[:, 2]) > 0
        assert not np.allclose(xyz[:, 2], realization.arrays["measured_depth"])
        assert realization.random_background_count > 0
        for diagnostic in realization.interval_diagnostics:
            assert sum(diagnostic.global_set_probabilities.values()) + diagnostic.random_probability == pytest.approx(1)


def test_camera_direction_does_not_contribute_density() -> None:
    project = _synthetic_project(include_random=False)
    service = BoreholeFractureService(project)
    fit = service.fit_global_sets(2, 42)
    constraints = service._intensity_constraints(fit)
    assert sum(len(site["components"]) for site in constraints["p_sites"]) == 2
    assert sum(len(rows) for rows in constraints["directions"].values()) == 3
    state = service.build_candidate(BoreholeFractureGenerationConfig(number_of_sets=2))
    assert all(
        item.random_component_diagnostic == "NOT_REPORTED_NO_RANDOM_ESTIMATE"
        for item in state.realizations[0].interval_diagnostics
    )


def test_cancel_never_commits_and_input_database_is_unchanged() -> None:
    project = _synthetic_project()
    database_before = project.borehole_database.model_copy(deep=True)
    service = BoreholeFractureService(project)
    with pytest.raises(BoreholeGenerationCancelled):
        service.build_candidate(BoreholeFractureGenerationConfig(number_of_sets=2), cancelled=lambda: True)
    assert project.borehole_fracture_state.realizations == []
    assert project.borehole_database == database_before


def test_camera_only_orientation_cannot_silently_supply_density() -> None:
    project = _synthetic_project()
    repository = BoreholeRepository(project)
    for record in list(repository.query(BoreholeDataType.ORIENTATION_POINTS, RecordState.FORMAL)):
        if str(record.values["observation_id"]).startswith("P"):
            repository.delete_record(record.record_id)
    state = BoreholeFractureService(project).build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=1, master_seed=9)
    )
    realization = state.realizations[0]
    assert realization.fracture_count == 0
    assert all(item.status == "BLOCKED_NO_LOCAL_INTENSITY" for item in realization.interval_diagnostics)


def test_memory_estimate_uses_compact_column_contract() -> None:
    project = _synthetic_project()
    details = BoreholeFractureService(project).estimate(
        BoreholeFractureGenerationConfig(number_of_sets=2, realization_count=3)
    )
    assert details["persistent_bytes"] == int(np.ceil(details["expected_fractures"])) * 42
    assert details["temporary_bytes"] > 0
    assert details["safety_margin_bytes"] > 0
    assert details["peak_bytes"] == (
        details["persistent_bytes"] + details["temporary_bytes"] + details["safety_margin_bytes"]
    )
    assert details["simultaneously_resident_realizations"] == 3


def test_two_phase_generation_preallocates_without_full_result_concatenation() -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    state = service.build_candidate(BoreholeFractureGenerationConfig(number_of_sets=2, master_seed=91))
    realization = state.realizations[0]
    assert "concatenate" not in inspect.getsource(service._generate_one)
    assert "concatenate" not in inspect.getsource(service._fill_interval)
    assert realization.fracture_count == len(realization.arrays["measured_depth"])
    assert all(len(array) == realization.fracture_count for array in realization.arrays.values())


def test_actual_poisson_count_is_budgeted_before_final_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    sampled_count = 100_000
    monkeypatch.setattr(HomogeneousPoissonSampler, "sample_count", lambda *_args, **_kwargs: sampled_count)
    allocation_called = False

    def fail_if_allocated(_count: int) -> dict[str, np.ndarray]:
        nonlocal allocation_called
        allocation_called = True
        raise AssertionError("final arrays must not be allocated before the actual-count budget check")

    monkeypatch.setattr(service, "_allocate_arrays", fail_if_allocated)
    config = BoreholeFractureGenerationConfig(number_of_sets=2, memory_budget_bytes=1024)
    with pytest.raises(MemoryError, match="Actual-count peak memory"):
        service.build_candidate(config)
    assert not allocation_called
    assert project.borehole_fracture_state.realizations == []


def test_actual_resource_estimate_owns_all_requested_realizations(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    sampled_count = 250
    monkeypatch.setattr(HomogeneousPoissonSampler, "sample_count", lambda *_args, **_kwargs: sampled_count)
    captured: list[dict[str, object]] = []
    original = service._resource_details

    def capture(config, realization_counts, **kwargs):
        details = original(config, realization_counts, **kwargs)
        captured.append({"counts": list(realization_counts), **details})
        return details

    monkeypatch.setattr(service, "_resource_details", capture)
    state = service.build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=2, realization_count=2, master_seed=19)
    )
    assert captured[-1]["estimate_basis"] == "actual_poisson_counts"
    assert captured[-1]["counts"] == [sampled_count * 2, sampled_count * 2]
    assert captured[-1]["expected_fractures"] == pytest.approx(130.0)
    assert captured[-1]["actual_fractures"] == sampled_count * 4
    assert captured[-1]["persistent_bytes"] == sampled_count * 4 * 42
    assert sum(item.fracture_count for item in state.realizations) == sampled_count * 4


def test_fitted_direction_uses_each_position_and_is_chunk_size_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _synthetic_project(include_random=True)
    repository = BoreholeRepository(project)
    for record in repository.query(BoreholeDataType.FRACTURES, RecordState.FORMAL):
        repository.edit_record(record.record_id, {**record.values, "fracture_spacing": 0.02})
    service = BoreholeFractureService(project)
    seen_targets: list[np.ndarray] = []
    original = service._directions_at

    def capture(idw, constraints, set_id, targets, fit):
        seen_targets.append(np.asarray(targets).copy())
        return original(idw, constraints, set_id, targets, fit)

    monkeypatch.setattr(service, "_directions_at", capture)
    base = BoreholeFractureGenerationConfig(
        number_of_sets=1,
        master_seed=731,
        direction_mode="SPATIALLY_FITTED_WITHIN_GLOBAL_SET",
        generation_chunk_size=256,
    )
    first = service.build_candidate(base)
    assert any(len(targets) > 1 and np.ptp(targets, axis=0).max() > 0 for targets in seen_targets)
    second = BoreholeFractureService(project).build_candidate(
        base.model_copy(update={"generation_chunk_size": 1024})
    )
    assert _array_hash(first) == _array_hash(second)
    assert first.realizations[0].generator_version == "borehole-phase2a-3"


def test_representative_kappa_status_and_stable_twenty_colour_palette() -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    one_per_group = service.fit_global_sets(3, 42)
    assert {item.kappa_status for item in one_per_group.sets} == {"UNRESOLVED"}
    mixed = service.fit_global_sets(2, 42)
    assert {item.kappa_status for item in mixed.sets} == {"UNRESOLVED", "SITE_MEAN_DISPERSION"}
    colours = [joint_set_color(set_id) for set_id in range(1, 21)]
    assert len(set(colours)) == 20
    assert colours == [joint_set_color(set_id) for set_id in range(1, 21)]


def test_spatial_direction_uses_available_representative_without_inventing_kappa() -> None:
    project = _synthetic_project()
    state = BoreholeFractureService(project).build_candidate(
        BoreholeFractureGenerationConfig(
            number_of_sets=3,
            master_seed=42,
            direction_mode="SPATIALLY_FITTED_WITHIN_GLOBAL_SET",
        )
    )
    assert state.realizations[0].fracture_count > 0


def test_two_far_generated_fractures_receive_distinct_local_direction_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _synthetic_project(include_random=False)
    service = BoreholeFractureService(project)
    captured_means: list[np.ndarray] = []
    original = service._directions_at

    monkeypatch.setattr(HomogeneousPoissonSampler, "sample_count", lambda *_args, **_kwargs: 2)

    def fixed_far_depths(_self, output, _rng, from_depth, to_depth, **_kwargs):
        output[:] = (from_depth, np.nextafter(to_depth, from_depth))

    monkeypatch.setattr(HomogeneousPoissonSampler, "fill_depths", fixed_far_depths)

    def capture(idw, constraints, set_id, targets, fit):
        means, valid = original(idw, constraints, set_id, targets, fit)
        captured_means.append(means.copy())
        return means, valid

    monkeypatch.setattr(service, "_directions_at", capture)
    state = service.build_candidate(
        BoreholeFractureGenerationConfig(
            number_of_sets=1,
            master_seed=123,
            direction_mode="SPATIALLY_FITTED_WITHIN_GLOBAL_SET",
        )
    )
    assert state.realizations[0].fracture_count == 4
    assert any(len(means) == 2 and not np.allclose(means[0], means[1]) for means in captured_means)


def test_fixed_group_mean_does_not_call_local_direction_idw(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("FIXED_GLOBAL_SET_MEAN must not use local direction IDW")

    monkeypatch.setattr(service, "_directions_at", unexpected)
    state = service.build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=2, direction_mode="FIXED_GLOBAL_SET_MEAN")
    )
    assert state.realizations[0].fracture_count > 0


def test_site_level_neighbors_expand_all_components_and_ignore_joint_num_for_intensity() -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    fit = service.fit_global_sets(2, 42)
    constraints = service._intensity_constraints(fit)
    constraints["search_mode"] = "RADIUS"
    target = np.asarray((10.0, 10.0, 90.0))
    support = service._local_probabilities(StableIDW(2.0, 1000.0, 1, 1), constraints, target, fit)
    assert support["selected_point_count"] == 1
    assert len(support["local_probabilities"]) == 2  # dominant + RANDOM at point A
    assert sum(support["local_probabilities"].values()) == pytest.approx(1.0)
    dominant_index = next(
        index
        for index, component in enumerate(fit.local_components)
        if component.observation_id == "P-A-1"
    )
    random_index = next(
        index
        for index, component in enumerate(fit.local_components)
        if component.observation_id == "P-A-R"
    )
    # joint_num=8 and 2 are diagnostic weights only; the probability ratio is reciprocal spacing.
    assert support["local_probabilities"][dominant_index] / support["local_probabilities"][random_index] == pytest.approx(2.5)


def test_all_sites_mode_records_long_range_extrapolation_distances() -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    fit = service.fit_global_sets(2, 42)
    state = service.build_candidate(
        BoreholeFractureGenerationConfig(
            number_of_sets=2,
            master_seed=42,
            idw_search_mode="ALL_WITHIN_DOMAIN",
            search_radius=0.01,
            max_neighbors=1,
        ),
        fit_override=fit,
    )
    for diagnostic in state.realizations[0].interval_diagnostics:
        assert diagnostic.selected_point_count == 2
        assert diagnostic.nearest_constraint_distance is not None
        assert diagnostic.farthest_constraint_distance is not None
        assert diagnostic.farthest_constraint_distance >= diagnostic.nearest_constraint_distance
        assert diagnostic.is_long_range_extrapolation


def test_component_probabilities_aggregate_to_global_sets_and_persist_in_rows() -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    state = service.build_candidate(BoreholeFractureGenerationConfig(number_of_sets=2, master_seed=314))
    realization = state.realizations[0]
    fit = state.global_fit
    assert fit is not None
    indices = realization.arrays["local_component_index"]
    assert indices.dtype == np.int32
    assert np.all((indices >= 0) & (indices < len(fit.local_components)))
    for diagnostic in realization.interval_diagnostics:
        assert sum(diagnostic.local_component_probabilities.values()) == pytest.approx(1.0)
        aggregate: dict[int, float] = {}
        for component_index, probability in diagnostic.local_component_probabilities.items():
            component = fit.local_components[component_index]
            if component.global_set_id is not None:
                aggregate[component.global_set_id] = aggregate.get(component.global_set_id, 0.0) + probability
        assert aggregate == pytest.approx(diagnostic.global_set_probabilities)


def test_poisson_total_stream_is_independent_of_local_component_assignment() -> None:
    project = _synthetic_project(include_random=False)
    config = BoreholeFractureGenerationConfig(number_of_sets=2, master_seed=867)
    service = BoreholeFractureService(project)
    state = service.build_candidate(config)
    intervals = service.observations.spacing_observations()
    expected = sum(
        HomogeneousPoissonSampler().sample_count(
            config.rng_for(0, index, 1), item.from_depth, item.to_depth, 1.0 / item.derived_p10
        )
        for index, item in enumerate(intervals)
    )
    assert state.realizations[0].fracture_count == expected


def test_fifteen_directional_components_keep_two_level_roles() -> None:
    project = _synthetic_project()
    repository = BoreholeRepository(project)
    for record in list(repository.query(BoreholeDataType.ORIENTATION_POINTS)):
        repository.delete_record(record.record_id)
    rows = []
    directions = ((15.0, 30.0), (130.0, 55.0), (255.0, 70.0))
    component_number = 0
    for point_index, count in enumerate((4, 4, 3)):
        point_id = f"SITE-{point_index + 1}"
        xyz = (10.0 + 20.0 * point_index, 15.0, 85.0 - 5.0 * point_index)
        for local_index in range(count):
            component_number += 1
            base_dd, base_dip = directions[local_index % 3]
            rows.append(
                {
                    "observation_id": f"P-S{point_index + 1}-{local_index + 1}",
                    "point_id": point_id,
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                    "dip": base_dip + point_index,
                    "dip_direction": (base_dd + point_index) % 360,
                    "local_set_id": f"LOCAL-{local_index + 1}",
                    "joint_spacing_m": 0.5 + 0.1 * local_index,
                    "joint_num": 2 + local_index,
                }
            )
        rows.append(
            {
                "observation_id": f"P-S{point_index + 1}-R",
                "point_id": point_id,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "dip": None,
                "dip_direction": None,
                "local_set_id": "RANDOM",
                "joint_spacing_m": 2.0,
                "joint_num": 1,
            }
        )
    for camera_index in range(4):
        base_dd, base_dip = directions[camera_index % 3]
        rows.append(
            {
                "observation_id": f"Z-C{camera_index + 1}-1",
                "point_id": f"CAM-{camera_index + 1}",
                "x": 12.0 + camera_index,
                "y": 18.0,
                "z": 80.0,
                "dip": base_dip + 0.5,
                "dip_direction": base_dd + 0.5,
                "local_set_id": "CAMERA-LOCAL",
                "joint_spacing_m": None,
                "joint_num": None,
            }
        )
    result = repository.import_dataframe(
        BoreholeDataType.ORIENTATION_POINTS,
        pd.DataFrame(rows),
        "synthetic-two-level-components.csv",
    )
    assert result["formal"] == 18
    service = BoreholeFractureService(project)
    fit = service.fit_global_sets(3, 42)
    constraints = service._intensity_constraints(fit)
    assert len(fit.mappings) == 15
    assert len(fit.local_components) == 18
    assert sum(item.source_kind == "POINT_CLOUD" and item.component_type == "LOCAL_DOMINANT_SET" for item in fit.local_components) == 11
    assert sum(item.source_kind == "BOREHOLE_CAMERA" for item in fit.local_components) == 4
    assert sum(item.component_type == "RANDOM_BACKGROUND" for item in fit.local_components) == 3
    assert len(constraints["p_sites"]) == 3
    assert sum(len(site["components"]) for site in constraints["p_sites"]) == 14
    assert sum(len(values) for values in constraints["directions"].values()) == 15
    support = service._local_probabilities(
        StableIDW(2.0, 1_000.0, 1, 1), constraints, np.asarray((10.0, 15.0, 85.0)), fit
    )
    assert support["selected_point_count"] == 1
    assert len(support["local_probabilities"]) == 5
    assert sum(support["local_probabilities"].values()) == pytest.approx(1.0)


def test_late_generation_exception_leaves_no_partial_realization(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    calls = 0
    original = service._fill_interval

    def fail_in_second_realization(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic generation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_fill_interval", fail_in_second_realization)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        service.build_candidate(BoreholeFractureGenerationConfig(number_of_sets=2, realization_count=2))
    assert project.borehole_fracture_state.realizations == []


def test_changed_input_invalidates_realizations_but_duplicate_and_transaction_rollback_do_not() -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    service.commit(service.build_candidate(BoreholeFractureGenerationConfig(number_of_sets=2)))
    original_state = project.borehole_fracture_state
    repository = BoreholeRepository(project)
    duplicate = repository.import_dataframe(
        BoreholeDataType.FRACTURES,
        pd.DataFrame(
            [{"hole_id": "SYN-A", "from_depth": 0.0, "to_depth": 20.0, "fracture_spacing": 0.5}]
        ),
        "synthetic-spacing.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "m"},
    )
    assert duplicate["duplicates"] == 1
    assert project.borehole_fracture_state is original_state

    with pytest.raises(RuntimeError):
        with repository.transaction():
            record = repository.query(BoreholeDataType.SURVEYS, RecordState.FORMAL)[1]
            repository.edit_record(record.record_id, {**record.values, "azimuth": 91.0})
            assert project.borehole_fracture_state.realizations == []
            raise RuntimeError("synthetic rollback")
    assert project.borehole_fracture_state is original_state

    record = repository.query(BoreholeDataType.SURVEYS, RecordState.FORMAL)[1]
    repository.edit_record(record.record_id, {**record.values, "azimuth": 91.0})
    assert project.borehole_fracture_state.realizations == []
