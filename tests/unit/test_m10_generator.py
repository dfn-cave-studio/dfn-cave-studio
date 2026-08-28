"""Scientific and behavioral tests for M10 explicit DFN generation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dfn_cave_studio.dfn.m10_generator import ConditionedObservation, M10ExplicitDFNGenerator
from dfn_cave_studio.dfn.m10_geometry import (
    authoritative_nbytes,
    fracture_ids,
    migrate_legacy_geometry,
    record_id,
    source_mask,
)
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.m10 import DeterministicStructure, M10FractureSource, M10GenerationConfig
from dfn_cave_studio.models.m10 import M10Realization
from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata, SizeModel, SizeModelSource
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES
from dfn_cave_studio.models.spatial_grid import VoxelCellState


def _input(
    *,
    shape=(1, 1, 1),
    p32=0.2,
    state=VoxelCellState.MODELED_VALUE,
    size_model=None,
    conditioned=(),
    deterministic=(),
    seed=42,
):
    metadata = ParameterFieldMetadata(
        shape=shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(10.0, 10.0, 10.0),
        field_names=[],
        set_ids=[1],
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=1,
        estimated_bytes=0,
    )
    arrays = {
        "cell_state": np.full(shape, CELL_STATE_CODES[state], dtype=np.uint8),
        "domain_id": np.ones(shape, dtype=np.int32),
        "set_1_p32": np.full(shape, p32, dtype=np.float32),
        "set_1_dip_direction": np.full(shape, 45.0, dtype=np.float32),
        "set_1_dip": np.full(shape, 60.0, dtype=np.float32),
        "set_1_kappa": np.full(shape, 100.0, dtype=np.float32),
    }
    model = size_model or SizeModel(
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
    generator = M10ExplicitDFNGenerator(
        metadata=metadata,
        arrays=arrays,
        generation_domain=ModelBounds(x_min=-5, x_max=25, y_min=-5, y_max=25, z_min=-5, z_max=25),
        size_models=[model],
        config=M10GenerationConfig(base_seed=seed),
        project_id="test-project",
        conditioned_observations=conditioned,
        deterministic_structures=deterministic,
    )
    return generator


def test_expected_area_uses_pi_times_e_r_squared_not_squared_mean():
    model = SizeModel(
        domain_id=1,
        set_id=1,
        distribution_type="uniform",
        parameters={},
        min_radius=1.0,
        max_radius=3.0,
        mean_radius=2.0,
        mean_squared_radius=13.0 / 3.0,
        source=SizeModelSource.USER_DEFINED,
    )
    assert M10ExplicitDFNGenerator.expected_fracture_area(model) == pytest.approx(math.pi * 13.0 / 3.0)
    assert M10ExplicitDFNGenerator.expected_fracture_area(model) != pytest.approx(math.pi * model.mean_radius**2)


@pytest.mark.parametrize(
    "state",
    [VoxelCellState.TRUE_ZERO, VoxelCellState.NO_DATA, VoxelCellState.OUTSIDE_MODEL, VoxelCellState.EXCAVATION],
)
def test_non_modelled_and_true_zero_cells_generate_no_random_fractures(state):
    result = _input(state=state).generate(0)
    assert result.fracture_count == 0
    assert result.quality.stochastic_count == 0


def test_constant_p32_single_voxel_poisson_and_center_is_inside_cell():
    p32 = math.pi * 8.0 / 1000.0
    result = _input(p32=p32, seed=19).generate(0)
    expected_count = int(np.random.default_rng(19).poisson(8.0))
    assert result.quality.stochastic_count == expected_count
    centers = result.geometry_arrays["center"]
    assert np.all((centers >= 0.0) & (centers < 10.0))


def test_poisson_exactly_one_returns_one_complete_unit_normal_with_seed_42():
    result = _input(p32=math.pi / 1000.0, seed=42).generate(0)
    assert result.quality.stochastic_count == 1
    assert result.geometry_arrays["normal"].shape == (1, 3)
    assert np.all(np.isfinite(result.geometry_arrays["normal"]))
    assert np.linalg.norm(result.geometry_arrays["normal"][0]) == pytest.approx(1.0, abs=1e-12)
    assert len(set(fracture_ids(result))) == result.fracture_count


def test_seed_42_conditioning_then_single_stochastic_sample_succeeds():
    observation = ConditionedObservation(
        observation_record_id="cal-full-seed-42",
        position=(5.0, 5.0, 5.0),
        dip_direction=120.0,
        dip=55.0,
        domain_id=1,
        set_id=1,
        hole_id="CAL-01",
    )
    result = _input(p32=1.2 * math.pi / 1000.0, conditioned=[observation], seed=42).generate(0)
    assert result.quality.conditioned_count == 1
    assert result.quality.stochastic_count == 1
    assert result.geometry_arrays["normal"].shape == (2, 3)
    assert len(set(fracture_ids(result))) == result.fracture_count


def test_seed_42_deterministic_structure_and_single_stochastic_sample_succeed():
    structure = DeterministicStructure(
        structure_id="FAULT-SEED-42",
        center_x=5,
        center_y=5,
        center_z=5,
        dip_direction=80,
        dip=70,
        radius=2,
        structure_type="fault",
        domain_id=1,
    )
    result = _input(p32=math.pi / 1000.0, deterministic=[structure], seed=42).generate(0)
    assert result.quality.deterministic_count == 1
    assert result.quality.stochastic_count == 1
    assert result.geometry_arrays["normal"].shape == (2, 3)
    assert len(set(fracture_ids(result))) == result.fracture_count


def test_multiple_voxels_with_zero_one_and_many_poisson_samples_succeed():
    generator = _input(shape=(3, 1, 1), p32=0.0, seed=42)
    generator.arrays["cell_state"][0, 0, 0] = CELL_STATE_CODES[VoxelCellState.TRUE_ZERO]
    generator.arrays["set_1_p32"][:, 0, 0] = [0.0, math.pi / 1000.0, 100.0 * math.pi / 1000.0]
    result = generator.generate(0)
    voxel_indices = result.geometry_arrays["voxel_index"]
    assert not np.any(np.all(voxel_indices == (0, 0, 0), axis=1))
    assert np.count_nonzero(np.all(voxel_indices == (1, 0, 0), axis=1)) == 1
    assert np.count_nonzero(np.all(voxel_indices == (2, 0, 0), axis=1)) > 1
    assert result.geometry_arrays["normal"].shape == (result.fracture_count, 3)
    assert len(set(fracture_ids(result))) == result.fracture_count


def test_fixed_seed_is_identical_and_different_seed_changes_geometry():
    first = _input(p32=0.1, seed=42).generate(0)
    second = _input(p32=0.1, seed=42).generate(0)
    third = _input(p32=0.1, seed=43).generate(0)
    assert first.geometry_arrays.keys() == second.geometry_arrays.keys()
    for name in first.geometry_arrays:
        np.testing.assert_equal(first.geometry_arrays[name], second.geometry_arrays[name])
    assert not np.array_equal(first.geometry_arrays["center"], third.geometry_arrays["center"])
    assert len(set(fracture_ids(first))) == first.fracture_count


def test_conditioned_full_orientation_passes_observation_and_deducts_budget():
    observation = ConditionedObservation(
        observation_record_id="obs-1",
        position=(5.0, 5.0, 5.0),
        dip_direction=120.0,
        dip=55.0,
        domain_id=1,
        set_id=1,
        hole_id="CAL-01",
    )
    result = _input(p32=0.001, conditioned=[observation]).generate(0)
    assert result.quality.conditioned_count == 1
    assert result.quality.stochastic_count == 0
    assert result.quality.over_conditioned_regions
    index = int(np.flatnonzero(source_mask(result, M10FractureSource.CONDITIONED_OBSERVATION.value))[0])
    center = result.geometry_arrays["center"][index]
    normal = result.geometry_arrays["normal"][index]
    radius = result.geometry_arrays["radius"][index]
    delta = np.asarray(observation.position) - center
    assert abs(float(np.dot(delta, normal))) < 1e-10
    assert np.linalg.norm(delta) <= radius + 1e-10
    assert record_id(result, index) == "obs-1"


def test_deterministic_structure_geometry_does_not_depend_on_seed():
    structure = DeterministicStructure(
        structure_id="FAULT-1",
        center_x=5,
        center_y=5,
        center_z=5,
        dip_direction=80,
        dip=70,
        radius=2,
        structure_type="fault",
        domain_id=1,
    )
    first = _input(p32=0.0, deterministic=[structure], seed=1).generate(0)
    second = _input(p32=0.0, deterministic=[structure], seed=999).generate(0)
    np.testing.assert_allclose(first.geometry_arrays["center"], second.geometry_arrays["center"])
    np.testing.assert_allclose(first.geometry_arrays["normal"], second.geometry_arrays["normal"])
    np.testing.assert_allclose(first.geometry_arrays["radius"], second.geometry_arrays["radius"])
    assert first.quality.deterministic_count == 1


def test_deterministic_structure_preserves_real_set_id_and_missing_set_is_explicit():
    with_set = DeterministicStructure(
        structure_id="SET-3", center_x=5, center_y=5, center_z=5, dip_direction=80, dip=70,
        radius=2, structure_type="fault", domain_id=1, set_id=3,
    )
    without_set = with_set.model_copy(update={"structure_id": "NO-SET", "set_id": None})
    result = _input(p32=0.0, deterministic=[with_set, without_set]).generate(0)
    deterministic = source_mask(result, M10FractureSource.DETERMINISTIC_STRUCTURE.value)
    assert result.geometry_arrays["set_id"][deterministic].tolist() == [3, -1]


def test_columnar_geometry_omits_per_fracture_strings_and_intact_vertices():
    result = _input(p32=0.2).generate(0)
    forbidden = {"fracture_id", "source", "distribution", "size_source", "orientation_source", "provenance", "vertices"}
    assert forbidden.isdisjoint(result.geometry_arrays)
    assert result.geometry_arrays["source_code"].dtype == np.uint8
    assert result.geometry_arrays["clipped_points"].shape[1:] == (3,)
    assert authoritative_nbytes(result) < result.fracture_count * 160


def test_compact_final_storage_estimate_is_within_twenty_percent():
    generator = _input(shape=(20, 1, 1), p32=0.2)
    estimate = generator.estimate_details()
    result = generator.generate(0)
    actual = authoritative_nbytes(result)
    assert abs(estimate["final_storage_bytes"] - actual) / actual <= 0.20

    actual_subgrid = sum(
        array.nbytes for name, array in result.geometry_arrays.items() if name.startswith("p32_subgrid_set_")
    )
    actual_fracture_arrays = actual - actual_subgrid - result.geometry_arrays["clipped_points"].nbytes
    assert estimate["subgrid_array_bytes"] == actual_subgrid
    assert estimate["fracture_array_bytes"] == (
        math.ceil(estimate["expected_fractures"]) * estimate["bytes_per_unclipped_fracture"]
    )
    assert abs(estimate["fracture_array_bytes"] - actual_fracture_arrays) / actual_fracture_arrays <= 0.20
    assert estimate["final_storage_bytes"] == (
        estimate["fracture_array_bytes"]
        + estimate["subgrid_array_bytes"]
        + estimate["clipped_geometry_bytes"]
        + estimate["multiscale_metadata_bytes"]
    )


def test_conditioned_offset_consumes_exactly_two_uniform_draws_for_fixed_radius():
    observation = ConditionedObservation(
        observation_record_id="offset", position=(5.0, 5.0, 5.0), dip_direction=120.0, dip=55.0,
        domain_id=1, set_id=1, hole_id="CAL",
    )
    result = _input(p32=0.0, conditioned=[observation], seed=42).generate(0)
    index = int(np.flatnonzero(source_mask(result, M10FractureSource.CONDITIONED_OBSERVATION.value))[0])
    rng = np.random.default_rng(42)
    radius_draw, angle_draw = rng.random(2)
    normal = result.geometry_arrays["normal"][index]
    u, v = M10ExplicitDFNGenerator._plane_basis(normal)
    expected_offset = math.sqrt(radius_draw) * (math.cos(2 * math.pi * angle_draw) * u + math.sin(2 * math.pi * angle_draw) * v)
    np.testing.assert_allclose(np.asarray(observation.position) - result.geometry_arrays["center"][index], expected_offset)


def test_unreleased_legacy_geometry_migrates_without_retaining_intact_vertices():
    legacy = M10Realization(
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
    assert migrate_legacy_geometry(legacy)
    assert "vertices" not in legacy.geometry_arrays
    assert legacy.geometry_arrays["clipped_points"].shape == (0, 3)
    assert legacy.provenance["migrated_from"] == "m10-object-array-v1"


def test_columnar_v2_geometry_remains_fully_explicit_and_gets_unknown_size_class():
    realization = _input(p32=0.02).generate(0)
    realization.provenance["geometry_format"] = "columnar-v2"
    realization.geometry_arrays.pop("size_class")
    for name in [key for key in realization.geometry_arrays if key.startswith("p32_subgrid_set_")]:
        realization.geometry_arrays.pop(name)
    count = realization.fracture_count
    assert migrate_legacy_geometry(realization)
    np.testing.assert_array_equal(realization.geometry_arrays["size_class"], np.zeros(count, dtype=np.uint8))
    assert realization.provenance["size_class_status"] == "legacy_unknown_all_explicit"
    assert realization.provenance["p32_subgrid"] == 0.0


@pytest.mark.parametrize(
    ("distribution", "parameters"),
    [
        ("fixed", {"radius": 1.5}),
        ("uniform", {}),
        ("truncated_lognormal", {"mu": 0.2, "sigma": 0.4}),
        ("truncated_power_law", {"exponent": 2.5}),
        ("truncated_exponential", {"rate": 1.2}),
    ],
)
def test_supported_size_distributions_are_seeded_and_strictly_truncated(distribution, parameters):
    model = SizeModel(
        domain_id=1,
        set_id=1,
        distribution_type=distribution,
        parameters=parameters,
        min_radius=1.0,
        max_radius=2.0,
        mean_radius=1.5,
        mean_squared_radius=2.4,
        source=SizeModelSource.USER_DEFINED,
    )
    generator = _input(size_model=model)
    first = generator._sample_radii(model, 100, np.random.default_rng(7))
    second = generator._sample_radii(model, 100, np.random.default_rng(7))
    np.testing.assert_equal(first, second)
    assert np.all((first >= 1.0) & (first <= 2.0))


def test_generation_domain_clipping_records_original_and_clipped_area():
    generator = _input(p32=0.0)
    size = next(iter(generator.size_models.values()))
    fracture = generator._make_fracture(
        realization_id="test",
        ordinal=0,
        center=np.array([-4.9, 5.0, 5.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        radius=2.0,
        domain_id=1,
        set_id=1,
        source=M10FractureSource.STOCHASTIC,
        observation_record_id="",
        size=size,
        orientation_source="domain_set_fisher_model",
        dip_only_density_evidence=False,
        voxel_index=(0, 0, 0),
    )
    assert 0.0 < fracture.clipped_area < fracture.original_area
    assert np.all(fracture.vertices[:, 0] >= -5.0 - 1e-10)


def test_experimental_size_model_requires_explicit_confirmation():
    model = SizeModel(
        domain_id=1,
        set_id=1,
        distribution_type="fixed",
        parameters={"radius": 1.0},
        min_radius=1.0,
        max_radius=1.0,
        mean_radius=1.0,
        mean_squared_radius=1.0,
        source=SizeModelSource.EXPERIMENTAL,
        converged=True,
    )
    with pytest.raises(ValueError, match="explicit user confirmation"):
        _input(size_model=model)


def test_missing_reliable_orientation_skips_domain_set_with_explicit_status():
    generator = _input(p32=0.2)
    generator.arrays["set_1_dip_direction"][:] = np.nan
    result = generator.generate(0)
    assert result.quality.stochastic_count == 0
    assert result.quality.domain_set[0].status == "INSUFFICIENT_ORIENTATION_DATA"
    assert result.quality.skipped_regions[0]["reason"] == "INSUFFICIENT_ORIENTATION_DATA"
