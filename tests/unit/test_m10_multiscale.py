"""Scientific regression tests for M10 multiscale radius budgeting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dfn_cave_studio.dfn.m10_generator import ConditionedObservation, M10ExplicitDFNGenerator
from dfn_cave_studio.dfn.m10_multiscale import (
    SizeClass,
    area_weighted_cdf,
    calculate_thresholds,
    classify_radius,
    sample_truncated_class,
)
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.m10 import DeterministicStructure, M10GenerationConfig
from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata, SizeModel, SizeModelSource
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES


def _uniform() -> SizeModel:
    return SizeModel(
        domain_id=1, set_id=1, distribution_type="uniform", parameters={}, min_radius=1.0,
        max_radius=4.0, mean_radius=2.5, mean_squared_radius=7.0,
        source=SizeModelSource.USER_DEFINED,
    )


def _generator(config: M10GenerationConfig) -> M10ExplicitDFNGenerator:
    shape = (2, 1, 1)
    metadata = ParameterFieldMetadata(
        shape=shape, origin=(0.0, 0.0, 0.0), spacing=(10.0, 10.0, 10.0),
        field_names=[], set_ids=[1], density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=1, estimated_bytes=0,
    )
    arrays = {
        "cell_state": np.full(shape, CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], dtype=np.uint8),
        "domain_id": np.ones(shape, dtype=np.int32),
        "set_1_p32": np.full(shape, 0.2, dtype=np.float32),
        "set_1_dip_direction": np.full(shape, 45.0, dtype=np.float32),
        "set_1_dip": np.full(shape, 60.0, dtype=np.float32),
        "set_1_kappa": np.full(shape, 20.0, dtype=np.float32),
    }
    return M10ExplicitDFNGenerator(
        metadata=metadata, arrays=arrays,
        generation_domain=ModelBounds(x_min=-5, x_max=25, y_min=-5, y_max=15, z_min=-5, z_max=15),
        size_models=[_uniform()], config=config, project_id="multiscale",
    )


def test_area_weighted_cdf_and_auto_thresholds_are_deterministic() -> None:
    model = _uniform()
    assert area_weighted_cdf(model, 2.0) == pytest.approx(7.0 / 63.0)
    first = calculate_thresholds(model)
    second = calculate_thresholds(model)
    assert first == second
    assert area_weighted_cdf(model, first.small_medium_radius) == pytest.approx(0.10)
    assert area_weighted_cdf(model, first.medium_large_radius) == pytest.approx(0.70)


def test_class_budgets_are_mutually_exclusive_and_conservative() -> None:
    result = calculate_thresholds(_uniform())
    assert sum(item.probability for item in result.budgets) == pytest.approx(1.0)
    assert sum(item.area_share for item in result.budgets) == pytest.approx(1.0)
    assert [item.area_share for item in result.budgets] == pytest.approx([0.1, 0.6, 0.3])
    assert classify_radius(result.small_medium_radius - 1e-9, result) == SizeClass.SMALL
    assert classify_radius(result.small_medium_radius, result) == SizeClass.MEDIUM
    assert classify_radius(result.medium_large_radius, result) == SizeClass.LARGE


def test_invalid_manual_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="0 <= r_sm < r_ml"):
        calculate_thresholds(
            _uniform(), mode="manual", manual_small_medium_radius=2.0, manual_medium_large_radius=2.0
        )


def test_fixed_radius_on_manual_boundary_obeys_half_open_class_rule() -> None:
    model = SizeModel(
        domain_id=1, set_id=1, distribution_type="fixed", parameters={"radius": 1.0},
        min_radius=1.0, max_radius=1.0, mean_radius=1.0, mean_squared_radius=1.0,
        source=SizeModelSource.ASSUMED,
    )
    result = calculate_thresholds(
        model, mode="manual", manual_small_medium_radius=1.0, manual_medium_large_radius=2.0
    )
    assert [item.probability for item in result.budgets] == [0.0, 1.0, 0.0]
    assert classify_radius(1.0, result) == SizeClass.MEDIUM


def test_conditional_samples_remain_inside_each_half_open_class() -> None:
    model = _uniform()
    thresholds = calculate_thresholds(model)
    rng = np.random.default_rng(4)
    for budget in thresholds.budgets:
        values = sample_truncated_class(model, budget, 200, rng)
        assert np.all(values >= max(model.min_radius, budget.lower))
        if math.isfinite(budget.upper):
            assert np.all(values <= budget.upper)


def test_default_omits_small_and_preserves_its_target_as_subgrid() -> None:
    generator = _generator(M10GenerationConfig(base_seed=42))
    result = generator.generate(0)
    assert result.geometry_arrays["size_class"].dtype == np.uint8
    assert not np.any(result.geometry_arrays["size_class"] == int(SizeClass.SMALL))
    assert np.any(result.geometry_arrays["p32_subgrid_set_1"] > 0.0)
    assert result.quality.p32_subgrid > 0.0
    assert result.quality.p32_target_conservation_error == pytest.approx(0.0, abs=1e-12)


def test_enabling_all_classes_removes_subgrid_and_is_seed_reproducible() -> None:
    config = M10GenerationConfig(base_seed=9, enabled_size_classes=["SMALL", "MEDIUM", "LARGE"])
    first = _generator(config).generate(0)
    second = _generator(config).generate(0)
    assert first.quality.p32_subgrid == pytest.approx(0.0)
    assert np.all(first.geometry_arrays["p32_subgrid_set_1"] == 0.0)
    for name in ("center", "normal", "radius", "size_class"):
        np.testing.assert_array_equal(first.geometry_arrays[name], second.geometry_arrays[name])


def test_threshold_calculation_does_not_consume_generation_rng() -> None:
    model = _uniform()
    before = np.random.default_rng(123).random(4)
    calculate_thresholds(model)
    after = np.random.default_rng(123).random(4)
    np.testing.assert_array_equal(before, after)


def test_deterministic_disk_bounds_relation_uses_full_disc_geometry() -> None:
    bounds = ModelBounds(x_min=0, x_max=10, y_min=0, y_max=10, z_min=0, z_max=10)
    normal = np.asarray((0.0, 0.0, 1.0))
    assert M10ExplicitDFNGenerator._disk_bounds_relation(np.asarray((5.0, 5.0, 5.0)), normal, 1.0, bounds) == "INSIDE"
    assert M10ExplicitDFNGenerator._disk_bounds_relation(np.asarray((11.0, 5.0, 5.0)), normal, 2.0, bounds) == "INTERSECTS"
    assert M10ExplicitDFNGenerator._disk_bounds_relation(np.asarray((20.0, 5.0, 5.0)), normal, 2.0, bounds) == "OUTSIDE"


def test_conditioned_and_deterministic_fractures_remain_explicit_when_random_classes_disabled() -> None:
    generator = _generator(M10GenerationConfig(enabled_size_classes=[]))
    generator.conditioned_observations = [
        ConditionedObservation(
            observation_record_id="obs-1", position=(5.0, 5.0, 5.0), dip_direction=45.0, dip=60.0,
            domain_id=1, set_id=1, hole_id="BH-1",
        )
    ]
    generator.deterministic_structures = [
        DeterministicStructure(
            structure_id="structure-1", center_x=5.0, center_y=5.0, center_z=5.0,
            dip_direction=120.0, dip=50.0, radius=1.0, structure_type="fault", domain_id=1, set_id=1,
        )
    ]
    realization = generator.generate(0)
    assert realization.quality.stochastic_count == 0
    assert realization.quality.conditioned_count == 1
    assert realization.quality.deterministic_count == 1
    assert np.all(realization.geometry_arrays["size_class"] > 0)
    assert realization.geometry_arrays["size_class"][1] == int(SizeClass.LARGE)


def test_fully_outside_deterministic_disc_is_audited_but_not_generated_by_default() -> None:
    generator = _generator(M10GenerationConfig(enabled_size_classes=[]))
    generator.deterministic_structures = [
        DeterministicStructure(
            structure_id="outside", center_x=100.0, center_y=100.0, center_z=100.0,
            dip_direction=0.0, dip=45.0, radius=1.0, structure_type="fault", set_id=1,
        )
    ]
    realization = generator.generate(0)
    assert realization.quality.deterministic_count == 0
    assert any("outside" in warning and "fully outside" in warning for warning in realization.quality.warnings)


def test_partial_invalid_orientation_keeps_all_budget_arrays_aligned() -> None:
    generator = _generator(M10GenerationConfig(base_seed=42))
    generator.arrays["set_1_dip_direction"][1, 0, 0] = np.nan
    realization = generator.generate(0)
    stochastic = realization.geometry_arrays["source_code"] == 0
    assert np.any(stochastic)
    assert np.all(realization.geometry_arrays["voxel_index"][stochastic, 0] == 0)
    subgrid = realization.geometry_arrays["p32_subgrid_set_1"]
    assert subgrid[0, 0, 0] > 0.0
    assert subgrid[1, 0, 0] == 0.0
    assert realization.quality.p32_unresolved_orientation > 0.0
    assert realization.quality.p32_target_conservation_error == pytest.approx(0.0, abs=1e-12)
    assert realization.quality.target_p32 == pytest.approx(
        realization.quality.p32_explicit_target
        + realization.quality.p32_subgrid
        + realization.quality.p32_unresolved_orientation
    )


def test_all_invalid_orientation_is_explicitly_unresolved_not_subgrid() -> None:
    generator = _generator(M10GenerationConfig(base_seed=42))
    generator.arrays["set_1_dip_direction"][:] = np.nan
    realization = generator.generate(0)
    assert realization.quality.stochastic_count == 0
    assert np.all(realization.geometry_arrays["p32_subgrid_set_1"] == 0.0)
    assert realization.quality.p32_explicit_target == 0.0
    assert realization.quality.p32_subgrid == 0.0
    assert realization.quality.p32_unresolved_orientation == pytest.approx(realization.quality.target_p32)
    assert realization.quality.domain_set[0].status == "INSUFFICIENT_ORIENTATION_DATA"
