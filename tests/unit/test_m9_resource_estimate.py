"""Resource-estimation and bounded batch-path tests for large M9 fields."""

from __future__ import annotations

import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.m9 import DensityMethod
from dfn_cave_studio.models.rock_mask import ExcavationMask, RockMask
from dfn_cave_studio.services.m9_service import _NearestDomainClassifier
from dfn_cave_studio.voxel.parameter_field import IDWInterpolator, SpatialSample
from dfn_cave_studio.voxel.resource_estimate import (
    estimate_parameter_field_resources,
    estimate_scalar_field_resources,
    format_resource_details,
    format_resource_estimate,
)


def _bounds() -> ModelBounds:
    return ModelBounds(x_min=0, x_max=3, y_min=0, y_max=2, z_min=0, z_max=2)


def test_parameter_estimate_matches_actual_small_contract_allocations() -> None:
    estimate = estimate_parameter_field_resources(
        _bounds(), VoxelConfig(), [1, 2, 4, 7, 9, 12, 15], DensityMethod.ORDINARY_KRIGING,
        chunk_size=5,
    )
    actual = {item.name: np.empty(item.shape, dtype=item.dtype) for item in estimate.arrays}
    assert estimate.grid_shape == (3, 2, 2)
    assert estimate.voxel_count == 12
    assert estimate.persistent_bytes == sum(array.nbytes for array in actual.values())
    assert len(actual) == 108
    assert estimate.temporary_arrays[0].shape == (5,)
    temporary_names = {item.name for item in estimate.temporary_arrays}
    assert {
        "domain_kdtree_query_distances",
        "domain_kdtree_query_indices",
        "domain_kdtree_tie_rows",
    } <= temporary_names


def test_method_specific_contract_and_worker_memory_are_explicit() -> None:
    global_estimate = estimate_parameter_field_resources(
        _bounds(), VoxelConfig(), range(7), DensityMethod.GLOBAL_CONSTANT, chunk_size=4
    )
    kriging_estimate = estimate_parameter_field_resources(
        _bounds(), VoxelConfig(), range(7), DensityMethod.ORDINARY_KRIGING, chunk_size=4
    )
    assert kriging_estimate.persistent_bytes - global_estimate.persistent_bytes == 8 * 12 * 4
    assert list(global_estimate.worker_extra_bytes) == [1, 2, 4, 8]
    assert global_estimate.worker_extra_bytes[8] == 8 * global_estimate.temporary_bytes


def test_scalar_estimate_matches_actual_array_nbytes_and_budget() -> None:
    estimate = estimate_scalar_field_resources(
        _bounds(), VoxelConfig(), chunk_size=3, budget_bytes=1
    )
    arrays = [np.empty(item.shape, dtype=item.dtype) for item in estimate.arrays]
    assert estimate.persistent_bytes == sum(array.nbytes for array in arrays)
    assert estimate.exceeds_budget


def test_very_large_grid_is_estimated_without_allocating_voxel_arrays() -> None:
    estimate = estimate_parameter_field_resources(
        ModelBounds(x_min=0, x_max=1000, y_min=0, y_max=1000, z_min=0, z_max=100),
        VoxelConfig(), range(7), DensityMethod.ORDINARY_KRIGING, chunk_size=8192,
    )
    assert estimate.voxel_count == 100_000_000
    assert estimate.persistent_bytes > 4 * 1024**3
    assert max(item.nbytes for item in estimate.temporary_arrays) <= 7 * 8192 * 8 * 4


def test_mask_batch_paths_equal_single_point_semantics() -> None:
    points = np.asarray([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
    rock = RockMask(x_min=0, x_max=3, y_min=0, y_max=3, z_min=0, z_max=3)
    excavation = ExcavationMask(excavation_boxes=[(1, 2.5, 1, 2.5, 1, 2.5)])
    np.testing.assert_array_equal(rock.contains_points(points), [rock.contains_point(*point) for point in points])
    np.testing.assert_array_equal(
        excavation.contains_points(points), [excavation.is_excavated(*point) for point in points]
    )


def test_idw_batch_matches_single_predictions() -> None:
    model = IDWInterpolator(
        [SpatialSample(0, 0, 0, 1, 3), SpatialSample(2, 0, 0, 3, 3), SpatialSample(0, 2, 0, 5, 3)],
        min_neighbors=1, max_neighbors=3,
    )
    points = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [3.0, 3.0, 1.0]])
    batch = model.predict_many(points, 3)
    single = [model.predict(tuple(point), 3) for point in points]
    np.testing.assert_array_equal(batch.values, [item.value for item in single])
    np.testing.assert_array_equal(batch.neighbor_counts, [item.neighbor_count for item in single])
    np.testing.assert_array_equal(batch.confidences, [item.confidence for item in single])


def test_nearest_domain_classifier_kdtree_matches_single_and_first_tie_rule() -> None:
    classifier = _NearestDomainClassifier(
        np.asarray([[0, 0, 0], [2, 0, 0], [0, 0, 0], [0, 2, 0]], dtype=float),
        (10, 20, 30, 40),
    )
    points = np.asarray([[0.1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1.9, 0]], dtype=float)
    batch = classifier.predict_many(points)
    assert batch.tolist() == [10, 10, 10, 40]
    assert batch.tolist() == [classifier(tuple(point)) for point in points]


def test_nearest_domain_classifier_does_not_build_broadcast_distance_matrix(monkeypatch) -> None:
    classifier = _NearestDomainClassifier(
        np.column_stack((np.arange(1000, dtype=float), np.zeros(1000), np.zeros(1000))),
        tuple(range(1000)),
    )
    original_norm = np.linalg.norm

    def guarded_norm(values, *args, **kwargs):
        assert np.asarray(values).ndim <= 2
        return original_norm(values, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", guarded_norm)
    result = classifier.predict_many(np.asarray([[0.25, 0, 0], [998.75, 0, 0]], dtype=float))
    assert result.tolist() == [0, 999]


def test_idw_batch_resolves_more_boundary_ties_than_max_neighbors() -> None:
    samples = [
        SpatialSample(1, 0, 0, 1, 1),
        SpatialSample(-1, 0, 0, 2, 1),
        SpatialSample(0, 1, 0, 4, 1),
        SpatialSample(0, -1, 0, 8, 1),
        SpatialSample(0, 0, 1, 16, 1),
        SpatialSample(0, 0, -1, 32, 1),
    ]
    model = IDWInterpolator(samples, min_neighbors=1, max_neighbors=3)
    points = np.asarray([[0, 0, 0], [0, 0, 0]], dtype=float)
    batch = model.predict_many(points, 1)
    single = [model.predict(tuple(point), 1) for point in points]
    np.testing.assert_array_equal(batch.values, [item.value for item in single])
    np.testing.assert_array_equal(batch.nearest_distances, [item.nearest_distance for item in single])
    np.testing.assert_array_equal(batch.neighbor_counts, [item.neighbor_count for item in single])
    np.testing.assert_array_equal(batch.confidences, [item.confidence for item in single])
    assert batch.values == pytest.approx([(1 + 2 + 4) / 3] * 2)


def test_idw_regular_batch_matches_single_without_boundary_slow_path() -> None:
    samples = [
        SpatialSample(0, 0, 0, 2, 1),
        SpatialSample(2, 0, 0, 4, 1),
        SpatialSample(0, 3, 0, 8, 1),
        SpatialSample(0, 0, 5, 16, 1),
        SpatialSample(7, 1, 2, 32, 1),
    ]
    model = IDWInterpolator(samples, min_neighbors=2, max_neighbors=3, search_radius=10.0)

    class _NoBoundaryQuery:
        def __init__(self, tree) -> None:
            self._tree = tree

        def query(self, *args, **kwargs):
            return self._tree.query(*args, **kwargs)

        def query_ball_point(self, *_args, **_kwargs):
            raise AssertionError("ordinary rows must not use the stable boundary slow path")

    model._domain_trees[1] = _NoBoundaryQuery(model._domain_trees[1])
    points = np.asarray([[0.2, 0.4, 0.7], [1.1, 1.3, 1.7], [3.4, 2.2, 0.8]], dtype=float)
    batch = model.predict_many(points, 1)
    single = [model.predict(tuple(point), 1) for point in points]
    np.testing.assert_array_equal(batch.values, [item.value for item in single])
    np.testing.assert_array_equal(batch.nearest_distances, [item.nearest_distance for item in single])
    np.testing.assert_array_equal(batch.neighbor_counts, [item.neighbor_count for item in single])
    np.testing.assert_array_equal(batch.confidences, [item.confidence for item in single])


def test_resource_summary_is_compact_and_details_retain_every_array() -> None:
    estimate = estimate_parameter_field_resources(
        _bounds(), VoxelConfig(), range(7), DensityMethod.ORDINARY_KRIGING, chunk_size=4
    )
    summary = format_resource_estimate(estimate)
    details = format_resource_details(estimate)
    assert summary.count("\n") == 3
    assert "cell_state" not in summary
    assert "cell_state" in details
    assert all(item.name in details for item in estimate.arrays)
    assert all(item.name in details for item in estimate.temporary_arrays)
