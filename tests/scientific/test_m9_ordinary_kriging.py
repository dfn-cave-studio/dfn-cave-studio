"""Scientific regression tests for three-dimensional ordinary kriging."""

from __future__ import annotations

import numpy as np
import pytest

from dfn_cave_studio.models.m9 import KrigingSettings, VariogramModel, VariogramMode
from dfn_cave_studio.voxel.ordinary_kriging import OrdinaryKrigingInterpolator, fit_variogram, semivariogram


@pytest.mark.parametrize("model", list(VariogramModel))
def test_semivariograms_start_at_zero_and_increase(model: VariogramModel) -> None:
    distances = np.asarray([0.0, 1.0, 5.0, 20.0])
    values = semivariogram(distances, model=model, nugget=0.1, sill=2.0, variogram_range=5.0)
    assert values[0] == 0.0
    assert np.all(np.diff(values) >= -1e-12)
    assert values[-1] <= 2.0 + 1e-12


def _manual_settings(**updates: object) -> KrigingSettings:
    values = {
        "mode": VariogramMode.MANUAL,
        "model": VariogramModel.SPHERICAL,
        "nugget": 0.0,
        "sill": 1.0,
        "range": 20.0,
        "minimum_neighbors": 2,
        "maximum_neighbors": 8,
    }
    values.update(updates)
    return KrigingSettings.model_validate(values)


def test_constant_field_is_unbiased_and_weights_sum_to_one() -> None:
    coordinates = np.asarray([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
    settings = _manual_settings()
    values = np.full(4, 7.5)
    interpolator = OrdinaryKrigingInterpolator(coordinates, values, fit_variogram(coordinates, values, settings), settings)
    prediction = interpolator.predict((2.0, 3.0, 4.0))
    assert prediction.status == "PREDICTED"
    assert prediction.estimate == pytest.approx(7.5)
    assert prediction.weights is not None and prediction.weights.sum() == pytest.approx(1.0)
    assert prediction.variance is not None and prediction.variance >= 0.0


def test_sample_location_is_reproduced_and_deterministic() -> None:
    coordinates = np.asarray([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    values = np.asarray([1.0, 3.0, 8.0])
    settings = _manual_settings()
    diagnostics = fit_variogram(coordinates, values, settings)
    first = OrdinaryKrigingInterpolator(coordinates, values, diagnostics, settings).predict((10.0, 0.0, 0.0))
    second = OrdinaryKrigingInterpolator(coordinates, values, diagnostics, settings).predict((10.0, 0.0, 0.0))
    assert first.estimate == second.estimate
    assert first.variance == second.variance
    assert first.weights is not None and second.weights is not None
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.estimate == 3.0
    assert first.variance == 0.0


def test_duplicate_coordinates_are_aggregated_without_singular_failure() -> None:
    coordinates = np.asarray([[0, 0, 0], [0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    values = np.asarray([2.0, 4.0, 5.0, 7.0])
    settings = _manual_settings()
    interpolator = OrdinaryKrigingInterpolator(coordinates, values, fit_variogram(coordinates, values, settings), settings)
    assert len(interpolator.coordinates) == 3
    assert interpolator.predict((0.0, 0.0, 0.0)).estimate == pytest.approx(3.0)


def test_singular_solver_path_records_regularization_and_pseudoinverse(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinates = np.asarray([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    values = np.asarray([1.0, 2.0, 4.0])
    settings = _manual_settings(regularization=0.0)
    diagnostics = fit_variogram(coordinates, values, settings)
    model = OrdinaryKrigingInterpolator(coordinates, values, diagnostics, settings)
    monkeypatch.setattr(np.linalg, "solve", lambda *_args, **_kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError()))
    prediction = model.predict((2.0, 2.0, 0.0))
    assert prediction.estimate is not None
    assert prediction.used_regularization is True
    assert prediction.used_pseudoinverse is True
    assert diagnostics.regularized_solve_count == 1
    assert diagnostics.pseudoinverse_solve_count == 1


def test_insufficient_data_has_explicit_status() -> None:
    coordinates = np.asarray([[0.0, 0.0, 0.0]])
    values = np.asarray([2.0])
    settings = _manual_settings(minimum_neighbors=2)
    interpolator = OrdinaryKrigingInterpolator(coordinates, values, fit_variogram(coordinates, values, settings), settings)
    result = interpolator.predict((1.0, 0.0, 0.0))
    assert result.status == "INSUFFICIENT_DATA"
    assert result.estimate is None


def test_auto_variogram_records_lags_and_fit_diagnostics() -> None:
    coordinates = np.asarray([[float(index), 0.0, 0.0] for index in range(8)])
    values = np.asarray([0.2, 0.3, 0.7, 1.1, 1.0, 1.4, 1.8, 2.0])
    settings = KrigingSettings(mode=VariogramMode.AUTO, lag_count=4, minimum_neighbors=2)
    diagnostics = fit_variogram(coordinates, values, settings)
    assert diagnostics.sample_count == 8
    assert diagnostics.pair_count > 0
    assert diagnostics.lags
    assert diagnostics.fit_status in {"converged", "failed"}


def test_batched_predictions_match_scalar_contract() -> None:
    coordinates = np.asarray([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
    values = np.asarray([1.0, 2.0, 3.0, 4.0]); settings = _manual_settings()
    model = OrdinaryKrigingInterpolator(coordinates, values, fit_variogram(coordinates, values, settings), settings)
    points = np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [0.0, 0.0, 0.0]])
    estimates, variances, counts, statuses = model.predict_many(points)
    for index, point in enumerate(points):
        scalar = model.predict(point)
        assert estimates[index] == pytest.approx(scalar.estimate)
        assert variances[index] == pytest.approx(scalar.variance)
        assert counts[index] == scalar.neighbor_count
        assert statuses[index] == scalar.status


def test_equidistant_neighbor_limit_uses_original_index_as_stable_tie_break() -> None:
    coordinates = np.asarray(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float
    )
    values = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    settings = _manual_settings(minimum_neighbors=2, maximum_neighbors=3)
    diagnostics = fit_variogram(coordinates, values, settings)
    model = OrdinaryKrigingInterpolator(coordinates, values, diagnostics, settings)
    first = model.predict((0.0, 0.0, 0.0))
    second = model.predict((0.0, 0.0, 0.0))
    assert first.estimate == second.estimate
    reference_coordinates = coordinates[:3]
    reference_values = values[:3]
    reference = OrdinaryKrigingInterpolator(
        reference_coordinates,
        reference_values,
        fit_variogram(reference_coordinates, reference_values, settings),
        settings,
    ).predict((0.0, 0.0, 0.0))
    assert first.estimate == pytest.approx(reference.estimate)
    batched = model.predict_many(np.asarray([[0.0, 0.0, 0.0]]))[0][0]
    assert batched == pytest.approx(first.estimate)


def test_domain_models_do_not_share_samples() -> None:
    settings = _manual_settings(minimum_neighbors=2, maximum_neighbors=4)
    left_xyz = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    right_xyz = left_xyz + np.asarray([100, 0, 0])
    left = OrdinaryKrigingInterpolator(left_xyz, np.ones(4), fit_variogram(left_xyz, np.ones(4), settings), settings)
    right = OrdinaryKrigingInterpolator(right_xyz, np.full(4, 100.0), fit_variogram(right_xyz, np.full(4, 100.0), settings), settings)
    assert left.predict((0.5, 0.5, 0)).estimate == pytest.approx(1.0)
    assert right.predict((100.5, 0.5, 0)).estimate == pytest.approx(100.0)


@pytest.mark.parametrize(
    "updates",
    ({"nugget": -1.0}, {"sill": 0.0}, {"range": 0.0}, {"nugget": 2.0, "sill": 1.0},
     {"minimum_neighbors": 8, "maximum_neighbors": 4}),
)
def test_invalid_variogram_parameters_are_rejected(updates: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _manual_settings(**updates)
