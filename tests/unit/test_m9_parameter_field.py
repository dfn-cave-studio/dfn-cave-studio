"""Unit tests for domain-safe M9 interpolation and field state semantics."""

import hashlib
import inspect

import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, KrigingSettings, P10Interval, P32Estimate, SizeModel
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.voxel.parameter_field import (
    CELL_STATE_CODES,
    IDWInterpolator,
    InterpolationResult,
    ParameterFieldBuilder,
    SpatialSample,
)


def test_idw_exact_point_domain_isolation_and_anisotropy():
    samples = [SpatialSample(0, 0, 0, 2, 1), SpatialSample(10, 0, 0, 8, 1), SpatialSample(0, 0, 0, 99, 2)]
    model = IDWInterpolator(samples, min_neighbors=1, anisotropy=(1, 10, 1))
    assert model.predict((0, 0, 0), 1).value == 2
    assert model.predict((0, 0, 0), 2).value == 99
    assert model.predict((0, 0, 0), 3).state == VoxelCellState.NO_DATA
    isotropic = IDWInterpolator([SpatialSample(10, 0, 0, 10, 1), SpatialSample(0, 10, 0, 0, 1)])
    anisotropic = IDWInterpolator(
        [SpatialSample(10, 0, 0, 10, 1), SpatialSample(0, 10, 0, 0, 0, 1)], anisotropy=(10, 1, 1)
    )
    assert anisotropic.predict((0, 0, 0), 1).value > isotropic.predict((0, 0, 0), 1).value


def test_parameter_field_total_and_cell_states():
    bounds = ModelBounds(x_min=0, x_max=3, y_min=0, y_max=1, z_min=0, z_max=1)
    settings = DensitySettings(method=DensityMethod.GLOBAL_CONSTANT)
    estimates = [
        P32Estimate(domain_id=1, set_id=1, fracture_count=1, raw_sample_length=1, effective_sample_length=1, mean_exposure=1, p32=0, observability="adequate", random_seed=1),
        P32Estimate(domain_id=1, set_id=2, fracture_count=2, raw_sample_length=1, effective_sample_length=1, mean_exposure=1, p32=2, observability="adequate", random_seed=1),
    ]
    sizes = [SizeModel(domain_id=1, set_id=1), SizeModel(domain_id=1, set_id=2)]
    metadata, arrays = ParameterFieldBuilder().build(
        bounds,
        VoxelConfig(cell_size_x=1, cell_size_y=1, cell_size_z=1),
        settings,
        [],
        estimates,
        [JointSetConfig(set_id=1), JointSetConfig(set_id=2)],
        sizes,
        random_seed=1,
        domain_at_point=lambda point: 1 if point[0] < 1 or point[0] >= 2 else 2,
        inside_model=lambda point: point[0] < 2,
    )
    assert metadata.shape == (3, 1, 1)
    assert ParameterFieldBuilder.estimate_bytes(bounds, VoxelConfig(), 2) == sum(array.nbytes for array in arrays.values())
    assert arrays["p32_total"][0, 0, 0] == arrays["set_1_p32"][0, 0, 0] + arrays["set_2_p32"][0, 0, 0]
    assert arrays["cell_state"][0, 0, 0] == CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
    assert arrays["cell_state"][1, 0, 0] == CELL_STATE_CODES[VoxelCellState.NO_DATA]
    assert arrays["cell_state"][2, 0, 0] == CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL]

    _, zero_arrays = ParameterFieldBuilder().build(
        ModelBounds(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1),
        VoxelConfig(),
        settings,
        [],
        [estimates[0]],
        [JointSetConfig(set_id=1)],
        [sizes[0]],
        random_seed=1,
        domain_at_point=lambda point: 1,
    )
    assert zero_arrays["cell_state"][0, 0, 0] == CELL_STATE_CODES[VoxelCellState.TRUE_ZERO]


def test_parameter_field_cancellation_and_memory_estimate():
    bounds = ModelBounds(x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=100)
    estimated = ParameterFieldBuilder.estimate_bytes(bounds, VoxelConfig(), 3)
    assert estimated > 100 * 100 * 100 * 4
    try:
        ParameterFieldBuilder().build(
            bounds, VoxelConfig(), DensitySettings(), [], [], [], [], random_seed=1, cancelled=lambda: True
        )
    except InterruptedError:
        pass
    else:
        raise AssertionError("cancel request was ignored")


def test_idw_without_fallback_returns_no_data_outside_search_radius():
    """A finite search with no neighbours stays NO_DATA when fallback is disabled."""
    model = IDWInterpolator(
        [SpatialSample(0, 0, 0, 2, 1)],
        search_radius=1.0,
        min_neighbors=1,
        fallback_by_domain=None,
    )
    result = model.predict((10, 0, 0), 1)
    assert result.value is None
    assert result.state == VoxelCellState.NO_DATA
    assert result.provenance == "no_data"


def test_parameter_field_builds_all_seven_and_non_contiguous_joint_sets():
    """M9 array creation derives names from actual IDs rather than three slots."""
    set_ids = [1, 2, 4, 7, 9, 12, 15]
    estimates = [
        P32Estimate(
            domain_id=1,
            set_id=set_id,
            fracture_count=4,
            raw_sample_length=10,
            effective_sample_length=8,
            mean_exposure=0.8,
            p32=0.01 * set_id,
            observability="adequate",
            random_seed=42,
        )
        for set_id in set_ids
    ]
    metadata, arrays = ParameterFieldBuilder().build(
        ModelBounds(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1),
        VoxelConfig(),
        DensitySettings(method=DensityMethod.GLOBAL_CONSTANT),
        [],
        estimates,
        [JointSetConfig(set_id=set_id) for set_id in set_ids],
        [SizeModel(domain_id=1, set_id=set_id) for set_id in set_ids],
        random_seed=42,
        domain_at_point=lambda _point: 1,
    )

    assert metadata.set_ids == set_ids
    assert all(f"set_{set_id}_p32" in arrays for set_id in set_ids)
    assert all(arrays[f"set_{set_id}_p32"][0, 0, 0] > 0.0 for set_id in set_ids)
    assert arrays["p32_total"][0, 0, 0] == sum(
        arrays[f"set_{set_id}_p32"][0, 0, 0] for set_id in set_ids
    )


def test_p32_ordinary_kriging_adds_variance_without_changing_idw_path():
    intervals = [
        P10Interval(hole_id=f"H{index}", from_depth=0, to_depth=1, domain_id=1, set_id=1,
                    observation_count=index + 1, sample_length=1, p10=float(index + 1), role="calibration",
                    center_x=float(x), center_y=float(y), center_z=0.5,
                    segment_directions=[(0.0, 0.0, -1.0, 1.0)])
        for index, (x, y) in enumerate(((0, 0), (2, 0), (0, 2), (2, 2)))
    ]
    settings = DensitySettings(
        method=DensityMethod.ORDINARY_KRIGING,
        monte_carlo_samples=100,
        kriging=KrigingSettings(mode="manual", nugget=0, sill=10, range=10,
                                minimum_neighbors=2, maximum_neighbors=4),
    )
    metadata, arrays = ParameterFieldBuilder().build(
        ModelBounds(x_min=0, x_max=2, y_min=0, y_max=2, z_min=0, z_max=1),
        VoxelConfig(cell_size_x=1, cell_size_y=1, cell_size_z=1), settings, intervals,
        [P32Estimate(domain_id=1, set_id=1, fracture_count=4, raw_sample_length=4,
                     effective_sample_length=4, mean_exposure=1, p32=2, observability="adequate", random_seed=42)],
        [JointSetConfig(set_id=1)], [SizeModel(domain_id=1, set_id=1)], random_seed=42,
        domain_at_point=lambda _point: 1,
    )
    assert np.isfinite(arrays["set_1_p32"]).all()
    assert np.isfinite(arrays["set_1_kriging_variance"]).all()
    assert (arrays["set_1_kriging_variance"] >= 0).all()
    assert "inter-set independence assumption" in metadata.provenance["kriging_variance_total_semantics"]
    audit = metadata.provenance["negative_prediction_audit"]
    assert audit["pre_adjustment_minimum"] == pytest.approx(float(np.min(arrays["set_1_p32"])))
    assert audit["pre_adjustment_maximum"] == pytest.approx(float(np.max(arrays["set_1_p32"])))
    assert audit["clipped_total_change"] == 0.0


def test_p32_reject_policy_audits_rejected_voxels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dfn_cave_studio.voxel.parameter_field.expected_orientation_exposure", lambda *_args, **_kwargs: 1.0)
    coordinates = np.asarray([
        [0.8025662504, 0.3957048590, 1.6810347667], [1.5436928907, 0.1426744999, 1.0196045744],
        [1.8575246436, 0.4696493671, 1.3012862086], [0.8860408773, 1.6175430478, 0.4499217102],
        [1.2840687489, 1.1536178579, 0.1364901646], [1.8608036851, 0.1657631769, 1.3273655112],
    ])
    values = [0.6203434192, 0.0979508128, 0.0361064494, 0.0805884798, 0.0146559873, 0.6837179537]
    target = np.asarray([1.7532164540, 1.2232780340, 0.7990488412])
    intervals = [
        P10Interval(
            hole_id=f"H{index}", from_depth=0, to_depth=1, domain_id=1, set_id=1,
            observation_count=1, sample_length=1, p10=value, role="calibration",
            center_x=float(point[0]), center_y=float(point[1]), center_z=float(point[2]),
            segment_directions=[(0.0, 0.0, -1.0, 1.0)],
        )
        for index, (point, value) in enumerate(zip(coordinates, values, strict=True))
    ]
    settings = DensitySettings(
        method=DensityMethod.ORDINARY_KRIGING,
        monte_carlo_samples=100,
        kriging=KrigingSettings(mode="manual", nugget=0, sill=1, range=2,
                                minimum_neighbors=2, maximum_neighbors=6,
                                non_negative_policy="reject"),
    )
    bounds = ModelBounds(x_min=target[0] - 0.5, x_max=target[0] + 0.5,
                         y_min=target[1] - 0.5, y_max=target[1] + 0.5,
                         z_min=target[2] - 0.5, z_max=target[2] + 0.5)
    metadata, arrays = ParameterFieldBuilder().build(
        bounds, VoxelConfig(), settings, intervals,
        [P32Estimate(domain_id=1, set_id=1, fracture_count=6, raw_sample_length=6,
                     effective_sample_length=6, mean_exposure=1, p32=1, observability="adequate", random_seed=42)],
        [JointSetConfig(set_id=1)], [SizeModel(domain_id=1, set_id=1)], random_seed=42,
        domain_at_point=lambda _point: 1,
    )
    audit = metadata.provenance["negative_prediction_audit"]
    assert audit["rejected_voxel_count"] == 1
    assert audit["rejected_set_voxel_count"] == 1
    assert audit["clipped_voxel_count"] == 0
    assert audit["clipped_set_voxel_count"] == 0
    assert audit["parameter_bounds"] == {"minimum": 0.0, "maximum": None}
    assert audit["pre_adjustment_minimum"] < 0
    assert audit["pre_adjustment_maximum"] < 0
    assert audit["clipped_total_change"] == 0.0
    assert not np.isfinite(arrays["p32_total"][0, 0, 0])


@pytest.mark.parametrize(
    ("policy", "expected_state", "expected_total"),
    [
        ("reject", VoxelCellState.NO_DATA, np.nan),
        ("clip_with_audit", VoxelCellState.TRUE_ZERO, 0.0),
    ],
)
def test_p32_audit_distinguishes_set_voxels_from_unique_voxels(
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    expected_state: VoxelCellState,
    expected_total: float,
) -> None:
    class _NegativeKriging:
        diagnostics = {}
        failures = {}

        def __init__(self, _samples, _settings) -> None:
            pass

        def predict(self, _point, _domain_id) -> InterpolationResult:
            return InterpolationResult(-1.0, VoxelCellState.MODELED_VALUE, kriging_variance=0.25)

    monkeypatch.setattr("dfn_cave_studio.voxel.parameter_field.DomainKrigingInterpolator", _NegativeKriging)
    estimates = [
        P32Estimate(
            domain_id=1,
            set_id=set_id,
            fracture_count=1,
            raw_sample_length=1,
            effective_sample_length=1,
            mean_exposure=1,
            p32=1,
            observability="adequate",
            random_seed=42,
        )
        for set_id in (1, 2)
    ]
    metadata, arrays = ParameterFieldBuilder().build(
        ModelBounds(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1),
        VoxelConfig(),
        DensitySettings(
            method=DensityMethod.ORDINARY_KRIGING,
            kriging=KrigingSettings(
                mode="manual",
                nugget=0,
                sill=1,
                range=2,
                minimum_neighbors=2,
                maximum_neighbors=6,
                non_negative_policy=policy,
            ),
        ),
        [],
        estimates,
        [JointSetConfig(set_id=1), JointSetConfig(set_id=2)],
        [SizeModel(domain_id=1, set_id=1), SizeModel(domain_id=1, set_id=2)],
        random_seed=42,
        domain_at_point=lambda _point: 1,
    )
    audit = metadata.provenance["negative_prediction_audit"]
    prefix = "rejected" if policy == "reject" else "clipped"
    assert audit[f"{prefix}_set_voxel_count"] == 2
    assert audit[f"{prefix}_voxel_count"] == 1
    assert audit["pre_adjustment_minimum"] == -1.0
    assert audit["pre_adjustment_maximum"] == -1.0
    assert audit["clipped_total_change"] == (2.0 if policy == "clip_with_audit" else 0.0)
    assert arrays["cell_state"][0, 0, 0] == CELL_STATE_CODES[expected_state]
    if np.isnan(expected_total):
        assert np.isnan(arrays["p32_total"][0, 0, 0])
    else:
        assert arrays["p32_total"][0, 0, 0] == expected_total


def test_p32_audit_counts_multiple_affected_spatial_voxels(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NegativeKriging:
        diagnostics = {}
        failures = {}

        def __init__(self, _samples, _settings) -> None:
            pass

        def predict(self, _point, _domain_id) -> InterpolationResult:
            return InterpolationResult(-1.0, VoxelCellState.MODELED_VALUE, kriging_variance=0.25)

    monkeypatch.setattr("dfn_cave_studio.voxel.parameter_field.DomainKrigingInterpolator", _NegativeKriging)
    metadata, _arrays = ParameterFieldBuilder().build(
        ModelBounds(x_min=0, x_max=2, y_min=0, y_max=1, z_min=0, z_max=1),
        VoxelConfig(),
        DensitySettings(
            method=DensityMethod.ORDINARY_KRIGING,
            kriging=KrigingSettings(
                mode="manual",
                nugget=0,
                sill=1,
                range=2,
                minimum_neighbors=2,
                maximum_neighbors=6,
                non_negative_policy="reject",
            ),
        ),
        [],
        [
            P32Estimate(
                domain_id=1,
                set_id=set_id,
                fracture_count=1,
                raw_sample_length=1,
                effective_sample_length=1,
                mean_exposure=1,
                p32=1,
                observability="adequate",
                random_seed=42,
            )
            for set_id in (1, 2)
        ],
        [JointSetConfig(set_id=1), JointSetConfig(set_id=2)],
        [SizeModel(domain_id=1, set_id=1), SizeModel(domain_id=1, set_id=2)],
        random_seed=42,
        domain_at_point=lambda _point: 1,
    )
    audit = metadata.provenance["negative_prediction_audit"]
    assert audit["rejected_set_voxel_count"] == 4
    assert audit["rejected_voxel_count"] == 2


def test_rejected_set_prediction_does_not_remove_positive_set_from_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter((-1.0, 2.5))

    class _MixedKriging:
        diagnostics = {}
        failures = {}

        def __init__(self, _samples, _settings) -> None:
            self.value = next(values)

        def predict(self, _point, _domain_id) -> InterpolationResult:
            return InterpolationResult(self.value, VoxelCellState.MODELED_VALUE, kriging_variance=0.25)

    monkeypatch.setattr("dfn_cave_studio.voxel.parameter_field.DomainKrigingInterpolator", _MixedKriging)
    metadata, arrays = ParameterFieldBuilder().build(
        ModelBounds(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1),
        VoxelConfig(),
        DensitySettings(
            method=DensityMethod.ORDINARY_KRIGING,
            kriging=KrigingSettings(
                mode="manual",
                nugget=0,
                sill=1,
                range=2,
                minimum_neighbors=2,
                maximum_neighbors=6,
                non_negative_policy="reject",
            ),
        ),
        [],
        [
            P32Estimate(
                domain_id=1,
                set_id=set_id,
                fracture_count=1,
                raw_sample_length=1,
                effective_sample_length=1,
                mean_exposure=1,
                p32=1,
                observability="adequate",
                random_seed=42,
            )
            for set_id in (1, 2)
        ],
        [JointSetConfig(set_id=1), JointSetConfig(set_id=2)],
        [SizeModel(domain_id=1, set_id=1), SizeModel(domain_id=1, set_id=2)],
        random_seed=42,
        domain_at_point=lambda _point: 1,
    )
    audit = metadata.provenance["negative_prediction_audit"]
    assert audit["rejected_set_voxel_count"] == 1
    assert audit["rejected_voxel_count"] == 1
    assert np.isnan(arrays["set_1_p32"][0, 0, 0])
    assert arrays["set_2_p32"][0, 0, 0] == pytest.approx(2.5)
    assert arrays["p32_total"][0, 0, 0] == pytest.approx(2.5)
    assert arrays["cell_state"][0, 0, 0] == CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]


def test_p32_audit_uses_constant_memory_unique_voxel_counters() -> None:
    source = inspect.getsource(ParameterFieldBuilder.build)
    assert "rejected_voxels" not in source
    assert "clipped_voxels" not in source
    assert "rejected_here" in source
    assert "clipped_here" in source


@pytest.mark.parametrize("method", [DensityMethod.IDW, DensityMethod.ORDINARY_KRIGING])
def test_seven_set_results_are_consistent_across_chunk_sizes(
    monkeypatch: pytest.MonkeyPatch, method: DensityMethod
) -> None:
    monkeypatch.setattr("dfn_cave_studio.voxel.parameter_field.expected_orientation_exposure", lambda *_a, **_k: 1.0)
    set_ids = [1, 2, 4, 7, 9, 12, 15]
    points = ((0.25, 0.25), (1.75, 0.25), (0.25, 1.75), (1.75, 1.75))
    intervals = [
        P10Interval(
            hole_id=f"SYN-{set_id}-{index}", from_depth=0, to_depth=1, domain_id=1, set_id=set_id,
            observation_count=index + 1, sample_length=1, p10=0.1 * set_id + index * 0.01,
            role="calibration", center_x=x, center_y=y, center_z=0.5,
            segment_directions=[(0.0, 0.0, -1.0, 1.0)],
        )
        for set_id in set_ids for index, (x, y) in enumerate(points)
    ]
    estimates = [
        P32Estimate(domain_id=1, set_id=set_id, fracture_count=4, raw_sample_length=4,
                    effective_sample_length=4, mean_exposure=1, p32=0.1 * set_id,
                    observability="adequate", random_seed=42)
        for set_id in set_ids
    ]
    settings = DensitySettings(
        method=method, min_neighbors=1, max_neighbors=4,
        kriging=KrigingSettings(mode="manual", nugget=0, sill=5, range=5,
                                minimum_neighbors=2, maximum_neighbors=4),
    )
    arguments = (
        ModelBounds(x_min=0, x_max=2, y_min=0, y_max=2, z_min=0, z_max=1), VoxelConfig(),
        settings, intervals, estimates, [JointSetConfig(set_id=set_id) for set_id in set_ids],
        [SizeModel(domain_id=1, set_id=set_id) for set_id in set_ids],
    )
    _, small = ParameterFieldBuilder().build(*arguments, random_seed=42, domain_at_point=lambda _p: 1, chunk_size=1)
    _, large = ParameterFieldBuilder().build(*arguments, random_seed=42, domain_at_point=lambda _p: 1, chunk_size=64)
    for name in small:
        np.testing.assert_allclose(small[name], large[name], rtol=0, atol=1e-6, equal_nan=True)


def test_idw_all_arrays_match_pre_kriging_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard every persisted IDW array, including its existing NaN bit patterns."""
    monkeypatch.setattr("dfn_cave_studio.voxel.parameter_field.expected_orientation_exposure", lambda *_a, **_k: 1.0)
    intervals = [
        P10Interval(hole_id=hole, from_depth=0, to_depth=1, domain_id=1, set_id=1,
                    observation_count=count, sample_length=1, p10=value, role="calibration",
                    center_x=x, center_y=0.5, center_z=0.5,
                    segment_directions=[(0.0, 0.0, -1.0, 1.0)])
        for hole, x, value, count in (("A", 0.5, 1.0, 1), ("B", 1.5, 3.0, 3))
    ]
    _, arrays = ParameterFieldBuilder().build(
        ModelBounds(x_min=0, x_max=2, y_min=0, y_max=2, z_min=0, z_max=1),
        VoxelConfig(), DensitySettings(method="idw", min_neighbors=1, max_neighbors=8), intervals,
        [P32Estimate(domain_id=1, set_id=1, fracture_count=4, raw_sample_length=2,
                     effective_sample_length=2, mean_exposure=1, p32=2,
                     observability="adequate", random_seed=42)],
        [JointSetConfig(set_id=1)], [SizeModel(domain_id=1, set_id=1)], random_seed=42,
        domain_at_point=lambda _point: 1, inside_model=lambda point: point[1] < 1,
    )
    expected = {
        "cell_state": "905c3e0e1bf85991fc02bb18a99f986ec86d99daf813aa29f256d3d6209a7465",
        "confidence": "5c0f50c6e283d64f560a7925aaed8b2540ff26747b986dbb10d97c7d4d19f5ee",
        "density_method": "27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1",
        "domain_id": "45c16df581e3889f3845994d2567332131fff444c1ad8e45fd50a0e4381f24b2",
        "effective_sample_length": "55e20760fddd52dbc77eb6ebb87d0ca11edfd3e176fa337b387ed480d95f9820",
        "nearest_data_distance": "31fca888075daac4c255e4007e81740ef50c0aef5ce28190b8c9ea74b7ad3494",
        "neighbour_count": "64ed86b909d6d0502b64b28db0ea1272ffb358e20e9b1d88b63ccb07fa900cf5",
        "observation_count": "60aade2bc7d803d24bceca5832090f3e23d06c1ca7f736e1a1578f3e04e8a3fa",
        "p32_total": "fde24fae9e325b6280a307108a4307f1e76d2293ea05f05655ea6ff03ef7e66f",
        "set_1_dip": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb",
        "set_1_dip_direction": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb",
        "set_1_kappa": "529ed9e4578731c902b6bdad665e3fe559f5ca7aecb5c69b618fb609b5144e84",
        "set_1_mean_radius": "d7b894f19b2a063c368662601410fb2ffcf1ad4a45d8d9c7e33626c4e467f2f9",
        "set_1_mean_squared_radius": "d7b894f19b2a063c368662601410fb2ffcf1ad4a45d8d9c7e33626c4e467f2f9",
        "set_1_p32": "fde24fae9e325b6280a307108a4307f1e76d2293ea05f05655ea6ff03ef7e66f",
        "set_1_probability": "d7b894f19b2a063c368662601410fb2ffcf1ad4a45d8d9c7e33626c4e467f2f9",
        "set_1_size_max_radius": "9b97afc7531daf705de7ce9de1d51cadc701f5030c46537e5873ef3e80e04767",
        "set_1_size_min_radius": "d7b894f19b2a063c368662601410fb2ffcf1ad4a45d8d9c7e33626c4e467f2f9",
        "set_1_size_parameter_1": "d7b894f19b2a063c368662601410fb2ffcf1ad4a45d8d9c7e33626c4e467f2f9",
        "set_1_size_parameter_2": "ef99cfd192ee2fe43a68cef2af40c85c2c215759f491c1b3fa09ed0f794f9201",
        "set_1_size_source": "037bbe3531313e516a83772c4d97c9ea7a34834c64bef8a5b6f61356ec4f0d68",
        "set_1_size_type": "7a7bf454c5f3cb1b9d9a20f81417f98d976fe3b3dd52c1b9968f02e89e7e8a2f",
    }
    actual = {
        name: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        for name, values in arrays.items()
    }
    assert actual == expected
