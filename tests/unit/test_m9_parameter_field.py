"""Unit tests for domain-safe M9 interpolation and field state semantics."""

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, P32Estimate, SizeModel
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES, IDWInterpolator, ParameterFieldBuilder, SpatialSample


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
