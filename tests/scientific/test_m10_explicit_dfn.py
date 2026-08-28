"""Small scientific validation cases for M10 parameter-field generation."""

import math

import numpy as np

from dfn_cave_studio.dfn.m10_generator import M10ExplicitDFNGenerator
from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.m10 import M10GenerationConfig
from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata, SizeModel, SizeModelSource
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES


def test_two_p32_voxels_three_sets_and_semantic_empty_cells():
    shape = (4, 1, 1)
    metadata = ParameterFieldMetadata(
        shape=shape,
        origin=(0, 0, 0),
        spacing=(10, 10, 10),
        field_names=[],
        set_ids=[1, 2, 3],
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=42,
        estimated_bytes=0,
    )
    arrays = {
        "cell_state": np.asarray(
            [
                [[CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]]],
                [[CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]]],
                [[CELL_STATE_CODES[VoxelCellState.TRUE_ZERO]]],
                [[CELL_STATE_CODES[VoxelCellState.NO_DATA]]],
            ],
            dtype=np.uint8,
        ),
        "domain_id": np.ones(shape, dtype=np.int32),
    }
    target_directions = {1: (20.0, 45.0), 2: (140.0, 65.0), 3: (280.0, 30.0)}
    for set_id, (direction, dip) in target_directions.items():
        arrays[f"set_{set_id}_p32"] = np.asarray([[[0.5]], [[1.0]], [[0.0]], [[np.nan]]], dtype=np.float32)
        arrays[f"set_{set_id}_dip_direction"] = np.full(shape, direction, dtype=np.float32)
        arrays[f"set_{set_id}_dip"] = np.full(shape, dip, dtype=np.float32)
        arrays[f"set_{set_id}_kappa"] = np.full(shape, 120.0, dtype=np.float32)
    sizes = [
        SizeModel(
            domain_id=1,
            set_id=set_id,
            distribution_type="fixed",
            parameters={"radius": 1.0},
            min_radius=1.0,
            max_radius=1.0,
            mean_radius=1.0,
            mean_squared_radius=1.0,
            source=SizeModelSource.ASSUMED,
        )
        for set_id in (1, 2, 3)
    ]
    result = M10ExplicitDFNGenerator(
        metadata=metadata,
        arrays=arrays,
        generation_domain=ModelBounds(x_min=-2, x_max=42, y_min=-2, y_max=12, z_min=-2, z_max=12),
        size_models=sizes,
        config=M10GenerationConfig(base_seed=77),
        project_id="scientific-case",
    ).generate(0)
    np.testing.assert_allclose(
        result.quality.target_fracture_count_expectation,
        3 * (0.5 + 1.0) * 1000.0 / math.pi,
        rtol=1e-7,
    )
    assert set(result.geometry_arrays["set_id"]) == {1, 2, 3}
    assert set(result.geometry_arrays["voxel_index"][:, 0]) <= {0, 1}
    for set_id, (direction, dip) in target_directions.items():
        normals = result.geometry_arrays["normal"][result.geometry_arrays["set_id"] == set_id]
        mean = normals.mean(axis=0)
        mean /= np.linalg.norm(mean)
        target = dip_dir_dip_to_normal(direction, dip)
        assert abs(float(np.dot(mean, target))) > 0.98
    assert "M11" in result.quality.local_p32_notice
