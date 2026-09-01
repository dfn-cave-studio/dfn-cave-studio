"""Scientific regression tests for M11.1 exact second voxelization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.m10 import M10QualitySummary, M10Realization
from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.voxel.second_voxelization import M11SecondVoxelizer, SecondVoxelizationConfig


def _case(
    centers: list[list[float]],
    normals: list[list[float]],
    radii: list[float],
    *,
    shape: tuple[int, int, int] = (2, 1, 1),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    set_ids: list[int] | None = None,
    subgrid: float = 0.0,
    worker_count: int = 1,
    cell_states: np.ndarray | None = None,
):
    count = len(radii)
    sets = np.asarray(set_ids or [1] * count, dtype=np.int16)
    metadata = ParameterFieldMetadata(
        shape=shape,
        origin=(0.0, 0.0, 0.0),
        spacing=spacing,
        field_names=[],
        set_ids=sorted(set(int(value) for value in sets)),
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=42,
        estimated_bytes=0,
    )
    modeled = CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
    parameter_arrays: dict[str, np.ndarray] = {
        "cell_state": (
            np.full(shape, modeled, dtype=np.uint8)
            if cell_states is None
            else np.asarray(cell_states, dtype=np.uint8).reshape(shape)
        ),
        "domain_id": np.ones(shape, dtype=np.int16),
    }
    geometry: dict[str, np.ndarray] = {
        "center": np.asarray(centers, dtype=np.float64).reshape(count, 3),
        "normal": np.asarray(normals, dtype=np.float64).reshape(count, 3),
        "radius": np.asarray(radii, dtype=np.float64),
        "ordinal": np.arange(count, dtype=np.uint64),
        "set_id": sets,
        "clipped_area": math.pi * np.square(radii),
    }
    for set_id in metadata.set_ids:
        parameter_arrays[f"set_{set_id}_p32"] = np.full(shape, 0.5, dtype=np.float32)
        geometry[f"p32_subgrid_set_{set_id}"] = np.full(shape, subgrid, dtype=np.float32)
    realization = M10Realization(
        realization_id="r-42",
        realization_index=0,
        seed=42,
        config_hash="m10-hash",
        quality=M10QualitySummary(p32_unresolved_orientation=0.125),
        geometry_arrays=geometry,
    )
    generation = ModelBounds(
        x_min=0.0,
        x_max=shape[0] * spacing[0],
        y_min=0.0,
        y_max=shape[1] * spacing[1],
        z_min=0.0,
        z_max=shape[2] * spacing[2],
    )
    return M11SecondVoxelizer(
        metadata,
        parameter_arrays,
        generation,
        realization,
        SecondVoxelizationConfig(worker_count=worker_count, fracture_batch_size=max(1, count)),
    ).compute()


def test_disk_crossing_two_voxels_conserves_area() -> None:
    result = _case([[1.0, 0.5, 0.5]], [[0.0, 0.0, 1.0]], [0.25])
    np.testing.assert_array_equal(result.arrays["voxel_flat_index"], [0, 1])
    np.testing.assert_allclose(result.arrays["intersection_area"], math.pi * 0.25**2 / 2.0, rtol=1e-12)
    assert result.conservation.absolute_error_total < 1e-12


@pytest.mark.parametrize(
    ("center", "normal", "expected_voxel"),
    [
        ([1.0, 0.5, 0.5], [1.0, 0.0, 0.0], 1),
        ([0.5, 1.0, 0.5], [0.0, 1.0, 0.0], 1),
        ([0.5, 0.5, 1.0], [0.0, 0.0, 1.0], 1),
    ],
)
def test_shared_face_is_owned_once(center, normal, expected_voxel) -> None:
    axis = int(np.argmax(normal))
    shape = tuple(2 if index == axis else 1 for index in range(3))
    result = _case([center], [normal], [0.2], shape=shape)
    np.testing.assert_array_equal(result.arrays["voxel_flat_index"], [expected_voxel])
    assert result.arrays["intersection_area"][0] == pytest.approx(math.pi * 0.2**2)


def test_outer_grid_maximum_face_is_included_by_last_voxel() -> None:
    result = _case([[1.0, 0.5, 0.5]], [[1.0, 0.0, 0.0]], [0.2], shape=(1, 1, 1))
    np.testing.assert_array_equal(result.arrays["voxel_flat_index"], [0])
    assert result.arrays["intersection_area"][0] == pytest.approx(math.pi * 0.2**2)


def test_joint_sets_remain_isolated_and_subgrid_is_added() -> None:
    result = _case(
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [0.1, 0.2],
        set_ids=[1, 2],
        subgrid=0.25,
    )
    assert np.count_nonzero(result.arrays["intersecting_fracture_count_set_1"]) == 1
    assert np.count_nonzero(result.arrays["intersecting_fracture_count_set_2"]) == 1
    np.testing.assert_allclose(
        result.arrays["p32_total"], result.arrays["p32_explicit_intersection"] + result.arrays["p32_subgrid"]
    )
    assert result.provenance["unresolved_orientation_reference"] == pytest.approx(0.125)
    assert result.p32_unresolved_orientation_reference == pytest.approx(0.125)
    assert result.provenance["unresolved_orientation_in_total"] is False


def test_tangent_candidate_has_no_sparse_pair() -> None:
    result = _case([[-0.25, 0.5, 0.5]], [[0.0, 0.0, 1.0]], [0.25])
    assert result.pair_count == 0


@pytest.mark.parametrize("worker_count", [2, 4, 8])
def test_worker_counts_produce_identical_sparse_arrays(worker_count: int) -> None:
    arguments = (
        [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [1.5, 0.5, 0.5]],
        [[1.0 / math.sqrt(14.0), 2.0 / math.sqrt(14.0), 3.0 / math.sqrt(14.0)], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [0.2, 0.3, 0.2],
    )
    one = _case(*arguments, worker_count=1)
    parallel = _case(*arguments, worker_count=worker_count)
    assert one.arrays.keys() == parallel.arrays.keys()
    for name in one.arrays:
        np.testing.assert_array_equal(one.arrays[name], parallel.arrays[name])


def test_cancellation_never_returns_partial_result() -> None:
    centers = [[0.5, 0.5, 0.5]]
    metadata = ParameterFieldMetadata(
        shape=(1, 1, 1), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), field_names=[],
        set_ids=[1], density_method=DensityMethod.GLOBAL_CONSTANT, random_seed=1, estimated_bytes=0
    )
    modeled = CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
    arrays = {"cell_state": np.full((1, 1, 1), modeled, np.uint8), "domain_id": np.ones((1, 1, 1), np.int16),
              "set_1_p32": np.ones((1, 1, 1), np.float32)}
    realization = M10Realization(
        realization_id="cancel", realization_index=0, seed=1, config_hash="x",
        geometry_arrays={"center": np.asarray(centers), "normal": np.asarray([[0.0, 0.0, 1.0]]),
                         "radius": np.asarray([0.2]), "ordinal": np.asarray([0], np.uint64),
                         "set_id": np.asarray([1], np.int16)},
    )
    bounds = ModelBounds(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1)
    with pytest.raises(InterruptedError):
        M11SecondVoxelizer(metadata, arrays, bounds, realization, cancelled=lambda: True).compute()


def test_generation_buffer_area_is_not_misreported_as_numerical_error() -> None:
    """Area outside the analysis grid is audited separately from conservation error."""
    metadata = ParameterFieldMetadata(
        shape=(1, 1, 1), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), field_names=[],
        set_ids=[1], density_method=DensityMethod.GLOBAL_CONSTANT, random_seed=1, estimated_bytes=0,
    )
    modeled = CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
    parameters = {
        "cell_state": np.full((1, 1, 1), modeled, np.uint8),
        "domain_id": np.ones((1, 1, 1), np.int16),
        "set_1_p32": np.ones((1, 1, 1), np.float32),
    }
    realization = M10Realization(
        realization_id="buffer", realization_index=0, seed=1, config_hash="x",
        geometry_arrays={
            "center": np.asarray([[1.0, 0.5, 0.5]]), "normal": np.asarray([[0.0, 0.0, 1.0]]),
            "radius": np.asarray([0.4]), "ordinal": np.asarray([0], np.uint64),
            "set_id": np.asarray([1], np.int16),
        },
    )
    generation = ModelBounds(x_min=-1, x_max=2, y_min=-1, y_max=2, z_min=-1, z_max=2)
    result = M11SecondVoxelizer(metadata, parameters, generation, realization).compute()
    assert result.conservation.generation_area_outside_analysis_total > 0.0
    assert result.conservation.absolute_error_total < 1e-12
    assert result.conservation.intersection_area_total <= result.conservation.target_area_total


def test_semantic_cell_states_are_preserved_verbatim() -> None:
    states = np.asarray(
        [
            CELL_STATE_CODES[VoxelCellState.MODELED_VALUE],
            CELL_STATE_CODES[VoxelCellState.TRUE_ZERO],
            CELL_STATE_CODES[VoxelCellState.NO_DATA],
            CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL],
        ],
        dtype=np.uint8,
    ).reshape(4, 1, 1)
    result = _case(
        [[0.5, 0.5, 0.5]], [[0.0, 0.0, 1.0]], [0.1], shape=(4, 1, 1), cell_states=states
    )
    np.testing.assert_array_equal(result.arrays["cell_state"], states)


def test_candidate_indexing_does_not_form_fracture_voxel_cartesian_product() -> None:
    centers = [[index + 0.5, 0.5, 0.5] for index in range(10)]
    result = _case(
        centers,
        [[0.0, 0.0, 1.0]] * 10,
        [0.1] * 10,
        shape=(100, 1, 1),
    )
    assert result.candidate_pair_count == 10
    assert result.candidate_pair_count < 10 * 100


@pytest.mark.parametrize("radius", [1e-4, 0.25, 1.75])
def test_oblique_disks_conserve_area_on_anisotropic_voxels(radius: float) -> None:
    normal = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    result = _case(
        [[2.0, 2.0, 2.0]],
        [normal.tolist()],
        [radius],
        shape=(4, 8, 4),
        spacing=(1.0, 0.5, 1.0),
    )
    assert result.conservation.absolute_error_total < max(1e-12, math.pi * radius**2 * 1e-10)
    assert result.conservation.intersection_area_total == pytest.approx(math.pi * radius**2, rel=1e-10)
