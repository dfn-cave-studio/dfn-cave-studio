"""Real off-screen VTK checks for M11 depth occlusion and masked holes."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import pyvista as pv

from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.models.m11 import M11SecondVoxelizationResult
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager
from dfn_cave_studio.visualization.m11_voxel_renderer import M11DisplayConfig, M11VoxelRenderer
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES


def _metadata(origin_z: float, shape: tuple[int, int, int] = (3, 3, 1)) -> ParameterFieldMetadata:
    return ParameterFieldMetadata(
        shape=shape,
        origin=(0.0, 0.0, origin_z),
        spacing=(1.0, 1.0, 1.0),
        field_names=["p32_total"],
        set_ids=[],
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=42,
        estimated_bytes=int(np.prod(shape) * 4),
    )


def _result(realization_id: str, values: np.ndarray) -> M11SecondVoxelizationResult:
    states = np.full(values.shape, CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], dtype=np.uint8)
    return M11SecondVoxelizationResult(
        realization_id=realization_id,
        source_m10_realization_id=realization_id,
        source_m10_config_hash="cfg",
        source_parameter_field_hash="field",
        arrays={"p32_total": np.asarray(values, dtype=np.float32), "cell_state": states},
    )


@contextmanager
def _plotter():
    plotter = pv.Plotter(off_screen=True, window_size=(240, 240))
    plotter.set_background("white")
    plotter.disable_anti_aliasing()
    plotter.ren_win.SetMultiSamples(0)
    try:
        yield plotter
    finally:
        plotter.close()


def _camera(plotter: pv.Plotter) -> None:
    plotter.camera_position = [(1.5, 1.5, 10.0), (1.5, 1.5, 0.0), (0.0, 1.0, 0.0)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = 1.8


def _config(display_mode: str, interpolation_mode: str, opacity: float = 1.0) -> M11DisplayConfig:
    return M11DisplayConfig(
        display_mode=display_mode,
        interpolation_mode=interpolation_mode,
        axis="z",
        fraction=0.5,
        range_mode="manual",
        manual_min=0.0,
        manual_max=1.0,
        opacity=opacity,
    )


def _render_pair(
    *,
    display_mode: str,
    interpolation_mode: str,
    front_values: np.ndarray,
    order: str,
    back_value: float = 0.0,
) -> tuple[np.ndarray, object, M11LayerManager, pv.Plotter]:
    plotter = pv.Plotter(off_screen=True, window_size=(240, 240))
    plotter.set_background("white")
    plotter.disable_anti_aliasing()
    plotter.ren_win.SetMultiSamples(0)
    manager = M11LayerManager(plotter)
    renderer = M11VoxelRenderer()
    back_values = np.full((3, 3, 1), back_value, dtype=np.float32)
    front = (_metadata(0.0), _result("front", front_values))
    back = (_metadata(-2.0), _result("back", back_values))
    records = {}
    for name, (metadata, result) in (("front", front), ("back", back)) if order == "FB" else (("back", back), ("front", front)):
        records[name] = renderer.render(
            manager,
            metadata,
            result,
            "p32_total",
            None,
            _config(display_mode, interpolation_mode),
        )
    _camera(plotter)
    image = plotter.screenshot(return_img=True)
    return image, records["front"], manager, plotter


def _center(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return image[height // 2, width // 2, :3].astype(np.int16)


def _reference(value: float, display_mode: str, interpolation_mode: str) -> np.ndarray:
    with _plotter() as plotter:
        manager = M11LayerManager(plotter)
        M11VoxelRenderer().render(
            manager,
            _metadata(0.0),
            _result("reference", np.full((3, 3, 1), value, dtype=np.float32)),
            "p32_total",
            None,
            _config(display_mode, interpolation_mode),
        )
        _camera(plotter)
        return _center(plotter.screenshot(return_img=True))


@pytest.mark.parametrize("display_mode", ["outer_surface", "orthogonal_section"])
@pytest.mark.parametrize("interpolation_mode", ["exact", "smooth"])
@pytest.mark.parametrize("order", ["FB", "BF"])
def test_finite_front_cloud_fully_occludes_rear_for_all_add_orders(
    display_mode: str, interpolation_mode: str, order: str
) -> None:
    front_reference = _reference(1.0, display_mode, interpolation_mode)
    image, record, _, plotter = _render_pair(
        display_mode=display_mode,
        interpolation_mode=interpolation_mode,
        front_values=np.ones((3, 3, 1), dtype=np.float32),
        order=order,
    )
    try:
        np.testing.assert_allclose(_center(image), front_reference, atol=2)
        plotter.render()
        assert record.actor.GetProperty().GetOpacity() == pytest.approx(1.0)
        assert not bool(record.actor.HasTranslucentPolygonalGeometry())
    finally:
        plotter.close()


@pytest.mark.parametrize("interpolation_mode", ["exact", "smooth"])
def test_nan_hole_exposes_rear_only_at_the_missing_cell(interpolation_mode: str) -> None:
    values = np.ones((3, 3, 1), dtype=np.float32)
    values[1, 1, 0] = np.nan
    image, _, _, plotter = _render_pair(
        display_mode="outer_surface",
        interpolation_mode=interpolation_mode,
        front_values=values,
        order="FB",
    )
    try:
        back_reference = _reference(0.0, "outer_surface", interpolation_mode)
        front_reference = _reference(1.0, "outer_surface", interpolation_mode)
        np.testing.assert_allclose(_center(image), back_reference, atol=2)
        # This sample is well inside the upper-left valid cell and away from anti-aliased edges.
        valid_pixel = image[80, 80, :3].astype(np.int16)
        assert np.linalg.norm(valid_pixel - front_reference) < np.linalg.norm(valid_pixel - back_reference)
    finally:
        plotter.close()


@pytest.mark.parametrize("interpolation_mode", ["exact", "smooth"])
def test_true_zero_remains_visible_and_opaque(interpolation_mode: str) -> None:
    image, record, _, plotter = _render_pair(
        display_mode="outer_surface",
        interpolation_mode=interpolation_mode,
        front_values=np.zeros((3, 3, 1), dtype=np.float32),
        order="BF",
        back_value=1.0,
    )
    try:
        np.testing.assert_allclose(_center(image), _reference(0.0, "outer_surface", interpolation_mode), atol=2)
        assert not bool(record.actor.HasTranslucentPolygonalGeometry())
    finally:
        plotter.close()


def test_opacity_round_trip_restores_opaque_depth_occlusion() -> None:
    front_reference = _reference(1.0, "orthogonal_section", "exact")
    image, record, manager, plotter = _render_pair(
        display_mode="orthogonal_section",
        interpolation_mode="exact",
        front_values=np.ones((3, 3, 1), dtype=np.float32),
        order="FB",
    )
    try:
        np.testing.assert_allclose(_center(image), front_reference, atol=2)
        assert manager.set_opacity(record.layer_id, 0.5)
        translucent = _center(plotter.screenshot(return_img=True))
        assert np.linalg.norm(translucent - front_reference) > 5
        assert bool(record.actor.HasTranslucentPolygonalGeometry())
        assert manager.set_opacity(record.layer_id, 1.0)
        restored = _center(plotter.screenshot(return_img=True))
        np.testing.assert_allclose(restored, front_reference, atol=2)
        assert not bool(record.actor.HasTranslucentPolygonalGeometry())
    finally:
        plotter.close()


@pytest.mark.parametrize("interpolation_mode", ["exact", "smooth"])
def test_cutaway_internal_face_is_opaque_in_real_offscreen_render(interpolation_mode: str) -> None:
    metadata = _metadata(0.0, shape=(3, 3, 3))
    result = _result("cutaway", np.ones((3, 3, 3), dtype=np.float32))
    config = M11DisplayConfig(
        display_mode="cutaway",
        interpolation_mode=interpolation_mode,
        plane_origin=(1.5, 1.5, 1.5),
        plane_normal=(1.0, 0.0, 0.0),
        range_mode="manual",
        manual_min=0.0,
        manual_max=1.0,
        opacity=1.0,
    )

    def render(with_rear_plane: bool) -> tuple[np.ndarray, object]:
        with _plotter() as plotter:
            if with_rear_plane:
                plotter.add_mesh(
                    pv.Plane(center=(2.2, 1.5, 1.5), direction=(1.0, 0.0, 0.0), i_size=3.0, j_size=3.0),
                    color="blue",
                    lighting=False,
                )
            record = M11VoxelRenderer().render(
                M11LayerManager(plotter), metadata, result, "p32_total", None, config
            )
            plotter.camera_position = [(-5.0, 1.5, 1.5), (1.5, 1.5, 1.5), (0.0, 0.0, 1.0)]
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = 1.8
            image = plotter.screenshot(return_img=True)
            translucent = bool(record.actor.HasTranslucentPolygonalGeometry())
            opacity = record.actor.GetProperty().GetOpacity()
            return _center(image), (translucent, opacity)

    reference, _ = render(False)
    occluded, actor_state = render(True)
    np.testing.assert_allclose(occluded, reference, atol=2)
    assert actor_state == (False, 1.0)


@pytest.mark.parametrize("interpolation_mode", ["exact", "smooth"])
def test_box_cutaway_new_internal_face_is_opaque_in_real_offscreen_render(interpolation_mode: str) -> None:
    metadata = _metadata(0.0, shape=(3, 3, 3))
    result = _result("box-cutaway", np.ones((3, 3, 3), dtype=np.float32))
    config = M11DisplayConfig(
        display_mode="box_cutaway",
        interpolation_mode=interpolation_mode,
        box_bounds=(1.0, 2.0, 1.0, 2.0, 1.0, 2.0),
        snap_box_to_voxel_faces=True,
        range_mode="manual",
        manual_min=0.0,
        manual_max=1.0,
        opacity=1.0,
    )

    def render(with_rear_plane: bool) -> tuple[np.ndarray, tuple[bool, float]]:
        with _plotter() as plotter:
            if with_rear_plane:
                plotter.add_mesh(
                    pv.Plane(center=(1.5, 1.5, 1.5), direction=(1.0, 0.0, 0.0), i_size=1.0, j_size=1.0),
                    color="blue",
                    lighting=False,
                )
            record = M11VoxelRenderer().render(
                M11LayerManager(plotter), metadata, result, "p32_total", None, config
            )
            plotter.camera_position = [(-5.0, 1.5, 1.5), (1.0, 1.5, 1.5), (0.0, 0.0, 1.0)]
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = 0.7
            image = plotter.screenshot(return_img=True)
            state = (
                bool(record.actor.HasTranslucentPolygonalGeometry()),
                float(record.actor.GetProperty().GetOpacity()),
            )
            return _center(image), state

    reference, _ = render(False)
    occluded, actor_state = render(True)
    np.testing.assert_allclose(occluded, reference, atol=2)
    assert actor_state == (False, 1.0)
