"""Real off-screen VTK regressions for opaque M9 parameter-field slices."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import pyvista as pv

from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.models.m11 import M11SecondVoxelizationResult
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.visualization.m9_layer_manager import M9LayerManager
from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager
from dfn_cave_studio.visualization.m11_voxel_renderer import M11DisplayConfig, M11VoxelRenderer
from dfn_cave_studio.visualization.parameter_field_renderer import ParameterFieldRenderer
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES

pytestmark = pytest.mark.real_vtk_render


def _metadata(z: float) -> ParameterFieldMetadata:
    return ParameterFieldMetadata(
        shape=(3, 3, 1),
        origin=(0.0, 0.0, z),
        spacing=(1.0, 1.0, 1.0),
        field_names=["p32_total"],
        set_ids=[],
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=42,
        estimated_bytes=36,
    )


def _states(*, true_zero: bool = False) -> np.ndarray:
    state = VoxelCellState.TRUE_ZERO if true_zero else VoxelCellState.MODELED_VALUE
    return np.full((3, 3, 1), CELL_STATE_CODES[state], dtype=np.uint8)


def _values(center: float) -> np.ndarray:
    values = np.full((3, 3, 1), 0.5, dtype=np.float32)
    values[0, 0, 0] = 0.0
    values[2, 2, 0] = 1.0
    values[1, 1, 0] = center
    return values


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


def _center(image: np.ndarray) -> np.ndarray:
    return image[image.shape[0] // 2, image.shape[1] // 2, :3].astype(np.int16)


def _render_m9(plotter: pv.Plotter, name: str, z: float, values: np.ndarray, opacity: float = 1.0):
    return ParameterFieldRenderer().render_slice(
        plotter,
        _metadata(z),
        {"p32_total": values, "cell_state": _states(true_zero=bool(np.all(values == 0.0)))},
        "p32_total",
        "z",
        0.5,
        opacity=opacity,
        actor_name=name,
        show_scalar_bar=False,
    )


def _reference(center: float) -> np.ndarray:
    with _plotter() as plotter:
        _render_m9(plotter, "m9_slice:reference:z:0", 0.0, _values(center))
        _camera(plotter)
        return _center(plotter.screenshot(return_img=True))


@pytest.mark.parametrize("order", ["front_back", "back_front"])
def test_m9_finite_front_slice_occludes_rear_independent_of_add_order(order: str) -> None:
    with _plotter() as plotter:
        actors = {
            "front": lambda: _render_m9(plotter, "m9_slice:front:z:0", 0.0, _values(1.0)),
            "back": lambda: _render_m9(plotter, "m9_slice:back:z:0", -2.0, _values(0.0)),
        }
        sequence = ("front", "back") if order == "front_back" else ("back", "front")
        rendered = {name: actors[name]() for name in sequence}
        _camera(plotter)
        image = plotter.screenshot(return_img=True)
        np.testing.assert_allclose(_center(image), _reference(1.0), atol=2)
        assert not bool(rendered["front"].HasTranslucentPolygonalGeometry())


def test_m9_nan_hole_exposes_rear_only_at_hole_and_true_zero_is_opaque() -> None:
    with _plotter() as plotter:
        _render_m9(plotter, "m9_slice:back:z:0", -2.0, _values(0.0))
        front = _values(1.0)
        front[1, 1, 0] = np.nan
        front_before = front.copy()
        _render_m9(plotter, "m9_slice:front:z:0", 0.0, front)
        _camera(plotter)
        image = plotter.screenshot(return_img=True)
        np.testing.assert_allclose(_center(image), _reference(0.0), atol=2)
        finite_pixel = image[80, 80, :3].astype(np.int16)
        assert np.linalg.norm(finite_pixel - _reference(0.5)) < np.linalg.norm(
            finite_pixel - _reference(0.0)
        )
        np.testing.assert_array_equal(front, front_before)

    with _plotter() as plotter:
        _render_m9(plotter, "m9_slice:back:z:0", -2.0, np.ones((3, 3, 1), np.float32))
        actor = _render_m9(plotter, "m9_slice:zero:z:0", 0.0, np.zeros((3, 3, 1), np.float32))
        _camera(plotter)
        zero_image = plotter.screenshot(return_img=True)
        with _plotter() as reference_plotter:
            _render_m9(
                reference_plotter,
                "m9_slice:zero-reference:z:0",
                0.0,
                np.zeros((3, 3, 1), np.float32),
            )
            _camera(reference_plotter)
            zero_reference = _center(reference_plotter.screenshot(return_img=True))
        np.testing.assert_allclose(_center(zero_image), zero_reference, atol=2)
        assert not bool(actor.HasTranslucentPolygonalGeometry())


def test_m9_opacity_round_trip_and_m11_namespaced_cleanup() -> None:
    with _plotter() as plotter:
        m9_manager = M9LayerManager(plotter)
        m9_actor = _render_m9(plotter, "m9_slice:p32_total:z:0", 0.0, _values(1.0))
        m9_bar_id = m9_manager.scalar_bar_id("p32_total")
        m9_bar = m9_manager.create_or_get_scalar_bar(m9_bar_id, "M9 P32", m9_actor)
        m9_record = m9_manager.add_or_replace(
            "m9_slice:p32_total:z:0",
            m9_actor,
            field_name="p32_total",
            axis="z",
            slice_index=0,
            coordinate=0.5,
            opacity=1.0,
            scalar_bar_id=m9_bar_id,
            scalar_bar_actor=m9_bar,
        )
        m11_manager = M11LayerManager(plotter)
        m11_values = _values(0.0)
        m11_result = M11SecondVoxelizationResult(
            realization_id="m11-back",
            source_m10_realization_id="back",
            source_m10_config_hash="cfg",
            source_parameter_field_hash="field",
            arrays={"p32_total": m11_values, "cell_state": _states()},
        )
        m11_record = M11VoxelRenderer().render(
            m11_manager,
            _metadata(-2.0),
            m11_result,
            "p32_total",
            None,
            M11DisplayConfig(
                display_mode="orthogonal_section",
                axis="z",
                range_mode="manual",
                manual_min=0.0,
                manual_max=1.0,
            ),
        )
        _camera(plotter)
        opaque = _center(plotter.screenshot(return_img=True))
        assert m9_manager.set_opacity(m9_record.layer_id, 0.5)
        translucent = _center(plotter.screenshot(return_img=True))
        assert np.linalg.norm(translucent - opaque) > 5
        assert m9_manager.set_opacity(m9_record.layer_id, 1.0)
        np.testing.assert_allclose(_center(plotter.screenshot(return_img=True)), opaque, atol=2)
        assert not bool(m9_actor.HasTranslucentPolygonalGeometry())
        assert m9_manager.remove(m9_record.layer_id)
        assert m11_manager.contains(m11_record.layer_id)
        assert m9_bar_id not in plotter.scalar_bars
        assert m11_record.scalar_bar_id in plotter.scalar_bars
