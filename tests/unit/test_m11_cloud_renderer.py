"""Scientific-display unit tests for M11.2 native VTK P32 clouds."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.models.m11 import M11SecondVoxelizationResult
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager
from dfn_cave_studio.visualization.m11_voxel_renderer import M11DisplayConfig, M11VoxelRenderer
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES


class FakeProperty:
    def SetOpacity(self, opacity):
        self.opacity = opacity


class FakeLookupTable:
    def __init__(self) -> None:
        self.table = [(0.1, 0.2, 0.3, 0.0), (0.4, 0.5, 0.6, 0.5)]
        self.nan_color = (0.6, 0.6, 0.6, 0.0)
        self.below_color = (0.0, 0.0, 0.0, 0.25)
        self.above_color = (1.0, 1.0, 1.0, 0.75)
        self.modified = False

    def GetNumberOfTableValues(self):
        return len(self.table)

    def GetTableValue(self, index):
        return self.table[index]

    def SetTableValue(self, index, red, green, blue, alpha):
        self.table[index] = (red, green, blue, alpha)

    def GetNanColor(self):
        return self.nan_color

    def SetNanColor(self, red, green, blue, alpha):
        self.nan_color = (red, green, blue, alpha)

    def GetBelowRangeColor(self):
        return self.below_color

    def SetBelowRangeColor(self, red, green, blue, alpha):
        self.below_color = (red, green, blue, alpha)

    def GetAboveRangeColor(self):
        return self.above_color

    def SetAboveRangeColor(self, red, green, blue, alpha):
        self.above_color = (red, green, blue, alpha)

    def Modified(self):
        self.modified = True


class FakeMapper:
    def __init__(self) -> None:
        self.lookup_table = FakeLookupTable()
        self.modified = False

    def GetLookupTable(self):
        return self.lookup_table

    def Modified(self):
        self.modified = True


class FakeActor:
    def __init__(self) -> None:
        self.visible = True
        self.mapper = FakeMapper()
        self.actor_property = FakeProperty()

    def SetVisibility(self, visible):
        self.visible = visible

    def GetProperty(self):
        return self.actor_property


class FakeScalarBar:
    def __init__(self) -> None:
        self.visible = True
        self.title = ""

    def SetVisibility(self, visible):
        self.visible = visible

    def SetTitle(self, title):
        self.title = title


class FakeWidget:
    def __init__(self, bounds=None) -> None:
        self.enabled = True
        self.bounds = bounds

    def Off(self):
        self.enabled = False

    def SetEnabled(self, enabled):
        self.enabled = enabled

    def PlaceWidget(self, *bounds):
        self.bounds = bounds


class FakePlotter:
    def __init__(self) -> None:
        self.renderer = SimpleNamespace(actors={"m9_slice:keep": FakeActor()})
        self.scalar_bars = {"m9_scalar:keep": FakeScalarBar()}
        self.meshes = {}
        self.mesh_kwargs = {}
        self.widgets = []
        self.render_count = 0
        self.fail_scalar_bar = False

    def add_mesh(self, mesh, *, name, **kwargs):
        actor = FakeActor()
        self.renderer.actors[name] = actor
        self.meshes[name] = mesh
        self.mesh_kwargs[name] = kwargs
        return actor

    def remove_actor(self, actor, *, render=False):
        del render
        for name, candidate in list(self.renderer.actors.items()):
            if candidate is actor:
                self.renderer.actors.pop(name)

    def add_scalar_bar(self, *, title, mapper, render=False):
        del mapper, render
        if self.fail_scalar_bar:
            raise RuntimeError("synthetic scalar-bar failure")
        if title in self.scalar_bars:
            return None
        actor = FakeScalarBar()
        self.scalar_bars[title] = actor
        return actor

    def remove_scalar_bar(self, *, title, render=False):
        del render
        self.scalar_bars.pop(title)

    def add_plane_widget(self, callback, **kwargs):
        del callback, kwargs
        widget = FakeWidget()
        self.widgets.append(widget)
        return widget

    def add_box_widget(self, callback, **kwargs):
        del callback
        widget = FakeWidget(bounds=kwargs.get("bounds"))
        self.widgets.append(widget)
        return widget

    def render(self):
        self.render_count += 1


def _metadata(shape=(3, 2, 2), *, origin=(100.0, 200.0, 300.0), spacing=(2.0, 3.0, 4.0)):
    return ParameterFieldMetadata(
        shape=shape,
        origin=origin,
        spacing=spacing,
        field_names=["p32_total"],
        set_ids=[1],
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=42,
        estimated_bytes=int(np.prod(shape) * 4),
    )


def _result(values, states=None):
    values = np.asarray(values, dtype=np.float32)
    if states is None:
        states = np.full(values.shape, CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], np.uint8)
    return M11SecondVoxelizationResult(
        realization_id="m11-r1",
        source_m10_realization_id="r1",
        source_m10_config_hash="cfg",
        source_parameter_field_hash="field",
        arrays={"p32_total": values, "cell_state": np.asarray(states, dtype=np.uint8)},
    )


def test_outer_surface_uses_only_valid_model_cells() -> None:
    values = np.arange(12, dtype=np.float32).reshape((3, 2, 2))
    states = np.full(values.shape, CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], np.uint8)
    states[2, 1, 1] = CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL]
    values[2, 1, 1] = 9999.0
    prepared = M11VoxelRenderer().prepare_display(
        _metadata(), _result(values, states), "p32_total", None, M11DisplayConfig(display_mode="outer_surface")
    )
    assert prepared.mesh.n_cells > 0
    assert 9999.0 not in np.asarray(prepared.mesh.cell_data["p32_total"])


def test_true_zero_is_retained_while_invalid_mask_states_are_removed() -> None:
    values = np.array([0.0, 9999.0], dtype=np.float32).reshape((2, 1, 1))
    states = np.array(
        [CELL_STATE_CODES[VoxelCellState.TRUE_ZERO], CELL_STATE_CODES[VoxelCellState.NO_DATA]],
        dtype=np.uint8,
    ).reshape((2, 1, 1))
    prepared = M11VoxelRenderer().prepare_display(
        _metadata((2, 1, 1)),
        _result(values, states),
        "p32_total",
        None,
        M11DisplayConfig(display_mode="voxel_cells"),
    )
    np.testing.assert_array_equal(prepared.mesh.cell_data["p32_total"], [0.0])


@pytest.mark.parametrize("interpolation_mode", ["exact", "smooth"])
def test_non_finite_field_cells_are_removed_before_geometry_operations(interpolation_mode) -> None:
    values = np.array([0.0, np.nan, 2.0], dtype=np.float32).reshape((3, 1, 1))
    prepared = M11VoxelRenderer().prepare_display(
        _metadata((3, 1, 1)),
        _result(values),
        "p32_total",
        None,
        M11DisplayConfig(display_mode="outer_surface", interpolation_mode=interpolation_mode),
    )
    scalars = (
        np.asarray(prepared.mesh.cell_data["p32_total"])
        if interpolation_mode == "exact"
        else np.asarray(prepared.mesh.point_data["p32_total"])
    )
    assert np.all(np.isfinite(scalars))
    assert np.any(scalars == 0.0)


def test_smooth_display_does_not_mix_masked_values_and_preserves_arrays() -> None:
    values = np.array([10.0, 1000.0], dtype=np.float32).reshape((2, 1, 1))
    states = np.array(
        [CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], CELL_STATE_CODES[VoxelCellState.NO_DATA]],
        dtype=np.uint8,
    ).reshape((2, 1, 1))
    result = _result(values, states)
    before = {name: array.copy() for name, array in result.arrays.items()}
    prepared = M11VoxelRenderer().prepare_display(
        _metadata((2, 1, 1)),
        result,
        "p32_total",
        None,
        M11DisplayConfig(display_mode="voxel_cells", interpolation_mode="smooth"),
    )
    np.testing.assert_allclose(prepared.mesh.point_data["p32_total"], 10.0)
    assert "p32_total" not in prepared.mesh.cell_data
    for name, expected in before.items():
        np.testing.assert_array_equal(result.arrays[name], expected)


@pytest.mark.parametrize("axis,index,coordinate", [("x", 1, 103.0), ("y", 0, 201.5), ("z", 0, 302.0)])
def test_orthogonal_section_location(axis, index, coordinate) -> None:
    prepared = M11VoxelRenderer().prepare_display(
        _metadata(),
        _result(np.ones((3, 2, 2))),
        "p32_total",
        None,
        M11DisplayConfig(display_mode="orthogonal_section", axis=axis, fraction=0.5),
    )
    assert prepared.slice_index == index
    assert prepared.coordinate == pytest.approx(coordinate)
    assert prepared.mesh.n_cells > 0


def test_continuous_section_snapshots_have_position_specific_identity() -> None:
    renderer = M11VoxelRenderer()
    result = _result(np.ones((3, 2, 2)))
    first = renderer.prepare_display(
        _metadata(),
        result,
        "p32_total",
        None,
        M11DisplayConfig(display_mode="orthogonal_section", axis="x", section_coordinate=100.5),
    )
    second = renderer.prepare_display(
        _metadata(),
        result,
        "p32_total",
        None,
        M11DisplayConfig(display_mode="orthogonal_section", axis="x", section_coordinate=102.0),
    )
    assert first.slice_index == second.slice_index == -1
    assert first.layer_id != second.layer_id


def test_arbitrary_plane_normal_validation_and_stable_identity() -> None:
    renderer = M11VoxelRenderer()
    assert renderer.normalize_plane_normal((0.0, 0.0, 2.0)) == (0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="non-zero"):
        renderer.normalize_plane_normal((0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        renderer.normalize_plane_normal((np.nan, 0.0, 1.0))
    config = M11DisplayConfig(
        display_mode="arbitrary_plane",
        plane_origin=(103.0, 203.0, 304.0),
        plane_normal=(1.0, 1.0, 1.0),
    )
    first = renderer.prepare_display(_metadata(), _result(np.ones((3, 2, 2))), "p32_total", None, config)
    second = renderer.prepare_display(_metadata(), _result(np.ones((3, 2, 2))), "p32_total", None, config)
    assert first.plane_id == second.plane_id
    assert first.mesh.n_cells > 0

    with pytest.raises(ValueError, match="finite"):
        renderer.prepare_display(
            _metadata(),
            _result(np.ones((3, 2, 2))),
            "p32_total",
            None,
            M11DisplayConfig(
                display_mode="arbitrary_plane",
                plane_origin=(np.inf, 203.0, 304.0),
                plane_normal=(0.0, 0.0, 1.0),
            ),
        )


def test_exact_uses_cell_scalars_and_global_ranges_match_sections() -> None:
    renderer = M11VoxelRenderer()
    result = _result(np.arange(12, dtype=np.float32).reshape((3, 2, 2)))
    x = renderer.prepare_display(
        _metadata(), result, "p32_total", None, M11DisplayConfig(display_mode="orthogonal_section", axis="x")
    )
    z = renderer.prepare_display(
        _metadata(), result, "p32_total", None, M11DisplayConfig(display_mode="orthogonal_section", axis="z")
    )
    assert x.scalar_preference == "cell"
    assert "p32_total" in x.mesh.cell_data
    assert x.color_range == z.color_range == (0.0, 11.0)


def test_manual_range_and_discrete_grid_render_options() -> None:
    renderer = M11VoxelRenderer()
    with pytest.raises(ValueError, match="min < max"):
        renderer.validate_manual_range(2.0, 1.0)
    plotter = FakePlotter()
    manager = M11LayerManager(plotter)
    config = M11DisplayConfig(
        display_mode="voxel_cells",
        range_mode="manual",
        manual_min=0.2,
        manual_max=0.8,
        color_mode="discrete",
        contour_bands=7,
        show_grid_lines=True,
    )
    record = renderer.render(manager, _metadata(), _result(np.ones((3, 2, 2))), "p32_total", None, config)
    kwargs = plotter.mesh_kwargs[record.layer_id]
    assert kwargs["clim"] == (0.2, 0.8)
    assert kwargs["n_colors"] == 7
    assert kwargs["show_edges"] is True
    assert kwargs["nan_opacity"] == 1.0
    lookup_table = record.actor.mapper.lookup_table
    assert all(value[3] == 1.0 for value in lookup_table.table)
    assert lookup_table.nan_color[3] == 1.0
    assert lookup_table.below_color[3] == 1.0
    assert lookup_table.above_color[3] == 1.0


def test_repeated_render_and_widget_cleanup_do_not_accumulate() -> None:
    renderer = M11VoxelRenderer()
    plotter = FakePlotter()
    manager = M11LayerManager(plotter)
    result = _result(np.ones((3, 2, 2)))
    config = M11DisplayConfig(
        display_mode="arbitrary_plane",
        plane_origin=(103.0, 203.0, 304.0),
        plane_normal=(0.0, 0.0, 1.0),
        interactive_plane=True,
    )
    for _ in range(10):
        renderer.render(manager, _metadata(), result, "p32_total", None, config)
    assert len(manager.list_layers()) == 1
    assert len([name for name in plotter.renderer.actors if name.startswith("m11_voxel:")]) == 1
    assert len([name for name in plotter.scalar_bars if name.startswith("m11_scalar_bar:")]) == 1
    assert sum(widget.enabled for widget in plotter.widgets) == 1
    manager.clear_m11_layers()
    assert not any(widget.enabled for widget in plotter.widgets)
    assert set(plotter.renderer.actors) == {"m9_slice:keep"}
    assert set(plotter.scalar_bars) == {"m9_scalar:keep"}


def test_replacing_colour_configuration_removes_superseded_scalar_bar() -> None:
    renderer = M11VoxelRenderer()
    plotter = FakePlotter()
    manager = M11LayerManager(plotter)
    result = _result(np.ones((3, 2, 2)))
    renderer.render(manager, _metadata(), result, "p32_total", None, M11DisplayConfig())
    renderer.render(
        manager,
        _metadata(),
        result,
        "p32_total",
        None,
        M11DisplayConfig(range_mode="manual", manual_min=0.0, manual_max=2.0),
    )
    assert len(manager.list_layers()) == 1
    assert len([key for key in plotter.scalar_bars if key.startswith("m11_scalar_bar:")]) == 1


def test_render_failure_rolls_back_mesh_scalar_bar_and_plane_widget() -> None:
    renderer = M11VoxelRenderer()
    plotter = FakePlotter()
    plotter.fail_scalar_bar = True
    manager = M11LayerManager(plotter)
    config = M11DisplayConfig(
        display_mode="arbitrary_plane",
        plane_origin=(103.0, 203.0, 304.0),
        interactive_plane=True,
    )
    with pytest.raises(RuntimeError, match="scalar-bar"):
        renderer.render(manager, _metadata(), _result(np.ones((3, 2, 2))), "p32_total", None, config)
    assert manager.list_layers() == []
    assert set(plotter.renderer.actors) == {"m9_slice:keep"}
    assert set(plotter.scalar_bars) == {"m9_scalar:keep"}
    assert not any(widget.enabled for widget in plotter.widgets)


def test_box_cutaway_retains_volume_and_has_stable_interactive_identity() -> None:
    renderer = M11VoxelRenderer()
    result = _result(np.arange(12, dtype=np.float32).reshape((3, 2, 2)))
    config = M11DisplayConfig(
        display_mode="box_cutaway",
        interactive_slot="box_cutaway",
        box_bounds=(100.0, 104.0, 200.0, 206.0, 300.0, 308.0),
    )
    clipped = renderer.prepare_display(_metadata(), result, "p32_total", None, config)
    assert clipped.mesh.n_cells > 0
    assert clipped.mesh.bounds.x_max == pytest.approx(104.0)
    changed = renderer.prepare_display(
        _metadata(),
        result,
        "p32_total",
        None,
        M11DisplayConfig(
            display_mode="box_cutaway",
            interactive_slot="box_cutaway",
            box_bounds=(100.0, 106.0, 200.0, 206.0, 300.0, 308.0),
        ),
    )
    assert changed.layer_id == clipped.layer_id


@pytest.mark.parametrize("interpolation_mode", ["exact", "smooth"])
def test_box_cutaway_central_voxel_has_six_new_scalar_surfaces(interpolation_mode) -> None:
    metadata = _metadata((3, 3, 3))
    values = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    result = _result(values)
    before = {name: array.copy() for name, array in result.arrays.items()}
    prepared = M11VoxelRenderer().prepare_display(
        metadata,
        result,
        "p32_total",
        None,
        M11DisplayConfig(
            display_mode="box_cutaway",
            interpolation_mode=interpolation_mode,
            box_bounds=(102.0, 104.0, 203.0, 206.0, 304.0, 308.0),
            snap_box_to_voxel_faces=True,
        ),
    )
    assert tuple(prepared.mesh.bounds) == pytest.approx((102.0, 104.0, 203.0, 206.0, 304.0, 308.0))
    assert prepared.mesh.n_cells == 6
    assert np.all(prepared.mesh.faces[::5] == 4)
    scalars = (
        np.asarray(prepared.mesh.cell_data["p32_total"])
        if interpolation_mode == "exact"
        else np.asarray(prepared.mesh.point_data["p32_total"])
    )
    assert np.all(np.isfinite(scalars))
    if interpolation_mode == "exact":
        np.testing.assert_allclose(scalars, values[1, 1, 1])
    else:
        assert float(np.min(scalars)) >= float(np.min(values))
        assert float(np.max(scalars)) <= float(np.max(values))
    for name, expected in before.items():
        np.testing.assert_array_equal(result.arrays[name], expected)


def test_box_snap_and_continuous_grid_line_semantics() -> None:
    renderer = M11VoxelRenderer()
    metadata = _metadata((3, 3, 3))
    snapped = renderer.snap_box_bounds(
        metadata, (101.7, 104.2, 202.6, 206.2, 303.8, 308.3)
    )
    assert snapped == (102.0, 104.0, 203.0, 206.0, 304.0, 308.0)
    plotter = FakePlotter()
    manager = M11LayerManager(plotter)
    result = _result(np.ones((3, 3, 3), dtype=np.float32))
    snapped_record = renderer.render(
        manager,
        metadata,
        result,
        "p32_total",
        None,
        M11DisplayConfig(
            display_mode="box_cutaway",
            box_bounds=snapped,
            show_grid_lines=True,
            snap_box_to_voxel_faces=True,
        ),
    )
    assert plotter.mesh_kwargs[snapped_record.layer_id]["show_edges"] is True
    continuous_record = renderer.render(
        manager,
        metadata,
        result,
        "p32_total",
        None,
        M11DisplayConfig(
            display_mode="box_cutaway",
            box_bounds=(101.5, 104.5, 202.5, 206.5, 303.5, 308.5),
            show_grid_lines=True,
            snap_box_to_voxel_faces=False,
        ),
    )
    assert plotter.mesh_kwargs[continuous_record.layer_id]["show_edges"] is False


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (_metadata((3, 3, 3), origin=(10.0, 20.0, 30.0), spacing=(2.0, 2.0, 2.0)), (10.0, 16.0, 20.0, 26.0, 30.0, 36.0)),
        (_metadata((3, 2, 4), origin=(10.125, -20.25, 30.5), spacing=(2.5, 3.25, 4.75)), (10.125, 17.625, -20.25, -13.75, 30.5, 49.5)),
    ],
)
def test_analysis_bounds_are_authoritative_first_and_last_voxel_edges(metadata, expected) -> None:
    renderer = M11VoxelRenderer()
    assert renderer.analysis_bounds(metadata) == expected
    for axis in range(3):
        edges = renderer.voxel_axis_edges(metadata, axis)
        assert edges[0] == expected[2 * axis]
        assert edges[-1] == expected[2 * axis + 1]
        assert len(edges) == metadata.shape[axis] + 1


def test_box_domain_boundary_tolerance_normalizes_only_nearby_roundoff() -> None:
    domain = (1000.0, 1100.0, -250.0, -150.0, 20.0, 120.0)
    tolerance = 100.0e-9
    near = (
        domain[0] - tolerance * 0.5,
        domain[1] + tolerance * 0.5,
        domain[2] - tolerance * 0.5,
        domain[3] + tolerance * 0.5,
        domain[4] - tolerance * 0.5,
        domain[5] + tolerance * 0.5,
    )
    assert M11VoxelRenderer.validate_box_bounds(near, domain) == domain
    assert M11VoxelRenderer.validate_box_bounds(domain, domain) == domain
    outside = list(domain)
    outside[1] += tolerance * 2.0
    with pytest.raises(ValueError, match="inside the Analysis Domain"):
        M11VoxelRenderer.validate_box_bounds(tuple(outside), domain)


def test_snap_accepts_all_six_outer_analysis_domain_faces() -> None:
    metadata = _metadata(
        (4, 3, 2),
        origin=(123.123456789, -456.987654321, 0.000000123),
        spacing=(1.25, 2.75, 3.5),
    )
    domain = M11VoxelRenderer.analysis_bounds(metadata)
    assert M11VoxelRenderer.snap_box_bounds(metadata, domain) == domain


@pytest.mark.parametrize("interpolation_mode", ["exact", "smooth"])
def test_full_domain_box_matches_uncut_valid_volume_surface(interpolation_mode) -> None:
    metadata = _metadata((3, 3, 3), origin=(10.125, 20.25, 30.5), spacing=(2.5, 3.25, 4.75))
    values = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    states = np.full(values.shape, CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], np.uint8)
    states[0, 0, 0] = CELL_STATE_CODES[VoxelCellState.TRUE_ZERO]
    values[0, 0, 0] = 0.0
    states[1, 1, 1] = CELL_STATE_CODES[VoxelCellState.NO_DATA]
    values[2, 2, 2] = np.nan
    result = _result(values, states)
    before = {name: array.copy() for name, array in result.arrays.items()}
    renderer = M11VoxelRenderer()
    volume = renderer.prepare_display(
        metadata,
        result,
        "p32_total",
        None,
        M11DisplayConfig(display_mode="voxel_cells", interpolation_mode=interpolation_mode),
    )
    outer = renderer.prepare_display(
        metadata,
        result,
        "p32_total",
        None,
        M11DisplayConfig(display_mode="outer_surface", interpolation_mode=interpolation_mode),
    )
    boxed = renderer.prepare_display(
        metadata,
        result,
        "p32_total",
        None,
        M11DisplayConfig(
            display_mode="box_cutaway",
            interpolation_mode=interpolation_mode,
            box_bounds=renderer.analysis_bounds(metadata),
            snap_box_to_voxel_faces=True,
        ),
    )
    assert volume.mesh.n_cells == 25
    assert boxed.mesh.n_cells == outer.mesh.n_cells
    assert boxed.mesh.area == pytest.approx(outer.mesh.area)
    assert tuple(boxed.mesh.bounds) == pytest.approx(renderer.analysis_bounds(metadata))
    for name, expected in before.items():
        np.testing.assert_array_equal(result.arrays[name], expected)


def test_box_cutaway_does_not_fill_no_data_and_rejects_empty_overlap() -> None:
    values = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    states = np.full(values.shape, CELL_STATE_CODES[VoxelCellState.NO_DATA], np.uint8)
    states[0, 0, 0] = CELL_STATE_CODES[VoxelCellState.TRUE_ZERO]
    values[0, 0, 0] = 0.0
    renderer = M11VoxelRenderer()
    kept = renderer.prepare_display(
        _metadata((3, 3, 3)),
        _result(values, states),
        "p32_total",
        None,
        M11DisplayConfig(
            display_mode="box_cutaway",
            box_bounds=(100.0, 102.0, 200.0, 203.0, 300.0, 304.0),
        ),
    )
    np.testing.assert_array_equal(kept.mesh.cell_data["p32_total"], np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="does not overlap"):
        renderer.prepare_display(
            _metadata((3, 3, 3)),
            _result(values, states),
            "p32_total",
            None,
            M11DisplayConfig(
                display_mode="box_cutaway",
                box_bounds=(104.0, 106.0, 206.0, 209.0, 308.0, 312.0),
            ),
        )


def test_box_cutaway_render_failure_leaves_no_actor_bar_or_widget() -> None:
    renderer = M11VoxelRenderer()
    plotter = FakePlotter()
    plotter.fail_scalar_bar = True
    manager = M11LayerManager(plotter)
    with pytest.raises(RuntimeError, match="scalar-bar"):
        renderer.render(
            manager,
            _metadata((3, 3, 3)),
            _result(np.ones((3, 3, 3), dtype=np.float32)),
            "p32_total",
            None,
            M11DisplayConfig(
                display_mode="box_cutaway",
                box_bounds=(102.0, 104.0, 203.0, 206.0, 304.0, 308.0),
                interactive_plane=True,
            ),
        )
    assert manager.list_layers() == []
    assert set(plotter.renderer.actors) == {"m9_slice:keep"}
    assert set(plotter.scalar_bars) == {"m9_scalar:keep"}
    assert not any(widget.enabled for widget in plotter.widgets)


@pytest.mark.parametrize("interpolation_mode", ["exact", "smooth"])
def test_cutaway_clips_valid_volume_then_exposes_scalar_coloured_internal_face(interpolation_mode) -> None:
    renderer = M11VoxelRenderer()
    values = np.arange(12, dtype=np.float32).reshape((3, 2, 2))
    result = _result(values)
    config = M11DisplayConfig(
        display_mode="cutaway",
        interpolation_mode=interpolation_mode,
        plane_origin=(101.5, 203.0, 304.0),
        plane_normal=(1.0, 0.0, 0.0),
    )
    prepared = renderer.prepare_display(_metadata(), result, "p32_total", None, config)
    assert prepared.display_mode == "cutaway"
    assert prepared.mesh.bounds.x_min == pytest.approx(101.5)
    assert prepared.mesh.bounds.x_max == pytest.approx(106.0)
    faces = []
    cursor = 0
    while cursor < prepared.mesh.faces.size:
        count = int(prepared.mesh.faces[cursor])
        faces.append(prepared.mesh.faces[cursor + 1 : cursor + 1 + count])
        cursor += count + 1
    assert any(np.allclose(prepared.mesh.points[face, 0], 101.5) for face in faces)
    scalars = (
        prepared.mesh.cell_data["p32_total"]
        if interpolation_mode == "exact"
        else prepared.mesh.point_data["p32_total"]
    )
    assert np.all(np.isfinite(scalars))


def test_cutaway_flip_side_has_distinct_snapshot_identity_and_opposite_geometry() -> None:
    renderer = M11VoxelRenderer()
    result = _result(np.arange(12, dtype=np.float32).reshape((3, 2, 2)))
    common = {
        "display_mode": "cutaway",
        "plane_origin": (103.0, 203.0, 304.0),
        "plane_normal": (1.0, 0.0, 0.0),
    }
    positive = renderer.prepare_display(_metadata(), result, "p32_total", None, M11DisplayConfig(**common))
    negative = renderer.prepare_display(
        _metadata(), result, "p32_total", None, M11DisplayConfig(**common, flip_side=True)
    )
    assert positive.mesh.bounds.x_min == pytest.approx(103.0)
    assert negative.mesh.bounds.x_max == pytest.approx(103.0)
    assert positive.layer_id != negative.layer_id


def test_cutaway_keeps_true_zero_and_does_not_bridge_no_data_cells() -> None:
    values = np.array([0.0, 9999.0, 2.0], dtype=np.float32).reshape((3, 1, 1))
    states = np.array(
        [
            CELL_STATE_CODES[VoxelCellState.TRUE_ZERO],
            CELL_STATE_CODES[VoxelCellState.NO_DATA],
            CELL_STATE_CODES[VoxelCellState.MODELED_VALUE],
        ],
        dtype=np.uint8,
    ).reshape((3, 1, 1))
    prepared = M11VoxelRenderer().prepare_display(
        _metadata((3, 1, 1)),
        _result(values, states),
        "p32_total",
        None,
        M11DisplayConfig(
            display_mode="cutaway",
            interpolation_mode="smooth",
            plane_origin=(100.0, 201.5, 302.0),
            plane_normal=(1.0, 0.0, 0.0),
        ),
    )
    scalars = np.asarray(prepared.mesh.point_data["p32_total"])
    assert np.any(scalars == 0.0)
    assert 9999.0 not in scalars
    assert np.all(np.isfinite(scalars))


@pytest.mark.parametrize(
    "bounds",
    [None, (0.0, 0.0, 0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 2.0, 1.0, 0.0, 1.0)],
)
def test_invalid_display_clip_bounds_are_rejected(bounds) -> None:
    with pytest.raises(ValueError, match="Box Cutaway"):
        M11VoxelRenderer.validate_box_bounds(bounds)


def test_box_bounds_outside_analysis_domain_are_rejected() -> None:
    with pytest.raises(ValueError, match="Analysis Domain"):
        M11VoxelRenderer.validate_box_bounds(
            (99.0, 104.0, 200.0, 206.0, 300.0, 308.0),
            (100.0, 106.0, 200.0, 206.0, 300.0, 308.0),
        )
