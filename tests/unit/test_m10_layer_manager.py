"""DFN layer namespace regression tests without OpenGL."""

from unittest.mock import MagicMock

import numpy as np

from dfn_cave_studio.models.m10 import M10Realization
from dfn_cave_studio.visualization.dfn_layer_manager import DFNLayerManager


class FakePlotter:
    def __init__(self):
        self.remove_actor = MagicMock()
        self.render = MagicMock()


class FakeGlyphPlotter(FakePlotter):
    def add_actor(self, actor, **kwargs):
        self.actor = actor
        return actor


def test_add_replace_and_clear_are_namespace_scoped():
    plotter = FakePlotter()
    manager = DFNLayerManager(plotter)
    first, second = MagicMock(), MagicMock()
    metadata = {
        "realization_id": "r1",
        "set_id": 1,
        "source": None,
        "opacity": 0.7,
        "color": "blue",
        "fracture_count": 10,
    }
    manager.add_or_replace("dfn:realization:r1:set:1", first, **metadata)
    for _ in range(10):
        manager.add_or_replace("dfn:realization:r1:set:1", second, **metadata)
    assert len(manager.list_layers()) == 1
    assert manager.list_layers()[0].actor is second
    assert manager.set_visible("missing", False) is False
    assert manager.remove("missing") is False
    assert manager.clear_dfn_layers() == 1
    assert len(manager.list_layers()) == 0
    assert not hasattr(plotter, "clear")


def test_visibility_and_opacity_update_actor():
    manager = DFNLayerManager(FakePlotter())
    actor = MagicMock()
    manager.add_or_replace(
        "dfn:realization:r1:all",
        actor,
        realization_id="r1",
        set_id=None,
        source=None,
        opacity=0.7,
        color="blue",
        fracture_count=3,
    )
    assert manager.set_visible("dfn:realization:r1:all", False)
    actor.SetVisibility.assert_called_with(False)
    assert manager.set_opacity("dfn:realization:r1:all", 0.25)
    actor.GetProperty.return_value.SetOpacity.assert_called_with(0.25)


def test_lod_uses_one_gpu_glyph_actor_and_does_not_expand_disc_vertices():
    count = 10_000
    realization = M10Realization(
        realization_id="r1",
        realization_index=0,
        seed=42,
        config_hash="hash",
        geometry_arrays={
            "center": np.zeros((count, 3), dtype=np.float64),
            "normal": np.tile((0.0, 0.0, 1.0), (count, 1)),
            "radius": np.ones(count, dtype=np.float64),
            "set_id": np.ones(count, dtype=np.int16),
            "source_code": np.zeros(count, dtype=np.uint8),
        },
    )
    plotter = FakeGlyphPlotter()
    manager = DFNLayerManager(plotter)
    layer = manager.render_realization(realization, mode="lod")
    assert layer.fracture_count == count
    assert len(manager.list_layers()) == 1
    mapper_input = plotter.actor.GetMapper().GetInput()
    assert mapper_input.GetNumberOfPoints() == count
    assert "vertices" not in realization.geometry_arrays
