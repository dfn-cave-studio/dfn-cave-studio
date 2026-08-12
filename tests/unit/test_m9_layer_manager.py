"""Unit regressions for the session-only M9 slice registry."""

from dfn_cave_studio.visualization.m9_layer_manager import M9LayerManager
from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.visualization.parameter_field_renderer import ParameterFieldRenderer


class _Property:
    def __init__(self) -> None:
        self.opacity = 1.0

    def SetOpacity(self, opacity: float) -> None:
        self.opacity = opacity


class _Actor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.visible = True
        self.property = _Property()

    def SetVisibility(self, visible: bool) -> None:
        self.visible = visible

    def GetProperty(self) -> _Property:
        return self.property


class _Plotter:
    def __init__(self) -> None:
        self.actors: list[_Actor] = []
        self.render_count = 0

    def remove_actor(self, actor: _Actor, render: bool = True) -> None:
        if actor in self.actors:
            self.actors.remove(actor)
        if render:
            self.render()

    def render(self) -> None:
        self.render_count += 1


def _add(manager: M9LayerManager, layer_id: str, actor: _Actor, opacity: float = 0.8) -> None:
    manager.plotter.actors.append(actor)
    _, field, axis, index = layer_id.split(":")
    manager.add_or_replace(
        layer_id,
        actor,
        field_name=field,
        axis=axis,
        slice_index=int(index),
        coordinate=float(index),
        opacity=opacity,
    )


def test_same_slice_replaced_ten_times_has_one_actor_and_record() -> None:
    plotter = _Plotter()
    manager = M9LayerManager(plotter)
    for attempt in range(10):
        _add(manager, "m9_slice:p32_total:z:3", _Actor(f"actor-{attempt}"))

    assert len(manager.list_layers()) == 1
    assert len(plotter.actors) == 1
    assert manager.get("m9_slice:p32_total:z:3").actor.name == "actor-9"


def test_visibility_opacity_remove_and_clear_are_namespaced() -> None:
    plotter = _Plotter()
    non_m9 = _Actor("voxel_analysis_domain")
    plotter.actors.append(non_m9)
    manager = M9LayerManager(plotter)
    first = _Actor("first")
    second = _Actor("second")
    _add(manager, "m9_slice:p32_total:z:3", first)
    _add(manager, "m9_slice:p32_set_1:x:5", second)

    assert manager.set_visible("m9_slice:p32_total:z:3", False)
    assert first.visible is False
    assert manager.set_visible("m9_slice:p32_total:z:3", True)
    assert first.visible is True
    assert manager.set_opacity("m9_slice:p32_total:z:3", 0.25)
    assert first.property.opacity == 0.25
    assert manager.remove("m9_slice:p32_total:z:3")
    assert first not in plotter.actors
    assert second in plotter.actors
    assert manager.clear_m9_layers() == 1
    assert non_m9 in plotter.actors
    assert manager.list_layers() == []


def test_missing_layer_operations_are_safe_and_invalid_registration_fails() -> None:
    manager = M9LayerManager(_Plotter())
    assert manager.remove("m9_slice:missing:z:0") is False
    assert manager.set_visible("m9_slice:missing:z:0", False) is False
    assert manager.set_opacity("m9_slice:missing:z:0", 0.5) is False

    try:
        manager.add_or_replace(
            "other:field:z:0",
            _Actor("bad"),
            field_name="field",
            axis="z",
            slice_index=0,
            coordinate=0,
            opacity=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-M9 actor namespace was accepted")


def test_slice_location_and_actor_name_are_stable() -> None:
    metadata = ParameterFieldMetadata(
        shape=(4, 5, 6),
        origin=(10, 20, 30),
        spacing=(2, 3, 4),
        field_names=["p32_set_1"],
        set_ids=[1],
        density_method=DensityMethod.IDW,
        random_seed=42,
        estimated_bytes=1,
    )
    assert ParameterFieldRenderer.slice_location(metadata, "z", 0.5) == (2, 40.0)
    assert ParameterFieldRenderer.layer_id("p32_set_1", "z", 2) == "m9_slice:p32_set_1:z:2"
