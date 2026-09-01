"""Unit tests for the session-only M11 slice registry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager


class ActorProperty:
    def __init__(self) -> None:
        self.opacity = 1.0

    def SetOpacity(self, opacity: float) -> None:
        self.opacity = opacity


class Actor:
    def __init__(self) -> None:
        self.visible = True
        self.actor_property = ActorProperty()

    def SetVisibility(self, visible: bool) -> None:
        self.visible = visible

    def GetProperty(self) -> ActorProperty:
        return self.actor_property

    def GetMapper(self):
        return object()


class ScalarBarActor:
    def __init__(self) -> None:
        self.visible = True
        self.title = ""

    def SetVisibility(self, visible: bool) -> None:
        self.visible = visible

    def SetTitle(self, title: str) -> None:
        self.title = title


class Plotter:
    def __init__(self) -> None:
        self.renderer = SimpleNamespace(actors={"m9_slice:keep": Actor(), "m10_dfn:keep": Actor()})
        self.scalar_bars = {"m9_scalar_bar:keep": ScalarBarActor()}

    def remove_actor(self, actor, *, render=False) -> None:
        del render
        for name, candidate in list(self.renderer.actors.items()):
            if candidate is actor:
                self.renderer.actors.pop(name)

    def render(self) -> None:
        pass

    def add_scalar_bar(self, *, title, mapper, render=False):
        del mapper, render
        if title in self.scalar_bars:
            return None
        actor = ScalarBarActor()
        self.scalar_bars[title] = actor
        return actor

    def remove_scalar_bar(self, *, title, render=False) -> None:
        del render
        self.scalar_bars.pop(title)


def _add(manager: M11LayerManager, *, field="p32_total", axis="z", index=1, pending=False):
    layer_id = manager.layer_id("m11-r1", field, None, axis, index)
    actor = Actor()
    manager.plotter.renderer.actors[layer_id] = actor
    scalar_bar_id = manager.scalar_bar_id("m11-r1", field, None)
    scalar_bar_actor = manager.create_or_get_scalar_bar(
        scalar_bar_id, f"{field} (m^-1)", actor, attach_mapper=True
    )
    return manager.add_or_replace(
        layer_id,
        actor,
        scalar_bar_id=scalar_bar_id,
        scalar_bar_actor=scalar_bar_actor,
        scalar_bar_title=f"{field} (m^-1)",
        realization_id="m11-r1",
        field_name=field,
        joint_set_id=None,
        axis=axis,
        slice_index=index,
        coordinate=index + 0.5,
        opacity=1.0,
        pending=pending,
    )


def test_layer_identity_includes_realization_field_set_axis_and_slice() -> None:
    assert M11LayerManager.layer_id("r1", "p32_total", 2, "X", 7) == (
        "m11_voxel:r1:orthogonal_section:p32_total:set_2:x:7:exact"
    )


def test_visibility_opacity_remove_and_namespace_clear() -> None:
    manager = M11LayerManager(Plotter())
    record = _add(manager)
    assert manager.set_visible(record.layer_id, False)
    assert not record.actor.visible
    assert manager.set_opacity(record.layer_id, 0.25)
    assert record.actor.actor_property.opacity == 0.25
    with pytest.raises(ValueError, match="between 0 and 1"):
        manager.set_opacity(record.layer_id, 2.0)
    assert manager.clear_m11_layers() == 1
    assert set(manager.plotter.renderer.actors) == {"m9_slice:keep", "m10_dfn:keep"}
    assert set(manager.plotter.scalar_bars) == {"m9_scalar_bar:keep"}


def test_pending_cleanup_does_not_remove_committed_layers() -> None:
    manager = M11LayerManager(Plotter())
    pending = _add(manager, field="p32_total", pending=True)
    committed = _add(manager, field="p32_subgrid", pending=False)
    assert manager.clear_pending("m11-r1") == 1
    assert not manager.contains(pending.layer_id)
    assert manager.contains(committed.layer_id)


def test_shared_scalar_bar_is_hidden_and_removed_only_after_last_layer() -> None:
    manager = M11LayerManager(Plotter())
    first = _add(manager, index=1)
    second = _add(manager, index=2)
    assert first.scalar_bar_actor is second.scalar_bar_actor
    assert len([key for key in manager.plotter.scalar_bars if key.startswith("m11_scalar_bar:")]) == 1

    manager.set_visible(first.layer_id, False)
    assert first.scalar_bar_actor.visible
    manager.set_visible(second.layer_id, False)
    assert not first.scalar_bar_actor.visible
    manager.set_visible(first.layer_id, True)
    assert first.scalar_bar_actor.visible

    manager.remove(first.layer_id)
    assert second.scalar_bar_id in manager.plotter.scalar_bars
    manager.remove(second.layer_id)
    assert second.scalar_bar_id not in manager.plotter.scalar_bars
