"""CPU-side M11 interactive cloud preparation benchmark (no OpenGL rendering)."""

from __future__ import annotations

import json
import platform
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from dfn_cave_studio.models.m11 import M11SecondVoxelizationResult
from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager
from dfn_cave_studio.visualization.m11_voxel_renderer import M11DisplayConfig, M11VoxelRenderer
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES


class _Actor:
    mapper = object()

    def SetVisibility(self, visible):
        self.visible = visible

    def GetProperty(self):
        return SimpleNamespace(SetOpacity=lambda value: None)


class _ScalarBar:
    def SetVisibility(self, visible):
        self.visible = visible

    def SetTitle(self, title):
        self.title = title


class _Widget:
    def __init__(self):
        self.enabled = True

    def Off(self):
        self.enabled = False

    def SetEnabled(self, enabled):
        self.enabled = bool(enabled)


class _Plotter:
    def __init__(self):
        self.renderer = SimpleNamespace(actors={})
        self.scalar_bars = {}
        self.widgets = []

    def add_mesh(self, mesh, *, name, **kwargs):
        del mesh, kwargs
        actor = _Actor()
        self.renderer.actors[name] = actor
        return actor

    def remove_actor(self, actor, *, render=False):
        del render
        for name, candidate in list(self.renderer.actors.items()):
            if candidate is actor:
                self.renderer.actors.pop(name)

    def add_scalar_bar(self, *, title, mapper, render=False):
        del mapper, render
        if title in self.scalar_bars:
            return None
        actor = _ScalarBar()
        self.scalar_bars[title] = actor
        return actor

    def remove_scalar_bar(self, *, title, render=False):
        del render
        self.scalar_bars.pop(title)

    def render(self):
        pass

    def add_plane_widget(self, callback, **kwargs):
        del callback, kwargs
        widget = _Widget()
        self.widgets.append(widget)
        return widget


def _case(renderer, metadata, result, config):
    started = perf_counter()
    prepared = renderer.prepare_display(metadata, result, "p32_total", None, config)
    elapsed = perf_counter() - started
    return {
        "seconds": elapsed,
        "cells": prepared.mesh.n_cells,
        "points": prepared.mesh.n_points,
        "array_and_point_bytes": renderer.estimated_display_bytes(prepared),
    }


def main() -> None:
    shape = (69, 50, 47)
    count = int(np.prod(shape))
    values = np.linspace(0.0, 1.0, count, dtype=np.float32).reshape(shape)
    states = np.full(shape, CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], np.uint8)
    metadata = ParameterFieldMetadata(
        shape=shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(5.0, 5.0, 5.0),
        field_names=["p32_total"],
        set_ids=[1],
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=42,
        estimated_bytes=count * 4,
    )
    result = M11SecondVoxelizationResult(
        realization_id="m11-benchmark-162150",
        source_m10_realization_id="benchmark",
        source_m10_config_hash="benchmark",
        source_parameter_field_hash="benchmark",
        arrays={"p32_total": values, "cell_state": states},
    )
    cases = {}
    configs = {
        "outer_surface_exact": M11DisplayConfig(display_mode="outer_surface"),
        "outer_surface_smooth": M11DisplayConfig(display_mode="outer_surface", interpolation_mode="smooth"),
        "section_x": M11DisplayConfig(display_mode="orthogonal_section", axis="x"),
        "section_y": M11DisplayConfig(display_mode="orthogonal_section", axis="y"),
        "section_z": M11DisplayConfig(display_mode="orthogonal_section", axis="z"),
        "arbitrary_plane": M11DisplayConfig(
            display_mode="arbitrary_plane",
            plane_origin=(172.5, 125.0, 117.5),
            plane_normal=(1.0, 1.0, 1.0),
        ),
        "clipped_outer_surface": M11DisplayConfig(
            display_mode="outer_surface",
            interactive_slot="surface",
            clip_enabled=True,
            clip_bounds=(0.0, 172.5, 0.0, 250.0, 0.0, 235.0),
        ),
    }
    for name, config in configs.items():
        renderer = M11VoxelRenderer()
        cases[name] = _case(renderer, metadata, result, config)

    renderer = M11VoxelRenderer()
    manager = M11LayerManager(_Plotter())
    config = M11DisplayConfig(
        display_mode="orthogonal_section",
        axis="z",
        section_coordinate=117.5,
        interactive_slot="xyz_z",
        interactive_plane=True,
    )
    started = perf_counter()
    renderer.render(manager, metadata, result, "p32_total", None, config)
    first_render = perf_counter() - started
    started = perf_counter()
    for coordinate in np.linspace(10.0, 225.0, 10):
        renderer.render(
            manager,
            metadata,
            result,
            "p32_total",
            None,
            M11DisplayConfig(
                display_mode="orthogonal_section",
                axis="z",
                section_coordinate=float(coordinate),
                interactive_slot="xyz_z",
                interactive_plane=coordinate == 10.0,
            ),
        )
    repeat_render = perf_counter() - started
    output = {
        "benchmark": "m11-interactive-cloud-cpu-no-opengl-2",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "voxel_shape": shape,
        "voxel_count": count,
        "cases": cases,
        "first_actor_render_seconds": first_render,
        "ten_interactive_updates_seconds": repeat_render,
        "actor_count_after_repeat": len(manager.list_layers()),
        "scalar_bar_count_after_repeat": len(manager.plotter.scalar_bars),
        "active_widget_count_after_repeat": sum(widget.enabled for widget in manager.plotter.widgets),
        "note": "CPU-side VTK geometry and actor registration only; real OpenGL remains for manual acceptance.",
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
