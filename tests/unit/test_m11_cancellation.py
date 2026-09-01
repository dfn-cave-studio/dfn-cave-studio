"""Cooperative cancellation regressions for M11 exact second voxelization."""

from __future__ import annotations

import math
import multiprocessing
import threading
import time

import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.models.m10 import M10Realization
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES
from dfn_cave_studio.voxel.second_voxelization import M11SecondVoxelizer, SecondVoxelizationConfig


def _large_candidate_voxelizer(
    *, worker_count: int, cancelled, fracture_count: int = 1
) -> M11SecondVoxelizer:
    shape = (60, 60, 1)
    metadata = ParameterFieldMetadata(
        shape=shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        field_names=[],
        set_ids=[1],
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=42,
        estimated_bytes=0,
    )
    centers = np.tile([[30.0, 30.0, 0.5]], (fracture_count, 1))
    normals = np.tile([[0.0, 0.0, 1.0]], (fracture_count, 1))
    radii = np.full(fracture_count, 45.0)
    realization = M10Realization(
        realization_id="cancel-large",
        realization_index=0,
        seed=42,
        config_hash="config",
        geometry_arrays={
            "center": centers,
            "normal": normals,
            "radius": radii,
            "ordinal": np.arange(fracture_count, dtype=np.uint64),
            "set_id": np.ones(fracture_count, dtype=np.int16),
            "clipped_area": math.pi * np.square(radii),
        },
    )
    modeled = CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
    parameters = {
        "cell_state": np.full(shape, modeled, dtype=np.uint8),
        "domain_id": np.ones(shape, dtype=np.int16),
        "set_1_p32": np.ones(shape, dtype=np.float32),
    }
    bounds = ModelBounds(x_min=0, x_max=60, y_min=0, y_max=60, z_min=0, z_max=1)
    return M11SecondVoxelizer(
        metadata,
        parameters,
        bounds,
        realization,
        SecondVoxelizationConfig(worker_count=worker_count, fracture_batch_size=1),
        cancelled=cancelled,
    )


def test_single_large_fracture_checks_cancel_inside_candidate_voxel_loop() -> None:
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 4

    with pytest.raises(InterruptedError):
        _large_candidate_voxelizer(worker_count=1, cancelled=cancelled).compute()
    assert checks < 20


def test_spawn_workers_cancel_cooperatively_and_leave_no_owned_processes() -> None:
    cancel = threading.Event()
    baseline = {process.pid for process in multiprocessing.active_children()}
    outcome: list[BaseException | object] = []
    voxelizer = _large_candidate_voxelizer(
        worker_count=2, cancelled=cancel.is_set, fracture_count=12
    )

    def run() -> None:
        try:
            outcome.append(voxelizer.compute())
        except Exception as exc:  # noqa: BLE001 - capture background-thread outcome for assertion
            outcome.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(0.25)
    cancel.set()
    thread.join(timeout=15.0)
    assert not thread.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], InterruptedError)
    assert voxelizer._submitted_batch_count <= 4
    assert voxelizer._submitted_batch_count < 12
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        remaining = {process.pid for process in multiprocessing.active_children()} - baseline
        if not remaining:
            break
        time.sleep(0.05)
    assert not ({process.pid for process in multiprocessing.active_children()} - baseline)


def test_new_calculation_succeeds_after_cancelled_calculation() -> None:
    with pytest.raises(InterruptedError):
        _large_candidate_voxelizer(worker_count=1, cancelled=lambda: True).compute()
    result = _large_candidate_voxelizer(
        worker_count=1, cancelled=lambda: False, fracture_count=1
    ).compute()
    assert result.complete
    assert result.positive_intersection_count > 0
