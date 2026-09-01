"""Measure cooperative M11 cancellation without modifying project state."""

from __future__ import annotations

import json
import math
import multiprocessing
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.models.m10 import M10Realization
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES
from dfn_cave_studio.voxel.second_voxelization import M11SecondVoxelizer, SecondVoxelizationConfig


def voxelizer(worker_count: int, cancelled) -> M11SecondVoxelizer:
    """Build a stable case whose fractures each span many candidate voxels."""
    shape = (80, 80, 1)
    count = 16
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
    radius = np.full(count, 60.0)
    realization = M10Realization(
        realization_id=f"cancel-{worker_count}",
        realization_index=0,
        seed=42,
        config_hash="cancel-benchmark",
        geometry_arrays={
            "center": np.tile([[40.0, 40.0, 0.5]], (count, 1)),
            "normal": np.tile([[0.0, 0.0, 1.0]], (count, 1)),
            "radius": radius,
            "ordinal": np.arange(count, dtype=np.uint64),
            "set_id": np.ones(count, dtype=np.int16),
            "clipped_area": math.pi * np.square(radius),
        },
    )
    arrays = {
        "cell_state": np.full(
            shape, CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], dtype=np.uint8
        ),
        "domain_id": np.ones(shape, dtype=np.int16),
        "set_1_p32": np.ones(shape, dtype=np.float32),
    }
    bounds = ModelBounds(x_min=0, x_max=80, y_min=0, y_max=80, z_min=0, z_max=1)
    return M11SecondVoxelizer(
        metadata,
        arrays,
        bounds,
        realization,
        SecondVoxelizationConfig(worker_count=worker_count, fracture_batch_size=1),
        cancelled=cancelled,
    )


def measure(worker_count: int) -> dict[str, object]:
    """Return request, stop, and owned-process-exit latency for one calculation."""
    cancel = threading.Event()
    baseline = {process.pid for process in multiprocessing.active_children()}
    terminal: list[str] = []

    def run() -> None:
        try:
            voxelizer(worker_count, cancel.is_set).compute()
            terminal.append("finished")
        except InterruptedError:
            terminal.append("cancelled")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(0.25)
    click = time.perf_counter()
    cancel.set()
    feedback = time.perf_counter()
    thread.join(timeout=30.0)
    stopped = time.perf_counter()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not ({process.pid for process in multiprocessing.active_children()} - baseline):
            break
        time.sleep(0.01)
    children_exited = time.perf_counter()
    remaining = sorted({process.pid for process in multiprocessing.active_children()} - baseline)
    return {
        "worker_count": worker_count,
        "cancel_to_ui_feedback_seconds": feedback - click,
        "cancel_to_calculation_stop_seconds": stopped - click,
        "cancel_to_all_owned_processes_exit_seconds": children_exited - click,
        "terminal_state": terminal[0] if terminal else "timeout",
        "thread_alive": thread.is_alive(),
        "remaining_owned_process_ids": remaining,
    }


def main() -> int:
    output = Path("benchmarks/m11_cancellation.json")
    payload = {
        "benchmark_version": "m11-cancellation-1",
        "generated_at": datetime.now(UTC).isoformat(),
        "case": "16 radius-60m disks over an 80x80x1 unit grid; cancel after 0.25s",
        "results": [measure(1), measure(2)],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# M11 Cooperative Cancellation Benchmark",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Workers | UI feedback s | Calculation stop s | Child exit s | Terminal |",
        "|---:|---:|---:|---:|---|",
    ]
    for result in payload["results"]:
        lines.append(
            f"| {result['worker_count']} | {result['cancel_to_ui_feedback_seconds']:.6f} | "
            f"{result['cancel_to_calculation_stop_seconds']:.6f} | "
            f"{result['cancel_to_all_owned_processes_exit_seconds']:.6f} | {result['terminal_state']} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
