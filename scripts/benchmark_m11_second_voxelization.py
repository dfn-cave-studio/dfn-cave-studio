"""Reproducible synthetic benchmark for M11.1 exact second voxelization."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.m10 import M10Realization
from dfn_cave_studio.models.m9 import DensityMethod, ParameterFieldMetadata
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig
from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES
from dfn_cave_studio.voxel.second_voxelization import M11SecondVoxelizer, SecondVoxelizationConfig


def process_memory_bytes() -> tuple[int, int]:
    """Return current RSS and peak working-set bytes using the standard library."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        memory_info.restype = wintypes.BOOL
        handle = get_process()
        if memory_info(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
        return 0, 0
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        peak *= 1024
    return peak, peak


def system_cpu_times() -> tuple[int, int, int] | None:
    """Return Windows system idle/kernel/user ticks for an auditable CPU-load estimate."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    idle = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None

    def ticks(value: wintypes.FILETIME) -> int:
        return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

    return ticks(idle), ticks(kernel), ticks(user)


def benchmark(count: int, worker_count: int) -> dict[str, object]:
    """Run one deterministic contained-disk benchmark and return audit data."""
    shape = (100, 100, 10)
    rng = np.random.default_rng(20260828)
    centers = np.column_stack(
        (
            rng.uniform(0.2, 99.8, count),
            rng.uniform(0.2, 99.8, count),
            rng.uniform(0.2, 9.8, count),
        )
    )
    normals = np.zeros((count, 3), dtype=np.float64)
    normals[:, 2] = 1.0
    radii = np.full(count, 0.1, dtype=np.float64)
    metadata = ParameterFieldMetadata(
        shape=shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        field_names=[],
        set_ids=[1],
        density_method=DensityMethod.GLOBAL_CONSTANT,
        random_seed=20260828,
        estimated_bytes=0,
    )
    state = CELL_STATE_CODES[VoxelCellState.MODELED_VALUE]
    parameter_arrays = {
        "cell_state": np.full(shape, state, dtype=np.uint8),
        "domain_id": np.ones(shape, dtype=np.int16),
        "set_1_p32": np.full(shape, 0.25, dtype=np.float32),
    }
    realization = M10Realization(
        realization_id=f"benchmark-{count}",
        realization_index=0,
        seed=20260828,
        config_hash="m11-benchmark-v1",
        geometry_arrays={
            "center": centers,
            "normal": normals,
            "radius": radii,
            "ordinal": np.arange(count, dtype=np.uint64),
            "set_id": np.ones(count, dtype=np.int16),
            "source_code": np.zeros(count, dtype=np.uint8),
            "size_class": np.full(count, 3, dtype=np.uint8),
            "clipped_area": np.full(count, np.pi * 0.01, dtype=np.float64),
            "p32_subgrid_set_1": np.zeros(shape, dtype=np.float32),
        },
    )
    bounds = ModelBounds(x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=10)
    rss_before, _ = process_memory_bytes()
    cpu_before = system_cpu_times()
    started = time.perf_counter()
    result = M11SecondVoxelizer(
        metadata,
        parameter_arrays,
        bounds,
        realization,
        SecondVoxelizationConfig(worker_count=worker_count),
    ).compute()
    elapsed = time.perf_counter() - started
    cpu_after = system_cpu_times()
    average_system_cpu_percent = None
    if cpu_before is not None and cpu_after is not None:
        idle_delta = cpu_after[0] - cpu_before[0]
        total_delta = (cpu_after[1] - cpu_before[1]) + (cpu_after[2] - cpu_before[2])
        if total_delta > 0:
            average_system_cpu_percent = 100.0 * (total_delta - idle_delta) / total_delta
    project = Project()
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=bounds, generation_domain=bounds)
    project.m9_state.parameter_field_metadata = metadata
    project.m9_state.parameter_field_arrays = parameter_arrays
    project.m10_state.realizations = [realization]
    project.m11_state.results = [result]
    with tempfile.TemporaryDirectory(prefix="m11-benchmark-") as directory:
        archive_path = Path(directory) / "m11-benchmark.dfnproj"
        save_started = time.perf_counter()
        ZipProjectStore().save(project, archive_path)
        save_seconds = time.perf_counter() - save_started
        reopen_started = time.perf_counter()
        reopened_project = ZipProjectStore().load(archive_path)
        reopen_seconds = time.perf_counter() - reopen_started
        archive_bytes = archive_path.stat().st_size
    reopened = reopened_project.m11_state.results[0].arrays
    assert reopened.keys() == result.arrays.keys()
    rss_after, peak_working_set = process_memory_bytes()
    return {
        "fracture_count": count,
        "worker_count": worker_count,
        "elapsed_seconds": elapsed,
        "fractures_per_second": count / elapsed,
        "average_system_cpu_percent": average_system_cpu_percent,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "parent_peak_working_set_bytes": peak_working_set,
        "result_array_bytes": sum(array.nbytes for array in result.arrays.values()),
        "sparse_bytes": result.sparse_bytes,
        "candidate_pairs": result.candidate_pair_count,
        "positive_pairs": result.positive_intersection_count,
        "candidate_rejection_ratio": (
            result.rejected_candidate_count / result.candidate_pair_count if result.candidate_pair_count else 0.0
        ),
        "dfnproj_save_seconds": save_seconds,
        "dfnproj_reopen_seconds": reopen_seconds,
        "dfnproj_bytes": archive_bytes,
        "conservation_absolute_error": result.conservation.absolute_error_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="+", type=int, default=[10_000, 100_000, 721_786])
    parser.add_argument("--workers", type=int, default=1, choices=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/m11_second_voxelization.json"))
    args = parser.parse_args()
    payload = {
        "benchmark_version": "m11-second-voxelization-benchmark-1",
        "algorithm_version": "m11-second-voxelization-1",
        "geometry_kernel_version": "analytic-circle-convex-polygon-1",
        "source_geometry": "synthetic contained horizontal disks; radius=0.1m; 100x100x10 unit grid",
        "generated_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "cases": [benchmark(count, args.workers) for count in args.counts],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = [
        "# M11.1 Exact Second Voxelization Benchmark",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "CPU percentage is the system-wide Windows load during the case. Peak memory is the parent-process working set and does not include simultaneous child working sets.",
        "Save/reopen timings use a complete `.dfnproj` containing the synthetic M9 field, M10 geometry, and M11 result.",
        "",
        "| Fractures | Workers | Seconds | Fractures/s | CPU % machine | Candidate pairs | Positive pairs | Result MiB | Parent peak MiB | Save s | Reopen s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        cpu_percent = case["average_system_cpu_percent"]
        cpu_text = "—" if cpu_percent is None else f"{cpu_percent:.1f}"
        markdown.append(
            f"| {case['fracture_count']:,} | {case['worker_count']} | {case['elapsed_seconds']:.3f} | "
            f"{case['fractures_per_second']:,.0f} | {cpu_text} | "
            f"{case['candidate_pairs']:,} | {case['positive_pairs']:,} | "
            f"{case['result_array_bytes'] / 1024**2:.2f} | "
            f"{case['parent_peak_working_set_bytes'] / 1024**2:.2f} | {case['dfnproj_save_seconds']:.3f} | "
            f"{case['dfnproj_reopen_seconds']:.3f} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
