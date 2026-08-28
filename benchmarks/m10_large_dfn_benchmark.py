"""Reproducible local benchmark for compact M10 explicit-DFN generation."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Callable

import numpy as np

from dfn_cave_studio.dfn.m10_geometry import authoritative_nbytes
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.m10_service import M10Service
from dfn_cave_studio.visualization.dfn_layer_manager import DFNLayerManager


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("page_fault_count", wintypes.DWORD),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    ]


_MEMORY_FUNCTION = ctypes.windll.kernel32.K32GetProcessMemoryInfo
_MEMORY_FUNCTION.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
_MEMORY_FUNCTION.restype = wintypes.BOOL
_PROCESS_HANDLE = ctypes.windll.kernel32.GetCurrentProcess()
_PROCESS_TIMES_FUNCTION = ctypes.windll.kernel32.GetProcessTimes
_PROCESS_TIMES_FUNCTION.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
]
_PROCESS_TIMES_FUNCTION.restype = wintypes.BOOL


def _rss_bytes() -> int:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    _MEMORY_FUNCTION(_PROCESS_HANDLE, ctypes.byref(counters), counters.cb)
    return int(counters.working_set_size)


def _cpu_seconds() -> float:
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    _PROCESS_TIMES_FUNCTION(
        _PROCESS_HANDLE,
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    )

    def seconds(value: wintypes.FILETIME) -> float:
        ticks = (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
        return ticks / 10_000_000.0

    return seconds(kernel) + seconds(user)


def _measure(function: Callable[[], Any]) -> tuple[Any, dict[str, float | int]]:
    stop = threading.Event()
    rss_samples: list[int] = []
    cpu_samples: list[tuple[float, float]] = []

    def poll() -> None:
        while not stop.is_set():
            cpu_samples.append((time.perf_counter(), _cpu_seconds()))
            rss_samples.append(_rss_bytes())
            time.sleep(0.01)

    baseline_rss = _rss_bytes()
    start_cpu = _cpu_seconds()
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        value = function()
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join()
    cpu_delta = _cpu_seconds() - start_cpu
    logical_cpus = max(1, os.cpu_count() or 1)
    average_cpu = 100.0 * cpu_delta / max(elapsed, 1e-9) / logical_cpus
    peak_cpu = 0.0
    for previous, current in zip(cpu_samples, cpu_samples[1:]):
        wall_delta = current[0] - previous[0]
        if wall_delta > 0:
            peak_cpu = max(peak_cpu, 100.0 * (current[1] - previous[1]) / wall_delta / logical_cpus)
    return value, {
        "elapsed_seconds": elapsed,
        "average_cpu_percent": average_cpu,
        "peak_cpu_percent": min(100.0, peak_cpu),
        "peak_rss_delta_bytes": max(rss_samples or [baseline_rss]) - baseline_rss,
    }


class _NoRenderPlotter:
    def add_actor(self, actor: Any, **_: Any) -> Any:
        return actor

    def remove_actor(self, *_: Any, **__: Any) -> None:
        return None

    def render(self) -> None:
        return None


def _scale_density(project: Any, requested_count: int) -> None:
    service = M10Service(project)
    current = service.estimate_details(project.m10_state.config)["expected_fractures"]
    fixed = len(service._conditioned_observations()) + len(project.m10_state.deterministic_structures)
    factor = max(0.0, (requested_count - fixed) / max(float(current) - fixed, 1e-12))
    for set_id in project.m9_state.parameter_field_metadata.set_ids:
        project.m9_state.parameter_field_arrays[f"set_{set_id}_p32"] *= factor


def run_case(source: Path, requested_count: int) -> dict[str, Any]:
    project = ZipProjectStore().load(source)
    project.m10_state.realizations = []
    project.m10_state.config = project.m10_state.config.model_copy(
        update={"realization_count": 1, "worker_count": 1, "validation_warning_acknowledged": True}
    )
    _scale_density(project, requested_count)
    service = M10Service(project)
    details = service.estimate_details(project.m10_state.config)
    realization, generation = _measure(
        lambda: service.generate_batch(project.m10_state.config, commit=False)[0]
    )
    service.commit_realizations(project.m10_state.config, [realization])
    with tempfile.TemporaryDirectory(prefix="m10-benchmark-") as directory:
        path = Path(directory) / "benchmark.dfnproj"
        _, save = _measure(lambda: ZipProjectStore().save(project, path))
        reopened, reopen = _measure(lambda: ZipProjectStore().load(path))
        manager = DFNLayerManager(_NoRenderPlotter())
        layer, lod = _measure(lambda: manager.render_realization(realization, mode="lod"))
        arrays_equal = all(
            np.array_equal(realization.geometry_arrays[name], reopened.m10_state.realizations[0].geometry_arrays[name])
            for name in realization.geometry_arrays
        )
        file_bytes = path.stat().st_size
    generation["fractures_per_second"] = realization.fracture_count / generation["elapsed_seconds"]
    actual_bytes = authoritative_nbytes(realization)
    estimate_error = abs(int(details["final_storage_bytes"]) - actual_bytes) / max(actual_bytes, 1)
    return {
        "requested_count": requested_count,
        "worker_count": 1,
        "expected_count": details["expected_fractures"],
        "actual_count": realization.fracture_count,
        "stochastic_count": realization.quality.stochastic_count,
        "conditioned_count": realization.quality.conditioned_count,
        "deterministic_count": realization.quality.deterministic_count,
        "generation": generation,
        "generator_version": realization.provenance.get("generator_version"),
        "geometry_format": realization.provenance.get("geometry_format"),
        "threshold_method_version": realization.provenance.get("size_threshold_method"),
        "threshold_mode": project.m10_state.config.size_threshold_mode,
        "enabled_size_classes": list(project.m10_state.config.enabled_size_classes),
        "size_parameters": {
            "small_area_share": project.m10_state.config.small_area_share,
            "medium_large_cumulative_share": project.m10_state.config.medium_large_cumulative_share,
            "manual_small_medium_radius": project.m10_state.config.manual_small_medium_radius,
            "manual_medium_large_radius": project.m10_state.config.manual_medium_large_radius,
        },
        "base_seed": project.m10_state.config.base_seed,
        "seed_strategy": project.m10_state.config.seed_strategy,
        "input_config_hash": realization.config_hash,
        "final_array_bytes": actual_bytes,
        "estimated_final_bytes": details["final_storage_bytes"],
        "final_estimate_error_percent": 100.0 * estimate_error,
        "fracture_array_estimate_bytes": details["fracture_array_bytes"],
        "subgrid_array_estimate_bytes": details["subgrid_array_bytes"],
        "multiscale_metadata_estimate_bytes": details["multiscale_metadata_bytes"],
        "estimated_peak_generation_bytes": details["peak_generation_bytes"],
        "modelled_voxels": details["modelled_voxels"],
        "voxel_volume": details["voxel_volume"],
        "target_breakdown": details["targets"],
        "save": save,
        "saved_project_bytes": file_bytes,
        "reopen": reopen,
        "reopened_arrays_equal": arrays_equal,
        "lod": lod,
        "lod_displayed_count": layer.fracture_count,
        "lod_actor_count": len(manager.list_layers()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", nargs="+", type=int, default=[10_000, 100_000, 690_000])
    arguments = parser.parse_args()
    results = {
        "source_project": str(arguments.project),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "logical_cpu_count": os.cpu_count(),
        "cases": [run_case(arguments.project, count) for count in arguments.counts],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = ["# M10 large explicit-DFN benchmark", "", f"Source: `{arguments.project}`", ""]
    first = results["cases"][0]
    markdown.extend(
        [
            f"Generated: `{results['generated_at']}`",
            f"Python: `{results['python_version']}` | NumPy: `{results['numpy_version']}` | logical CPUs: {results['logical_cpu_count']}",
            f"Generator: `{first['generator_version']}` | geometry: `{first['geometry_format']}` | threshold: `{first['threshold_method_version']}`",
            f"Threshold mode: `{first['threshold_mode']}` | enabled: `{', '.join(first['enabled_size_classes'])}` | base seed: {first['base_seed']} | strategy: `{first['seed_strategy']}`",
            f"Size parameters: `{json.dumps(first['size_parameters'], sort_keys=True)}`",
            "",
        ]
    )
    markdown.append("| Actual | Workers | Time (s) | fractures/s | Peak RSS (MiB) | Arrays (MiB) | Estimate error | Save (s) | Reopen (s) | LOD (s) |")
    markdown.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case in results["cases"]:
        markdown.append(
            f"| {case['actual_count']:,} | {case['worker_count']} | {case['generation']['elapsed_seconds']:.3f} | "
            f"{case['generation']['fractures_per_second']:,.0f} | {case['generation']['peak_rss_delta_bytes']/1024**2:.1f} | "
            f"{case['final_array_bytes']/1024**2:.1f} | {case['final_estimate_error_percent']:.2f}% | {case['save']['elapsed_seconds']:.3f} | "
            f"{case['reopen']['elapsed_seconds']:.3f} | {case['lod']['elapsed_seconds']:.3f} |"
        )
    arguments.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
