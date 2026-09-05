"""Reproducible synthetic benchmark for the 162,150-cell M9 kriging case."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import tracemalloc

import numpy as np
import scipy

from dfn_cave_studio.models.m9 import KrigingSettings, VariogramMode
from dfn_cave_studio.voxel.ordinary_kriging import OrdinaryKrigingInterpolator, fit_variogram


SHAPE = (50, 69, 47)
SEED = 20260903
CHUNK_SIZE = 4096


def run_benchmark() -> dict[str, object]:
    """Run the fixed synthetic case and return auditable measurements."""
    # Start before every benchmark-owned sample, target, and output array is
    # created.  This is still a Python allocator measurement; native library
    # allocations may not be fully visible to tracemalloc.
    tracemalloc.start()
    rng = np.random.default_rng(SEED)
    coordinates = rng.uniform((0.0, 0.0, 0.0), (500.0, 690.0, 470.0), size=(12, 3))
    values = 40.0 + 0.02 * coordinates[:, 0] - 0.01 * coordinates[:, 1] + rng.normal(0.0, 0.5, 12)
    settings = KrigingSettings(
        mode=VariogramMode.MANUAL,
        nugget=0.1,
        sill=25.0,
        range=250.0,
        minimum_neighbors=4,
        maximum_neighbors=8,
    )
    model = OrdinaryKrigingInterpolator(
        coordinates, values, fit_variogram(coordinates, values, settings), settings
    )
    indices = np.column_stack(np.unravel_index(np.arange(np.prod(SHAPE)), SHAPE))
    targets = (indices + 0.5) * np.asarray((10.0, 10.0, 10.0))
    voxel_count = int(np.prod(SHAPE))
    estimates = np.full(voxel_count, np.nan, dtype=np.float32)
    variances = np.full(voxel_count, np.nan, dtype=np.float32)
    neighbours = np.zeros(voxel_count, dtype=np.int16)
    cell_state = np.zeros(voxel_count, dtype=np.uint8)
    domain_id = np.ones(voxel_count, dtype=np.int32)
    started = time.perf_counter()
    for start in range(0, voxel_count, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, voxel_count)
        chunk_estimates, chunk_variances, chunk_neighbours, status = model.predict_many(targets[start:stop])
        estimates[start:stop] = chunk_estimates
        variances[start:stop] = chunk_variances
        neighbours[start:stop] = chunk_neighbours
        cell_state[start:stop] = np.where(status == "INSUFFICIENT_DATA", 1, 3)
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    final_bytes = estimates.nbytes + variances.nbytes + neighbours.nbytes + cell_state.nbytes + domain_id.nbytes
    output_hash = hashlib.sha256()
    for array in (estimates, variances, neighbours, cell_state, domain_id):
        output_hash.update(np.ascontiguousarray(array).tobytes())
    return {
        "algorithm_version": "m9-scalar-kriging-1",
        "generated_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "random_seed": SEED,
        "synthetic_data_only": True,
        "voxel_shape": list(SHAPE),
        "voxel_count": int(np.prod(SHAPE)),
        "calibration_sample_count": len(coordinates),
        "variogram_mode": settings.mode.value,
        "variogram_model": settings.model.value,
        "nugget": settings.nugget,
        "sill": settings.sill,
        "range": settings.range,
        "minimum_neighbors": settings.minimum_neighbors,
        "maximum_neighbors": settings.maximum_neighbors,
        "chunk_size": CHUNK_SIZE,
        "elapsed_seconds": elapsed,
        "voxels_per_second": int(np.prod(SHAPE) / elapsed),
        "persistent_output_array_mib": final_bytes / 1024**2,
        "benchmark_tracemalloc_peak_mib": peak / 1024**2,
        "modeled_voxel_count": int(np.count_nonzero(cell_state == 3)),
        "output_array_sha256": output_hash.hexdigest(),
        "memory_note": (
            "benchmark_tracemalloc_peak_mib starts before benchmark-owned input, target, and output arrays are "
            "created, but tracemalloc does not fully capture NumPy/SciPy native allocator peaks or VTK buffers"
        ),
        "performance_scope": "This machine-specific measurement is not a speed guarantee for other systems.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=Path("benchmarks/m9_ordinary_kriging_162150.json"))
    parser.add_argument("--markdown", type=Path, default=Path("benchmarks/m9_ordinary_kriging_162150.md"))
    args = parser.parse_args()
    result = run_benchmark()
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(
        "# M9 Ordinary Kriging - 162,150 voxel benchmark\n\n"
        "This fixed-seed benchmark uses synthetic data only.\n\n"
        f"- Elapsed: {result['elapsed_seconds']:.3f} s\n"
        f"- Throughput: {result['voxels_per_second']:,} voxels/s\n"
        f"- Persistent output arrays: {result['persistent_output_array_mib']:.3f} MiB\n"
        f"- Benchmark tracemalloc peak: {result['benchmark_tracemalloc_peak_mib']:.3f} MiB\n"
        f"- Modelled cells: {result['modeled_voxel_count']:,}\n"
        f"- CPU: {result['cpu']} ({result['logical_cpu_count']} logical cores)\n\n"
        f"{result['memory_note']}. {result['performance_scope']}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
