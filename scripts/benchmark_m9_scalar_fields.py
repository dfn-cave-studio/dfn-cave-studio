"""Compare baseline/current M9 scalar fields on 162,150 synthetic voxels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc
import types
from statistics import median

import numpy as np

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, KrigingSettings, ScalarParameterSample
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.services.scalar_field_service import ScalarParameterFieldService


BASELINE = "709ca10"
SEED = 20260903
SHAPE = (50, 69, 47)


def _baseline_service_class():
    """Load the two baseline modules without changing the worktree."""
    parameter_name = "dfn_cave_studio.voxel.parameter_field"
    scalar_name = "dfn_cave_studio.services.scalar_field_service"
    current_parameter = sys.modules[parameter_name]
    current_scalar = sys.modules[scalar_name]
    try:
        parameter_source = subprocess.check_output(
            ["git", "show", f"{BASELINE}:src/dfn_cave_studio/voxel/parameter_field.py"],
            text=True, encoding="utf-8",
        )
        baseline_parameter = types.ModuleType(parameter_name)
        baseline_parameter.__package__ = "dfn_cave_studio.voxel"
        sys.modules[parameter_name] = baseline_parameter
        exec(compile(parameter_source, "baseline_parameter_field.py", "exec"), baseline_parameter.__dict__)
        scalar_source = subprocess.check_output(
            ["git", "show", f"{BASELINE}:src/dfn_cave_studio/services/scalar_field_service.py"],
            text=True, encoding="utf-8",
        )
        baseline_scalar = types.ModuleType(scalar_name)
        baseline_scalar.__package__ = "dfn_cave_studio.services"
        sys.modules[scalar_name] = baseline_scalar
        exec(compile(scalar_source, "baseline_scalar_field_service.py", "exec"), baseline_scalar.__dict__)
        return baseline_scalar.ScalarParameterFieldService
    finally:
        sys.modules[parameter_name] = current_parameter
        sys.modules[scalar_name] = current_scalar


def _project() -> Project:
    rng = np.random.default_rng(SEED)
    project = Project()
    bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=690, z_min=0, z_max=470)
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=bounds, generation_domain=bounds)
    project.voxel_config = VoxelConfig(cell_size_x=10, cell_size_y=10, cell_size_z=10)
    coordinates = rng.uniform((0, 0, 0), (500, 690, 470), size=(12, 3))
    values = 40 + 0.02 * coordinates[:, 0] - 0.01 * coordinates[:, 1] + rng.normal(0, 0.5, 12)
    project.m9_state.scalar_samples = [
        ScalarParameterSample(
            sample_id=f"SYN-{index}", borehole_id=f"SYN-{index}", from_depth=0, to_depth=1,
            parameter_name="UCS", value=float(value), unit="MPa", midpoint_x=float(point[0]),
            midpoint_y=float(point[1]), midpoint_z=float(point[2]), domain_id=1, role="calibration",
        )
        for index, (point, value) in enumerate(zip(coordinates, values, strict=True))
    ]
    holdout = HoldoutService()
    hole_ids = [sample.borehole_id for sample in project.m9_state.scalar_samples]
    holdout.select_manual(hole_ids, [])
    holdout.lock()
    set_holdout(project, holdout)
    return project


def _hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    return digest.hexdigest()


def _run(service_class, implementation: str, method: DensityMethod) -> dict[str, object]:
    project = _project()
    settings = DensitySettings(
        method=method, min_neighbors=1, max_neighbors=8,
        kriging=KrigingSettings(mode="manual", nugget=0.1, sill=25, range=250,
                                minimum_neighbors=4, maximum_neighbors=8),
    )
    tracemalloc.start()
    started = time.perf_counter()
    result = service_class(project).build("UCS", settings, chunk_size=4096)
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "implementation": implementation, "method": method.value, "shape": list(SHAPE),
        "voxel_count": int(np.prod(SHAPE)), "elapsed_seconds": elapsed,
        "voxels_per_second": int(np.prod(SHAPE) / elapsed),
        "persistent_array_bytes": sum(array.nbytes for array in result.arrays.values()),
        "tracemalloc_peak_bytes": peak, "array_sha256": _hash(result.arrays),
    }


def _aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    """Summarize repeated measurements without discarding their raw values."""
    hashes = {str(run["array_sha256"]) for run in runs}
    if len(hashes) != 1:
        raise RuntimeError(f"Repeated benchmark outputs differ: {sorted(hashes)}")
    elapsed = float(median(float(run["elapsed_seconds"]) for run in runs))
    voxel_count = int(runs[0]["voxel_count"])
    return {
        "implementation": runs[0]["implementation"],
        "method": runs[0]["method"],
        "shape": runs[0]["shape"],
        "voxel_count": voxel_count,
        "elapsed_seconds": elapsed,
        "voxels_per_second": int(voxel_count / elapsed),
        "persistent_array_bytes": runs[0]["persistent_array_bytes"],
        "tracemalloc_peak_bytes": int(median(int(run["tracemalloc_peak_bytes"]) for run in runs)),
        "array_sha256": runs[0]["array_sha256"],
        "runs": [
            {
                "repetition": index + 1,
                "elapsed_seconds": run["elapsed_seconds"],
                "voxels_per_second": run["voxels_per_second"],
                "tracemalloc_peak_bytes": run["tracemalloc_peak_bytes"],
                "array_sha256": run["array_sha256"],
            }
            for index, run in enumerate(runs)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if args.warmup_runs < 1:
        parser.error("--warmup-runs must be at least 1")
    baseline_class = _baseline_service_class()
    cases = []
    for method in (DensityMethod.IDW, DensityMethod.ORDINARY_KRIGING):
        implementations = (
            (baseline_class, f"baseline-{BASELINE}"),
            (ScalarParameterFieldService, "optimized"),
        )
        for _warmup in range(args.warmup_runs):
            for service_class, implementation in implementations:
                _run(service_class, implementation, method)
                print(f"warmup {method.value} {implementation}", file=sys.stderr, flush=True)
        measurements: dict[str, list[dict[str, object]]] = {
            implementation: [] for _service_class, implementation in implementations
        }
        execution_order: list[str] = []
        for repetition in range(args.repetitions):
            ordered = implementations if repetition % 2 == 0 else tuple(reversed(implementations))
            for service_class, implementation in ordered:
                result = _run(service_class, implementation, method)
                measurements[implementation].append(result)
                execution_order.append(f"{repetition + 1}:{implementation}")
                print(
                    f"measurement {method.value} {repetition + 1}/{args.repetitions} "
                    f"{implementation}: {result['elapsed_seconds']:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
        for _service_class, implementation in implementations:
            case = _aggregate(measurements[implementation])
            case["execution_order"] = execution_order
            cases.append(case)
    payload = {
        "random_seed": SEED,
        "synthetic_data_only": True,
        "warmup_runs": args.warmup_runs,
        "repetitions": args.repetitions,
        "summary_statistic": "median",
        "cases": cases,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
