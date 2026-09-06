"""Fixed-seed synthetic benchmark for the M9 first voxel parameter field."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import tracemalloc

import numpy as np

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, KrigingSettings, P10Interval, P32Estimate, SizeModel
from dfn_cave_studio.voxel.parameter_field import ParameterFieldBuilder


SEED = 20260905
SHAPE = (24, 24, 12)


def _inputs(set_ids: list[int]) -> tuple[list[P10Interval], list[P32Estimate], list[JointSetConfig], list[SizeModel]]:
    rng = np.random.default_rng(SEED)
    intervals: list[P10Interval] = []
    estimates: list[P32Estimate] = []
    sizes: list[SizeModel] = []
    for domain_id in (1, 2):
        for set_id in set_ids:
            estimates.append(P32Estimate(
                domain_id=domain_id, set_id=set_id, fracture_count=8, raw_sample_length=8.0,
                effective_sample_length=8.0, mean_exposure=1.0, p32=0.05 * set_id + 0.1 * domain_id,
                observability="adequate", random_seed=SEED,
            ))
            sizes.append(SizeModel(domain_id=domain_id, set_id=set_id))
            for sample_index in range(4):
                x = rng.uniform(0.5, 11.5) + (domain_id - 1) * 12.0
                y = rng.uniform(0.5, 23.5)
                z = rng.uniform(0.5, 11.5)
                intervals.append(P10Interval(
                    hole_id=f"SYN-{domain_id}-{set_id}-{sample_index}", from_depth=0.0, to_depth=1.0,
                    domain_id=domain_id, set_id=set_id, observation_count=sample_index + 1,
                    sample_length=1.0, p10=0.05 * set_id + 0.02 * sample_index + 0.1 * domain_id,
                    role="calibration", center_x=x, center_y=y, center_z=z,
                    segment_directions=[(0.0, 0.0, -1.0, 1.0)],
                ))
    return intervals, estimates, [JointSetConfig(set_id=item) for item in set_ids], sizes


def _array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = arrays[name]
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def run_case(method: DensityMethod, set_ids: list[int], chunk_size: int) -> dict[str, object]:
    intervals, estimates, joint_sets, sizes = _inputs(set_ids)
    bounds = ModelBounds(x_min=0, x_max=24, y_min=0, y_max=24, z_min=0, z_max=12)
    voxel = VoxelConfig()
    settings = DensitySettings(
        method=method, min_neighbors=1, max_neighbors=4, monte_carlo_samples=100,
        kriging=KrigingSettings(mode="manual", nugget=0.01, sill=2.0, range=20.0,
                                minimum_neighbors=2, maximum_neighbors=4),
    )
    tracemalloc.start()
    started = time.perf_counter()
    metadata, arrays = ParameterFieldBuilder().build(
        bounds, voxel, settings, intervals, estimates, joint_sets, sizes, random_seed=SEED,
        domain_at_point=lambda point: 1 if point[0] < 12 else 2,
        inside_model=lambda point: not (point[1] < 2 and point[2] < 2), chunk_size=chunk_size,
    )
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    persistent = sum(array.nbytes for array in arrays.values())
    return {
        "method": method.value, "set_ids": set_ids, "set_count": len(set_ids),
        "shape": list(metadata.shape), "voxel_count": math.prod(metadata.shape),
        "field_count": len(arrays), "array_contract": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype), "nbytes": array.nbytes}
            for name, array in sorted(arrays.items())
        },
        "persistent_array_bytes": persistent, "elapsed_seconds": elapsed,
        "voxels_per_second": int(math.prod(metadata.shape) / elapsed),
        "tracemalloc_peak_bytes": peak, "array_sha256": _array_hash(arrays), "chunk_size": chunk_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()
    cases = []
    for set_ids in ([1, 2, 3, 4, 5, 6, 7], [1, 2, 4, 7, 9, 12, 15]):
        for method in (DensityMethod.IDW, DensityMethod.ORDINARY_KRIGING):
            cases.append(run_case(method, list(set_ids), args.chunk_size))
    payload = {"label": args.label, "random_seed": SEED, "synthetic_data_only": True, "cases": cases}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
