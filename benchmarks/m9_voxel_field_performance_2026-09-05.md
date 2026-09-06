# M9 large voxel parameter-field performance report

All benchmark inputs are fixed-seed synthetic data. Baseline is `origin/main@709ca10`; measurements were made on the
same Windows machine and Python 3.14 runtime. Results are machine-specific and are not a speed guarantee.

## Baseline bottlenecks and evidence

- `ParameterFieldBuilder` generated an index tuple and XYZ tuple per voxel, invoked mask/domain callbacks point by
  point, constructed `InterpolationResult` objects per joint set and voxel, and repeatedly filtered IDW samples and
  rebuilt coordinate/value arrays.
- The generic scalar service was already chunked and used local `cKDTree`/bounded kriging neighbours, but its rock and
  excavation masks and IDW path were still Python point loops.
- Final dense scientific arrays dominate large-grid memory. Parallelism cannot reduce this allocation.

## Adopted design

- Generate coordinates and grid indices once per bounded chunk.
- Add batch protocols for active-rock masks and nearest-domain assignment, retaining legacy point callback fallback.
- Cache domain-local IDW samples and `cKDTree` objects; use deterministic bounded batch prediction.
- Use the existing Ordinary Kriging batch solver from the First Parameter Field path.
- Group each chunk by structural domain and joint set, then write preallocated flat array views in batches.
- Check cancellation at every chunk, domain, and joint-set boundary.
- Return uncommitted worker candidates to the GUI; cancelled/stale candidates cannot mutate project state.
- Derive resource estimates from named shape/dtype contracts. GUI shows voxel count, persistent arrays, primary chunk
  temporaries, 1/2/4/8-worker array payloads, user budget, and 50% of currently available physical RAM as a system-safe
  limit. Exceeding either limit requires an explicit Yes confirmation and never changes grid spacing.

## First Voxel Parameter Field benchmark

Shape is `24 x 24 x 12` (6,912 voxels), seven joint sets, 100 fields for IDW and 108 for Ordinary Kriging. Persistent
payload is 2,419,200 bytes (2.307 MiB) for IDW and 2,640,384 bytes (2.518 MiB) for Kriging.

| Method | Set IDs | Baseline | Optimized | Baseline voxel/s | Optimized voxel/s | Speed ratio | Full array hash |
|---|---|---:|---:|---:|---:|---:|---|
| IDW | 1,2,3,4,5,6,7 | 16.240 s | 3.459 s | 425 | 1,998 | 4.69x | identical |
| Kriging | 1,2,3,4,5,6,7 | 27.574 s | 2.638 s | 250 | 2,619 | 10.45x | identical |
| IDW | 1,2,4,7,9,12,15 | 14.514 s | 3.457 s | 476 | 1,999 | 4.20x | identical |
| Kriging | 1,2,4,7,9,12,15 | 27.680 s | 2.635 s | 249 | 2,622 | 10.50x | identical |

Chunk sizes 512, 2,048, and 8,192 produced the same complete-array SHA-256 for all four cases. Optimized tracemalloc
peaks were 3.38-3.96 MiB versus 2.37-2.62 MiB at baseline: a bounded 1.0-1.34 MiB increase in exchange for removing
Python object churn. The allocation-contract estimate for seven-set Kriging at chunk 2,048 is 708 KiB of named primary
temporaries. Tracemalloc includes final Python/NumPy objects but does not reliably capture every NumPy/SciPy native
allocation; peak process RSS was therefore not claimed.

## Generic Scalar Field benchmark

Shape is `50 x 69 x 47` (162,150 voxels), 12 synthetic calibration samples, chunk 4,096. The five persistent arrays
occupy 2,432,250 bytes (2.320 MiB).

Each implementation was warmed once, followed by three fixed-seed measurements in alternating execution order. The
summary uses the median wall time; every individual run is retained in the comparison JSON.

| Method | Baseline median | Optimized median | Baseline voxel/s | Optimized voxel/s | Speed ratio | Full array hash |
|---|---:|---:|---:|---:|---:|---|
| IDW | 20.923 s | 0.785 s | 7,750 | 206,482 | 26.64x | identical |
| Kriging | 23.386 s | 17.499 s | 6,933 | 9,266 | 1.34x | identical |

| Method | Implementation | Run 1 | Run 2 | Run 3 | Median |
|---|---|---:|---:|---:|---:|
| IDW | Baseline | 20.973 s | 20.923 s | 20.913 s | 20.923 s |
| IDW | Optimized | 0.785 s | 0.803 s | 0.779 s | 0.785 s |
| Kriging | Baseline | 23.370 s | 23.386 s | 23.446 s | 23.386 s |
| Kriging | Optimized | 17.499 s | 17.488 s | 17.592 s | 17.499 s |

The complete-array SHA-256 is
`c1c56969d332348aa52d9d86d1d1ddb75447fa162a51adb09b2a20ce23c0d9fa` for IDW and
`c1322379516428b17b9cdb12f3d481e6631bc67ee5949ac08d49a0d780e5ea64` for Ordinary Kriging. Baseline and optimized
outputs are byte-for-byte identical for each method.

The independent legacy Kriging benchmark also retained its exact output hash. Its repeated timings (16.986 s baseline,
17.013 s current) show expected run-to-run noise and are retained rather than rewritten. The service-level comparison
above measures the mask/domain batching that the standalone interpolator benchmark does not exercise.

## Memory categories

1. **Persistent scientific arrays:** exact sum of every named array's `shape * dtype.itemsize`; dtype, NaN/NO_DATA
   semantics, set IDs, and M10-required fields are unchanged.
2. **Temporary computation arrays:** bounded by `chunk_size`; the estimator names coordinate, index, mask, prediction,
   diagnostic, validity, and aggregation buffers. No full voxel coordinate matrix is created by estimation or building.
3. **Python objects/loops:** reduced on the normal batch path. Point-only third-party callbacks retain a compatibility
   loop and its Python overhead is not represented as NumPy `nbytes`.
4. **Multiprocessing:** not enabled. Actual extra process-copy/IPC memory is therefore zero. For seven-set Kriging at
   chunk 8,192, chunk-array payload lower bounds for 1/2/4/8 workers are 2.766/5.531/11.063/22.125 MiB; Python runtime,
   SciPy/BLAS working sets, serialization, and task queues would add more.
5. **VTK display buffers:** not included in scientific-array or tracemalloc estimates. Rendering remains single-threaded,
   reads only the selected field/slice, and session layer cleanup behavior is unchanged.

At `1 x 1 x 1 m`, a `224 x 224 x 224` grid with seven Kriging joint sets requires 4,293,459,968 persistent scientific
array bytes (3.999 GiB) before VTK or process overhead. This is output memory, not a temporary or parallelism cost. The
GUI now displays this allocation before work starts and requires confirmation when it exceeds the configured budget or
half of currently available physical RAM. No automatic coarsening or `.dfnproj` format change was introduced.

## Multiprocessing decision and deferred storage options

Multiprocessing was intentionally not implemented. The vectorized single-process path removed the dominant overhead,
while Windows spawn would add interpreter/SciPy working sets and IPC complexity without reducing the final 4 GiB-class
output. Memmap, transactional chunk storage, and lazy field loading remain future options; changing persistence now
would risk compatibility and requires a separate save/reopen/migration design.

## Modified files

- `src/dfn_cave_studio/voxel/parameter_field.py`
- `src/dfn_cave_studio/voxel/resource_estimate.py`
- `src/dfn_cave_studio/models/rock_mask.py`
- `src/dfn_cave_studio/services/m9_service.py`
- `src/dfn_cave_studio/services/scalar_field_service.py`
- `src/dfn_cave_studio/ui/dialogs/m9_dialogs.py`
- `src/dfn_cave_studio/ui/dialogs/m9_scalar_field_dialog.py`
- `scripts/benchmark_m9_parameter_field.py`
- `scripts/benchmark_m9_scalar_fields.py`
- `tests/unit/test_m9_parameter_field.py`
- `tests/unit/test_m9_resource_estimate.py`
- `tests/gui/test_m9_workflow.py`
- `tests/gui/test_m9_scalar_field_dialog.py`
- New benchmark JSON/Markdown artifacts under `benchmarks/`; existing benchmark history was not deleted or rewritten.

## Scientific and application regression

- M9 complete: 138 passed, 4 deselected.
- M10/M11.1 related: 188 passed, 18 deselected.
- Complete non-GUI suite: 788 passed, 21 deselected.
- Complete GUI suite: 178 passed.
- The original First Parameter Field IDW contract remains byte-for-byte stable across all 22 arrays, including dtype,
  shape, values, NaN locations and full-array SHA-256.
- M10/M11.1 scientific regression: 188 passed, 18 deselected; committed M10/M11.1 scientific results remain unchanged.
- IDW persisted-array SHA-256 regression, Kriging single/batch consistency, seven/non-contiguous sets, chunk consistency,
  Holdout isolation, Reject/Clip set-voxel and unique-voxel audit, cancellation/stale result rejection, save/reopen,
  M10 consumption, M11.1 scientific arrays, and GUI budget confirmation all passed.
- `compileall`, changed-file Ruff F/E9, and `git diff --check` passed.

## Known limitations

- Legacy point-only mask/domain callbacks cannot be vectorized without changing their contract; they use the preserved
  compatibility path.
- tracemalloc is not peak RSS and excludes some native-library and all VTK/OpenGL allocations.
- Worker-count figures are static array-payload estimates only because multiprocessing was not activated.
- The dense persisted format remains the limiting factor near 4 GiB; no unsafe persistence rewrite was attempted.

No `git add`, commit, push, merge, rebase, stash, tag, or release operation was executed.
