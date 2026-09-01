# DFN Cave Studio Roadmap

This roadmap is fixed from M8 onward. Published M0–M7 milestone numbers and
tags remain unchanged.

## End-to-end flow

`Borehole database → cleaning → validation holdout → domains/joint sets → input parameter voxel field → explicit DFN → result voxel field → validation → export/fragmentation`

## Published baseline

- Published version: `v0.7.0-M7`
- Baseline commit: `ad4cd158c272a9b44deb93d40515729dc17361e5`
- Delivered: import, traceable cleaning, validation-hole holdout, borehole
  domain intervals, joint-set identification, and `.dfnproj` restoration.

## M8 — Borehole database and spatial-grid foundation

**Version:** `v0.8.0-M8`

- Maintain collars, surveys, fractures, RQD, and domain intervals through one
  project repository.
- Support independent, arbitrary-order, incremental imports with preview,
  mapping, duplicate handling, append/replace/cancel, and Pending relinking.
- Preserve Raw, Formal, Excluded, Pending, provenance, and edit history.
- Provide searchable database UI with per-hole and per-table views.
- Define separate Voxel Analysis and DFN Generation domains.
- Validate full trajectories and spatial observations against automatic or
  manual bounds.
- Compute anisotropic voxel dimensions with ceil coverage and allocation-free
  memory estimates.
- Preview bounds, trajectories, observations, sampled grid lines, and outliers
  without rendering every voxel.

**Exit condition:** the user can establish and inspect the maintained borehole
database and confirm a reasonable voxel range.

## M9 — Local DFN parameter field and first voxelization

**Version:** `v0.9.1-M9`

- Estimate joint-set orientation, borehole P10, density, Kappa, set
  probabilities, and size-distribution parameters by structural domain.
- Assign traceable parameters and uncertainty to voxels.
- Validation boreholes remain completely outside fitting.
- RQD is auxiliary information and is never directly converted to P32.

**Exit condition:** every input voxel has traceable DFN generation parameters
and uncertainty.

## M10 — Conditional explicit DFN

**Version:** `v0.10.0-M10`

- Combine parameterized deterministic structures, Calibration observation constraints, and seeded stochastic fractures.
- Generate only from valid M9 modelled cells over the buffered generation domain.
- Manage reproducible realizations, merged display layers, compressed persistence, basic exports, and preliminary quality reports.
- Do not claim centre-assigned voxel statistics are exact local P32.

**Exit condition:** reproducible explicit geometry can be generated, inspected, saved, reopened, and exported.

## M11.1 — Exact Second Voxelization

**Version:** `v0.11.0-M11.1`
**Status:** PUBLISHED

- Compute exact circular-fracture/voxel intersection areas and intersection-derived P32 using sparse deterministic indexing.
- Persist per-set/all-set sparse results, `P32_explicit_intersection + P32_subgrid`, diagnostics, provenance, and invalidation state.
- Provide P32 clouds, orthogonal/arbitrary sections, plane and box cutaways, and session-only layer/scalar-bar management.
- Support transactional background execution, fast cancellation, save/reopen, bilingual UI, and audited cross-platform rendering CI.

**Exit condition:** exact second voxelization can be calculated, audited, visualized, saved, reopened, cancelled, and invalidated without changing M10 geometry.

## Later M11 — Connectivity, validation, and external export

**Status:** PLANNED

- Compute fracture-fracture intersections, connected clusters, boundary-spanning paths, and flow/percolation channels.
- Validate P10, orientation, and set proportions on held-out holes.
- Quantify multiple-realization uncertainty, error, and confidence intervals.
- Assign fracture mechanics and export 3DEC, PFC, VTK, CSV, and ML-ready
  input/output voxel datasets with provenance, seeds, and versions.
- Evaluate interpolation extensions such as Kriging only as a separately validated future method.

**Exit condition:** the model is independently validated and transferable.

M11.1 is intentionally limited to second voxelization. It does not claim
connectivity, percolation, block cutting, mechanics, formal simulator export, or Kriging.

## M12 — Fragmentation, integration, and formal release

**Version:** `v1.0.0-M12`

- Explicit block cutting, block shape/volume statistics, D20/D50/D80, and
  uncertainty across realizations.
- Geometric caveability-related indicators without equating DFN alone with
  real caveability.
- Spatial indexing, large-model performance, background progress/cancel,
  documentation, Windows installer, regression, and scientific validation.

**Exit condition:** installable, usable, validated, and exportable v1.0.0.

Published baselines extend through `v0.11.0-M11.1`; later M11 work and M12 remain planned as described above.
