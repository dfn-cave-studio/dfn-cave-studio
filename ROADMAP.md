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

**Version:** `v0.9.0-M9`

- Estimate joint-set orientation, borehole P10, density, Kappa, set
  probabilities, and size-distribution parameters by structural domain.
- Assign traceable parameters and uncertainty to voxels.
- Validation boreholes remain completely outside fitting.
- RQD is auxiliary information and is never directly converted to P32.

**Exit condition:** every input voxel has traceable DFN generation parameters
and uncertainty.

## M10 — Conditional explicit DFN and second voxelization

**Version:** `v0.10.0-M10`

- Combine deterministic major structures with seeded stochastic fractures.
- Generate through the analysis domain plus buffer.
- Compute real fracture–fracture and fracture–voxel intersections,
  connectivity graphs, and spanning paths.
- Re-voxelize explicit DFN results and compare input and result fields.

**Exit condition:** the database-to-explicit-DFN-to-result-grid path runs end
to end.

## M11 — Validation, external simulation, and ML export

**Version:** `v0.11.0-M11`

- Validate P10, orientation, and set proportions on held-out holes.
- Quantify multiple-realization uncertainty, error, and confidence intervals.
- Assign fracture mechanics and export 3DEC, PFC, VTK, CSV, and ML-ready
  input/output voxel datasets with provenance, seeds, and versions.

**Exit condition:** the model is independently validated and transferable.

## M12 — Fragmentation, integration, and formal release

**Version:** `v1.0.0-M12`

- Explicit block cutting, block shape/volume statistics, D20/D50/D80, and
  uncertainty across realizations.
- Geometric caveability-related indicators without equating DFN alone with
  real caveability.
- Spatial indexing, large-model performance, background progress/cancel,
  documentation, Windows installer, regression, and scientific validation.

**Exit condition:** installable, usable, validated, and exportable v1.0.0.

M9 work must not begin until M8 external review is accepted.
