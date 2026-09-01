# DFN Cave Studio Project Plan

## Governance

Published M0–M7 versions retain their existing milestone numbers and tags.
Development after the `v0.7.0-M7` baseline follows the fixed M8–M12 route in
this document and `ROADMAP.md`. Scientific claims, tests, provenance, units,
coordinate conventions, and random-seed policy remain governed by
`AGENTS.md`, `ARCHITECTURE.md`, and `SCIENTIFIC_SPEC.md`.

## Product flow

`Borehole database → data cleaning → validation holdout → structural domains / joint sets → input parameter voxel field → explicit DFN → result voxel field → validation → export / fragmentation`

## M8 — v0.8.0-M8

Scope is limited to a sustainable borehole database and spatial-grid
foundation.

### Borehole database work packages

1. Establish a canonical project repository for collars, surveys, fractures,
   RQD, and domain intervals, extensible to future logged tables.
2. Persist immutable Raw values and Formal, Excluded, and Pending
   dispositions with record ID, file, row, timestamp, reason, and modification
   history.
3. Support independent and arbitrary-order preview/mapped imports, batch
   append, audited replacement, duplicate reporting, transactional cancel, and
   automatic Pending relinking.
4. Project the repository’s Formal records into the M7
   `BoreholeCollection` compatibility model used by existing scientific
   services.
5. Migrate v0.7.0 projects without losing boreholes, surveys, fractures, RQD,
   domain intervals, holdout, workflow, or joint sets.
6. Provide a searchable and sortable database dock with hole/table/state
   views, real row values, counts, issues, audited edit/delete, and import.
7. Invalidate only results that depend on changed upstream tables and mark the
   project dirty only after accepted changes.

### Spatial-grid work packages

1. Persist distinct Voxel Analysis and DFN Generation domains; generation must
   contain analysis.
2. Derive automatic bounds from complete collar-to-toe trajectories and valid
   spatial observations plus outward margin.
3. Permit manual bounds before or after import, then report outside holes,
   trajectory points, observation points, directional exceedance, affected
   holes, and recommended expansion.
4. Require explicit user acknowledgement before retaining a boundary that
   clips data.
5. Support independent dx/dy/dz and ceil-covered nx/ny/nz, voxel counts, active
   counts when a mask exists, memory estimates, and large-grid warnings.
6. Preserve separate states for outside-model, no-data, and true-zero cells.
7. Preview analysis/generation boxes, trajectories, fracture points, axes,
   sampled grid lines, and outliers without allocating or drawing a dense
   voxel grid.

### M8 verification

Tests cover collars-only projects; Pending relinking; arbitrary-order import;
duplicate, append, replace, and cancel behavior; real UI table contents;
state/provenance round-trip; M7 migration; complete-trajectory bounds;
collar-inside/trajectory-outside detection; recommended expansion;
anisotropic ceil coverage; domain containment; exact memory estimates;
allocation-free preview; dirty lifecycle; and repeated save/autosave/reopen.

M8 produces no P10/P32 interpolation, parameter field, conditional DFN,
fragmentation, external-simulation export, or formal release.

## M9 — v0.9.1-M9

Build local DFN parameter fields and the first voxelization: structural-domain
orientation statistics, borehole P10, density, Kappa and set-probability
fields, fracture-size parameters, traceability, and uncertainty. Validation
holes are excluded from fitting. RQD may be auxiliary but is never directly
converted to P32.

## M10 — v0.10.0-M10

Generate seeded conditional explicit DFNs from deterministic major structures
and M9 small-fracture parameters over the buffered domain. M10 includes
Calibration observation conditioning, multi-realization management,
visualization, compressed persistence, generic export, and preliminary
quality reporting. M10 defers exact intersections and second voxelization to
M11.1; connectivity remains deferred to later M11 work.

## M11.1 — v0.11.0-M11.1 (READY FOR RELEASE)

This bounded release candidate implements the first M11 work package: analytic circular disk/AABB
intersection area, sparse fracture-voxel pairs, intersection-derived explicit
P32, combination with M10 subgrid P32, conservation diagnostics, background
execution, fast cancellation, persistence, invalidation, and cloud/section/cutaway
inspection. It also includes bilingual UI, M10 save protection, M9/M11 opaque
rendering fixes, and Windows/Linux rendering CI.

## Later M11 work packages — PLANNED

Compute fracture-fracture intersections, connected clusters, boundary-spanning
paths, flow/percolation channels, independent held-out validation, uncertainty,
mechanical properties, and formal 3DEC/PFC export. Kriging is not implemented.
Connectivity and every later M11 work package remain explicitly outside M11.1.

## M12 — v1.0.0-M12

Implement explicit block cutting, block statistics and D20/D50/D80 curves,
multi-realization fragmentation uncertainty, carefully bounded geometric
caveability indicators, large-model performance, background progress/cancel,
documentation, Windows packaging, and complete regression/scientific
validation.

## M9 implementation work packages

1. Generate half-open fixed-length or domain-bound P10 intervals from Formal records, retaining independent Validation intervals.
2. Estimate domain/set P32 by `N / sum(Lj * E(|n·uj|))`, using seeded Fisher integration and an explicit low-observability state.
3. Provide replaceable GLOBAL_CONSTANT and three-dimensional IDW density models with anisotropy, domain isolation, diagnostics, fallback provenance, and distinct zero/no-data states.
4. Fit only real radius/diameter/trace/mapped-size fields by MLE with likelihood/AIC/BIC, or store explicitly assumed/user-defined priors.
5. Build the first parameter voxel field in chunks with progress/cancel and persist arrays once in compressed NPZ form.
6. Evaluate held-out boreholes only after fitting, reporting interval errors and aggregate metrics or `INSUFFICIENT_VALIDATION`.
7. Export P10/P32 tables, density/size JSON, validation outputs, NPZ and VTI; do not generate explicit DFN geometry.

M9 and M10 are published historical foundations for the M11.1 release candidate. Later M11 work remains separately planned.
