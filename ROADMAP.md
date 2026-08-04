# DFN Cave Studio — Development Roadmap

## Overview

DFN Cave Studio is a Discrete Fracture Network (DFN) modeling and visualization application for geotechnical and hydrogeological analysis. This roadmap defines 13 milestones (M0 through M12) from project inception through a stable v1.0 release.

---

## Version Tag Convention

Each milestone is tagged according to the pattern `v<major>.<minor>.<patch>-M<N>`:

| Milestone | Tag        |
|-----------|------------|
| M0        | v0.1.0-M0  |
| M1        | v0.2.0-M1  |
| M2        | v0.3.0-M2  |
| M3        | v0.4.0-M3  |
| M4        | v0.5.0-M4  |
| M5        | v0.6.0-M5  |
| M6        | v0.7.0-M6  |
| M7        | v0.8.0-M7  |
| M8        | v0.9.0-M8  |
| M9        | v0.10.0-M9 |
| M10       | v0.11.0-M10|
| M11       | v1.0.0-rc1 |
| M12       | v1.0.0     |

---

## Dependency Graph

```
M0 ──────► M1 ──────► M2 ──────► M3 ──────► M4 ──────► M5
                                                         │
             ┌───────────────────────────────────────────┘
             ▼
            M6 ──────► M7 ──────► M8
             │                      │
             └──────────► M9 ◄──────┘
                            │
                            ▼
                           M10 ─────► M11 ─────► M12
```

M0 through M5 are strictly sequential core math and data model milestones.
M6 depends on M5 (data model complete).
M7 depends on M6 (computation engine needs import/export).
M8 depends on M7 (GUI needs computation engine).
M9 depends on M6 and M8 (3D visualization needs data model and GUI framework).
M10 depends on M9 (analysis tools need visualization).
M11 depends on M10 (release candidate needs all features).
M12 depends on M11 (final release after RC stabilization).

---

## Milestones

---

### M0 — Project Skeleton and Math Foundation

**Tag:** `v0.1.0-M0`
**Status:** NOT STARTED
**Estimated Duration:** 1-2 weeks

**Goal:** Establish the project repository, build system, CI pipeline, and core linear algebra / geometry primitives.

**Scope:**
- Repository initialization with `.gitignore`, `README.md`, `LICENSE`
- Python package scaffold (`pyproject.toml`, `setup.cfg`, or equivalent)
- CI pipeline (linting, type-checking, unit tests on push)
- `Vector3` class: construction, addition, subtraction, scalar multiplication, dot product, cross product, magnitude, normalization
- `Plane` class: construction from point + normal, from three points, from dip-direction/dip
- `Line3D` / `Ray3D` class
- `Polygon` class (ordered vertices, area computation)
- `Box` class (axis-aligned bounding box, oriented bounding box)
- Unit test suite covering all math primitives
- Code formatting (black/ruff) and linting (ruff/mypy) configured

**Dependencies:** None (starting point)
**Completed Work:** N/A
**Next:** M1 — orientation distributions and stochastic geometry

---

### M1 — Orientation Distributions and Sampling

**Tag:** `v0.2.0-M1`
**Status:** NOT STARTED
**Estimated Duration:** 1-2 weeks

**Goal:** Implement statistical orientation models used in DFN generation (Fisher distribution), plus deterministic helpers.

**Scope:**
- Dip direction / dip angle representation and conversion to/from unit normal vectors
- Fisher distribution implementation:
  - PDF evaluation
  - Random sampling (acceptance-rejection or Woodcock algorithm)
  - Concentration parameter kappa mapping
- Fixed-seed reproducibility for all random sampling
- Uniform orientation sampling (sphere)
- Orientation set: collection of sampled orientations from a distribution
- Statistical validation tests (chi-squared or KS test against expected distribution)
- Unit tests for all conversions, sampling, and reproducibility

**Dependencies:** M0 (Vector3, Plane)
**Completed Work:** N/A
**Next:** M2 — fracture primitives

---

### M2 — Fracture Primitives

**Tag:** `v0.3.0-M2`
**Status:** NOT STARTED
**Estimated Duration:** 2-3 weeks

**Goal:** Define the fracture entity and implement geometric intersection operations.

**Scope:**
- `Fracture` class:
  - Centroid (x, y, z)
  - Orientation (dip direction, dip) or normal vector
  - Shape: polygon (arbitrary vertex list), ellipse (semi-major, semi-minor), or circle (radius)
  - Aperture (hydraulic)
  - Unique identifier
- Fracture polygon clipping against a bounding box (Sutherland-Hodgman or equivalent)
- Fracture-plane intersection with a bounding box
- Fracture-fracture intersection (line segment result)
- Fracture-voxel intersection detection and area computation
- Fracture area calculation (planar polygon area)
- Unit tests for all intersection operations with known-geometry cases

**Dependencies:** M1 (orientations, Plane, Box)
**Completed Work:** N/A
**Next:** M3 — DFN set generation

---

### M3 — DFN Set Generation

**Tag:** `v0.4.0-M3`
**Status:** NOT STARTED
**Estimated Duration:** 2-3 weeks

**Goal:** Generate collections of fractures according to statistical parameters, forming a complete fracture network.

**Scope:**
- `FractureSet` class: a group of fractures sharing an orientation distribution and size distribution
- Size distribution models:
  - Log-normal
  - Power-law (truncated)
  - Exponential
  - Constant (fixed size)
- Fracture density specification: P32 (fracture area per unit volume)
- Stochastic realization: given a bounding region, generate N fractures from a set definition that achieve (on expectation) the target P32
- Multiple fracture sets within one DFN model
- `DFNModel` class: collection of fracture sets, bounding region
- Unit tests for P32 computation, statistical properties of generated sets

**Dependencies:** M2 (Fracture, clipping)
**Completed Work:** N/A
**Next:** M4 — spatial models and intensity functions

---

### M4 — Spatial Models and Intensity Functions

**Tag:** `v0.5.0-M4`
**Status:** NOT STARTED
**Estimated Duration:** 2-3 weeks

**Goal:** Support spatially varying fracture intensity (non-homogeneous Poisson processes for fracture locations).

**Scope:**
- Intensity function interface (abstract)
- Constant (homogeneous) intensity model — uniform random centroid placement
- Depth-dependent intensity (e.g., decreasing P32 with depth)
- Domain-based intensity: user-defined sub-regions with different P32 values
- Grid-defined intensity: intensity sampled from a 3D scalar grid
- Fracture set generation with non-uniform spatial density (rejection sampling or thinning)
- Domain definition: axis-aligned boxes, convex hulls, import from mesh
- Unit tests verifying spatial intensity distributions

**Dependencies:** M3 (DFNModel, FractureSet)
**Completed Work:** N/A
**Next:** M5 — project persistence and data model

---

### M5 — Project Persistence and Data Model

**Tag:** `v0.6.0-M5`
**Status:** NOT STARTED
**Estimated Duration:** 1-2 weeks

**Goal:** Save and load complete DFN projects to/from disk with a versioned file format.

**Scope:**
- Project file format specification (JSON-based or HDF5)
- `Project` class: metadata (name, author, created date, version), DFN model, display settings
- Serialize: project to file
- Deserialize: file to project, with version detection
- Version migration: detect older file format versions and upgrade data model
- Import/export of fracture data (CSV, VTK)
- Error handling for corrupted or incompatible files
- Unit tests for save/load round-trip fidelity, version upgrades, bad-data rejection

**Dependencies:** M4 (complete DFN model with spatial domains)
**Completed Work:** N/A
**Next:** M6 — computation engine

---

### M6 — Computation Engine

**Tag:** `v0.7.0-M6`
**Status:** NOT STARTED
**Estimated Duration:** 3-4 weeks

**Goal:** Compute derived quantities from a DFN model: fracture connectivity, block fragmentation, equivalent permeability.

**Scope:**
- Fracture connectivity graph construction (intersection detection at network scale)
- Connected component analysis
- Path finding between user-specified points through the fracture network
- Block identification: regions of intact rock bounded by fractures
- Block volume and shape statistics
- P10 computation along virtual boreholes (linear fracture intensity)
- P32 computation (area per volume) for the realized network
- Computations run in background threads; support cancellation
- Progress reporting interface
- Unit tests with synthetic networks of known connectivity

**Dependencies:** M5 (Project persistence for loading models to compute on)
**Completed Work:** N/A
**Next:** M7 — command-line interface

---

### M7 — Command-Line Interface

**Tag:** `v0.8.0-M7`
**Status:** NOT STARTED
**Estimated Duration:** 1-2 weeks

**Goal:** Provide a CLI for headless DFN generation, computation, and export.

**Scope:**
- CLI entry point (`dfn-cave-studio` or similar)
- Subcommands:
  - `generate`: create a DFN from a parameter file, save project
  - `compute`: run connectivity/block analysis on a project, output results
  - `export`: export fractures to CSV/VTK
  - `info`: print project metadata and summary statistics
- Parameter file format (YAML/JSON) for headless generation
- Exit codes for success/failure
- `--help` documentation for all commands
- Integration tests for CLI workflows

**Dependencies:** M6 (computation engine)
**Completed Work:** N/A
**Next:** M8 — GUI foundation

---

### M8 — GUI Foundation

**Tag:** `v0.9.0-M8`
**Status:** NOT STARTED
**Estimated Duration:** 3-4 weeks

**Goal:** Build the desktop GUI with project management, parameter editing, and non-graphical views.

**Scope:**
- GUI framework: PySide6 or PyQt6
- Main window with menu bar (File, Edit, View, Compute, Help)
- Project panel: create new project, open, save, save-as
- Parameter editor: fracture set definitions, domain configuration, intensity models
- Object tree / layer panel: list fracture sets, toggle visibility
- Console / log panel: display computation progress and messages
- Status bar
- Error dialogs for invalid inputs, file errors, computation failures
- Background computation does not block the UI thread
- Unit tests for UI logic (model-view separation)
- Manual GUI test specifications

**Dependencies:** M7 (CLI for headless testing of the same operations)
**Completed Work:** N/A
**Next:** M9 — 3D visualization

---

### M9 — 3D Visualization

**Tag:** `v0.10.0-M9`
**Status:** NOT STARTED
**Estimated Duration:** 3-4 weeks

**Goal:** Interactive 3D rendering of the DFN model, domain boundary, and analysis results.

**Scope:**
- 3D viewport using VTK / PyVista or OpenGL
- Render fracture polygons with orientation-dependent coloring
- Render domain bounding box / boundary surface
- Render borehole traces
- Render intersection lines between fractures
- Camera controls: orbit, pan, zoom, fit-to-view
- Object picking: click to select a fracture, highlight it
- Fracture set toggling: show/hide individual sets
- Screenshot / high-resolution export
- Performance optimization for large fracture counts (LOD, instancing)
- Integration tests for viewport rendering (buffer-based comparisons)

**Dependencies:** M6 (computation results to visualize) and M8 (GUI framework to host the viewport)
**Completed Work:** N/A
**Next:** M10 — analysis and reporting

---

### M10 — Analysis and Reporting

**Tag:** `v0.11.0-M10`
**Status:** NOT STARTED
**Estimated Duration:** 2-3 weeks

**Goal:** In-app analysis dashboards, statistical summaries, and exportable reports.

**Scope:**
- Fracture statistics panel: orientation stereonet, size histogram, spacing distribution
- Block size distribution chart
- Connectivity metrics summary
- P10 / P32 comparison table (target vs achieved)
- Stereonet plot (Schmidt equal-area projection)
- Report generation: PDF or HTML summary of model parameters and results
- Batch processing: run analysis on multiple projects
- Export analysis results to CSV
- Unit and integration tests for all analysis outputs

**Dependencies:** M9 (visualization for charts and stereonets)
**Completed Work:** N/A
**Next:** M11 — release candidate

---

### M11 — Release Candidate

**Tag:** `v1.0.0-rc1`
**Status:** NOT STARTED
**Estimated Duration:** 2-3 weeks

**Goal:** Stabilization, packaging, documentation, and pre-release testing.

**Scope:**
- Full acceptance test suite execution (see ACCEPTANCE_TESTS.md)
- Bug fixing from test results
- Performance profiling and optimization
- Packaging for distribution:
  - Windows: standalone installer (PyInstaller / Nuitka)
  - Linux: AppImage or Flatpak
  - macOS: .app bundle
- User documentation:
  - Installation guide
  - Quick-start tutorial
  - Parameter reference
  - Theory manual (math behind DFN generation)
- Sample project files
- Changelog
- Contributor guide
- Version compatibility policy documented

**Dependencies:** M10 (all features complete)
**Completed Work:** N/A
**Next:** M12 — v1.0.0 final release

---

### M12 — v1.0.0 Final Release

**Tag:** `v1.0.0`
**Status:** NOT STARTED
**Estimated Duration:** 1 week

**Goal:** Publish the stable v1.0.0 release.

**Scope:**
- Finalize changelog
- Tag and sign the v1.0.0 release in git
- Publish packages to distribution channels
- Publish documentation online
- Announcement / release notes
- Archive the M0-M11 milestone artifacts

**Dependencies:** M11 (release candidate validated)
**Completed Work:** N/A
**Next:** Post-1.0 feature planning

---

## Timeline Summary

| Milestone | Estimated Duration | Cumulative |
|-----------|-------------------|------------|
| M0        | 1-2 weeks         | 1-2 weeks  |
| M1        | 1-2 weeks         | 2-4 weeks  |
| M2        | 2-3 weeks         | 4-7 weeks  |
| M3        | 2-3 weeks         | 6-10 weeks |
| M4        | 2-3 weeks         | 8-13 weeks |
| M5        | 1-2 weeks         | 9-15 weeks |
| M6        | 3-4 weeks         | 12-19 weeks|
| M7        | 1-2 weeks         | 13-21 weeks|
| M8        | 3-4 weeks         | 16-25 weeks|
| M9        | 3-4 weeks         | 19-29 weeks|
| M10       | 2-3 weeks         | 21-32 weeks|
| M11       | 2-3 weeks         | 23-35 weeks|
| M12       | 1 week            | 24-36 weeks|

**Total estimated project duration:** 6-9 months (single developer, full-time).
With a small team (2-3 developers), the timeline could compress to 4-6 months due to parallelizable work (M8 and M9 can partially overlap, M10 can start during M9).

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VTK/PyVista API instability | Medium | High | Pin dependency versions; abstract rendering backend |
| Large-DFN performance bottlenecks | Medium | Medium | Profile early (M6); design for LOD from M9 start |
| Cross-platform packaging issues | High | Medium | Test packaging on all targets in M10, not just M11 |
| Scope creep in visualization (M9) | High | Medium | Define MVP visualization features; defer advanced rendering |
| Fisher sampling statistical bias | Low | High | Validate against published implementations in M1 |

---

## Change Log

| Date       | Change                                        |
|------------|-----------------------------------------------|
| 2026-08-04 | Initial roadmap created. All milestones pending. |
