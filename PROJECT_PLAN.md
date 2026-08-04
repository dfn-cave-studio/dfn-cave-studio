# DFN Cave Studio -- Project Plan

## Project Overview

**DFN Cave Studio** is a desktop scientific application for underground block cave mining Discrete Fracture Network (DFN) modeling. The software provides geomechanical and mining engineers with an integrated environment to generate, analyse, validate, and export stochastic DFN models for block cave mining operations.

### Core Goals

1. **Stochastic DFN Generation**: Produce realistic 3D fracture networks using Fisher distribution sampling with global and local parameter control (P32, orientation, size distributions).
2. **Voxel-based Spatial Analysis**: Partition the domain into a voxel grid, compute explicit DFN-voxel intersections, and derive local intensity metrics (P32, P10, P21) for spatial characterisation.
3. **Connectivity and Percolation**: Build fracture connectivity graphs, identify connected clusters, and evaluate boundary percolation critical for cave propagation assessment.
4. **Block Fragmentation**: Cut explicit blocks from the fracture network, compute fragmentation size distributions, and generate statistical plots for cave draw analysis.
5. **Structural Integration**: Import deterministic large structures, borehole data, and support structural domain zoning with spatially varying DFN parameters.
6. **Export and Interop**: Assign mechanical properties to fractures and export to numerical simulation codes (e.g. 3DEC, FLAC3D, PFC).

### Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Scientific ecosystem, rapid development |
| GUI Framework | PySide6 (Qt 6) | Mature desktop toolkit, 3D via Qt 3D or vispy |
| 3D Visualisation | vispy / pyvista | Hardware-accelerated volume and mesh rendering |
| Numerical Core | NumPy, SciPy | Array operations, spatial data structures, statistics |
| Geometry | Shapely, trimesh | Explicit geometric intersections, mesh operations |
| Storage | HDF5 (h5py) + SQLite | Chunked voxel storage, project metadata |
| Packaging | PyInstaller + NSIS | Single-file Windows installer |
| Testing | pytest, pytest-qt, pytest-benchmark | Unit, integration, GUI, and performance tests |
| Linting / Formatting | ruff, mypy, black | Code quality enforcement |
| CI | GitHub Actions | Windows runner, automated test suite |

---

## Dependency Pinning Strategy

All direct dependencies are pinned to a **minor version floor** with an upper bound below the next breaking release:

```
numpy>=1.26,<2
scipy>=1.12,<2
pyside6>=6.6,<6.8
vispy>=0.14,<0.15
pyvista>=0.43,<0.45
shapely>=2.0,<2.1
trimesh>=4.1,<4.2
h5py>=3.10,<3.12
pyinstaller>=6.0,<6.5
pytest>=8.0,<9
pytest-qt>=4.4,<5
pytest-benchmark>=4.0,<5
ruff>=0.3,<0.4
mypy>=1.8,<2
```

- `requirements.in` + `pip-compile` produces a fully hashed `requirements.txt` checked into the repository.
- `dev-requirements.in` covers test, lint, and type-check tooling.
- Nightly CI job tests against the upper bound of each dependency range to catch regressions early.

---

## Testing Strategy

### Levels

| Level | Scope | Tool | Trigger |
|---|---|---|---|
| Unit | Individual functions and classes | pytest | Every push |
| Integration | Module interactions (e.g. voxeliser + intersection engine) | pytest | Every push |
| GUI | Qt widget behaviour, signal/slot wiring | pytest-qt | PR, main merge |
| Performance | Regression benchmarks for critical paths | pytest-benchmark | Nightly |
| End-to-End | Full workflow from generation to export | pytest + subprocess | Release candidate |

### Coverage Targets

- Core geometry and statistics modules: >= 90% line coverage.
- GUI layer: >= 60% line coverage (signal/slot logic; view code excluded).
- Overall project: >= 80% line coverage.

### Test Data

- A curated set of small (100 fracture), medium (1,000 fracture), and large (10,000 fracture) reference DFN realisations stored in `tests/data/`.
- Known-answer tests (KATs) for P32 computation, intersection counts, and cluster sizes validated against independent analytical or high-resolution Monte Carlo results.

---

## Scientific Validation Approach

1. **Analytical Benchmarks**: For uniform Poisson DFN in a cube, P32 equals total fracture area per unit volume. Validate computed P32 against the known analytical input within 0.1% tolerance for large realisations.
2. **Fisher Distribution**: Generate 1e6 samples and compare the empirical CDF of the colatitude angle against the theoretical Fisher CDF using the Kolmogorov-Smirnov test at alpha = 0.01.
3. **Intersection Correctness**: Cross-validate explicit fracture-fracture and fracture-voxel intersections against a brute-force O(N^2) reference implementation on small networks (N <= 50).
4. **Cluster Statistics**: Compare connected-component size distributions against published literature values (e.g. percolation threshold for Poisson disc networks).
5. **Borehole Sampling**: Simulate a virtual borehole through a network with known P10/P21, verify the line/plane intersection counts match analytical expectations.
6. **Fragmentation**: Compare block volume distributions against analytical results for orthogonal persistent joint sets.
7. **Reproducibility**: All stochastic routines use a seedable RNG; release tests run with fixed seeds and assert bitwise-identical output.

---

## Performance Targets

| Model Size | Fractures | Voxels (effective) | DFN Gen (s) | Intersection (s) | Graph Build (s) | Total RAM (GB) |
|---|---|---|---|---|---|---|
| Small | 500 | 8,000 (20^3) | < 1 | < 2 | < 1 | < 0.5 |
| Medium | 5,000 | 64,000 (40^3) | < 5 | < 15 | < 5 | < 2 |
| Large | 50,000 | 1,000,000 (100^3) | < 30 | < 120 | < 30 | < 8 |

Target hardware: Windows 10/11 x64, Intel i7-12700H or equivalent, 16 GB RAM, NVIDIA RTX 3060 or equivalent.

---

## Milestone Breakdown

---

### M0 -- Foundation and Project Scaffolding

**Objectives**: Establish repository structure, development environment, architecture documentation, and a minimal runnable Qt application shell.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M0.1 | Repository initialisation | Create Git repository with `.gitignore` (Python, Windows, IDE), `README.md`, `LICENSE` (MIT). |
| M0.2 | Python virtual environment | Create `venv` with Python 3.11; pin dependencies in `requirements.in` / `dev-requirements.in`; compile with `pip-compile --generate-hashes`. |
| M0.3 | Project directory layout | Establish `src/dfn_cave_studio/` package with subpackages: `core/`, `geometry/`, `generation/`, `voxel/`, `analysis/`, `io/`, `ui/`, `utils/`. |
| M0.4 | AGENTS.md | Write comprehensive agent instructions covering coding conventions, architecture, key design decisions, dependency rules, and contribution workflow. |
| M0.5 | Minimal Qt window | Create `src/dfn_cave_studio/ui/app.py` with a `QApplication`, `QMainWindow`, menu bar (File > Exit, Help > About), status bar showing "DFN Cave Studio v0.1.0". |
| M0.6 | Configuration system | Implement `src/dfn_cave_studio/utils/config.py` reading a TOML config file with sensible defaults. |
| M0.7 | Logging infrastructure | Set up structured logging via `logging` with console and file handlers in `src/dfn_cave_studio/utils/logging.py`. |
| M0.8 | CI pipeline | GitHub Actions workflow: checkout, setup Python, install dependencies, run ruff + mypy + pytest. |
| M0.9 | Entry point | `__main__.py` and `pyproject.toml` `[project.scripts]` entry point `dfn-cave-studio`. |

#### Deliverables

- Runnable `dfn-cave-studio` command opening an empty Qt window.
- Clean `ruff` and `mypy` checks (zero errors).
- Passing test scaffold (`tests/test_app.py` -- window opens and closes without crash).
- `AGENTS.md` with full project documentation for AI-assisted development.

#### Completion Criteria

1. `dfn-cave-studio` launches and displays the main window on Windows 10/11.
2. `pytest` collects and passes the scaffold test.
3. CI pipeline green on push.

#### Review Package (for stakeholder / peer review)

- `README.md`
- `AGENTS.md`
- `pyproject.toml`
- `requirements.txt`
- `src/dfn_cave_studio/ui/app.py`
- CI workflow YAML
- Screenshot of the running window

---

### M1 -- Unified Data Models

**Objectives**: Define the core geometry, fracture, joint set, and coordinate system data models that all downstream modules depend upon.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M1.1 | Coordinate system module | `src/dfn_cave_studio/geometry/coords.py`: define `Vec3`, `BBox3`, `Transform` (translation, rotation, scale). Implement vector operations and bounding-box queries (contains, intersects, volume). |
| M1.2 | Plane and polygon primitives | `src/dfn_cave_studio/geometry/primitives.py`: `Plane` (point + normal), `Polygon3` (ordered vertices, area, centroid, normal). |
| M1.3 | Disc fracture model | `src/dfn_cave_studio/geometry/fracture.py`: `Fracture` dataclass -- id (UUID), centre `Vec3`, radius `float`, normal `Vec3`, aperture `float`, transmissivity `float`. Methods: `area()`, `as_polygon(n_segments)`, `intersects_plane()`. |
| M1.4 | Joint set definition | `src/dfn_cave_studio/generation/joint_set.py`: `JointSet` -- name, orientation (Fisher kappa, mean pole), size distribution (lognormal mu/sigma or power-law alpha/Dmin/Dmax), intensity (P32 or P30 target), shape (disc / polygon). |
| M1.5 | Fracture network container | `src/dfn_cave_studio/generation/network.py`: `FractureNetwork` -- ordered collection of `Fracture` objects, bounding box, joint set index per fracture, summary statistics (count, total area, P32). |
| M1.6 | Serialisation | JSON round-trip for `JointSet`, `FractureNetwork`; HDF5 for large fracture arrays. |
| M1.7 | Unit tests | 100% coverage on geometry primitives; property-based tests for BBox operations. |

#### Deliverables

- Fully typed geometry and data model modules.
- Serialisation round-trip tests.
- Jupyter notebook demonstrating the data model API (for review).

#### Completion Criteria

1. All geometry operations numerically validated (area within 1e-12 of analytical).
2. HDF5 read/write round-trip produces byte-identical arrays.
3. `mypy --strict` passes on `geometry/` and `generation/` packages.

#### Review Package

- `src/dfn_cave_studio/geometry/coords.py`
- `src/dfn_cave_studio/geometry/primitives.py`
- `src/dfn_cave_studio/geometry/fracture.py`
- `src/dfn_cave_studio/generation/joint_set.py`
- `src/dfn_cave_studio/generation/network.py`
- Unit test files
- API demo notebook

---

### M2 -- Global Stochastic DFN Generation and 3D Display

**Objectives**: Implement constant-parameter DFN generation for disc fractures following Fisher orientation and user-specified size distributions. Control global P32. Render the generated network in an interactive 3D viewport.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M2.1 | Fisher distribution sampler | `src/dfn_cave_studio/generation/fisher.py`: sample unit vectors from Fisher distribution with given mean direction and kappa. Validate against analytical CDF. |
| M2.2 | Size distribution samplers | Lognormal, power-law (truncated Pareto), exponential samplers in `src/dfn_cave_studio/generation/size_samplers.py`. |
| M2.3 | DFN generator core | `src/dfn_cave_studio/generation/generator.py`: `DFNGenerator` class. Given domain `BBox3`, list of `JointSet`, and target P32, generate fractures by Poisson disc process: (a) draw N from Poisson(rate) where rate = P32_target * V_domain / mean_fracture_area; (b) for each fracture, sample centre uniformly, orientation from Fisher, radius from size distribution; (c) filter fractures completely outside domain. |
| M2.4 | P32 control loop | Iterative refinement: measure achieved P32, adjust N, regenerate until within 0.5% of target, or cap iterations at 10. |
| M2.5 | 3D viewport | Integrate vispy or pyvista `QtRenderWindowInteractor` into a `QWidget`. Render fractures as semi-transparent coloured discs, colour-coded by joint set. |
| M2.6 | Viewport controls | Orbit/pan/zoom. Bounding-box wireframe. Joint set legend with visibility toggles. |
| M2.7 | DFN statistics panel | Dock widget displaying fracture count, P32, intensity by joint set, orientation rose diagram (2D projection). |
| M2.8 | Performance tests | Benchmark generator on small/medium/large target sizes; assert within performance targets. |

#### Deliverables

- Interactive 3D view showing generated fracture discs.
- Dockable statistics panel with live P32 readout.
- Orientation stereonet / rose diagram widget.
- Benchmark report for M2.8.

#### Completion Criteria

1. Generated P32 within 0.5% of target for >= 95% of trials (100 random seeds, small model).
2. Fisher sampling passes KS test at alpha=0.01 for kappa in [5, 50].
3. 3D viewport renders 5,000 fractures at >= 10 fps during orbit.
4. Performance targets met for small and medium models.

#### Review Package

- `src/dfn_cave_studio/generation/fisher.py`
- `src/dfn_cave_studio/generation/generator.py`
- `src/dfn_cave_studio/ui/viewport.py`
- `src/dfn_cave_studio/ui/stats_panel.py`
- Validation test results (KS test, P32 convergence plots)
- Benchmark JSON output

---

### M3 -- Voxel Space and Sparse Storage

**Objectives**: Partition the domain into a voxel grid. Implement a sparse/chunked storage backend in HDF5 for efficient access to large grids.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M3.1 | Voxel grid definition | `src/dfn_cave_studio/voxel/grid.py`: `VoxelGrid` class -- origin, voxel size (dx), grid dimensions (nx, ny, nz), bounding box. Methods: `voxel_index(point)`, `voxel_bbox(i,j,k)`, `iter_voxels()`. |
| M3.2 | Surface mask | `src/dfn_cave_studio/voxel/mask.py`: `SurfaceMask` -- binary 3D array (or sparse) marking voxels inside a closed triangulated surface (e.g. orebody wireframe). Import from STL/OBJ. Method: `is_active(i,j,k)`. |
| M3.3 | Sparse voxel storage | `src/dfn_cave_studio/voxel/storage.py`: `SparseVoxelStore` backed by HDF5. Chunked layout (e.g. 32^3 chunks). Supports `get_chunk(cx,cy,cz)`, `set_voxel(i,j,k, data)`. Data schema: P32 (float32), fracture_count (int32), per-joint-set P32 (float32 array). |
| M3.4 | Masked iterator | Efficient iterator over active voxels only, respecting the surface mask and skipping empty chunks. |
| M3.5 | Unit + integration tests | Verify grid indexing, chunk I/O, masked iteration on synthetic data. |

#### Deliverables

- Voxel grid module with sparse HDF5 backend.
- Surface mask import from STL/OBJ.
- Test datasets: a 100^3 grid with a spherical mask.

#### Completion Criteria

1. Voxel index computation correct to machine precision.
2. Chunked read/write round-trip preserves all data.
3. Sparse iteration over a 100^3 grid with 10% active voxels completes in < 0.5 s.
4. Surface mask correctly classifies >= 99.9% of voxels against a reference ray-casting implementation.

#### Review Package

- `src/dfn_cave_studio/voxel/grid.py`
- `src/dfn_cave_studio/voxel/mask.py`
- `src/dfn_cave_studio/voxel/storage.py`
- HDF5 schema documentation
- Mask validation report

---

### M4 -- Explicit DFN-Voxel Intersection and Local P32

**Objectives**: Compute explicit intersections between every fracture and the voxel grid. Accumulate per-voxel P32 and fracture counts for spatial intensity mapping.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M4.1 | Fracture-voxel intersection | `src/dfn_cave_studio/voxel/intersection.py`: for each fracture, determine which voxels its bounding sphere intersects, then test disc-polygon intersection against each candidate voxel's faces. Accumulate intersected area into the voxel's P32 sum. |
| M4.2 | Spatial indexing | Use a grid-based spatial index to accelerate fracture lookup by voxel. Build a dictionary mapping `(cx, cy, cz)` chunk to list of fracture indices whose bounding spheres overlap the chunk. |
| M4.3 | Parallel execution | Use `concurrent.futures.ProcessPoolExecutor` or `numba` to parallelise the intersection loop over fractures or chunks. |
| M4.4 | Voxel field computation | Compute per-voxel: P32 (total fracture area / voxel volume), fracture count, per-joint-set P32 breakdown. |
| M4.5 | Voxel visualisation | Render voxel grid as a 3D scalar field. Active voxels coloured by P32 (colormap: blue-green-yellow-red). Support an opacity transfer function and isosurface extraction. |
| M4.6 | Performance optimisation | Profile and optimise the hot path (disc-AABB intersection). Target: medium model (5,000 fractures, 64k voxels) in < 15 s. |
| M4.7 | Validation | Compare computed local P32 against global P32 for a uniform network (should match within sampling error). |

#### Deliverables

- Intersection engine with spatial indexing and parallelisation.
- Voxel P32 field viewer with colormap and opacity controls.
- Performance profiling report.

#### Completion Criteria

1. Local P32 integrated over domain equals global P32 within 0.1% for uniform networks.
2. Medium model intersection completes within 15 s.
3. Voxel viewer renders and updates colour mapping at interactive rates (>= 5 fps for 64k voxels).

#### Review Package

- `src/dfn_cave_studio/voxel/intersection.py`
- `src/dfn_cave_studio/ui/voxel_viewer.py`
- Validation report (local vs global P32)
- Profiling flamegraph / report

---

### M5 -- Fracture Connectivity Graph and Percolation

**Objectives**: Build a fracture connectivity graph, identify connected clusters, and analyse boundary-to-boundary percolation critical for caveability assessment.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M5.1 | Fracture-fracture intersection | `src/dfn_cave_studio/analysis/intersection.py`: pairwise disc-disc intersection test. Return intersection line segment and intersection point for each pair of intersecting fractures. |
| M5.2 | Connectivity graph | `src/dfn_cave_studio/analysis/graph.py`: build undirected graph (networkx or custom sparse adjacency). Nodes = fractures, edges = intersections. Edge weight = intersection length or area. |
| M5.3 | Spatial acceleration | Use an R-tree (via `rtree` or `scipy.spatial.KDTree` on bounding sphere centres) to prune non-candidate pairs before pairwise intersection testing. |
| M5.4 | Connected components | Compute connected components (clusters) of the fracture graph. Compute cluster statistics: size distribution, largest cluster fraction, mean cluster size. |
| M5.5 | Boundary percolation | `src/dfn_cave_studio/analysis/percolation.py`: identify clusters that connect specified boundary faces (e.g. top-to-bottom, north-to-south). Compute percolation probability vs P32. |
| M5.6 | Cluster visualisation | Render connected clusters in 3D with distinct colours per cluster. Highlight percolating clusters. |
| M5.7 | Graph analysis tools | Compute additional graph metrics: degree distribution, average clustering coefficient, betweenness centrality for key fractures. |
| M5.8 | Validation | Validate percolation threshold against known results for Poisson disc networks (literature comparison). |

#### Deliverables

- Fracture connectivity graph module.
- Cluster identification and percolation analysis.
- Cluster visualisation with percolation highlighting.
- Validation report against published percolation thresholds.

#### Completion Criteria

1. Graph construction for medium model (5,000 fractures) completes within 5 s.
2. Percolation threshold for uniform disc network within 10% of published values.
3. Interactive cluster colouring updates within 2 s on cluster recomputation.

#### Review Package

- `src/dfn_cave_studio/analysis/intersection.py`
- `src/dfn_cave_studio/analysis/graph.py`
- `src/dfn_cave_studio/analysis/percolation.py`
- Cluster visualisation screenshot
- Percolation validation plot

---

### M6 -- Deterministic Structures and Borehole Import

**Objectives**: Import large deterministic geological structures (faults, dykes, shear zones) from industry formats and borehole fracture logs for hybrid DFN construction.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M6.1 | Large structure import | `src/dfn_cave_studio/io/structures.py`: import deterministic structures from DXF, VTK, or CSV (triangulated surfaces). Parse and convert to internal `Fracture` (as polygon fractures) or dedicated `LargeStructure` class with persistence (deterministic, not stochastic). |
| M6.2 | Borehole data import | `src/dfn_cave_studio/io/borehole.py`: import borehole trajectories (collar + survey) and fracture logs (depth, alpha, beta, aperture, infill) from CSV, Excel, or LAS formats. |
| M6.3 | Borehole trajectory model | `src/dfn_cave_studio/geometry/borehole.py`: `Borehole` class with collar position, azimuth, dip, and survey points defining a 3D polyline. |
| M6.4 | Borehole fracture log | `src/dfn_cave_studio/geometry/borehole.py`: `BoreholeFracture` -- depth, orientation relative to core axis (alpha/beta), aperture, infill, roughness. Methods to convert to world-space plane. |
| M6.5 | Hybrid DFN construction | `src/dfn_cave_studio/generation/hybrid.py`: merge deterministic structures with stochastically generated background fractures. Ensure no duplicate fractures at deterministic locations. |
| M6.6 | Import UI dialogs | File > Import > Structures, File > Import > Boreholes. Preview imported geometry in 3D view. |
| M6.7 | Unit tests | Round-trip import/export tests with synthetic borehole and structure data. |

#### Deliverables

- Deterministic structure and borehole import modules.
- Import dialogs with preview.
- Hybrid DFN construction (deterministic + stochastic).

#### Completion Criteria

1. DXF, CSV, and LAS formats parse correctly for representative industry sample files (provided in test data).
2. Borehole trajectory and fracture log import produce correct world-space orientations.
3. Hybrid DFN correctly excludes stochastic fractures that would intersect deterministic structures (optional; configurable overlap tolerance).

#### Review Package

- `src/dfn_cave_studio/io/structures.py`
- `src/dfn_cave_studio/io/borehole.py`
- `src/dfn_cave_studio/geometry/borehole.py`
- `src/dfn_cave_studio/generation/hybrid.py`
- Sample import files (anonymised)
- Import dialog screenshots

---

### M7 -- Virtual Borehole Sampling and DFN Validation

**Objectives**: Simulate virtual boreholes through the generated DFN to compute P10 (fractures per metre) and P21 (fracture trace length per unit area). Compare against target values for model validation.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M7.1 | Virtual borehole P10 | `src/dfn_cave_studio/analysis/borehole_sampling.py`: generate a virtual borehole (straight line) through the DFN. Intersect with all fractures. Count intersections per unit length = P10. |
| M7.2 | Virtual borehole P21 | For a virtual borehole, compute the trace length of each intersecting fracture on a planar window aligned with the borehole. P21 = total trace length / window area. |
| M7.3 | Scanline and window sampling | Extend to multiple parallel virtual boreholes (scanline array) and rectangular sampling windows. |
| M7.4 | Orientation bias correction | Apply Terzaghi weighting to correct for orientation bias (fractures sub-parallel to borehole are under-sampled). |
| M7.5 | Validation report | `src/dfn_cave_studio/analysis/validation.py`: compare virtual P10/P21 against input joint set targets. Produce report with per-joint-set statistics and overall error. |
| M7.6 | Validation UI | Dock widget showing virtual borehole locations in 3D view, intersection markers, and a table of P10/P21 by joint set vs target. |
| M7.7 | Batch validation | Run N virtual boreholes with random orientations, aggregate statistics, produce box plots of P10/P21 error. |

#### Deliverables

- Virtual borehole sampling engine with P10 and P21 computation.
- Orientation bias correction.
- Validation report generation.
- Validation UI with inline comparison to targets.

#### Completion Criteria

1. Virtual P10 on a uniform DFN agrees with analytical P10 = P32 * mean_fracture_diameter / 4 within 5% for >= 100 boreholes.
2. Terzaghi correction reduces orientation bias to < 2% for boreholes at 30-degree inclination to fracture set mean.
3. Batch validation completes for 100 boreholes through 5,000 fractures in < 10 s.

#### Review Package

- `src/dfn_cave_studio/analysis/borehole_sampling.py`
- `src/dfn_cave_studio/analysis/validation.py`
- Validation report PDF / HTML sample
- Validation UI screenshot

---

### M8 -- Explicit Block Cutting and Fragmentation

**Objectives**: Cut the rock mass into blocks defined by the fracture network. Compute block volume distribution and fragmentation metrics for cave draw assessment.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M8.1 | Block cutting algorithm | `src/dfn_cave_studio/analysis/fragmentation.py`: implement a block cutting algorithm. Approach: use generalised half-space intersection or extend fracture planes to domain boundaries and apply 3D cell decomposition. Alternative: use `pysph` or custom plane-based subdivision. Store blocks as closed polyhedra. |
| M8.2 | Domain boundary handling | Clip blocks to the domain bounding box or to an imported orebody surface. Blocks outside the orebody are discarded. |
| M8.3 | Block volume computation | Compute volume of each block via divergence theorem or tetrahedral decomposition. |
| M8.4 | Fragmentation metrics | Compute: D10, D30, D50, D80 (characteristic block sizes by volume equivalent sphere diameter). Coefficient of uniformity (D60/D10). Block count. |
| M8.5 | Statistical plots | Block volume histogram (log-log), cumulative passing curve, block shape metrics (sphericity, aspect ratio). |
| M8.6 | Fragmentation UI | Dock widget with fragmentation curve plot, block size distribution table, and 3D view of individual blocks coloured by volume. |
| M8.7 | Validation | Validate on a network of three orthogonal persistent joint sets with known spacing -- analytical block volume distribution should match within sampling error. |
| M8.8 | Performance | Medium model (5,000 fractures producing ~10,000 blocks) completes cutting in < 30 s. |

#### Deliverables

- Block cutting engine with domain clipping.
- Fragmentation metrics and statistical plot module.
- Fragmentation visualisation.
- Validation against orthogonal joint set analytical solution.

#### Completion Criteria

1. D50 for orthogonal persistent sets matches analytical block size within 5%.
2. Block volumes sum to domain volume within 0.1%.
3. Medium model cutting within 30 s.

#### Review Package

- `src/dfn_cave_studio/analysis/fragmentation.py`
- Fragmentation plot screenshots
- Validation report (analytical vs computed)
- Performance benchmark

---

### M9 -- Structural Domain Division and Zonal DFN

**Objectives**: Divide the modelling domain into structural sub-domains, each with independent joint set definitions and DFN parameters, to represent geological heterogeneity.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M9.1 | Domain boundary definition | `src/dfn_cave_studio/generation/domain.py`: `StructuralDomain` -- name, bounding surface (closed triangulated mesh), list of `JointSet` specific to this domain. |
| M9.2 | Domain assignment | For each point in the domain, determine which structural domain it belongs to (point-in-mesh test). Handle domain boundaries (shared faces, priority rules). |
| M9.3 | Zonal DFN generator | Extend `DFNGenerator` to generate fractures domain-by-domain. Each domain uses its own joint sets and P32 target. Fractures are clipped to their domain boundaries. |
| M9.4 | Domain transition handling | Implement strategies for domain boundaries: (a) truncate fractures at boundary, (b) allow fractures to cross boundaries with probability p, (c) generate boundary-crossing fractures from merged statistics. |
| M9.5 | Domain visualisation | Render domain boundaries as semi-transparent surfaces in 3D. Colour fractures by their parent domain. |
| M9.6 | Domain editing UI | UI for creating, editing, and deleting structural domains. Import domain surfaces. Assign joint sets to domains via drag-and-drop or property editor. |
| M9.7 | Validation | Generate a two-domain model with a vertical planar boundary. Verify fracture count per domain matches P32 targets. Check boundary crossing behaviour. |

#### Deliverables

- Structural domain data model and editor.
- Zonal DFN generator with boundary handling.
- Domain visualisation.

#### Completion Criteria

1. P32 per domain matches target within 1% for three-domain model.
2. Domain editor supports import and manual creation of domain boundaries.
3. Boundary-crossing fractures correctly handled (no gaps or overlaps along domain seams).

#### Review Package

- `src/dfn_cave_studio/generation/domain.py`
- Domain editor UI screenshot
- Multi-domain DFN visualisation
- Domain P32 validation report

---

### M10 -- Continuous Local DFN Parameter Field

**Objectives**: Replace discrete structural domains with a continuous spatially varying DFN parameter field, allowing P32, orientation, and size to vary smoothly across the domain.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M10.1 | Parameter field definition | `src/dfn_cave_studio/generation/field.py`: `ParameterField` -- a 3D scalar/vector field defined on the voxel grid. Field types: P32 (scalar), Fisher kappa (scalar), mean pole (vector3), mean radius (scalar). |
| M10.2 | Field interpolation | Tri-linear interpolation within voxels. Option for Gaussian process (kriging) interpolation when conditioned on sparse data. |
| M10.3 | Field-conditioned generation | `FieldDFNGenerator`: for each fracture candidate, sample centre uniformly. Look up local P32 at centre; accept/reject based on local vs global intensity ratio (thinning Poisson process). Sample orientation from Fisher with local kappa and mean pole. Sample radius from distribution with local mean. |
| M10.4 | Field editor | UI for defining parameter fields: (a) import from grid file (GSLIB, VTK), (b) manual definition via control points with kriging interpolation, (c) derive from borehole data via inverse distance weighting. |
| M10.5 | Field visualisation | Render parameter field as a 3D volume with slice planes, isosurfaces, and colour mapping. |
| M10.6 | Validation | Generate a linear P32 gradient field (0 at top, P32_max at bottom). Verify that local P32 computed via M4 intersection reproduces the input gradient. |
| M10.7 | Field export | Export parameter fields to VTK, GSLIB, or CSV for use in external tools. |

#### Deliverables

- Continuous parameter field data model
- Field-conditioned DFN generator
- Field editor UI with import and kriging
- Field visualisation with slice planes

#### Completion Criteria

1. Generated local P32 reproduces input parameter field with R^2 >= 0.95 for a smooth field on a 40^3 grid.
2. Kriging interpolation from 10 control points reproduces a known analytical field with RMS error < 5%.
3. Field import/export round-trip preserves data to float32 precision.

#### Review Package

- `src/dfn_cave_studio/generation/field.py`
- Field editor screenshots
- Parameter field visualisation
- Field validation plots

---

### M11 -- Mechanical Property Assignment and Simulation Export

**Objectives**: Assign mechanical properties to fractures (stiffness, friction, cohesion) based on joint set or local rules. Export the DFN to formats consumable by numerical simulation codes (3DEC, FLAC3D, PFC, UDEC).

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M11.1 | Mechanical property model | `src/dfn_cave_studio/analysis/mechanics.py`: `MechanicalProperties` -- joint normal stiffness (kn), shear stiffness (ks), friction angle, cohesion, tensile strength, dilation angle. Assign per joint set with optional degradation rules (e.g. friction reduces with slip). |
| M11.2 | Property assignment | Assign properties to each fracture in the network. Support: (a) constant per joint set, (b) spatially varying (linked to parameter fields from M10), (c) imported property maps. |
| M11.3 | 3DEC export | `src/dfn_cave_studio/io/export_3dec.py`: write 3DEC `jset` commands for each fracture. Format: `jset id N x y z dip dd radius kn ks fric coh ten`. Optionally write block cutting commands. |
| M11.4 | FLAC3D export | `src/dfn_cave_studio/io/export_flac3d.py`: export fractures as interface elements or as ubiquitous joint model parameters. |
| M11.5 | Generic export formats | VTK unstructured grid (polydata) with cell data arrays for all properties. CSV table of all fractures with properties. |
| M11.6 | PFC export | Ball/particle positions and bonds if the DFN drives a bonded particle model. |
| M11.7 | Export UI | File > Export menu with format-specific option dialogs. Preview export summary before writing. |
| M11.8 | Validation | Export a simple DFN, import into 3DEC (or parse the command file), verify fracture count, positions, and properties match. |

#### Deliverables

- Mechanical property model and assignment system.
- 3DEC, FLAC3D, VTK, and CSV export modules.
- Export UI with preview.

#### Completion Criteria

1. 3DEC export of a 100-fracture network produces a valid command file that runs without syntax errors in 3DEC.
2. Property assignment correctly handles per-joint-set and spatially varying cases.
3. VTK export loads correctly in ParaView with all cell data arrays.

#### Review Package

- `src/dfn_cave_studio/analysis/mechanics.py`
- `src/dfn_cave_studio/io/export_3dec.py`
- `src/dfn_cave_studio/io/export_flac3d.py`
- Sample export files
- Export UI screenshots

---

### M12 -- Integration, Hardening, and Packaging

**Objectives**: Complete project management features, error handling, performance optimisation, thorough testing, and Windows installer packaging.

#### Tasks

| ID | Task | Detail |
|---|---|---|
| M12.1 | Project management | `src/dfn_cave_studio/ui/project.py`: DFN project file (`.dfnp`) -- HDF5 container bundling all data: fracture network, voxel grid, parameter fields, analysis results, view state. File > New / Open / Save / Save As. Recent files list. |
| M12.2 | Undo/redo framework | Command-pattern undo stack for all destructive operations (delete fractures, edit joint set, modify domain). Integrated with Qt's `QUndoStack`. |
| M12.3 | Comprehensive error handling | Replace all bare `except:` with specific exception types. User-facing error dialogs for recoverable errors (file not found, invalid input). Crash reporter writing stack trace to log. |
| M12.4 | Progress reporting | All long-running operations (generation, intersection, block cutting) emit progress signals (0-100%). UI shows `QProgressBar` in status bar. Cancellation support via `QThread` / `QtConcurrent` with cancellation tokens. |
| M12.5 | Performance optimisation | Profile full workflow (generation -> intersection -> graph -> blocks -> export) on large model. Identify and resolve bottlenecks. Target: large model end-to-end in < 5 minutes. |
| M12.6 | Memory optimisation | Replace in-memory copies with memory-mapped HDF5 arrays where possible. Lazy loading of chunks in voxel viewer. `__slots__` on high-count data classes. |
| M12.7 | Full test suite | Achieve >= 80% overall line coverage. Add integration tests for complete workflows. Add GUI tests for all dialogs and panels. |
| M12.8 | User documentation | In-app help system (`QHelpEngine`). Tooltips on all controls. Status bar hints. |
| M12.9 | Windows packaging | PyInstaller spec file producing a single `.exe` or installer directory. NSIS installer script with license, shortcuts, file associations (`.dfnp`). |
| M12.10 | Release checklist | Version bump to 1.0.0. Tagged release on GitHub. Installer smoke test on clean Windows 10 and Windows 11 VMs. Signed executable (optional: self-signed for internal use). |

#### Deliverables

- Complete DFN Cave Studio application.
- Windows installer (`.exe`).
- User documentation (in-app help).
- Release test report.

#### Completion Criteria

1. Full workflow (new project -> define joint sets -> generate DFN -> analyse -> export) completes without error.
2. Installer installs and launches on clean Windows 10 and Windows 11.
3. All 145+ tests pass.
4. Large model end-to-end within 5 minutes.
5. `.dfnp` project file round-trip preserves all data.

#### Review Package

- Windows installer
- Release notes
- Test coverage report
- Performance benchmark report
- In-app help screenshots

---

## Milestone Dependency Graph

```
M0 (Foundation)
 |
M1 (Data Models)
 |
M2 (Global DFN + 3D View)
 |
+----+----+
|         |
M3        M5
(Voxel)   (Graph)
|         |
M4        |
(Intersection)
|         |
+----+----+
|         |
M7 <------+ (uses both voxel and graph for borehole analysis)
(Borehole)
|
M8 (Fragmentation -- needs intersection and graph)
|
M6 (Deterministic + Boreholes -- can run parallel to M5-M8)
|
M9 (Domains)
|
M10 (Fields)
|
M11 (Mechanics + Export)
|
M12 (Hardening + Packaging)
```

Milestones M6 can be developed in parallel with M5-M8. M9 and M10 can be partially overlapped with M11 if staffing allows.

---

## Appendix A -- Coding Conventions

- **Package layout**: `src/` layout with `pyproject.toml`; all imports use fully qualified `dfn_cave_studio.xxx`.
- **Type hints**: All public functions and methods must have complete type annotations; `mypy --strict` on core modules.
- **Docstrings**: NumPy-style docstrings with Parameters, Returns, Raises sections.
- **Immutability**: Geometry primitives (`Vec3`, `BBox3`) are immutable (frozen dataclasses or named tuples).
- **RNG**: All stochastic functions accept an explicit `rng: np.random.Generator` parameter; no global RNG state.
- **Naming**: `snake_case` for modules, functions, variables; `PascalCase` for classes.
- **Logging**: Use `logger = logging.getLogger(__name__)` at module level; no `print()` in library code.

## Appendix B -- Review Checklist (per milestone)

Before marking a milestone complete, the following must be verified by a reviewer who is not the primary developer of that milestone:

1. All tasks marked done with evidence (commits, test results).
2. Deliverables are present and correspond to the task list.
3. Completion criteria are met with objective evidence (test logs, screenshots, benchmark output).
4. Code review completed on all new modules (>= 1 approval).
5. `ruff` and `mypy` pass with zero errors.
6. Documentation (docstrings, inline comments) is adequate for a new developer.
7. No regressions in prior milestones (full test suite passes).
8. Performance targets met or an accepted deviation documented with rationale.

## Appendix C -- Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Block cutting algorithm complexity exceeds estimate | Medium | High | Research and prototype in M5; fall back to simplified cell decomposition if full half-space intersection too slow. |
| 3D rendering performance insufficient for large models | Medium | Medium | Implement LOD (level-of-detail) rendering; decimate discs to hexagons at far zoom; use instanced rendering. |
| HDF5 concurrent access issues with parallel intersection | Low | Medium | Use process-local HDF5 file handles; write results in independent chunks; merge in single-threaded pass. |
| Numerical simulation export formats change between versions | Low | Low | Pin to documented format versions; maintain format compatibility table; add format version detection in import. |
| Windows packaging (PyInstaller) compatibility with scientific stack | Medium | Medium | Test packaging from M0 onwards; maintain a dedicated packaging CI job; have a conda-based fallback. |
| Insufficient test coverage for GUI components | Medium | Low | Enforce minimum coverage per module; use pytest-qt for signal/slot testing; record-and-replay for integration tests. |

## Appendix D -- Glossary

| Term | Definition |
|---|---|
| DFN | Discrete Fracture Network -- a collection of individual fracture planes representing rock mass discontinuities. |
| P10 | Linear fracture frequency: number of fractures intersected per unit length along a borehole or scanline (m^-1). |
| P21 | Areal fracture intensity: total trace length of fractures per unit area of a sampling window (m/m^2 = m^-1). |
| P32 | Volumetric fracture intensity: total fracture area per unit volume of rock mass (m^2/m^3 = m^-1). |
| P30 | Volumetric fracture density: number of fracture centres per unit volume (m^-3). |
| Fisher distribution | A spherical probability distribution describing the orientation dispersion of fracture poles around a mean direction. Controlled by kappa (concentration parameter). |
| Kappa | Fisher concentration parameter; high kappa = tightly clustered orientations, low kappa = widely dispersed. |
| Percolation | The existence of a connected fracture cluster that spans the domain from one boundary to another, enabling fluid flow or affecting cave propagation. |
| Block cave mining | An underground mining method where an orebody is undercut, causing it to fracture and collapse under gravity into drawpoints for extraction. |
| Structural domain | A sub-volume of the rock mass with statistically homogeneous fracture characteristics (orientation, intensity, size). |
| Joint set | A group of fractures sharing similar orientation and genetic origin, described by statistical distributions. |
| Terzaghi correction | A weighting factor applied to borehole fracture counts to correct for the lower probability of intersecting fractures oriented sub-parallel to the borehole. |
| RQD | Rock Quality Designation -- the percentage of intact core pieces longer than 10 cm, used as an index of rock mass quality. |
