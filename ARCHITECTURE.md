# DFN Cave Studio -- Architecture Document

**DFN Cave Studio** is a desktop scientific application for underground block cave mining DFN (Discrete Fracture Network) modeling. It provides geologists and mining engineers with tools to generate, analyze, and export fracture networks, compute block fragmentation, assess connectivity, and prepare simulation models for external numerical solvers (3DEC, FLAC3D, etc.).

This document describes the high-level architecture, module dependencies, data flows, threading model, file formats, and extension mechanisms that govern the project.

---

## 1. High-Level Layered Architecture

The application is organized into seven horizontal layers. Each layer depends only on layers below it; no upward dependencies are permitted.

```
+---------------------------------------------------------------+
|                    Qt UI Layer  (PySide6)                      |
|  Widgets, dialogs, panels, menus, toolbars, viewports, plots   |
+---------------------------------------------------------------+
|               3D Visualization Layer  (PyVista/VTK)            |
|  Render windows, actors, filters, pickers, 3D interaction      |
+---------------------------------------------------------------+
|               Background Task Layer  (QRunnable workers)       |
|  Long-running computation offload, progress signalling         |
+---------------------------------------------------------------+
|               Scientific Algorithm Layer                       |
|  geometry | dfn | voxel | connectivity | fragmentation |       |
|  geology  | borehole | mechanics                               |
+---------------------------------------------------------------+
|               Data Model Layer  (Pydantic / dataclass)         |
|  Immutable value objects, command objects, config schemas      |
+---------------------------------------------------------------+
|               File I/O Layer  (persistence + services)         |
|  Project JSON save/restore, HDF5/Zarr bulk arrays, exporters   |
+---------------------------------------------------------------+
|               Project Save / Restore Layer                     |
|  Project manifest, asset registry, version migration           |
+---------------------------------------------------------------+
```

### Layer Details

#### 1.1 Data Model Layer (`models/`)

Contains all domain concepts as Pydantic `BaseModel` subclasses and standard-library
`dataclass` types. Models are entirely decoupled from I/O and UI concerns.

Key models:
- `Project` -- top-level container holding metadata, file manifests, and references to domain objects.
- `DFNModel` -- a named fracture network with a set of `Fracture` objects.
- `Fracture` -- shape (disc/polygon), center, orientation, radius, aperture, mechanical properties.
- `VoxelGrid` -- regular 3D grid with origin, spacing, dimensions, and chunked data arrays.
- `ConnectivityGraph` -- nodes (fractures) and edges (intersection segments) with computed metrics.
- `FragmentationResult` -- set of polyhedral blocks derived from DFN and tunnel geometries.
- `GeologicalDomain` -- lithology units, faults, contacts, grade shells.
- `Borehole` / `BoreholeInterval` -- logged intervals with fracture, alteration, and assay data.

Pydantic is chosen for built-in validation, serialization round-tripping, and JSON Schema
generation that the persistence layer uses.

#### 1.2 Scientific Algorithm Layer

Pure computational modules with no side effects. Each module operates on data-model
objects and returns results as new data-model objects.

| Module            | Responsibility                                               |
|-------------------|--------------------------------------------------------------|
| `geometry/`       | Vec3, Plane, AABB, OBB, intersection tests, convex hull      |
| `dfn/`            | Stochastic DFN generation (FracMan-compatible, enhanced-Baecher, etc.) |
| `voxel/`          | Grid construction, fracture-to-grid intersection, property population |
| `connectivity/`   | Graph construction, component finding, percolation analysis   |
| `fragmentation/`  | Block tracing, polygon clipping, GMG/GeneralBlock algorithms |
| `geology/`        | Domain boundary surfaces, implicit modelling (RBF, kriging)   |
| `borehole/`       | Interval processing, stereonet statistics, intensity (P10/P32) |
| `mechanics/`      | Property assignment rules, equivalent continuum upscaling     |

#### 1.3 File I/O Layer (`persistence/`, `services/`)

- `persistence/` -- low-level readers/writers for project files, CSV, DXF, VTK, HDF5, Zarr.
- `services/` -- higher-level orchestration that coordinates reading, writing, and model assembly.

#### 1.4 3D Visualization Layer (`visualization/`)

Thin wrappers around PyVista that accept data-model objects and produce renderable scenes.
Exposes functions like `render_df_network(dfn) -> pv.Plotter`, `render_voxel_grid(grid, property) -> ...`.
The wrappers exist so the Qt UI layer never imports PyVista/VTK directly, enabling
testing the UI with a headless or mock visualisation backend.

#### 1.5 Qt UI Layer (`ui/`)

Widgets, dialogs, dockable panels, main-window layout, menus, toolbars, and settings
editors. All UI classes delegate computation to the Scientific Algorithm Layer (or
Background Task Layer for long-running work) and render geometry through the Visualization Layer.

#### 1.6 Background Task Layer (`workers/`)

Each long-running operation (DFN generation, voxelization, fragmentation, export) has a
dedicated `QRunnable` subclass. Workers emit signals through a `QObject` signal proxy.
The main thread remains responsive while progress, partial results, and final results
are delivered via queued connections.

#### 1.7 Project Save / Restore Layer

A dedicated orchestration layer that serializes the entire application state (open
models, viewport camera, UI layout, undo history) into a structured JSON-based project
file. Large numeric arrays (voxel data, matrix properties) are stored in HDF5 or Zarr
sidecar files referenced from the JSON manifest.

---

## 2. Module Dependency Diagram

```
src/dfn_cave_studio/
|-- app/                    Entry point, application bootstrap
|-- utils/                  Shared utility functions (no deps on other app modules)
|
|-- models/                 Pure data (depends on pydantic, numpy; NO app deps)
|
|-- geometry/               Depends on models (Vec3, Plane, etc.)
|-- dfn/                    Depends on geometry, models
|-- voxel/                  Depends on geometry, models
|-- connectivity/           Depends on dfn, geometry, models
|-- fragmentation/          Depends on connectivity, voxel, geometry, models
|-- geology/                Depends on geometry, models
|-- borehole/               Depends on geology, geometry, models
|-- mechanics/              Depends on dfn, voxel, models
|-- simulation_export/      Depends on dfn, voxel, fragmentation, mechanics, models
|
|-- persistence/            Depends on models, utils (reads/writes files)
|-- services/               Depends on models, persistence, ALL scientific modules
|
|-- visualization/          Depends on models, pyvista
|-- workers/                Depends on services, models
|-- ui/                     Depends on visualization, workers, services, models, utils
|
|-- app/                    Depends on ui, core, services, persistence
```

### Text Dependency Graph

```
                         +------------+
                         |    utils   |
                         +-----+------+
                               |
                               v
                         +-----+------+
                         |   models   |
                         +-----+------+
                               |
                +--------------+---------------+
                |              |               |
                v              v               v
          +-----------+ +-----------+ +--------------+
          | geometry  | | geology   | | borehole     |
          +-----+-----+ +-----+-----+ +------+-------+
                |              |               |
                v              |               |
          +-----------+       |               |
          |    dfn    |       |               |
          +-----+-----+       |               |
                |              |               |
        +-------+-------+     |               |
        v               v     v               v
  +-----------+ +--------------+ +-----------------+
  |   voxel   | | connectivity | | fragmentation   |
  +-----+-----+ +------+-------+ +--------+--------+
        |               |                  |
        +-------+-------+                  |
                v                          |
          +-----------+                    |
          | mechanics |                    |
          +-----+-----+                    |
                |                          |
                +-------------+------------+
                              |
                              v
                   +--------------------+
                   | simulation_export  |
                   +---------+----------+
                             |
                             v
                   +--------------------+
                   |    persistence     |
                   +---------+----------+
                             |
                             v
                   +--------------------+
                   |      services      |
                   +---------+----------+
                             |
                  +----------+----------+
                  |          |          |
                  v          v          v
          +-----------+ +--------+ +---------+
          |visualizatn| |workers | |   ui    |
          +-----------+ +--------+ +----+----+
                                        |
                                        v
                                  +-----------+
                                  |    app    |
                                  +-----------+
```

---

## 3. Data Flow Diagrams

### 3.1 DFN Generation

```
 User triggers "Generate DFN"
           |
           v
 [UI] DFNGenerationDialog
   |-- collects: domain box, orientation distribution (Fisher/Bingham),
   |             size distribution (power-law/lognormal),
   |             intensity (P32), truncation rules
   |
   v
 [Worker] DFNGenerateWorker (runs in thread pool)
   |-- emits progress signal (fractures placed / total)
   |
   v
 [Service] DFNService.generate_dfn(params)
   |-- calls dfn/ generator
   |
   v
 [Scientific] dfn.generator.DFNGenerator
   |-- 1. Sample orientations from distribution
   |-- 2. Sample sizes from distribution
   |-- 3. Place centers (Poisson process / enhanced Baecher)
   |-- 4. Apply truncation (domain boundary, borehole conditioning)
   |-- 5. Compute intersection with domain boundary (clip discs/polygons)
   |
   v
 [Model] DFNModel (list of Fracture objects)
   |
   v
 [Worker] signals completion
   |
   v
 [UI] MainWindow receives DFNModel, adds to project tree,
      triggers visualization refresh
```

### 3.2 Voxelization

```
 User selects DFN + domain box, triggers "Voxelize"
           |
           v
 [Worker] VoxelizeWorker
   |-- validates input DFN is non-empty
   |-- emits progress (z-slices completed)
   |
   v
 [Service] VoxelService.build_grid(dfn, grid_spec)
   |
   v
 [Scientific] voxel.grid.Voxelizer
   |-- 1. Allocate VoxelGrid (origin, spacing, dims)
   |-- 2. For each fracture, compute intersection with grid cells
   |       (AABB test -> triangle/plane intersection -> cell marking)
   |-- 3. Compute cell properties:
   |       - fracture count per cell (P32 estimator)
   |       - mean orientation per cell
   |       - fracture intensity tensor
   |       - equivalent permeability tensor (Oda's method)
   |-- 4. Return populated VoxelGrid
   |
   v
 [Model] VoxelGrid (ndarray properties: frac_count, k_tensor, etc.)
   |-- optionally persisted to HDF5/Zarr sidecar
   |
   v
 [UI] Visualization updated with voxel rendering (volume / slices)
```

### 3.3 Connectivity Analysis

```
 User selects DFN, triggers "Analyze Connectivity"
           |
           v
 [Worker] ConnectivityWorker
   |
   v
 [Service] ConnectivityService.analyze(dfn, domain_box)
   |
   v
 [Scientific] connectivity.graph.ConnectivityAnalyzer
   |-- 1. Build fracture intersection matrix:
   |       - For each fracture pair, test disc-disc or polygon-polygon intersection
   |       - Record intersection segment (line segment in 3D)
   |-- 2. Construct undirected graph:
   |       - Nodes = fractures
   |       - Edges = intersections (weight = intersection length)
   |-- 3. Compute metrics:
   |       - Connected components (union-find)
   |       - Largest component size / fraction
   |       - Percolation: does largest component span domain in X/Y/Z?
   |       - Degree distribution
   |       - Clustering coefficient, average path length
   |       - Betweenness centrality
   |-- 4. Return ConnectivityGraph
   |
   v
 [Model] ConnectivityGraph (networkx-compatible graph + metrics dict)
   |
   v
 [UI] Results panel shows metrics table, graph panel colours by component
```

### 3.4 Block Fragmentation

```
 User selects DFN + tunnel/drawpoint geometry, triggers "Fragment"
           |
           v
 [Worker] FragmentationWorker
   |-- emits progress (total blocks traced)
   |
   v
 [Service] FragmentationService.fragment(dfn, tunnels, domain)
   |
   v
 [Scientific] fragmentation.block_tracer.BlockTracer
   |
   |-- 1. Pre-processing:
   |       - Merge coplanar/colinear fractures within tolerance
   |       - Extend fractures to domain boundaries
   |       - Build fracture intersection graph (face adjacency)
   |
   |-- 2. Block tracing (GMG algorithm):
   |       - Start from each tunnel face triangle
   |       - Traverse fracture planes to close polyhedra
   |       - Validate block closure and non-degeneracy
   |
   |-- 3. Block classification:
   |       - Shape metrics: volume, surface area, aspect ratio
   |       - Stability: removable? tapered? (key-block theory)
   |       - Drawpoint connectivity: is block reachable?
   |
   |-- 4. Post-processing:
   |       - Merge very small blocks into neighbours (threshold)
   |       - Assign mechanical properties
   |
   v
 [Model] FragmentationResult (list of PolyhedralBlock, statistics)
   |
   v
 [UI] Fragment size distribution plot, 3D block rendering,
      CSV export of block properties
```

### 3.5 Model Export

```
 User selects model + target format, triggers "Export"
           |
           v
 [Worker] ExportWorker
   |
   v
 [Service] SimulationExportService.export(model, format, options)
   |
   v
 [Scientific] simulation_export/ (per-format writers)
   |
   |-- 3DEC exporter:
   |     - Convert fracture polygons to joints (face commands)
   |     - Convert blocks to polyhedral zones
   |     - Write block properties (density, modulus, strength)
   |     - Write 3DEC command (.dat) file
   |
   |-- FLAC3D exporter:
   |     - Map voxel grid to FLAC3D zones
   |     - Write zone properties (DFN-derived tensors)
   |     - Write .f3grid or .f3dat file
   |
   |-- VTK exporter:
   |     - Fracture set -> vtkPolyData (.vtp)
   |     - Voxel grid -> vtkImageData / vtkUnstructuredGrid (.vtu)
   |     - Blocks -> vtkPolyData (.vtp)
   |
   |-- CSV exporter:
   |     - Tabular export of fracture/block/voxel properties
   |
   v
 [File I/O] persistence writers
   |
   v
 Output files written to disk
```

---

## 4. Qt Adapter Pattern (PySide6 / PyQt6 Compatibility)

The project targets **PySide6** as its primary Qt binding but maintains compatibility
with **PyQt6** through an adapter layer. No module outside `ui/` imports from `PySide6` or `PyQt6` directly.

### Adapter Module

File: `src/dfn_cave_studio/ui/qt_adapter.py`

```python
"""Qt binding abstraction layer.

All imports of PySide6 / PyQt6 are centralized here.  Other modules
import from this adapter rather than directly from either binding.
"""

import os

# Late-binding flag -- set before any Qt class is instantiated.
_QT_BACKEND: str = os.environ.get("DFN_QT_BACKEND", "pyside6")

if _QT_BACKEND == "pyside6":
    from PySide6.QtCore import *
    from PySide6.QtGui import *
    from PySide6.QtWidgets import *
    from PySide6.QtUiTools import QUiLoader
elif _QT_BACKEND == "pyqt6":
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *
    from PyQt6.uic import loadUi as _loadUi
else:
    raise ImportError(f"Unsupported Qt backend: {_QT_BACKEND}")

# Re-export with canonical names to hide binding-specific differences.
Signal = Signal
Slot = Slot
QObject = QObject
```

### Usage Convention

Every UI module imports from `ui.qt_adapter`:

```python
from dfn_cave_studio.ui.qt_adapter import (
    QWidget, QVBoxLayout, QPushButton, Signal, Slot, Qt,
)
```

### Shims for Known Differences

Where PySide6 and PyQt6 APIs diverge, the adapter provides thin wrappers:

| Area              | PySide6                          | PyQt6                          | Adapter Shim                      |
|-------------------|----------------------------------|--------------------------------|-----------------------------------|
| `QFileDialog` enum| `QFileDialog.Option.xxx`         | `QFileDialog.Option.xxx`       | (identical -- no shim needed)     |
| `uic` loading     | `PySide6.QtUiTools.QUiLoader`    | `PyQt6.uic.loadUi`             | `load_ui(path, parent)` function  |
| `qApp`            | `QApplication.instance()`        | `QApplication.instance()`      | `get_qapp()` helper               |
| `exec()` keyword  | method is `exec()` (deprecated)  | method is `exec()` (deprecated)| use `exec_()` on QDialog          |
| `long` params     | accepts `int` (Python 3)         | accepts `int` (Python 3)       | `int` everywhere                  |

---

## 5. Threading Model

### Thread Architecture

```
+-------------------------------------------------------+
|                    Main thread (Qt)                     |
|  - Event loop                                          |
|  - All QWidget instantiation and method calls           |
|  - Signal/slot connections                             |
|  - Plotting and data-slice access (lightweight)         |
+---------------------------+---------------------------+
                            |
              queued connections (thread-safe)
                            |
          +-----------------+-----------------+
          |                                   |
          v                                   v
+---------------------+          +---------------------+
|  QThreadPool        |          |  QThreadPool        |
|  (global, up to     |          |  (secondary, up to  |
|   N-1 workers)      |          |   2 workers)        |
|                     |          |                     |
|  - DFN generation   |          |  - File I/O         |
|  - Voxelization     |          |  - Export writing   |
|  - Connectivity     |          |  - Import parsing   |
|  - Fragmentation    |          |                     |
|  - Geometry ops     |          |                     |
+---------------------+          +---------------------+
```

### Worker Design

Each worker follows the same pattern:

```python
class BaseWorker(QObject):
    """Signal proxy held by the main thread; the QRunnable uses it to emit."""
    finished = Signal(object)       # result payload
    error = Signal(str, str)        # short message, traceback
    progress = Signal(int, int)     # current, total
    partial_result = Signal(object) # intermediate result for streaming

class ComputeTask(QRunnable):
    def __init__(self, worker_proxy, *args):
        super().__init__()
        self._proxy = worker_proxy
        self._args = args

    def run(self):
        try:
            result = heavy_computation(*self._args, progress_cb=self._proxy.progress.emit)
            self._proxy.finished.emit(result)
        except Exception as exc:
            self._proxy.error.emit(str(exc), traceback.format_exc())
```

### Thread Safety Rules

1. **All QWidget mutations MUST happen on the main thread.** Workers emit signals; slots connected with `Qt.QueuedConnection` (the default for cross-thread) handle the mutation safely.
2. **Scientific modules are stateless and thread-safe.** They accept immutable input objects and return new outputs.
3. **File I/O workers use a separate, smaller thread pool** to avoid starving compute workers during large exports.
4. **Cancellation** is cooperative: workers periodically check a `threading.Event` and return early if set.
5. **Numpy arrays crossing thread boundaries** are passed by reference (shared memory) but ownership is transferred (the worker does not mutate the array after emitting `finished`).

---

## 6. Project File Format Specification

### 6.1 Overview

A DFN Cave Studio project is a directory containing one JSON manifest and optional
sidecar files for large numeric data.

```
my_project.dfnproj/                # directory (or .zip archive)
|-- manifest.json                  # project metadata, references, UI state
|-- arrays/
|   |-- voxel_grid_001.h5          # HDF5 file for a VoxelGrid
|   |-- voxel_grid_001.zarr/       # OR Zarr directory (alternative)
|   |-- dfn_properties_002.h5
|-- exports/                       # last-exported files (optional, not versioned)
|-- thumbnails/                    # PNG snapshots for project browser
|-- history.json                   # undo/redo stack (optional, ephemeral)
```

### 6.2 manifest.json Specification

```jsonc
{
  "format_version": "1.0.0",
  "app_version": "0.5.0",
  "created": "2026-08-04T10:30:00Z",
  "modified": "2026-08-04T14:22:00Z",
  "project": {
    "name": "Cadia East Panel 2",
    "description": "...",
    "units": "metric",            // "metric" | "imperial"
    "coordinate_system": {        // optional CRS
      "epsg": 32755,
      "origin_easting": 650000.0,
      "origin_northing": 6300000.0,
      "origin_elevation": 500.0
    }
  },
  "domain_models": [
    {
      "id": "domain-001",
      "type": "GeologicalDomain",
      "name": "Orebody",
      "data_file": null,              // small data inlined
      "geometry": { /* serialized surface/volume */ }
    }
  ],
  "dfn_models": [
    {
      "id": "dfn-001",
      "name": "Stage 1 DFN",
      "fractures_count": 125000,
      "arrays_file": "arrays/dfn-001.h5",
      "arrays_key": "/fractures",
      "stats": { "p32": 2.3, "p10": 4.1, "mean_radius": 5.2 }
    }
  ],
  "voxel_grids": [
    {
      "id": "voxel-001",
      "name": "Voxel 10m",
      "arrays_file": "arrays/voxel-001.h5",
      "grid_spec": {
        "origin": [650000.0, 6300000.0, 500.0],
        "spacing": [10.0, 10.0, 10.0],
        "dims": [80, 120, 60]
      }
    }
  ],
  "connectivity_graphs": [
    {
      "id": "conn-001",
      "name": "Connectivity Analysis",
      "dfn_id": "dfn-001",
      "graph_data": { /* inlined JSON -- nodes, edges, metrics */ }
    }
  ],
  "fragmentation_results": [
    {
      "id": "frag-001",
      "name": "Frag Run 1",
      "dfn_id": "dfn-001",
      "tunnel_geometry_id": "domain-002",
      "arrays_file": "arrays/frag-001.h5",
      "stats": { "total_blocks": 34520, "mean_volume_m3": 12.5 }
    }
  ],
  "ui_state": {
    "camera": { "position": [...], "focal_point": [...], "view_up": [...] },
    "visible_items": ["dfn-001", "voxel-001"],
    "dock_layout": "<base64 encoded QByteArray>",
    "plots": {
      "stereonet": { "show_contours": true, "grid_spacing": 5.0 }
    }
  }
}
```

### 6.3 HDF5 / Zarr Conventions

For data sets expected to be large (>1000 elements), arrays are stored in HDF5 or Zarr
sidecars rather than inlined in JSON.

**HDF5 layout example** (for a DFN model):

```
/                          # root group
/fractures/                # group
/fractures/center          # shape (N, 3) float64
/fractures/normal          # shape (N, 3) float64
/fractures/radius          # shape (N,)  float64
/fractures/aperture        # shape (N,)  float64
/fractures/set_id          # shape (N,)  int32
/dfn_params/               # group -- generation parameters (attrs)
/dfn_params/.attrs/p32     # scalar
/dfn_params/.attrs/seed    # scalar
```

**Zarr is preferred** when:
- Concurrent read/write from multiple processes is needed (e.g., parallel voxelization).
- Cloud-backed storage (S3, GCS) is required in the future.

**HDF5 is preferred** when:
- Single-file portability matters.
- Compression and chunked access from a single process suffice.

The `persistence/array_store.py` module provides a unified interface:

```python
class ArrayStore(ABC):
    """Abstract interface for bulk array storage."""
    @abstractmethod
    def write_array(self, path: str, array: np.ndarray, **meta) -> None: ...
    @abstractmethod
    def read_array(self, path: str) -> np.ndarray: ...
    @abstractmethod
    def keys(self) -> list[str]: ...

class HDF5Store(ArrayStore): ...
class ZarrStore(ArrayStore): ...
class MemoryStore(ArrayStore): ...  # for testing
```

### 6.4 Version Migration

When `format_version` in `manifest.json` differs from the current application version,
a migration pipeline runs at project load time:

```
ManifestLoader
  -> detect version mismatch
  -> lookup migration chain in migration_registry
  -> apply each migration in order (1.0 -> 1.1 -> 1.2 -> ...)
  -> write updated manifest (or ask user before overwriting)
```

Migrations are pure functions:
```python
def migrate_1_0_to_1_1(manifest: dict) -> dict: ...
```

---

## 7. Error Handling Strategy

### 7.1 Principles

1. **Fail fast on programmer errors.** Assertions, type mismatches, and violated
   invariants raise exceptions immediately during development and testing.
2. **Fail gracefully on user/data errors.** Invalid input files, out-of-memory
   conditions, and numerical failures are caught, reported to the user via the UI,
   and logged. The application continues running.
3. **Never lose the user's data.** The project auto-save mechanism runs on a timer
   and before any potentially destructive operation.

### 7.2 Exception Hierarchy

```
DFNCaveStudioError               # base for all app exceptions
|-- ModelValidationError         # pydantic validation failure
|-- ComputationError             # algorithmic failure (e.g., mesh non-manifold)
|   |-- ConnectivityError
|   |-- FragmentationError
|   |-- VoxelizationError
|-- IOFIleError                  # file not found, permission denied, corrupt
|   |-- ProjectLoadError
|   |-- ProjectSaveError
|   |-- ExportError
|-- CancellationError            # user cancelled a background task
|-- ConfigurationError           # invalid config file / environment
```

### 7.3 Error Propagation Path

```
Scientific layer:    raises ComputationError
       |
       v
Service layer:       catches, wraps, logs, optionally retries
       |
       v
Worker layer:        catches all exceptions, emits error signal
       |
       v
UI layer:            slot displays QMessageBox with user-friendly text
                     and detailed stack trace in expandable section
```

### 7.4 Context-Rich Errors

Errors carry structured context rather than bare strings:

```python
class ComputationError(DFNCaveStudioError):
    def __init__(self, message: str, *, module: str, operation: str, details: dict = None):
        super().__init__(message)
        self.module = module
        self.operation = operation
        self.details = details or {}
```

This allows the UI to render targeted recovery actions (e.g., "Try reducing the DFN
density and re-run" after an `OutOfMemoryError`-like condition).

### 7.5 Defensive Checks

- All public service methods validate inputs with Pydantic models before dispatching
  to the scientific layer.
- File I/O operations verify checksums (SHA-256) stored in HDF5 attributes.
- Numerical results are checked for NaN/Inf before being returned to the UI.

---

## 8. Configuration Management

### 8.1 Configuration Layers (Cascading Priority)

```
1. Environment variable          DFN_QT_BACKEND=pyside6
2. User config file              ~/.dfn_cave_studio/config.yaml
3. Project-local config          <project_dir>/.dfnconfig.yaml
4. Built-in defaults             src/dfn_cave_studio/core/defaults.yaml
```

Lower-numbered layers override higher-numbered ones.

### 8.2 Configuration Schema (Pydantic `BaseSettings`)

```python
class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DFN_",
        yaml_file="~/.dfn_cave_studio/config.yaml",
    )

    # General
    language: str = "en"
    units_default: Literal["metric", "imperial"] = "metric"

    # Compute
    num_worker_threads: int = Field(default_factory=lambda: max(1, os.cpu_count() - 1))
    memory_limit_gb: float = 0.0  # 0 = auto-detect
    double_precision: bool = True
    default_random_seed: int = 42

    # Visualization
    render_backend: Literal["auto", "vtk", "pyvista"] = "auto"
    anti_aliasing: bool = True
    point_size: float = 3.0
    background_color: str = "#1e1e1e"

    # Persistence
    array_format: Literal["hdf5", "zarr"] = "hdf5"
    auto_save_interval_minutes: int = 5
    max_recent_projects: int = 10

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Path | None = None  # None = console only

    # Advanced
    debug_mode: bool = False
    telemetry_enabled: bool = True
```

### 8.3 Accessing Config

The `core/config.py` module provides a singleton:

```python
from dfn_cave_studio.core.config import get_config

cfg = get_config()
print(cfg.num_worker_threads)
```

For tests, a context manager temporarily overrides values:

```python
with config_override(num_worker_threads=1, debug_mode=True):
    run_test()
```

---

## 9. Logging Strategy

### 9.1 Logger Hierarchy

```
dfn_cave_studio                    # root logger
|-- dfn_cave_studio.app            # application lifecycle
|-- dfn_cave_studio.ui             # UI events, widget lifecycle
|-- dfn_cave_studio.dfn            # DFN generation details
|-- dfn_cave_studio.voxel          # voxelization progress
|-- dfn_cave_studio.connectivity   # graph construction steps
|-- dfn_cave_studio.fragmentation  # block tracing details
|-- dfn_cave_studio.mechanics      # property assignment
|-- dfn_cave_studio.export         # export operations
|-- dfn_cave_studio.persistence    # file I/O, project load/save
|-- dfn_cave_studio.workers        # task start/stop/cancel
|-- dfn_cave_studio.geometry       # geometric algorithm tracing (verbose)
```

### 9.2 Configuration

Controlled via `AppConfig.log_level` and `AppConfig.log_file`:

- **Console handler**: always active, coloured output, level from config.
- **File handler**: optional, rotating (5 MB x 3 backups), structured JSON for machine parsing.
- **Signal handler**: a custom `logging.Handler` that emits a Qt signal so the UI can display log messages in a dockable Log Panel (filterable by logger name and level).

### 9.3 Structured Logging

All log records in the file handler use JSON formatting for downstream analysis:

```json
{"timestamp": "2026-08-04T14:22:01.123Z", "level": "INFO", "logger": "dfn_cave_studio.dfn",
 "message": "DFN generation complete", "fractures": 125000, "elapsed_s": 12.4,
 "module": "dfn.generator", "function": "generate"}
```

### 9.4 Sensitive-Data Policy

- File paths and project names are logged.
- Fracture coordinates, borehole data, and assay values are **never** logged at INFO
  level and above. They may appear at DEBUG level and only when `debug_mode` is enabled.

---

## 10. Extension Points and Plugin Architecture

### 10.1 Plugin Discovery

Plugins are Python packages installed in the environment. DFN Cave Studio discovers
them through setuptools entry points:

```toml
# pyproject.toml of a third-party plugin
[project.entry-points."dfn_cave_studio.plugins"]
my_plugin = "my_package.plugin:MyPlugin"
```

The plugin registry scans entry points at startup:

```python
# core/plugin_registry.py
class PluginRegistry:
    def discover(self) -> list[Plugin]:
        eps = importlib.metadata.entry_points(group="dfn_cave_studio.plugins")
        return [ep.load()() for ep in eps]
```

### 10.2 Extension Points

#### A. Scientific Algorithm Extensions

Plugins can register alternative implementations of core algorithms:

```python
class DFNGeneratorPlugin(ABC):
    @abstractmethod
    def generate(self, params: DFNGenerationParams) -> DFNModel: ...

# In the plugin package:
class MyCustomGenerator(DFNGeneratorPlugin):
    name = "Custom Fracture Generator"
    def generate(self, params): ...

# Registration via entry point.
```

#### B. Export Format Extensions

```python
class SimulationExporterPlugin(ABC):
    """Add a new export target format."""
    format_id: str          # e.g., "openfoam"
    format_label: str       # e.g., "OpenFOAM mesh"
    file_extension: str     # e.g., ".foam"
    filter_string: str      # e.g., "OpenFOAM files (*.foam)"

    @abstractmethod
    def export(self, model: ExportableModel, path: Path, options: dict) -> None: ...
```

#### C. Import Format Extensions

```python
class ImporterPlugin(ABC):
    """Add support for importing external data formats."""
    format_id: str
    format_label: str
    file_extensions: list[str]

    @abstractmethod
    def can_read(self, path: Path) -> bool: ...
    @abstractmethod
    def read(self, path: Path, options: dict) -> ImportedData: ...
```

#### D. Visualization Plugin Extensions

```python
class VisualizationPlugin(ABC):
    """Add custom 3D actors or overlay renderers."""
    def render(self, plotter: pv.Plotter, model: Any, options: dict) -> None: ...
    def toolbar_actions(self) -> list[QAction]: ...
```

#### E. UI Panel Extensions

```python
class UIPanelPlugin(ABC):
    """Register a custom dockable panel widget."""
    panel_id: str
    panel_title: str
    default_area: str       # "left" | "right" | "bottom"

    def create_widget(self, parent: QWidget, main_window) -> QWidget: ...
```

### 10.3 Plugin Lifecycle

```
Application start
  -> PluginRegistry.discover()
  -> for each plugin:
       plugin.on_load()
  -> Plugins appear in menus, toolbars, and export dialogs

Application shutdown
  -> for each plugin:
       plugin.on_unload()
```

### 10.4 Plugin Isolation

- Plugins run in the same process as the main application (no sandboxing).
- Plugin authors are trusted; they have access to the full Python environment.
- A plugin manifest declares minimum DFN Cave Studio version, dependencies, and an
  optional "safe mode" flag. Safe-mode plugins are guaranteed not to use `subprocess`,
  `socket`, or file-system write outside the project directory (enforced via a static
  analysis lint check in the plugin submission pipeline, not enforced at runtime).

### 10.5 Built-in Extension Modules

The following are implemented as internal plugins (same API as external) and shipped
with the application:

| Plugin                     | Extension Point        | Description                               |
|----------------------------|------------------------|-------------------------------------------|
| `enhanced_baecher`         | DFN Generator          | Enhanced Baecher disc model               |
| `fracman_compat`           | DFN Generator          | FracMan-compatible generator              |
| `gmg_block_tracer`         | Fragmentation          | GeneralBlock / GMG block tracing           |
| `oda_upscaling`            | Mechanics              | Oda's method for equivalent continuum      |
| `export_3dec`              | Simulation Exporter    | 3DEC joint/block export                   |
| `export_flac3d`            | Simulation Exporter    | FLAC3D zone export                        |
| `export_vtk`               | Simulation Exporter    | VTK/VTP/VTU export                        |
| `export_csv`               | Simulation Exporter    | CSV tabular export                        |
| `import_dxf`               | Importer              | DXF polyline/surface import               |
| `import_deswik`            | Importer              | Deswik mine planning import               |
| `import_leapfrog`          | Importer              | Leapfrog geological model import          |
| `stereonet_panel`          | UI Panel              | Stereonet plotting panel                  |
| `borehole_log_panel`       | UI Panel              | Borehole log viewer                       |

---

## Appendix A: Key Technical Decisions

| Decision                     | Rationale                                                                 |
|------------------------------|---------------------------------------------------------------------------|
| Pydantic for models          | Validation, serialisation, JSON Schema, IDE autocomplete                   |
| PyVista (not raw VTK)        | Higher-level API, rapid prototyping, good documentation                    |
| HDF5 + Zarr for big arrays   | Chunked, compressible, ubiquitous in scientific Python                     |
| `QRunnable` over `QThread`   | Better thread-pool management, lighter weight, built-in auto-delete         |
| Plugin entry points          | Standard Python mechanism, no custom registry file needed                  |
| Pydantic `BaseSettings`      | TOML/YAML/env cascade with validation out of the box                       |
| Structured JSON logging      | Machine-parseable, integrates with ELK/Loki/datadog if needed in future    |

## Appendix B: Performance Targets

| Operation                | Model Size      | Target Wall Time   | Notes                                    |
|--------------------------|-----------------|---------------------|------------------------------------------|
| DFN generation           | 100k fractures  | < 30 s              | Single-threaded, uniform distribution    |
| Voxelization             | 1M cells        | < 60 s              | 8-thread parallel                        |
| Connectivity             | 50k fractures   | < 120 s             | O(n^2) intersection test; R-tree indexed |
| Fragmentation            | 100k fractures   | < 300 s             | Block tracing is the bottleneck          |
| Project save (full)      | 500k fractures   | < 10 s              | HDF5 with compression                    |
| Project load (full)      | 500k fractures   | < 15 s              | Lazy-load arrays on access               |
| 3D render (interactive)  | 500k discs       | 30+ FPS             | LOD-based point-cloud fallback           |

---

*Document version: 1.0 -- 2026-08-04*
