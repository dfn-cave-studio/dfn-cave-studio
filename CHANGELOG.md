# Changelog

All notable changes to DFN Cave Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0-M0] — 2026-08-04

### Added (Milestone M0: Foundation)
- Project repository initialized with full directory structure (17 source packages)
- `AGENTS.md` — developer conventions, coordinate/unit standards, architecture rules
- `README.md` — project overview, installation, milestone status
- `PROJECT_PLAN.md` — detailed 13-milestone plan with task breakdowns (M0-M12)
- `ARCHITECTURE.md` — 7-layer architecture, data flow diagrams, threading model
- `SCIENTIFIC_SPEC.md` — complete scientific method documentation with 35 references
- `ROADMAP.md` — development roadmap with dependency graph
- `ACCEPTANCE_TESTS.md` — 48 acceptance tests across 4 categories
- `RISK_REGISTER.md` — 8 identified risks with mitigations
- `THIRD_PARTY_NOTICES.md` — dependency licenses and compliance verification
- `LICENSE` — MIT license
- `requirements.txt` — pinned dependency versions (Python 3.14.5, Windows 11 x64)
- `pyproject.toml` — project metadata and tool configuration
- `.gitignore` — comprehensive ignore rules
- `conftest.py` — shared pytest fixtures (qapp, main_window, rng, config)
- `.github/workflows/milestone-review.yml` — CI: test, coverage, review package, release

### Source Code
- **`core/config.py`**: AppConfig, AppVersion, UnitConfig, RandomConfig, UIConfig, PerformanceConfig, LoggingConfig — versioned pydantic models
- **`core/logger.py`**: Structured logging with console + file handlers
- **`ui/qt_adapter.py`**: PySide6 abstraction layer (QtCore, QtGui, QtWidgets re-exports) for future PyQt6 migration
- **`ui/main_window.py`**: Modern scientific desktop UI (menus: File/Data/Voxel/DFN/Domains/Analysis/Visualization/Export/Tools/Help; docks: Project Explorer, Properties, Log; toolbar; status bar with progress)
- **`models/bounds.py`**: ModelBounds with validation, VoxelConfig with memory estimation and danger warnings
- **`models/enums.py`**: FractureType, SizeDistributionType, ExportFormat, ProjectStatus, etc.
- **Resources**: Professional light theme stylesheet (`resources/styles/app.qss`)

### Tests (26 passing)
- **Unit (14)**: AppVersion, UnitConfig, RandomConfig, AppConfig, all enums
- **GUI (12)**: Window creation, menus, docks, callbacks, status/progress, project tree

### Build Infrastructure
- `scripts/build_review_package.py` — milestone review package generator
- GitHub Actions CI workflow (test + review package + release)

### Environment
- Python 3.14.5, Windows 11 Pro x64, 32GB RAM, 663GB free disk
- PySide6 6.11.1, PyVista 0.48.4, VTK 9.6.2
- NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.3
- trimesh 4.12.2, NetworkX 3.6.1, Matplotlib 3.10.9
- h5py 3.16.0, zarr 3.3.0, pydantic 2.13.4
- pytest 9.1.1, pytest-qt 4.5.0

## [0.2.0-M1] — 2026-08-04

### Added (Milestone M1: Data Models & Project Management)
- **Borehole models**: Collar, SurveyStation, BoreholeSurvey, FractureObservation, RQDInterval, Borehole, BoreholeCollection
- **BoreholeSurvey**: Minimum curvature trajectory computation
- **BoreholeCollection**: CSV import, duplicate detection, angle validation, quality report
- **StructuralDomain**: Global, box, polygon, fault_buffer boundary types with spatial point query
- **StructuralDomainCollection**: Prioritized domain lookup with global fallback
- **RockMask**: Box and surface-based masks with point/AABB containment testing
- **ExcavationMask**: Void region masks for excavations and caves
- **SurfaceModel**: Triangulated surface with barycentric elevation_at() and above_surface()
- **SpatialAttributeConfig**: Per-voxel attribute flags (active_mask, material_id, domain_id, etc.)
- **FractureMechanicalProperties**: Cohesion, friction, stiffness, dilation, residual strength
- **MechanicalPropertyTemplate**: Multi-criteria matching (set, domain, type, filling)
- **MechanicalPropertyLibrary**: Priority-based template resolution
- **Project model**: Unified container for all project data with JSON serialization
- **Project schema versioning**: Forward/backward compatible migration
- **ProjectStore**: Save/load/auto-save with atomic writes and backup
- **RecentProjectsManager**: Persistent recent files (max 20)
- **MainWindow**: Project CRUD (new, open, save, save-as) with unsaved-changes tracking
- **Project tree**: Dynamic display of project structure
- **Sample data**: 5 borehole collars, 14 fracture observations, 1 sample project
- **63 new tests** (200 total, all passing)
