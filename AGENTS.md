# AGENTS.md — Developer Conventions and Guidelines

## Coordinate and Unit Conventions

### Coordinate System (Right-handed)
- **X**: Easting (positive east)
- **Y**: Northing (positive north)
- **Z**: Elevation (positive up)

### Units (SI Internal)
| Quantity | Unit | Symbol |
|----------|------|--------|
| Length | meter | m |
| Area | square meter | m² |
| Volume | cubic meter | m³ |
| Angle (internal) | radian | rad |
| Angle (UI/IO) | degree | ° |
| P10 | per meter | m⁻¹ |
| P32 | square meter per cubic meter | m²/m³ ≡ m⁻¹ |
| Stress | Pascal | Pa |

### Angle Conventions
- **Dip direction**: 0°–360° (clockwise from north)
- **Dip**: 0°–90° (from horizontal)
- **Internal trig functions use radians** — convert at IO boundaries
- `math.radians()` / `math.degrees()` for conversion

### Random Number Policy
- ALL stochastic algorithms MUST accept `random_seed: int`
- Same seed + same input MUST produce identical output
- Use `numpy.random.Generator` with explicit seed, NOT `numpy.random` global state
- Store seed in project file for reproducibility

### Prohibited Practices
- ❌ Do NOT equate RQD directly with P10 or P32 without documented method
- ❌ Do NOT treat voxel boundaries as fracture boundaries
- ❌ Do NOT equate "adjacent voxels both have fractures" with network percolation
- ❌ Do NOT claim high P32 equals guaranteed cavability
- ❌ Do NOT use unseeded random algorithms
- ❌ Do NOT ignore units
- ❌ Do NOT ignore coordinate systems
- ❌ Do NOT fake validation results
- ❌ Do NOT lower test standards to pass tests
- ❌ Do NOT claim completion when tests fail
- ❌ Do NOT hide unimplemented features
- ❌ Do NOT copy external code without license verification
- ❌ Do NOT write scientific algorithms in Qt button callbacks
- ❌ Do NOT put all code in a single file

## Architecture Rules

### Layer Separation (Strict)
1. **Data Model Layer** (`models/`) — Pydantic/dataclass models
2. **Scientific Algorithm Layer** (`geometry/`, `dfn/`, `voxel/`, `connectivity/`, `fragmentation/`) — Pure computation, NO Qt imports
3. **File I/O Layer** (`persistence/`, `services/`) — File reading/writing
4. **3D Visualization Layer** (`visualization/`) — PyVista/VTK wrappers
5. **Qt UI Layer** (`ui/`) — PySide6 widgets only
6. **Background Task Layer** (`workers/`) — QRunnable tasks
7. **Project Save/Restore Layer** (`persistence/`) — Project serialization

### Import Rules
- `core/` and `models/` — zero Qt imports, zero VTK imports
- `geometry/` — only NumPy/SciPy
- `dfn/`, `voxel/`, `connectivity/`, `fragmentation/` — only NumPy/SciPy + models
- `visualization/` — may import PyVista/VTK
- `ui/` — may import PySide6 + visualization wrappers
- `workers/` — may import core + models

### Qt Adapter Layer
The project uses PySide6. If PyQt6 is needed in future:
- `ui/qt_adapter.py` provides a thin abstraction
- All Qt imports go through this adapter
- NEVER scatter `from PySide6 import ...` throughout the codebase

## Code Style

- Python 3.12+ type hints required on all public functions
- Google-style docstrings
- `black` formatting (120 char line length)
- `ruff` linting
- `pydantic` for configuration and data models
- `dataclasses` for lightweight internal structs

## Testing Requirements

- NO feature is complete without tests
- Unit tests: every public function
- Integration tests: cross-module workflows
- Scientific tests: known-result validation cases
- GUI tests: smoke tests for all windows
- Run `pytest` before every commit

## Git Conventions

- Branch: `milestone/M<N>/<description>`
- Commit: `<type>(<scope>): <description>`
- Tag: `v<major>.<minor>.<patch>-M<milestone>`
- NEVER force-push to main
- NEVER modify published tags

## Milestone Completion Checklist

- [ ] Code complete
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Scientific validation passes
- [ ] Documentation updated
- [ ] Review package generated
- [ ] GitHub Pages updated
- [ ] Release published
- [ ] REVIEW_MANIFEST consistent with actual output
- [ ] No hidden failures
