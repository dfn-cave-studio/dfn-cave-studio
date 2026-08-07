# DFN Cave Studio

**Discrete Fracture Network modeling and analysis for underground block cave mining research.**

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Milestone](https://img.shields.io/badge/Milestone-M7-orange.svg)](ROADMAP.md)

## Quick Start (M7)

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m dfn_cave_studio.app
```

### M7 Data Pipeline Workflow

1. **Import** — Open M7 Import Wizard, load `examples/m7_demo/` CSV files
2. **Clean** — Open Cleaning dialog, review issues, exclude bad rows, accept auto-fixes
3. **Holdout** — Split boreholes into calibration / validation (random, seed=42, 25%)
4. **Domains** — Create structural domains, assign borehole depth intervals
5. **Joint Sets** — Identify joint sets from imported set_id or auto-cluster
6. **Save** — Save as `.dfnproj`, reopen to continue

### Overview

DFN Cave Studio is a desktop scientific software for:

- **DFN Modeling** — Stochastic and deterministic discrete fracture network generation
- **Voxelization** — Spatial discretization with sparse/chunked storage
- **Connectivity Analysis** — Fracture network graph construction and percolation analysis
- **Fragmentation Analysis** — In-situ block size distribution estimation
- **Numerical Export** — Model export for 3DEC, FLAC3D, and other solvers

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m dfn_cave_studio
```

## Quick Start

See [docs/USER_MANUAL.md](docs/USER_MANUAL.md) for detailed instructions.

```python
from dfn_cave_studio import create_app
app = create_app()
app.run()
```

## Project Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| M0 | 🟡 In Progress | Repository, environment, architecture, minimal Qt window |
| M1 | ⬜ Planned | Geometry, coordinate, fracture data models |
| M2 | ⬜ Planned | Global constant-parameter stochastic DFN |
| M3 | ⬜ Planned | Voxel space, surface mask, sparse storage |
| M4 | ⬜ Planned | DFN-voxel intersection |
| M5 | ⬜ Planned | Connectivity graph and percolation |
| M6 | ⬜ Planned | Deterministic structures, borehole import |
| M7 | ⬜ Planned | Virtual borehole P10/P21 validation |
| M8 | ⬜ Planned | Block fragmentation analysis |
| M9 | ⬜ Planned | Structural domains |
| M10 | ⬜ Planned | Continuous DFN parameter fields |
| M11 | ⬜ Planned | Mechanical properties and export |
| M12 | ⬜ Planned | Complete project management, packaging |

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Software architecture
- [SCIENTIFIC_SPEC.md](SCIENTIFIC_SPEC.md) — Scientific methods and algorithms
- [AGENTS.md](AGENTS.md) — Developer conventions and guidelines
- [ROADMAP.md](ROADMAP.md) — Development roadmap
- [RISK_REGISTER.md](RISK_REGISTER.md) — Known risks and mitigations
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — Third-party licenses

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src/dfn_cave_studio

# Run GUI tests only
pytest tests/gui/

# Run scientific validation
pytest tests/scientific/
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## References

See [SCIENTIFIC_SPEC.md](SCIENTIFIC_SPEC.md) for scientific references.
