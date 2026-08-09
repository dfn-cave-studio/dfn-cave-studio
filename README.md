# DFN Cave Studio

**Discrete Fracture Network modeling and analysis for underground block cave mining research.**

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Milestone](https://img.shields.io/badge/Milestone-M8-orange.svg)](ROADMAP.md)

## Quick Start (M8)

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m dfn_cave_studio.app
```

### M8 Workflow

1. **Borehole Database** — Independently preview/map/import collars, surveys, fractures, RQD, and domain intervals.
2. **Data Quality** — Review Raw, Formal, Excluded, and Pending records and traceable reasons.
3. **Holdout** — Split boreholes into calibration / validation.
4. **Domains and Joint Sets** — Maintain borehole intervals and identify sets from calibration fractures.
5. **Model Boundary** — Calculate or enter the Voxel Analysis Domain and check complete trajectories.
6. **Voxel Preview** — Confirm anisotropic resolution, memory estimate, and buffered DFN Generation Domain.
7. **Save** — Save as `.dfnproj`; all database states and spatial settings restore on reopen.

### Overview

DFN Cave Studio is a desktop scientific software for:

- **DFN Modeling** — Stochastic and deterministic discrete fracture network generation
- **Voxelization** — Spatial discretization with sparse/chunked storage
- **Connectivity Analysis** — Fracture network graph construction and percolation analysis
- **Fragmentation Analysis** — Planned for the integrated v1.0.0-M12 release
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
| M0–M7 | ✅ Published | Existing historical milestones through `v0.7.0-M7` |
| M8 | 🚧 In development | Borehole database, spatial boundaries, and voxel-grid definition |
| M9 | ⬜ Planned | Local DFN parameter fields and first voxelization |
| M10 | ⬜ Planned | Conditional explicit DFN and second voxelization |
| M11 | ⬜ Planned | Independent validation, external simulation, and ML export |
| M12 | ⬜ Planned | Fragmentation, integration, performance, and Windows release |

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
