# DFN Cave Studio

**Discrete Fracture Network modeling and analysis for underground block cave mining research.**

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Milestone](https://img.shields.io/badge/Milestone-M11.1-brightgreen.svg)](ROADMAP.md)

## Quick Start (M11.1)

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m dfn_cave_studio.app
```

### M8–M11.1 Workflow

1. **Borehole Database** — Independently preview/map/import collars, surveys, fractures, RQD, and domain intervals.
2. **Data Quality** — Review Raw, Formal, Excluded, and Pending records and traceable reasons.
3. **Holdout** — Split boreholes into calibration / validation.
4. **Domains and Joint Sets** — Maintain borehole intervals and identify sets from calibration fractures.
5. **Model Boundary** — Calculate or enter the Voxel Analysis Domain and check complete trajectories.
6. **Voxel Preview** — Confirm anisotropic resolution, memory estimate, and buffered DFN Generation Domain.
7. **Save** — Save as `.dfnproj`; all database states and spatial settings restore on reopen.

8. **Fracture Density Model** — Compute fixed/domain P10 intervals and direction-corrected Poisson-MLE P32 from Calibration holes.
9. **Fracture Size Distribution** — Fit real size observations or enter a clearly labelled assumed/user model.
10. **First Voxel Parameter Field** — Build a domain-aware GLOBAL_CONSTANT or IDW input field in a cancelable worker.
11. **Validation** — Compare field predictions against held-out Validation-hole P10 without refitting.
12. **Explicit DFN Generation** — Generate seeded conditional explicit DFNs from the M9 field, manage merged display layers, save complete geometry, and export CSV/JSON/NPZ/VTP.
13. **Exact Second Voxelization** — Compute analytic disk–voxel intersection area, persist sparse pairs, and inspect per-set or aggregate `P32_explicit_intersection + P32_subgrid` clouds and sections.

Current status: v0.11.0-M11.1 is **READY FOR RELEASE** as the bounded **M11.1 Exact Second Voxelization** phase. Its tag and GitHub Release have not been created. Fracture-fracture connectivity, spanning/percolation, block cutting, mechanics, formal 3DEC/PFC export, and Kriging remain planned; this candidate does not claim completion of all M11 work.

### Overview

DFN Cave Studio is a desktop scientific software for:

- **DFN Modeling** — Stochastic and deterministic discrete fracture network generation
- **Voxelization** — Spatial discretization with sparse/chunked storage
- **Connectivity Analysis** — Planned after M11.1; fracture graph, clusters, and percolation are not in this release
- **Fragmentation Analysis** — Planned for the integrated v1.0.0-M12 release
- **Numerical Export** — Generic audited exports exist; formal 3DEC/PFC export remains planned

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
| M8 | ✅ Published baseline | Borehole database, spatial boundaries, and voxel-grid definition |
| M9 | ✅ Published | Local DFN parameter fields, first voxelization, and dip-only degradation |
| M10 | ✅ Published | Seeded conditional explicit DFN, multi-scale columnar geometry, subgrid P32, display, persistence, and basic export |
| M11.1 | 🟦 Ready for release | Planned tag `v0.11.0-M11.1`: analytic disk–voxel intersection, sparse second voxelization, P32 clouds/sections/cutaways, persistence, cancellation, and bilingual UI |
| Later M11 | ⬜ Planned | Fracture graph, connected clusters, boundary spanning, flow channels, mechanics, formal 3DEC/PFC export, and Kriging |
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
