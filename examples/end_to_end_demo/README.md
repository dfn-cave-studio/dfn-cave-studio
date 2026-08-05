# DFN Cave Studio — End-to-End Demo Dataset

## Overview
Synthetic demo dataset for end-to-end testing of the DFN Cave Studio GUI workflow.
Generated with fixed random seed (42) for full reproducibility.

## Contents
- `collars.csv` — 5 borehole collars (100×100×200m block)
- `surveys.csv` — Survey stations at 10m intervals
- `fractures.csv` — 50+ fracture observations across boreholes

## Quick Start (GUI)
1. Launch DFN Cave Studio: `python -m dfn_cave_studio.app`
2. File → New Project → name it
3. Data → Borehole Manager → Import collars, surveys, fractures
4. Voxel → Voxel Settings → Set bounds (0-100, 0-100, 0-200), cell=2m
5. DFN → Joint Set Manager → Configure 3 joint sets
6. DFN → Generate DFN → Start Computation
7. View results in 3D viewport
8. Export → VTK

## Joint Set Suggestions
| Set | Dip Direction | Dip | Kappa | P32 |
|-----|--------------|-----|-------|-----|
| Main | 45° | 60° | 30 | 0.3 |
| Secondary | 135° | 75° | 25 | 0.2 |
| Minor | 270° | 30° | 20 | 0.1 |

## Model Parameters
- Model: 100×100×200m (2,000,000 m³)
- Voxel: 2m cells → 50×50×100 = 250,000 voxels
- Seed: 42
