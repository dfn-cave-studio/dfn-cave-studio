#!/usr/bin/env python
"""Generate synthetic demo data for end-to-end testing of DFN Cave Studio.

Creates:
  - collars.csv: 5 boreholes in a 100×100×200m block
  - surveys.csv: Survey stations at 10m intervals
  - fractures.csv: 50 fracture observations across boreholes
  - README.md: Step-by-step walkthrough

Fixed seed for full reproducibility.
"""

import csv
import math
from pathlib import Path

import numpy as np

SEED = 42
OUT_DIR = Path(__file__).resolve().parent

# Model: 100×100×200m block
BOUNDS = {"x": (0, 100), "y": (0, 100), "z": (0, 200)}

# 5 boreholes in a fan pattern
BOREHOLE_DEFS = [
    {"id": "BH-001", "x": 25, "y": 25, "z": 200, "az": 0, "dip": -90, "depth": 200},
    {"id": "BH-002", "x": 75, "y": 25, "z": 200, "az": 45, "dip": -60, "depth": 150},
    {"id": "BH-003", "x": 75, "y": 75, "z": 200, "az": 135, "dip": -75, "depth": 180},
    {"id": "BH-004", "x": 25, "y": 75, "z": 200, "az": 225, "dip": -50, "depth": 120},
    {"id": "BH-005", "x": 50, "y": 50, "z": 200, "az": 0, "dip": -90, "depth": 200},
]

# 3 joint sets
JOINT_SETS = [
    {"id": 1, "name": "Main Set", "dd": 45, "dip": 60, "kappa": 30},
    {"id": 2, "name": "Secondary Set", "dd": 135, "dip": 75, "kappa": 25},
    {"id": 3, "name": "Minor Set", "dd": 270, "dip": 30, "kappa": 20},
]


def generate():
    rng = np.random.default_rng(SEED)

    # ── collars.csv ──
    with open(OUT_DIR / "collars.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["borehole_id", "x", "y", "z", "azimuth", "dip", "final_depth"])
        for bh in BOREHOLE_DEFS:
            w.writerow([bh["id"], bh["x"], bh["y"], bh["z"], bh["az"], bh["dip"], bh["depth"]])

    # ── surveys.csv ──
    with open(OUT_DIR / "surveys.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["borehole_id", "depth", "azimuth", "dip"])
        for bh in BOREHOLE_DEFS:
            depth = 0.0
            while depth <= bh["depth"]:
                az_jitter = rng.normal(0, 0.5)
                dip_jitter = rng.normal(0, 0.3)
                w.writerow([
                    bh["id"],
                    round(depth, 1),
                    round(bh["az"] + az_jitter, 1),
                    round(bh["dip"] + dip_jitter, 1),
                ])
                depth += 10.0

    # ── fractures.csv ──
    with open(OUT_DIR / "fractures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["borehole_id", "depth", "dip_direction", "dip", "set_id"])
        for bh in BOREHOLE_DEFS:
            n_frac = rng.integers(5, 15)
            for _ in range(n_frac):
                js = rng.choice(JOINT_SETS)
                depth = rng.uniform(5, bh["depth"] - 2)
                dd = js["dd"] + rng.normal(0, 5)
                dip = js["dip"] + rng.normal(0, 3)
                w.writerow([
                    bh["id"],
                    round(depth, 1),
                    round(max(0, min(360, dd)), 1),
                    round(max(0, min(90, dip)), 1),
                    js["id"],
                ])

    # ── README.md ──
    readme = """# DFN Cave Studio — End-to-End Demo Dataset

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
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Demo data generated in: {OUT_DIR}")
    for f in ["collars.csv", "surveys.csv", "fractures.csv", "README.md"]:
        p = OUT_DIR / f
        print(f"  {f}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    generate()
