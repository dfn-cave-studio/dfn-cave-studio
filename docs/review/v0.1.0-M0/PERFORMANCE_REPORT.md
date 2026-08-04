# Performance Report — v0.1.0-M0

## Test Execution Performance

| Metric | Value |
|--------|-------|
| Total test time | 3.12s |
| Unit test time | ~2.5s |
| GUI test time | ~0.6s |
| DFN generation (50³ model, 1 set) | <0.1s |
| DFN generation (50³ model, 2 sets) | <0.2s |

## Import Time

| Module | Import Time |
|--------|-------------|
| dfn_cave_studio (core) | <0.1s |
| dfn_cave_studio.ui | ~0.5s (includes Qt init) |
| Full application startup | ~1.5s |

## Memory (Baseline)

| State | Memory |
|-------|--------|
| Python idle | ~30 MB |
| After imports | ~80 MB |
| With MainWindow | ~120 MB |
| After DFN generation (50³, ~100 fractures) | ~130 MB |

## Notes
- Performance profiling infrastructure will be built in M3
- No large models tested yet (expected in M3/M4)
- Qt startup dominates launch time
- PyVistaQt 3D viewport not performance-tested in headless CI
