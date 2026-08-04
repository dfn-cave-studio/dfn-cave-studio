# Known Issues — v0.1.0-M0

## M0 Issues

### KI-001: 3D Viewport Requires GPU
**Severity:** Low  
**Impact:** PyVistaQt 3D viewport will not render without GPU with OpenGL 3.2+ support.  
**Workaround:** Software rendering via Mesa3D or using the headless CI mode.  
**Fix:** M1 will add graceful fallback to 2D-only mode.

### KI-002: Menu Items Are Placeholder Stubs
**Severity:** Low  
**Impact:** Most menu actions show informational dialogs instead of functional dialogs.  
**Expected:** Functional dialogs implemented progressively in M1–M12.  
**Workaround:** None needed for M0 (this is by design).

### KI-003: No Project Persistence
**Severity:** Medium  
**Impact:** Cannot save or load projects.  
**Expected:** M1 implements project save/load with JSON serialization.  

### KI-004: No DFN 3D Rendering
**Severity:** Medium  
**Impact:** DFN generation works (tested in unit tests) but cannot be viewed in 3D.  
**Expected:** M2 adds 3D rendering of generated fractures.

### KI-005: Python 3.14 Edge Cases
**Severity:** Low  
**Impact:** Python 3.14 is very recent (May 2026). Some edge cases may exist in third-party packages.  
**Mitigation:** All 137 tests pass; major packages (PySide6, PyVista, VTK, NumPy, SciPy) confirmed compatible.

## Resolution Plan
- KI-001: M1
- KI-002: M1–M12 (ongoing)
- KI-003: M1
- KI-004: M2
- KI-005: Continuous monitoring
