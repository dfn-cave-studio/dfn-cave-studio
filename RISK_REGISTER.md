# RISK_REGISTER.md

## Active Risks

### R1: Python 3.14 Compatibility
- **Severity**: High
- **Status**: 🟡 Monitoring
- **Description**: Python 3.14 is very new (released ~2026). Some scientific packages may have edge cases.
- **Mitigation**: All key packages installed successfully. Monitor for runtime issues.
- **Fallback**: Use Python 3.12 via conda if blocking issues arise.

### R2: 3D Block Cutting Robustness
- **Severity**: High
- **Status**: 🟡 Mitigated
- **Description**: Full 3D polyhedral block cutting from fracture planes is notoriously fragile.
- **Mitigation**: 
  - M8 implements experimental algorithm with clear labeling
  - Cross-validate against simple known geometries
  - Provide trimesh/Shapely-based fallback
  - Document limitations explicitly
- **Fallback**: Voxel-based proxy fragmentation metrics if explicit cutting fails

### R3: Large Model Performance
- **Severity**: Medium
- **Status**: 🟡 Mitigated
- **Description**: Dense 1000³ voxel arrays would require ~8GB+ per attribute.
- **Mitigation**:
  - Sparse/chunked storage from M3
  - Multi-resolution display
  - Memory usage warnings for dangerous resolutions
  - Chunked parallel processing
- **Fallback**: Hard limits with user override

### R4: Fisher Distribution Sampling Accuracy
- **Severity**: Low
- **Status**: 🟢 Mitigated
- **Description**: Fisher distribution sampling must be statistically correct.
- **Mitigation**: 
  - Scientific validation case for Fisher sampling
  - Compare against known analytical results
  - Fixed-seed reproducibility testing

### R5: P32 Target Accuracy
- **Severity**: Medium
- **Status**: 🟡 Mitigated
- **Description**: Achieved P32 may differ from target P32.
- **Mitigation**:
  - Iterative generation with convergence check
  - Default 5% tolerance
  - Report reasons when target cannot be met
  - Virtual borehole validation

### R6: Connectivity False Positives
- **Severity**: High
- **Status**: 🟡 Mitigated
- **Description**: Easy to incorrectly identify connectivity.
- **Mitigation**:
  - Explicit geometric intersection testing
  - Distinguish intersection types
  - Validation against known cases
  - Conservative defaults

### R7: No C Compiler Available
- **Severity**: Low
- **Status**: 🟢 Resolved
- **Description**: Some packages may need compilation from source.
- **Status**: All packages installed from pre-built wheels. No compilation needed yet.

### R8: External Code License Contamination
- **Severity**: Critical
- **Status**: 🟢 Mitigated
- **Description**: Risk of inadvertently incorporating GPL/AGPL code.
- **Mitigation**:
  - All code written from scratch or verified permissive licenses
  - THIRD_PARTY_NOTICES.md maintained
  - License checks before any third-party code inclusion

## Risk Matrix

| Risk | Likelihood | Impact | Residual |
|------|-----------|--------|----------|
| R1: Python 3.14 | Low | High | Low |
| R2: Block cutting | Medium | High | Medium |
| R3: Performance | Medium | Medium | Low-Medium |
| R4: Fisher accuracy | Low | Low | Low |
| R5: P32 accuracy | Low | Medium | Low |
| R6: Connectivity | Medium | High | Medium |
| R7: No compiler | Low | Low | Resolved |
| R8: License | Low | Critical | Low |

## Review Schedule
- After each milestone completion
- Before any external release
- When new external dependencies are added
