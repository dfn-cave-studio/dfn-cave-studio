# DFN Cave Studio v0.11.0-M11.1 review

**Phase:** M11.1 Exact Second Voxelization
**Commit baseline:** `e066e6cedb05cac2b01caea1464ba86b6a29f930`
**Milestone identifier:** `M11.1`
**Status:** READY FOR RELEASE

## Test jobs (not summed across duplicated Python matrix runs)

| Job | Core | GUI | Coverage |
|---|---:|---:|---:|
| Windows Python 3.12 | 710 | 155 | 87.25% |
| Windows Python 3.13 | 710 | 155 | 87.25% |

Linux Mesa/Xvfb real VTK: 22/22 passed.

## Completed M11.1 scope

- Analytic circular disk-voxel exact area intersection
- Exact second voxelization
- P32_explicit_intersection + P32_subgrid
- Per-joint-set and all-set results
- Sparse persistence, reopen, and invalidation management
- M11 cloud, orthogonal/arbitrary sections, plane cutaway, and box cutaway
- M11 layer and scalar-bar lifecycle
- Chinese and English interface
- M10 configuration save validation and recovery protection
- M9/M11 opaque finite-value occlusion fixes
- Cooperative fast cancellation
- Windows Python 3.12/3.13 CI and Linux Mesa/Xvfb real VTK rendering tests

## Not implemented in M11.1

- Fracture-fracture intersection graph
- Connected clusters and boundary-spanning paths
- Flow/percolation and preferential channels
- Block cutting and fragmentation statistics
- Mechanical properties
- Formal 3DEC/PFC export
- Kriging interpolation

M11.1 is ready for release as a bounded phase. Its tag and GitHub Release have not been created, and it does not claim completion of all M11 work.
