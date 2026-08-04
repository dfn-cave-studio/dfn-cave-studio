# DFN Cave Studio — Acceptance Test Specifications

This document defines the acceptance test suite for DFN Cave Studio. Every test includes: a unique ID, a description of inputs, expected outputs, pass criteria, and the file location where the test implementation lives.

Tests are organized into four categories:

- **MG** — Math & Geometry
- **SV** — Synthetic Scientific Validation
- **PR** — Project (persistence, import/export)
- **UI** — GUI / User Interface

---

## MG — Math & Geometry Tests

---

### MG-01: Vector Normalization

**Description:** Verify that `Vector3.normalize()` returns a unit vector for arbitrary non-zero inputs, and raises or returns a zero vector for the zero vector.

**Inputs:**
- `Vector3(3, 0, 0)` — axis-aligned
- `Vector3(1, 2, 3)` — general
- `Vector3(-4.5, 0.1, 999.9)` — large component
- `Vector3(0, 0, 0)` — zero vector

**Expected Outputs:**
- `Vector3(1, 0, 0)` for input 1
- Unit vector in direction (1, 2, 3) for input 2: magnitude == 1.0 within tolerance
- Unit vector for input 3: magnitude == 1.0 within tolerance
- `ValueError` raised, OR `Vector3(0,0,0)` returned (design choice must be documented and consistent)

**Pass Criteria:**
- All normalized results have magnitude within `1e-12` of 1.0
- Original vector direction is preserved (cross product of original and normalized is zero vector)
- Zero-vector case is handled consistently

**Test File:** `tests/math/test_vector.py::TestVectorNormalization`

---

### MG-02: Dip Direction / Dip to Normal Vector Round-Trip

**Description:** Verify that converting between dip-direction/dip-angle and a unit normal vector is lossless (round-trip) and correct for known orientations.

**Inputs (dip direction, dip angle in degrees):**
- `(0, 0)` — horizontal plane, north dip (edge case)
- `(90, 45)` — east-dipping at 45 degrees
- `(180, 90)` — vertical plane striking east-west
- `(270, 30)` — west-dipping at 30 degrees
- `(45, 60)` — northeast-dipping at 60 degrees

**Expected Outputs:**
- Round-trip: `dip_to_normal(normal_to_dip(n))` returns the original normal `n` within tolerance
- Round-trip: `normal_to_dip(dip_to_normal(dd, d))` returns `(dd, d)` within tolerance
- Known conversion check:
  - `(0, 0)` normal points straight down: `Vector3(0, 0, -1)` (convention-dependent; document convention)
  - `(90, 90)` normal is horizontal, pointing north

**Pass Criteria:**
- All round-trips preserve values within `1e-10` tolerance
- All known-conversion checks match documented convention

**Test File:** `tests/math/test_orientation.py::TestDipNormalRoundTrip`

---

### MG-03: Fisher Orientation Sampling — Statistical Validation

**Description:** Verify that Fisher-distributed random orientations match the theoretical distribution to within statistical confidence.

**Inputs:**
- Fisher distribution with mean direction `(dd=45, dip=30)`, kappa = 15
- Sample size: N = 10,000

**Expected Outputs:**
- Sample mean direction converges to input mean within 2 degrees
- Sample circular variance matches theoretical: `1 - (bessel_ratio(kappa))` within 0.02
- Empirical CDF of angular deviations from mean matches theoretical Fisher CDF (Kolmogorov-Smirnov test, p > 0.05)

**Pass Criteria:**
- KS test passes at alpha = 0.05 significance level
- Mean direction within 2 degrees of target
- Circular variance within 0.02 of theoretical

**Test File:** `tests/math/test_fisher.py::TestFisherStatisticalValidation`

---

### MG-04: Fixed Seed Reproducibility

**Description:** Verify that setting a fixed random seed produces identical results across two independent runs.

**Inputs:**
- Fisher distribution: mean `(0, 0)`, kappa = 10
- Sample size: N = 500
- Fixed seed: `42`

**Expected Outputs:**
- Two independent calls to `FisherDistribution.sample(n=500)` with the same seed produce the exact same sequence of 500 orientations, element by element.

**Pass Criteria:**
- All 500 orientation pairs (dd, dip) match exactly between the two runs (or all normal vectors match component-wise within `1e-15`)

**Test File:** `tests/math/test_fisher.py::TestFisherSeedReproducibility`

---

### MG-05: Plane-Line Intersection

**Description:** Verify computation of the intersection point between a plane and a line.

**Inputs:**
- Plane: point `(0,0,0)`, normal `(0,0,1)` (horizontal plane at z=0)
- Lines:
  1. Point `(1,1,5)`, direction `(0,0,-1)` — intersects at `(1,1,0)`
  2. Point `(2,3,-5)`, direction `(0,0,1)` — intersects at `(2,3,0)`
  3. Point `(1,0,1)`, direction `(1,0,0)` — parallel, no intersection

**Expected Outputs:**
- Case 1: intersection at `(1, 1, 0)`
- Case 2: intersection at `(2, 3, 0)`
- Case 3: `None` or exception indicating parallel/no-intersection

**Pass Criteria:**
- Intersection points correct within `1e-10`
- Parallel case correctly identified

**Test File:** `tests/math/test_intersection.py::TestPlaneLineIntersection`

---

### MG-06: Plane-Box Intersection

**Description:** Verify detection and computation of plane-box intersection (the polygon formed by clipping a plane against an axis-aligned box).

**Inputs:**
- Box: axis-aligned from `(0,0,0)` to `(10,10,10)`
- Planes:
  1. `z = 5`, normal `(0,0,1)` — horizontal mid-plane, intersection is the full `10x10` square
  2. `x = 3`, normal `(1,0,0)` — vertical mid-plane, intersection is `10x10` square
  3. `z = -1`, normal `(0,0,1)` — no intersection (plane outside box)
  4. Plane through `(5,5,5)` with normal `(1,1,1).normalized()` — diagonal cut

**Expected Outputs:**
- Case 1: polygon with 4 vertices forming a `10x10` square in z=5 plane
- Case 2: polygon with 4 vertices forming a `10x10` square in x=3 plane
- Case 3: empty polygon (no intersection)
- Case 4: polygon with 3 to 6 vertices (diagonal slice through a cube produces a hexagon or triangle)

**Pass Criteria:**
- Case 1 polygon area equals 100 within tolerance
- Case 2 polygon area equals 100 within tolerance
- Case 3 returns empty/no-intersection indicator
- Case 4 polygon area is positive and all vertices lie on the plane and within the box

**Test File:** `tests/math/test_intersection.py::TestPlaneBoxIntersection`

---

### MG-07: Fracture-Voxel Intersection

**Description:** Verify that a fracture intersected with a voxel (small sub-box) correctly reports intersection existence and the intersecting polygon area.

**Inputs:**
- Voxel grid: `10x10x10` cube, voxel size `1x1x1`
- Fracture: horizontal square, center `(5,5,5)`, normal `(0,0,1)`, side length 8 (extends from `(1,1,5)` to `(9,9,5)`)
- Test: intersect fracture with voxel at grid index `(0,0,5)` (the voxel from `(0,0,5)` to `(1,1,6)`)

**Expected Outputs:**
- Voxel `(0,0,5)` does not intersect (fracture extends from x=1 to x=9, voxel is x=0 to x=1 — check convention: fracture edge at x=1; if voxel occupies [0,1), no intersection)
- Voxel `(1,1,5)` intersects with area = 1.0 (full voxel face)

**Pass Criteria:**
- Non-intersecting voxels correctly identified
- Intersecting voxel area correct within `1e-10`
- Boundary-touching cases handled consistently per documented convention

**Test File:** `tests/math/test_intersection.py::TestFractureVoxelIntersection`

---

### MG-08: Fracture-Fracture Intersection

**Description:** Verify that the intersection of two fractures is correctly computed as a line segment (or empty).

**Inputs:**
- Fracture A: horizontal square, center `(5,5,5)`, normal `(0,0,1)`, side length 6
- Fracture B: vertical square, center `(5,5,5)`, normal `(1,0,0)` (oriented east-west), side length 6

**Expected Outputs:**
- Intersection is a line segment along y-direction at x=5, z=5 from y=2 to y=8
- Length of intersection segment = 6.0

**Pass Criteria:**
- Intersection detected (not empty)
- Intersection segment endpoints correct within `1e-10`
- Segment length correct within `1e-10`

**Test File:** `tests/math/test_intersection.py::TestFractureFractureIntersection`

---

### MG-09: Polygon Area Calculation

**Description:** Verify area computation for planar polygons of varying complexity.

**Inputs:**
- Triangle in XY plane: `(0,0,0)`, `(3,0,0)`, `(0,4,0)` — area = 6.0
- Unit square in XY plane: `(0,0,0)`, `(1,0,0)`, `(1,1,0)`, `(0,1,0)` — area = 1.0
- Regular hexagon in XY plane, circumradius = 2
- Non-convex (bow-tie) shape — should compute signed area
- Vertices given in clockwise vs counter-clockwise order (magnitude should match, sign may differ)

**Expected Outputs:**
- Triangle area: 6.0
- Square area: 1.0
- Hexagon area: `(3 * sqrt(3) / 2) * (2^2)` = `6 * sqrt(3)` = approx 10.3923
- Signed area magnitude matches regardless of winding order

**Pass Criteria:**
- All areas correct within `1e-10`
- Self-intersecting polygon at least produces a documented result (may warn)
- Clockwise vs CCW yields same magnitude

**Test File:** `tests/math/test_polygon.py::TestPolygonArea`

---

### MG-10: Fracture Clipping to Boundary

**Description:** Verify that a fracture polygon clipped against a bounding box produces the correct clipped polygon.

**Inputs:**
- Box: `(0,0,0)` to `(10,10,10)`
- Fracture: square centered at `(8,5,5)`, normal `(0,0,1)`, side length 10 (extends beyond box on +x side and slightly on -x side)

**Expected Outputs:**
- Clipped polygon has 4 vertices (rectangle)
- Clipped polygon area = 10 * 8 = 80 (fracture from x=-2 to x=18, clipped to x=0 to x=10: width=10; y from y=0 to y=10: width=10)

Wait — recheck: center at x=8, side length 10 means x range = [3, 13]. Clipped to box x in [0,10] gives [3,10], width = 7. y range = [0,10], width = 10. Area = 70.

**Pass Criteria:**
- All vertices of clipped polygon lie within or on the box boundary
- Clipped polygon lies in the fracture plane
- Area computed correctly from geometry, matching expected value within `1e-10`

**Test File:** `tests/math/test_fracture.py::TestFractureClipping`

---

### MG-11: P32 Computation

**Description:** Verify that P32 (total fracture area per unit volume) is computed correctly for a known fracture set.

**Inputs:**
- Domain: unit cube `(0,0,0)` to `(1,1,1)`, volume = 1.0
- 10 identical horizontal square fractures, each of area 0.5, placed fully inside the cube

**Expected Outputs:**
- Total fracture area = 10 * 0.5 = 5.0 (after clipping to domain if any extend outside)
- P32 = 5.0 / 1.0 = 5.0

**Pass Criteria:**
- P32 computed as `sum(clipped_areas) / domain_volume`
- Result correct within `1e-10`

**Test File:** `tests/math/test_dfm_metrics.py::TestP32Computation`

---

### MG-12: P10 Computation

**Description:** Verify that P10 (number of fracture intersections per unit length along a scanline/borehole) is computed correctly.

**Inputs:**
- Domain: unit cube
- 5 horizontal fractures at z = 0.1, 0.3, 0.5, 0.7, 0.9, each fully spanning the cube in x and y
- Borehole: vertical line from `(0.5, 0.5, 0)` to `(0.5, 0.5, 1)`, length = 1.0

**Expected Outputs:**
- Number of intersections = 5
- P10 = 5 / 1.0 = 5.0

**Pass Criteria:**
- All 5 intersections detected
- P10 = 5.0 within tolerance
- Test with borehole that misses all fractures returns P10 = 0

**Test File:** `tests/math/test_dfm_metrics.py::TestP10Computation`

---

## SV — Synthetic Scientific Validation

---

### SV-01: Single Horizontal Fracture Through a Cube

**Description:** Generate a single horizontal fracture through the center of a unit cube and verify geometric properties.

**Inputs:**
- Domain: unit cube `(0,0,0)` to `(1,1,1)`
- One fracture: center `(0.5, 0.5, 0.5)`, dip direction = 0, dip = 0 (horizontal), shape = square, side length = 2 (oversized to fill)

**Expected Outputs:**
- Clipped fracture is the full unit square at z=0.5, area = 1.0
- P32 = 1.0

**Pass Criteria:**
- Fracture area within domain = 1.0
- P32 = 1.0
- Fracture normal is `(0, 0, -1)` or `(0, 0, 1)` depending on convention

**Test File:** `tests/validation/test_synthetic_cases.py::TestSingleHorizontalFracture`

---

### SV-02: Single Dipping Fracture Through Regular Voxels

**Description:** Generate a single dipping fracture through a regular voxel grid and verify per-voxel intersection areas.

**Inputs:**
- Domain: unit cube
- Voxel grid: 10x10x10 (voxel size 0.1 each)
- One fracture: passes through the cube, dip direction = 90 (east), dip = 45 degrees, center at `(0.5, 0.5, 0.5)`, circular, radius = 1.0

**Expected Outputs:**
- A contiguous set of intersected voxels
- Sum of all per-voxel intersection areas equals the total clipped fracture area
- The intersected voxels form a planar band through the cube

**Pass Criteria:**
- Sum of voxel intersection areas equals total fracture area within `1e-8` relative tolerance
- All intersected voxel centers lie approximately on or near the fracture plane (within half a voxel diagonal)
- No gaps in the voxel intersection pattern (connectivity check)

**Test File:** `tests/validation/test_synthetic_cases.py::TestSingleDippingFractureVoxels`

---

### SV-03: Two Intersecting Fractures

**Description:** Generate two orthogonal fractures that intersect and verify the intersection line is correct.

**Inputs:**
- Domain: unit cube
- Fracture A: horizontal at z=0.5, square, side 2
- Fracture B: vertical at x=0.5, striking north-south, square, side 2

**Expected Outputs:**
- Intersection detected
- Intersection line: from `(0.5, 0.0, 0.5)` to `(0.5, 1.0, 0.5)`, length = 1.0
- Total fracture count = 2

**Pass Criteria:**
- Intersection segment endpoints correct within `1e-10`
- Length = 1.0

**Test File:** `tests/validation/test_synthetic_cases.py::TestTwoIntersectingFractures`

---

### SV-04: Three Orthogonal Fracture Sets Forming Regular Blocks

**Description:** Generate three orthogonal fracture sets with regular spacing to produce a known block pattern.

**Inputs:**
- Domain: `(0,0,0)` to `(2,2,2)`
- Set 1: fractures at x = 0.5, 1.0, 1.5 (spacing 0.5), normal `(1,0,0)`
- Set 2: fractures at y = 0.5, 1.0, 1.5 (spacing 0.5), normal `(0,1,0)`
- Set 3: fractures at z = 0.5, 1.0, 1.5 (spacing 0.5), normal `(0,0,1)`

**Expected Outputs:**
- 9 fractures total
- Blocks formed: 64 blocks (4x4x4 grid of 0.5-sized cubes), though boundary-touching depends on clipping — at minimum 27 fully interior blocks of size 0.5 (3x3x3)
- All blocks are axis-aligned rectangular prisms

**Pass Criteria:**
- Block count matches expected (document expected count based on clipping convention)
- Block volume distribution is uniform (all equal size)
- Total block volume sums to domain volume (within tolerance)

**Test File:** `tests/validation/test_synthetic_cases.py::TestOrthogonalFractureSets`

---

### SV-05: Known-Spacing Fractures Producing Known Fragmentation

**Description:** Generate parallel equally spaced fractures and verify the resulting block sizes.

**Inputs:**
- Domain: unit cube
- 9 equally spaced horizontal fractures at z = 0.1, 0.2, ..., 0.9 (spacing = 0.1)

**Expected Outputs:**
- 10 horizontal layers (blocks), each of thickness 0.1
- Expected block volume per layer: 1.0 * 1.0 * 0.1 = 0.1
- Total fragmentation: 10 blocks

**Pass Criteria:**
- Block count = 10
- Mean block volume = 0.1 within `1e-10`
- Block volume standard deviation approx 0 (all equal)

**Test File:** `tests/validation/test_synthetic_cases.py::TestKnownSpacingFragmentation`

---

### SV-06: Target P32 vs Achieved P32 Comparison

**Description:** Generate a stochastic DFN with a target P32, measure the achieved P32, and verify it converges to the target as the number of fractures increases.

**Inputs:**
- Domain: unit cube
- Target P32 = 2.0
- Fractures: circular, radius drawn from log-normal (mean=0.1, sigma=0.05)
- Realizations at N = 10, 50, 100, 500, 1000 fractures

**Expected Outputs:**
- As N increases, achieved P32 converges toward 2.0
- At N = 1000, relative error |achieved - target| / target < 5%

**Pass Criteria:**
- Convergence trend is monotonic or near-monotonic (measured by decreasing relative error with increasing N, aside from stochastic noise)
- At N >= 500, relative error < 10%
- Repeat with 5 different seeds: all pass the 10% threshold (test is not seed-flaky)

**Test File:** `tests/validation/test_p32_convergence.py::TestP32Convergence`

---

### SV-07: Virtual Borehole P10

**Description:** Place virtual boreholes through a known DFN and verify P10 matches expectation.

**Inputs:**
- Domain: unit cube
- 100 horizontal fractures uniformly distributed in z
- Vertical borehole through center: expected intersections approx 100 (depending on fracture sizes and positions)
- Instead, deterministic variant: 10 fractures fully spanning x-y at z = 0.05, 0.15, ..., 0.95, vertical borehole

**Expected Outputs (deterministic variant):**
- P10 = 10 intersections / 1.0 m borehole = 10.0

**Pass Criteria:**
- All fractures intersected
- P10 = 10.0 exactly

**Test File:** `tests/validation/test_borehole.py::TestVirtualBoreholeP10`

---

### SV-08: Global Constant Parameter Model

**Description:** Generate a homogeneous DFN (same intensity everywhere) and verify spatial uniformity of fracture density.

**Inputs:**
- Domain: `(0,0,0)` to `(10,10,10)`
- Single fracture set with constant intensity (P32 = 1.0), uniform random centroid placement
- Subdivide domain into 8 octants, compute P32 per octant

**Expected Outputs:**
- P32 per octant is approximately 1.0 (within statistical noise)
- Chi-squared test for uniform distribution of fracture counts across octants: p > 0.05

**Pass Criteria:**
- No octant deviates from expected P32 by more than 3 standard deviations
- Chi-squared test passes at alpha = 0.05
- Repeat with 3 seeds: all pass

**Test File:** `tests/validation/test_spatial_models.py::TestGlobalConstantModel`

---

### SV-09: Two-Domain Different P32 Model

**Description:** Generate a DFN with two spatial domains having different target P32 values and verify the spatial variation.

**Inputs:**
- Domain: `(0,0,0)` to `(10,10,10)`
- Sub-domain A: `(0,0,0)` to `(5,10,10)` with P32 = 1.0
- Sub-domain B: `(5,0,0)` to `(10,10,10)` with P32 = 4.0
- Generate with sufficient fractures for statistical significance

**Expected Outputs:**
- P32 in domain A approximately 1.0
- P32 in domain B approximately 4.0
- Ratio P32_B / P32_A approximately 4.0

**Pass Criteria:**
- P32_A within 20% of 1.0
- P32_B within 20% of 4.0
- P32_B > P32_A with high confidence (t-test, p < 0.01)

**Test File:** `tests/validation/test_spatial_models.py::TestTwoDomainModel`

---

### SV-10: Same-Seed Full Reproducibility

**Description:** Generate a complete DFN model twice with the same seed and verify bitwise-identical output.

**Inputs:**
- Full model definition with 3 fracture sets, spatial domains, size distributions
- Fixed seed: 12345
- Two independent `generate()` calls

**Expected Outputs:**
- Both generated models have the same number of fractures
- Corresponding fractures have identical centroids, orientations, shapes, and areas
- P32, P10, connectivity graph, block counts all match exactly

**Pass Criteria:**
- All fracture properties match element-by-element within `1e-15`
- All derived metrics match exactly

**Test File:** `tests/validation/test_reproducibility.py::TestSameSeedReproducibility`

---

### SV-11: Different-Seed Statistical Consistency but Geometric Difference

**Description:** Verify that two runs with different seeds produce geometrically different but statistically equivalent DFNs.

**Inputs:**
- Same model definition as SV-10
- Seeds: 12345 and 67890

**Expected Outputs:**
- Fracture centroids differ (not bitwise identical)
- Statistical properties are consistent:
  - Achieved P32 within 10% of each other
  - Orientation distributions not significantly different (two-sample KS test, p > 0.05)
  - Fracture size distributions not significantly different (two-sample KS test, p > 0.05)

**Pass Criteria:**
- At least one fracture centroid differs between runs
- P32 difference < 10%
- Both KS tests pass at alpha = 0.05

**Test File:** `tests/validation/test_reproducibility.py::TestDifferentSeedConsistency`

---

### SV-12: Fracture Connectivity Path

**Description:** Verify that the connectivity engine correctly finds paths between points connected by intersecting fractures.

**Inputs:**
- Domain: unit cube
- Three fractures:
  - F1: horizontal at z=0.5, spanning the cube
  - F2: vertical at x=0.5, spanning from y=0 to y=0.6, z=0 to z=1
  - F3: vertical at y=0.7, spanning from x=0.5 to x=1.0, z=0 to z=1
- F1 intersects F2 (they touch at y <= 0.6 on the x=0.5, z=0.5 line)
- F1 intersects F3 (they touch at x >= 0.5 on the y=0.7, z=0.5 line)
- F2 and F3 do NOT intersect (F2 ends at y=0.6, F3 is at y=0.7)
- Start point: on F2 at `(0.5, 0.3, 0.5)`
- End point: on F3 at `(0.8, 0.7, 0.5)`

**Expected Outputs:**
- Path exists: Start -> F2 -> F1 -> F3 -> End
- Path length and segment details returned
- Direct F2-to-F3 path does NOT exist

**Pass Criteria:**
- Connectivity graph correctly identifies F1-F2 and F1-F3 intersections
- F2-F3 non-intersection correctly identified
- Path found traverses F1-F2 and F1-F3
- No false path from F2 directly to F3

**Test File:** `tests/validation/test_connectivity.py::TestConnectivityPath`

---

## PR — Project Tests

---

### PR-01: New Project Creation

**Description:** Verify that a new project is created with correct default values.

**Inputs:**
- Project name: `"Test Project"`
- Author: `"Tester"`
- No existing file

**Expected Outputs:**
- Project object created with:
  - `name == "Test Project"`
  - `author == "Tester"`
  - `created_date` is today's date
  - `version` matches current software version
  - DFN model is empty (no fracture sets)
  - Modified flag is `True` (unsaved changes)

**Pass Criteria:**
- All fields match expected values
- Project is in valid state (no exceptions on access)

**Test File:** `tests/project/test_project.py::TestNewProject`

---

### PR-02: Save Project

**Description:** Verify that a project can be saved to disk and the file is valid.

**Inputs:**
- A project with one fracture set containing 50 fractures
- Save path: a temporary file

**Expected Outputs:**
- File exists at save path after save
- File is non-empty
- File is valid per the project format schema (JSON parseable or HDF5 readable)
- Project modified flag is `False` after save

**Pass Criteria:**
- File created and non-empty
- File passes schema validation
- Modified flag cleared

**Test File:** `tests/project/test_persistence.py::TestSaveProject`

---

### PR-03: Open Project

**Description:** Verify that a saved project can be opened and all data is correctly restored.

**Inputs:**
- A project saved to disk (as in PR-02)

**Expected Outputs:**
- Opened project has same name, author, version
- Same number of fracture sets
- Each fracture set has same number of fractures
- Fracture properties (centroids, orientations, shapes) match within `1e-12`
- Derived metrics (P32, etc.) match

**Pass Criteria:**
- Full structural and numeric equality with the original project
- Modified flag is `False` after open

**Test File:** `tests/project/test_persistence.py::TestOpenProject`

---

### PR-04: Version Upgrade

**Description:** Verify that a project saved by an older software version is correctly upgraded when opened by a newer version.

**Inputs:**
- A project file created by software version 0.5.0 (mock: manually construct file with old version tag and old schema)
- New software version 0.7.0 (current)
- Old schema missing a field that was added in 0.7.0 (e.g., `fracture_set.color`)

**Expected Outputs:**
- Project opens without error
- Missing fields populated with documented defaults
- After save, the file uses the new schema version
- All original data preserved

**Pass Criteria:**
- No exception on open
- Defaulted fields have correct default values
- Re-saved file schema version is current
- Round-trip: open upgraded file again, data matches

**Test File:** `tests/project/test_versioning.py::TestVersionUpgrade`

---

### PR-05: Import Bad Data Handling

**Description:** Verify that importing corrupted or invalid fracture data produces clear error messages rather than crashes.

**Inputs:**
- CSV file with missing columns (no `dip_direction` column)
- CSV file with non-numeric values in numeric columns
- CSV file with inconsistent row lengths
- Empty CSV file (header only)
- Binary garbage file with `.csv` extension

**Expected Outputs:**
- Each case raises a specific, documented exception type
- Error message describes the problem (which column is missing, which row has bad data)
- No partial data is loaded into the project
- Application does not crash

**Pass Criteria:**
- Appropriate exceptions raised for each case
- Error messages are human-readable and specific
- Project state unchanged after failed import

**Test File:** `tests/project/test_import.py::TestBadDataHandling`

---

### PR-06: Computation Cancellation

**Description:** Verify that a long-running computation can be cancelled and the system returns to a consistent state.

**Inputs:**
- Large DFN model (e.g., 10,000 fractures) requiring connectivity analysis
- Initiate computation, then request cancellation after a short delay (e.g., 100 ms)

**Expected Outputs:**
- Computation thread terminates
- Cancellation is acknowledged (event or callback fires)
- Project data is unchanged (partial results are not committed)
- A new computation can be started without issues

**Pass Criteria:**
- Computation stops within 500 ms of cancellation request
- Project state identical to pre-computation state
- Subsequent computation runs successfully

**Test File:** `tests/project/test_computation.py::TestCancellation`

---

### PR-07: Export-Then-Import Round Trip

**Description:** Verify that exporting fractures to CSV and reimporting them produces an equivalent DFN.

**Inputs:**
- A DFN model with 2 fracture sets, 100 fractures each
- Export to CSV
- Create a new empty project
- Import the CSV

**Expected Outputs:**
- Reimported project has same number of fracture sets and fractures
- Fracture properties match within tolerance (export precision may limit this; document the precision)
- P32 matches within 1%

**Pass Criteria:**
- Fracture count preserved
- Geometric properties match within documented export precision
- P32 difference < 1%

**Test File:** `tests/project/test_import_export.py::TestExportImportRoundTrip`

---

### PR-08: Restore After Restart

**Description:** Verify that a project saved before application restart can be reopened correctly.

**Inputs:**
- Create, populate, and save a project to a known file path
- Simulate "restart" by clearing all in-memory state
- Open the project from the saved file

**Expected Outputs:**
- Project opens correctly
- All data intact as in PR-03

**Pass Criteria:**
- Full data integrity as in PR-03
- No reliance on unsaved in-memory state (e.g., temp files, singleton caches)

**Test File:** `tests/project/test_persistence.py::TestRestoreAfterRestart`

---

## UI — GUI Tests

These tests verify the graphical user interface. They may be semi-automated (using QtTest or similar) or manual. Manual tests include step-by-step procedures.

---

### UI-01: Main Window Launches

**Description:** Verify that the application main window launches without errors and displays core UI elements.

**Inputs:**
- Execute: `python -m dfn_cave_studio` or launch the installed application

**Expected Outputs:**
- Main window appears
- Title bar contains "DFN Cave Studio"
- Menu bar visible with File, Edit, View, Compute, Help menus
- Central widget area present (empty viewport or welcome screen)
- Status bar visible at bottom
- No console errors or exception dialogs

**Pass Criteria:**
- Window renders within 5 seconds of launch
- All required UI elements present
- No errors logged

**Test File:** `tests/gui/test_main_window.py::TestLaunch`

---

### UI-02: Menus Functional

**Description:** Verify that all menu items are present and trigger the expected actions.

**Inputs:**
- Freshly launched application

**Expected Outputs:**
- File > New Project: opens new project dialog or creates default project
- File > Open: opens file chooser dialog
- File > Save: triggers save (may show save-as if new project)
- File > Save As: opens file chooser for save path
- File > Exit: closes application
- Edit > Preferences/Preferences: opens settings dialog
- View > toggle panels: shows/hides respective panels
- Compute > Run Analysis: starts computation
- Help > About: shows about dialog

**Pass Criteria:**
- Each menu action executes without error
- Dialogs appear when expected
- Keyboard shortcuts (if defined) work

**Test File:** `tests/gui/test_menus.py::TestMenusFunctional`

---

### UI-03: Parameter Changes Take Effect

**Description:** Verify that modifying fracture set parameters in the UI updates the generated DFN.

**Inputs:**
- Create a project with one fracture set
- Edit the fracture set parameters in the UI (change dip from 45 to 60, change P32 from 1.0 to 3.0)
- Regenerate the DFN (or trigger recomputation)

**Expected Outputs:**
- Fractures generated with new dip angle (mean dip approx 60)
- P32 of generated fractures approx 3.0 (not 1.0)

**Pass Criteria:**
- Parameter change is reflected in generated output
- Reverting parameters and regenerating restores original behavior

**Test File:** `tests/gui/test_parameters.py::TestParameterChangesTakeEffect`

---

### UI-04: Error Dialogs

**Description:** Verify that invalid user actions produce error dialogs rather than silent failures or crashes.

**Inputs:**
- Attempt to open a non-existent file
- Attempt to save to a read-only path
- Enter negative P32 value in parameter editor
- Enter dip angle > 90 degrees

**Expected Outputs:**
- Error dialog appears with descriptive message
- Dialog is modal
- Application remains responsive after dialog dismissed
- Invalid input is rejected (field reverts to last valid value or is highlighted)

**Pass Criteria:**
- Error dialog shown for each case
- No crash or hang
- Invalid state not accepted

**Test File:** `tests/gui/test_error_handling.py::TestErrorDialogs`

---

### UI-05: Background Computation Does Not Freeze UI

**Description:** Verify that the UI remains responsive during a long computation.

**Inputs:**
- Large DFN model requiring 5+ seconds of computation
- Initiate computation via menu or button
- Attempt to interact with UI during computation (move window, click menus, scroll panels)

**Expected Outputs:**
- UI remains responsive: window can be moved, menus can be opened
- Progress indicator updates (progress bar or spinner)
- Cancel button is active and functional
- Computation completes and results appear

**Pass Criteria:**
- Window does not show "Not Responding" (Windows) or beachball (macOS)
- User interactions processed within 200 ms
- Progress updates at least once per second

**Test File:** `tests/gui/test_responsiveness.py::TestBackgroundComputation`

---

### UI-06: Layer Toggling

**Description:** Verify that fracture set visibility can be toggled in the layer panel and the 3D view updates accordingly.

**Inputs:**
- Project with 3 fracture sets, all visible
- Toggle visibility of set 2 off

**Expected Outputs:**
- Set 2 fractures disappear from 3D view
- Sets 1 and 3 remain visible
- Toggle set 2 back on: fractures reappear in same positions

**Pass Criteria:**
- Visual change occurs within 500 ms of toggle
- Correct set is hidden/shown
- Toggle state is preserved across save/load

**Test File:** `tests/gui/test_layers.py::TestLayerToggling`

---

### UI-07: Object Selection

**Description:** Verify that clicking on a fracture in the 3D view selects it and displays its properties.

**Inputs:**
- DFN with fractures visible in 3D viewport
- Click on a specific fracture

**Expected Outputs:**
- Clicked fracture is highlighted (different color or outline)
- Properties panel updates to show selected fracture's centroid, orientation, area, aperture
- Clicking empty space deselects

**Pass Criteria:**
- Correct fracture selected (verify by checking displayed centroid)
- Highlight visible
- Properties panel updates within 200 ms

**Test File:** `tests/gui/test_selection.py::TestObjectSelection`

---

### UI-08: Save Project from UI

**Description:** Verify end-to-end save workflow triggered from the UI.

**Inputs:**
- Create/modify a project in the UI
- File > Save As, choose a path
- Close the application
- Reopen the application
- File > Open, choose the saved file

**Expected Outputs:**
- Saved file is valid (as in PR-02)
- Reopened project matches the state at save time
- Modified indicator (e.g., asterisk in title bar) cleared after save

**Pass Criteria:**
- Full data integrity after reopen (as in PR-08)
- No "unsaved changes" warning on close after save
- Modified indicator behavior correct

**Test File:** `tests/gui/test_save.py::TestSaveFromUI`

---

## Test Execution Order

Tests should be run in the following order during CI or full-suite runs:

1. **MG** tests first (no dependencies beyond math library)
2. **SV** tests second (depend on math + fracture generation)
3. **PR** tests third (depend on math + generation + persistence)
4. **UI** tests last (depend on everything above + GUI framework)

Within each category, tests are independent except where noted.

---

## Test Environment Requirements

- Python version: as specified in `pyproject.toml` (minimum 3.10)
- Required packages: `pytest`, `pytest-qt` (for GUI tests), `numpy`, `scipy` (for statistical tests)
- GUI tests require a display server (Xvfb on Linux CI, or equivalent headless setup)
- Tolerance constants:
  - Geometric tolerance: `1e-10` (positions, distances, areas)
  - Angular tolerance: `1e-8` radians
  - Statistical significance: `alpha = 0.05` unless otherwise specified

---

## Continuous Integration

All acceptance tests (MG, SV, PR) must pass on every push to `main` and on every pull request. GUI tests (UI) run on merge to `main` and nightly.

A test failing in CI blocks the release for the corresponding milestone.

---

## Test Coverage Targets

| Milestone | Minimum Coverage |
|-----------|-----------------|
| M0        | 90% (math module) |
| M1-M5     | 85% (core library) |
| M6-M10    | 80% (full application) |
| M11       | 85% (full application, release quality) |

---

## Change Log

| Date       | Change                                          |
|------------|-------------------------------------------------|
| 2026-08-04 | Initial acceptance test specifications created. |
