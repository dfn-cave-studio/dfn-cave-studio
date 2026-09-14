# DFN Cave Studio — Scientific Specification

**Document Version:** 1.2.0 (M10 review draft)
**Status:** Authoritative Reference
**Scope:** All scientific methods, algorithms, formulae, and validation approaches

## M10 Conditional explicit DFN (v0.10.0)

For every MODELED_VALUE voxel and joint set, M10 independently computes target area `A_t=P32·V` and expected count `λ=A_remaining/(πE[R²])`, then samples `N~Poisson(λ)`. It never substitutes `π(E[R])²`. TRUE_ZERO generates zero fractures; NO_DATA, OUTSIDE_MODEL, EXCAVATION, and cells without a reliable Domain/Set Fisher model generate none.

M10 multiscale generation partitions each supported radius distribution into SMALL, MEDIUM, and LARGE by the deterministic area-weighted CDF `F_A(r)=E[R² I(R≤r)]/E[R²]`. Defaults place approximately 10%, 60%, and 30% of target P32 in the three classes for continuous distributions. Poisson counts and radii are sampled directly within enabled classes; disabled class P32 is retained per voxel and set as `P32_subgrid`. Fixed-radius distributions are degenerate and are classified without inventing unattainable shares. These thresholds are numerical resolution recommendations, not universal geological classes.

Stochastic centres are uniform within their source voxel. Directions use the saved Domain/Set Fisher mean and Kappa with an explicit seed; sizes use the saved FIXED, UNIFORM, TRUNCATED_LOGNORMAL, TRUNCATED_POWER_LAW, or TRUNCATED_EXPONENTIAL model and preserve source provenance. EXPERIMENTAL models require explicit confirmation.

Calibration FULL_ORIENTATION observations may create one conditioned disc each. The measured point lies in the disc plane and inside its radius; Validation and DIP_ONLY observations never receive fabricated conditioning directions. Conditioned original area is deducted before stochastic Poisson sampling. Parameterized deterministic discs are seed-independent and affect the random budget only under an explicit option with a supplied set ID.

Generation-boundary clipping preserves both `original_area` and polygon `clipped_area`. M10 centre assignment is labelled `CENTER_ASSIGNED_PRELIMINARY`; exact local P32 from fracture–voxel intersection and second voxelization are published in M11.1, while fracture connectivity remains later M11 work.

## M9 Local DFN parameter field (v0.9.0)

M9 samples each Formal borehole trace in fixed-length or structural-domain intervals. Adjacent intervals are half-open and the final interval is end-inclusive. For valid length `L`, `P10=N/L`; `N=0` with valid support is TRUE_ZERO, while absent support is NO_DATA.

For joint-set normal `n` and local trajectory direction `u`, `P10 = P32 E(|n·u|)`. Curved holes use actual trajectory segments. Calibration data are fitted by `P32_hat = N / sum_j(L_j E(|n·u_j|))`. Fisher expectations use an explicit seed. Exposure below the threshold yields LOW_OBSERVABILITY and no unstable P32. Validation holes are evaluated only after fitting.

GLOBAL_CONSTANT applies the fitted domain/set value. IDW uses anisotropically scaled 3D distance, exact observation recovery, neighbor/radius limits, and no cross-domain interpolation. Unsupported cells remain NO_DATA unless an explicit provenance-labelled fallback is enabled.

Ordinary Kriging is available as a separately selected, three-dimensional isotropic estimator. For local Calibration samples `x_i`, prediction at `x_0` solves
`sum_j(w_j gamma(x_i-x_j)) + mu = gamma(x_i-x_0)` with `sum_j(w_j)=1`. Spherical, exponential, and Gaussian semivariograms are supported with either manually validated nugget/sill/range values or an experimental-variogram fit. Models and searches are isolated by structural domain; the current locked borehole Holdout is authoritative at calculation time, so Validation intervals never enter fitting. Equal-distance neighbours are ordered deterministically by distance and original sample index. Kriging variance is conditional on the chosen/fitted semivariogram and is not total geological uncertainty.

Generic continuous interval parameters use the long-table contract `borehole_id, from_depth, to_depth, parameter_name, value, unit` plus optional source and quality fields. Interval midpoints are located on the computed three-dimensional borehole trajectory, not by treating measured depth as elevation. P32, UCS, joint spacing, and joint density are non-negative; RQD and RMR are bounded to `[0,100]`. Out-of-range predictions are either rejected or clipped with an explicit voxel-count and magnitude audit. RMR system/version interpretation remains the data owner's provenance responsibility. Categorical rock classes are not interpolated as continuous values. The first implementation is isotropic only: anisotropic, co-kriging, indicator, and non-stationary Kriging are not implemented.

For main-P32 bounds auditing, a **set-voxel count** is one joint-set prediction at one spatial voxel, whereas a **voxel count** is the number of unique spatial voxels affected by one or more adjusted set predictions. The audit records both counts, the minimum and maximum of every finite pre-adjustment prediction, and the total value added by clipping. With the reject policy, the rejected set prediction is omitted from `p32_total`; other valid set predictions at that voxel remain in the sum. The aggregate cell is `NO_DATA` only if no valid set prediction remains.

Size fitting accepts only radius, diameter, trace length, or mapped length. Without measurements, source is ASSUMED or USER_DEFINED; aperture, RQD, and set ID are never size. Supported distributions are fixed, uniform, truncated lognormal, truncated power law, and truncated exponential, with consistent `E[R]`, `E[R²]`, and seeded sampling.

Automatic truncated-distribution fitting in v0.9.0-M9 is **EXPERIMENTAL**. Truncation bounds use the sample minimum and maximum, and the truncated-lognormal candidate uses moment-based initialization rather than a full truncated-likelihood optimization. Candidate records persist sample count, convergence state, optimizer message, and an AIC/BIC parameter count that includes both truncation bounds. Failed optimizations retain an explicit failed status without publishing fitted parameters. The demonstration fixed radius of 2 m remains explicitly **ASSUMED**.

The first voxelization is an input parameter field, not an explicit DFN. Cell state distinguishes OUTSIDE_MODEL, NO_DATA, TRUE_ZERO, and MODELED_VALUE; `P32_total` equals the sum of valid set-level P32 values.
**Last Updated:** 2026-08-04

---

## Table of Contents

1. [Coordinate System and Units](#1-coordinate-system-and-units)
2. [Orientation Representation](#2-orientation-representation)
3. [Fisher Distribution](#3-fisher-distribution)
4. [Fracture Geometry](#4-fracture-geometry)
5. [P32 Definition and Calculation](#5-p32-definition-and-calculation)
6. [P10 and Scanline Sampling](#6-p10-and-scanline-sampling)
7. [Fracture-Fracture Intersection](#7-fracture-fracture-intersection)
8. [Fracture-Voxel Intersection](#8-fracture-voxel-intersection)
9. [Connectivity Analysis](#9-connectivity-analysis)
10. [Fragmentation Analysis](#10-fragmentation-analysis)
11. [RQD and Fracture Frequency](#11-rqd-and-fracture-frequency)
12. [Mechanical Properties](#12-mechanical-properties)
13. [Validation Methodology](#13-validation-methodology)
14. [Known Limitations](#14-known-limitations)
15. [References](#15-references)

---

## 1. Coordinate System and Units

### 1.1 Primary Coordinate System

The software uses a **right-handed Cartesian coordinate system**:

| Axis | Direction        | Description                         |
|------|------------------|-------------------------------------|
| X    | Easting          | Positive eastward                   |
| Y    | Northing         | Positive northward                  |
| Z    | Elevation        | Positive upward                     |

This convention is consistent with standard mine surveying practice and follows the right-hand rule: rotating from +X toward +Y produces a positive +Z (thumb).

### 1.2 Angular Conventions

| Quantity          | Range        | Convention                                      |
|-------------------|-------------|--------------------------------------------------|
| Dip direction     | [0, 360)    | Degrees clockwise from true north (Y-axis)       |
| Dip               | [0, 90]     | Degrees downward from horizontal (XY-plane)      |
| Strike            | [0, 360)    | Right-hand rule: strike + 90 = dip direction     |
| Internal trig     | radians     | All `sin`, `cos`, `tan`, `arctan2` in radians    |
| UI and file I/O   | degrees     | Converted at module boundaries                   |

### 1.3 Unit System

All internal computations use **SI units** (meters). Conversion to/from other units occurs at I/O boundaries only.

| Quantity       | Internal Unit     | Symbol     |
|----------------|-------------------|------------|
| Length         | meter             | m          |
| Area           | square meter      | m²         |
| Volume         | cubic meter       | m³         |
| Angle (internal)| radian            | rad        |
| Angle (I/O)    | degree            | °          |
| P10 intensity  | per meter         | m⁻¹        |
| P21 intensity  | meter per meter   | m/m (or m²/m²) |
| P32 intensity  | square meter per cubic meter | m²/m³ |
| P33 intensity  | cubic meter per cubic meter | m³/m³ (dimensionless) |

### 1.4 Conversion Functions

```text
degrees_to_radians(deg) = deg * pi / 180
radians_to_degrees(rad) = rad * 180 / pi
```

All angle conversions must use `math.radians()` / `math.degrees()` or the equivalent `numpy` functions, never hand-coded constants, to avoid loss of precision.

---

## 2. Orientation Representation

### 2.1 Dip Direction / Dip Convention

A fracture plane is oriented by two angles:

- **Dip direction** `phi` (often written as `alpha_d`): the compass bearing of the steepest line on the plane, measured clockwise from true north (Y-axis), in degrees, range [0, 360).
- **Dip** `theta` (often written as `beta_d`): the angle the plane makes with the horizontal, measured in the dip direction, in degrees, range [0, 90].

This is the standard convention in engineering geology (ISRM, 1978).

### 2.2 Plane Normal Vector

Given dip direction `phi` (degrees) and dip `theta` (degrees), the unit normal vector `n = (n_x, n_y, n_z)` pointing **upward** (i.e., out of the upper half-space) is computed as:

Convert to radians:

```
phi_rad = phi * pi / 180
theta_rad = theta * pi / 180
```

Then:

```
n_x = -sin(theta_rad) * sin(phi_rad)
n_y = -sin(theta_rad) * cos(phi_rad)
n_z =  cos(theta_rad)
```

**Derivation:** The upward normal points opposite to the dip vector. The dip vector in Cartesian coordinates is `(sin(theta)*sin(phi), sin(theta)*cos(phi), -cos(theta))`. Negating gives the upward normal.

### 2.3 Normal Vector to Dip Direction / Dip (Inverse Conversion)

Given a unit normal vector `n = (n_x, n_y, n_z)` with `n_z >= 0` (upper hemisphere convention), the dip and dip direction are recovered as:

```
theta_rad = arccos(n_z)               // dip
phi_rad   = arctan2(-n_x, -n_y)       // dip direction
```

Where `arctan2(y, x)` returns the angle in the correct quadrant. The result `phi_rad` is normalized to [0, 2*pi).

If `n_z < 0` (lower hemisphere), the vector is first negated: `n = -n`, which flips it to the upper hemisphere. This is equivalent to using the opposite pole in stereographic projection.

**Pole vector convention:** For stereographic projection, the lower-hemisphere pole `p = -n` is used, with `p_z < 0`. The pole vector points downward and plots on the lower hemisphere of the Schmidt net.

### 2.4 Rotation Matrices

The orientation of a fracture plane in 3D can be expressed as a rotation from the reference horizontal plane (XY-plane with normal along +Z).

The rotation matrix that takes the reference plane to the fracture plane orientation is:

```
R = R_z(phi_rad) * R_y(theta_rad)
```

Where:

```
        [cos(alpha)  -sin(alpha)  0]
R_z(a) = [sin(alpha)   cos(alpha)  0]
        [   0            0         1]

        [cos(beta)   0   sin(beta)]
R_y(b) = [   0       1      0     ]
        [-sin(beta)  0   cos(beta)]
```

This rotates first by dip `theta` about the Y-axis, then by dip direction `phi` about the Z-axis. The resulting matrix maps the reference normal `(0, 0, 1)` to the fracture normal `n`.

### 2.5 Stereographic Projection

For equal-angle (Wulff) projection of a lower-hemisphere pole `p = (p_x, p_y, p_z)` with `p_z < 0`:

```
x_proj = p_x / (1 - p_z)
y_proj = p_y / (1 - p_z)
```

For equal-area (Schmidt) projection:

```
r = sqrt(2 / (1 - p_z))
x_proj = p_x * r
y_proj = p_y * r
```

The software uses **equal-area (Schmidt) projection** by default, consistent with structural geology convention for contouring pole densities (see Section 3).

### 2.6 Strike

Strike is derived from dip direction by the right-hand rule:

```
strike = dip_direction - 90
```

Normalized to [0, 360). The strike direction is such that the plane dips to the right when looking along the strike direction.

---

## 3. Fisher Distribution

The Fisher distribution (Fisher, 1953) is the fundamental probability distribution for orientations on the unit sphere. It is the spherical analogue of the isotropic Gaussian distribution on the circle (von Mises distribution).

### 3.1 Probability Density Function

For a unit vector `x` on the sphere, given a mean direction unit vector `mu` and concentration parameter `kappa >= 0`:

```
f(x; mu, kappa) = [kappa / (4 * pi * sinh(kappa))] * exp(kappa * (mu . x))
```

Where `mu . x` is the dot product (cosine of the angular deviation).

Equivalently, in terms of the angular deviation `delta` from the mean:

```
f(delta; kappa) = [kappa / (2 * pi * (e^kappa - e^(-kappa)))] * exp(kappa * cos(delta)) * sin(delta)
```

For `delta` in [0, pi].

**Special cases:**
- `kappa = 0`: Uniform distribution on the sphere (all orientations equally likely)
- `kappa → infinity`: Concentrated at the mean direction (deterministic)

### 3.2 Expected Angular Dispersion

The expected cosine (mean resultant length) is:

```
R_bar = E[cos(delta)] = coth(kappa) - 1/kappa
```

The approximate standard deviation (in radians) for large kappa (`kappa > 5`):

```
sigma_rad ≈ 1 / sqrt(kappa)
```

The spherical standard deviation (Fisher dispersion parameter):

```
delta_68 ≈ arccos(1 - log(0.32) / kappa)   // cone containing ~68% of vectors
```

The aperture of the cone containing probability `p`:

```
cos(alpha_p) = 1 + (1/kappa) * log(1 - p)
```

### 3.3 Sampling Algorithm

Implemented using the method of Woodcock (1977), also described in Fisher, Lewis, and Embleton (1987, Section 3.3.3).

**Algorithm: Generate a Fisher-distributed unit vector with mean `mu` and concentration `kappa`:**

1. Generate `v_1 ~ Uniform(0, 1)`, compute the azimuthal angle `tau`:
   ```
   tau = 2 * pi * v_1
   ```

2. Generate `v_2 ~ Uniform(0, 1)`, compute the colatitude `delta` (deviation from mean):
   ```
   cos_delta = 1 + (1/kappa) * log(1 - v_2 + v_2 * exp(-2*kappa))
   cos_delta = clamp(cos_delta, -1, 1)
   sin_delta = sqrt(1 - cos_delta^2)
   ```

3. Construct a vector in local coordinates (mean direction aligned with +Z):
   ```
   x_local = sin_delta * cos(tau)
   y_local = sin_delta * sin(tau)
   z_local = cos_delta
   ```

4. Rotate to world coordinates using a rotation matrix that maps `(0,0,1)` to `mu`:
   - Construct an orthonormal basis `{u, v, mu}` where `u` and `v` span the plane perpendicular to `mu`.
   - ```
     result = x_local * u + y_local * v + z_local * mu
     ```

**Orthonormal basis construction:** Given mean direction `mu = (m_x, m_y, m_z)`:

```
If |m_z| < 0.999:
    u_raw = (-m_y, m_x, 0)    // vector perpendicular to mu
    u = u_raw / ||u_raw||
Else (mu ~ (0,0,+-1)):
    u = (1, 0, 0)
v = mu × u                      // cross product
```

**Implementation note:** All generation uses `numpy.random.Generator` with a caller-supplied `random_seed: int` parameter. No global RNG state is ever used. This ensures full reproducibility (see Section 13.4).

**Reference:** Fisher, N.I., Lewis, T., and Embleton, B.J.J. (1987). *Statistical Analysis of Spherical Data*. Cambridge University Press.

### 3.4 Parameter Estimation from Data

Given a sample of N unit vectors `{x_i}`, the maximum likelihood estimate of the mean direction is:

```
R = || sum(x_i) ||                   // resultant length
mu_hat = sum(x_i) / R                 // normalized resultant
```

The MLE of kappa is approximated by:

```
For N <= 15:
    kappa_hat = (N - 1) / (N - R)

For N > 15:
    kappa_hat = (N - 2) / (N - R)    // Mardia and Jupp (2000) correction
```

---

## 4. Fracture Geometry

### 4.1 Baecher Disk Model

The Baecher disk model (Baecher et al., 1977) represents each fracture as a planar circular disk in 3D space, defined by:

- **Center point** `c = (c_x, c_y, c_z)` in world coordinates
- **Radius** `r` (a scalar, meters)
- **Orientation** (see Section 2), which defines the plane's unit normal `n`

The fracture plane is the set of points `p` satisfying:

```
n . (p - c) = 0
||p - c|| <= r
```

Where `||.||` denotes Euclidean distance. This model is the most widely used in DFN analysis (Dershowitz and Einstein, 1988).

**Radius distribution:** Radii are typically drawn from a power-law (Pareto) or lognormal distribution. The default distribution is power-law (Dershowitz, 1984):

```
f(r) = a * D * r_min^a / r^(a+1)    for r >= r_min
```

Where `a` is the shape parameter (typically 2.5-4.0 for natural fracture sets) and `r_min` is the minimum radius (truncation).

### 4.2 Polygon Model

The polygon model represents each fracture as a convex polygon in its local plane, defined by an ordered list of vertices. This is useful for deterministic fractures mapped from borehole imagery or for fractures bounded by lithological contacts (Ivanova et al., 2014).

Properties:
- **Vertices:** N points in world coordinates, all coplanar (to tolerance 1e-9 m)
- **Normal:** Computed from the first three non-collinear vertices via Newell's method for numerical stability
- **Area:** Computed via the shoelace formula in the projected 2D plane

**Area computation (shoelace formula, projected):**

1. Project 3D vertices onto the best-fit 2D plane (choose the coordinate plane with the largest component of the normal to minimize projection distortion).
2. Apply the shoelace formula in 2D:

```
Area = 0.5 * |sum_{i=0}^{N-1} (x_i * y_{i+1} - x_{i+1} * y_i)|  // indices modulo N
```

### 4.3 Finite Rectangular Plane Model

A special case of the polygon model where the fracture is an axis-aligned rectangle in its local plane, defined by half-width `h_w` and half-height `h_h` (both in meters). The four corner points in local coordinates `(u, v)` are `(+-h_w, +-h_h)`.

This model is used for:
- Known geological structures (faults, veins) with measured extents
- Validation test cases with analytical solutions
- Export to 3DEC where joints are specified as finite planes

**Area:** `Area = 4 * h_w * h_h`

### 4.4 Area Calculation

Disk model area:

```
A = pi * r^2
```

Polygon model area: See Section 4.2 shoelace formula.

Rectangle model area:

```
A = width * height
```

### 4.5 Boundary Intersection

For fractures intersecting a bounding volume (e.g., a rectangular domain or cylindrical cave column), the fracture is clipped to the intersection of the infinite fracture plane with the bounding volume.

**Algorithm (rectangular domain):**

1. Compute the intersection of the infinite plane with the 12 edges of the box.
2. Collect all intersection points that lie within the edge segment and within the fracture bounds (disk radius or polygon).
3. Order intersection points angularly in the fracture plane to form a polygon.
4. Compute the clipped area using the shoelace formula.

This produces the "in-domain fracture area" used for P32 calculation (Section 5).

---

## 5. P32 Definition and Calculation

### 5.1 Definition

P32 is the **total fracture area per unit volume** (Dershowitz and Herda, 1992):

```
P32 = (1 / V_domain) * sum_{i=1}^{N} A_i
```

Where:
- `V_domain` is the volume of the domain of interest (m³)
- `A_i` is the area of fracture `i` within the domain (m²)
- `N` is the number of fractures intersecting the domain

**Units:** m²/m³ = m⁻¹

P32 is the primary intensity measure used in DFN modeling because it is (a) independent of orientation bias, (b) directly related to hydraulic and mechanical properties of the fractured rock mass, and (c) invariant under translation and rotation of the coordinate system.

### 5.2 Relationship to Other Intensity Measures

| Measure | Name | Definition | Units |
|---------|------|-----------|-------|
| P10 | Linear intensity | Fractures per meter along a scanline | m⁻¹ |
| P21 | Areal intensity | Fracture trace length per unit area on a plane | m/m² |
| P32 | Volumetric intensity | Fracture area per unit volume | m²/m³ |
| P33 | Volumetric density | Fracture volume per unit volume | m³/m³ |

For a population of parallel disks with mean radius `r_mean`:

```
P32 = pi * r_mean^2 * P30     // P30 = number density (m⁻³)
```

### 5.3 Target P32 Convergence Algorithm

When generating a stochastic DFN to match a target P32, the software uses the following iterative algorithm:

**Algorithm: Rejection-sampling DFN generation with P32 targeting**

```
Input: target_P32, domain_volume, radius_distribution, orientation_distribution, tolerance, max_iterations

1. n_guess = ceil(target_P32 * domain_volume / (pi * r_mean^2))   // initial estimate
2. For iteration = 1 to max_iterations:
   a. Generate n_guess fractures (centers, orientations, radii)
   b. Clip all fractures to domain, compute in-domain areas A_i
   c. current_P32 = (1/V_domain) * sum(A_i)
   d. error = |current_P32 - target_P32| / target_P32
   e. If error < tolerance: return generated DFN
   f. n_guess *= target_P32 / current_P32   // proportional correction
3. If max_iterations reached: warn user, return best DFN
```

The proportional correction in step 2f is justified because P32 scales approximately linearly with the number of fractures, assuming the fracture population characteristics (radius distribution, orientation distribution) are stationary.

**Convergence tolerance:** Default 0.05 (5%), configurable.

**Reference:** Adapted from the FracMan P32 generation methodology described in Dershowitz et al. (1998).

### 5.4 Spatial P32 Field

For heterogeneous DFNs, P32 is computed on a regular grid of subdomains (voxels) to produce a spatially varying intensity field. The calculation for voxel `k` with volume `V_k` is:

```
P32_k = (1 / V_k) * sum_{i in voxel_k} A_{i,k}
```

Where `A_{i,k}` is the area of fracture `i` within voxel `k` (see Section 8).

---

## 6. P10 and Scanline Sampling

### 6.1 P10 Definition

P10 is the number of fractures intersected per unit length along a linear sample (scanline or borehole):

```
P10 = N_intersections / L_scanline
```

Where `N_intersections` is the count of fracture intersections and `L_scanline` is the scanline length.

**Units:** m⁻¹

### 6.2 Terzaghi Correction for Orientation Bias

Scanline sampling preferentially intersects fractures oriented perpendicular to the scanline. Fractures parallel to the scanline are undersampled. The **Terzaghi correction** (Terzaghi, 1965) compensates for this bias.

For a scanline with direction unit vector `s`, a fracture with unit normal `n` (upper hemisphere), the probability of intersection is proportional to:

```
cos(alpha) = |s . n|
```

Where `alpha` is the acute angle between the scanline direction and the fracture normal. The corrected count is:

```
N_corrected = sum_{i in intersections} (1 / |s . n_i|)
```

**Weight clipping:** To avoid infinite weights for fractures nearly parallel to the scanline (`|s . n| ≈ 0`), a minimum weight cap is applied:

```
w_i = 1 / max(|s . n_i|, cos_max_angle)
```

Where `cos_max_angle = cos(max_alpha)` and `max_alpha` defaults to 85 degrees (cos 85° ≈ 0.087). Fractures at angles > 85° are given the maximum weight and a warning is emitted about potential undercounting in sets subparallel to the scanline.

**Corrected P10:**

```
P10_corrected = (1 / L) * sum(w_i)
```

### 6.3 Virtual Scanline Methodology

The software simulates physical scanline sampling by casting rays (virtual scanlines) through a generated DFN and recording intersections. This is used for:

1. **Validation:** Compare virtual scanline P10 against the known P32 of the generated DFN
2. **Borehole simulation:** Predict what a real borehole would observe
3. **Forward modeling:** Relate observable 1D measures (P10) to true 3D intensity (P32)

**Algorithm: Virtual scanline intersection**

```
Input: scanline origin o, direction s (unit vector), fracture_list, scanline_length L

1. intersections = []
2. For each fracture f in fracture_list:
   a. Compute the intersection point of the infinite line with the fracture plane
   b. If no intersection (line parallel to plane): continue
   c. Parameter t such that p = o + t*s lies on the fracture plane:
        t = (n . (c - o)) / (n . s)
   d. If t < 0 or t > L: continue (outside scanline segment)
   e. p = o + t*s
   f. If ||p - c|| <= r (for disk model): append (t, f) to intersections
   g. If p is inside polygon (for polygon model): append (t, f) to intersections
3. Sort intersections by t
4. Return intersections
```

The line-plane intersection parameter `t` is derived from:

```
o + t*s lies on plane: n . (o + t*s - c) = 0
=> n . o + t*(n . s) - n . c = 0
=> t = (n . (c - o)) / (n . s)
```

Reference: Priest (1993), *Discontinuity Analysis for Rock Engineering*.

### 6.4 Virtual Borehole Methodology

Identical to virtual scanline but with cylindrical borehole geometry. The borehole is modeled as a finite cylinder with radius `r_bh`. A fracture intersects the borehole if:

1. The fracture plane intersects the borehole axis (same calculation as scanline), AND
2. The intersection point `p` and the fracture center `c` satisfy the disk radius check: `||p - c|| <= r_fracture + r_bh` (conservative, includes near-misses where the borehole wall contacts the fracture edge).

For borehole televiewer simulation, the apparent dip and dip direction of each intersected fracture are computed in the borehole reference frame.

### 6.5 P21 from Virtual Windows

P21 (areal intensity) is the total trace length per unit area on a sampling window plane:

```
P21 = sum(trace_lengths) / window_area
```

A virtual window is a planar rectangle placed within the domain. The intersection of each fracture with this plane produces a line segment (trace). The trace length is computed as the length of the intersection segment of the fracture disk/polygon with the window plane, clipped to the window boundary.

**Conversion from P21 to P32 (stereological relation, Underwood 1970):**

For a population of parallel fractures:

```
P32 = P21
```

For randomly oriented fractures:

```
P32 = (4/pi) * P21
```

More generally, for a Fisher distribution with concentration `kappa`:

```
P32 = C(kappa) * P21
```

Where `C(kappa)` is a correction factor that accounts for orientation bias. This factor is computed numerically by the software via Monte Carlo integration of the expected projected area.

**Reference:** Underwood, E.E. (1970). *Quantitative Stereology*. Addison-Wesley. See also Dershowitz and Einstein (1988).

---

## 7. Fracture-Fracture Intersection

### 7.1 Line of Intersection Between Two Planes

Given two fractures with normals `n_1` and `n_2`, and centers `c_1` and `c_2`, the line of intersection of their infinite planes has direction:

```
d = n_1 × n_2    // cross product
```

If `||d|| < epsilon` (planes are parallel or subparallel, where `epsilon = 1e-9`), there is no intersection line, and the fractures do not intersect (unless they are coplanar and overlapping -- this degenerate case is handled by a separate coplanarity check).

To find a point `p_0` on the intersection line, solve the linear system:

```
n_1 . p_0 = D_1    where D_1 = n_1 . c_1
n_2 . p_0 = D_2    where D_2 = n_2 . c_2
d . p_0 = 0        // anchor to the plane through origin perpendicular to d
```

This is a 3x3 linear system solved by Gaussian elimination or LU decomposition. The intersection line is then parameterized as:

```
p(t) = p_0 + t * d_hat     where d_hat = d / ||d||
```

### 7.2 Intersection Segment Within Finite Fractures

For the infinite intersection line to represent a real fracture-fracture intersection, it must intersect both finite fracture bodies (disks or polygons).

**For two disk fractures (radii r_1, r_2):**

1. Find the interval `[t_1_min, t_1_max]` where the line is within disk 1:
   - The line intersects disk 1 where `||p(t) - c_1|| <= r_1`
   - This gives a quadratic in `t`: `||p_0 + t*d_hat - c_1||^2 = r_1^2`
   - Solve for `t`: `a*t^2 + b*t + c = 0` where:
     - `a = ||d_hat||^2 = 1`
     - `b = 2 * d_hat . (p_0 - c_1)`
     - `c = ||p_0 - c_1||^2 - r_1^2`
   - Roots: `t = (-b +- sqrt(b^2 - 4ac)) / 2a`
   - If discriminant < 0: no intersection with this disk, return None
   - `[t_1_min, t_1_max]` = sorted roots

2. Similarly find `[t_2_min, t_2_max]` for disk 2.

3. The intersection segment is `[max(t_1_min, t_2_min), min(t_1_max, t_2_max)]`.
   - If `t_max < t_min`: no intersecting segment
   - Otherwise: intersection segment with endpoints `p(t_min)` and `p(t_max)`
   - Length of intersection segment: `|t_max - t_min|`

**For polygon fractures:** The intersection of the line with the polygon is found by computing the intersection of the line with each polygon edge, collecting the two extreme intersection points, and verifying they lie within the polygon. This uses the ray-casting / half-plane approach.

### 7.3 Intersection Detection Algorithm (Pairwise)

```
Input: fracture_1, fracture_2

1. Compute intersection line of infinite planes (Section 7.1)
2. If planes parallel: return None_intersect (or handle coplanar case)
3. Compute intersection segment (Section 7.2)
4. If no segment or segment_length ≈ 0: return None_intersect
5. Return Intersection(start_point, end_point, length, fracture_1_id, fracture_2_id)
```

### 7.4 Spatial Indexing for N-Body Intersection

Naive pairwise checking of N fractures requires O(N²) comparisons, which is infeasible for large DFNs (N > 10,000).

The software uses an **R-tree** spatial index (via `rtree` or `scipy.spatial`) to accelerate intersection detection:

1. Insert the axis-aligned bounding box (AABB) of each fracture into the R-tree.
2. For each fracture, query the R-tree for all fractures whose AABB overlaps its own.
3. Perform exact intersection testing only on the candidate pairs.

**Bounding box computation (disk):**

```
bbox = [c_x - r, c_x + r, c_y - r, c_y + r, c_z - r, c_z + r]
```

This uses a conservative cubic AABB. The optimal (tighter) AABB for a disk uses the plane orientation to compute the actual projected extents, reducing false-positive candidate pairs.

**Reference:** Guttman (1984), "R-trees: A Dynamic Index Structure for Spatial Searching." (Also: Samet (2006), *Foundations of Multidimensional and Metric Data Structures*.)

---

## 8. Fracture-Voxel Intersection

### 8.1 Voxel Grid Definition

The domain is subdivided into a regular 3D grid of voxels (volume elements). Each voxel is an axis-aligned rectangular prism (typically a cube) defined by:

- Corner `(x_min, y_min, z_min)`
- Size `(dx, dy, dz)`
- Grid dimensions `(N_x, N_y, N_z)`

**Storage:** The software uses a sparse/chunked voxel representation (via `zarr` or HDF5) to handle potentially large grids (e.g., 1000x1000x1000) where most voxels contain no fractures.

### 8.2 Candidate Search

Given a fracture (center `c`, normal `n`, radius `r` for disk model), the candidate voxels that may intersect it are those whose AABB overlaps the expanded AABB of the fracture.

**Algorithm:**

1. Compute the fracture AABB: `[c_x - r, c_x + r] × [c_y - r, c_y + r] × [c_z - r, c_z + r]`
2. Map AABB to voxel index range: `[i_min..i_max] × [j_min..j_max] × [k_min..k_max]`
3. For tighter culling, test whether the candidate voxel's AABB intersects the fracture plane slab (plane extruded by `+-epsilon` outward) -- this is the separating axis theorem (SAT) applied to one axis (the plane normal).

### 8.3 Exact Intersection Testing

**Disk fracture vs. voxel:**

A voxel (AABB) intersects the fracture disk if and only if:

1. The AABB intersects the infinite plane (distance from voxel center to plane <= half-diagonal along normal), AND
2. The AABB intersects the circular disk (the projection of the voxel onto the plane intersects the disk).

The exact test uses the method of Eberly (2008) for AABB-disk intersection:

1. Test if any voxel corner is within distance `r` of the fracture center in the fracture plane: compute the signed distance of each of the 8 voxel corners from the plane, and the in-plane distance from the fracture center for corners on both sides of the plane.
2. Alternatively, project the voxel onto the fracture plane as an axis-aligned (in 2D plane coordinates) rectangle, then test rectangle-circle intersection.
3. For robust results, use the SAT variant: test separation along (a) plane normal, (b) cross products of edge directions with plane normal.

**Implementation note:** A conservative approach is used for performance: a voxel is flagged as "intersecting" if the minimum distance from the voxel center to the fracture plane is less than the half-diagonal of the voxel and the in-plane distance condition is satisfied. This is exact for cubic voxels and adequate for scientific analysis.

**Reference:** Eberly, D. (2008). "Intersection of a Disk and a Box." Geometric Tools (www.geometrictools.com). See also: Ericson, C. (2004). *Real-Time Collision Detection*. Morgan Kaufmann.

### 8.4 Intersection Area Calculation

The area of fracture `f` within voxel `v` is computed by:

1. Computing the convex polygon resulting from clipping the fracture shape (disk or polygon) to the voxel's six half-spaces.
2. Computing the area of the resulting polygon (shoelace formula, Section 4.2).

**For disk fractures:** The clipped polygon has up to 7 vertices (the six faces of the voxel can each contribute one edge, plus the circular boundary).

**For polygon fractures:** Sutherland-Hodgman polygon clipping algorithm (Sutherland and Hodgman, 1974) against each of the six voxel face planes.

The intersection area is used for:
- Spatially varying P32 calculation (Section 5.4)
- Voxel-based fracture intensity mapping
- Export of effective permeability tensors

---

### 8.5 M11.1 Exact Second-Voxelization Contract (normative)

**Published scientific scope:** `v0.11.0-M11.1` implements this contract as the
bounded **M11.1 Exact Second Voxelization** phase. Publication of M11.1 does not
mean that the complete M11 roadmap is finished.

The earlier conservative intersection description is retained for historical
context only. M11.1 uses the following normative definition for explicit
circular fractures:

```
A_fv = Area(D_f intersect B_v intersect GenerationDomain)
P32_explicit_intersection(v, set) = sum_f(A_fv) / voxel_volume
P32_total = P32_explicit_intersection + P32_subgrid
```

Tangency by only a point or line has zero effective area.
`p32_unresolved_orientation` remains a separate audit quantity and is not
silently treated as explicit or subgrid P32.

Candidate voxels come only from the deterministic index range of the tight disk
AABB, whose axis `j` extent is `R * sqrt(1 - n_j^2)`. The implementation never
constructs a fracture-by-all-voxels Cartesian product. Candidate, positive, and
rejected-pair counts are retained.

The disk plane is mapped to a deterministic orthonormal 2-D basis. The six AABB
half spaces clip a square containing the circle. The area shared by that convex
polygon and the true circle is then integrated with straight-line and
circular-arc boundary terms. No fixed 8-, 16-, or 32-sided approximation is
substituted for the circle. Computation uses float64, finite unit normals,
positive finite radii, and a saved scale-relative tolerance policy based on
`128 * machine_epsilon * coordinate_scale`.

Voxel ownership is half-open (`[minimum, maximum)`) on all internal faces, with
the outer grid maximum included in the final voxel. A disk exactly coplanar with
a shared X, Y, or Z face is assigned once to the minimum-inclusive voxel on the
positive side.

Positive intersections are stored as stable columns sorted by
`(voxel_flat_index, fracture_ordinal)`:

```
fracture_ordinal
voxel_flat_index
intersection_area  # float64, m^2
```

Per-set and aggregate voxel arrays retain explicit-intersection P32, subgrid
P32, total P32, intersecting-fracture count, and semantic `cell_state`. For each
fracture, exact area in both the Generation Domain and Voxel Analysis Domain is
computed independently. The sum assigned to analysis voxels must equal the
analysis-domain target within the stored tolerance and must not exceed the
generation-domain target. Error totals, maximum, p50/p95/p99, offending
ordinals, and difference from the M10 clipped-area cache are persisted. M10
geometry and its cache are not overwritten.

M11.1 does not implement fracture-fracture intersections, connectivity,
connected clusters, boundary spanning, flow/percolation channels, block cutting,
mechanics, formal 3DEC/PFC export, or Kriging interpolation.

---

## 9. Connectivity Analysis

### 9.1 Graph Construction

The DFN is converted to an undirected graph `G = (V, E)` where:

- **Nodes (V):** Each fracture is a node.
- **Edges (E):** An edge exists between two nodes if and only if the corresponding fractures intersect (Section 7).

**Edge weights (optional):** The length of the intersection segment normalized by the dominant fracture radius:

```
w_{ij} = L_intersection / max(r_i, r_j)
```

This weight can be interpreted as a measure of the relative importance of the connection.

### 9.2 Connected Components

Connected components are identified using depth-first search (DFS) or union-find (disjoint set) algorithm:

**Algorithm (union-find):**

```
Input: graph G = (V, E)

1. Initialize each node as its own component
2. For each edge (i, j) in E:
       union(i, j)
3. Group nodes by their root representative
4. Return list of connected components, each being a set of fracture IDs
```

**Output metrics per component:**
- Number of fractures
- Total fracture area
- Component bounding box extents
- Component effective P32

Components are sorted by size (number of fractures), and the largest component is analyzed for percolation.

**Reference:** Cormen, Leiserson, Rivest, and Stein (2009). *Introduction to Algorithms*, 3rd ed. (DFS: Section 22.3; Disjoint sets: Chapter 21).

### 9.3 Percolation Detection

A connected component **percolates** if it contains fractures that intersect (or extend across) opposite boundaries of the domain. Percolation is the critical property determining whether the DFN forms a continuous network spanning the rock mass -- essential for fluid flow and cave propagation assessment (Elmo et al., 2014).

**Algorithm: Boundary-to-boundary percolation:**

```
Input: connected_component, domain_boundaries

1. For each of the 3 axis pairs (X_min/X_max, Y_min/Y_max, Z_min/Z_max):
   a. Identify fractures in the component that intersect the lower boundary face
   b. Identify fractures in the component that intersect the upper boundary face
   c. If both sets are non-empty AND there exists a path between them
      (guaranteed since they are in the same connected component):
          percolation_axis[i] = True
2. total_percolation = any(percolation_axis)
```

**Boundary intersection test:** A fracture intersects a domain boundary face if the fracture disk/polygon has any portion lying on or within the boundary face plane (to tolerance epsilon = 1e-6 m). The test is equivalent to: the fracture shape clipped to the boundary face half-space yields a non-empty polygon.

**Percolation threshold P32:** The minimum P32 at which percolation occurs for a given fracture set (orientation distribution, radius distribution, domain size). This is estimated by generating DFNs at incrementally increasing P32 and running percolation analysis.

### 9.4 Network Connectivity Index (NCI)

The NCI (La Pointe, 2002) is a dimensionless measure of fracture network connectivity:

```
NCI = C1 + C2 + C3
```

Where:
- `C1`: Number of intersections per fracture (average degree of the graph)
- `C2`: Proportion of fractures in the percolating cluster (fractures in largest component / total fractures)
- `C3`: A measure based on the cyclomatic number (number of independent cycles)

Specifically, for a graph with `V` nodes and `E` edges:

**Average degree:**

```
d_avg = 2 * |E| / |V|
```

**Proportion in largest cluster:**

```
P_inf = |V_largest| / |V|
```

**Cyclomatic number (Betti number):**

```
beta_1 = |E| - |V| + |C|
```

Where `|C|` is the number of connected components. The cyclomatic number represents the number of independent loops (redundant flow paths).

**Reference:** La Pointe, P.R. (2002). "Derivation of fracture intensity and connectivity parameters from geologic field data for use in discrete fracture network and boundary element models." SKB Report R-02-21, Swedish Nuclear Fuel and Waste Management Co.

---

## 10. Fragmentation Analysis

### 10.1 Block Definition via Fracture Planes

The rock mass is partitioned into discrete blocks by the network of fracture planes. Each block is a convex polyhedron (or the intersection of a convex set with the domain boundary) defined by the half-spaces of the bounding fracture planes.

**Algorithm: 3D block cutting (simplified)**

```
Input: fracture_list, domain_bbox

1. Start with the domain as a single polyhedron (the bounding box)
2. For each fracture plane (in arbitrary order):
   a. For each existing block that straddles the plane:
        Split the block into two: the portion on the positive side of the plane
        (n . p > D) and the portion on the negative side (n . p < D)
   b. If a block lies entirely on one side, leave it intact
3. After all planes are processed, we have the set of all blocks
```

**Implementation approach:** The software uses a Binary Space Partition (BSP) tree approach or, for moderate fracture counts, a brute-force half-space intersection using `scipy.spatial.HalfspaceIntersection` for each subset of fracture planes that form a closed block.

**Important caveat:** General fracture networks produce non-convex blocks and blocks that may span far beyond the domain boundaries. The block-cutting algorithm is an **approximation** (see Section 10.4 for known issues). The method is most reliable for well-connected fracture networks where blocks are fully bounded by fractures (not by the domain boundary).

**Reference:** Elmouttie, M., Poropat, G., and Krahenbuhl, G. (2010). "Polyhedral modelling of rock mass structure." *International Journal of Rock Mechanics and Mining Sciences*, 47(4), 577-585.

### 10.2 Block Properties

For each identified block, the following properties are computed:

**Volume:** For a convex polyhedron defined by vertices, volume is computed using the divergence theorem (triangulating the surface and summing signed tetrahedron volumes from an interior point):

```
V = (1/6) * |sum_{faces} sum_{triangles} (v_0 × v_1) . v_2|
```

Where each face is triangulated from its centroid.

**Surface area:** Sum of face areas (computed via shoelace formula for each face).

**Equivalent sphere diameter (Deq):**

```
Deq = (6 * V / pi)^(1/3)
```

**Bounding box:** Axis-aligned and oriented minimum bounding box.

### 10.3 Block Shape Descriptors

**Elongation index:**

```
I_e = L_intermediate / L_longest
```

Where `L_longest`, `L_intermediate`, `L_shortest` are the dimensions of the oriented minimum bounding box, sorted.

**Flatness index:**

```
I_f = L_shortest / L_intermediate
```

**Sphericity (Wadell):**

```
psi = pi^(1/3) * (6 * V)^(2/3) / A
```

Where `V` is volume and `A` is surface area. Perfect sphere: `psi = 1`. Thin disks: `psi << 1`.

**Reference:** Blott, S.J. and Pye, K. (2008). "Particle shape: a review and new methods of characterization." *Sedimentology*, 55(1), 31-63.

### 10.4 Size Distribution

The block size distribution (BSD) is a fundamental output for cave mining engineering (Laubscher, 1994; Elmo et al., 2014).

**Cumulative distribution:** Blocks sorted by volume (ascending), cumulative percentage passing computed:

```
P(V <= V_i) = (sum_{j=1}^{i} V_j) / (sum_{j=1}^{N} V_j) * 100%
```

**Percentile values:**
- **D20:** Block size (equivalent diameter Deq) at which 20% of the total volume passes
- **D50:** Median block size (50% passing)
- **D80:** 80% passing

Interpolation between adjacent ranked block volumes using linear or logarithmic interpolation as appropriate.

**Fitting to distribution models:**

**Rosin-Rammler (Weibull) distribution:**

```
P(Deq <= x) = 1 - exp(-(x / x_c)^k)
```

Where `x_c` is the characteristic size and `k` is the uniformity coefficient. Parameters estimated via least-squares fit to log(1-P) vs. log(x).

**Power-law distribution (for volume):**

```
N(V > v) = C * v^(-b)
```

Where `b` is the power-law exponent. Estimated via maximum likelihood (Clauset et al., 2009) on the tail of the distribution.

**Important caveat:** Blocks intersected by the domain boundary are truncated and their true volume is underestimated. These "boundary blocks" MUST be either (a) excluded from the distribution, or (b) flagged with a warning about potential bias. The software reports both the full BSD and the "interior-only" BSD (blocks not touching the domain boundary).

### 10.5 Fragmentation Caveability Indicators

For block cave mining, the following derived metrics are computed from the BSD:

- **D50 / drawpoint_width ratio:** If `D50 < 0.15 * drawpoint_width`, the cave is expected to produce granular, flowable material
- **Fines fraction (Deq < 0.1 m):** Percentage of material below 10 cm equivalent diameter
- **Oversize fraction (Deq > 1.0 m):** Percentage of material exceeding 1 m, requiring secondary breakage

These are **engineering heuristics** based on the empirical work of Laubscher (1994) and are provided for reference. They are NOT guarantees of cave performance.

**Reference:** Laubscher, D.H. (1994). "Cave mining -- the state of the art." *Journal of the South African Institute of Mining and Metallurgy*, 94(10), 279-293.

---

## 11. RQD and Fracture Frequency

### 11.1 Definition of RQD

Rock Quality Designation (RQD) is defined by Deere (1964) as:

```
RQD = (sum of core pieces >= 100 mm length) / (total core run length) * 100%
```

RQD is a directional measure -- it depends on the borehole orientation. The software computes RQD for a given borehole trajectory by identifying fracture intersections along the borehole axis and measuring the intact lengths between consecutive intersections.

### 11.2 Relationship to P10

RQD is **related** to P10 (linear fracture frequency) but is **not equivalent**. The empirical relationship proposed by Priest and Hudson (1976) for an **assumed Poisson process of fracture spacing** (negative exponential distribution) is:

```
RQD = 100 * exp(-P10 * t) * (P10 * t + 1)
```

Where `t = 0.1 m` (the threshold length for "intact core piece").

**Inverse:** Given RQD, the implied P10 for the Poisson model is:

```
P10_implied = numerical solve of: 100 * exp(-x * 0.1) * (x * 0.1 + 1) = RQD
```

### 11.3 Conversion Limitations and Assumptions

The Priest-Hudson relationship assumes:

1. Fracture spacings follow a negative exponential distribution (Poisson process along the borehole)
2. Fractures are planar and infinite (no termination within the core)
3. The borehole is perpendicular to the fracture set (or Terzaghi-corrected)
4. No mechanical breaks or drilling-induced fractures

**The software MUST NOT:**
- Present RQD as directly convertible to/from P32 without explicit statement of assumptions
- Claim that RQD from one borehole orientation characterizes the entire rock mass
- Use RQD as an input parameter for DFN generation (it is an output metric only)

**The software MUST:**
- Report RQD alongside the borehole orientation and fracture set characteristics
- Flag when the Poisson assumption is violated (e.g., regular or clustered spacing patterns)
- Provide both the raw (uncorrected) RQD and the Terzaghi-corrected RQD for different fracture sets

### 11.4 Fracture Frequency (FF)

Linear fracture frequency (`FF = P10`) and volumetric fracture count (`JV`) are reported alongside RQD:

```
JV = sum_{sets} (1 / S_i)     // Palmstrom (1982)
```

Where `S_i` is the mean spacing of fracture set `i`. This is an alternative to RQD that does not depend on the 100 mm threshold.

**Reference:** Palmstrom, A. (1982). "The volumetric joint count -- a useful and simple measure of the degree of rock mass jointing." *Proc. 4th Int. Congress IAEG*, Delhi, V.221-V.228.

---

## 12. Mechanical Properties

### 12.1 Mohr-Coulomb Parameters

The Mohr-Coulomb failure criterion for intact rock and fracture surfaces:

```
tau = c + sigma_n * tan(phi)
```

Where:
- `tau` is the shear strength (MPa)
- `sigma_n` is the normal stress on the fracture plane (MPa)
- `c` is the cohesion (MPa)
- `phi` is the friction angle (degrees)

**Principal stress form:**

```
sigma_1 = (2 * c * cos(phi)) / (1 - sin(phi)) + sigma_3 * (1 + sin(phi)) / (1 - sin(phi))
```

### 12.2 Barton-Bandis Joint Model

The Barton-Bandis empirical model (Barton, 1973; Barton and Choubey, 1977; Barton and Bandis, 1990) is the primary model for fracture shear strength:

```
tau = sigma_n * tan( JRC * log10(JCS / sigma_n) + phi_r )
```

Where:
- `JRC` is the Joint Roughness Coefficient (dimensionless, 0-20)
- `JCS` is the Joint Wall Compressive Strength (MPa)
- `phi_r` is the residual friction angle (degrees)
- `sigma_n` is the effective normal stress (MPa)

**Scale corrections (Bandis et al., 1981):**

```
JRC_n = JRC_0 * (L_n / L_0)^(-0.02 * JRC_0)
JCS_n = JCS_0 * (L_n / L_0)^(-0.03 * JRC_0)
```

Where `L_0` is the laboratory scale (typically 100 mm) and `L_n` is the in-situ block size.

**Normal closure (hyperbolic model, Bandis et al., 1983):**

```
delta_V = sigma_n * V_m / (K_ni * V_m + sigma_n)
```

Where:
- `delta_V` is the normal closure (mm)
- `V_m` is the maximum closure (mm)
- `K_ni` is the initial normal stiffness (MPa/mm)

### 12.3 Property Assignment Strategies

The software supports three strategies for assigning mechanical properties to individual fractures:

**Strategy 1: Set-based assignment (default)**

Each fracture set (defined by an orientation distribution) is assigned a single set of mechanical parameters (JRC, JCS, phi_r, etc.). Every fracture in the set inherits these values. This is the simplest and most common approach (Rogers et al., 2014).

**Strategy 2: Size-dependent assignment**

Fracture mechanical properties vary with fracture size, following the scale-correction formulae in Section 12.2:

```
JRC = JRC_lab * (L / 0.1)^(-0.02 * JRC_lab)
```

Where `L` is the fracture radius (or equivalent diameter for polygon fractures). Larger fractures are expected to have lower JRC (smoother at large scales).

**Strategy 3: Correlated random fields**

For advanced analysis, mechanical properties are generated as spatially correlated random fields. Each fracture inherits the property value at its center from the underlying field. This requires Gaussian random field simulation (e.g., sequential Gaussian simulation or spectral methods).

**All strategies support:**
- Mean and standard deviation for each parameter
- Distribution type (normal, lognormal, uniform, truncated normal)
- Seed-based reproducibility

**Export targets:**
- 3DEC `joint` commands with Mohr-Coulomb or Barton-Bandis parameters
- FLAC3D interface elements
- Custom CSV/JSON for other geomechanical codes

**Reference:** Rogers, S., Elmo, D., Webb, G., and Catalan, A. (2014). "Volumetric fracture intensity measurement for improved rock mass characterisation and fragmentation assessment." *Journal of the Southern African Institute of Mining and Metallurgy*, 114(7), 555-562.

---

## 13. Validation Methodology

### 13.1 Round-Trip Conversions

All orientation conversions must pass round-trip testing:

```
For all test vectors:
    dip_dir_1, dip_1 = normal_to_dip_direction_dip(dip_direction_dip_to_normal(dip_dir_0, dip_0))
    assert abs(angular_difference_dd(dip_dir_0, dip_dir_1)) < 1e-10
    assert abs(dip_0 - dip_1) < 1e-10
```

Test grid covering:
- Dip directions: 0, 45, 90, 135, 180, 225, 270, 315 degrees
- Dips: 0, 15, 30, 45, 60, 75, 90 degrees
- Edge cases: dip_dir = 0 (north), dip_dir = 359.999, dip = 0 (horizontal), dip = 90 (vertical)

### 13.2 Known-Geometry Test Cases

**Test: Two perpendicular vertical fractures**

- Fracture A: dip_dir=0, dip=90, center=(0,0,0), radius=5
- Fracture B: dip_dir=90, dip=90, center=(0,0,0), radius=5
- Analytical solution: intersection line is vertical (Z-axis), from z=-5 to z=5, length=10

**Test: Horizontal fracture intersecting vertical fracture**

- Fracture A: dip_dir=0, dip=0 (horizontal), center=(0,0,0), radius=10
- Fracture B: dip_dir=0, dip=90, center=(0,0,0), radius=10
- Analytical solution: intersection is a line along the X-axis (y=0, z=0), from x=-10 to x=10, length=20

**Test: Two parallel fractures** (should produce NO intersection)

- Both: dip_dir=45, dip=60, radius=5
- Centers: (0,0,0) and (0,0,10) -- separated by normal offset
- Expected: no intersection (confirmed by algorithm)

**Test: P32 unit test**

Generate 1000 parallel unit-area (r=1/sqrt(pi)) vertical fractures in a unit cube (1 m³).
- P32_target = 10 m⁻¹
- Expected: ~10 fractures (since each has area 1 m²)
- Tolerance: +-5% of target

**Test: P32 in a 10x10x10 m cube**

- 1000 fractures, r=1m, isotropic orientation
- Each fracture area: pi * 1² ≈ 3.1416 m²
- Domain volume: 1000 m³
- Expected P32 ≈ 1000 * 3.1416 / 1000 = 3.1416 m⁻¹
- Verify that computed P32 matches to within 1% (accounting for boundary truncation)

### 13.3 Statistical Consistency Checks

**Fisher distribution generation check:**

Generate N=10,000 vectors from Fisher(mu, kappa=20):
- Estimated kappa (from MLE) should lie within [19.5, 20.5]
- Mean direction should be within 1 degree of mu
- Spherical variance `1 - R_bar` should approximately equal `1/kappa`

**P10-P32 consistency:**

For isotropic Fisher(kappa=0) fractures in a large domain (to minimize boundary effects):
- Generate DFN with known P32
- Run virtual scanlines in multiple directions
- Mean P10 * pi * r_mean^2 (stereological relation) should approximately equal P32

**P32 convergence check:**

- Target P32 = 5.0 m⁻¹
- Convergence algorithm must achieve error < 5% within 10 iterations
- Test with varying domain volumes (1, 10, 100, 1000 m³) to verify robustness

### 13.4 Reproducibility with Fixed Seeds

Every stochastic function accepts `random_seed: int` and uses `numpy.random.Generator(PCG64(seed))`. The following must produce identical results across runs:

```python
rng1 = np.random.default_rng(seed=42)
result1 = generate_fisher_vectors(mean_direction=(0,0,1), kappa=20, n=1000, random_seed=42)

rng2 = np.random.default_rng(seed=42)
result2 = generate_fisher_vectors(mean_direction=(0,0,1), kappa=20, n=1000, random_seed=42)

assert np.array_equal(result1, result2)
```

Reproducibility is tested:
1. Within a single Python session (as above)
2. Across separate Python sessions
3. Across different operating systems (same numpy version)
4. With seed values at extremes (0, 2^32-1, negative values mapped to positive)

**Caveat:** Reproducibility across numpy versions requires the same BitGenerator (PCG64). This is documented in the user manual.

---

## 14. Known Limitations

### 14.1 Fracture Geometry Model

| Can Do | Cannot Do |
|--------|-----------|
| Planar disk fractures (Baecher) | Non-planar fractures (curved joints) |
| Planar polygonal fractures | Fractures with thickness (represented as zero-thickness surfaces) |
| Rectangular planes | Fracture tip process zones or propagation |
| Stochastic size distributions (power-law, lognormal) | Deterministic fracture termination at bedding planes (unless explicitly modeled as polygons) |

**Assumption:** All fractures are planar. In reality, natural fractures can be curved, especially at large scales. This assumption is standard in DFN modeling and is adequate for block cave fragmentation analysis where block definition depends on fracture intersection topology rather than precise fracture curvature.

### 14.2 Fracture Intersection

**Assumption:** Fractures are rigid planes. Intersection is purely geometric; no mechanical interaction is considered.

**Resolution dependency:** Very small fractures (radius << voxel size) may be missed in voxel-based analyses. The minimum detectable fracture size is effectively the voxel size.

**False negatives:** Two fractures that pass very close to each other (within a fraction of the fracture radius) but do not geometrically intersect are treated as disconnected. In the physical rock mass, these near-intersections might form a mechanical connection. This is a conservative assumption for stability analysis but may underestimate connectivity.

**Coplanarity threshold:** The epsilon threshold for treating two planes as parallel (Section 7.1) defaults to 1e-9. Fracture sets with very similar orientations may occasionally be misclassified as parallel. Users can configure this threshold.

### 14.3 Fragmentation / Block Cutting

**Known issues:**

1. **Boundary blocks:** Blocks intersecting the domain boundary are truncated, and their true volume is unknown. The software reports BSD for interior-only and all-blocks separately. Boundary blocks are flagged.

2. **Algorithmic complexity:** General 3D polyhedral block cutting has worst-case exponential complexity in the number of fractures. The software uses spatial partitioning to make this tractable, but very dense fracture networks (>10,000 fractures contributing to a single region) may be computationally prohibitive.

3. **Non-convex blocks:** The standard half-space intersection method produces only convex blocks. In reality, blocks can be non-convex. The software's BSD should be interpreted as an approximation of the true distribution, biased toward smaller volumes (since non-convex shapes have more surface area per unit volume than convex ones).

4. **Very elongated blocks:** Blocks with extreme aspect ratios (>100:1) may have numerical precision issues in volume computation due to near-degenerate geometry.

**What fragmentation analysis CAN do:**
- Estimate the BSD for well-connected fracture networks
- Identify the dominant block size (D50) to within an order of magnitude
- Provide comparative analysis (which fracture set dominantly controls block size)
- Support caveability assessments when combined with geomechanical analysis

**What fragmentation analysis CANNOT do:**
- Guarantee exact block volumes for a given realization
- Predict secondary fragmentation during draw (requires DEM/FLAC3D modeling)
- Account for rock bridges that are invisible at the modeling scale

### 14.4 Connectivity and Percolation

- **Finite domain effects:** Percolation threshold estimates are domain-size dependent. The software provides percolation curves for the selected domain size; extrapolation to larger domains should be done with caution.
- **Edge effects:** Fractures intersecting the domain boundary may connect to fractures outside the domain (not modeled). The software identifies boundary-connected components separately.
- **Percolation does NOT imply cavability:** A percolating fracture network is a necessary but not sufficient condition for cave propagation. Additional factors (stress state, fracture shear strength, rock mass strength, undercut geometry) must be analyzed separately.

### 14.5 RQD Limitations

- RQD is a **directional** measure. A single RQD value does not characterize the 3D fracture network.
- The Priest-Hudson formula assumes a Poisson spacing distribution. Real fracture spacings often show clustering or regularity.
- Mechanical breaks in core recovery can artificially reduce RQD; the software does not model mechanical breaks.
- RQD has low sensitivity in poor-quality rock masses (RQD < 25%) and in excellent rock masses (RQD > 90%), where small changes in fracture count produce large changes in RQD.

### 14.6 Mechanical Properties

- The software assigns mechanical properties to fractures but does **not** perform stress analysis, deformation analysis, or failure simulation.
- Barton-Bandis parameters are empirical and scale-dependent. Laboratory-derived values should be scaled to in-situ block sizes using the formulae in Section 12.2.
- Mohr-Coulomb and Barton-Bandis models assume drained conditions and do not account for pore pressure effects (effective stress principle must be applied by the user in the target geomechanical code).

### 14.7 Stochastic Sampling

- The Fisher distribution sampling algorithm (Section 3.3) is accurate for `kappa > 0.01`. For very low kappa (near-uniform), results may show minor numerical drift from true uniformity.
- The power-law radius distribution is truncated at `r_min` and `r_max`. Truncation values must be geologically justified; the default `r_max` is set to the domain diagonal unless explicitly specified.

---

## 15. References

### Primary Scientific References

1. Baecher, G.B., Lanney, N.A., and Einstein, H.H. (1977). "Statistical description of rock properties and sampling." *Proc. 18th U.S. Symposium on Rock Mechanics*, American Rock Mechanics Association, pp. 5C1-1 -- 5C1-8.

2. Bandis, S.C., Lumsden, A.C., and Barton, N.R. (1981). "Experimental studies of scale effects on the shear behaviour of rock joints." *International Journal of Rock Mechanics and Mining Sciences*, 18(1), 1-21.

3. Bandis, S.C., Lumsden, A.C., and Barton, N.R. (1983). "Fundamentals of rock joint deformation." *International Journal of Rock Mechanics and Mining Sciences*, 20(6), 249-268.

4. Barton, N. (1973). "Review of a new shear-strength criterion for rock joints." *Engineering Geology*, 7(4), 287-332.

5. Barton, N. and Bandis, S. (1990). "Review of predictive capabilities of JRC-JCS model in engineering practice." *Proc. Int. Symposium on Rock Joints*, Loen, Norway, pp. 603-610.

6. Barton, N. and Choubey, V. (1977). "The shear strength of rock joints in theory and practice." *Rock Mechanics*, 10(1-2), 1-54.

7. Blott, S.J. and Pye, K. (2008). "Particle shape: a review and new methods of characterization." *Sedimentology*, 55(1), 31-63.

8. Clauset, A., Shalizi, C.R., and Newman, M.E.J. (2009). "Power-law distributions in empirical data." *SIAM Review*, 51(4), 661-703.

9. Cormen, T.H., Leiserson, C.E., Rivest, R.L., and Stein, C. (2009). *Introduction to Algorithms*, 3rd Edition. MIT Press.

10. Deere, D.U. (1964). "Technical description of rock cores for engineering purposes." *Rock Mechanics and Engineering Geology*, 1(1), 16-22.

11. Dershowitz, W.S. (1984). "Rock joint systems." Ph.D. Thesis, Massachusetts Institute of Technology, Cambridge, MA.

12. Dershowitz, W.S. and Einstein, H.H. (1988). "Characterizing rock joint geometry with joint system models." *Rock Mechanics and Rock Engineering*, 21(1), 21-51.

13. Dershowitz, W.S. and Herda, H.H. (1992). "Interpretation of fracture spacing and intensity." *Proc. 33rd U.S. Symposium on Rock Mechanics*, Santa Fe, NM, pp. 757-766.

14. Dershowitz, W., Lee, G., Geier, J., Foxford, T., La Pointe, P., and Thomas, A. (1998). "FracMan: Interactive Discrete Feature Data Analysis, Geometric Modeling, and Exploration Simulation. User Documentation." Golder Associates Inc., Seattle, WA.

15. Eberly, D. (2008). "Intersection of a Disk and a Box." Geometric Tools, Redmond, WA. Available at: https://www.geometrictools.com/

16. Elmo, D., Rogers, S., Stead, D., and Eberhardt, E. (2014). "Discrete fracture network approach to characterise rock mass fragmentation and implications for geomechanical upscaling." *Mining Technology*, 123(3), 149-161.

17. Elmouttie, M., Poropat, G., and Krahenbuhl, G. (2010). "Polyhedral modelling of rock mass structure." *International Journal of Rock Mechanics and Mining Sciences*, 47(4), 577-585.

18. Ericson, C. (2004). *Real-Time Collision Detection*. Morgan Kaufmann, San Francisco, CA.

19. Fisher, R.A. (1953). "Dispersion on a sphere." *Proceedings of the Royal Society of London A*, 217(1130), 295-305.

20. Fisher, N.I., Lewis, T., and Embleton, B.J.J. (1987). *Statistical Analysis of Spherical Data*. Cambridge University Press, Cambridge, UK.

21. Guttman, A. (1984). "R-trees: A Dynamic Index Structure for Spatial Searching." *Proc. ACM SIGMOD*, pp. 47-57.

22. ISRM (1978). "Suggested methods for the quantitative description of discontinuities in rock masses." *International Journal of Rock Mechanics and Mining Sciences*, 15(6), 319-368.

23. Ivanova, V.M., Sousa, R., Murrihy, B., and Einstein, H.H. (2014). "Mathematical algorithm development and parametric studies with the GEOFRAC three-dimensional stochastic model of natural rock fracture systems." *Computers and Geosciences*, 67, 100-109.

24. La Pointe, P.R. (2002). "Derivation of fracture intensity and connectivity parameters from geologic field data for use in discrete fracture network and boundary element models." SKB Report R-02-21, Swedish Nuclear Fuel and Waste Management Co., Stockholm.

25. Laubscher, D.H. (1994). "Cave mining -- the state of the art." *Journal of the South African Institute of Mining and Metallurgy*, 94(10), 279-293.

26. Mardia, K.V. and Jupp, P.E. (2000). *Directional Statistics*. John Wiley & Sons, Chichester, UK.

27. Palmstrom, A. (1982). "The volumetric joint count -- a useful and simple measure of the degree of rock mass jointing." *Proc. 4th Int. Congress IAEG*, Delhi, V.221-V.228.

28. Priest, S.D. (1993). *Discontinuity Analysis for Rock Engineering*. Chapman & Hall, London.

29. Priest, S.D. and Hudson, J.A. (1976). "Discontinuity spacings in rock." *International Journal of Rock Mechanics and Mining Sciences*, 13(5), 135-148.

30. Rogers, S., Elmo, D., Webb, G., and Catalan, A. (2014). "Volumetric fracture intensity measurement for improved rock mass characterisation and fragmentation assessment." *Journal of the Southern African Institute of Mining and Metallurgy*, 114(7), 555-562.

31. Samet, H. (2006). *Foundations of Multidimensional and Metric Data Structures*. Morgan Kaufmann, San Francisco, CA.

32. Sutherland, I.E. and Hodgman, G.W. (1974). "Reentrant polygon clipping." *Communications of the ACM*, 17(1), 32-42.

33. Terzaghi, R.D. (1965). "Sources of error in joint surveys." *Geotechnique*, 15(3), 287-304.

34. Underwood, E.E. (1970). *Quantitative Stereology*. Addison-Wesley, Reading, MA.

35. Woodcock, N.H. (1977). "Specification of fabric shapes using an eigenvalue method." *Geological Society of America Bulletin*, 88(9), 1231-1236.

### Software and Methodology References

- *FracMan* (Golder Associates / WSP): DFN generation and analysis software suite. Primary design inspiration for DFN Cave Studio.
- *DFNWorks* (LANL): Open-source DFN generation for subsurface flow and transport. Reference for generation algorithms.
- *MoFrac* (University of Toronto): DFN generation code. Reference for fracture intersection algorithms.
- *3DEC* (Itasca Consulting Group): 3-Dimensional Distinct Element Code. Primary export target for geomechanical analysis.
- *FLAC3D* (Itasca Consulting Group): Fast Lagrangian Analysis of Continua in 3D. Secondary export target.

---

## Phase 2A: measured-point constrained along-hole realizations

Phase 2A is an independent derived-data workflow; it does not write generated rows into Formal fracture observations and does not automatically start M9. Its orientation fit uses only complete P point-cloud dominant-set representatives and complete Z borehole-camera representatives. Each imported representative remains an immutable Local Orientation Component identified by source kind, point, local set and observation; a Global Joint Set is only its confirmed parent label. Mapping several components to one global set never deletes or physically merges the local records. The input is an axial unit normal (a plane pole), not a Euclidean pair of dip and dip direction. P representatives carry `joint_num` as a fit weight; Z representatives have weight one. This weight therefore describes clustered representative orientations and must not be interpreted as raw individual-fracture orientation fitting. P `RANDOM` rows and every missing direction are excluded from Global K.

The number of imported local representatives is not the number of confirmed global joint sets. The latter is exactly the user-selected global `K`; point-local `local_set_id` values are mapped to, but never equated with, the confirmed global identifiers. A global group supported by one representative has `kappa_status=UNRESOLVED`; two or more representatives can provide only `SITE_MEAN_DISPERSION`, not original within-set Fisher dispersion. Phase 2A defaults to `LOCAL_REPRESENTATIVE`, which does not pretend that an unresolved Kappa was fitted; `FIXED_GLOBAL_SET_MEAN` remains an explicit legacy-compatible display/generation choice. P/Z representative observations contain no fracture-size or trace-length evidence, so their size status is `UNRESOLVED`; the stored legacy model-shaped placeholders are inactive and cannot satisfy the M10 size-model preflight. Likewise their target intensity is `DERIVED_LATER_BY_M9`, not an automatically fitted `P32=0.5`.

Local P dominant-set intensity is `lambda_g(x_i) = 1 / spacing_g(x_i)`. A P random row similarly defines `lambda_random`; Z rows never provide intensity. Each intensity is interpolated independently in XYZ with deterministic three-dimensional IDW, then the available positive intensities are normalized into component probabilities. An unreported component is missing information, not a measured zero. A spacing interval with no valid nearby P intensity support is reported as blocked rather than assigned equal probabilities. Representative directions are combined as sign-aligned axial pole vectors; dip and dip direction are never interpolated as ordinary scalars.

Every spacing record retains a `measurement_basis`: `BOREHOLE_ALONG_HOLE`, `TRUE_NORMAL`, or `SCANLINE_APPARENT`. Only true-normal spacing can support the explicit approximation `P32 ≈ 1/S`. Along-hole and scanline-apparent spacing require directional correction before any P32 interpretation. Phase 2A does not perform that conversion: P-site reciprocal spacing supplies relative component allocation, while the borehole interval spacing alone supplies the Poisson total `L/S`, so intensity is not counted twice.

For an original spacing-only borehole interval `[a,b)` with length `L=b-a` and spacing `S`, the implemented `HOMOGENEOUS_POISSON` sampler uses:

```text
N ~ Poisson(L / S)
MD_i | N ~ sorted Uniform[a,b)
```

For each borehole interval midpoint, P sites are first selected by `(distance, stable point index)`. `max_neighbors` therefore counts sites, not component rows. Every valid dominant and reported RANDOM component at each selected site receives the same spatial kernel; component contribution is `(1 / spacing) × kernel`, and all contributions are normalized. `joint_num` is not multiplied again. Z components never enter this probability calculation. An unlimited search is an explicit extrapolation mode and records nearest/farthest constraint distance and selected site count.

Measured depths are located on the persisted survey trajectory. Dominant fractures use the selected local representative, an axial-normal spatial fit within its confirmed Global Set (P and Z direction anchors), or the legacy fixed global-set mean. Random-background normals are isotropic on an axial hemisphere and retain a distinct component code instead of a fabricated global set. Every stochastic stream is derived deterministically from `master_seed` and realization index.

Generated scientific rows are compact aligned NumPy columns (`interval_index`, measured depth, origin-relative XYZ, global set, `local_component_index`, component type, dip, dip direction, and status). The component index references the fit-level metadata table; strings are not repeated per fracture. Generator version `borehole-phase2a-3` records this local-mixture allocation change. Multiple realizations, their input hash, mappings, diagnostics, seeds, and random-component declarations are persisted separately from the source database. On reopen only the selected realization arrays are loaded.

The M9 adapter requires an explicit realization ID and never mixes it with Formal observations, chooses the newest realization automatically, rewrites Formal data, or refits Global K. It derives set-specific `P10=N/L` from generated intersections and actual survey-trajectory length, then uses the existing directional-exposure correction for P32. `RANDOM_BACKGROUND` remains unassigned and is excluded with an audited count. A set/interval without P intensity support is `NO_DATA`; a zero count is `TRUE_ZERO` only where the Phase 2A interval diagnostics record positive support for that global set. Only-Z direction support therefore cannot become a false geological zero.

This adapter is an internal stochastic-consistency path, not a new P32 conversion formula. Validation holes remain excluded from every fit by the locked whole-hole Holdout, but because their realization was generated from their own spacing, subsequent comparison is labelled `internal_consistency_not_independent`; it is not reported as independent validation or observed P32 truth. Realization ID, generator version, master/derived seed, source hash, excluded count and set support states are persisted in M9 provenance. A missing realization or mismatched current input hash blocks downstream field construction. M10/M11 do not consume this adapter in the present phase.

---

## Appendix A: Notation Index

| Symbol | Meaning | Units |
|--------|---------|-------|
| `phi` / `alpha_d` | Dip direction | degrees (I/O) or radians (internal) |
| `theta` / `beta_d` | Dip | degrees (I/O) or radians (internal) |
| `n` | Unit normal vector (upper hemisphere) | dimensionless |
| `p` | Pole vector (lower hemisphere) | dimensionless |
| `kappa` | Fisher concentration parameter | dimensionless |
| `mu` | Fisher mean direction | dimensionless (unit vector) |
| `r` | Fracture radius (disk model) | m |
| `c` | Fracture center point | m |
| `A_i` | Area of fracture i | m² |
| `V` | Volume | m³ |
| `P10` | Linear fracture intensity | m⁻¹ |
| `P21` | Areal fracture intensity | m/m² |
| `P32` | Volumetric fracture intensity | m²/m³ |
| `P33` | Volumetric fracture density | m³/m³ |
| `L` | Scanline/borehole length | m |
| `s` | Scanline direction | dimensionless (unit vector) |
| `JRC` | Joint Roughness Coefficient | dimensionless |
| `JCS` | Joint Wall Compressive Strength | MPa |
| `phi_r` | Residual friction angle | degrees |
| `tau` | Shear strength | MPa |
| `sigma_n` | Normal stress | MPa |
| `c_m` | Cohesion | MPa |
| `phi_m` | Friction angle (Mohr-Coulomb) | degrees |
| `D50` | Median block equivalent diameter | m |
| `Deq` | Equivalent sphere diameter | m |
| `d_avg` | Average graph degree | dimensionless |
| `beta_1` | Cyclomatic number | dimensionless |

---

## Appendix B: Default Numerical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `EPSILON_PARALLEL` | 1e-9 | Threshold for treating two planes as parallel |
| `EPSILON_COPLANAR` | 1e-6 | Threshold for coplanarity check (m) |
| `MAX_TERZAGHI_ANGLE` | 85° | Maximum angle for Terzaghi correction |
| `P32_TOLERANCE` | 0.05 | Default P32 convergence tolerance (5%) |
| `MAX_P32_ITERATIONS` | 50 | Maximum iterations for P32 convergence |
| `DEFAULT_SEED` | 42 | Master random seed |
| `VXL_CHUNK_SIZE` | 32 | Default voxel chunk dimension |
| `MIN_BLOCK_VOLUME` | 1e-12 | Minimum reportable block volume (m³) |
| `RQD_THRESHOLD` | 0.1 | RQD intact length threshold (m) |

---

## M11 interactive P32 cloud visualization (display-only)

The interactive M11 visualization layer visualizes persisted M11.1 intersection-derived fields without changing them. Supported fields are `p32_explicit_intersection`, `p32_subgrid`, and `p32_total`, for all joint sets or an individual set. The scientific identity remains:

```text
p32_total = p32_explicit_intersection + p32_subgrid
```

`p32_unresolved_orientation` is not added to `p32_total` by the visualization layer.

**Exact Cell Colours** is the authoritative audit view. It maps the original cell scalar directly to each valid voxel or section cell. **Smooth Display** is display interpolation only: invalid cells (`OUTSIDE_MODEL`, `NO_DATA`, and `EXCAVATION`) are removed before cell values are converted to point values. Consequently, invalid values cannot contribute to interpolation at the model boundary. Smooth arrays are temporary VTK display arrays and are neither written back to the M11 result nor saved in `.dfnproj`.

The outer surface is extracted from the actual set of valid cells, not from the rectangular model bounds. Orthogonal and arbitrary sections use VTK plane slicing. Global colour ranges are calculated from all finite valid values in the selected M11 field so separate sections remain visually comparable. Manual ranges and discrete colour bands alter only mapper configuration.

The non-modal M11 dock owns stable interactive slots for the three orthogonal planes, one arbitrary plane, one cutaway plane, and one box cutaway. Coordinate input, sliders, and VTK widgets update the same slot; snapshots are explicit separate layers. Orthogonal normals are fixed to X, Y, or Z and only their origin may translate. Arbitrary and cutaway normals are validated and normalized on temporary display values without modifying user input or scientific arrays. Interactive widgets and box bounds are constrained to the Analysis Domain, while initial/reset positions use the effective valid-cell mask bounds.

A plane cutaway first clips the finite, valid-cell volume by a plane and then extracts the retained volume surface. Consequently, the newly exposed internal face carries P32 display scalars; it is not an open outer shell. Flip Side selects the opposite half-space.

The axis-aligned Box Cutaway follows the same volume-first rule. It masks invalid or non-finite cells, retains the valid volume inside six axis-aligned bounds, and only then extracts a display surface. Box faces therefore expose the P32 values of their source cells wherever valid volume exists; the renderer does not fabricate faces across `NO_DATA` holes or outside the model. `TRUE_ZERO` remains a valid displayed value. **Snap Box to Voxel Faces** is enabled by default and retains complete source voxels with bounds on real voxel faces. When snapping is disabled, continuous geometric clipping is display-only and does not create new scientific voxel values. Algorithm-generated triangulation edges are hidden in continuous mode rather than being represented as voxel grid lines.

Slice, plane/box cutaway, and scientific domains are distinct. A slice constructs a plane intersection for viewing. A cutaway retains part of the valid display volume and exposes its display surface. Neither operation changes the Analysis Domain, Generation Domain, persisted cell mask, sparse intersection table, or any P32 array.

This feature is not Kriging, spatial estimation, second scientific interpolation, or volume rendering. Cloud actors, scalar bars, plane widgets, and preparation caches are session-only state. They cannot change P32 conservation, sparse fracture/voxel intersections, workflow state, or project dirty state.

---

*End of Scientific Specification*
