"""Generate synthetic M7 demo dataset with known issues for testing import/cleaning.

Creates:
  - 8 boreholes across 2 structural domains
  - 3 real joint sets (Fisher-distributed orientations)
  - Deliberate data issues: missing values, duplicate rows, out-of-range
    depths, overlapping RQD intervals, 359°/1° direction wrapping
  - 2 validation boreholes
"""

import os
import math
import numpy as np

OUT = os.path.dirname(__file__)
SEED = 42
rng = np.random.default_rng(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Collars (8 boreholes)
# ═══════════════════════════════════════════════════════════════════════════
hole_ids = [f"BH-{i:02d}" for i in range(1, 9)]
eastings = [100 + i * 50 for i in range(8)]
northings = [200 + i * 30 for i in range(8)]
elevations = [500 + rng.uniform(-5, 5) for _ in range(8)]
total_depths = [150, 200, 180, 155, 190, 205, 175, 160]
azimuths = [0, 0, 0, 0, 0, 0, 0, 0]
dips = [-90, -90, -90, -90, -90, -90, -90, -90]
domain_ids = [1, 1, 1, 2, 2, 2, 2, 1]  # 4 in domain 1, 4 in domain 2

with open(os.path.join(OUT, "collars.csv"), "w") as f:
    f.write("hole_id,easting,northing,elevation,total_depth,azimuth,dip,domain_id\n")
    for i in range(8):
        f.write(f"{hole_ids[i]},{eastings[i]},{northings[i]},{elevations[i]:.1f},"
                f"{total_depths[i]},{azimuths[i]},{dips[i]},{domain_ids[i]}\n")
    # Bad row: missing hole_id
    f.write(",999,999,999,100,0,-90,1\n")
    # Bad row: negative total_depth
    f.write("BH-BAD,300,300,500,-50,0,-90,1\n")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Surveys (stations every 10m)
# ═══════════════════════════════════════════════════════════════════════════
with open(os.path.join(OUT, "surveys.csv"), "w") as f:
    f.write("hole_id,measured_depth,azimuth,dip\n")
    for i in range(8):
        for md in range(0, total_depths[i] + 10, 10):
            az = rng.uniform(-1, 1)
            dp = -90 + rng.uniform(-0.5, 0.5)
            f.write(f"{hole_ids[i]},{md},{az:.2f},{dp:.2f}\n")
    # Bad row: depth exceeding total
    f.write("BH-01,999,0,-90\n")
    # Bad row: duplicate depth for same hole
    f.write("BH-01,10,0,-90\n")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Fractures (3 Fisher-distributed joint sets)
# ═══════════════════════════════════════════════════════════════════════════
# Set 1:  dd≈45, dip≈60 (sub-vertical NE-SW)
# Set 2:  dd≈135, dip≈75 (sub-vertical NW-SE)
# Set 3:  dd≈270, dip≈30 (shallow east-dipping)
# Include 359°/1° wrapping case in Set 3

def fisher_sample(dd_mean, dip_mean, kappa, n, rng):
    """Generate Fisher-distributed dip_direction/dip pairs."""
    # Mean direction to unit vector
    alpha = math.radians(dd_mean)
    beta = math.radians(dip_mean)
    mean_vec = np.array([
        -math.sin(alpha) * math.sin(beta),
        -math.cos(alpha) * math.sin(beta),
        math.cos(beta),
    ])
    # Sample from Fisher using Woodcock method
    samples = []
    for _ in range(n):
        # Random direction from Fisher distribution around (0,0,1)
        cos_theta = 1 + math.log(1 - rng.random() * (1 - math.exp(-2 * kappa))) / kappa
        cos_theta = max(-1, min(1, cos_theta))
        theta = math.acos(cos_theta)
        phi = rng.uniform(0, 2 * math.pi)
        v = np.array([math.sin(theta) * math.cos(phi),
                       math.sin(theta) * math.sin(phi), math.cos(theta)])
        # Rotate v to align with mean_vec
        z = np.array([0, 0, 1])
        if np.allclose(mean_vec, z):
            rotated = v
        elif np.allclose(mean_vec, -z):
            rotated = np.array([v[0], -v[1], -v[2]])
        else:
            axis = np.cross(z, mean_vec)
            axis = axis / np.linalg.norm(axis)
            cos_a = np.dot(z, mean_vec)
            sin_a = math.sqrt(1 - cos_a**2)
            # Rodrigues rotation
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])
            R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
            rotated = R @ v
        # Ensure upper hemisphere
        if rotated[2] < 0:
            rotated = -rotated
        # Back to dd/dip
        nz = max(-1, min(1, rotated[2]))
        dip_rad = math.acos(nz)
        dip_val = math.degrees(dip_rad)
        sin_b = math.sin(dip_rad)
        if sin_b < 1e-12:
            dd_val = 0
        else:
            sin_a = -rotated[0] / sin_b
            cos_a = -rotated[1] / sin_b
            dd_val = math.degrees(math.atan2(sin_a, cos_a)) % 360
        samples.append((round(dd_val, 1), round(dip_val, 1)))
    return samples

set1 = fisher_sample(45, 60, 30, 25, rng)
set2 = fisher_sample(135, 75, 25, 20, rng)
set3 = fisher_sample(270, 30, 20, 15, rng)
# Add 359°/1° wrapping specifically
set3.append((359.0, 30.0))
set3.append((1.0, 30.0))
rng.shuffle(set3)

with open(os.path.join(OUT, "fractures.csv"), "w") as f:
    f.write("hole_id,depth,dip_direction,dip,set_id\n")
    for i in range(8):
        for j, obs_set in enumerate([set1, set2, set3]):
            n_obs = min(len(obs_set) // 8 + 1, 8)
            for k in range(n_obs):
                idx = (i * n_obs + k) % len(obs_set)
                dd, dip = obs_set[idx]
                depth = rng.uniform(5, total_depths[i] - 5)
                f.write(f"{hole_ids[i]},{depth:.1f},{dd:.1f},{dip:.1f},{j+1}\n")
    # Bad row: depth exceeding total
    f.write("BH-01,9999,45,60,1\n")
    # Bad row: dip out of range
    f.write("BH-02,50,45,999,1\n")
    # Bad row: invalid set_id
    f.write("BH-03,50,45,60,not_a_number\n")

# ═══════════════════════════════════════════════════════════════════════════
# 4. RQD intervals (with deliberate overlap)
# ═══════════════════════════════════════════════════════════════════════════
with open(os.path.join(OUT, "rqd.csv"), "w") as f:
    f.write("hole_id,from_depth,to_depth,rqd\n")
    for i in range(8):
        intervals = [
            (0, 50, rng.uniform(60, 95)),
            (50, 100, rng.uniform(50, 90)),
            (100, total_depths[i], rng.uniform(40, 85)),
        ]
        for fd, td, rqd_val in intervals:
            f.write(f"{hole_ids[i]},{fd},{td},{rqd_val:.1f}\n")
    # Bad row: overlap (90-150 overlaps with 100-200)
    f.write("BH-01,90,150,75\n")
    # Bad row: RQD > 100
    f.write("BH-02,50,100,150\n")
    # Bad row: from > to
    f.write("BH-03,100,50,80\n")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Domain intervals
# ═══════════════════════════════════════════════════════════════════════════
# BH-05 crosses two domains (non-overlapping): 0-100 Domain 2, 100-190 Domain 1
# BH-06 has three non-overlapping sections across two domains:
#   0-50 Domain 2, 50-150 Domain 3, 150-205 Domain 2
with open(os.path.join(OUT, "domain_intervals.csv"), "w") as f:
    f.write("hole_id,from_depth,to_depth,domain_id,domain_name\n")
    f.write("BH-01,0,150,1,Domain 1\n")
    f.write("BH-02,0,200,1,Domain 1\n")
    f.write("BH-03,0,180,1,Domain 1\n")
    f.write("BH-04,0,155,2,Domain 2\n")
    f.write("BH-05,0,100,2,Domain 2\n")
    f.write("BH-06,0,50,2,Domain 2\n")
    f.write("BH-07,0,175,2,Domain 2\n")
    f.write("BH-08,0,160,1,Domain 1\n")
    f.write("BH-05,100,190,1,Domain 1\n")
    f.write("BH-06,50,150,3,Domain 3\n")
    f.write("BH-06,150,205,2,Domain 2\n")

print("M7 demo dataset generated in:", OUT)
print(f"  collars.csv: {8}+2 rows")
print(f"  surveys.csv: ~{sum(td//10+1 for td in total_depths)}+2 rows")
print(f"  fractures.csv: ~{len(set1)+len(set2)+len(set3)}+3 rows")
print(f"  rqd.csv: {24}+3 rows")
print(f"  domain_intervals.csv: 11 rows")
print("  Issues: missing hole_id, negative depth, depth exceed, RQD overlap,")
print("          RQD>100, from>to, 359/1 wrapping, non-integer set_id")
