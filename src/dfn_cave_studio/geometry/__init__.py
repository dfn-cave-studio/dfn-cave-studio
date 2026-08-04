"""
Geometric primitives and algorithms for DFN Cave Studio.

Pure computation layer — no Qt, no VTK, no PyVista imports.
Only depends on NumPy and SciPy.
"""

from dfn_cave_studio.geometry.vector import (
    normalize,
    length,
    dot,
    cross,
    angle_between,
    distance,
    midpoint,
    project_point_to_line,
    project_point_to_plane,
    are_parallel,
    are_orthogonal,
    plane_from_three_points,
    polygon_area_3d,
)

from dfn_cave_studio.geometry.coordinate import (
    dip_dir_dip_to_normal,
    normal_to_dip_dir_dip,
    normal_to_pole,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    rotate_vector,
    dip_dir_to_strike,
    strike_dip_to_normal,
    batch_dip_dir_dip_to_normal,
    batch_normal_to_dip_dir_dip,
)

from dfn_cave_studio.geometry.intersection import (
    line_plane_intersection,
    point_in_aabb,
    aabb_aabb_overlap,
    plane_aabb_intersects,
    disk_aabb_intersects,
    fracture_fracture_intersects,
)

__all__ = [
    # Vector
    "normalize",
    "length",
    "dot",
    "cross",
    "angle_between",
    "distance",
    "midpoint",
    "project_point_to_line",
    "project_point_to_plane",
    "are_parallel",
    "are_orthogonal",
    "plane_from_three_points",
    "polygon_area_3d",
    # Coordinate
    "dip_dir_dip_to_normal",
    "normal_to_dip_dir_dip",
    "normal_to_pole",
    "rotation_matrix_x",
    "rotation_matrix_y",
    "rotation_matrix_z",
    "rotate_vector",
    "dip_dir_to_strike",
    "strike_dip_to_normal",
    "batch_dip_dir_dip_to_normal",
    "batch_normal_to_dip_dir_dip",
    # Intersection
    "line_plane_intersection",
    "point_in_aabb",
    "aabb_aabb_overlap",
    "plane_aabb_intersects",
    "disk_aabb_intersects",
    "fracture_fracture_intersects",
]
