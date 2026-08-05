"""
Fracture connectivity graph analysis (M5).

Constructs a graph where:
  - Nodes = fractures
  - Edges = geometric intersections between fractures

From the graph, we compute:
  - Connected components (clusters)
  - Largest cluster size and fraction
  - Boundary-percolating clusters (spanning model)
  - From undercut to upper region percolation
  - Inter-set intersection matrix
  - Connectivity statistics (degree distribution, etc.)

DO NOT:
  - Equate "adjacent voxels both have fractures" with "fractures connected"
  - Treat voxel boundaries as fracture boundaries
  - Claim percolation without explicit fracture-fracture intersection

References:
  - SCIENTIFIC_SPEC.md Section 9.
"""

from __future__ import annotations

import math
from typing import Optional, List, Dict, Set, Tuple, Callable

import numpy as np
from numpy.typing import NDArray

from dfn_cave_studio.models.fracture import StochasticFracture
from dfn_cave_studio.models.dfn_realization import DFNRealization
from dfn_cave_studio.geometry.intersection import fracture_fracture_intersects


# =============================================================================
# Connectivity Graph
# =============================================================================

class ConnectivityGraph:
    """Graph representation of fracture network connectivity.

    Stores adjacency list for efficient traversal and analysis.
    Uses NetworkX for graph algorithms when available, with fallback
    to custom implementations for core functionality.
    """

    def __init__(self, realization: DFNRealization):
        self.realization = realization
        self.fractures = realization.stochastic_fractures
        self.n_fractures = len(self.fractures)

        # Adjacency: dict[fracture_index] → set(neighbor_indices)
        self._adjacency: Dict[int, Set[int]] = {}
        self._edges_computed: bool = False
        self._n_edges: int = 0

        # Cached results
        self._components: Optional[List[Set[int]]] = None
        self._component_labels: Optional[NDArray[np.int32]] = None

    # ── Edge Computation ──────────────────────────────────────────────────

    def compute_edges(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """Compute all fracture-fracture intersections.

        Uses pairwise exact intersection tests (O(N²) worst case).
        For large DFNs (>10K fractures), consider spatial indexing.

        Args:
            progress_callback: Optional (current, total, message) callback.

        Returns:
            Number of edges (intersecting fracture pairs).
        """
        n = self.n_fractures
        self._adjacency = {i: set() for i in range(n)}
        self._n_edges = 0

        for i in range(n):
            if progress_callback and i % 500 == 0:
                progress_callback(i, n, f"Computing edges {i}/{n}")

            fi = self.fractures[i]
            for j in range(i + 1, n):
                fj = self.fractures[j]

                ri = fi.radius if fi.radius > 0 else (fi.geometry.radius or 1.0)
                rj = fj.radius if fj.radius > 0 else (fj.geometry.radius or 1.0)

                # Bounding sphere fast-reject
                center_dist = float(np.linalg.norm(
                    fi.geometry.center - fj.geometry.center
                ))
                if center_dist > (ri + rj):
                    continue  # Too far apart to intersect

                # Exact intersection test
                if fracture_fracture_intersects(
                    fi.geometry.center, fi.geometry.normal, ri,
                    fj.geometry.center, fj.geometry.normal, rj,
                ):
                    self._adjacency[i].add(j)
                    self._adjacency[j].add(i)
                    self._n_edges += 1

        self._edges_computed = True
        return self._n_edges

    # ── Graph Properties ──────────────────────────────────────────────────

    @property
    def n_edges(self) -> int:
        return self._n_edges

    @property
    def n_nodes(self) -> int:
        return self.n_fractures

    def neighbors(self, fracture_index: int) -> Set[int]:
        """Get neighbors of a fracture."""
        return self._adjacency.get(fracture_index, set())

    def degree(self, fracture_index: int) -> int:
        """Get degree (number of connections) of a fracture."""
        return len(self.neighbors(fracture_index))

    def average_degree(self) -> float:
        """Average degree across all fractures."""
        if self.n_fractures == 0:
            return 0.0
        total_degree = sum(self.degree(i) for i in range(self.n_fractures))
        return total_degree / self.n_fractures

    # ── Connected Components ──────────────────────────────────────────────

    def find_components(self) -> List[Set[int]]:
        """Find all connected components using depth-first search.

        Returns:
            List of components, each a set of fracture indices.
            Sorted by size (largest first).
        """
        if self._components is not None:
            return self._components

        if not self._edges_computed:
            self.compute_edges()

        visited = set()
        components = []

        def dfs(node: int, component: Set[int]) -> None:
            visited.add(node)
            component.add(node)
            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for i in range(self.n_fractures):
            if i not in visited:
                component: Set[int] = set()
                dfs(i, component)
                components.append(component)

        # Sort by size descending
        components.sort(key=len, reverse=True)
        self._components = components

        # Build label array
        self._component_labels = np.full(self.n_fractures, -1, dtype=np.int32)
        for label, comp in enumerate(components):
            for idx in comp:
                self._component_labels[idx] = label

        return components

    def component_labels(self) -> NDArray[np.int32]:
        """Get component label for each fracture (-1 = uncomputed)."""
        if self._component_labels is None:
            self.find_components()
        return self._component_labels

    def largest_component_size(self) -> int:
        """Size of the largest connected component."""
        comps = self.find_components()
        return len(comps[0]) if comps else 0

    def largest_component_fraction(self) -> float:
        """Fraction of fractures in the largest component."""
        if self.n_fractures == 0:
            return 0.0
        return self.largest_component_size() / self.n_fractures

    def n_components(self) -> int:
        """Number of connected components."""
        return len(self.find_components())

    def isolated_fractures(self) -> int:
        """Number of isolated fractures (degree 0, component of size 1)."""
        comps = self.find_components()
        return sum(1 for c in comps if len(c) == 1)

    # ── Percolation Analysis ──────────────────────────────────────────────

    def find_percolating_clusters(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        direction: str = "any",
    ) -> List[int]:
        """Find component labels that span from one model face to the opposite face.

        Uses real fracture-boundary intersection (disk-plane distance test)
        rather than bounding-sphere approximation.

        Args:
            x_range: (x_min, x_max) of model.
            y_range: (y_min, y_max) of model.
            z_range: (z_min, z_max) of model.
            direction: Percolation direction to check.
                       "x", "y", "z", or "any" (default).

        Returns:
            List of component labels that percolate in the specified direction.
        """
        if self._component_labels is None:
            self.find_components()

        comps = self.find_components()
        percolating = []

        for label, comp in enumerate(comps):
            if self._component_percolates(
                comp, x_range, y_range, z_range, direction
            ):
                percolating.append(label)

        return percolating

    def percolation_detail(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
    ) -> Dict:
        """Detailed percolation analysis for all directions.

        Returns a dict with percolating status in X, Y, Z, and the
        largest percolating cluster size for each direction.
        Geomechanical connectivity requires additional checks
        (stress state, dilation, shear displacement) not performed here.

        Returns:
            Dict with keys: percolates_x, percolates_y, percolates_z,
            largest_x_size, largest_y_size, largest_z_size,
            geometric_only (True — mechanical filtering not yet applied).
        """
        comps = self.find_components()
        result = {
            "percolates_x": False,
            "percolates_y": False,
            "percolates_z": False,
            "largest_x_size": 0,
            "largest_y_size": 0,
            "largest_z_size": 0,
            "geometric_only": True,
            "x_percolating_labels": [],
            "y_percolating_labels": [],
            "z_percolating_labels": [],
        }

        for label, comp in enumerate(comps):
            spans = self._component_span_directions(comp, x_range, y_range, z_range)
            if spans["x"]:
                result["percolates_x"] = True
                result["x_percolating_labels"].append(label)
                result["largest_x_size"] = max(result["largest_x_size"], len(comp))
            if spans["y"]:
                result["percolates_y"] = True
                result["y_percolating_labels"].append(label)
                result["largest_y_size"] = max(result["largest_y_size"], len(comp))
            if spans["z"]:
                result["percolates_z"] = True
                result["z_percolating_labels"].append(label)
                result["largest_z_size"] = max(result["largest_z_size"], len(comp))

        return result

    def _component_percolates(
        self,
        component: Set[int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        direction: str = "any",
    ) -> bool:
        """Check if a component spans the model in the specified direction(s).

        Uses real fracture geometry: a fracture touches a boundary if the
        perpendicular distance from the fracture center to the boundary
        plane, projected along the boundary normal, is ≤ the fracture
        radius times the cosine of the angle between the fracture normal
        and the boundary normal (i.e., the fracture disk intersects the
        boundary plane within its radius).
        """
        def _touches_boundary(frac, bound_normal, bound_val, dim_idx, range_a, range_b):
            """Check if fracture disk intersects a FINITE model boundary face.

            A fracture touches a boundary if:
            1. The perpendicular distance from the disk center to the boundary
               plane ≤ the disk's projected radius (infinite plane check)
            2. The intersection between the disk and the boundary plane lies
               within the finite face rectangle (finite face check).

            Args:
                frac: Fracture object.
                bound_normal: Inward-pointing normal of the boundary (3,).
                bound_val: Coordinate value of the boundary (e.g., x_min).
                dim_idx: Index of the boundary dimension (0=x, 1=y, 2=z).
                range_a: (min, max) for the first non-boundary dimension.
                range_b: (min, max) for the second non-boundary dimension.
            """
            r = frac.radius if frac.radius > 0 else (frac.geometry.radius or 1.0)
            center = frac.geometry.center
            n = frac.geometry.normal

            # 1. Infinite plane check: distance from center to boundary plane
            dist = abs(float(center[dim_idx] - bound_val))
            # Projected radius in the boundary normal direction
            # cos(θ) = |n·bound_normal|, effective radius = r·sin(θ) = r·√(1-cos²θ)
            cos_theta = abs(float(n[dim_idx]))  # bound_normal is axis-aligned
            proj_radius = r * math.sqrt(max(0.0, 1.0 - cos_theta ** 2))
            if dist > proj_radius + 1e-6:
                return False

            # 2. Finite face check: the intersection of the disk with the
            #    boundary plane must overlap the face rectangle.
            #    The intersection of the disk plane with the boundary plane
            #    is a line. Parametrize the line, find the segment within
            #    the disk, and check if it overlaps the face.
            if proj_radius < 1e-12:
                # Disk is parallel to boundary plane — only the center
                # matters for the infinite plane check.
                # Check if center lies within face rectangle.
                d1, d2 = (1, 2) if dim_idx == 0 else ((0, 2) if dim_idx == 1 else (0, 1))
                return (range_a[0] - 1e-6 <= center[d1] <= range_a[1] + 1e-6 and
                        range_b[0] - 1e-6 <= center[d2] <= range_b[1] + 1e-6)

            # Compute the intersection line. The line direction is
            # n_disk × n_boundary (both lie in the boundary plane).
            # Simpler: solve for the line in the boundary plane coordinates.

            # The disk plane: n·(P - c) = 0
            # On boundary plane x = x_min (for dim=0):
            #   nx*(x-x_min) + nx*(x_min-cx) + ny*(y-cy) + nz*(z-cz) = 0
            # But P is on x=x_min, so first term is 0:
            #   ny*(y-cy) + nz*(z-cz) + nx*(x_min-cx) = 0
            # This is a line in the (y,z) plane.

            d1, d2 = (1, 2) if dim_idx == 0 else ((0, 2) if dim_idx == 1 else (0, 1))
            a = float(n[d1])   # coefficient for first free coordinate
            b = float(n[d2])   # coefficient for second free coordinate
            c_offset = float(n[dim_idx]) * (bound_val - float(center[dim_idx]))

            # Line equation in the face plane: a·(u-c_u) + b·(v-c_v) + c_offset = 0
            # where u = coord[d1], v = coord[d2]
            # The distance from center projection to this line is |c_offset| / sqrt(a²+b²)

            line_normal_mag = math.sqrt(a**2 + b**2)
            if line_normal_mag < 1e-12:
                # Disk normal is parallel to boundary normal — shouldn't reach here
                # (handled by proj_radius check above)
                return False

            # Distance from (center[d1], center[d2]) to the line in the face plane
            dist_face = abs(c_offset) / line_normal_mag
            if dist_face > proj_radius + 1e-6:
                return False

            # The intersection segment on the line within the disk has
            # half-length h = sqrt(proj_radius² - dist_face²) in the face plane
            h_face = math.sqrt(max(0.0, proj_radius**2 - dist_face**2))

            # The line's closest point to (center[d1], center[d2]) in the face plane:
            p0_u = float(center[d1]) - a * c_offset / line_normal_mag**2
            p0_v = float(center[d2]) - b * c_offset / line_normal_mag**2

            # Direction along the line in the face plane: perpendicular to (a,b)
            line_dir_u = -b / line_normal_mag
            line_dir_v = a / line_normal_mag

            # Intersection segment on the line: from p0 ± h_face * line_dir
            seg_u_min = p0_u - h_face * abs(line_dir_u) - 1e-6
            seg_u_max = p0_u + h_face * abs(line_dir_u) + 1e-6
            seg_v_min = p0_v - h_face * abs(line_dir_v) - 1e-6
            seg_v_max = p0_v + h_face * abs(line_dir_v) + 1e-6

            # Check overlap with the face rectangle
            face_u_min, face_u_max = range_a
            face_v_min, face_v_max = range_b

            return (seg_u_min <= face_u_max and seg_u_max >= face_u_min and
                    seg_v_min <= face_v_max and seg_v_max >= face_v_min)

        # Aggregate boundary-touching flags across all fractures in the component
        x_min_comp = False
        x_max_comp = False
        y_min_comp = False
        y_max_comp = False
        z_min_comp = False
        z_max_comp = False

        for fi in component:
            f = self.fractures[fi]

            if not x_min_comp:
                x_min_comp = _touches_boundary(
                    f, None, x_range[0], 0, y_range, z_range,
                )
            if not x_max_comp:
                x_max_comp = _touches_boundary(
                    f, None, x_range[1], 0, y_range, z_range,
                )
            if not y_min_comp:
                y_min_comp = _touches_boundary(
                    f, None, y_range[0], 1, x_range, z_range,
                )
            if not y_max_comp:
                y_max_comp = _touches_boundary(
                    f, None, y_range[1], 1, x_range, z_range,
                )
            if not z_min_comp:
                z_min_comp = _touches_boundary(
                    f, None, z_range[0], 2, x_range, y_range,
                )
            if not z_max_comp:
                z_max_comp = _touches_boundary(
                    f, None, z_range[1], 2, x_range, y_range,
                )

        percolates_x = x_min_comp and x_max_comp
        percolates_y = y_min_comp and y_max_comp
        percolates_z = z_min_comp and z_max_comp

        if direction == "x":
            return percolates_x
        elif direction == "y":
            return percolates_y
        elif direction == "z":
            return percolates_z
        else:  # "any"
            return percolates_x or percolates_y or percolates_z

    @staticmethod
    def _fracture_touches_face(
        frac,
        bound_val: float,
        dim_idx: int,
        face_range_a: Tuple[float, float],
        face_range_b: Tuple[float, float],
    ) -> bool:
        """Check if a fracture touches a finite model boundary face.

        Performs both:
        1. Infinite plane check (distance + projected radius)
        2. Finite face rectangle overlap check

        Args:
            frac: Fracture object with .radius and .geometry (center, normal).
            bound_val: Coordinate of the boundary plane (e.g., x_min).
            dim_idx: Index of the boundary dimension (0=x, 1=y, 2=z).
            face_range_a: (min, max) for the first free dimension.
            face_range_b: (min, max) for the second free dimension.

        Returns:
            True if the fracture disk intersects the finite boundary face.
        """
        r = frac.radius if frac.radius > 0 else (frac.geometry.radius or 1.0)
        center = frac.geometry.center
        n = frac.geometry.normal

        # 1. Distance to boundary plane
        dist = abs(float(center[dim_idx]) - bound_val)
        cos_theta = abs(float(n[dim_idx]))
        proj_radius = r * math.sqrt(max(0.0, 1.0 - cos_theta**2))
        if dist > proj_radius + 1e-6:
            return False

        # 2. Finite face: check center coordinates fall within face bounds
        #    (with margin = proj_radius for the disk's extent)
        d1, d2 = (1, 2) if dim_idx == 0 else ((0, 2) if dim_idx == 1 else (0, 1))
        c1, c2 = float(center[d1]), float(center[d2])

        # The disk's footprint on the boundary face extends up to proj_radius
        # from the center's projection in all face-plane directions
        return (c1 - proj_radius <= face_range_a[1] + 1e-6 and
                c1 + proj_radius >= face_range_a[0] - 1e-6 and
                c2 - proj_radius <= face_range_b[1] + 1e-6 and
                c2 + proj_radius >= face_range_b[0] - 1e-6)

    def _component_span_directions(
        self,
        component: Set[int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
    ) -> Dict[str, bool]:
        """Check which directions a component spans (for percolation_detail)."""
        x_min_c = x_max_c = y_min_c = y_max_c = z_min_c = z_max_c = False
        for fi in component:
            f = self.fractures[fi]

            if not x_min_c:
                x_min_c = self._fracture_touches_face(f, x_range[0], 0, y_range, z_range)
            if not x_max_c:
                x_max_c = self._fracture_touches_face(f, x_range[1], 0, y_range, z_range)
            if not y_min_c:
                y_min_c = self._fracture_touches_face(f, y_range[0], 1, x_range, z_range)
            if not y_max_c:
                y_max_c = self._fracture_touches_face(f, y_range[1], 1, x_range, z_range)
            if not z_min_c:
                z_min_c = self._fracture_touches_face(f, z_range[0], 2, x_range, y_range)
            if not z_max_c:
                z_max_c = self._fracture_touches_face(f, z_range[1], 2, x_range, y_range)

        return {
            "x": x_min_c and x_max_c,
            "y": y_min_c and y_max_c,
            "z": z_min_c and z_max_c,
        }

    # ── Inter-set Matrix ──────────────────────────────────────────────────

    def inter_set_matrix(self) -> Dict[Tuple[int, int], int]:
        """Compute intersection counts between fracture sets.

        Returns:
            Dict[(set_id_a, set_id_b), count] mapping.
        """
        matrix: Dict[Tuple[int, int], int] = {}

        for i in range(self.n_fractures):
            si = self.fractures[i].set_id
            for j in self._adjacency.get(i, set()):
                if i < j:  # Count each edge once
                    sj = self.fractures[j].set_id
                    key = (min(si, sj), max(si, sj))
                    matrix[key] = matrix.get(key, 0) + 1

        return matrix

    # ── Statistics ─────────────────────────────────────────────────────────

    def statistics(self) -> Dict:
        """Compute comprehensive connectivity statistics.

        Returns:
            Dictionary of connectivity metrics.
        """
        comps = self.find_components()
        sizes = [len(c) for c in comps]

        return {
            "n_fractures": self.n_fractures,
            "n_edges": self.n_edges,
            "n_components": len(comps),
            "largest_component_size": sizes[0] if sizes else 0,
            "largest_component_fraction": sizes[0] / self.n_fractures if sizes and self.n_fractures > 0 else 0.0,
            "isolated_fractures": sum(1 for s in sizes if s == 1),
            "average_degree": self.average_degree(),
            "max_degree": max(self.degree(i) for i in range(self.n_fractures)) if self.n_fractures > 0 else 0,
            "component_size_distribution": sizes[:20],  # Top 20
        }
