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
    ) -> List[int]:
        """Find component labels that span from one model face to the opposite face.

        Checks percolation in X, Y, and Z directions.

        Args:
            x_range: (x_min, x_max) of model.
            y_range: (y_min, y_max) of model.
            z_range: (z_min, z_max) of model.

        Returns:
            List of component labels that percolate in at least one direction.
        """
        if self._component_labels is None:
            self.find_components()

        comps = self.find_components()
        percolating = []

        for label, comp in enumerate(comps):
            if self._component_percolates(comp, x_range, y_range, z_range):
                percolating.append(label)

        return percolating

    def _component_percolates(
        self,
        component: Set[int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
    ) -> bool:
        """Check if a component spans the model in any direction."""
        x_min_bound = x_max_bound = False
        y_min_bound = y_max_bound = False
        z_min_bound = z_max_bound = False

        tol = 0.01  # 1 cm tolerance

        for fi in component:
            f = self.fractures[fi]
            r = f.radius if f.radius > 0 else (f.geometry.radius or 1.0)
            cx, cy, cz = f.geometry.center_x, f.geometry.center_y, f.geometry.center_z

            # Check X bounds
            if cx - r <= x_range[0] + tol:
                x_min_bound = True
            if cx + r >= x_range[1] - tol:
                x_max_bound = True

            # Check Y bounds
            if cy - r <= y_range[0] + tol:
                y_min_bound = True
            if cy + r >= y_range[1] - tol:
                y_max_bound = True

            # Check Z bounds
            if cz - r <= z_range[0] + tol:
                z_min_bound = True
            if cz + r >= z_range[1] - tol:
                z_max_bound = True

            # Early exit if spanning found
            if (x_min_bound and x_max_bound) or (y_min_bound and y_max_bound) or (z_min_bound and z_max_bound):
                return True

        return False

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
