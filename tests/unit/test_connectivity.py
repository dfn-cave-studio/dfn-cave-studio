"""Tests for fracture connectivity graph (M5)."""

import math
import numpy as np
import pytest

from dfn_cave_studio.models.fracture import create_fracture_from_dip
from dfn_cave_studio.models.dfn_realization import DFNRealization
from dfn_cave_studio.connectivity.connectivity_graph import ConnectivityGraph


@pytest.fixture
def empty_graph():
    real = DFNRealization(realization_number=0)
    return ConnectivityGraph(real)


@pytest.fixture
def two_intersecting():
    """Two intersecting orthogonal fractures."""
    f1 = create_fracture_from_dip((0, 0, 0), 90, 90, 5.0, set_id=1)  # EW vertical
    f2 = create_fracture_from_dip((0, 0, 0), 0, 90, 5.0, set_id=2)    # NS vertical
    real = DFNRealization(realization_number=0, stochastic_fractures=[f1, f2])
    return ConnectivityGraph(real)


@pytest.fixture
def two_separated():
    """Two fractures far apart."""
    f1 = create_fracture_from_dip((0, 0, 0), 90, 90, 1.0, set_id=1)
    f2 = create_fracture_from_dip((100, 100, 100), 0, 90, 1.0, set_id=2)
    real = DFNRealization(realization_number=0, stochastic_fractures=[f1, f2])
    return ConnectivityGraph(real)


@pytest.fixture
def chain_of_three():
    """Three fractures: f1-f2 intersect, f2-f3 intersect, f1-f3 don't."""
    f1 = create_fracture_from_dip((0, 0, 0), 90, 90, 5.0, set_id=1)
    f2 = create_fracture_from_dip((0, 0, 0), 0, 90, 5.0, set_id=2)
    f3 = create_fracture_from_dip((0, 0, 0), 45, 45, 5.0, set_id=1)
    # f1 and f2 intersect at origin, f1 and f3 also intersect, f2 and f3 also intersect
    real = DFNRealization(realization_number=0, stochastic_fractures=[f1, f2, f3])
    return ConnectivityGraph(real)


class TestConnectivityGraph:
    def test_empty_graph(self, empty_graph):
        assert empty_graph.n_fractures == 0
        assert empty_graph.average_degree() == 0.0

    def test_two_intersecting(self, two_intersecting):
        g = two_intersecting
        n_edges = g.compute_edges()
        assert n_edges >= 1  # Should find the intersection
        assert g.n_edges >= 1

    def test_two_separated(self, two_separated):
        g = two_separated
        n_edges = g.compute_edges()
        assert n_edges == 0

    def test_components_two_separated(self, two_separated):
        g = two_separated
        g.compute_edges()
        comps = g.find_components()
        assert len(comps) == 2  # Two isolated fractures
        assert g.n_components() == 2
        assert g.isolated_fractures() == 2

    def test_components_two_intersecting(self, two_intersecting):
        g = two_intersecting
        g.compute_edges()
        comps = g.find_components()
        assert len(comps) == 1  # One connected component
        assert g.largest_component_size() == 2

    def test_largest_component_fraction(self, two_intersecting):
        g = two_intersecting
        g.compute_edges()
        frac = g.largest_component_fraction()
        assert frac == 1.0  # All fractures in one component

    def test_degree(self, two_intersecting):
        g = two_intersecting
        g.compute_edges()
        # Both fractures should have degree 1 (connected to each other)
        assert g.degree(0) >= 1
        assert g.degree(1) >= 1

    def test_component_labels(self, two_separated):
        g = two_separated
        g.compute_edges()
        labels = g.component_labels()
        assert labels[0] != labels[1]  # Different components

    def test_inter_set_matrix(self, two_intersecting):
        g = two_intersecting
        g.compute_edges()
        matrix = g.inter_set_matrix()
        # Should have edge between set 1 and set 2
        assert (1, 2) in matrix

    def test_statistics(self, two_intersecting):
        g = two_intersecting
        g.compute_edges()
        stats = g.statistics()
        assert "n_fractures" in stats
        assert "n_edges" in stats
        assert "largest_component_size" in stats

    def test_percolation(self, two_intersecting):
        g = two_intersecting
        g.compute_edges()
        percolating = g.find_percolating_clusters(
            (-50, 50), (-50, 50), (-50, 50),
        )
        assert isinstance(percolating, list)
