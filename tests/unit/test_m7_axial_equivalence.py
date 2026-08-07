"""Test axial equivalence: n and -n represent the same fracture plane."""

import numpy as np
import pytest
from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal


class TestAxialEquivalence:
    """Joint set clustering must treat n and -n as the same orientation."""

    def test_n_and_neg_n_same_cosine_abs(self):
        """abs(dot(n1, n2)) is identical for n2 and -n2."""
        n1 = dip_dir_dip_to_normal(45, 60)
        n2 = dip_dir_dip_to_normal(225, 60)  # opposite dip direction = -n1 approx
        cos_abs_1 = abs(np.dot(n1, n2))
        cos_abs_2 = abs(np.dot(n1, -n2))
        assert abs(cos_abs_1 - cos_abs_2) < 1e-10

    def test_n_and_neg_n_are_same_fracture_plane(self):
        """abs(dot(n, -n)) = 1.0 — n and -n represent the identical fracture plane."""
        n = dip_dir_dip_to_normal(45, 60)
        cos_dist = 1.0 - abs(np.dot(n, -n))
        assert cos_dist < 0.01, f"n and -n must be same plane, dist={cos_dist}"

    def test_angular_difference_uses_abs_dot(self):
        """Two directions 120° apart on sphere become 60° with axial equivalence."""
        n1 = dip_dir_dip_to_normal(0, 60)
        n2 = dip_dir_dip_to_normal(180, 60)
        # Without axial: dot ≈ -0.5, distance ≈ 1.5
        # With axial: abs(dot) ≈ 0.5, distance ≈ 0.5  (60° apart)
        axial_cos_dist = 1.0 - abs(np.dot(n1, n2))
        assert 0.3 < axial_cos_dist < 0.7, f"Should be ~60° apart axially, got {axial_cos_dist:.3f}"

    def test_perpendicular_planes_are_far(self):
        """Orthogonal fractures should have abs(dot) ≈ 0."""
        n1 = dip_dir_dip_to_normal(0, 0)    # horizontal
        n2 = dip_dir_dip_to_normal(0, 90)   # vertical E-W
        cos_dist = 1.0 - abs(np.dot(n1, n2))
        assert cos_dist > 0.9  # Should be far apart

    def test_359_and_1_wrap(self):
        """359° and 1° are nearly identical directions."""
        n1 = dip_dir_dip_to_normal(359, 60)
        n2 = dip_dir_dip_to_normal(1, 60)
        cos_dist = 1.0 - abs(np.dot(n1, n2))
        assert cos_dist < 0.01
