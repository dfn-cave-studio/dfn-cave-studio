"""Tests for coordinate transforms and orientation conversions.

Scientific validation cases SV-01, SV-02, SV-03.
"""

import math
import numpy as np
import pytest

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


class TestDipDirDipToNormal:
    """Test dip direction/dip → normal vector conversion."""

    def test_horizontal_fracture(self):
        """Dip=0 → normal points straight up (Z=1)."""
        n = dip_dir_dip_to_normal(45, 0)
        np.testing.assert_array_almost_equal(n, [0, 0, 1], decimal=10)

    def test_vertical_north_dipping(self):
        """Dip direction 0 (north), dip=90 → vertical fracture."""
        n = dip_dir_dip_to_normal(0, 90)
        # Should be [-sin(0)*1, -cos(0)*1, 0] = [0, -1, 0]
        np.testing.assert_array_almost_equal(n, [0, -1, 0], decimal=10)

    def test_vertical_east_dipping(self):
        """Dip direction 90 (east), dip=90."""
        n = dip_dir_dip_to_normal(90, 90)
        # [-sin(90)*1, -cos(90)*1, 0] = [-1, 0, 0]
        np.testing.assert_array_almost_equal(n, [-1, 0, 0], decimal=10)

    def test_45_degree_dip(self):
        """Intermediate dip test."""
        n = dip_dir_dip_to_normal(0, 45)
        # [-sin(0)*sin(45), -cos(0)*sin(45), cos(45)]
        expected = [0, -math.sqrt(2) / 2, math.sqrt(2) / 2]
        np.testing.assert_array_almost_equal(n, expected, decimal=10)

    def test_upper_hemisphere(self):
        """All normals should have Z >= 0."""
        for dd in [0, 45, 90, 135, 180, 225, 270, 315]:
            for dip in [15, 30, 45, 60, 75]:
                n = dip_dir_dip_to_normal(dd, dip)
                assert n[2] >= -1e-12, f"Z negative for dd={dd}, dip={dip}"

    def test_unit_length(self):
        """All outputs should be unit vectors."""
        for dd in range(0, 360, 30):
            for dip in range(0, 91, 30):
                n = dip_dir_dip_to_normal(dd, dip)
                assert abs(np.linalg.norm(n) - 1.0) < 1e-12


class TestNormalToDipDirDip:
    """Test normal vector → dip direction/dip conversion."""

    def test_round_trip_random(self, fixed_seed: int):
        """SV-01: Round-trip conversion for random orientations."""
        rng = np.random.default_rng(fixed_seed)
        for _ in range(100):
            dd_in = rng.uniform(0, 360)
            dip_in = rng.uniform(0, 90)
            n = dip_dir_dip_to_normal(dd_in, dip_in)
            dd_out, dip_out = normal_to_dip_dir_dip(n)

            # Check tolerance
            dd_diff = min(abs(dd_out - dd_in), 360 - abs(dd_out - dd_in))
            assert dd_diff < 1e-9, f"Dip direction mismatch: {dd_in} → {dd_out}"
            assert abs(dip_out - dip_in) < 1e-9, f"Dip mismatch: {dip_in} → {dip_out}"

    def test_horizontal(self):
        """Horizontal fractures: dip=0, direction undefined → 0."""
        dd, dip = normal_to_dip_dir_dip([0, 0, 1])
        assert dip == 0.0

    def test_vertical(self):
        """Vertical fracture."""
        dd, dip = normal_to_dip_dir_dip([0, -1, 0])
        assert abs(dd) < 1e-9
        assert abs(dip - 90) < 1e-9

    def test_lower_hemisphere_flipped(self):
        """Lower hemisphere normal should be flipped to upper."""
        dd, dip = normal_to_dip_dir_dip([0, 0, -1])
        assert dip == 0.0  # Should be treated as horizontal

    def test_known_values(self):
        """Test known conversion values."""
        # East-dipping 30° → dip_dir=90, dip=30
        n = dip_dir_dip_to_normal(90, 30)
        dd, dip = normal_to_dip_dir_dip(n)
        assert abs(dd - 90) < 1e-9
        assert abs(dip - 30) < 1e-9


class TestNormalToPole:
    """Test pole vector conversion for stereographic projection."""

    def test_pole_points_down(self):
        """Pole should point to lower hemisphere."""
        n = np.array([0, 0, 1])  # Horizontal fracture
        pole = normal_to_pole(n)
        assert pole[2] <= 0


class TestRotationMatrices:
    """Test rotation matrices."""

    def test_rx_90(self):
        """90° rotation about X maps Y → Z."""
        R = rotation_matrix_x(math.pi / 2)
        v = np.array([0, 1, 0])
        result = R @ v
        np.testing.assert_array_almost_equal(result, [0, 0, 1])

    def test_rz_90(self):
        """90° rotation about Z maps X → Y."""
        R = rotation_matrix_z(math.pi / 2)
        v = np.array([1, 0, 0])
        result = R @ v
        np.testing.assert_array_almost_equal(result, [0, 1, 0])

    def test_determinant_is_1(self):
        """Rotation matrices should have determinant 1."""
        for R in [rotation_matrix_x(0.5), rotation_matrix_y(0.5), rotation_matrix_z(0.5)]:
            assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_rotate_vector(self):
        """Test rotate_vector helper."""
        v = np.array([1, 0, 0])
        result = rotate_vector(v, 'z', math.pi / 2)
        np.testing.assert_array_almost_equal(result, [0, 1, 0])

    def test_rotate_invalid_axis(self):
        with pytest.raises(ValueError):
            rotate_vector([1, 0, 0], 'w', 0.5)


class TestStrikeConversion:
    """Test strike ↔ dip direction conversions."""

    def test_dip_dir_to_strike_rh(self):
        """Right-hand rule: strike = dip_dir - 90."""
        strike = dip_dir_to_strike(90, right_hand_rule=True)
        assert abs(strike) < 1e-9  # 0°

    def test_strike_dip_to_normal_rh(self):
        """Strike 0, dip 90 east → dip_dir=90, dip=90."""
        n = strike_dip_to_normal(0, 90, right_hand_rule=True)
        # dip_dir=90, dip=90 → n = [-sin(90), -cos(90), 0] = [-1, 0, 0]
        np.testing.assert_array_almost_equal(n, [-1, 0, 0], decimal=10)


class TestBatchOperations:
    """Test batch conversion functions."""

    def test_batch_round_trip(self, fixed_seed: int):
        """SV-01 batch: Round-trip for multiple orientations."""
        rng = np.random.default_rng(fixed_seed)
        N = 50
        dd_in = rng.uniform(0, 360, N)
        dip_in = rng.uniform(0, 90, N)

        normals = batch_dip_dir_dip_to_normal(dd_in, dip_in)
        dd_out, dip_out = batch_normal_to_dip_dir_dip(normals)

        dd_diff = np.minimum(np.abs(dd_out - dd_in), 360 - np.abs(dd_out - dd_in))
        assert np.all(dd_diff < 1e-9)
        assert np.all(np.abs(dip_out - dip_in) < 1e-9)

    def test_batch_output_shapes(self):
        """Batch functions should return correct shapes."""
        N = 10
        dd = np.linspace(0, 350, N)
        dip = np.linspace(10, 80, N)

        normals = batch_dip_dir_dip_to_normal(dd, dip)
        assert normals.shape == (N, 3)

        dd_out, dip_out = batch_normal_to_dip_dir_dip(normals)
        assert dd_out.shape == (N,)
        assert dip_out.shape == (N,)
