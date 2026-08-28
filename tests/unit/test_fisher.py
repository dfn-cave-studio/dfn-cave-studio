"""Tests for Fisher distribution sampling.

Scientific validation: SV-03 (Fisher distribution), SV-06 (seed reproducibility).
"""

import numpy as np
import pytest

from dfn_cave_studio.dfn.fisher import (
    fisher_sample,
    estimate_kappa,
    _build_tangent_basis,
)


class TestFisherSample:
    """Test Fisher distribution sampling."""

    def test_output_shape(self, fixed_seed):
        rng = np.random.default_rng(fixed_seed)
        result = fisher_sample(45, 30, 20, rng, n_samples=100)
        assert result.shape == (100, 3)

    def test_single_sample(self, fixed_seed):
        rng = np.random.default_rng(fixed_seed)
        result = fisher_sample(45, 30, 20, rng, n_samples=1)
        assert result.shape == (1, 3)
        assert np.linalg.norm(result[0]) == pytest.approx(1.0, abs=1e-12)

    def test_two_samples_have_stable_matrix_shape(self):
        result = fisher_sample(45, 30, 20, np.random.default_rng(42), n_samples=2)
        assert result.shape == (2, 3)

    def test_single_sample_seed_42_values_are_unchanged(self):
        result = fisher_sample(45, 30, 20, np.random.default_rng(42), n_samples=1)
        expected_pre_fix_vector = np.array(
            [-0.19465424770630507, -0.021412520373810362, 0.9806381737527525]
        )
        np.testing.assert_array_equal(result[0], expected_pre_fix_vector)

    def test_zero_samples_return_empty_matrix_without_advancing_rng(self):
        rng = np.random.default_rng(42)
        before = rng.bit_generator.state
        result = fisher_sample(45, 30, 20, rng, n_samples=0)
        after = rng.bit_generator.state
        assert result.shape == (0, 3)
        assert before == after

    def test_negative_sample_count_is_rejected(self):
        with pytest.raises(ValueError, match="n_samples must be non-negative"):
            fisher_sample(45, 30, 20, np.random.default_rng(42), n_samples=-1)

    def test_unit_length(self, fixed_seed):
        rng = np.random.default_rng(fixed_seed)
        samples = fisher_sample(90, 45, 20, rng, n_samples=200)
        norms = np.linalg.norm(samples, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-12)

    def test_mean_direction(self, fixed_seed):
        """The mean of many samples should be close to the input direction."""
        rng = np.random.default_rng(fixed_seed)
        dd, dip = 45, 30
        kappa = 50  # High kappa → tightly clustered
        samples = fisher_sample(dd, dip, kappa, rng, n_samples=1000)

        from dfn_cave_studio.geometry.coordinate import normal_to_dip_dir_dip
        mean_normal = np.mean(samples, axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        dd_out, dip_out = normal_to_dip_dir_dip(mean_normal)

        # With high kappa, the mean should be within ~10° of target
        dd_diff = min(abs(dd_out - dd), 360 - abs(dd_out - dd))
        assert dd_diff < 15.0, f"Mean dip direction {dd_out} too far from {dd}"
        assert abs(dip_out - dip) < 15.0

    def test_seed_reproducibility(self):
        """SV-03: Same seed → same samples."""
        seed = 42
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        samples1 = fisher_sample(90, 45, 30, rng1, n_samples=50)
        samples2 = fisher_sample(90, 45, 30, rng2, n_samples=50)

        np.testing.assert_array_almost_equal(samples1, samples2)

    def test_different_seed_different(self):
        """Different seeds → different samples."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)

        samples1 = fisher_sample(90, 45, 30, rng1, n_samples=100)
        samples2 = fisher_sample(90, 45, 30, rng2, n_samples=100)

        # Should be different (very unlikely to be identical)
        assert not np.allclose(samples1, samples2)

    def test_kappa_effect(self, fixed_seed):
        """Higher kappa → lower dispersion."""
        rng = np.random.default_rng(fixed_seed)
        samples_low_k = fisher_sample(0, 45, 5, rng, n_samples=500)
        rng = np.random.default_rng(fixed_seed + 1)
        samples_high_k = fisher_sample(0, 45, 50, rng, n_samples=500)

        # Angular deviation from mean
        from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal
        mean_n = dip_dir_dip_to_normal(0, 45)

        def angular_deviation(samples, mean):
            dots = np.clip(np.abs(np.dot(samples, mean)), 0.0, 1.0)
            return np.degrees(np.arccos(dots))

        dev_low = angular_deviation(samples_low_k, mean_n)
        dev_high = angular_deviation(samples_high_k, mean_n)

        # High kappa should have lower mean deviation
        assert np.mean(dev_high) < np.mean(dev_low)

    def test_upper_hemisphere(self, fixed_seed):
        """All sampled normals should be upper-hemisphere (Z >= 0)."""
        rng = np.random.default_rng(fixed_seed)
        for dd in [0, 90, 180, 270]:
            for dip in [15, 45, 75]:
                samples = fisher_sample(dd, dip, 20, rng, n_samples=50)
                assert np.all(samples[:, 2] >= -1e-12)

    def test_invalid_kappa(self):
        with pytest.raises(ValueError):
            fisher_sample(0, 0, -1, n_samples=10)


class TestEstimateKappa:
    """Test Fisher kappa estimation."""

    def test_high_concentration(self, fixed_seed):
        """Tightly clustered samples → high kappa."""
        rng = np.random.default_rng(fixed_seed)
        samples = fisher_sample(90, 60, 100, rng, n_samples=500)
        k_est = estimate_kappa(samples)
        assert k_est > 30  # Should be high

    def test_low_concentration(self, fixed_seed):
        """Dispersed samples → low kappa."""
        rng = np.random.default_rng(fixed_seed)
        samples = fisher_sample(90, 60, 5, rng, n_samples=500)
        k_est = estimate_kappa(samples)
        assert k_est < 30  # Should be moderate/low

    def test_insufficient_samples(self):
        with pytest.raises(ValueError):
            estimate_kappa(np.array([[1, 0, 0]]))


class TestTangentBasis:
    def test_orthonormal(self):
        from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal
        for dd in [0, 45, 90, 135]:
            for dip in [10, 45, 80]:
                w = dip_dir_dip_to_normal(dd, dip)
                u, v = _build_tangent_basis(w)
                # Unit length
                assert abs(np.linalg.norm(u) - 1) < 1e-12
                assert abs(np.linalg.norm(v) - 1) < 1e-12
                # Orthogonal
                assert abs(np.dot(u, v)) < 1e-12
                assert abs(np.dot(u, w)) < 1e-12
                assert abs(np.dot(v, w)) < 1e-12
