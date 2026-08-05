"""Integration tests for DFN generation → intersection → connectivity pipeline."""

import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.enums import SizeDistributionType
from dfn_cave_studio.models.fracture_set import (
    JointSetConfig, OrientationDistribution, SizeDistribution,
)
from dfn_cave_studio.models.dfn_realization import DFNGenerationConfig
from dfn_cave_studio.dfn.generator import DFNGenerator


class TestDFNPipeline:
    """DFN generation with statistical verification."""

    @pytest.fixture
    def small_bounds(self):
        return ModelBounds(x_min=0, x_max=50, y_min=0, y_max=50, z_min=0, z_max=50)

    @pytest.fixture
    def fixed_radius_set(self):
        return JointSetConfig(
            set_id=1,
            name="Test Set",
            orientation=OrientationDistribution(
                mean_dip_direction=45, mean_dip=30, kappa=50,
            ),
            size=SizeDistribution(
                distribution_type=SizeDistributionType.FIXED,
                min_radius=2.99, max_radius=3.01,
            ),
            target_p32=1.0,
        )

    def test_generation_produces_fractures(self, small_bounds, fixed_radius_set):
        """DFN generation produces the expected number of fractures."""
        config = DFNGenerationConfig(
            joint_sets=[fixed_radius_set],
            master_seed=42,
        )
        gen = DFNGenerator(config, small_bounds)
        realization = gen.generate(realization_number=0)

        assert realization.total_p32 > 0
        assert len(realization.stochastic_fractures) > 0
        # With fixed r=3m, area=π*9=28.27, volume=125000m³, target P32=1.0
        # Expected count = 1.0 * 125000 / 28.27 ≈ 4421
        n = len(realization.stochastic_fractures)
        assert 4000 < n < 5000, f"Expected ~4421 fractures, got {n}"

    def test_reproducibility(self, small_bounds, fixed_radius_set):
        """Same seed must produce identical results."""
        config = DFNGenerationConfig(
            joint_sets=[fixed_radius_set],
            master_seed=42,
        )
        gen1 = DFNGenerator(config, small_bounds)
        r1 = gen1.generate()

        gen2 = DFNGenerator(config, small_bounds)
        r2 = gen2.generate()

        assert r1.total_p32 == r2.total_p32
        assert len(r1.stochastic_fractures) == len(r2.stochastic_fractures)
        # Verify first fracture is identical
        f1 = r1.stochastic_fractures[0]
        f2 = r2.stochastic_fractures[0]
        assert f1.geometry.center_x == f2.geometry.center_x
        assert f1.geometry.center_y == f2.geometry.center_y
        assert f1.geometry.center_z == f2.geometry.center_z

    def test_different_seeds_different_results(self, small_bounds, fixed_radius_set):
        """Different seeds produce geometrically different fractures."""
        config1 = DFNGenerationConfig(joint_sets=[fixed_radius_set], master_seed=42)
        config2 = DFNGenerationConfig(joint_sets=[fixed_radius_set], master_seed=99)

        gen1 = DFNGenerator(config1, small_bounds)
        r1 = gen1.generate()

        gen2 = DFNGenerator(config2, small_bounds)
        r2 = gen2.generate()

        # P32 values should be statistically close (same distribution)
        p32_ratio = max(r1.total_p32, r2.total_p32) / min(r1.total_p32, r2.total_p32)
        assert p32_ratio < 1.02, f"P32 too different: {r1.total_p32} vs {r2.total_p32}"

        # Fracture positions should differ
        positions1 = np.array([[f.geometry.center_x, f.geometry.center_y, f.geometry.center_z]
                               for f in r1.stochastic_fractures])
        positions2 = np.array([[f.geometry.center_x, f.geometry.center_y, f.geometry.center_z]
                               for f in r2.stochastic_fractures])
        assert not np.allclose(positions1, positions2)

    def test_multi_set_generation(self, small_bounds):
        """Multiple joint sets generate fractures with correct set_ids."""
        set1 = JointSetConfig(
            set_id=1, name="Set A",
            orientation=OrientationDistribution(mean_dip_direction=0, mean_dip=90, kappa=30),
            size=SizeDistribution(distribution_type=SizeDistributionType.FIXED,
                                  min_radius=1.99, max_radius=2.01),
            target_p32=0.5,
        )
        set2 = JointSetConfig(
            set_id=2, name="Set B",
            orientation=OrientationDistribution(mean_dip_direction=90, mean_dip=0, kappa=30),
            size=SizeDistribution(distribution_type=SizeDistributionType.FIXED,
                                  min_radius=1.99, max_radius=2.01),
            target_p32=0.5,
        )

        config = DFNGenerationConfig(joint_sets=[set1, set2], master_seed=42)
        gen = DFNGenerator(config, small_bounds)
        realization = gen.generate()

        set_ids = set(f.set_id for f in realization.stochastic_fractures)
        assert set_ids == {1, 2}

        count1 = sum(1 for f in realization.stochastic_fractures if f.set_id == 1)
        count2 = sum(1 for f in realization.stochastic_fractures if f.set_id == 2)
        assert count1 > 0
        assert count2 > 0

    def test_p32_convergence_fixed_size(self, small_bounds):
        """P32 should converge to target within tolerance for fixed-size distribution."""
        joint_set = JointSetConfig(
            set_id=1, name="Test",
            orientation=OrientationDistribution(mean_dip_direction=45, mean_dip=30, kappa=50),
            size=SizeDistribution(
                distribution_type=SizeDistributionType.FIXED,
                min_radius=2.99, max_radius=3.01,
            ),
            target_p32=1.0,
            p32_tolerance=0.05,
        )
        config = DFNGenerationConfig(joint_sets=[joint_set], master_seed=42)
        gen = DFNGenerator(config, small_bounds)
        realization = gen.generate()

        p32 = realization.total_p32
        assert 0.90 < p32 < 1.10, f"P32 {p32} outside ±10% of target 1.0"

    def test_lognormal_size_produces_variable_radii(self, small_bounds):
        """Lognormal size distribution produces variable fracture radii."""
        joint_set = JointSetConfig(
            set_id=1, name="Lognormal",
            orientation=OrientationDistribution(mean_dip_direction=45, mean_dip=30, kappa=50),
            size=SizeDistribution(
                distribution_type=SizeDistributionType.LOGNORMAL,
                lognormal_mu=1.0, lognormal_sigma=0.5,
                min_radius=0.5, max_radius=10.0,
            ),
            target_p32=1.0,
        )
        config = DFNGenerationConfig(joint_sets=[joint_set], master_seed=42)
        gen = DFNGenerator(config, small_bounds)
        realization = gen.generate()

        radii = [f.radius for f in realization.stochastic_fractures]
        assert len(set(round(r, 2) for r in radii[:50])) > 5  # Variable radii
        # Check E[πR²] is correctly computed (not π·E[R]²)
        expected_mean_area = joint_set.expected_mean_area()
        mean_r = joint_set.size.mean_radius
        naive_area = np.pi * mean_r ** 2
        # With σ=0.5, Jensen correction should be exp(σ²) ≈ 1.284×
        assert expected_mean_area > naive_area * 1.1, \
            f"E[πR²]={expected_mean_area:.3f} should be > π·E[R]²={naive_area:.3f}"


class TestP32Calculation:
    """Verify E[πR²] vs π(E[R])² correction."""

    def test_lognormal_jensens_gap(self):
        """Lognormal: E[R²] = exp(2μ+2σ²) > (E[R])² = exp(2μ+σ²)."""
        sd = SizeDistribution(
            distribution_type=SizeDistributionType.LOGNORMAL,
            lognormal_mu=1.0,
            lognormal_sigma=0.5,
            min_radius=0.5, max_radius=10.0,
        )
        mean_r = sd.mean_radius
        mean_r2 = sd.mean_squared_radius

        naive_r2 = mean_r ** 2
        ratio = mean_r2 / naive_r2
        # exp(σ²) = exp(0.25) ≈ 1.284
        expected_ratio = np.exp(0.5 ** 2)
        assert abs(ratio - expected_ratio) < 0.01

    def test_fixed_radius_no_jensens_gap(self):
        """Fixed radius: E[R²] = (E[R])² = r²."""
        sd = SizeDistribution(
            distribution_type=SizeDistributionType.FIXED,
            min_radius=2.999, max_radius=3.001,
        )
        # E[R] ≈ 3.0, E[R²] ≈ 9.0, the gap should be tiny for near-fixed radius
        assert abs(sd.mean_squared_radius - 9.0) < 0.01
        # Jensen gap should be negligible
        jensen_gap = abs(sd.mean_radius ** 2 - sd.mean_squared_radius)
        assert jensen_gap < 0.001
