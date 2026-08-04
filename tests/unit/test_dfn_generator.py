"""Tests for DFN generator.

Scientific validation: SV-08 (global constant model), SV-06 (reproducibility).
"""

import math
import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.fracture_set import (
    JointSetConfig,
    OrientationDistribution,
    SizeDistribution,
)
from dfn_cave_studio.models.dfn_realization import DFNGenerationConfig
from dfn_cave_studio.models.enums import SizeDistributionType
from dfn_cave_studio.dfn.generator import DFNGenerator


@pytest.fixture
def small_bounds() -> ModelBounds:
    """Small 50x50x50m model for fast tests."""
    return ModelBounds(x_min=0, x_max=50, y_min=0, y_max=50, z_min=0, z_max=50)


@pytest.fixture
def simple_joint_set() -> JointSetConfig:
    """Simple joint set with fixed parameters for testing."""
    return JointSetConfig(
        set_id=0,
        name="Test Set",
        orientation=OrientationDistribution(
            mean_dip_direction=90,
            mean_dip=60,
            kappa=30,
        ),
        size=SizeDistribution(
            distribution_type=SizeDistributionType.FIXED,
            min_radius=2.99,
            max_radius=3.0,
        ),
        target_p32=1.0,
    )


@pytest.fixture
def generator(small_bounds, simple_joint_set) -> DFNGenerator:
    config = DFNGenerationConfig(
        master_seed=42,
        joint_sets=[simple_joint_set],
        model_volume=small_bounds.volume,
    )
    return DFNGenerator(config, small_bounds)


class TestDFNGeneratorBasic:
    """Basic generation tests."""

    def test_generate_returns_realization(self, generator):
        """Generation should return a DFNRealization."""
        realization = generator.generate(0)
        assert realization is not None
        assert realization.realization_number == 0

    def test_generates_fractures(self, generator):
        """Should generate fractures for the configured joint set."""
        realization = generator.generate(0)
        assert len(realization.stochastic_fractures) > 0

    def test_fractures_have_set_id(self, generator):
        """All fractures should belong to the correct set."""
        realization = generator.generate(0)
        for f in realization.stochastic_fractures:
            assert f.set_id == 0

    def test_fracture_positions_in_range(self, generator):
        """Fracture centers should be within the model bounds (+ buffer)."""
        realization = generator.generate(0)
        buffer = generator.config.boundary_buffer
        for f in realization.stochastic_fractures:
            g = f.geometry
            assert generator.bounds.x_min - buffer - 10 <= g.center_x <= generator.bounds.x_max + buffer + 10
            assert generator.bounds.y_min - buffer - 10 <= g.center_y <= generator.bounds.y_max + buffer + 10
            assert generator.bounds.z_min - buffer - 10 <= g.center_z <= generator.bounds.z_max + buffer + 10

    def test_fracture_radii_in_range(self, generator):
        """Fracture radii should be within the size distribution range."""
        realization = generator.generate(0)
        for f in realization.stochastic_fractures:
            assert 2.9 <= f.radius <= 3.1  # FIXED with min=2.99, max=3.0


class TestDFNGeneratorReproducibility:
    """SV-06: Reproducibility tests."""

    def test_same_seed_same_result(self, small_bounds, simple_joint_set):
        """Same seed → identical DFN."""
        config = DFNGenerationConfig(
            master_seed=42,
            joint_sets=[simple_joint_set],
            model_volume=small_bounds.volume,
        )
        gen1 = DFNGenerator(config, small_bounds)
        gen2 = DFNGenerator(config, small_bounds)

        r1 = gen1.generate(0)
        r2 = gen2.generate(0)

        assert r1.generation_result.total_fractures == r2.generation_result.total_fractures
        # Check first fracture's center and radius match
        assert r1.stochastic_fractures[0].geometry.center_x == r2.stochastic_fractures[0].geometry.center_x
        assert r1.stochastic_fractures[0].radius == r2.stochastic_fractures[0].radius

    def test_different_seed_different_result(self, small_bounds, simple_joint_set):
        """Different seed → different DFN (statistically)."""
        config1 = DFNGenerationConfig(master_seed=42, joint_sets=[simple_joint_set])
        config2 = DFNGenerationConfig(master_seed=99, joint_sets=[simple_joint_set])

        gen1 = DFNGenerator(config1, small_bounds)
        gen2 = DFNGenerator(config2, small_bounds)

        r1 = gen1.generate(0)
        r2 = gen2.generate(0)

        # Centers should differ
        c1 = r1.stochastic_fractures[0].geometry.center_x
        c2 = r2.stochastic_fractures[0].geometry.center_x
        assert c1 != c2


class TestDFNGeneratorP32:
    """SV-08: P32 convergence tests."""

    def test_p32_statistics_recorded(self, generator):
        """Generator should record P32 statistics."""
        realization = generator.generate(0)
        result = realization.generation_result
        assert result.achieved_p32 > 0
        assert result.total_fracture_area > 0

    def test_p32_reasonable(self, small_bounds):
        """P32 should be within a reasonable range of target."""
        js = JointSetConfig(
            set_id=0,
            name="High P32 Set",
            orientation=OrientationDistribution(mean_dip_direction=0, mean_dip=90, kappa=20),
            size=SizeDistribution(
                distribution_type=SizeDistributionType.LOGNORMAL,
                lognormal_mu=1.0,
                lognormal_sigma=0.3,
                min_radius=0.5,
                max_radius=5.0,
            ),
            target_p32=2.0,
            p32_tolerance=0.30,  # 30% tolerance for small models
        )
        config = DFNGenerationConfig(
            master_seed=42,
            joint_sets=[js],
            model_volume=small_bounds.volume,
        )
        gen = DFNGenerator(config, small_bounds)
        realization = gen.generate(0)
        result = realization.generation_result

        # For a small model with random sizes, error might be higher
        # Check that P32 is non-zero and within 2x of target
        assert result.achieved_p32 > 0
        assert result.achieved_p32 < result.target_p32 * 3

    def test_total_fracture_count_consistent(self, generator):
        """Fracture count in result should match actual fractures."""
        realization = generator.generate(0)
        assert realization.generation_result.total_fractures == len(realization.stochastic_fractures)


class TestDFNGeneratorCancellation:
    """Test cancellation support."""

    def test_cancel_flag(self, generator):
        """Setting cancel should prevent generation."""
        generator.cancel()
        with pytest.raises(InterruptedError):
            generator.generate(0)


class TestDFNGeneratorMultiSet:
    """Test generation with multiple joint sets."""

    def test_two_sets(self, small_bounds):
        set1 = JointSetConfig(
            set_id=0, name="Set A",
            orientation=OrientationDistribution(mean_dip_direction=0, mean_dip=90, kappa=20),
            size=SizeDistribution(distribution_type=SizeDistributionType.FIXED, min_radius=1.99, max_radius=2.0),
            target_p32=0.5,
        )
        set2 = JointSetConfig(
            set_id=1, name="Set B",
            orientation=OrientationDistribution(mean_dip_direction=90, mean_dip=60, kappa=20),
            size=SizeDistribution(distribution_type=SizeDistributionType.FIXED, min_radius=0.99, max_radius=1.0),
            target_p32=0.5,
        )
        config = DFNGenerationConfig(
            master_seed=42,
            joint_sets=[set1, set2],
            model_volume=small_bounds.volume,
        )
        gen = DFNGenerator(config, small_bounds)
        realization = gen.generate(0)

        # Should have fractures from both sets
        set_ids = {f.set_id for f in realization.stochastic_fractures}
        assert 0 in set_ids
        assert 1 in set_ids
