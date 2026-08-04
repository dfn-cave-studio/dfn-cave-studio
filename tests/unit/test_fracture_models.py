"""Tests for fracture and fracture set data models."""

import math
import numpy as np
import pytest

from dfn_cave_studio.models.fracture import (
    FractureGeometry,
    DeterministicFracture,
    StochasticFracture,
    create_fracture_from_dip,
)
from dfn_cave_studio.models.fracture_set import (
    OrientationDistribution,
    SizeDistribution,
    JointSetConfig,
    SpatialDistribution,
)
from dfn_cave_studio.models.enums import (
    SizeDistributionType,
    FractureType,
    FractureSource,
)


class TestFractureGeometry:
    def test_default_disk(self):
        geo = FractureGeometry()
        assert geo.geometry_type == "disk"
        np.testing.assert_array_equal(geo.center, [0, 0, 0])
        np.testing.assert_array_equal(geo.normal, [0, 0, 1])

    def test_custom_geometry(self):
        geo = FractureGeometry(
            geometry_type="disk",
            center_x=10, center_y=20, center_z=30,
            radius=5.0,
            normal_x=0, normal_y=0, normal_z=1,
            dip_direction=0, dip=0,
            area=math.pi * 25,
        )
        assert geo.radius == 5.0
        assert geo.area == math.pi * 25


class TestDeterministicFracture:
    def test_default_creation(self):
        f = DeterministicFracture()
        assert f.fracture_type == FractureType.JOINT
        assert f.source == FractureSource.USER_DEFINED
        assert f.confidence == 1.0

    def test_with_provenance(self):
        f = DeterministicFracture(
            name="Fault F1",
            fracture_type=FractureType.FAULT,
            data_source="Surface mapping - East pit",
            data_quality="measured",
            confidence=0.95,
        )
        assert f.name == "Fault F1"
        assert f.fracture_type == FractureType.FAULT
        assert f.data_source == "Surface mapping - East pit"
        assert f.confidence == 0.95


class TestStochasticFracture:
    def test_create_from_dip(self):
        f = create_fracture_from_dip(
            center=(50, 50, 50),
            dip_direction_deg=45,
            dip_deg=60,
            radius=5.0,
            set_id=1,
            realization_id=0,
            random_seed=42,
        )
        assert f.set_id == 1
        assert f.radius == 5.0
        assert f.source == FractureSource.STOCHASTIC
        assert f.random_seed == 42
        assert f.geometry.radius == 5.0

    def test_create_fracture_area(self):
        f = create_fracture_from_dip(
            center=(0, 0, 0), dip_direction_deg=90, dip_deg=30, radius=3.0
        )
        expected_area = math.pi * 9.0
        assert abs(f.geometry.area - expected_area) < 1e-10


class TestOrientationDistribution:
    def test_defaults(self):
        od = OrientationDistribution()
        assert od.mean_dip_direction == 0.0
        assert od.mean_dip == 0.0
        assert od.kappa == 20.0

    def test_high_kappa_std(self):
        """High kappa = tightly clustered → small std."""
        od = OrientationDistribution(kappa=100)
        std = od.angular_std_dev_deg
        assert std < 15  # Should be about 8.1°

    def test_low_kappa_std(self):
        """Low kappa = dispersed → large std."""
        od = OrientationDistribution(kappa=5)
        std = od.angular_std_dev_deg
        assert std > 20


class TestSizeDistribution:
    def test_defaults(self):
        sd = SizeDistribution()
        assert sd.distribution_type == SizeDistributionType.LOGNORMAL
        assert sd.min_radius == 0.5
        assert sd.max_radius == 10.0

    def test_validation_min_max(self):
        with pytest.raises(ValueError):
            SizeDistribution(min_radius=10.0, max_radius=5.0)

    def test_lognormal_mean(self):
        """Mean of lognormal: exp(μ + σ²/2)."""
        sd = SizeDistribution(lognormal_mu=1.0, lognormal_sigma=0.01)
        # exp(1.0 + 0.0001/2) ≈ e
        assert abs(sd.mean_radius - math.e) < 0.1

    def test_fixed_mean(self):
        sd = SizeDistribution(
            distribution_type=SizeDistributionType.FIXED,
            min_radius=2.99,
            max_radius=3.0,
        )
        assert abs(sd.mean_radius - 3.0) < 0.5


class TestJointSetConfig:
    def test_defaults(self):
        js = JointSetConfig(set_id=0, name="Test Set")
        assert js.target_p32 == 1.0
        assert js.orientation.kappa == 20.0

    def test_expected_mean_area(self):
        js = JointSetConfig(
            set_id=1,
            size=SizeDistribution(
                distribution_type=SizeDistributionType.FIXED,
                min_radius=1.99,
                max_radius=2.0,
            )
        )
        # FIXED distribution mean is (min + max) / 2 = 1.995
        # Expected area ≈ π * 1.995² ≈ 12.5
        expected = math.pi * 1.995 ** 2
        assert abs(js.expected_mean_area() - expected) < 0.1

    def test_expected_fracture_count(self):
        js = JointSetConfig(
            set_id=1,
            target_p32=2.0,
            size=SizeDistribution(
                distribution_type=SizeDistributionType.FIXED,
                min_radius=0.99,
                max_radius=1.0,
            )
        )
        # mean_area ≈ π * 0.995² ≈ 3.11
        # N = 2 * 1000 / 3.11 ≈ 643
        count = js.expected_fracture_count(1000.0)
        assert count > 0
        assert count < 10000  # reasonable range
