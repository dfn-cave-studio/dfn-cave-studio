"""Tests for core configuration module."""

import pytest

from dfn_cave_studio.core.config import (
    AppConfig,
    AppVersion,
    UnitConfig,
    RandomConfig,
    UIConfig,
    PerformanceConfig,
    LoggingConfig,
    CoordinateSystem,
    AngleUnit,
    LengthUnit,
    get_config,
    set_config,
    reset_config,
)


class TestAppVersion:
    """Test application version tracking."""

    def test_default_version(self):
        v = AppVersion()
        assert v.major == 0
        assert v.minor == 1
        assert v.patch == 0
        assert v.milestone == "M0"

    def test_version_string(self):
        v = AppVersion(major=1, minor=2, patch=3, milestone="M5")
        assert str(v) == "1.2.3-M5"

    def test_version_no_milestone(self):
        v = AppVersion(major=1, minor=0, patch=0, milestone=None)
        assert str(v) == "1.0.0"


class TestUnitConfig:
    """Test unit configuration."""

    def test_default_units(self):
        uc = UnitConfig()
        assert uc.length_unit == LengthUnit.METERS
        assert uc.angle_unit == AngleUnit.DEGREES

    def test_length_to_meters(self):
        uc_m = UnitConfig(length_unit=LengthUnit.METERS)
        assert uc_m.length_to_meters == 1.0

        uc_mm = UnitConfig(length_unit=LengthUnit.MILLIMETERS)
        assert uc_mm.length_to_meters == 0.001

        uc_ft = UnitConfig(length_unit=LengthUnit.FEET)
        assert abs(uc_ft.length_to_meters - 0.3048) < 0.0001


class TestRandomConfig:
    """Test random seed configuration."""

    def test_default_seed(self):
        rc = RandomConfig()
        assert rc.master_seed == 42

    def test_get_seed_fallback(self):
        rc = RandomConfig(master_seed=99)
        assert rc.get_seed("dfn") == 99
        assert rc.get_seed("voxel") == 99
        assert rc.get_seed("block") == 99

    def test_get_seed_override(self):
        rc = RandomConfig(master_seed=42, dfn_seed=100, voxel_seed=200)
        assert rc.get_seed("dfn") == 100
        assert rc.get_seed("voxel") == 200
        assert rc.get_seed("block") == 42  # Falls back to master


class TestAppConfig:
    """Test root application configuration."""

    def test_default_creation(self):
        config = AppConfig()
        assert config.version.major == 0
        assert config.units.length_unit == LengthUnit.METERS
        assert config.random.master_seed == 42

    def test_global_singleton(self):
        reset_config()
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_set_config(self):
        c1 = AppConfig(random=RandomConfig(master_seed=999))
        set_config(c1)
        assert get_config().random.master_seed == 999
        reset_config()


class TestEnumValues:
    """Test enumeration values."""

    def test_coordinate_systems(self):
        assert CoordinateSystem.LOCAL_CARTESIAN is not None
        assert CoordinateSystem.UTM is not None

    def test_angle_units(self):
        assert AngleUnit.DEGREES.value == "degrees"
        assert AngleUnit.RADIANS.value == "radians"

    def test_length_units(self):
        assert LengthUnit.METERS.value == "m"
        assert LengthUnit.MILLIMETERS.value == "mm"
