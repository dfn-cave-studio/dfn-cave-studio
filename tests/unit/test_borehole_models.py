"""Tests for borehole data models (M1)."""

import math
import numpy as np
import pytest

from dfn_cave_studio.models.borehole import (
    Collar, SurveyStation, BoreholeSurvey, FractureObservation,
    RQDInterval, Borehole, BoreholeCollection, BoreholeValidationError,
    TrajectoryMethod,
)
from dfn_cave_studio.models.enums import FractureType


class TestCollar:
    def test_default_creation(self):
        c = Collar(borehole_id="BH-001", collar_x=10, collar_y=20, collar_z=100,
                    azimuth=45, dip=-60, final_depth=200)
        assert c.borehole_id == "BH-001"
        assert c.azimuth == 45
        np.testing.assert_array_equal(c.position, [10, 20, 100])

    def test_angle_validation(self):
        with pytest.raises(Exception):
            Collar(borehole_id="X", collar_x=0, collar_y=0, collar_z=0, azimuth=400)

    def test_dip_negative(self):
        """Dip should accept negative values (downward)."""
        c = Collar(borehole_id="X", collar_x=0, collar_y=0, collar_z=0, dip=-90)
        assert c.dip == -90


class TestBoreholeSurvey:
    def test_straight_trajectory(self):
        collar = Collar(borehole_id="BH-1", collar_x=0, collar_y=0, collar_z=100,
                         azimuth=0, dip=-90, final_depth=50)
        survey = BoreholeSurvey(stations=[])
        points, mds = survey.compute_trajectory_3d(collar, step_length=5)
        assert points.shape[0] > 2
        # Straight down: x,y constant, z decreasing
        np.testing.assert_allclose(points[:, 0], 0, atol=1e-10)
        np.testing.assert_allclose(points[:, 1], 0, atol=1e-10)
        assert points[-1, 2] < points[0, 2]

    def test_min_curvature_trajectory(self):
        collar = Collar(borehole_id="BH-2", collar_x=0, collar_y=0, collar_z=100,
                         azimuth=0, dip=-45, final_depth=100)
        stations = [
            SurveyStation(measured_depth=0, azimuth=0, dip=-45),
            SurveyStation(measured_depth=50, azimuth=90, dip=-60),
            SurveyStation(measured_depth=100, azimuth=180, dip=-30),
        ]
        survey = BoreholeSurvey(
            stations=stations,
            trajectory_method=TrajectoryMethod.MINIMUM_CURVATURE,
        )
        points, mds = survey.compute_trajectory_3d(collar, step_length=2)
        assert points.shape[0] > 10
        assert len(points) == len(mds)


class TestFractureObservation:
    def test_valid_observation(self):
        obs = FractureObservation(
            borehole_id="BH-1", measured_depth=50,
            dip_direction=45, dip=30, fracture_type=FractureType.JOINT,
        )
        assert obs.measured_depth == 50
        assert obs.confidence == 1.0

    def test_invalid_dip(self):
        with pytest.raises(Exception):
            FractureObservation(borehole_id="X", measured_depth=10, dip_direction=0, dip=100)


class TestRQDInterval:
    def test_valid_interval(self):
        rqd = RQDInterval(borehole_id="BH-1", from_depth=10, to_depth=20, rqd_value=85)
        assert rqd.interval_length == 10
        assert rqd.rqd_value == 85

    def test_invalid_range(self):
        with pytest.raises(Exception):
            RQDInterval(borehole_id="X", from_depth=20, to_depth=10, rqd_value=50)


class TestBorehole:
    def test_create_borehole(self):
        collar = Collar(borehole_id="BH-001", collar_x=0, collar_y=0, collar_z=100,
                         azimuth=45, dip=-60, final_depth=150)
        bh = Borehole(borehole_id="BH-001", collar=collar)
        assert bh.borehole_id == "BH-001"
        assert bh.observed_fracture_count == 0

    def test_with_observations(self):
        collar = Collar(borehole_id="BH-002", collar_x=0, collar_y=0, collar_z=100,
                         azimuth=0, dip=-90, final_depth=100)
        obs = [
            FractureObservation(borehole_id="BH-002", measured_depth=20,
                                dip_direction=45, dip=60),
            FractureObservation(borehole_id="BH-002", measured_depth=50,
                                dip_direction=135, dip=30),
        ]
        bh = Borehole(borehole_id="BH-002", collar=collar, fracture_observations=obs)
        assert bh.observed_fracture_count == 2


class TestBoreholeCollection:
    def test_add_borehole(self):
        coll = BoreholeCollection()
        bh = Borehole(
            borehole_id="BH-001",
            collar=Collar(borehole_id="BH-001", collar_x=0, collar_y=0, collar_z=100,
                          azimuth=0, dip=-90, final_depth=100),
        )
        coll.add(bh)
        assert len(coll) == 1

    def test_duplicate_rejected(self):
        coll = BoreholeCollection()
        bh1 = Borehole(
            borehole_id="BH-001",
            collar=Collar(borehole_id="BH-001", collar_x=0, collar_y=0, collar_z=100,
                          azimuth=0, dip=-90, final_depth=100),
        )
        bh2 = Borehole(
            borehole_id="BH-001",
            collar=Collar(borehole_id="BH-001", collar_x=1, collar_y=1, collar_z=101,
                          azimuth=45, dip=-60, final_depth=80),
        )
        coll.add(bh1)
        with pytest.raises(ValueError, match="Duplicate"):
            coll.add(bh2)

    def test_validate_all_empty(self):
        coll = BoreholeCollection()
        errors = coll.validate_all()
        assert len(errors) == 0

    def test_duplicate_detection(self):
        coll = BoreholeCollection()
        bh = Borehole(
            borehole_id="BH-001",
            collar=Collar(borehole_id="BH-001", collar_x=0, collar_y=0, collar_z=100,
                          azimuth=0, dip=-90, final_depth=100),
        )
        coll.add(bh)
        # Manually add duplicate (bypassing add() check)
        coll.boreholes.append(bh)
        errors = coll.validate_all()
        dup_errors = [e for e in errors if "Duplicate" in e.issue]
        assert len(dup_errors) >= 1

    def test_quality_report(self):
        coll = BoreholeCollection()
        bh = Borehole(
            borehole_id="BH-001",
            collar=Collar(borehole_id="BH-001", collar_x=0, collar_y=0, collar_z=100,
                          azimuth=0, dip=-90, final_depth=100),
        )
        coll.add(bh)
        report = coll.quality_report()
        assert "All checks passed" in report
