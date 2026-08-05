"""Scientific validation of minimum curvature trajectory algorithm.

Reference: Taylor, H.L. & Mason, C.M. (1972).
           A systematic approach to well surveying calculations.
"""

import math
import numpy as np

from dfn_cave_studio.models.borehole import (
    Collar, Borehole, BoreholeSurvey, SurveyStation, TrajectoryMethod,
)


class TestMinimumCurvatureScientific:
    """Validate the minimum curvature method against analytic solutions."""

    @staticmethod
    def _spatial_length(points):
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    def test_straight_vertical_hole(self):
        """10 m straight vertical hole → spatial length = 10 m."""
        collar = Collar(borehole_id="V", collar_x=0, collar_y=0, collar_z=0,
                        azimuth=0, dip=-90, final_depth=10)
        bh = Borehole(borehole_id="V", collar=collar)
        points, mds = bh.compute_trajectory(step_length=1.0)

        length = self._spatial_length(points)
        assert abs(length - 10.0) < 0.02, f"Expected 10m, got {length}m"

    def test_straight_horizontal_hole(self):
        """10 m straight horizontal hole → spatial length = 10 m."""
        collar = Collar(borehole_id="H", collar_x=0, collar_y=0, collar_z=100,
                        azimuth=90, dip=0, final_depth=10)
        bh = Borehole(borehole_id="H", collar=collar)
        points, mds = bh.compute_trajectory(step_length=1.0)

        length = self._spatial_length(points)
        assert abs(length - 10.0) < 0.02, f"Expected 10m, got {length}m"

    def test_45_degree_inclined_straight_hole(self):
        """10 m hole at 45° dip → spatial length = 10 m."""
        collar = Collar(borehole_id="I", collar_x=0, collar_y=0, collar_z=100,
                        azimuth=0, dip=-45, final_depth=10)
        bh = Borehole(borehole_id="I", collar=collar)
        points, mds = bh.compute_trajectory(step_length=1.0)

        length = self._spatial_length(points)
        assert abs(length - 10.0) < 0.02, f"Expected 10m, got {length}m"

    def test_survey_straight_hole_matches_no_survey(self):
        """With survey stations all at same direction, result = no-survey result."""
        collar = Collar(borehole_id="S", collar_x=0, collar_y=0, collar_z=0,
                        azimuth=45, dip=-30, final_depth=20)

        # No survey
        bh1 = Borehole(borehole_id="S", collar=collar)
        p1, _ = bh1.compute_trajectory(step_length=1.0)

        # With survey (all stations at same orientation)
        survey = BoreholeSurvey(
            stations=[
                SurveyStation(measured_depth=0, azimuth=45, dip=-30),
                SurveyStation(measured_depth=10, azimuth=45, dip=-30),
                SurveyStation(measured_depth=20, azimuth=45, dip=-30),
            ],
            trajectory_method=TrajectoryMethod.MINIMUM_CURVATURE,
        )
        bh2 = Borehole(borehole_id="S", collar=collar, survey=survey)
        p2, _ = bh2.compute_trajectory(step_length=1.0)

        # End points should match
        np.testing.assert_allclose(p1[-1], p2[-1], rtol=1e-4)

    def test_quarter_circle_approximation(self):
        """90° bend over 10m arc → spatial length ≈ arc length.

        For a 90° circular arc with ΔMD = 10 m:
          - Arc length = 10 m (the measured depth is the arc length)
          - Chord length = (20√2)/π ≈ 9.003 m (straight line from start to end)
          - Sum of sub-chord lengths (polygonal approximation) ≈ arc length

        With step_length = 0.5 m (20 sub-steps), the polygon length should
        be close to the arc length of 10 m, not the chord of 9.003 m.
        """
        collar = Collar(borehole_id="Q", collar_x=0, collar_y=0, collar_z=0,
                        azimuth=0, dip=0, final_depth=10)
        survey = BoreholeSurvey(
            stations=[
                SurveyStation(measured_depth=0, azimuth=0, dip=0),    # horizontal N
                SurveyStation(measured_depth=10, azimuth=0, dip=-90),  # vertical down
            ],
            trajectory_method=TrajectoryMethod.MINIMUM_CURVATURE,
        )
        bh = Borehole(borehole_id="Q", collar=collar, survey=survey)
        points, mds = bh.compute_trajectory(step_length=0.5)

        # The direct displacement from first to last point = chord
        chord = float(np.linalg.norm(points[-1] - points[0]))
        expected_chord = (20 * math.sqrt(2)) / math.pi  # ≈ 9.003 m
        assert abs(chord - expected_chord) < 0.05, \
            f"Chord should be ~{expected_chord:.3f}m, got {chord:.3f}m"

        # The polygonal path length (sum of sub-chords) ≈ arc length
        path_length = self._spatial_length(points)
        assert abs(path_length - 10.0) < 0.1, \
            f"Path length should be ~10m (arc), got {path_length:.3f}m"

    def test_rf_factor_greater_than_one(self):
        """For a curved hole, RF = tan(γ/2)/(γ/2) > 1.

        Without RF correction, spatial length is systematically underestimated.
        """
        # 60° dogleg
        gamma = math.radians(60)
        rf = math.tan(gamma / 2) / (gamma / 2)
        assert rf > 1.0, f"RF = {rf} should be > 1 for curved well"

        # For small dogleg, RF ≈ 1
        gamma_small = math.radians(1.0)
        rf_small = math.tan(gamma_small / 2) / (gamma_small / 2)
        assert abs(rf_small - 1.0) < 0.001

    def test_tangential_vs_minimum_curvature(self):
        """Minimum curvature and tangential produce different trajectories.

        Tangential follows a straight line in the start direction.
        Minimum curvature follows a curved arc, resulting in different
        endpoint positions (though similar total spatial lengths).
        """
        collar = Collar(borehole_id="T", collar_x=0, collar_y=0, collar_z=0,
                        azimuth=0, dip=-30, final_depth=10)

        stations = [
            SurveyStation(measured_depth=0, azimuth=0, dip=-30),
            SurveyStation(measured_depth=10, azimuth=90, dip=-30),
        ]

        # Minimum curvature
        survey_mc = BoreholeSurvey(
            stations=stations,
            trajectory_method=TrajectoryMethod.MINIMUM_CURVATURE,
        )
        bh_mc = Borehole(borehole_id="T", collar=collar, survey=survey_mc)
        p_mc, _ = bh_mc.compute_trajectory(step_length=0.5)

        # Tangential
        survey_tg = BoreholeSurvey(
            stations=stations,
            trajectory_method=TrajectoryMethod.TANGENTIAL,
        )
        bh_tg = Borehole(borehole_id="T", collar=collar, survey=survey_tg)
        p_tg, _ = bh_tg.compute_trajectory(step_length=0.5)

        # Endpoints should differ (MC curves, TG goes straight)
        endpoint_diff = float(np.linalg.norm(p_mc[-1] - p_tg[-1]))
        assert endpoint_diff > 0.1, \
            f"MC and TG endpoints should differ for curved hole: diff={endpoint_diff:.3f}m"
