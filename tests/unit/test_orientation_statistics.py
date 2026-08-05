"""Tests for OrientationStatisticsCalculator — Fisher stats from borehole observations.

Verifies:
  - dip=0,30,60,90 round-trip through coordinate transforms
  - Multi-set Fisher computation
  - <3 observations raises ValueError
  - No-set_id observations skipped
  - locate_all_observations_3d
  - Tightly clustered vs dispersed orientations
  - Kappa consistency
"""

import math
import pytest
import numpy as np

from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal, normal_to_dip_dir_dip
from dfn_cave_studio.models.borehole import (
    BoreholeCollection, Borehole, Collar, BoreholeSurvey,
    SurveyStation, FractureObservation,
)
from dfn_cave_studio.models.enums import FractureType
from dfn_cave_studio.borehole.orientation_statistics import OrientationStatisticsCalculator


class TestCoordinateRoundTrip:
    """Dip inversion acceptance tests — 0→0, 30→30, 60→60, 90→90."""

    @pytest.mark.parametrize("dip", [0, 30, 60, 90])
    @pytest.mark.parametrize("dd", [0, 45, 90, 180, 270])
    def test_dip_round_trip(self, dd, dip):
        n = dip_dir_dip_to_normal(dd, dip)
        out_dd, out_dip = normal_to_dip_dir_dip(n)
        assert abs(out_dip - dip) < 0.1, f"dip={dip}→{out_dip:.1f} (dd={dd})"


class TestOrientationStatistics:
    """Comprehensive Fisher statistics tests."""

    @pytest.fixture
    def calculator(self):
        return OrientationStatisticsCalculator()

    @pytest.fixture
    def empty_collection(self):
        return BoreholeCollection(boreholes=[])

    def _make_obs(self, dd: float, dip: float, set_id: int = 1) -> FractureObservation:
        return FractureObservation(
            borehole_id="BH-001",
            measured_depth=10.0,
            dip_direction=dd,
            dip=dip,
            set_id=set_id,
        )

    def _make_bh_with_obs(self, bh_id: str, observations: list) -> Borehole:
        """Create a borehole with given observations and a simple trajectory."""
        bh = Borehole(
            borehole_id=bh_id,
            collar=Collar(
                borehole_id=bh_id,
                collar_x=0, collar_y=0, collar_z=100,
                final_depth=200,
            ),
            survey=BoreholeSurvey(stations=[
                SurveyStation(measured_depth=0, azimuth=0, dip=-90),
                SurveyStation(measured_depth=200, azimuth=0, dip=-90),
            ]),
        )
        bh.fracture_observations = observations
        return bh

    # ── Error on < 3 observations ──────────────────────────────────────

    def test_less_than_3_observations_raises(self, calculator):
        obs = [self._make_obs(45, 60, 1), self._make_obs(47, 62, 1)]
        with pytest.raises(ValueError, match="at least 3"):
            calculator.compute_fisher_stats(obs)

    # ── compute_by_set with multiple set_ids ───────────────────────────

    def test_multi_set_fisher_stats(self, calculator):
        """Three sets with Fisher-distributed orientations."""
        rng = np.random.default_rng(42)
        bh = self._make_bh_with_obs("BH-001", [
            self._make_obs(45 + rng.normal(0, 2), 60 + rng.normal(0, 2), 1)
            for _ in range(15)
        ] + [
            self._make_obs(135 + rng.normal(0, 2), 75 + rng.normal(0, 2), 2)
            for _ in range(12)
        ] + [
            self._make_obs(270 + rng.normal(0, 3), 30 + rng.normal(0, 3), 3)
            for _ in range(10)
        ])
        coll = BoreholeCollection(boreholes=[bh])
        stats = calculator.compute_by_set(coll)

        assert len(stats) == 3
        assert 1 in stats and 2 in stats and 3 in stats
        # Set 1: mean near (45, 60)
        assert 40 < stats[1].mean_dip_direction < 50
        assert 55 < stats[1].mean_dip < 65
        assert stats[1].kappa > 5  # tightly clustered
        # Set 2: mean near (135, 75)
        assert 130 < stats[2].mean_dip_direction < 140
        assert 70 < stats[2].mean_dip < 80
        # Set 3: mean near (270, 30)
        assert 265 < stats[3].mean_dip_direction < 275
        assert 25 < stats[3].mean_dip < 35

    # ── No set_id observations are skipped ────────────────────────────

    def test_no_set_id_skipped(self, calculator):
        obs_with = [self._make_obs(45, 60, 1) for _ in range(10)]
        obs_without = [self._make_obs(90, 45, None) for _ in range(5)]
        bh = self._make_bh_with_obs("BH-001", obs_with + obs_without)
        coll = BoreholeCollection(boreholes=[bh])
        stats = calculator.compute_by_set(coll)
        assert len(stats) == 1  # Only set_id=1
        assert 1 in stats

    def test_all_no_set_id_returns_empty(self, calculator):
        obs = [self._make_obs(45, 60, None) for _ in range(10)]
        bh = self._make_bh_with_obs("BH-001", obs)
        coll = BoreholeCollection(boreholes=[bh])
        stats = calculator.compute_by_set(coll)
        assert len(stats) == 0

    # ── Tightly clustered vs dispersed ────────────────────────────────

    def test_tightly_clustered_gives_high_kappa(self, calculator):
        """All observations at exactly the same orientation → high kappa."""
        obs = [self._make_obs(45.0, 60.0, 1) for _ in range(10)]
        result = calculator.compute_fisher_stats(obs)
        assert result.kappa > 100  # Very high concentration
        assert abs(result.mean_dip_direction - 45.0) < 1
        assert abs(result.mean_dip - 60.0) < 1

    def test_dispersed_gives_low_kappa(self, calculator):
        """Widely dispersed orientations → low kappa."""
        rng = np.random.default_rng(42)
        obs = [self._make_obs(rng.uniform(0, 360), rng.uniform(0, 90), 1)
               for _ in range(20)]
        result = calculator.compute_fisher_stats(obs)
        assert result.kappa < 15  # Low concentration
        assert 0 <= result.mean_dip_direction < 360
        assert 0 <= result.mean_dip <= 90

    # ── Dip=0, 30, 60, 90 ────────────────────────────────────────────

    @pytest.mark.parametrize("dip", [0, 30, 60, 90])
    def test_dip_recovery(self, calculator, dip):
        """Fisher stats recovers the correct dip for uniform orientations."""
        obs = [self._make_obs(45.0, float(dip), 1) for _ in range(10)]
        result = calculator.compute_fisher_stats(obs)
        assert abs(result.mean_dip - float(dip)) < 0.5, (
            f"dip={dip} recovered as {result.mean_dip}")

    @pytest.mark.parametrize("dd", [0, 45, 90, 180, 270])
    def test_dd_recovery(self, calculator, dd):
        """Fisher stats recovers the correct dip direction."""
        obs = [self._make_obs(float(dd), 60.0, 1) for _ in range(10)]
        result = calculator.compute_fisher_stats(obs)
        assert abs(result.mean_dip_direction - float(dd)) < 0.5, (
            f"dd={dd} recovered as {result.mean_dip_direction}")

    # ── Kappa formula — small vs large sample ──────────────────────────

    def test_kappa_small_sample_correction(self, calculator):
        """n=3 uses (n-2)/(n-R) * n/(n-1) correction."""
        obs = [self._make_obs(45.0, 60.0, 1),
               self._make_obs(46.0, 61.0, 1),
               self._make_obs(44.0, 59.0, 1)]
        result = calculator.compute_fisher_stats(obs)
        assert result.kappa > 0
        assert result.kappa <= 999.0

    def test_kappa_large_sample(self, calculator):
        """n=20 uses (n-1)/(n-R) formula."""
        obs = [self._make_obs(45.0, 60.0, 1) for _ in range(20)]
        result = calculator.compute_fisher_stats(obs)
        assert result.kappa > 100  # Exact same → very high

    # ── Resultant near zero (uniform) ──────────────────────────────────

    def test_uniform_distribution(self, calculator):
        """Opposing directions nearly cancel → uses first observation."""
        rng = np.random.default_rng(123)
        obs = [self._make_obs(float(dd), 45.0, 1)
               for dd in np.linspace(0, 350, 20)]
        result = calculator.compute_fisher_stats(obs)
        assert 0 <= result.mean_dip_direction < 360
        assert result.kappa >= 0.1

    # ── MIN_OBSERVATIONS constraint ─────────────────────────────────────

    def test_min_observations_is_3(self):
        assert OrientationStatisticsCalculator.MIN_OBSERVATIONS == 3

    # ── locate_all_observations_3d ─────────────────────────────────────
    def test_locate_all_observations_3d(self, calculator):
        """3D positions are returned for all located observations."""
        bh = Borehole(
            borehole_id="BH-001",
            collar=Collar(
                borehole_id="BH-001",
                collar_x=25, collar_y=25, collar_z=200,
                final_depth=200,
            ),
            survey=BoreholeSurvey(stations=[
                SurveyStation(measured_depth=0, azimuth=0, dip=-90),
                SurveyStation(measured_depth=200, azimuth=0, dip=-90),
            ]),
            fracture_observations=[
                FractureObservation(measured_depth=50, dip_direction=45, dip=60),
                FractureObservation(measured_depth=100, dip_direction=135, dip=75),
                FractureObservation(measured_depth=150, dip_direction=270, dip=30),
            ],
        )
        coll = BoreholeCollection(boreholes=[bh])
        positions = calculator.locate_all_observations_3d(coll)
        assert "BH-001" in positions
        assert len(positions["BH-001"]) == 3
        for pos in positions["BH-001"]:
            assert pos.shape == (3,)
            assert all(isinstance(v, float) for v in pos)

    def test_locate_observations_returns_empty_for_no_obs(self, calculator):
        """Borehole with no fracture observations returns empty positions."""
        bh = Borehole(
            borehole_id="BH-001",
            collar=Collar(
                borehole_id="BH-001",
                collar_x=0, collar_y=0, collar_z=100,
                final_depth=100,
            ),
            survey=BoreholeSurvey(stations=[
                SurveyStation(measured_depth=0, azimuth=0, dip=-90),
                SurveyStation(measured_depth=100, azimuth=0, dip=-90),
            ]),
            fracture_observations=[],  # No observations
        )
        coll = BoreholeCollection(boreholes=[bh])
        positions = calculator.locate_all_observations_3d(coll)
        assert "BH-001" not in positions  # No valid positions
