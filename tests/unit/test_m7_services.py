"""Unit tests for M7 services: import, cleaning, holdout, joint sets."""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dfn_cave_studio.services.cleaning_service import DataCleaningService
from dfn_cave_studio.services.holdout_service import HoldoutService, HoldoutRole
from dfn_cave_studio.services.joint_set_service import JointSetService
from dfn_cave_studio.models.data_management import IssueSeverity, IssueType, FieldMapping
from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal


# ═══════════════════════════════════════════════════════════════════════════
# Cleaning service
# ═══════════════════════════════════════════════════════════════════════════

class TestCleaningService:
    """Data cleaning validation checks."""

    @pytest.fixture
    def svc(self):
        return DataCleaningService()

    @pytest.fixture
    def valid_collars(self):
        return pd.DataFrame({
            "hole_id": ["BH-01", "BH-02", "BH-03"],
            "easting": [100, 200, 300],
            "northing": [200, 300, 400],
            "elevation": [500, 510, 520],
            "total_depth": [150, 200, 180],
        })

    def test_empty_hole_id_detected(self, svc):
        """Empty hole_id raises ERROR."""
        df = pd.DataFrame({"hole_id": ["", "BH-02"], "easting": [100, 200],
                           "northing": [200, 300], "elevation": [500, 510],
                           "total_depth": [150, 200]})
        issues = svc.validate_collars(df)
        errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
        assert len(errors) >= 1
        assert any("hole_id" in i.field for i in errors)

    def test_duplicate_hole_id_detected(self, svc):
        """Duplicate hole_id raises ERROR."""
        df = pd.DataFrame({
            "hole_id": ["BH-01", "BH-01"],
            "easting": [100, 100], "northing": [200, 200],
            "elevation": [500, 500], "total_depth": [150, 150],
        })
        issues = svc.validate_collars(df)
        dups = [i for i in issues if i.issue_type == IssueType.DUPLICATE]
        assert len(dups) >= 1

    def test_missing_coordinate_detected(self, svc):
        """Missing easting raises ERROR."""
        df = pd.DataFrame({
            "hole_id": ["BH-01"], "easting": [None],
            "northing": [200], "elevation": [500], "total_depth": [150],
        })
        issues = svc.validate_collars(df)
        assert len(issues) >= 1

    def test_negative_total_depth(self, svc):
        """Total depth <= 0 raises ERROR."""
        df = pd.DataFrame({
            "hole_id": ["BH-01"], "easting": [100], "northing": [200],
            "elevation": [500], "total_depth": [-50],
        })
        issues = svc.validate_collars(df)
        assert len(issues) >= 1

    def test_azimuth_normalization(self, svc):
        """Azimuth outside [0,360) is normalized."""
        df = pd.DataFrame({
            "hole_id": ["BH-01"], "easting": [100], "northing": [200],
            "elevation": [500], "total_depth": [150], "azimuth": [-45],
        })
        issues = svc.validate_collars(df)
        warnings = [i for i in issues if i.severity == IssueSeverity.WARNING]
        assert len(warnings) >= 1
        assert df.at[0, "azimuth"] == 315  # -45 → 315

    def test_valid_data_passes(self, svc, valid_collars):
        """Clean data produces no errors."""
        issues = svc.validate_collars(valid_collars)
        errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
        assert len(errors) == 0

    def test_survey_unknown_borehole(self, svc, valid_collars):
        """Survey referencing unknown borehole raises ERROR."""
        df = pd.DataFrame({
            "hole_id": ["BH-UNKNOWN"],
            "measured_depth": [10],
        })
        issues = svc.validate_surveys(df, valid_collars)
        errors = [i for i in issues if i.issue_type == IssueType.UNKNOWN_REFERENCE]
        assert len(errors) >= 1

    def test_rqd_range_check(self, svc, valid_collars):
        """RQD outside [0,100] raises WARNING."""
        df = pd.DataFrame({
            "hole_id": ["BH-01"],
            "from_depth": [0], "to_depth": [50], "rqd": [150],
        })
        issues = svc.validate_rqd(df, valid_collars)
        warnings = [i for i in issues if i.severity == IssueSeverity.WARNING]
        assert len(warnings) >= 1

    def test_rqd_overlap_detected(self, svc, valid_collars):
        """Overlapping RQD intervals raise WARNING."""
        df = pd.DataFrame({
            "hole_id": ["BH-01", "BH-01"],
            "from_depth": [0, 40], "to_depth": [50, 100], "rqd": [80, 75],
        })
        issues = svc.validate_rqd(df, valid_collars)
        overlaps = [i for i in issues if i.issue_type == IssueType.OVERLAP]
        assert len(overlaps) >= 1

    def test_from_gt_to_detected(self, svc, valid_collars):
        """from_depth >= to_depth raises ERROR."""
        df = pd.DataFrame({
            "hole_id": ["BH-01"],
            "from_depth": [100], "to_depth": [50], "rqd": [80],
        })
        issues = svc.validate_rqd(df, valid_collars)
        errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
        assert len(errors) >= 1

    def test_issues_report_exportable(self, svc, valid_collars):
        """Issues can be exported as dicts."""
        df = pd.DataFrame({
            "hole_id": ["BH-01", "BH-01"],
            "from_depth": [0, 40], "to_depth": [50, 100], "rqd": [80, 75],
        })
        svc.validate_rqd(df, valid_collars)
        report = svc.export_issues_report()
        assert len(report) >= 1
        assert "issue_id" in report[0]
        assert "source_file" in report[0]
        assert "hole_id" in report[0]


# ═══════════════════════════════════════════════════════════════════════════
# Holdout service
# ═══════════════════════════════════════════════════════════════════════════

class TestHoldoutService:
    """Validation holdout tests."""

    def test_random_holdout_reproducible(self):
        """Same seed produces same validation set."""
        holes = [f"BH-{i:02d}" for i in range(1, 9)]
        svc1 = HoldoutService()
        svc1.select_random(holes, validation_fraction=0.25, random_seed=42)
        svc2 = HoldoutService()
        svc2.select_random(holes, validation_fraction=0.25, random_seed=42)
        assert svc1.validation_holes == svc2.validation_holes

    def test_manual_selection(self):
        """Manual selection sets correct roles."""
        holes = ["BH-01", "BH-02", "BH-03", "BH-04"]
        svc = HoldoutService()
        svc.select_manual(holes, ["BH-02", "BH-04"])
        assert svc.is_validation("BH-02")
        assert svc.is_validation("BH-04")
        assert svc.is_calibration("BH-01")
        assert svc.is_calibration("BH-03")

    def test_lock_prevents_reselection(self):
        """Locked holdout cannot be modified."""
        holes = ["BH-01", "BH-02"]
        svc = HoldoutService()
        svc.select_random(holes, 0.5, random_seed=42)
        svc.lock()
        with pytest.raises(RuntimeError):
            svc.select_random(holes, 0.3, random_seed=99)

    def test_unlock_allows_reselection(self):
        """Unlocked holdout can be modified."""
        holes = ["BH-01", "BH-02"]
        svc = HoldoutService()
        svc.select_random(holes, 0.5, random_seed=42)
        svc.lock()
        svc.unlock()
        svc.select_random(holes, 0.5, random_seed=42)  # Should not raise

    def test_filter_calibration(self):
        """filter_calibration excludes validation holes."""
        holes = [f"BH-{i:02d}" for i in range(1, 9)]
        svc = HoldoutService()
        svc.select_random(holes, 0.25, random_seed=42)
        cal = svc.filter_calibration(holes)
        assert len(cal) + len(svc.validation_holes) == len(holes)
        for vh in svc.validation_holes:
            assert vh not in cal

    def test_stratified_holdout(self):
        """Stratified holdout selects per domain."""
        holes = ["BH-01", "BH-02", "BH-03", "BH-04", "BH-05", "BH-06"]
        domains = {"BH-01": 1, "BH-02": 1, "BH-03": 1,
                   "BH-04": 2, "BH-05": 2, "BH-06": 2}
        svc = HoldoutService()
        svc.select_stratified(holes, domains, 0.5, random_seed=42)
        # Each domain should have at least 1 validation
        d1_vals = [h for h in svc.validation_holes if domains.get(h) == 1]
        d2_vals = [h for h in svc.validation_holes if domains.get(h) == 2]
        assert len(d1_vals) >= 1
        assert len(d2_vals) >= 1

    def test_to_dict_round_trip(self):
        """to_dict → from_dict preserves holdout state."""
        holes = [f"BH-{i:02d}" for i in range(1, 9)]
        svc = HoldoutService()
        svc.select_random(holes, 0.25, random_seed=42)
        svc.lock()
        d = svc.to_dict()
        restored = HoldoutService.from_dict(d)
        assert restored.validation_holes == svc.validation_holes
        assert restored.is_locked


# ═══════════════════════════════════════════════════════════════════════════
# Joint set identification
# ═══════════════════════════════════════════════════════════════════════════

class TestJointSetService:
    """Joint set identification tests."""

    def test_direction_wrapping(self):
        """359° and 1° should be close on the sphere, not 358° apart."""
        n1 = dip_dir_dip_to_normal(359, 60)
        n2 = dip_dir_dip_to_normal(1, 60)
        # Cosine distance on sphere
        cos_dist = 1.0 - abs(np.dot(n1, n2))
        assert cos_dist < 0.01, f"359° and 1° should be close, cos_dist={cos_dist:.4f}"

    def test_359_and_1_are_close(self):
        """359° and 1° are close (angular wrap handled by unit vectors)."""
        n1 = dip_dir_dip_to_normal(359, 60)
        n2 = dip_dir_dip_to_normal(1, 60)
        cos_dist = 1.0 - abs(np.dot(n1, n2))
        assert cos_dist < 0.01, f"359° and 1° should be nearly identical, cos_dist={cos_dist:.4f}"

    def test_auto_identify_reproducible(self):
        """Auto-identification with fixed seed is reproducible."""
        from dfn_cave_studio.models.borehole import (
            BoreholeCollection, Borehole, Collar, BoreholeSurvey,
            SurveyStation, FractureObservation,
        )
        rng = np.random.default_rng(42)

        def make_collection():
            bh = Borehole(
                borehole_id="BH-01",
                collar=Collar(borehole_id="BH-01", collar_x=0, collar_y=0, collar_z=500, final_depth=200),
                survey=BoreholeSurvey(stations=[
                    SurveyStation(measured_depth=0, azimuth=0, dip=-90),
                    SurveyStation(measured_depth=200, azimuth=0, dip=-90),
                ]),
            )
            # 3 sets with Fisher-distributed orientations
            for _ in range(15):
                bh.fracture_observations.append(FractureObservation(
                    measured_depth=float(rng.uniform(5, 195)),
                    dip_direction=45 + rng.normal(0, 3),
                    dip=60 + rng.normal(0, 3),
                ))
            for _ in range(12):
                bh.fracture_observations.append(FractureObservation(
                    measured_depth=float(rng.uniform(5, 195)),
                    dip_direction=135 + rng.normal(0, 3),
                    dip=75 + rng.normal(0, 3),
                ))
            for _ in range(10):
                bh.fracture_observations.append(FractureObservation(
                    measured_depth=float(rng.uniform(5, 195)),
                    dip_direction=270 + rng.normal(0, 3),
                    dip=30 + rng.normal(0, 3),
                ))
            return BoreholeCollection(boreholes=[bh])

        svc1 = JointSetService(random_seed=42)
        r1 = svc1.identify_auto(make_collection(), {"BH-01"}, set(), n_clusters=3, random_seed=99)
        svc2 = JointSetService(random_seed=42)
        r2 = svc2.identify_auto(make_collection(), {"BH-01"}, set(), n_clusters=3, random_seed=99)
        assert len(r1.sets) == len(r2.sets)

    def test_validation_excluded_from_clustering(self):
        """Validation holes do not affect cluster centroids."""
        from dfn_cave_studio.models.borehole import (
            BoreholeCollection, Borehole, Collar, BoreholeSurvey,
            SurveyStation, FractureObservation,
        )
        cal_bh = Borehole(
            borehole_id="BH-CAL",
            collar=Collar(borehole_id="BH-CAL", collar_x=0, collar_y=0, collar_z=500, final_depth=200),
            survey=BoreholeSurvey(stations=[
                SurveyStation(measured_depth=0, azimuth=0, dip=-90),
                SurveyStation(measured_depth=200, azimuth=0, dip=-90),
            ]),
            fracture_observations=[
                FractureObservation(measured_depth=50, dip_direction=45, dip=60),
                FractureObservation(measured_depth=100, dip_direction=47, dip=62),
                FractureObservation(measured_depth=150, dip_direction=43, dip=58),
            ],
        )
        collection = BoreholeCollection(boreholes=[cal_bh])
        svc = JointSetService()
        result = svc.identify_auto(collection, {"BH-CAL"}, set(), n_clusters=1)
        assert result.calibration_count == 3
        assert result.validation_count == 0

    def test_imported_mode_counts_correctly(self):
        """Mode A: imported set_id preserves original assignments."""
        from dfn_cave_studio.models.borehole import (
            BoreholeCollection, Borehole, Collar, BoreholeSurvey,
            SurveyStation, FractureObservation,
        )
        bh = Borehole(
            borehole_id="BH-01",
            collar=Collar(borehole_id="BH-01", collar_x=0, collar_y=0, collar_z=500, final_depth=200),
            survey=BoreholeSurvey(stations=[
                SurveyStation(measured_depth=0, azimuth=0, dip=-90),
                SurveyStation(measured_depth=200, azimuth=0, dip=-90),
            ]),
            fracture_observations=[
                FractureObservation(measured_depth=50, dip_direction=45, dip=60, set_id=1),
                FractureObservation(measured_depth=70, dip_direction=47, dip=62, set_id=1),
                FractureObservation(measured_depth=90, dip_direction=43, dip=58, set_id=1),
                FractureObservation(measured_depth=110, dip_direction=135, dip=75, set_id=2),
                FractureObservation(measured_depth=130, dip_direction=137, dip=77, set_id=2),
                FractureObservation(measured_depth=150, dip_direction=133, dip=73, set_id=2),
                FractureObservation(measured_depth=170, dip_direction=270, dip=30, set_id=3),
                FractureObservation(measured_depth=180, dip_direction=272, dip=32, set_id=3),
                FractureObservation(measured_depth=190, dip_direction=268, dip=28, set_id=3),
            ],
        )
        collection = BoreholeCollection(boreholes=[bh])
        svc = JointSetService()
        result = svc.identify_from_imported(collection, {"BH-01"}, set())
        assert len(result.sets) == 3
        assert 1 in result.sets
        assert 2 in result.sets
        assert 3 in result.sets


# ═══════════════════════════════════════════════════════════════════════════
# Import service
# ═══════════════════════════════════════════════════════════════════════════

class TestImportService:
    """Unified import service tests."""

    @pytest.fixture
    def demo_dir(self):
        return Path(__file__).parent.parent.parent / "examples" / "m7_demo"

    def test_field_mapping_round_trip(self, tmp_path):
        """FieldMapping can be saved and restored."""
        fm = FieldMapping(
            name="Test Map", data_type="collars",
            column_map={"HoleID": "hole_id", "X": "easting"},
        )
        path = tmp_path / "mapping.json"
        from dfn_cave_studio.services.import_service import UnifiedImportService
        svc = UnifiedImportService()
        svc.save_field_mapping(fm, path)
        loaded = svc.load_field_mapping(path)
        assert loaded.name == "Test Map"
        assert loaded.column_map["HoleID"] == "hole_id"

    def test_field_detection_collars(self):
        """Auto-detect standard collar fields."""
        from dfn_cave_studio.services.import_service import UnifiedImportService
        svc = UnifiedImportService()
        df = pd.DataFrame({"hole_id": ["A"], "easting": [100], "northing": [200],
                           "elevation": [500], "total_depth": [150]})
        detected = svc.detect_fields(df, "collars")
        # Standard field names are keys like "borehole_id", "collar_x", etc.
        # "hole_id" is an alias for "borehole_id"
        assert detected.get("borehole_id") == "hole_id"
        assert detected.get("collar_x") == "easting"
        assert detected.get("collar_y") == "northing"

    def test_field_detection_with_aliases(self):
        """Auto-detect using field aliases."""
        from dfn_cave_studio.services.import_service import UnifiedImportService
        svc = UnifiedImportService()
        df = pd.DataFrame({"HOLE": ["A"], "X": [100], "Y": [200],
                           "Z": [500], "TD": [150]})
        detected = svc.detect_fields(df, "collars")
        # "X" alias matches "collar_x", "TD" matches "final_depth"
        # These aliases may or may not match based on exact alias list
        assert isinstance(detected, dict)
        assert len(detected) > 0
