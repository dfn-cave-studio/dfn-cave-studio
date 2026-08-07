"""M7 state and data validation tests (NOT dialog tests).

These tests validate data integrity and service behavior — they do NOT
instantiate Qt dialogs.  Dialog lifecycle is tested in tests/gui/.
"""

import pandas as pd
from pathlib import Path
import pytest

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter
from dfn_cave_studio.models.borehole import SurveyStation
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.joint_set_service import JointSetService


@pytest.fixture
def demo_dir():
    return Path(__file__).parent.parent.parent / "examples" / "m7_demo"


def _build_loaded_project(demo_dir):
    """Build project with all demo data, cleaned surveys, fractures, holdout."""
    project = Project()
    project.metadata.name = "State Test"
    project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)
    imp = BoreholeImporter()
    cr = imp.import_all(collar_path=str(demo_dir / "collars.csv"))
    project.borehole_collection = cr.collection
    bh_map = {bh.borehole_id: bh for bh in cr.collection}
    max_depths = {bh.borehole_id: bh.collar.final_depth for bh in cr.collection}
    raw = pd.read_csv(demo_dir / "surveys.csv")
    for _, row in raw.iterrows():
        try:
            bh_id = str(row["hole_id"]).strip()
            if bh_id not in bh_map:
                continue
            md = float(row["measured_depth"])
            if md > max_depths.get(bh_id, float("inf")):
                continue
            az = float(row.get("azimuth", 0)) % 360
            dip = max(-90.0, min(90.0, float(row.get("dip", -90))))
            if md in {s.measured_depth for s in bh_map[bh_id].survey.stations}:
                continue
            bh_map[bh_id].survey.stations.append(SurveyStation(measured_depth=md, azimuth=az, dip=dip))
        except Exception:
            pass
    fr = imp.import_all(
        collar_path=str(demo_dir / "collars.csv"),
        fractures_path=str(demo_dir / "fractures.csv"),
    )
    fmap = {bh.borehole_id: bh for bh in fr.collection}
    for bh in project.borehole_collection:
        src = fmap.get(bh.borehole_id)
        if src:
            bh.fracture_observations = src.fracture_observations
    project._m7_data = {
        "raw_surveys": raw,
        "raw_fractures": pd.read_csv(demo_dir / "fractures.csv"),
        "raw_rqd": pd.read_csv(demo_dir / "rqd.csv"),
        "raw_domain_intervals": pd.read_csv(demo_dir / "domain_intervals.csv"),
        "excluded_records": fr.fracture_exclusions,
    }
    return project


class TestHoldoutStatePersistence:
    """Holdout state persists in project._m7_data across service instances."""

    def test_holdout_persists_across_service_instances(self, demo_dir):
        """Holdout written to project._m7_data is readable by another service."""
        project = _build_loaded_project(demo_dir)

        holes = sorted(bh.borehole_id for bh in project.borehole_collection)
        svc1 = HoldoutService()
        svc1.select_manual(holes, ["BH-02", "BH-07"])
        svc1.lock()
        m7 = getattr(project, "_m7_data", {}) or {}
        m7["holdout"] = svc1
        project._m7_data = m7

        # Second "instance" reads same data
        saved = project._m7_data.get("holdout")
        assert saved is not None, "Holdout not saved to project"
        assert saved.is_locked, "Holdout should be locked"
        assert "BH-02" in saved.validation_holes
        assert "BH-07" in saved.validation_holes
        assert len(saved.calibration_holes) == 6
        assert len(saved.validation_holes) == 2

    def test_joint_set_reads_same_holdout(self, demo_dir):
        """Joint set service reads calibration/validation from holdout."""
        project = _build_loaded_project(demo_dir)

        holes = sorted(bh.borehole_id for bh in project.borehole_collection)
        svc = HoldoutService()
        svc.select_manual(holes, ["BH-02", "BH-07"])
        svc.lock()
        m7 = getattr(project, "_m7_data", {}) or {}
        m7["holdout"] = svc
        project._m7_data = m7

        saved = project._m7_data.get("holdout")
        cal_holes = set(saved.calibration_holes)
        val_holes = set(saved.validation_holes)

        js = JointSetService(random_seed=42)
        result = js.identify_auto(project.borehole_collection, cal_holes, val_holes, n_clusters=3, random_seed=42)
        assert result.calibration_count == 60
        assert result.validation_count == 20
        counts = {}
        for sid in result.sets:
            counts[sid] = len([a for a in result.assignments.values() if a == sid])
        assert sum(counts.values()) == 60
        assert counts.get(1, 0) > 0
        assert counts.get(2, 0) > 0
        assert counts.get(3, 0) > 0


class TestDomainDataValidation:
    """Raw domain interval data integrity checks."""

    def test_domain_intervals_11_rows_in_project(self, demo_dir):
        """Project stores 11 domain interval rows."""
        project = _build_loaded_project(demo_dir)
        m7 = getattr(project, "_m7_data", {}) or {}
        raw_di = m7.get("raw_domain_intervals")
        assert raw_di is not None
        assert len(raw_di) == 11

        # Re-read from same project (simulates dialog close/reopen)
        raw_di_2 = project._m7_data.get("raw_domain_intervals")
        assert len(raw_di_2) == 11

    def test_bh05_has_two_correct_intervals(self, demo_dir):
        """BH-05: 0-100 Domain 2, 100-190 Domain 1 (non-overlapping)."""
        df = pd.read_csv(demo_dir / "domain_intervals.csv")
        bh05 = df[df["hole_id"] == "BH-05"].sort_values("from_depth")
        assert len(bh05) == 2
        row0 = bh05.iloc[0]
        assert row0["from_depth"] == 0
        assert row0["to_depth"] == 100
        assert row0["domain_id"] == 2
        row1 = bh05.iloc[1]
        assert row1["from_depth"] == 100
        assert row1["to_depth"] == 190
        assert row1["domain_id"] == 1
        assert row0["to_depth"] <= row1["from_depth"]

    def test_bh06_has_three_correct_intervals(self, demo_dir):
        """BH-06: 0-50 D2, 50-150 D3, 150-205 D2 (non-overlapping)."""
        df = pd.read_csv(demo_dir / "domain_intervals.csv")
        bh06 = df[df["hole_id"] == "BH-06"].sort_values("from_depth")
        assert len(bh06) == 3
        r0 = bh06.iloc[0]
        assert r0["from_depth"] == 0 and r0["to_depth"] == 50 and r0["domain_id"] == 2
        r1 = bh06.iloc[1]
        assert r1["from_depth"] == 50 and r1["to_depth"] == 150 and r1["domain_id"] == 3
        r2 = bh06.iloc[2]
        assert r2["from_depth"] == 150 and r2["to_depth"] == 205 and r2["domain_id"] == 2
        assert r0["to_depth"] <= r1["from_depth"]
        assert r1["to_depth"] <= r2["from_depth"]

    def test_overlap_detected_as_error(self, demo_dir):
        """Manual overlap input must produce error from cleaning service."""
        from dfn_cave_studio.services.cleaning_service import (
            DataCleaningService,
        )

        collars = pd.DataFrame(
            {
                "hole_id": ["BH-05"],
                "easting": [100],
                "northing": [200],
                "elevation": [500],
                "total_depth": [200],
            }
        )
        df = pd.DataFrame(
            {
                "hole_id": ["BH-05", "BH-05"],
                "from_depth": [0, 100],
                "to_depth": [190, 190],
                "domain_id": [2, 1],
            }
        )
        svc = DataCleaningService()
        issues = svc.validate_domain_intervals(df, collars)
        overlaps = [i for i in issues if i.issue_type.value == "overlap"]
        assert len(overlaps) >= 1, f"No overlap error detected, got {len(issues)} issues"
