"""Regression tests for M7 fixes: validation leak, domain visibility, round-trip, clear."""

import pandas as pd
from pathlib import Path
import tempfile
import pytest

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter
from dfn_cave_studio.models.borehole import SurveyStation
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.joint_set_service import JointSetService
from dfn_cave_studio.services.workflow_controller import WorkflowController, StepStatus
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.m7_state import get_excluded_records


@pytest.fixture
def demo_dir():
    return Path(__file__).parent.parent.parent / "examples" / "m7_demo"


@pytest.fixture
def loaded_project(demo_dir):
    """Project with all demo data loaded and cleaned."""
    project = Project()
    project.metadata.name = "Regression Test"
    project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

    imp = BoreholeImporter()
    cr = imp.import_all(collar_path=str(demo_dir / "collars.csv"))
    project.borehole_collection = cr.collection
    bh_map = {bh.borehole_id: bh for bh in cr.collection}
    max_depths = {bh.borehole_id: bh.collar.final_depth for bh in cr.collection}

    # Import and clean surveys
    raw = pd.read_csv(demo_dir / "surveys.csv")
    attached = 0
    excluded = 0
    for _, row in raw.iterrows():
        try:
            bh_id = str(row["hole_id"]).strip()
            if bh_id not in bh_map:
                excluded += 1
                continue
            md = float(row["measured_depth"])
            if md > max_depths.get(bh_id, float("inf")):
                excluded += 1
                continue
            az = float(row.get("azimuth", 0)) % 360
            dip = max(-90.0, min(90.0, float(row.get("dip", -90))))
            if md in {s.measured_depth for s in bh_map[bh_id].survey.stations}:
                excluded += 1
                continue
            bh_map[bh_id].survey.stations.append(SurveyStation(measured_depth=md, azimuth=az, dip=dip))
            attached += 1
        except Exception:
            excluded += 1

    # Import fractures
    fr = imp.import_all(
        collar_path=str(demo_dir / "collars.csv"),
        fractures_path=str(demo_dir / "fractures.csv"),
    )
    raw_fractures = pd.read_csv(demo_dir / "fractures.csv")
    assert fr.fracture_rows_raw == 83
    assert fr.fracture_rows_imported == 80
    assert fr.fracture_rows_excluded == 3
    assert len(fr.fracture_exclusions) == 3
    assert {record["file_line"] for record in fr.fracture_exclusions} == {
        82,
        83,
        84,
    }
    assert {record["field"] for record in fr.fracture_exclusions} == {
        "measured_depth",
        "dip",
        "set_id",
    }
    fr_map = {bh.borehole_id: bh for bh in fr.collection}
    for bh in project.borehole_collection:
        src = fr_map.get(bh.borehole_id)
        if src:
            bh.fracture_observations = src.fracture_observations

    # Setup holdout: BH-02, BH-07 = validation
    hole_ids = sorted(bh_map.keys())
    ho = HoldoutService()
    ho.select_manual(hole_ids, ["BH-02", "BH-07"])
    ho.lock()

    project._m7_data = {
        "raw_surveys": raw,
        "raw_fractures": raw_fractures,
        "raw_rqd": pd.read_csv(demo_dir / "rqd.csv"),
        "raw_domain_intervals": pd.read_csv(demo_dir / "domain_intervals.csv"),
        "excluded_records": fr.fracture_exclusions,
        "holdout": ho,
    }
    return project, ho


class TestValidationLeak:
    """Joint set identification only uses calibration holes."""

    def test_joint_set_uses_calibration_only(self, loaded_project):
        project, ho = loaded_project
        collection = project.borehole_collection

        # Count total fractures
        total_fracs = sum(len(bh.fracture_observations) for bh in collection)
        assert total_fracs == 80, f"Total fractures: {total_fracs}"
        assert all(
            observation.measured_depth <= borehole.collar.final_depth
            and 0.0 <= observation.dip <= 90.0
            and isinstance(observation.set_id, int)
            for borehole in collection
            for observation in borehole.fracture_observations
        )

        # Count validation fractures
        val_fracs = sum(len(bh.fracture_observations) for bh in collection if ho.is_validation(bh.borehole_id))
        assert val_fracs == 20, f"Validation fractures: {val_fracs}"

        cal_holes = set(ho.calibration_holes)
        val_holes = set(ho.validation_holes)

        js = JointSetService(random_seed=42)
        result = js.identify_auto(collection, cal_holes, val_holes, n_clusters=3, random_seed=42)

        # The invalid BH-01 depth=9999 row must not enter fitting.
        assert result.calibration_count == 60
        assert result.validation_count == 20

        # Every valid calibration fracture is assigned exactly once.
        assert len(result.assignments) == 60
        assert len(set(result.assignments)) == 60
        assert (
            sum(
                len([assignment for assignment in result.assignments.values() if assignment == set_id])
                for set_id in result.sets
            )
            == 60
        )

        # No validation fractures in training assignments
        for bh in collection:
            if bh.borehole_id in val_holes:
                for i, obs in enumerate(bh.fracture_observations):
                    record_id = f"{bh.borehole_id}:{i}"
                    assert record_id not in result.assignments, f"Validation record {record_id} leaked into clustering"


class TestDomainVisibility:
    """Imported domain intervals are visible."""

    def test_imported_domain_intervals_in_project(self, loaded_project):
        project, _ = loaded_project
        m7 = getattr(project, "_m7_data", {})
        raw_di = m7.get("raw_domain_intervals")
        assert raw_di is not None
        assert len(raw_di) == 11, f"Domain intervals: {len(raw_di)}"


class TestRoundTrip:
    """Save/reopen preserves workflow, holdout, domains."""

    def test_m7_project_round_trip_restores_workflow(self, loaded_project):
        project, ho = loaded_project

        # Setup workflow
        wf = WorkflowController()
        wf.complete_step("import")
        wf.complete_step("clean")
        wf.complete_step("holdout")
        wf.complete_step("domains")
        wf.complete_step("joint_sets")
        project._m7_data["workflow"] = wf

        # Save
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "regression.dfnproj"
        zps = ZipProjectStore()
        zps.save(project, path)

        # Reopen into new project
        reopened = zps.load(path)
        m7 = getattr(reopened, "_m7_data", {}) or {}

        # Verify data counts
        assert len(reopened.borehole_collection) == 8
        s = sum(len(bh.survey.stations) for bh in reopened.borehole_collection)
        assert s == 148, f"Survey stations: {s}"
        f_total = sum(len(bh.fracture_observations) for bh in reopened.borehole_collection)
        assert f_total == 80

        # Verify raw data preserved
        assert len(m7.get("raw_surveys", [])) == 153
        assert len(m7.get("raw_fractures", [])) == 83
        assert len(m7.get("raw_rqd", [])) == 27
        assert len(m7.get("raw_domain_intervals", [])) == 11
        exclusions = get_excluded_records(reopened)
        assert len(exclusions) == 3
        assert {record["source_row"] for record in exclusions} == {80, 81, 82}

        # Verify workflow
        restored_wf = m7.get("workflow")
        assert restored_wf is not None
        assert restored_wf.is_step_done("import") or restored_wf.get_step("import").status == StepStatus.HAS_ISSUES
        assert restored_wf.is_step_done("holdout") or restored_wf.get_step("holdout").status == StepStatus.COMPLETED

        # Verify holdout
        restored_ho = m7.get("holdout")
        assert restored_ho is not None
        assert restored_ho.is_locked
        assert "BH-02" in restored_ho.validation_holes
        assert "BH-07" in restored_ho.validation_holes
        assert len(restored_ho.calibration_holes) == 6

        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


class TestNewProjectClears:
    """New Project clears all state."""

    def test_new_project_via_store(self, loaded_project):
        project, _ = loaded_project
        from dfn_cave_studio.persistence.project_store import ProjectStore

        store = ProjectStore()
        store._current_project = project
        new_proj = store.new_project("Clean Project")
        assert new_proj.metadata.name == "Clean Project"
        assert new_proj.borehole_collection is not None  # empty collection
        assert len(new_proj.borehole_collection) == 0
        assert len(new_proj.joint_sets) == 0
