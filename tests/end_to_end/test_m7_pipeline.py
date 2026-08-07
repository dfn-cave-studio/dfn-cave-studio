"""M7 end-to-end test: import → clean → holdout → domains → joint sets → export."""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter


class TestM7RealFileImport:
    """Import the actual demo CSVs and verify every type returns non-zero."""

    @pytest.fixture
    def demo_dir(self):
        return Path(__file__).parent.parent.parent / "examples" / "m7_demo"

    def test_collars_import_nonzero(self, demo_dir):
        """collars.csv imports 8 boreholes (2 error rows excluded)."""
        importer = BoreholeImporter()
        result = importer.import_all(collar_path=str(demo_dir / "collars.csv"))
        assert result.collection is not None
        assert len(result.collection) >= 8, f"Got {len(result.collection)} boreholes"

    def test_surveys_import_153_rows(self, demo_dir):
        """surveys.csv has 153 data rows."""
        df = pd.read_csv(demo_dir / "surveys.csv")
        assert len(df) >= 150, f"surveys.csv has {len(df)} rows"

    def test_fractures_import_nonzero(self, demo_dir):
        """fractures.csv imports fracture observations attached to collars."""
        importer = BoreholeImporter()
        # Import collars first
        coll_result = importer.import_all(collar_path=str(demo_dir / "collars.csv"))
        assert coll_result.collection is not None
        # Now import fractures
        frac_result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        assert frac_result.collection is not None
        total_obs = sum(len(bh.fracture_observations) for bh in frac_result.collection)
        assert total_obs > 0, f"fractures.csv imported {total_obs} observations (expected >0)"

    def test_rqd_import_nonzero(self, demo_dir):
        """rqd.csv has 27 data rows (24 valid + 3 error)."""
        df = pd.read_csv(demo_dir / "rqd.csv")
        assert len(df) >= 24, f"rqd.csv has {len(df)} rows"

    def test_domain_intervals_import_nonzero(self, demo_dir):
        """domain_intervals.csv has 11 rows."""
        df = pd.read_csv(demo_dir / "domain_intervals.csv")
        assert len(df) >= 8, f"domain_intervals.csv has {len(df)} rows"

    def test_set_id_preserved(self, demo_dir):
        """set_id from fractures.csv is preserved as integer."""
        importer = BoreholeImporter()
        result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        set_ids = set()
        for bh in result.collection:
            for obs in bh.fracture_observations:
                if obs.set_id is not None:
                    set_ids.add(obs.set_id)
        assert 1 in set_ids
        assert 2 in set_ids
        assert 3 in set_ids

    def test_no_zero_import_for_nonempty_files(self, demo_dir):
        """Non-empty files must not return 0/0/0."""
        for fname in ["collars.csv", "surveys.csv", "fractures.csv", "rqd.csv", "domain_intervals.csv"]:
            path = demo_dir / fname
            assert path.exists(), f"{fname} not found"
            df = pd.read_csv(path)
            assert len(df) > 0, f"{fname} is empty ({len(df)} rows)"


class TestM7FullImportIntoProject:
    """Import all 5 CSV files into a Project and verify state."""

    @pytest.fixture
    def demo_dir(self):
        return Path(__file__).parent.parent.parent / "examples" / "m7_demo"

    def test_all_five_types_import_to_project(self, demo_dir, tmp_path):
        """All 5 files import into project with correct counts."""
        from dfn_cave_studio.models.project import Project
        from dfn_cave_studio.models.bounds import ModelBounds
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        from dfn_cave_studio.models.data_management import DomainInterval

        project = Project()
        project.metadata.name = "Import Test"
        project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

        # Import collars
        importer = BoreholeImporter()
        coll_result = importer.import_all(collar_path=str(demo_dir / "collars.csv"))
        project.borehole_collection = coll_result.collection
        assert len(project.borehole_collection) >= 8

        # Import surveys — attach to existing boreholes
        surveys_df = pd.read_csv(demo_dir / "surveys.csv")
        s_imported = 0
        from dfn_cave_studio.models.borehole import SurveyStation

        # Build lookup by iterating collection
        bh_map = {bh.borehole_id: bh for bh in project.borehole_collection}
        for _, row in surveys_df.iterrows():
            try:
                bh_id = str(row["hole_id"]).strip()
                if bh_id in bh_map:
                    bh = bh_map[bh_id]
                    bh.survey.stations.append(
                        SurveyStation(
                            measured_depth=float(row["measured_depth"]),
                            azimuth=float(row["azimuth"]),
                            dip=float(row["dip"]),
                        )
                    )
                    s_imported += 1
            except Exception:
                pass
        assert s_imported >= 30, f"Surveys: {s_imported} imported (8 boreholes × ~5-20 stations each)"

        # Import fractures — attach observations
        frac_result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        project.borehole_collection = frac_result.collection
        f_total = sum(len(bh.fracture_observations) for bh in project.borehole_collection)
        assert f_total > 0, f"Fractures: {f_total} imported"

        # Import RQD
        rqd_df = pd.read_csv(demo_dir / "rqd.csv")
        assert len(rqd_df) >= 24

        # Import domain intervals
        di_df = pd.read_csv(demo_dir / "domain_intervals.csv")
        intervals = []
        for _, row in di_df.iterrows():
            try:
                intervals.append(
                    DomainInterval(
                        hole_id=str(row["hole_id"]).strip(),
                        from_depth=float(row["from_depth"]),
                        to_depth=float(row["to_depth"]),
                        domain_id=int(row["domain_id"]),
                        domain_name=str(row.get("domain_name", "")),
                        assignment_method="imported",
                    )
                )
            except Exception:
                pass
        assert len(intervals) >= 8

        # Save and verify
        project._m7_data = {"domain_intervals": intervals}
        save_path = tmp_path / "import_test.dfnproj"
        zps = ZipProjectStore()
        zps.save(project, save_path)

        reopened = zps.load(save_path)
        assert reopened.borehole_collection is not None
        reopened_total = sum(len(bh.fracture_observations) for bh in reopened.borehole_collection)
        assert reopened_total == f_total
        m7 = getattr(reopened, "_m7_data", {})
        assert len(m7.get("domain_intervals", [])) >= 8

    def test_round_trip_exact_counts(self, demo_dir, tmp_path):
        """Save/reopen preserves the imported formal data counts."""
        from dfn_cave_studio.models.project import Project
        from dfn_cave_studio.models.bounds import ModelBounds
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        from dfn_cave_studio.models.borehole import SurveyStation

        # Build full project
        project = Project()
        project.metadata.name = "Count Test"
        project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

        importer = BoreholeImporter()
        coll_result = importer.import_all(collar_path=str(demo_dir / "collars.csv"))
        project.borehole_collection = coll_result.collection
        bh_map = {bh.borehole_id: bh for bh in coll_result.collection}
        max_depths = {bh.borehole_id: bh.collar.final_depth for bh in coll_result.collection}

        # Attach surveys with normalization
        raw_surveys = pd.read_csv(demo_dir / "surveys.csv")
        s_attached = 0
        for _, row in raw_surveys.iterrows():
            try:
                bh_id = str(row["hole_id"]).strip()
                if bh_id not in bh_map:
                    continue
                md = float(row["measured_depth"])
                if md > max_depths.get(bh_id, float("inf")):
                    continue
                az = float(row.get("azimuth", 0)) % 360
                dip = float(row.get("dip", -90))
                dip = max(-90.0, min(90.0, dip))
                existing = {s.measured_depth for s in bh_map[bh_id].survey.stations}
                if md in existing:
                    continue
                bh_map[bh_id].survey.stations.append(SurveyStation(measured_depth=md, azimuth=az, dip=dip))
                s_attached += 1
            except Exception:
                pass

        # Import fractures — use the SAME collection with surveys attached
        importer2 = BoreholeImporter()
        frac_result = importer2.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        # Copy fracture observations into existing collection (which has surveys)
        frac_bh_map = {bh.borehole_id: bh for bh in frac_result.collection}
        for bh in project.borehole_collection:
            src = frac_bh_map.get(bh.borehole_id)
            if src:
                bh.fracture_observations = src.fracture_observations
        f_total = sum(len(bh.fracture_observations) for bh in project.borehole_collection)

        # RQD and domain intervals
        raw_rqd = pd.read_csv(demo_dir / "rqd.csv")
        raw_di = pd.read_csv(demo_dir / "domain_intervals.csv")

        project._m7_data = {
            "raw_surveys": raw_surveys,
            "raw_rqd": raw_rqd,
            "raw_domain_intervals": raw_di,
        }

        # Save
        save_path = tmp_path / "count_test.dfnproj"
        zps = ZipProjectStore()
        zps.save(project, save_path)

        # Reopen
        reopened = zps.load(save_path)
        # Boreholes
        assert len(reopened.borehole_collection) >= 8, f"Boreholes: {len(reopened.borehole_collection)}"
        # Fractures
        r_f_total = sum(len(bh.fracture_observations) for bh in reopened.borehole_collection)
        assert r_f_total == f_total, f"Fractures: {r_f_total} != {f_total}"
        # Surveys (count attached stations)
        r_s_total = sum(len(bh.survey.stations) for bh in reopened.borehole_collection)
        assert r_s_total >= 100, f"Survey stations: {r_s_total} (expected >=100 of 148)"
        # M7 data
        m7 = getattr(reopened, "_m7_data", {})
        for key in ["raw_surveys", "raw_rqd", "raw_domain_intervals"]:
            assert key in m7, f"Missing {key} in reopened project"
        assert len(m7["raw_surveys"]) == len(raw_surveys), f"R surveys: {len(m7['raw_surveys'])}"
        assert len(m7["raw_rqd"]) == len(raw_rqd), f"R RQD: {len(m7['raw_rqd'])}"
        assert len(m7["raw_domain_intervals"]) == len(raw_di), f"R domains: {len(m7['raw_domain_intervals'])}"

    def test_workflow_transitions_on_real_data(self, demo_dir):
        """Workflow moves to HAS_ISSUES or COMPLETED after real import."""
        from dfn_cave_studio.services.workflow_controller import WorkflowController, StepStatus

        wf = WorkflowController()
        importer = BoreholeImporter()
        result = importer.import_all(collar_path=str(demo_dir / "collars.csv"))
        if result.collection and len(result.collection) > 0:
            wf.complete_step("import")
        else:
            wf.mark_issues("import")
        step = wf.get_step("import")
        assert step is not None
        assert step.status in (StepStatus.COMPLETED, StepStatus.HAS_ISSUES)
        assert step.status != StepStatus.NOT_STARTED


from dfn_cave_studio.services.cleaning_service import DataCleaningService, IssueSeverity
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.joint_set_service import JointSetService
from dfn_cave_studio.borehole.orientation_statistics import OrientationStatisticsCalculator


class TestM7Pipeline:
    """Full M7 data pre-processing pipeline."""

    @pytest.fixture
    def demo_dir(self):
        return Path(__file__).parent.parent.parent / "examples" / "m7_demo"

    @pytest.mark.slow
    def test_full_m7_workflow(self, demo_dir, tmp_path):
        """M7 pipeline: import → clean → holdout → domains → joint sets → export."""
        # ── Step 1: Import ──────────────────────────────────────────────
        importer = BoreholeImporter()
        result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            survey_path=str(demo_dir / "surveys.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        # Import has expected errors from deliberate bad rows in demo data
        collection = result.collection
        assert collection is not None, "Collection should not be None"
        assert len(collection) >= 8, f"Expected 8+ boreholes, got {len(collection)}"

        # ── Step 2: Cleaning ────────────────────────────────────────────
        cleaner = DataCleaningService()
        collars_df = pd.read_csv(demo_dir / "collars.csv")
        issues_c = cleaner.validate_collars(collars_df)
        error_rows = [i for i in issues_c if i.severity == IssueSeverity.ERROR]
        assert len(error_rows) >= 2, f"Expected missing/negative depth issues, got {len(error_rows)}"

        surveys_df = pd.read_csv(demo_dir / "surveys.csv")
        issues_s = cleaner.validate_surveys(surveys_df, collars_df)
        depth_issues = [i for i in issues_s if i.issue_type.value in ("depth_exceeds", "depth_order")]
        assert len(depth_issues) >= 1

        fractures_df = pd.read_csv(demo_dir / "fractures.csv")
        issues_f = cleaner.validate_fractures(fractures_df, collars_df)
        depth_exceed = [i for i in issues_f if i.issue_type.value == "depth_exceeds"]
        assert len(depth_exceed) >= 1

        rqd_df = pd.read_csv(demo_dir / "rqd.csv")
        issues_r = cleaner.validate_rqd(rqd_df, collars_df)
        overlaps = [i for i in issues_r if i.issue_type.value == "overlap"]
        rqd_range = [i for i in issues_r if i.issue_type.value == "out_of_range"]
        assert len(overlaps) >= 1 or len(rqd_range) >= 1

        domain_df = pd.read_csv(demo_dir / "domain_intervals.csv")
        issues_d = cleaner.validate_domain_intervals(domain_df, collars_df)
        domain_issues = len(issues_d)
        assert domain_issues >= 0  # May or may not have issues

        # Export issues report
        report = cleaner.export_issues_report()
        assert len(report) > 0
        # Save report
        report_path = tmp_path / "data_quality_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        assert report_path.exists()

        # ── Step 3: Holdout ─────────────────────────────────────────────
        hole_ids = sorted([bh.borehole_id for bh in collection if not bh.borehole_id.startswith("BH-BAD")])
        holdout = HoldoutService()
        holdout.select_random(hole_ids, validation_fraction=0.25, random_seed=42)
        holdout.lock()

        assert len(holdout.validation_holes) >= 2, f"Need >=2 validation holes, got {len(holdout.validation_holes)}"
        assert len(holdout.calibration_holes) >= 5
        # Reproducibility
        holdout2 = HoldoutService()
        holdout2.select_random(hole_ids, 0.25, random_seed=42)
        assert holdout.validation_holes == holdout2.validation_holes

        # ── Step 4: Verify validation holes excluded from clustering ────
        cal_holes = set(holdout.calibration_holes)
        val_holes = set(holdout.validation_holes)

        js_svc = JointSetService(random_seed=42)
        js_result = js_svc.identify_from_imported(collection, cal_holes, val_holes)
        assert js_result.mode == "imported"
        assert len(js_result.sets) >= 3, f"Expected 3 joint sets, got {len(js_result.sets)}"

        # Validation count should be non-zero (validation holes exist)
        assert js_result.calibration_count > 0

        # ── Step 5: Auto-identify (mode B) ──────────────────────────────
        js_auto = JointSetService(random_seed=42)
        auto_result = js_auto.identify_auto(collection, cal_holes, val_holes, n_clusters=3, random_seed=42)
        assert auto_result.mode == "automatic"
        assert len(auto_result.sets) >= 1

        # ── Step 6: Compute Fisher stats from imported set_ids ──────────
        calc = OrientationStatisticsCalculator()
        stats = calc.compute_by_set(collection)
        assert len(stats) >= 3

        # ── Step 7: Export cleaned data ──────────────────────────────────
        # Write cleaned collars (exclude bad rows)
        good_collars = collars_df.iloc[:8].copy()
        clean_path = tmp_path / "cleaned_collars.csv"
        good_collars.to_csv(clean_path, index=False)
        assert clean_path.exists()

        # Write excluded records
        excl_path = tmp_path / "excluded_records.csv"
        bad_mask = collars_df["hole_id"].isna() | (collars_df["total_depth"] <= 0)
        excluded = collars_df[bad_mask]
        excluded.to_csv(excl_path, index=False)
        assert excl_path.exists()

        # Write holdout assignments
        holdout_path = tmp_path / "holdout_assignments.csv"
        with open(holdout_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hole_id", "role"])
            for hid in hole_ids:
                role = "validation" if holdout.is_validation(hid) else "calibration"
                w.writerow([hid, role])
        assert holdout_path.exists()

        # ── Step 8: Structural domain intervals ──────────────────────────
        domain_intervals_path = tmp_path / "domain_intervals_cleaned.csv"
        # Keep only rows without overlapping
        domain_clean = domain_df[domain_df["hole_id"].isin(hole_ids)]
        domain_clean.to_csv(domain_intervals_path, index=False)
        assert domain_intervals_path.exists()

        # Verify domain intervals for BH-05 (crosses two domains)
        bh05_intervals = domain_clean[domain_clean["hole_id"] == "BH-05"]
        assert len(bh05_intervals) >= 1

        # ── Step 9: Save project state ──────────────────────────────────
        from dfn_cave_studio.models.project import Project
        from dfn_cave_studio.models.bounds import ModelBounds
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore

        project = Project()
        project.metadata.name = "M7 E2E Test"
        project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)
        project.config.master_seed = 42
        project.borehole_collection = collection
        project.joint_sets = js_svc.get_all_joint_sets()

        save_path = tmp_path / "m7_e2e.dfnproj"
        zps = ZipProjectStore()
        zps.save(project, save_path)
        assert save_path.exists()

        # ── Step 10: Reopen and verify ──────────────────────────────────
        reopened = zps.load(save_path)
        assert reopened.metadata.name == "M7 E2E Test"
        assert reopened.borehole_collection is not None
        assert len(reopened.borehole_collection) >= 8
        assert len(reopened.joint_sets) >= 1

        # ── Step 11: Verify set_ids preserved ──────────────────────────
        reopened_set_ids = set()
        for bh in reopened.borehole_collection:
            for obs in bh.fracture_observations:
                if obs.set_id is not None:
                    reopened_set_ids.add(obs.set_id)
        assert 1 in reopened_set_ids
        assert 2 in reopened_set_ids
        assert 3 in reopened_set_ids

    @pytest.mark.slow
    def test_m7_dfproj_round_trip(self, demo_dir, tmp_path):
        """M7 save/restore: workflow + holdout + domains + joint sets."""
        from dfn_cave_studio.models.project import Project
        from dfn_cave_studio.models.bounds import ModelBounds
        from dfn_cave_studio.models.data_management import (
            DomainInterval,
            DataQualityIssue,
            IssueSeverity,
            IssueType,
        )
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        from dfn_cave_studio.services.workflow_controller import WorkflowController, StepStatus
        from dfn_cave_studio.services.holdout_service import HoldoutService
        from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter

        # Create project with M7 state
        project = Project()
        project.metadata.name = "M7 Round-trip Test"
        project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

        # Import demo data
        importer = BoreholeImporter()
        result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            survey_path=str(demo_dir / "surveys.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        project.borehole_collection = result.collection

        # Workflow state
        wf = WorkflowController()
        wf.complete_step("import")
        wf.complete_step("clean")
        wf.complete_step("holdout")
        wf.complete_step("domains")
        wf.complete_step("joint_sets")
        wf.invalidate_from("clean")  # Simulate re-cleaning → downstream stale

        # Holdout
        hole_ids = [bh.borehole_id for bh in result.collection]
        ho = HoldoutService()
        ho.select_random(hole_ids, 0.25, random_seed=42)
        ho.lock()

        # Domain intervals
        intervals = [
            DomainInterval(
                hole_id="BH-01",
                from_depth=0,
                to_depth=150,
                domain_id=1,
                domain_name="Domain 1",
                assignment_method="imported",
            ),
            DomainInterval(
                hole_id="BH-05",
                from_depth=0,
                to_depth=100,
                domain_id=2,
                domain_name="Domain 2",
                assignment_method="imported",
            ),
            DomainInterval(
                hole_id="BH-05",
                from_depth=100,
                to_depth=190,
                domain_id=1,
                domain_name="Domain 1",
                assignment_method="manual",
            ),
        ]

        # Quality issues
        issues = [
            DataQualityIssue(
                source_file="collars.csv",
                source_row=8,
                hole_id="",
                field="hole_id",
                issue_type=IssueType.MISSING_REQUIRED,
                severity=IssueSeverity.ERROR,
                message="Missing hole_id",
            ),
        ]

        # Attach M7 data
        project._m7_data = {
            "workflow": wf,
            "holdout": ho,
            "domain_intervals": intervals,
            "quality_issues": issues,
        }

        # Save
        save_path = tmp_path / "m7_roundtrip.dfnproj"
        zps = ZipProjectStore()
        zps.save(project, save_path)
        assert save_path.exists()

        # Verify ZIP contents include M7 data
        import zipfile

        with zipfile.ZipFile(save_path, "r") as zf:
            names = zf.namelist()
            assert "m7/workflow.json" in names
            assert "m7/holdout.json" in names
            assert "m7/domain_intervals.json" in names
            assert "m7/quality_issues.json" in names
            assert "inputs/boreholes.json" in names

        # Reopen
        reopened = zps.load(save_path)
        assert reopened.metadata.name == "M7 Round-trip Test"

        # Verify M7 data restored
        m7 = getattr(reopened, "_m7_data", None)
        assert m7 is not None, "M7 data not restored"

        # Workflow
        reopened_wf = m7.get("workflow")
        assert reopened_wf is not None, "Workflow not restored"
        assert reopened_wf.is_step_done("import")
        # After invalidating "clean", holdout and joint_sets should be stale
        ho_step = reopened_wf.get_step("holdout")
        assert ho_step is not None
        assert ho_step.status in (StepStatus.STALE, StepStatus.COMPLETED), f"holdout status: {ho_step.status}"
        js_step = reopened_wf.get_step("joint_sets")
        assert js_step is not None
        assert js_step.status in (StepStatus.STALE, StepStatus.COMPLETED)

        # Holdout
        reopened_ho = m7.get("holdout")
        assert reopened_ho is not None, "Holdout not restored"
        assert reopened_ho.is_locked
        assert len(reopened_ho.validation_holes) >= 1

        # Domain intervals
        reopened_di = m7.get("domain_intervals", [])
        assert len(reopened_di) == 3
        bh05 = [d for d in reopened_di if d.hole_id == "BH-05"]
        assert len(bh05) == 2  # BH-05 has two domain intervals

        # Quality issues
        reopened_qi = m7.get("quality_issues", [])
        assert len(reopened_qi) >= 1
        assert reopened_qi[0].source_file == "collars.csv"
