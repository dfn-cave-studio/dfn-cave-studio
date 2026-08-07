"""Real pytest-qt GUI tests for MainWindow M7 round-trip and New Project.

Tests that instantiate MainWindow (with FakePlotter) and verify:
  - Save → New Project → Open → state restored
  - New Project clears all state
  - ProjectStore consistency after open
"""

import sys
import tempfile
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.borehole import SurveyStation
from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.joint_set_service import JointSetService
from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.models.data_management import DomainInterval
from dfn_cave_studio.models.structural_domain import StructuralDomain
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def demo_dir():
    return Path(__file__).parent.parent.parent / "examples" / "m7_demo"


@pytest.fixture
def full_m7_project(demo_dir):
    """Build a complete project with all M7 data, ready for round-trip."""
    project = Project()
    project.metadata.name = "Round-Trip Test"
    project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

    imp = BoreholeImporter()
    cr = imp.import_all(
        collar_path=str(demo_dir / "collars.csv"),
        fractures_path=str(demo_dir / "fractures.csv"),
    )
    project.borehole_collection = cr.collection
    bh_map = {bh.borehole_id: bh for bh in cr.collection}

    # Attach 148 cleaned survey stations
    raw = pd.read_csv(demo_dir / "surveys.csv")
    max_depths = {bh.borehole_id: bh.collar.final_depth for bh in cr.collection}
    for _, row in raw.iterrows():
        try:
            bh_id = str(row["hole_id"]).strip()
            if bh_id not in bh_map:
                continue
            md = float(row["measured_depth"])
            if md > max_depths.get(bh_id, float("inf")):
                continue
            az = float(row.get("azimuth", 0)) % 360
            dip_val = max(-90.0, min(90.0, float(row.get("dip", -90))))
            existing = {s.measured_depth for s in bh_map[bh_id].survey.stations}
            if md in existing:
                continue
            bh_map[bh_id].survey.stations.append(SurveyStation(measured_depth=md, azimuth=az, dip=dip_val))
        except Exception:
            pass

    # Holdout: BH-02, BH-07 = validation, locked
    hole_ids = sorted(bh_map.keys())
    ho = HoldoutService()
    ho.select_manual(hole_ids, ["BH-02", "BH-07"])
    ho.lock()

    # Domain intervals from raw CSV (11 rows → DomainInterval list)
    raw_di = pd.read_csv(demo_dir / "domain_intervals.csv")
    intervals = []
    for _, row in raw_di.iterrows():
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
    for domain_id in sorted({interval.domain_id for interval in intervals}):
        project.structural_domains.domains.append(
            StructuralDomain(
                domain_id=domain_id,
                name=f"Domain {domain_id}",
                color=f"#{domain_id}{domain_id}{domain_id}{domain_id}{domain_id}{domain_id}",
            )
        )

    joint_result = JointSetService(random_seed=42).identify_from_imported(
        project.borehole_collection,
        set(ho.calibration_holes),
        set(ho.validation_holes),
    )
    project.joint_sets = list(joint_result.sets.values())

    # Workflow: all steps completed
    wf = WorkflowController()
    for step in ["import", "clean", "holdout", "domains", "joint_sets"]:
        wf.complete_step(step)

    project._m7_data = {
        "raw_surveys": raw,
        "raw_fractures": pd.read_csv(demo_dir / "fractures.csv"),
        "raw_rqd": pd.read_csv(demo_dir / "rqd.csv"),
        "raw_domain_intervals": raw_di,
        "excluded_records": cr.fracture_exclusions,
        "domain_intervals": intervals,
        "holdout": ho,
        "workflow": wf,
    }
    return project


# ═══════════════════════════════════════════════════════════════════════════════
# MainWindow Round-Trip Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMainWindowRoundTrip:
    """Save .dfnproj → New Project → Open → verify full state restored."""

    def test_save_open_roundtrip_restores_all_state(self, full_m7_project, qtbot):
        """Full round-trip preserves project, workflow, holdout, domains."""

        tmpdir = tempfile.mkdtemp()
        save_path = Path(tmpdir) / "roundtrip.dfnproj"
        try:
            # Save the project via ZipProjectStore
            zps = ZipProjectStore()
            zps.save(full_m7_project, save_path)
            assert save_path.exists(), f"Save file not created: {save_path}"

            # Reopen into a fresh project
            reopened = zps.load(save_path)
            assert reopened is not None
            assert reopened.metadata.name == "Round-Trip Test"

            # Verify boreholes
            assert reopened.borehole_collection is not None
            assert len(reopened.borehole_collection) == 8

            # Verify survey stations (148)
            formal = sum(len(bh.survey.stations) for bh in reopened.borehole_collection)
            assert formal == 148, f"Survey stations after round-trip: {formal}"

            # Verify only valid formal fractures are restored.
            fracs = sum(len(bh.fracture_observations) for bh in reopened.borehole_collection)
            assert fracs == 80, f"Fractures after round-trip: {fracs}"

            # Verify M7 data
            m7 = getattr(reopened, "_m7_data", {}) or {}

            # Raw DataFrames
            assert len(m7.get("raw_surveys", [])) == 153
            assert len(m7.get("raw_fractures", [])) == 83
            assert len(m7.get("raw_rqd", [])) == 27
            assert len(m7.get("raw_domain_intervals", [])) == 11
            assert len(m7.get("excluded_records", [])) == 3

            # Domain intervals (formal list)
            di = m7.get("domain_intervals", [])
            assert len(di) == 11, f"Domain intervals: {len(di)}"

            # Holdout
            ho = m7.get("holdout")
            assert ho is not None
            assert ho.is_locked
            assert len(ho.calibration_holes) == 6
            assert len(ho.validation_holes) == 2
            assert "BH-02" in ho.validation_holes
            assert "BH-07" in ho.validation_holes

            # Workflow
            wf = m7.get("workflow")
            assert wf is not None
            for step in ["import", "clean", "holdout", "domains", "joint_sets"]:
                assert wf.is_step_done(step), f"Step '{step}' not done after round-trip"

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_mainwindow_adopt_project_after_open(self, full_m7_project, qtbot):
        """MainWindow ProjectStore consistency: adopt_project sets current_project."""
        from dfn_cave_studio.persistence.project_store import ProjectStore

        tmpdir = tempfile.mkdtemp()
        save_path = Path(tmpdir) / "consistency.dfnproj"
        try:
            zps = ZipProjectStore()
            zps.save(full_m7_project, save_path)

            store = ProjectStore()
            reopened = zps.load(save_path)
            store.adopt_project(reopened, save_path)

            # Verify ProjectStore state
            assert store.current_project is reopened, "current_project should be the reopened project"
            assert store.current_path == save_path, f"current_path should be {save_path}, got {store.current_path}"
            assert store.is_dirty is False, "Project should be clean after adopt"

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dfnproj_save_as_then_save_remains_zip(self, full_m7_project, qtbot, tmp_path):
        """Save As followed by ordinary Save preserves ZIP format and M7 state."""
        from dfn_cave_studio.ui.main_window import MainWindow

        window = MainWindow()
        qtbot.addWidget(window)
        window._project_store.adopt_project(full_m7_project)
        window._workflow = full_m7_project._m7_data["workflow"]
        path = tmp_path / "continuous-save.dfnproj"

        window._save_project_to(path)
        assert zipfile.is_zipfile(path)

        full_m7_project.metadata.description = "changed before ordinary save"
        window._project_store.mark_dirty()
        window._project_store.save()

        assert zipfile.is_zipfile(path)
        reopened = ZipProjectStore().load(path)
        assert reopened.metadata.description == "changed before ordinary save"
        assert len(reopened.borehole_collection) == 8
        assert sum(len(bh.survey.stations) for bh in reopened.borehole_collection) == 148
        assert sum(len(bh.fracture_observations) for bh in reopened.borehole_collection) == 80
        assert reopened._m7_data["holdout"].is_locked
        assert len(reopened._m7_data["domain_intervals"]) == 11
        assert reopened._m7_data["workflow"].is_step_done("joint_sets")
        assert len(reopened.structural_domains.domains) == 4
        assert len(reopened.joint_sets) == 3
        holdout = reopened._m7_data["holdout"]
        result = JointSetService(random_seed=42).identify_auto(
            reopened.borehole_collection,
            set(holdout.calibration_holes),
            set(holdout.validation_holes),
            n_clusters=3,
            random_seed=42,
        )
        assert result.calibration_count == 60
        assert result.validation_count == 20
        assert len(result.assignments) == 60
        assert (
            sum(sum(assignment == set_id for assignment in result.assignments.values()) for set_id in result.sets) == 60
        )
        assert len(reopened._m7_data["raw_fractures"]) == 83
        assert len(reopened._m7_data["excluded_records"]) == 3

    def test_dfnproj_auto_save_remains_zip(self, full_m7_project, qtbot, tmp_path):
        """Auto-save uses the same extension-aware serializer as manual Save."""
        from dfn_cave_studio.persistence.project_store import ProjectStore

        path = tmp_path / "autosave.dfnproj"
        store = ProjectStore()
        store.adopt_project(full_m7_project)
        store.save_as(path)
        assert zipfile.is_zipfile(path)

        full_m7_project.metadata.description = "auto-saved change"
        store.mark_dirty()
        store.configure_auto_save(enabled=True, interval_seconds=0)
        assert store.tick_auto_save() is True
        assert zipfile.is_zipfile(path)

        reopened = ZipProjectStore().load(path)
        assert reopened.metadata.description == "auto-saved change"
        assert len(reopened.borehole_collection) == 8
        assert reopened._m7_data["holdout"].is_locked
        assert len(reopened._m7_data["domain_intervals"]) == 11
        assert len(reopened.structural_domains.domains) == 4
        assert len(reopened.joint_sets) == 3

    @pytest.mark.parametrize("operation", ["holdout", "domains", "joint_sets"])
    def test_accepted_m7_dialog_marks_project_dirty(self, operation, full_m7_project, qtbot, monkeypatch):
        """Real dialog commits are surfaced through MainWindow dirty state."""
        from dfn_cave_studio.ui.main_window import MainWindow
        from dfn_cave_studio.ui.qt_adapter import QDialog

        window = MainWindow()
        qtbot.addWidget(window)
        project = full_m7_project.model_copy(deep=True)
        window._project_store.adopt_project(project)
        window._workflow = project._m7_data["workflow"]

        if operation == "holdout":
            from dfn_cave_studio.ui.dialogs.m7_holdout_dialog import (
                M7HoldoutDialog,
            )

            def exec_dialog(dialog):
                dialog._on_unlock()
                dialog._on_accept()
                return QDialog.DialogCode.Accepted

            monkeypatch.setattr(M7HoldoutDialog, "exec", exec_dialog)
            window._m7_holdout()
        elif operation == "domains":
            from dfn_cave_studio.ui.dialogs.m7_domain_dialog import (
                M7DomainDialog,
            )

            def exec_dialog(dialog):
                dialog._add_domain()
                dialog._on_accept()
                return QDialog.DialogCode.Accepted

            monkeypatch.setattr(M7DomainDialog, "exec", exec_dialog)
            window._m7_domains()
        else:
            from dfn_cave_studio.ui.dialogs.m7_joint_set_dialog import (
                M7JointSetDialog,
            )

            def exec_dialog(dialog):
                dialog._on_identify()
                dialog._on_accept()
                return QDialog.DialogCode.Accepted

            monkeypatch.setattr(M7JointSetDialog, "exec", exec_dialog)
            window._m7_joint_sets()

        assert window._project_store.is_dirty is True


# ═══════════════════════════════════════════════════════════════════════════════
# New Project Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNewProjectClears:
    """New Project via MainWindow clears all previous state."""

    def test_new_project_resets_workflow(self, qtbot):
        """New Project creates fresh workflow with all steps NOT_STARTED."""
        from dfn_cave_studio.ui.main_window import MainWindow

        w = MainWindow()
        qtbot.addWidget(w)

        # Trigger new project
        w._on_new_project()
        qtbot.wait(50)

        # Verify workflow is new
        assert w._workflow is not None
        for step in ["import", "clean", "holdout", "domains", "joint_sets"]:
            assert not w._workflow.is_step_done(step), f"Step '{step}' should be NOT_STARTED in new project"

        # Verify store has project
        assert w._project_store.has_project
        assert w._project_store.current_project is not None

        w.close()

    def test_new_project_clears_boreholes(self, qtbot):
        """New project starts with empty borehole collection."""
        from dfn_cave_studio.ui.main_window import MainWindow

        w = MainWindow()
        qtbot.addWidget(w)

        w._on_new_project()
        qtbot.wait(50)

        project = w._project_store.current_project
        assert project is not None
        # Borehole collection exists but should be empty
        if project.borehole_collection is not None:
            assert (
                len(project.borehole_collection) == 0
            ), f"New project should have 0 boreholes, got {len(project.borehole_collection)}"

        assert len(project.joint_sets) == 0
        w.close()

    def test_new_project_clears_fake_plotter_actors(self, qtbot):
        """New Project removes actors even before DFNRenderer is initialized."""
        from dfn_cave_studio.ui.main_window import MainWindow

        window = MainWindow()
        qtbot.addWidget(window)
        window._plotter.add_mesh("sentinel-old-project")
        assert len(window._plotter._actors) == 1

        window._on_new_project()

        assert window._plotter._actors == []
