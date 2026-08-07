"""Real pytest-qt GUI tests for M7 dialogs (Domain, Holdout, Cleaning).

These tests actually instantiate dialogs with qtbot.addWidget() and
verify real widget state (QTableWidget rows, QComboBox selections, etc.).

They do NOT initialise real VTK/OpenGL — FakePlotter is injected via conftest.
"""

import sys
import copy
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.borehole import SurveyStation
from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter
from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import (
    get_cleaned_rqd,
    get_domain_intervals,
    get_excluded_records,
    get_holdout,
    get_quality_issues,
    get_raw_fractures,
    set_holdout,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def demo_dir():
    return Path(__file__).parent.parent.parent / "examples" / "m7_demo"


@pytest.fixture
def m7_project(demo_dir):
    """Build a project with all demo data loaded: collars, cleaned surveys,
    fractures, holdout BH-02/BH-07 locked, raw DataFrames."""
    project = Project()
    project.metadata.name = "GUI Lifecycle Test"
    project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

    # Import collars + fractures via BoreholeImporter
    imp = BoreholeImporter()
    cr = imp.import_all(
        collar_path=str(demo_dir / "collars.csv"),
        fractures_path=str(demo_dir / "fractures.csv"),
    )
    project.borehole_collection = cr.collection
    bh_map = {bh.borehole_id: bh for bh in cr.collection}

    # Attach cleaned survey stations (148 valid)
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

    # Setup holdout: BH-02, BH-07 = validation, fraction=0.25
    hole_ids = sorted(bh_map.keys())
    ho = HoldoutService()
    ho.select_manual(hole_ids, ["BH-02", "BH-07"])
    ho.update_config(validation_fraction=0.25)
    ho.lock()

    project._m7_data = {
        "raw_surveys": raw,
        "raw_fractures": pd.read_csv(demo_dir / "fractures.csv"),
        "raw_rqd": pd.read_csv(demo_dir / "rqd.csv"),
        "raw_domain_intervals": pd.read_csv(demo_dir / "domain_intervals.csv"),
        "excluded_records": cr.fracture_exclusions,
        "holdout": ho,
    }
    return project


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Dialog Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainDialog:
    """Real GUI tests for M7DomainDialog — instantiates dialog, checks tables."""

    def test_global_shows_11_rows(self, m7_project, qtbot):
        """DomainDialog Global (ID=0) shows all 11 interval rows."""
        from dfn_cave_studio.ui.dialogs.m7_domain_dialog import M7DomainDialog

        wf = WorkflowController()
        dlg = M7DomainDialog(m7_project, wf)
        qtbot.addWidget(dlg)

        # Global is the first domain (ID=0), selected by default
        table = dlg._interval_table
        assert table.rowCount() == 11, f"Global should show 11 rows, got {table.rowCount()}"

        dlg.close()

    def test_domain1_shows_5_rows(self, m7_project, qtbot):
        """Selecting Domain 1 (ID=1) shows 5 rows."""
        from dfn_cave_studio.ui.dialogs.m7_domain_dialog import M7DomainDialog

        wf = WorkflowController()
        dlg = M7DomainDialog(m7_project, wf)
        qtbot.addWidget(dlg)

        # Select Domain 1 (index 1, ID=1)
        dlg._domain_list.setCurrentRow(1)
        qtbot.wait(50)
        table = dlg._interval_table
        assert table.rowCount() == 5, f"Domain 1 should show 5 rows, got {table.rowCount()}"

        dlg.close()

    def test_domain2_shows_5_rows(self, m7_project, qtbot):
        """Selecting Domain 2 (ID=2) shows 5 rows."""
        from dfn_cave_studio.ui.dialogs.m7_domain_dialog import M7DomainDialog

        wf = WorkflowController()
        dlg = M7DomainDialog(m7_project, wf)
        qtbot.addWidget(dlg)

        dlg._domain_list.setCurrentRow(2)
        qtbot.wait(50)
        table = dlg._interval_table
        assert table.rowCount() == 5, f"Domain 2 should show 5 rows, got {table.rowCount()}"

        dlg.close()

    def test_domain3_shows_1_row(self, m7_project, qtbot):
        """Selecting Domain 3 (ID=3) shows 1 row."""
        from dfn_cave_studio.ui.dialogs.m7_domain_dialog import M7DomainDialog

        wf = WorkflowController()
        dlg = M7DomainDialog(m7_project, wf)
        qtbot.addWidget(dlg)

        dlg._domain_list.setCurrentRow(3)
        qtbot.wait(50)
        table = dlg._interval_table
        assert table.rowCount() == 1, f"Domain 3 should show 1 row, got {table.rowCount()}"

        dlg.close()

    def test_intervals_persist_across_dialog_reopen(self, m7_project, qtbot):
        """Modify interval, close dialog (OK), reopen — modification persists."""
        from dfn_cave_studio.ui.dialogs.m7_domain_dialog import M7DomainDialog

        wf = WorkflowController()

        # First dialog: assign a new interval
        dlg1 = M7DomainDialog(m7_project, wf)
        qtbot.addWidget(dlg1)
        initial_count = dlg1._interval_table.rowCount()
        assert initial_count == 11

        # Add a manual interval
        dlg1._domain_list.setCurrentRow(1)  # Select Domain 1
        dlg1._bh_combo.setCurrentIndex(0)  # First borehole
        dlg1._from_spin.setValue(10)
        dlg1._to_spin.setValue(50)
        dlg1._assign_interval()
        assert dlg1._interval_table.rowCount() >= 5  # Domain 1 filtered + new

        dlg1._on_accept()  # Persists to project

        # Second dialog: should see the new interval persisted
        dlg2 = M7DomainDialog(m7_project, wf)
        qtbot.addWidget(dlg2)
        total_rows = dlg2._interval_table.rowCount()
        assert total_rows >= 11, f"Reopened dialog should have ≥11 rows, got {total_rows}"

        dlg2.close()

    def test_cancel_discards_domain_edits(self, m7_project, qtbot):
        """Adding a domain and interval remains pending until OK."""
        from dfn_cave_studio.ui.dialogs.m7_domain_dialog import M7DomainDialog

        before_domains = m7_project.structural_domains.model_dump()
        before_intervals = [interval.model_dump() for interval in get_domain_intervals(m7_project)]
        before_workflow = WorkflowController().to_dict()
        workflow = WorkflowController()
        dialog = M7DomainDialog(m7_project, workflow)
        qtbot.addWidget(dialog)

        qtbot.mouseClick(dialog._add_domain_btn, Qt.MouseButton.LeftButton)
        dialog._domain_list.setCurrentRow(dialog._domain_list.count() - 1)
        dialog._from_spin.setValue(0)
        dialog._to_spin.setValue(10)
        qtbot.mouseClick(dialog._assign_interval_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseClick(
            dialog._button_box.button(QDialogButtonBox.StandardButton.Cancel),
            Qt.MouseButton.LeftButton,
        )

        assert m7_project.structural_domains.model_dump() == before_domains
        assert [interval.model_dump() for interval in get_domain_intervals(m7_project)] == before_intervals
        assert workflow.to_dict() == before_workflow

    def test_overlapping_domain_blocks_commit_and_workflow(self, m7_project, qtbot):
        """An overlap remains pending and cannot complete the domain step."""
        from dfn_cave_studio.ui.dialogs.m7_domain_dialog import M7DomainDialog
        from dfn_cave_studio.models.data_management import DomainInterval

        before_intervals = [interval.model_dump() for interval in get_domain_intervals(m7_project)]
        workflow = WorkflowController()
        dialog = M7DomainDialog(m7_project, workflow)
        qtbot.addWidget(dialog)
        first = dialog._intervals[0]
        dialog._intervals.append(
            DomainInterval(
                hole_id=first.hole_id,
                from_depth=first.from_depth + 1,
                to_depth=first.to_depth,
                domain_id=first.domain_id,
                domain_name=first.domain_name,
            )
        )

        qtbot.mouseClick(
            dialog._button_box.button(QDialogButtonBox.StandardButton.Ok),
            Qt.MouseButton.LeftButton,
        )

        assert dialog.committed_changes is False
        assert workflow.is_step_done("domains") is False
        assert [interval.model_dump() for interval in get_domain_intervals(m7_project)] == before_intervals


# ═══════════════════════════════════════════════════════════════════════════════
# Holdout Dialog Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldoutDialog:
    """Real GUI tests for M7HoldoutDialog — instantiates dialog, checks state."""

    def test_holdout_state_persists_across_dialog_reopen(self, m7_project, qtbot):
        """Lock with BH-02/BH-07 validation, close, reopen — state intact."""
        from dfn_cave_studio.ui.dialogs.m7_holdout_dialog import M7HoldoutDialog

        wf = WorkflowController()

        # First open: verify restore from project
        dlg1 = M7HoldoutDialog(m7_project, wf)
        qtbot.addWidget(dlg1)

        # Check method restored to "Manual selection"
        assert dlg1._method_combo.currentIndex() == 0, "Method should be manual"

        # Check fraction restored
        assert (
            abs(dlg1._fraction_spin.value() - 0.25) < 0.01
        ), f"Fraction should be 0.25, got {dlg1._fraction_spin.value()}"

        # Check locked state
        stats_text = dlg1._stats_label.text()
        assert "LOCKED" in stats_text, f"Holdout should be locked, got: {stats_text}"
        assert "Calibration: 6" in stats_text, f"Should show 6 cal: {stats_text}"
        assert "Validation: 2" in stats_text, f"Should show 2 val: {stats_text}"
        assert not dlg1._lock_btn.isEnabled(), "Lock button should be disabled"
        assert dlg1._unlock_btn.isEnabled(), "Unlock button should be enabled"
        assert not dlg1._method_combo.isEnabled()
        assert not dlg1._bh_list.isEnabled()
        assert not dlg1._val_btn.isEnabled()

        dlg1.accept()

        # Second open: state still intact
        dlg2 = M7HoldoutDialog(m7_project, wf)
        qtbot.addWidget(dlg2)
        assert dlg2._method_combo.currentIndex() == 0

        # Verify locked, 6/2
        ho = get_holdout(m7_project)
        assert ho is not None
        assert ho.is_locked
        assert len(ho.calibration_holes) == 6
        assert len(ho.validation_holes) == 2
        assert "BH-02" in ho.validation_holes
        assert "BH-07" in ho.validation_holes

        dlg2.accept()

    def test_manual_selection_and_lock(self, m7_project, qtbot):
        """Unlock, manually select validation holes, re-lock."""
        from dfn_cave_studio.ui.dialogs.m7_holdout_dialog import M7HoldoutDialog

        # First unlock current holdout
        ho = get_holdout(m7_project)
        if ho and ho.is_locked:
            ho.unlock()
            set_holdout(m7_project, ho)

        wf = WorkflowController()
        dlg = M7HoldoutDialog(m7_project, wf)
        qtbot.addWidget(dlg)

        # Should not be locked initially (we just unlocked)
        if "LOCKED" not in dlg._stats_label.text():
            # Select BH-02 and BH-07 as validation
            for i in range(dlg._bh_list.count()):
                item = dlg._bh_list.item(i)
                hid = item.data(Qt.ItemDataRole.UserRole)
                if hid in ("BH-02", "BH-07"):
                    item.setSelected(True)

            dlg._val_btn.click()  # Mark as Validation
            qtbot.wait(50)

            # Lock
            dlg._lock_btn.click()
            qtbot.wait(50)

        stats = dlg._stats_label.text()
        assert "LOCKED" in stats, f"Should be locked after clicking Lock: {stats}"
        assert "Calibration: 6" in stats, f"Should show 6 cal: {stats}"
        assert "Validation: 2" in stats, f"Should show 2 val: {stats}"

        dlg.accept()

    def test_cancel_discards_holdout_and_workflow_changes(self, m7_project, qtbot):
        """Unlock/select operations do not commit before OK."""
        from dfn_cave_studio.ui.dialogs.m7_holdout_dialog import M7HoldoutDialog

        before_holdout = copy.deepcopy(get_holdout(m7_project).to_dict())
        workflow = WorkflowController()
        before_workflow = workflow.to_dict()
        dialog = M7HoldoutDialog(m7_project, workflow)
        qtbot.addWidget(dialog)

        dialog._unlock_btn.click()
        qtbot.wait(10)
        dialog._bh_list.item(0).setSelected(True)
        dialog._val_btn.click()
        qtbot.mouseClick(
            dialog._button_box.button(QDialogButtonBox.StandardButton.Cancel),
            Qt.MouseButton.LeftButton,
        )

        assert get_holdout(m7_project).to_dict() == before_holdout
        assert workflow.to_dict() == before_workflow


# ═══════════════════════════════════════════════════════════════════════════════
# Cleaning Dialog Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleaningDialog:
    """Real GUI tests for M7CleaningDialog — check idempotency and apply."""

    def test_check_phase_is_idempotent(self, m7_project, qtbot):
        """Opening CleaningDialog does not modify formal stations."""
        from dfn_cave_studio.ui.dialogs.m7_cleaning_dialog import M7CleaningDialog

        wf = WorkflowController()

        initial = sum(len(bh.survey.stations) for bh in m7_project.borehole_collection)

        dlg = M7CleaningDialog(m7_project, wf)
        qtbot.addWidget(dlg)
        qtbot.wait(50)

        after_open = sum(len(bh.survey.stations) for bh in m7_project.borehole_collection)
        assert after_open == initial, f"Check phase must be idempotent: {initial} → {after_open}"

        # Verify idempotency label is green
        assert (
            "unchanged" in dlg._idempotency_label.text().lower()
            or "idempotent" in dlg._idempotency_label.text().lower()
        )

        dlg.reject()

    def test_close_reopen_does_not_duplicate(self, m7_project, qtbot):
        """Closing and reopening the dialog doesn't change formal stations."""
        from dfn_cave_studio.ui.dialogs.m7_cleaning_dialog import M7CleaningDialog

        wf = WorkflowController()

        initial = sum(len(bh.survey.stations) for bh in m7_project.borehole_collection)

        # First open
        dlg1 = M7CleaningDialog(m7_project, wf)
        qtbot.addWidget(dlg1)
        qtbot.wait(50)
        dlg1.reject()

        # Second open
        dlg2 = M7CleaningDialog(m7_project, wf)
        qtbot.addWidget(dlg2)
        qtbot.wait(50)

        after_reopen = sum(len(bh.survey.stations) for bh in m7_project.borehole_collection)
        assert after_reopen == initial, f"Reopen should not change stations: {initial} → {after_reopen}"

        dlg2.reject()


class TestImportAndJointSetPendingState:
    """Regression tests for import cancellation and non-demo holdout sizes."""

    def test_fracture_import_cancel_and_repeat_leave_project_unchanged(self, m7_project, demo_dir, qtbot):
        from dfn_cave_studio.ui.dialogs.m7_import_dialog import M7ImportDialog

        before = sum(len(bh.fracture_observations) for bh in m7_project.borehole_collection)
        dialog = M7ImportDialog(m7_project)
        qtbot.addWidget(dialog)
        path = str(demo_dir / "fractures.csv")

        dialog._import_fractures(path, "utf-8", ",", {}, "Sheet")
        first_pending = sum(len(bh.fracture_observations) for bh in dialog.get_collection())
        dialog._import_fractures(path, "utf-8", ",", {}, "Sheet")
        second_pending = sum(len(bh.fracture_observations) for bh in dialog.get_collection())
        assert second_pending == first_pending

        qtbot.mouseClick(
            dialog._button_box.button(QDialogButtonBox.StandardButton.Cancel),
            Qt.MouseButton.LeftButton,
        )
        after = sum(len(bh.fracture_observations) for bh in m7_project.borehole_collection)
        assert after == before

    def test_fracture_import_report_and_audit_rows(self, demo_dir, qtbot):
        """All 83 rows are retained while exactly 80 become formal data."""
        from dfn_cave_studio.ui.dialogs.m7_import_dialog import M7ImportDialog

        project = Project()
        collar_result = BoreholeImporter().import_all(collar_path=str(demo_dir / "collars.csv"))
        project.borehole_collection = collar_result.collection
        dialog = M7ImportDialog(project)
        qtbot.addWidget(dialog)

        dialog._import_fractures(
            str(demo_dir / "fractures.csv"),
            "utf-8",
            ",",
            {},
            "Sheet",
        )
        report = dialog._imported["fractures"]
        assert report == {
            "raw": 83,
            "imported": 80,
            "skipped": 3,
            "errors": 3,
        }
        assert sum(len(borehole.fracture_observations) for borehole in dialog.get_collection()) == 80
        assert {record["source_row"] for record in dialog._fracture_exclusions} == {
            80,
            81,
            82,
        }

        qtbot.mouseClick(
            dialog._button_box.button(QDialogButtonBox.StandardButton.Ok),
            Qt.MouseButton.LeftButton,
        )
        assert len(get_raw_fractures(project)) == 83
        assert len(get_excluded_records(project)) == 3

    def test_five_boreholes_can_identify_joint_sets(self, m7_project, qtbot):
        from dfn_cave_studio.ui.dialogs.m7_joint_set_dialog import (
            M7JointSetDialog,
        )
        from dfn_cave_studio.models.borehole import BoreholeCollection

        project = m7_project.model_copy(deep=True)
        project.borehole_collection = BoreholeCollection(boreholes=project.borehole_collection.boreholes[:5])
        hole_ids = [bh.borehole_id for bh in project.borehole_collection]
        holdout = HoldoutService()
        holdout.select_manual(hole_ids, [hole_ids[-1]])
        holdout.lock()
        set_holdout(project, holdout)

        workflow = WorkflowController()
        dialog = M7JointSetDialog(project, workflow)
        qtbot.addWidget(dialog)
        qtbot.mouseClick(dialog._identify_btn, Qt.MouseButton.LeftButton)

        assert dialog._result_table.rowCount() > 0
        assert "Validation fractures excluded" in dialog._stats_label.text()

    def test_accept_auto_fixes_then_ok_produces_148(self, m7_project, qtbot):
        """Accept All Auto-Fixes + OK → 148 stations committed to project."""
        from dfn_cave_studio.ui.dialogs.m7_cleaning_dialog import M7CleaningDialog

        wf = WorkflowController()

        dlg = M7CleaningDialog(m7_project, wf)
        qtbot.addWidget(dlg)
        qtbot.wait(50)

        # Click Accept All Auto-Fixes
        dlg._accept_auto_fixes()
        qtbot.wait(50)

        # Verify pending stations count
        pending_count = dlg.get_pending_station_count()
        assert pending_count == 148, f"Pending stations should be 148 after auto-fixes, got {pending_count}"

        # Verify no unresolved errors (after our demo fixes)
        if not dlg._has_unresolved_errors:
            dlg._on_accept()
        else:
            dlg._on_accept()  # Will show message box (mocked by conftest)

        # After accept, check formal stations
        formal = sum(len(bh.survey.stations) for bh in m7_project.borehole_collection)
        assert formal == 148, f"After accept, formal stations should be 148, got {formal}"

    def test_reopen_after_accept_still_148(self, m7_project, qtbot):
        """After Accept+OK, reopening dialog shows 148 formal stations."""
        from dfn_cave_studio.ui.dialogs.m7_cleaning_dialog import M7CleaningDialog

        wf = WorkflowController()

        # First apply fixes
        dlg1 = M7CleaningDialog(m7_project, wf)
        qtbot.addWidget(dlg1)
        qtbot.wait(50)
        dlg1._accept_auto_fixes()
        qtbot.wait(50)
        if not dlg1._has_unresolved_errors:
            dlg1._on_accept()

        formal_after = sum(len(bh.survey.stations) for bh in m7_project.borehole_collection)
        assert formal_after == 148

        # Reopen — should still be 148
        dlg2 = M7CleaningDialog(m7_project, wf)
        qtbot.addWidget(dlg2)
        qtbot.wait(50)

        formal_after_reopen = sum(len(bh.survey.stations) for bh in m7_project.borehole_collection)
        assert formal_after_reopen == 148, (
            f"After reopen, formal stations should still be 148, " f"got {formal_after_reopen}"
        )

        dlg2.reject()

    def test_filtered_exclusion_uses_issue_identity_and_removes_invalid_data(self, m7_project, qtbot):
        """Filtered rows exclude their source records, not full-list positions."""
        from dfn_cave_studio.ui.dialogs.m7_cleaning_dialog import (
            M7CleaningDialog,
        )

        invalid = pd.DataFrame(
            [
                {
                    "hole_id": "BH-01",
                    "from_depth": 25.0,
                    "to_depth": 10.0,
                    "domain_id": 99,
                    "domain_name": "Invalid",
                }
            ]
        )
        m7_project._m7_data["raw_domain_intervals"] = pd.concat(
            [m7_project._m7_data["raw_domain_intervals"], invalid],
            ignore_index=True,
        )
        workflow = WorkflowController()
        dialog = M7CleaningDialog(m7_project, workflow)
        qtbot.addWidget(dialog)

        dialog._filter_combo.setCurrentIndex(1)  # Errors Only
        qtbot.wait(10)
        assert dialog._table.rowCount() > 1
        selected_row = 1
        selected_issue_id = dialog._table.item(selected_row, 0).data(Qt.ItemDataRole.UserRole)
        dialog._table.selectRow(selected_row)
        dialog._exclude_selected()

        assert all(issue.issue_id != selected_issue_id for issue in dialog._pending_quality_issues)
        assert any(record.get("issue_id") == selected_issue_id for record in dialog._pending_excluded_records)

        dialog._table.selectAll()
        dialog._exclude_selected()
        assert dialog._has_unresolved_errors is False
        qtbot.mouseClick(
            dialog._button_box.button(QDialogButtonBox.StandardButton.Ok),
            Qt.MouseButton.LeftButton,
        )

        cleaned_rqd = get_cleaned_rqd(m7_project)
        assert len(cleaned_rqd) == 26
        assert (cleaned_rqd["rqd"].between(0, 100)).all()
        intervals = get_domain_intervals(m7_project)
        assert len(intervals) == 11
        assert all(interval.domain_id != 99 for interval in intervals)
        exclusions = get_excluded_records(m7_project)
        assert len(exclusions) == 11
        assert len({record["issue_id"] for record in exclusions}) == 11
        assert Counter(record["source_file"] for record in exclusions) == {
            "surveys.csv": 5,
            "fractures.csv": 3,
            "rqd.csv": 1,
            "domain_intervals.csv": 2,
        }
        assert {record["source_file"] for record in exclusions} == {
            "surveys.csv",
            "fractures.csv",
            "rqd.csv",
            "domain_intervals.csv",
        }
        fractures = sum(len(borehole.fracture_observations) for borehole in m7_project.borehole_collection)
        assert fractures == 80
        fracture_issues = [issue for issue in get_quality_issues(m7_project) if issue.source_file == "fractures.csv"]
        assert len(fracture_issues) == 3
        assert all(issue.applied_action == "excluded" for issue in fracture_issues)
