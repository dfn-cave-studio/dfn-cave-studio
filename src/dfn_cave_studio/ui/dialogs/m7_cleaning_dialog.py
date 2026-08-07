"""M7 data cleaning dialog — separates check (idempotent) from apply (commits).

Check phase:
  - Reads raw data, validates, produces pending stations/issues.
  - NEVER modifies project.borehole_collection or bh.survey.stations.

Apply phase (Accept All Auto-Fixes):
  - Re-processes raw surveys with normalization (azimuth % 360, dip clamp).
  - Updates pending state only — still does NOT touch the project.

Commit phase (OK):
  - Refuses if unresolved ERRORs remain.
  - Writes pending stations to bh.survey.stations (replaces, not appends).
  - Persists quality_issues, domain_intervals, excluded_records to project.

Idempotency:
  - Opening, closing, reopening the dialog never changes formal station count.
  - Accept Auto-Fixes → 148 stations. Reopen → still 148. Recheck → still 148.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from dfn_cave_studio.ui.qt_adapter import (
    Qt,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDialogButtonBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
)
from dfn_cave_studio.services.cleaning_service import (
    DataCleaningService,
    IssueSeverity,
    IssueType,
)
from dfn_cave_studio.models.data_management import DataQualityIssue, DomainInterval
from dfn_cave_studio.services.m7_state import (
    get_raw_fractures,
    get_raw_surveys,
    get_raw_rqd,
    get_raw_domain_intervals,
    get_quality_issues,
    set_quality_issues,
    get_domain_intervals,
    set_domain_intervals,
    get_cleaned_rqd,
    set_cleaned_rqd,
    get_excluded_records,
    set_excluded_records,
)

_logger = logging.getLogger(__name__)


class M7CleaningDialog(QDialog):
    """Data cleaning dialog with separated check/apply/commit phases.

    Check = idempotent read-only validation → populates pending state.
    Apply = auto-fix normalization on pending state only.
    Commit = write pending to project (OK button, only if no ERRORs).
    """

    def __init__(self, project, workflow, parent=None):
        super().__init__(parent)
        self._project = project
        self._workflow = workflow

        # ── Pending state (NOT in project until OK) ─────────────────────
        self._pending_survey_stations: Dict[str, list] = {}  # hole_id → [SurveyStation]
        self._pending_quality_issues: List[DataQualityIssue] = []
        self._resolved_quality_issues: List[DataQualityIssue] = []
        self._pending_excluded_records: List[dict] = []
        self._pending_domain_intervals: List[DomainInterval] = []
        self._pending_domain_sources: Dict[int, str] = {}
        self._pending_rqd = pd.DataFrame()

        # Stats for display
        self._stats = {
            "raw_surveys": 0,
            "raw_fractures": 0,
            "cleaned": 0,
            "attached": 0,
            "excluded": 0,
            "raw_rqd": 0,
            "raw_domain_intervals": 0,
            "n_errors": 0,
            "n_warnings": 0,
            "n_info": 0,
            "collars_ok": 0,
            "collars_errors": 0,
            "fractures_ok": 0,
            "fractures_errors": 0,
        }

        # Station count BEFORE dialog opened (for idempotency verification)
        self._initial_station_count = self._count_formal_stations()
        self._initial_commit_signature = self._commit_signature()
        self._committed_changes = False

        self.setWindowTitle("Data Cleaning & Quality Check")
        self.resize(950, 620)
        self._init_ui()
        self._run_checks()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _count_formal_stations(self) -> int:
        """Count SurveyStation objects currently in project boreholes."""
        if self._project.borehole_collection is None:
            return 0
        return sum(len(bh.survey.stations) for bh in self._project.borehole_collection)

    def _commit_signature(self) -> str:
        """Return a stable signature of project state owned by cleaning."""
        stations = []
        for borehole in self._project.borehole_collection or []:
            stations.append(
                (
                    borehole.borehole_id,
                    [station.model_dump(mode="json") for station in borehole.survey.stations],
                )
            )
        cleaned_rqd = get_cleaned_rqd(self._project)
        payload = {
            "stations": stations,
            "quality_issues": [
                issue.model_dump(mode="json") if hasattr(issue, "model_dump") else issue
                for issue in get_quality_issues(self._project)
            ],
            "domain_intervals": [interval.model_dump(mode="json") for interval in get_domain_intervals(self._project)],
            "cleaned_rqd": (
                cleaned_rqd.to_json(orient="split", date_format="iso")
                if isinstance(cleaned_rqd, pd.DataFrame)
                else None
            ),
            "excluded_records": get_excluded_records(self._project),
            "workflow": self._workflow.to_dict(),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @property
    def _has_unresolved_errors(self) -> bool:
        return any(i.severity == IssueSeverity.ERROR for i in self._pending_quality_issues)

    # ── UI ────────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Top: Run checks + filter
        top = QHBoxLayout()
        self._run_btn = QPushButton("Re-run All Checks")
        self._run_btn.clicked.connect(self._on_re_run_checks)
        top.addWidget(self._run_btn)

        self._filter_combo = QComboBox()
        self._filter_combo.addItem("All Issues", None)
        self._filter_combo.addItem("Errors Only", "error")
        self._filter_combo.addItem("Warnings Only", "warning")
        self._filter_combo.addItem("Info Only", "info")
        self._filter_combo.currentIndexChanged.connect(self._refresh_table)
        top.addWidget(QLabel("Filter:"))
        top.addWidget(self._filter_combo)

        self._stats_label = QLabel("")
        top.addWidget(self._stats_label)
        top.addStretch()
        layout.addLayout(top)

        # Idempotency indicator
        self._idempotency_label = QLabel("")
        layout.addWidget(self._idempotency_label)

        # Issues table
        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels(
            [
                "Type",
                "Hole ID",
                "Row",
                "Field",
                "Original Value",
                "Severity",
                "Message",
                "Action",
            ]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        # Buttons row
        actions = QHBoxLayout()
        self._exclude_btn = QPushButton("Exclude Selected Rows")
        self._exclude_btn.clicked.connect(self._exclude_selected)
        actions.addWidget(self._exclude_btn)

        self._accept_fixes_btn = QPushButton("Accept All Auto-Fixes")
        self._accept_fixes_btn.clicked.connect(self._accept_auto_fixes)
        actions.addWidget(self._accept_fixes_btn)

        export_btn = QPushButton("Export Report JSON")
        export_btn.clicked.connect(self._export_report)
        actions.addWidget(export_btn)
        actions.addStretch()
        layout.addLayout(actions)

        # Bottom row
        btn_row = QHBoxLayout()
        self._error_warning = QLabel("")
        btn_row.addWidget(self._error_warning)
        btn_row.addStretch()
        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        btn_row.addWidget(self._button_box)
        layout.addLayout(btn_row)

    # ── Check phase (idempotent, never modifies project) ─────────────────

    def _on_re_run_checks(self):
        """Re-run checks.  Idempotent — does not modify project stations."""
        self._run_checks()

    def _run_checks(self):
        """Run all data quality checks on raw imported data.

        IDEMPOTENT: reads raw DataFrames and project collection,
        populates pending state.  Does NOT write to bh.survey.stations.
        """
        self._pending_quality_issues.clear()
        self._resolved_quality_issues.clear()
        self._pending_survey_stations.clear()
        self._pending_excluded_records = [dict(record) for record in get_excluded_records(self._project)]
        self._pending_domain_intervals.clear()
        self._pending_domain_sources.clear()
        self._pending_rqd = pd.DataFrame()
        self._stats = {k: 0 for k in self._stats}

        coll = self._project.borehole_collection
        if coll is None:
            self._stats_label.setText("No borehole data loaded.")
            self._refresh_table()
            return

        # Build collars DataFrame from collection (read-only)
        bh_map = {bh.borehole_id: bh for bh in coll}
        max_depths = {bh.borehole_id: bh.collar.final_depth for bh in coll}
        collars_rows = []
        for bh in coll:
            collars_rows.append(
                {
                    "hole_id": bh.borehole_id,
                    "borehole_id": bh.borehole_id,
                    "easting": bh.collar.collar_x,
                    "northing": bh.collar.collar_y,
                    "elevation": bh.collar.collar_z,
                    "total_depth": bh.collar.final_depth,
                    "azimuth": bh.collar.azimuth,
                    "dip": bh.collar.dip,
                }
            )
        collars_df = pd.DataFrame(collars_rows)

        # ── Validate collars ────────────────────────────────────────────
        if not collars_df.empty:
            svc = DataCleaningService()
            collar_issues = svc.validate_collars(collars_df, "collars.csv")
            self._pending_quality_issues.extend(collar_issues)
            self._stats["collars_errors"] = len([i for i in collar_issues if i.severity == IssueSeverity.ERROR])
            self._stats["collars_ok"] = len(coll) - self._stats["collars_errors"]

        # ── Surveys: check only, build pending stations ─────────────────
        raw_surveys = get_raw_surveys(self._project)
        if raw_surveys is not None and not raw_surveys.empty:
            self._stats["raw_surveys"] = len(raw_surveys)
            self._check_surveys(raw_surveys, bh_map, max_depths)

        # ── Fractures: check only (from collection) ─────────────────────
        raw_fractures = get_raw_fractures(self._project)
        if raw_fractures is not None and not raw_fractures.empty:
            fractures_df = raw_fractures
        else:
            fractures_df = pd.DataFrame(
                [
                    {
                        "hole_id": bh.borehole_id,
                        "depth": obs.measured_depth,
                        "measured_depth": obs.measured_depth,
                        "dip_direction": obs.dip_direction,
                        "dip": obs.dip,
                        "set_id": obs.set_id,
                    }
                    for bh in coll
                    for obs in bh.fracture_observations
                ]
            )
        self._stats["raw_fractures"] = len(fractures_df)
        if not fractures_df.empty:
            if not fractures_df.empty and not collars_df.empty:
                svc2 = DataCleaningService()
                frac_issues = svc2.validate_fractures(fractures_df, collars_df, "fractures.csv")
                self._pending_quality_issues.extend(frac_issues)
                self._stats["fractures_ok"] = len(fractures_df) - len(
                    [i for i in frac_issues if i.severity == IssueSeverity.ERROR]
                )
                self._stats["fractures_errors"] = len([i for i in frac_issues if i.severity == IssueSeverity.ERROR])

        # ── RQD: validate from raw DataFrame ────────────────────────────
        raw_rqd = get_raw_rqd(self._project)
        if raw_rqd is not None and not raw_rqd.empty:
            self._stats["raw_rqd"] = len(raw_rqd)
            svc3 = DataCleaningService()
            rqd_issues = svc3.validate_rqd(raw_rqd, collars_df, "rqd.csv")
            self._pending_quality_issues.extend(rqd_issues)
            invalid_rows = {issue.source_row for issue in rqd_issues if issue.severity == IssueSeverity.ERROR}
            cleaned_rqd = raw_rqd.drop(index=[idx for idx in invalid_rows if idx in raw_rqd.index]).copy()
            if "rqd" in cleaned_rqd.columns:
                cleaned_rqd["rqd"] = pd.to_numeric(cleaned_rqd["rqd"], errors="coerce").clip(0.0, 100.0)
                cleaned_rqd = cleaned_rqd.dropna(subset=["rqd"])
            self._pending_rqd = cleaned_rqd

        # ── Domain intervals: validate from raw DataFrame ────────────────
        raw_di = get_raw_domain_intervals(self._project)
        if raw_di is not None and not raw_di.empty:
            self._stats["raw_domain_intervals"] = len(raw_di)
            svc4 = DataCleaningService()
            di_issues = svc4.validate_domain_intervals(raw_di, collars_df, "domain_intervals.csv")
            self._pending_quality_issues.extend(di_issues)
            invalid_rows = {issue.source_row for issue in di_issues if issue.severity == IssueSeverity.ERROR}
            for idx, row in raw_di.iterrows():
                if idx in invalid_rows:
                    continue
                try:
                    interval = DomainInterval(
                        hole_id=str(row["hole_id"]).strip(),
                        from_depth=float(row["from_depth"]),
                        to_depth=float(row["to_depth"]),
                        domain_id=int(row["domain_id"]),
                        domain_name=str(row.get("domain_name", "")),
                        assignment_method="imported",
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                self._pending_domain_intervals.append(interval)
                self._pending_domain_sources[int(idx)] = interval.interval_id

        # ── Count severities ─────────────────────────────────────────────
        excluded_source_rows = {
            (record.get("source_file"), record.get("source_row")) for record in self._pending_excluded_records
        }
        unresolved_issues = []
        for issue in self._pending_quality_issues:
            if (issue.source_file, issue.source_row) in excluded_source_rows:
                issue.applied_action = "excluded"
                self._resolved_quality_issues.append(issue)
            else:
                unresolved_issues.append(issue)
        self._pending_quality_issues = unresolved_issues

        self._stats["n_errors"] = len([i for i in self._pending_quality_issues if i.severity == IssueSeverity.ERROR])
        self._stats["n_warnings"] = len(
            [i for i in self._pending_quality_issues if i.severity == IssueSeverity.WARNING]
        )
        self._stats["n_info"] = len([i for i in self._pending_quality_issues if i.severity == IssueSeverity.INFO])

        self._refresh_table()
        self._update_idempotency_label()

    def _check_surveys(self, df, bh_map, max_depths):
        """Build pending survey stations from raw DataFrame.  IDEMPOTENT.

        Does NOT modify bh.survey.stations.  Populates:
          - self._pending_survey_stations[hole_id] = list of SurveyStation
          - self._pending_quality_issues with survey-related issues
          - self._pending_excluded_records
        """
        from dfn_cave_studio.models.borehole import SurveyStation

        for idx, row in df.iterrows():
            bh_id = str(row.get("hole_id", "")).strip()

            if bh_id not in bh_map:
                self._pending_quality_issues.append(
                    DataQualityIssue(
                        source_file="surveys.csv",
                        source_row=int(idx),
                        hole_id=bh_id,
                        field="hole_id",
                        issue_type=IssueType.UNKNOWN_REFERENCE,
                        severity=IssueSeverity.ERROR,
                        original_value=bh_id,
                        message=f"Borehole '{bh_id}' not in collars",
                        suggested_action="Exclude row",
                    )
                )
                self._stats["excluded"] += 1
                continue

            # Parse depth
            try:
                md = float(row["measured_depth"])
            except (ValueError, TypeError):
                self._pending_quality_issues.append(
                    DataQualityIssue(
                        source_file="surveys.csv",
                        source_row=int(idx),
                        hole_id=bh_id,
                        field="measured_depth",
                        issue_type=IssueType.INVALID_TYPE,
                        severity=IssueSeverity.ERROR,
                        original_value=str(row.get("measured_depth", "")),
                        message="Non-numeric measured_depth",
                        suggested_action="Exclude row",
                    )
                )
                self._stats["excluded"] += 1
                continue

            # Check depth exceed
            max_d = max_depths.get(bh_id, float("inf"))
            if md > max_d:
                self._pending_quality_issues.append(
                    DataQualityIssue(
                        source_file="surveys.csv",
                        source_row=int(idx),
                        hole_id=bh_id,
                        field="measured_depth",
                        issue_type=IssueType.DEPTH_EXCEEDS,
                        severity=IssueSeverity.ERROR,
                        original_value=str(md),
                        message=f"Depth {md} exceeds total depth {max_d}",
                        suggested_action="Exclude row",
                    )
                )
                self._stats["excluded"] += 1
                continue

            # Parse azimuth — normalize % 360
            raw_az = row.get("azimuth", 0)
            try:
                az = float(raw_az)
            except (ValueError, TypeError):
                az = 0.0
            az_original = az
            az = az % 360
            if abs(az - az_original) > 0.01:
                self._pending_quality_issues.append(
                    DataQualityIssue(
                        source_file="surveys.csv",
                        source_row=int(idx),
                        hole_id=bh_id,
                        field="azimuth",
                        issue_type=IssueType.OUT_OF_RANGE,
                        severity=IssueSeverity.WARNING,
                        original_value=str(az_original),
                        message=f"Azimuth {az_original} normalized to {az:.1f}",
                        suggested_action="Normalized to [0, 360)",
                        applied_action="accepted",
                    )
                )

            # Parse dip — flag out of range
            raw_dip = row.get("dip", -90)
            try:
                dip = float(raw_dip)
            except (ValueError, TypeError):
                dip = -90.0
            if dip < -90 or dip > 90:
                self._pending_quality_issues.append(
                    DataQualityIssue(
                        source_file="surveys.csv",
                        source_row=int(idx),
                        hole_id=bh_id,
                        field="dip",
                        issue_type=IssueType.OUT_OF_RANGE,
                        severity=IssueSeverity.WARNING,
                        original_value=str(dip),
                        message=f"Dip {dip} outside [-90, 90], clamped",
                        suggested_action="Clamp to [-90, 90]",
                        applied_action="",
                    )
                )
                dip = max(-90.0, min(90.0, dip))

            # Check for duplicate depth in same borehole (within pending)
            pending = self._pending_survey_stations.setdefault(bh_id, [])
            existing_mds = {s.measured_depth for s in pending}
            if md in existing_mds:
                issue = DataQualityIssue(
                    source_file="surveys.csv",
                    source_row=int(idx),
                    hole_id=bh_id,
                    field="measured_depth",
                    issue_type=IssueType.DUPLICATE,
                    severity=IssueSeverity.WARNING,
                    original_value=str(md),
                    message=f"Duplicate measured_depth {md} for {bh_id}",
                    suggested_action="Keep first occurrence",
                    applied_action="auto_excluded_keep_first",
                )
                self._pending_quality_issues.append(issue)
                self._pending_excluded_records.append(
                    {
                        "issue_id": issue.issue_id,
                        "source_file": issue.source_file,
                        "source_row": issue.source_row,
                        "hole_id": issue.hole_id,
                        "reason": issue.message,
                    }
                )
                self._stats["excluded"] += 1
                continue

            # Build valid SurveyStation
            station = SurveyStation(measured_depth=md, azimuth=az, dip=dip)
            pending.append(station)
            self._stats["cleaned"] += 1
            self._stats["attached"] += 1

    # ── Apply phase (auto-fixes on pending state only) ───────────────────

    def _accept_auto_fixes(self):
        """Apply auto-fixes: re-process raw surveys with normalization.

        Operates on PENDING state only — does NOT write to project.
        Clears pending stations, reprocesses raw_surveys with azimuth % 360
        and dip clamping applied during construction.
        """
        # Clear pending state
        self._pending_survey_stations.clear()
        self._pending_excluded_records = [
            record for record in self._pending_excluded_records if record.get("source_file") != "surveys.csv"
        ]
        # Remove only survey-related issues (others preserved)
        self._pending_quality_issues = [i for i in self._pending_quality_issues if i.source_file != "surveys.csv"]
        self._stats["cleaned"] = 0
        self._stats["attached"] = 0
        self._stats["excluded"] = 0

        raw_surveys = get_raw_surveys(self._project)
        if raw_surveys is None or raw_surveys.empty:
            self._refresh_table()
            return

        bh_map = {bh.borehole_id: bh for bh in self._project.borehole_collection}
        max_depths = {bh.borehole_id: bh.collar.final_depth for bh in self._project.borehole_collection}

        # Re-run the check (which builds pending with normalization)
        self._check_surveys(raw_surveys, bh_map, max_depths)

        # Re-count severities
        self._stats["n_errors"] = len([i for i in self._pending_quality_issues if i.severity == IssueSeverity.ERROR])
        self._stats["n_warnings"] = len(
            [i for i in self._pending_quality_issues if i.severity == IssueSeverity.WARNING]
        )

        self._refresh_table()

        ss = self._stats
        self._stats_label.setText(
            f"Auto-fixes applied. "
            f"Surveys: raw={ss['raw_surveys']} cleaned={ss['cleaned']} "
            f"attached={ss['attached']} excluded={ss['excluded']}"
        )
        self._update_idempotency_label()

    # ── Exclude selected ─────────────────────────────────────────────────

    def _exclude_selected(self):
        """Exclude selected source records by stable issue identity."""
        selected_issue_ids = set()
        for row in {index.row() for index in self._table.selectedIndexes()}:
            item = self._table.item(row, 0)
            if item is not None:
                issue_id = item.data(Qt.ItemDataRole.UserRole)
                if issue_id:
                    selected_issue_ids.add(issue_id)
        if not selected_issue_ids:
            return

        keep = []
        for issue in self._pending_quality_issues:
            if issue.issue_id not in selected_issue_ids:
                keep.append(issue)
                continue
            issue.applied_action = "excluded"
            self._resolved_quality_issues.append(issue)
            existing_record = next(
                (
                    record
                    for record in self._pending_excluded_records
                    if issue.source_file == "fractures.csv"
                    and record.get("source_file") == issue.source_file
                    and record.get("source_row") == issue.source_row
                ),
                None,
            )
            if existing_record is None:
                self._pending_excluded_records.append(
                    {
                        "issue_id": issue.issue_id,
                        "source_file": issue.source_file,
                        "source_row": issue.source_row,
                        "hole_id": issue.hole_id,
                        "reason": issue.message,
                        "applied_action": "excluded_during_cleaning",
                    }
                )
            else:
                existing_record["quality_issue_id"] = issue.issue_id
                existing_record["applied_action"] = "confirmed_excluded_during_cleaning"
            self._remove_pending_source_record(issue)
        self._pending_quality_issues = keep
        self._stats["n_errors"] = len([i for i in self._pending_quality_issues if i.severity == IssueSeverity.ERROR])
        self._stats["n_warnings"] = len(
            [i for i in self._pending_quality_issues if i.severity == IssueSeverity.WARNING]
        )
        self._refresh_table()

    def _remove_pending_source_record(self, issue: DataQualityIssue) -> None:
        """Remove one source record from the appropriate pending dataset."""
        if issue.source_file == "surveys.csv":
            raw = get_raw_surveys(self._project)
            if raw is None or issue.source_row not in raw.index:
                return
            row = raw.loc[issue.source_row]
            hole_id = str(row.get("hole_id", "")).strip()
            try:
                measured_depth = float(row.get("measured_depth"))
            except (TypeError, ValueError):
                return
            self._pending_survey_stations[hole_id] = [
                station
                for station in self._pending_survey_stations.get(hole_id, [])
                if station.measured_depth != measured_depth
            ]
        elif issue.source_file == "rqd.csv":
            if issue.source_row in self._pending_rqd.index:
                self._pending_rqd = self._pending_rqd.drop(index=issue.source_row)
        elif issue.source_file == "domain_intervals.csv":
            interval_id = self._pending_domain_sources.pop(issue.source_row, None)
            if interval_id is not None:
                self._pending_domain_intervals = [
                    interval for interval in self._pending_domain_intervals if interval.interval_id != interval_id
                ]

    # ── Commit phase (OK button) ─────────────────────────────────────────

    def _on_accept(self):
        """Commit all pending state to project.  Blocks on unresolved ERRORs."""
        if self._has_unresolved_errors:
            n = self._stats["n_errors"]
            QMessageBox.critical(
                self,
                "Unresolved Errors",
                f"There are still {n} ERROR-level data quality issues.\n\n"
                "These errors will block downstream processing (holdout, domains, joint sets).\n"
                "Please resolve or exclude the errors before proceeding.",
            )
            # Do NOT accept — keep dialog open
            return

        # ── Commit survey stations to project ───────────────────────────
        if self._pending_survey_stations:
            for bh in self._project.borehole_collection:
                cleaned = self._pending_survey_stations.get(bh.borehole_id, [])
                bh.survey.stations = cleaned  # Replace, not append
        # ── Persist quality issues ───────────────────────────────────────
        set_quality_issues(
            self._project,
            self._pending_quality_issues + self._resolved_quality_issues,
        )
        if self._stats["raw_rqd"]:
            set_cleaned_rqd(self._project, self._pending_rqd.copy())
        set_excluded_records(self._project, self._pending_excluded_records)

        # ── Update domain intervals if no ERRORs from domain validation ──
        if self._stats["raw_domain_intervals"]:
            set_domain_intervals(self._project, self._pending_domain_intervals)

        # ── Mark workflow ────────────────────────────────────────────────
        if self._stats["n_errors"] == 0:
            self._workflow.complete_step("clean")
        else:
            self._workflow.mark_issues("clean")
        self._workflow.mark_ready("holdout")

        self._committed_changes = self._commit_signature() != self._initial_commit_signature
        self.accept()

    # ── UI helpers ────────────────────────────────────────────────────────

    def _refresh_table(self):
        filt = self._filter_combo.currentData()
        if filt == "error":
            issues = [i for i in self._pending_quality_issues if i.severity == IssueSeverity.ERROR]
        elif filt == "warning":
            issues = [i for i in self._pending_quality_issues if i.severity == IssueSeverity.WARNING]
        elif filt == "info":
            issues = [i for i in self._pending_quality_issues if i.severity == IssueSeverity.INFO]
        else:
            issues = self._pending_quality_issues

        self._table.setRowCount(len(issues))
        for row, issue in enumerate(issues):
            type_item = QTableWidgetItem(issue.issue_type.value)
            type_item.setData(Qt.ItemDataRole.UserRole, issue.issue_id)
            self._table.setItem(row, 0, type_item)
            self._table.setItem(row, 1, QTableWidgetItem(issue.hole_id))
            self._table.setItem(row, 2, QTableWidgetItem(str(issue.source_row)))
            self._table.setItem(row, 3, QTableWidgetItem(issue.field))
            self._table.setItem(row, 4, QTableWidgetItem(str(issue.original_value)))
            sev_item = QTableWidgetItem(issue.severity.value.upper())
            if issue.severity == IssueSeverity.ERROR:
                sev_item.setForeground(Qt.GlobalColor.red)
            else:
                sev_item.setForeground(Qt.GlobalColor.darkYellow)
            self._table.setItem(row, 5, sev_item)
            self._table.setItem(row, 6, QTableWidgetItem(issue.message))
            self._table.setItem(row, 7, QTableWidgetItem(issue.suggested_action))

        ss = self._stats
        stats_text = (
            f"{len(issues)} issues ({ss['n_errors']} errors, " f"{ss['n_warnings']} warnings, {ss['n_info']} info)"
        )
        if ss.get("raw_surveys"):
            stats_text += (
                f" | Surveys: raw={ss['raw_surveys']} cleaned={ss['cleaned']} "
                f"attached={ss['attached']} excluded={ss['excluded']}"
            )
        if ss.get("raw_rqd"):
            stats_text += f" | RQD: {ss['raw_rqd']} rows"
        if ss.get("raw_domain_intervals"):
            stats_text += f" | Domains: {ss['raw_domain_intervals']} rows"
        self._stats_label.setText(stats_text)

        # Update error warning
        if self._has_unresolved_errors:
            self._error_warning.setText(
                f"⚠ {ss['n_errors']} unresolved ERRORs — " f"cleaning cannot be COMPLETED until resolved"
            )
            self._error_warning.setStyleSheet("color: red; font-weight: bold;")
        else:
            self._error_warning.setText("")
            self._error_warning.setStyleSheet("")

    def _update_idempotency_label(self):
        current = self._count_formal_stations()
        initial = self._initial_station_count
        pending = sum(len(v) for v in self._pending_survey_stations.values())
        if current == initial:
            self._idempotency_label.setText(
                f"✓ Idempotent: formal stations unchanged ({current}). "
                f"Pending: {pending} stations ready for commit."
            )
            self._idempotency_label.setStyleSheet("color: green;")
        else:
            self._idempotency_label.setText(
                f"⚠ WARNING: formal station count changed from "
                f"{initial} to {current} — check phase is not idempotent!"
            )
            self._idempotency_label.setStyleSheet("color: red; font-weight: bold;")

    def _export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Quality Report", "quality_report.json", "JSON (*.json)")
        if not path:
            return
        report = [
            {
                "issue_id": i.issue_id,
                "source_file": i.source_file,
                "source_row": i.source_row,
                "hole_id": i.hole_id,
                "field": i.field,
                "original_value": str(i.original_value),
                "issue_type": i.issue_type.value,
                "severity": i.severity.value,
                "message": i.message,
                "suggested_action": i.suggested_action,
            }
            for i in self._pending_quality_issues
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        self._stats_label.setText(self._stats_label.text() + f"  |  Exported to {Path(path).name}")

    # ── Public accessors ──────────────────────────────────────────────────

    def get_issues(self):
        return self._pending_quality_issues

    def get_pending_station_count(self) -> int:
        return sum(len(v) for v in self._pending_survey_stations.values())

    @property
    def committed_changes(self) -> bool:
        """Whether OK changed cleaning-owned project state."""
        return self._committed_changes
