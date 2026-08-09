"""Transactional M8 borehole data-quality review dialog."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dfn_cave_studio.models.borehole_database import (
    BoreholeDataType,
    QualityIssueStatus,
    QualitySeverity,
)
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.ui.qt_adapter import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    Qt,
    QVBoxLayout,
)


class M8QualityDialog(QDialog):
    """Review and resolve findings against the canonical M8 database."""

    COLUMNS = (
        "data_type",
        "severity",
        "status",
        "hole_id",
        "source_file",
        "source_row",
        "field",
        "original_value",
        "current_value",
        "suggested_value",
        "reason",
    )

    def __init__(self, project: Any, workflow: Any, parent: Any = None):
        super().__init__(parent)
        self._project = project
        self._workflow = workflow
        self._database_snapshot = project.borehole_database.model_copy(deep=True)
        self._workflow_snapshot = workflow.to_dict()
        self._service = BoreholeQualityService(project)
        self._changed = False
        self._committed_changes = False
        self._quality_completed = False
        self._quality_operation_active = False
        self.setWindowTitle("M8 Data Quality / 数据质量")
        self.resize(1280, 720)
        self._build_ui()
        self._service.run_checks()
        self._sync_workflow_gate()
        self._changed = (
            self._database_snapshot.model_dump_json() != project.borehole_database.model_dump_json()
            or self._workflow_snapshot != workflow.to_dict()
        )
        self.refresh()

    @property
    def committed_changes(self) -> bool:
        """Whether accepted actions changed persisted project state."""
        return self._committed_changes

    @property
    def quality_completed(self) -> bool:
        """Whether this session confirmed the quality gate."""
        return self._quality_completed

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        filters = QHBoxLayout()
        self._type_filter = QComboBox()
        self._type_filter.addItem("All data types", None)
        for data_type in BoreholeDataType:
            self._type_filter.addItem(data_type.value, data_type.value)
        self._severity_filter = QComboBox()
        self._severity_filter.addItem("All severities", None)
        for severity in QualitySeverity:
            self._severity_filter.addItem(severity.value.upper(), severity.value)
        self._status_filter = QComboBox()
        self._status_filter.addItem("All states", None)
        for status in QualityIssueStatus:
            self._status_filter.addItem(status.value.title(), status.value)
        for combo in (self._type_filter, self._severity_filter, self._status_filter):
            combo.currentIndexChanged.connect(self.refresh)
            filters.addWidget(combo)
        layout.addLayout(filters)

        self._table = QTableWidget()
        self._table.setColumnCount(len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setDefaultSectionSize(120)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

        self._summary = QLabel()
        layout.addWidget(self._summary)

        checks = QHBoxLayout()
        self._rerun_button = QPushButton("Re-run checks")
        self._configure_action_button(self._rerun_button)
        self._rerun_button.clicked.connect(self._rerun)
        checks.addWidget(self._rerun_button)
        self._selected_fix_button = QPushButton("Accept selected auto-fix")
        self._configure_action_button(self._selected_fix_button)
        self._selected_fix_button.clicked.connect(self._accept_selected_fixes)
        checks.addWidget(self._selected_fix_button)
        self._all_fix_button = QPushButton("Accept all auto-fixes")
        self._configure_action_button(self._all_fix_button)
        self._all_fix_button.clicked.connect(self._accept_all_fixes)
        checks.addWidget(self._all_fix_button)
        edit = QPushButton("Manual edit record…")
        edit.clicked.connect(self._manual_edit)
        checks.addWidget(edit)
        exclude = QPushButton("Exclude record…")
        exclude.clicked.connect(self._exclude)
        checks.addWidget(exclude)
        retry = QPushButton("Retry / reclassify")
        retry.clicked.connect(self._retry)
        checks.addWidget(retry)
        layout.addLayout(checks)

        audit = QHBoxLayout()
        confirm_selected = QPushButton("Confirm selected exclusions")
        confirm_selected.clicked.connect(self._confirm_selected_exclusions)
        audit.addWidget(confirm_selected)
        self._confirm_all_button = QPushButton("Confirm all documented exclusions")
        self._configure_action_button(self._confirm_all_button)
        self._confirm_all_button.clicked.connect(self._confirm_all_exclusions)
        audit.addWidget(self._confirm_all_button)
        export_json = QPushButton("Export JSON…")
        export_json.clicked.connect(self._export_json)
        audit.addWidget(export_json)
        export_csv = QPushButton("Export CSV…")
        export_csv.clicked.connect(self._export_csv)
        audit.addWidget(export_csv)
        self._complete_button = QPushButton("Confirm data quality complete")
        self._configure_action_button(self._complete_button)
        self._complete_button.clicked.connect(self._confirm_complete)
        audit.addWidget(self._complete_button)
        layout.addLayout(audit)

        self._dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._dialog_buttons.accepted.connect(self.accept)
        self._dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(self._dialog_buttons)

    def refresh(self, *_args: Any) -> None:
        """Refresh the filtered issue table and quality-gate counts."""
        data_type = self._type_filter.currentData()
        severity = self._severity_filter.currentData()
        status = self._status_filter.currentData()
        issues = [
            issue
            for issue in self._service.issues
            if (data_type is None or issue.data_type == data_type)
            and (severity is None or issue.severity.value == severity)
            and (status is None or issue.status.value == status)
        ]
        self._table.setUpdatesEnabled(False)
        try:
            self._table.setRowCount(len(issues))
            for row, issue in enumerate(issues):
                values = {name: getattr(issue, name) for name in self.COLUMNS}
                for column, name in enumerate(self.COLUMNS):
                    value = values[name]
                    if name in {"severity", "status"}:
                        value = value.value
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False, default=str)
                    item = QTableWidgetItem("" if value is None else str(value))
                    item.setData(Qt.ItemDataRole.UserRole, (issue.issue_id, issue.record_id))
                    self._table.setItem(row, column, item)
        finally:
            self._table.setUpdatesEnabled(True)
        counts = self._project.borehole_database.counts()
        confirmed = self._project.borehole_database.quality_confirmed_at
        self._summary.setText(
            f"Raw {counts['raw']} | Formal {counts['formal']} | Excluded {counts['excluded']} | "
            f"Pending {counts['pending']} | Unresolved ERROR {self._service.unresolved_error_count} | "
            f"Quality confirmed: {confirmed.isoformat() if confirmed else 'no'}"
        )

    def reject(self) -> None:
        """Rollback every staged project mutation when Cancel is selected."""
        self._service.repository.restore(self._database_snapshot)
        self._workflow.from_dict(self._workflow_snapshot)
        self._changed = False
        self._committed_changes = False
        super().reject()

    def accept(self) -> None:
        """Commit staged-change metadata and close without recomputation."""
        if self._quality_operation_active:
            return
        self._committed_changes = self._changed
        super().accept()

    def _selected_ids(self) -> tuple[list[str], list[str]]:
        issue_ids: list[str] = []
        record_ids: list[str] = []
        for index in self._table.selectionModel().selectedRows():
            item = self._table.item(index.row(), 0)
            if item is not None:
                issue_id, record_id = item.data(Qt.ItemDataRole.UserRole)
                issue_ids.append(issue_id)
                record_ids.append(record_id)
        return list(dict.fromkeys(issue_ids)), list(dict.fromkeys(record_ids))

    def _rerun(self) -> None:
        def operation() -> bool:
            before = self._project.borehole_database.model_dump_json()
            self._service.run_checks()
            self._sync_workflow_gate()
            self._changed |= (
                before != self._project.borehole_database.model_dump_json()
                or self._workflow_snapshot != self._workflow.to_dict()
            )
            return True

        self._run_quality_operation(operation, "Quality check failed")

    def _accept_selected_fixes(self) -> None:
        issue_ids, _ = self._selected_ids()
        if not issue_ids:
            return
        self._run_quality_operation(
            lambda: self._apply_auto_fixes(issue_ids),
            "Selected auto-fix failed",
        )

    def _accept_all_fixes(self) -> None:
        self._run_quality_operation(
            lambda: self._apply_auto_fixes(None),
            "Automatic correction failed",
        )

    def _apply_auto_fixes(self, issue_ids: list[str] | None) -> bool:
        applied = self._service.apply_auto_fixes(issue_ids)
        self._changed |= applied > 0
        self._sync_workflow_gate()
        return applied > 0

    @staticmethod
    def _configure_action_button(button: QPushButton) -> None:
        button.setCheckable(False)
        button.setAutoRepeat(False)

    def _run_quality_operation(self, operation: Callable[[], bool | None], title: str) -> None:
        """Run one guarded quality operation and always restore UI controls."""
        if self._quality_operation_active:
            return
        self._quality_operation_active = True
        action_buttons = (self._rerun_button, self._selected_fix_button, self._all_fix_button)
        for button in action_buttons:
            button.setEnabled(False)
        refresh_required = True
        try:
            refresh_required = operation() is not False
        except Exception as error:
            QMessageBox.critical(self, title, str(error))
        finally:
            self._quality_operation_active = False
            for button in action_buttons:
                button.setEnabled(True)
            if refresh_required:
                try:
                    self.refresh()
                except Exception as error:
                    QMessageBox.critical(self, "Quality table refresh failed", str(error))

    def _manual_edit(self) -> None:
        _, record_ids = self._selected_ids()
        if len(record_ids) != 1:
            QMessageBox.information(self, "Select one record", "Select exactly one issue row to edit its record.")
            return
        record = self._service.repository.get_record(record_ids[0])
        text, accepted = QInputDialog.getMultiLineText(
            self,
            "Manual cleaned-value edit",
            "JSON values (original_values remain unchanged):",
            json.dumps(record.values, indent=2, ensure_ascii=False, default=str),
        )
        if not accepted:
            return
        try:
            values = json.loads(text)
            if not isinstance(values, dict):
                raise TypeError("Edited JSON must be an object")
            self._service.edit_record(record.record_id, values)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid edit", str(error))
            return
        self._changed = True
        self._sync_workflow_gate()
        self.refresh()

    def _exclude(self) -> None:
        _, record_ids = self._selected_ids()
        if len(record_ids) != 1:
            QMessageBox.information(self, "Select one record", "Select exactly one issue row to exclude its record.")
            return
        reason, accepted = QInputDialog.getText(self, "Exclude record", "Required exclusion reason:")
        if not accepted:
            return
        try:
            self._service.exclude_record(record_ids[0], reason)
        except ValueError as error:
            QMessageBox.warning(self, "Cannot exclude", str(error))
            return
        self._changed = True
        self._sync_workflow_gate()
        self.refresh()

    def _retry(self) -> None:
        _, record_ids = self._selected_ids()
        for record_id in record_ids:
            self._service.retry_record(record_id)
        self._changed |= bool(record_ids)
        self._sync_workflow_gate()
        self.refresh()

    def _confirm_selected_exclusions(self) -> None:
        _, record_ids = self._selected_ids()
        self._run_quality_operation(
            lambda: self._apply_exclusion_confirmation(record_ids),
            "Exclusion confirmation failed",
        )

    def _confirm_all_exclusions(self) -> None:
        self._run_quality_operation(
            lambda: self._apply_exclusion_confirmation(None),
            "Exclusion confirmation failed",
        )

    def _apply_exclusion_confirmation(self, record_ids: list[str] | None) -> bool:
        confirmed = self._service.confirm_exclusions(record_ids)
        self._changed |= confirmed > 0
        self._sync_workflow_gate()
        return confirmed > 0

    def _confirm_complete(self) -> None:
        try:
            self._service.confirm_complete(self._workflow)
        except ValueError as error:
            QMessageBox.warning(self, "Quality gate blocked", str(error))
            self.refresh()
            return
        except Exception as error:
            QMessageBox.critical(self, "Quality confirmation failed", str(error))
            self.refresh()
            return
        self._quality_completed = True
        self._changed = True
        self.refresh()

    def _sync_workflow_gate(self) -> None:
        """Reflect only unresolved quality blockers in workflow state."""
        counts = self._project.borehole_database.counts()
        if counts["raw"]:
            self._workflow.complete_step("import")
        if counts["pending"] or self._service.unresolved_error_count:
            self._workflow.mark_issues("clean")
        elif self._project.borehole_database.quality_confirmed_at is not None:
            self._workflow.complete_step("clean")
        else:
            self._workflow.mark_ready("clean")

    def _export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export quality report", "quality-report.json", "JSON (*.json)")
        if path:
            self._service.export_json(Path(path))

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export quality report", "quality-report.csv", "CSV (*.csv)")
        if path:
            self._service.export_csv(Path(path))
