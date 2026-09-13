"""Dockable M8 borehole database browser and maintenance panel."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

from dfn_cave_studio.models.borehole_database import BoreholeDataType, RecordState
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.ui.qt_adapter import (
    QComboBox,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class BoreholeDatabasePanel(QDockWidget):
    """Searchable, sortable view over the canonical project repository."""

    def __init__(self, project, on_changed: Callable[..., None] | None = None, parent=None):
        super().__init__("Borehole Database / 钻孔数据库", parent)
        self._project = project
        self._repository = BoreholeRepository(project)
        self._on_changed = on_changed
        self._build_ui()
        self.refresh()

    def set_project(self, project) -> None:
        """Bind the panel to another project."""
        self._project = project
        self._repository = BoreholeRepository(project)
        self.refresh()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        filters = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search hole, source, value…")
        self._search.textChanged.connect(self.refresh)
        filters.addWidget(self._search)
        self._type = QComboBox()
        self._type.addItem("All data types", None)
        for data_type in BoreholeDataType:
            self._type.addItem(data_type.value, data_type.value)
        self._type.currentIndexChanged.connect(self.refresh)
        filters.addWidget(self._type)
        self._state = QComboBox()
        self._state.addItem("Raw (all imported)", "raw")
        for state in RecordState:
            self._state.addItem(state.value.title(), state.value)
        self._state.currentIndexChanged.connect(self.refresh)
        filters.addWidget(self._state)
        self._orientation = QComboBox()
        self._orientation.addItem("All orientation records", None)
        self._orientation.addItem("Full orientation", "full_orientation")
        self._orientation.addItem("Dip only", "dip_only")
        self._orientation.currentIndexChanged.connect(self.refresh)
        filters.addWidget(self._orientation)
        layout.addLayout(filters)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._holes = QListWidget()
        self._holes.currentItemChanged.connect(self.refresh)
        splitter.addWidget(self._holes)
        self._table = QTableWidget()
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setDefaultSectionSize(120)
        self._table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self._table)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self._orientation_summary = QTableWidget(0, 6)
        self._orientation_summary.setHorizontalHeaderLabels(
            ["Point key", "Input observations", "Includes RANDOM", "Jv estimate", "RQD from Jv", "Method"]
        )
        self._orientation_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._orientation_summary)
        self._counts = QLabel()
        layout.addWidget(self._counts)

        buttons = QHBoxLayout()
        import_button = QPushButton("Import / Append…")
        import_button.clicked.connect(self._open_import)
        buttons.addWidget(import_button)
        edit_button = QPushButton("Edit selected…")
        edit_button.clicked.connect(self._edit_selected)
        buttons.addWidget(edit_button)
        delete_button = QPushButton("Delete selected…")
        delete_button.clicked.connect(self._delete_selected)
        buttons.addWidget(delete_button)
        layout.addLayout(buttons)
        self.setWidget(root)

    def refresh(self, *_args) -> None:
        """Refresh holes, counts, and real record values."""
        selected_hole = self._holes.currentItem().data(Qt.ItemDataRole.UserRole) if self._holes.currentItem() else None
        hole_ids = sorted({record.hole_id for record in self._repository.query(raw=True) if record.hole_id})
        if (
            self._holes.count() != len(hole_ids) + 1
            or [self._holes.item(index).data(Qt.ItemDataRole.UserRole) for index in range(1, self._holes.count())]
            != hole_ids
        ):
            self._holes.blockSignals(True)
            self._holes.clear()
            all_item = QListWidgetItem("All boreholes")
            all_item.setData(Qt.ItemDataRole.UserRole, None)
            self._holes.addItem(all_item)
            for hole_id in hole_ids:
                item = QListWidgetItem(hole_id)
                item.setData(Qt.ItemDataRole.UserRole, hole_id)
                self._holes.addItem(item)
            target = 0
            if selected_hole in hole_ids:
                target = hole_ids.index(selected_hole) + 1
            self._holes.setCurrentRow(target)
            self._holes.blockSignals(False)

        data_type = self._type.currentData()
        state_value = self._state.currentData()
        state = None if state_value == "raw" else state_value
        hole_id = self._holes.currentItem().data(Qt.ItemDataRole.UserRole) if self._holes.currentItem() else None
        records = self._repository.query(data_type=data_type, state=state, hole_id=hole_id, raw=state_value == "raw")
        orientation = self._orientation.currentData()
        if orientation is not None:
            records = [
                record
                for record in records
                if record.data_type == BoreholeDataType.FRACTURES
                and record.values.get("orientation_completeness") == orientation
            ]
        search = self._search.text().strip().lower()
        if search:
            records = [
                record
                for record in records
                if search
                in json.dumps(
                    {
                        "hole_id": record.hole_id,
                        "source": record.source_file,
                        "values": record.values,
                        "reason": record.exclusion_reason,
                    },
                    default=str,
                ).lower()
            ]
        value_fields = sorted({key for record in records for key in record.values})
        columns = [
            "record_id",
            "data_type",
            "state",
            "hole_id",
            "source_file",
            "source_row",
            "reason",
            *value_fields,
        ]
        self._table.setUpdatesEnabled(False)
        self._table.setSortingEnabled(False)
        try:
            self._table.setRowCount(len(records))
            self._table.setColumnCount(len(columns))
            self._table.setHorizontalHeaderLabels(columns)
            for row, record in enumerate(records):
                metadata = {
                    "record_id": record.record_id,
                    "data_type": record.data_type,
                    "state": record.state.value,
                    "hole_id": record.hole_id,
                    "source_file": record.source_file,
                    "source_row": record.source_row,
                    "reason": record.exclusion_reason or "",
                }
                for column, name in enumerate(columns):
                    value = metadata.get(name, record.values.get(name, ""))
                    display = "—" if value is None and name == "dip_direction" else ("" if value is None else str(value))
                    item = QTableWidgetItem(display)
                    item.setData(Qt.ItemDataRole.UserRole, record.record_id)
                    self._table.setItem(row, column, item)
        finally:
            self._table.setSortingEnabled(True)
            self._table.setUpdatesEnabled(True)
        count_records = self._repository.query(data_type=data_type, hole_id=hole_id, raw=True)
        counts = {
            "raw": len(count_records),
            "formal": sum(record.state == RecordState.FORMAL for record in count_records),
            "excluded": sum(record.state == RecordState.EXCLUDED for record in count_records),
            "pending": sum(record.state == RecordState.PENDING for record in count_records),
        }
        missing_text = ""
        if hole_id:
            missing = [
                table.value
                for table in BoreholeDataType
                if not self._repository.query(table.value, RecordState.FORMAL, hole_id=hole_id)
            ]
            missing_text = f" | Missing: {', '.join(missing) if missing else 'none'}"
        self._counts.setText(
            f"Raw {counts['raw']} | Formal {counts['formal']} | "
            f"Excluded {counts['excluded']} | Pending {counts['pending']} | Visible {len(records)}{missing_text}"
        )
        if data_type in (None, BoreholeDataType.FRACTURES):
            orientation_counts = self._repository.database.orientation_counts()
            self._counts.setText(
                f"{self._counts.text()} | Full orientation {orientation_counts['full_orientation']} | "
                f"Dip only {orientation_counts['dip_only']}"
            )
        summaries = self._repository.database.orientation_point_summaries
        show_summaries = data_type == BoreholeDataType.ORIENTATION_POINTS
        self._orientation_summary.setVisible(show_summaries)
        self._orientation_summary.setRowCount(len(summaries) if show_summaries else 0)
        if show_summaries:
            for row, summary in enumerate(summaries):
                values = (
                    summary.point_key,
                    ", ".join(summary.input_observation_ids),
                    "Yes" if summary.includes_random else "No",
                    f"{summary.jv_estimated:g}",
                    f"{summary.rqd_from_jv:g}",
                    summary.method_code,
                )
                for column, value in enumerate(values):
                    self._orientation_summary.setItem(row, column, QTableWidgetItem(value))

    def _selected_record_id(self) -> str | None:
        items = self._table.selectedItems()
        return items[0].data(Qt.ItemDataRole.UserRole) if items else None

    def _open_import(self) -> None:
        from dfn_cave_studio.ui.dialogs.m8_import_dialog import M8ImportDialog

        dialog = M8ImportDialog(self._project, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.committed_changes:
            self.refresh()
            self._notify_changed(dialog.committed_data_types)

    def _edit_selected(self) -> None:
        record_id = self._selected_record_id()
        if not record_id:
            return
        record = next(record for record in self._repository.database.records if record.record_id == record_id)
        text, accepted = QInputDialog.getMultiLineText(
            self,
            "Edit cleaned/formal values",
            "JSON values (Raw remains unchanged):",
            json.dumps(record.values, indent=2, ensure_ascii=False, default=str),
        )
        if not accepted:
            return
        try:
            values = json.loads(text)
            if not isinstance(values, dict):
                raise TypeError("Edited JSON must be an object")
        except (json.JSONDecodeError, TypeError) as error:
            QMessageBox.warning(self, "Invalid values", str(error))
            return
        if QMessageBox.question(self, "Confirm edit", "Apply this audited edit?") != QMessageBox.StandardButton.Yes:
            return
        self._repository.edit_record(record_id, values)
        self.refresh()
        self._notify_changed({record.data_type})

    def _delete_selected(self) -> None:
        record_id = self._selected_record_id()
        if not record_id:
            return
        record = next(record for record in self._repository.database.records if record.record_id == record_id)
        if (
            QMessageBox.question(self, "Confirm delete", "Move this record to Excluded? Raw data will be retained.")
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._repository.delete_record(record_id)
        self.refresh()
        self._notify_changed({record.data_type})

    def _notify_changed(self, data_types: Iterable[str] | None = None) -> None:
        if self._on_changed is not None:
            if data_types is None:
                self._on_changed()
            else:
                self._on_changed(set(data_types))
