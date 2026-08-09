"""Independent M8 borehole-table import dialog with transactional cancel."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from dfn_cave_studio.models.borehole_database import BoreholeDataType
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.import_service import read_input_table
from dfn_cave_studio.ui.qt_adapter import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class M8ImportDialog(QDialog):
    """Preview, map, and stage one or more independent imports."""

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._repository = BoreholeRepository(project)
        self._snapshot = self._repository.snapshot()
        m7_data = getattr(project, "_m7_data", {}) or {}
        self._legacy_domain_snapshot = deepcopy(m7_data.get("domain_intervals", []))
        self._preview: pd.DataFrame | None = None
        self._staged_changes = False
        self._committed_changes = False
        self._results: list[dict] = []
        self.setWindowTitle("M8 Independent Borehole Import")
        self.resize(900, 620)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._type_combo = QComboBox()
        for data_type in BoreholeDataType:
            self._type_combo.addItem(data_type.value, data_type.value)
        form.addRow("Data type:", self._type_combo)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Append (skip exact duplicates)", "append")
        self._mode_combo.addItem("Replace formal table (retain audit history)", "replace")
        form.addRow("Commit behavior:", self._mode_combo)
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        path_row.addWidget(self._path_edit)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)
        preview = QPushButton("Preview")
        preview.clicked.connect(self._load_preview)
        path_row.addWidget(preview)
        form.addRow("CSV/XLSX:", path_row)
        layout.addLayout(form)

        self._preview_table = QTableWidget()
        self._preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._preview_table)
        layout.addWidget(QLabel("Field mapping: edit the Standard Field column before staging."))
        self._mapping_table = QTableWidget(0, 2)
        self._mapping_table.setHorizontalHeaderLabels(["Source Column", "Standard Field"])
        self._mapping_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._mapping_table)

        self._status = QLabel("No project data is changed unless OK is pressed.")
        layout.addWidget(self._status)
        stage = QPushButton("Stage This Import")
        stage.clicked.connect(self._stage_import)
        layout.addWidget(stage)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select borehole table",
            "",
            "Data files (*.csv *.xlsx *.xls);;All files (*)",
        )
        if path:
            self._path_edit.setText(path)
            self._load_preview()

    def _load_preview(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            return
        try:
            self._preview = read_input_table(path, nrows=100)
        except (OSError, ValueError, ImportError, KeyError) as error:
            QMessageBox.critical(self, "Preview failed", str(error))
            return
        dataframe = self._preview
        self._preview_table.setRowCount(len(dataframe))
        self._preview_table.setColumnCount(len(dataframe.columns))
        self._preview_table.setHorizontalHeaderLabels([str(column) for column in dataframe.columns])
        for row in range(len(dataframe)):
            for column, name in enumerate(dataframe.columns):
                self._preview_table.setItem(row, column, QTableWidgetItem(str(dataframe.iloc[row][name])))
        self._mapping_table.setRowCount(len(dataframe.columns))
        for row, column in enumerate(dataframe.columns):
            self._mapping_table.setItem(row, 0, QTableWidgetItem(str(column)))
            self._mapping_table.setItem(row, 1, QTableWidgetItem(str(column)))
        self._status.setText(f"Preview: {len(dataframe)} rows × {len(dataframe.columns)} fields")

    def _stage_import(self) -> None:
        path = self._path_edit.text().strip()
        if self._preview is None or not path:
            self._load_preview()
        if self._preview is None:
            return
        rename = {}
        for row in range(self._mapping_table.rowCount()):
            source = self._mapping_table.item(row, 0)
            target = self._mapping_table.item(row, 1)
            if source and target and target.text().strip():
                rename[source.text()] = target.text().strip()
        try:
            dataframe = read_input_table(path).rename(columns=rename)
        except (OSError, ValueError, ImportError, KeyError) as error:
            QMessageBox.critical(self, "Import failed", str(error))
            return
        result = self._repository.import_dataframe(
            self._type_combo.currentData(),
            dataframe,
            source_file=str(Path(path)),
            mode=self._mode_combo.currentData(),
            modification_source="staged_import",
        )
        self._results.append(dict(result))
        self._staged_changes = self._staged_changes or result.changed
        self._status.setText(
            f"Staged raw={result['raw']}, formal={result['formal']}, excluded={result['excluded']}, "
            f"pending={result['pending']}, duplicates={result['duplicates']}"
        )

    def _accept(self) -> None:
        self._committed_changes = self._staged_changes
        if self._committed_changes:
            for data_type in {result["data_type"] for result in self._results if result.get("raw", 0) > 0}:
                self._repository.invalidate_dependent_results(data_type)
        self.accept()

    def reject(self) -> None:
        """Rollback all staged repository changes."""
        self._repository.restore(self._snapshot)
        m7_data = getattr(self._repository.project, "_m7_data", None)
        if m7_data is None:
            self._repository.project._m7_data = {}
            m7_data = self._repository.project._m7_data
        m7_data["domain_intervals"] = deepcopy(self._legacy_domain_snapshot)
        self._committed_changes = False
        super().reject()

    @property
    def committed_changes(self) -> bool:
        """Whether OK committed at least one new raw record."""
        return self._committed_changes

    @property
    def results(self) -> list[dict]:
        """Return staged batch summaries."""
        return list(self._results)
