"""Independent M8 borehole-table import dialog with transactional cancel."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from dfn_cave_studio.models.borehole_database import BoreholeDataType
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.import_service import read_input_table
from dfn_cave_studio.borehole.borehole_importer import STANDARD_COLLAR_FIELDS, STANDARD_FRACTURE_FIELDS
from dfn_cave_studio.services.field_mapping import detect_field_mapping, validate_field_mapping
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
        self._preview_path = ""
        self._mapping_conflicts: list[str] = []
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
        self._type_combo.currentIndexChanged.connect(self._clear_preview_mapping)
        form.addRow("Data type:", self._type_combo)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Append (skip exact duplicates)", "append")
        self._mode_combo.addItem("Replace formal table (retain audit history)", "replace")
        form.addRow("Commit behavior:", self._mode_combo)
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.textChanged.connect(self._clear_preview_mapping)
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
        self._mapping_table = QTableWidget(0, 3)
        self._mapping_table.setHorizontalHeaderLabels(["Source Column", "Standard Field", "Required"])
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

    def _clear_preview_mapping(self, *_args) -> None:
        """Remove old mapping editors before a file/type preview is rebuilt."""
        for row in range(self._mapping_table.rowCount()):
            widget = self._mapping_table.cellWidget(row, 1)
            if widget is not None:
                self._mapping_table.removeCellWidget(row, 1)
                widget.setParent(None)
                widget.deleteLater()
        self._mapping_table.setRowCount(0)
        self._mapping_table.clearContents()
        self._preview_table.setRowCount(0)
        self._preview_table.setColumnCount(0)
        self._preview = None
        self._preview_path = ""

    @staticmethod
    def _standard_field_options(data_type: str, current: str) -> list[str]:
        """Return mapping choices without changing existing detection rules."""
        fields = {
            BoreholeDataType.COLLARS: list(STANDARD_COLLAR_FIELDS),
            BoreholeDataType.SURVEYS: ["hole_id", "measured_depth", "azimuth", "dip"],
            BoreholeDataType.FRACTURES: [
                "hole_id",
                "depth",
                "dip_direction",
                "dip",
                "set_id",
                "aperture",
                "fracture_type",
                "confidence",
            ],
            BoreholeDataType.RQD: ["hole_id", "from_depth", "to_depth", "rqd", "core_recovery"],
            BoreholeDataType.DOMAIN_INTERVALS: [
                "hole_id",
                "from_depth",
                "to_depth",
                "domain_id",
                "domain_name",
            ],
        }.get(data_type, [])
        options = [""] + fields
        if current and current not in options:
            options.append(current)
        return options

    def _load_preview(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            return
        self._clear_preview_mapping()
        try:
            self._preview = read_input_table(path, nrows=100)
        except (OSError, ValueError, ImportError, KeyError) as error:
            QMessageBox.critical(self, "Preview failed", str(error))
            return
        self._preview_path = path
        dataframe = self._preview
        self._preview_table.setRowCount(len(dataframe))
        self._preview_table.setColumnCount(len(dataframe.columns))
        self._preview_table.setHorizontalHeaderLabels([str(column) for column in dataframe.columns])
        for row in range(len(dataframe)):
            for column, name in enumerate(dataframe.columns):
                value = dataframe.iloc[row][name]
                self._preview_table.setItem(row, column, QTableWidgetItem("" if pd.isna(value) else str(value)))
        self._mapping_table.setRowCount(len(dataframe.columns))
        data_type = self._type_combo.currentData()
        required = {
            BoreholeDataType.COLLARS: {"borehole_id", "collar_x", "collar_y", "collar_z", "final_depth"},
            BoreholeDataType.SURVEYS: {"hole_id", "measured_depth", "azimuth", "dip"},
            BoreholeDataType.FRACTURES: {"hole_id", "depth", "dip"},
            BoreholeDataType.RQD: {"hole_id", "from_depth", "to_depth", "rqd"},
            BoreholeDataType.DOMAIN_INTERVALS: {"hole_id", "from_depth", "to_depth", "domain_id"},
        }.get(data_type, set())
        fracture_aliases = {
            alias.lower(): standard
            for standard, aliases in STANDARD_FRACTURE_FIELDS.items()
            for alias in aliases
        }
        fracture_targets = {"borehole_id": "hole_id", "measured_depth": "depth"}
        collar_mapping: dict[str, str] = {}
        if data_type == BoreholeDataType.COLLARS:
            detection = detect_field_mapping(dataframe.columns, STANDARD_COLLAR_FIELDS)
            collar_mapping = detection.mapping
            self._mapping_conflicts = detection.conflicts
        else:
            self._mapping_conflicts = []
        for row, column in enumerate(dataframe.columns):
            self._mapping_table.setItem(row, 0, QTableWidgetItem(str(column)))
            standard = collar_mapping.get(str(column), "") if data_type == BoreholeDataType.COLLARS else str(column)
            if data_type == BoreholeDataType.FRACTURES:
                standard = fracture_targets.get(fracture_aliases.get(standard.lower(), standard), fracture_aliases.get(standard.lower(), standard))
            selector = QComboBox()
            selector.addItems(self._standard_field_options(data_type, standard))
            selector.setCurrentText(standard)
            self._mapping_table.setCellWidget(row, 1, selector)
            required_item = QTableWidgetItem("Yes" if standard in required else "")
            self._mapping_table.setItem(row, 2, required_item)
            selector.currentTextChanged.connect(
                lambda value, item=required_item, required_fields=required: item.setText(
                    "Yes" if value in required_fields else ""
                )
            )
        self._status.setText(f"Preview: {len(dataframe)} rows × {len(dataframe.columns)} fields")

        if self._mapping_conflicts:
            self._status.setText(f"Mapping conflict: {'; '.join(self._mapping_conflicts)}")

    def _stage_import(self) -> None:
        path = self._path_edit.text().strip()
        if self._preview is None or self._preview_path != path or not path:
            self._load_preview()
        if self._preview is None:
            return
        rename = {}
        for row in range(self._mapping_table.rowCount()):
            source = self._mapping_table.item(row, 0)
            target = self._mapping_table.cellWidget(row, 1)
            if source and isinstance(target, QComboBox) and target.currentText().strip():
                rename[source.text()] = target.currentText().strip()
        data_type = self._type_combo.currentData()
        try:
            raw_dataframe = read_input_table(path)
            if data_type == BoreholeDataType.COLLARS:
                rename = validate_field_mapping(raw_dataframe.columns, rename, STANDARD_COLLAR_FIELDS)
            dataframe = raw_dataframe.rename(columns=rename)
        except (OSError, ValueError, ImportError, KeyError) as error:
            QMessageBox.critical(self, "Import failed", str(error))
            return
        required = {
            BoreholeDataType.COLLARS: {"borehole_id", "collar_x", "collar_y", "collar_z", "final_depth"},
            BoreholeDataType.SURVEYS: {"hole_id", "measured_depth", "azimuth", "dip"},
            BoreholeDataType.FRACTURES: {"hole_id", "depth", "dip"},
            BoreholeDataType.RQD: {"hole_id", "from_depth", "to_depth", "rqd"},
            BoreholeDataType.DOMAIN_INTERVALS: {"hole_id", "from_depth", "to_depth", "domain_id"},
        }.get(data_type, set())
        missing = sorted(required - set(dataframe.columns))
        if missing:
            QMessageBox.critical(self, "Import failed", f"Missing required fields: {', '.join(missing)}")
            return
        result = self._repository.import_dataframe(
            data_type,
            raw_dataframe if data_type == BoreholeDataType.COLLARS else dataframe,
            source_file=str(Path(path)),
            mode=self._mode_combo.currentData(),
            modification_source="staged_import",
            field_mapping=rename if data_type == BoreholeDataType.COLLARS else None,
        )
        self._results.append(dict(result))
        self._staged_changes = self._staged_changes or result.changed
        orientation = (
            f", total fracture rows={result.get('total_fracture_rows', 0)}, "
            f"full orientation={result.get('full_orientation', 0)}, dip only={result.get('dip_only', 0)}, "
            f"excluded/error={result.get('excluded_error', 0)}"
            if result.get("data_type") == BoreholeDataType.FRACTURES
            else ""
        )
        self._status.setText(
            f"Staged raw={result['raw']}, formal={result['formal']}, excluded={result['excluded']}, "
            f"pending={result['pending']}, duplicates={result['duplicates']}{orientation}"
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
