"""Data import dialog for borehole collar, survey, and fracture data.

Provides file selection, field mapping with preview, validation stats,
and mandatory field enforcement. Supports CSV and LAS formats.
"""

from pathlib import Path
from typing import Optional, Dict, List

from dfn_cave_studio.ui.qt_adapter import (
    Qt, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QTextEdit, QDialogButtonBox, QTabWidget, QWidget,
    QMessageBox, QProgressBar,
)
from dfn_cave_studio.borehole.borehole_importer import (
    BoreholeImporter, STANDARD_COLLAR_FIELDS, STANDARD_SURVEY_FIELDS,
    STANDARD_FRACTURE_FIELDS, STANDARD_RQD_FIELDS, ImportResult,
)
from dfn_cave_studio.models.borehole import BoreholeCollection


class DataImportDialog(QDialog):
    """Dialog for importing borehole and fracture data.

    Three tabs: Collar (required), Survey (optional), Fractures (optional).
    Each tab has file selection, field mapping, preview, and validation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Borehole & Fracture Data")
        self.resize(900, 650)
        self._importer = BoreholeImporter()
        self._collection: Optional[BoreholeCollection] = None

        self._collar_path: str = ""
        self._survey_path: str = ""
        self._fractures_path: str = ""

        self._init_ui()
        self._update_stats()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Tab widget
        self._tabs = QTabWidget()

        # Collar tab
        collar_tab = QWidget()
        cl = QVBoxLayout(collar_tab)
        cl.addWidget(self._make_file_row("Collar CSV:", "collar"))
        self._collar_preview = QTextEdit()
        self._collar_preview.setReadOnly(True)
        self._collar_preview.setMaximumHeight(120)
        cl.addWidget(QLabel("Preview (first 10 rows):"))
        cl.addWidget(self._collar_preview)
        self._collar_stats = QLabel("No file selected")
        cl.addWidget(self._collar_stats)
        self._tabs.addTab(collar_tab, "Collar (Required)")

        # Survey tab
        survey_tab = QWidget()
        sl = QVBoxLayout(survey_tab)
        sl.addWidget(self._make_file_row("Survey CSV (optional):", "survey"))
        self._survey_preview = QTextEdit()
        self._survey_preview.setReadOnly(True)
        self._survey_preview.setMaximumHeight(120)
        sl.addWidget(QLabel("Preview (first 10 rows):"))
        sl.addWidget(self._survey_preview)
        self._tabs.addTab(survey_tab, "Survey (Optional)")

        # Fractures tab
        frac_tab = QWidget()
        fl = QVBoxLayout(frac_tab)
        fl.addWidget(self._make_file_row("Fractures CSV (optional):", "fractures"))
        self._frac_preview = QTextEdit()
        self._frac_preview.setReadOnly(True)
        self._frac_preview.setMaximumHeight(120)
        fl.addWidget(QLabel("Preview (first 10 rows):"))
        fl.addWidget(self._frac_preview)
        self._tabs.addTab(frac_tab, "Fractures (Optional)")

        layout.addWidget(self._tabs)

        # Overall stats
        self._overall_stats = QLabel("Ready to import")
        layout.addWidget(self._overall_stats)

        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        self._ok_btn = btn_box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setText("Import")
        layout.addWidget(btn_box)

    def _make_file_row(self, label: str, key: str) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        le = QLineEdit()
        le.setPlaceholderText("Select file...")
        le.setReadOnly(True)
        setattr(self, f"_{key}_path_le", le)
        row.addWidget(le, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file(key))
        row.addWidget(btn)
        return w

    def _browse_file(self, key: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {key} file", "",
            "CSV Files (*.csv);;LAS Files (*.las);;Excel Files (*.xlsx *.xls);;All Files (*)",
        )
        if not path:
            return
        setattr(self, f"_{key}_path", path)
        le = getattr(self, f"_{key}_path_le")
        le.setText(path)
        self._update_preview(key, path)
        self._update_stats()

    def _update_preview(self, key: str, path: str) -> None:
        try:
            import pandas as pd
            df = pd.read_csv(path, nrows=10)
            preview = getattr(self, f"_{key}_preview")
            preview.setText(df.head(10).to_string())
        except Exception as e:
            preview = getattr(self, f"_{key}_preview")
            preview.setText(f"Error reading file: {e}")

    def _update_stats(self) -> None:
        has_collar = bool(self._collar_path)
        if has_collar:
            try:
                import pandas as pd
                df = pd.read_csv(self._collar_path)
                n_rows = len(df)
                self._collar_stats.setText(f"Found: {n_rows} rows. Columns: {', '.join(df.columns[:8])}")
            except Exception as e:
                self._collar_stats.setText(f"Error: {e}")
        else:
            self._collar_stats.setText("No file selected")
        self._ok_btn.setEnabled(has_collar)

    def _on_accept(self) -> None:
        if not self._collar_path:
            QMessageBox.warning(self, "Missing Data", "A collar CSV file is required.")
            return

        result = self._importer.import_all(
            collar_path=self._collar_path,
            survey_path=self._survey_path if self._survey_path else None,
            fractures_path=self._fractures_path if self._fractures_path else None,
        )

        if result.success:
            self._collection = result.collection
            n_bh = len(self._collection) if self._collection else 0
            n_frac = sum(bh.observed_fracture_count for bh in (self._collection or []))
            QMessageBox.information(
                self, "Import Complete",
                f"Imported {n_bh} boreholes with {n_frac} fracture observations.\n"
                f"Warnings: {len(result.warnings)}"
            )
            self.accept()
        else:
            QMessageBox.critical(
                self, "Import Failed",
                f"Errors ({len(result.errors)}):\n" + "\n".join(result.errors[:10])
            )

    def get_collection(self) -> Optional[BoreholeCollection]:
        return self._collection
