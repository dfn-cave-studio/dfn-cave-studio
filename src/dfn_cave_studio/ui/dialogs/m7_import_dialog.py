"""M7 unified data import wizard — imports all data types into the project.

Key fix: collars are loaded FIRST so fractures/surveys/rqd can attach to
existing boreholes.  All five data types are accumulated across separate
import actions and written to the project on dialog accept.
"""

import copy
import os
from pathlib import Path
from typing import Optional, Dict, Any

from dfn_cave_studio.ui.qt_adapter import (
    Qt,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QDialogButtonBox,
    QGroupBox,
    QWidget,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QSplitter,
    QTextEdit,
)
import pandas as pd

from dfn_cave_studio.borehole.borehole_importer import (
    BoreholeImporter,
    STANDARD_COLLAR_FIELDS,
    STANDARD_SURVEY_FIELDS,
    STANDARD_FRACTURE_FIELDS,
    STANDARD_RQD_FIELDS,
)
from dfn_cave_studio.services.import_service import (
    STANDARD_DOMAIN_INTERVAL_FIELDS,
    read_input_table,
)
from dfn_cave_studio.services.m7_state import (
    get_excluded_records,
    set_raw_domain_intervals,
    set_raw_fractures,
    set_raw_rqd,
    set_raw_surveys,
    set_excluded_records,
)
from dfn_cave_studio.models.borehole import BoreholeCollection

# ── Required field definitions per data type ───────────────────────────

# Required fields MUST use the same standard names as BoreholeImporter/FIELD_DEFS.
# After mapping, columns are renamed to these standard names before validation.
REQUIRED_FIELDS = {
    "collars": {"borehole_id", "collar_x", "collar_y", "collar_z", "final_depth"},
    "surveys": {"hole_id", "measured_depth", "azimuth", "dip"},
    "fractures": {"hole_id", "depth", "dip"},
    "rqd": {"hole_id", "from_depth", "to_depth", "rqd"},
    "domain_intervals": {"hole_id", "from_depth", "to_depth", "domain_id"},
}

FIELD_DEFS = {
    "collars": STANDARD_COLLAR_FIELDS,
    "surveys": STANDARD_SURVEY_FIELDS,
    "fractures": STANDARD_FRACTURE_FIELDS,
    "rqd": STANDARD_RQD_FIELDS,
    "domain_intervals": STANDARD_DOMAIN_INTERVAL_FIELDS,
}

# Alias: extra standard-field keys that some callers use
_REQUIRED_ALIASES = {
    "hole_id": "borehole_id",
    "easting": "collar_x",
    "northing": "collar_y",
    "elevation": "collar_z",
    "total_depth": "final_depth",
}

DATA_TYPE_LABELS = [
    ("collars", "Collar data (collars)"),
    ("surveys", "Survey data (surveys)"),
    ("fractures", "Fracture observations (fractures)"),
    ("rqd", "RQD intervals (rqd)"),
    ("domain_intervals", "Domain intervals"),
]


class M7ImportDialog(QDialog):
    """Wizard for importing all M7 data types.

    Collars are imported first (via BoreholeImporter).  Subsequent
    surveys/fractures/rqd are attached to the existing borehole
    collection.  Domain intervals are stored as DataFrame.

    On accept, all accumulated data is written to the project.
    """

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = project
        self._importer = BoreholeImporter()
        self._collection: Optional[BoreholeCollection] = (
            copy.deepcopy(project.borehole_collection) if project.borehole_collection is not None else None
        )
        self._preview_df: Optional[pd.DataFrame] = None
        self._file_paths: Dict[str, str] = {}
        # Accumulated import results
        self._imported: Dict[str, Any] = {}
        # RQD DataFrame
        self._rqd_df: Optional[pd.DataFrame] = None
        # Domain intervals
        self._domain_df: Optional[pd.DataFrame] = None
        self._processed_fracture_files: set[str] = set()
        self._raw_fractures: Optional[pd.DataFrame] = None
        self._fracture_exclusions: list[dict[str, Any]] = []
        self._committed_changes = False

        self.setWindowTitle("M7 Data Import Wizard")
        self.resize(900, 650)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Top: file type + path
        top = QGroupBox("Select Data Files")
        top_layout = QFormLayout(top)

        self._type_combo = QComboBox()
        for key, label in DATA_TYPE_LABELS:
            self._type_combo.addItem(label, key)
        top_layout.addRow("Data type:", self._type_combo)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select CSV or XLSX file...")
        path_row.addWidget(self._path_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(browse_btn)
        top_layout.addRow("File:", path_row)

        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("Encoding:"))
        self._encoding_combo = QComboBox()
        self._encoding_combo.addItems(["utf-8", "latin-1", "cp1252"])
        csv_row.addWidget(self._encoding_combo)
        csv_row.addWidget(QLabel("Delimiter:"))
        self._delimiter_combo = QComboBox()
        self._delimiter_combo.addItems([",", ";", "\\t", "|"])
        csv_row.addWidget(self._delimiter_combo)
        csv_row.addWidget(QLabel("Sheet:"))
        self._sheet_edit = QLineEdit("Sheet")
        self._sheet_edit.setMaximumWidth(80)
        csv_row.addWidget(self._sheet_edit)
        top_layout.addRow("Options:", csv_row)

        preview_btn = QPushButton("Preview & Map Fields")
        preview_btn.clicked.connect(self._on_preview)
        top_layout.addRow(preview_btn)
        layout.addWidget(top)

        # Middle: preview + field mapping
        middle = QSplitter(Qt.Orientation.Horizontal)
        self._preview_table = QTableWidget()
        self._preview_table.setAlternatingRowColors(True)
        middle.addWidget(self._preview_table)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Field Mapping (editable):"))
        self._mapping_table = QTableWidget(0, 3)
        self._mapping_table.setHorizontalHeaderLabels(["Standard Field", "Source Column", "Required"])
        self._mapping_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rl.addWidget(self._mapping_table)
        rl.addWidget(QLabel("<i>Double-click Source Column cell to edit.</i>"))
        middle.addWidget(right)
        layout.addWidget(middle)

        # Status
        self._status_label = QLabel("Ready.  Import collars first, then other files.")
        layout.addWidget(self._status_label)

        # Import stats summary
        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setMaximumHeight(100)
        layout.addWidget(self._summary_text)

        # Buttons
        self._button_box = QDialogButtonBox()
        self._import_btn = QPushButton("Import This File")
        self._import_btn.clicked.connect(self._on_import)
        self._button_box.addButton(self._import_btn, QDialogButtonBox.ButtonRole.ActionRole)
        self._button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self._button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    # ── Browse ───────────────────────────────────────────────────────

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "Data Files (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)
            dtype = self._type_combo.currentData()
            self._file_paths[dtype] = path

    # ── Preview ──────────────────────────────────────────────────────

    def _on_preview(self):
        path = self._path_edit.text()
        if not path:
            return
        delim = self._delimiter_combo.currentText().replace("\\t", "\t")
        enc = self._encoding_combo.currentText()
        sheet = self._sheet_edit.text() or "Sheet"
        try:
            df = read_input_table(path, encoding=enc, delimiter=delim, sheet_name=sheet, nrows=50)
        except (OSError, ValueError, ImportError, KeyError) as e:
            self._status_label.setText(f"Read error: {e}")
            return
        self._preview_df = df
        self._status_label.setText(f"Preview: {len(df)} rows, {len(df.columns)} columns")
        self._populate_preview(df)
        self._populate_mapping(df)

    def _populate_preview(self, df: pd.DataFrame):
        self._preview_table.setRowCount(min(len(df), 50))
        self._preview_table.setColumnCount(len(df.columns))
        self._preview_table.setHorizontalHeaderLabels(list(df.columns))
        for i in range(min(len(df), 50)):
            for j, col in enumerate(df.columns):
                val = str(df.iloc[i, j]) if pd.notna(df.iloc[i, j]) else ""
                self._preview_table.setItem(i, j, QTableWidgetItem(val))

    def _populate_mapping(self, df: pd.DataFrame):
        dtype = self._type_combo.currentData()
        field_defs = FIELD_DEFS.get(dtype, {})
        required = REQUIRED_FIELDS.get(dtype, set())

        self._mapping_table.setRowCount(len(field_defs))
        row = 0
        for std_name, aliases in field_defs.items():
            self._mapping_table.setItem(row, 0, QTableWidgetItem(std_name))
            # Auto-detect: try aliases first, then exact match
            found = ""
            for alias in aliases:
                if alias in df.columns:
                    found = alias
                    break
            if not found and std_name in df.columns:
                found = std_name
            src_item = QTableWidgetItem(found)
            self._mapping_table.setItem(row, 1, src_item)
            req_text = "Yes" if std_name in required else ""
            self._mapping_table.setItem(row, 2, QTableWidgetItem(req_text))
            row += 1

    # ── Import ────────────────────────────────────────────────────────

    def _build_rename_map(self) -> dict:
        """Build rename map from mapping table: {source_column: standard_field}.

        Mapping direction: standard_field → source_column (in table).
        Applied as: df.rename(columns={source: standard}).
        """
        rename = {}
        for i in range(self._mapping_table.rowCount()):
            std = self._mapping_table.item(i, 0).text() if self._mapping_table.item(i, 0) else ""
            src = self._mapping_table.item(i, 1).text() if self._mapping_table.item(i, 1) else ""
            if std and src:
                rename[src] = std
        return rename

    def _check_required_fields(self, df_columns: set, dtype: str) -> list:
        """Return list of missing required fields, resolving aliases."""
        required = REQUIRED_FIELDS.get(dtype, set())
        # Expand df_columns with known aliases so "hole_id" counts as "borehole_id"
        expanded = set(df_columns)
        for alias, canonical in _REQUIRED_ALIASES.items():
            if alias in df_columns:
                expanded.add(canonical)
        return [f for f in required if f not in expanded]

    def _apply_mapping_and_validate(self, df, dtype: str, rename_map: dict):
        """Apply column rename, validate required fields. Returns (mapped_df, error_msg)."""
        if rename_map:
            # Only rename columns that exist in the DataFrame
            applicable = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=applicable)

        missing = self._check_required_fields(set(df.columns), dtype)
        if missing:
            return None, (
                f"Missing required fields for {dtype}: {', '.join(missing)}\n\n"
                f"Available columns: {', '.join(df.columns)}\n\n"
                f"Please map the required fields in the Field Mapping table."
            )
        return df, None

    def _on_import(self):
        dtype = self._type_combo.currentData()
        path = self._path_edit.text()
        if not path:
            QMessageBox.warning(self, "No File", "Please select a file.")
            return
        self._status_label.setText(f"Importing {dtype}...")

        rename_map = self._build_rename_map()

        delim = self._delimiter_combo.currentText().replace("\\t", "\t")
        enc = self._encoding_combo.currentText()
        sheet = self._sheet_edit.text() or "Sheet"

        try:
            if dtype == "collars":
                self._import_collars(path, enc, delim, rename_map, sheet)
            elif dtype == "surveys":
                self._import_surveys(path, enc, delim, rename_map, sheet)
            elif dtype == "fractures":
                self._import_fractures(path, enc, delim, rename_map, sheet)
            elif dtype == "rqd":
                self._import_rqd(path, enc, delim, rename_map, sheet)
            elif dtype == "domain_intervals":
                self._import_domain_intervals(path, enc, delim, rename_map, sheet)
        except (OSError, ValueError, ImportError, KeyError, TypeError) as e:
            self._status_label.setText(f"Import error: {e}")
            QMessageBox.critical(self, "Import Error", str(e))

    def _import_collars(self, path, enc, delim, rename_map, sheet):
        """Import collars — creates the borehole collection."""
        df = read_input_table(path, encoding=enc, delimiter=delim, sheet_name=sheet)
        mapped_df, err = self._apply_mapping_and_validate(df, "collars", rename_map)
        if err:
            QMessageBox.critical(self, "Field Mapping Error", err)
            return

        # Write mapped CSV to temp file so BoreholeImporter can read with
        # standard column names (it has its own field detection)
        import tempfile

        # Use mkstemp for safe cross-platform temp file creation
        fd, tmp_path = tempfile.mkstemp(suffix=".csv")
        try:
            os.close(fd)  # Close the fd so to_csv can open the file
            mapped_df.to_csv(tmp_path, index=False)
            result = self._importer.import_all(collar_path=tmp_path)
            if result.collection:
                self._collection = result.collection
            self._imported["collars"] = {
                "imported": len(result.collection) if result.collection else 0,
                "skipped": result.rows_skipped,
                "errors": len(result.errors),
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        self._add_summary("collars", self._imported["collars"])
        self._status_label.setText(
            f"Collars: {self._imported['collars']['imported']} imported, "
            f"{self._imported['collars']['errors']} errors"
        )

    def _import_surveys(self, path, enc, delim, rename_map, sheet):
        """Import surveys — store ALL raw rows for later cleaning."""
        coll = self._collection or self._project.borehole_collection
        if coll is None:
            QMessageBox.warning(self, "No Collars", "Import collars first.")
            return
        self._collection = coll
        df = read_input_table(path, encoding=enc, delimiter=delim, sheet_name=sheet)
        mapped_df, err = self._apply_mapping_and_validate(df, "surveys", rename_map)
        if err:
            QMessageBox.critical(self, "Field Mapping Error", err)
            return
        self._raw_surveys = mapped_df
        self._imported["surveys"] = {"imported": len(mapped_df), "skipped": 0, "errors": 0}
        self._add_summary("surveys", self._imported["surveys"])
        self._status_label.setText(f"Surveys: {len(mapped_df)} raw rows stored (cleaning will process)")

    def _import_fractures(self, path, enc, delim, rename_map, sheet):
        """Import fractures — attach to boreholes in current project."""
        normalized_path = str(Path(path).resolve())
        if normalized_path in self._processed_fracture_files:
            self._status_label.setText(f"Fractures: {Path(path).name} already imported in this dialog")
            return

        coll = self._collection
        if coll is None:
            QMessageBox.warning(self, "No Collars", "Import collars first.")
            return
        self._collection = coll

        raw_df = read_input_table(path, encoding=enc, delimiter=delim, sheet_name=sheet)
        df, err = self._apply_mapping_and_validate(raw_df, "fractures", rename_map)
        if err:
            QMessageBox.critical(self, "Field Mapping Error", err)
            return
        self._raw_fractures = df.copy()
        self._fracture_exclusions = []

        bh_map = {bh.borehole_id: bh for bh in coll}
        imported = 0
        errors = 0
        first_errors = []
        from dfn_cave_studio.models.borehole import FractureObservation
        from dfn_cave_studio.models.enums import FractureType

        existing = {
            (
                bh.borehole_id,
                float(obs.measured_depth),
                obs.dip_direction,
                float(obs.dip),
                obs.set_id,
                None if obs.aperture is None else float(obs.aperture),
            )
            for bh in coll
            for obs in bh.fracture_observations
        }
        for idx, row in df.iterrows():
            try:
                bh_id = str(row.get("hole_id", "")).strip()
                if bh_id not in bh_map:
                    errors += 1
                    if len(first_errors) < 5:
                        first_errors.append(f"Row {idx}: unknown borehole '{bh_id}'")
                    continue
                depth = float(row.get("depth", row.get("measured_depth", 0)))
                raw_direction = row.get("dip_direction")
                dd = None if pd.isna(raw_direction) or str(raw_direction).strip().lower() in {"", "na", "n/a", "null", "none"} else float(raw_direction)
                dip = float(row.get("dip", 0))
                if depth > bh_map[bh_id].collar.final_depth:
                    reason = (
                        f"Fracture depth {depth} exceeds borehole " f"total depth {bh_map[bh_id].collar.final_depth}"
                    )
                    errors += 1
                    self._fracture_exclusions.append(
                        self._importer._fracture_exclusion_record(
                            row,
                            idx,
                            Path(path).name,
                            bh_id,
                            "depth",
                            row.get("depth", row.get("measured_depth")),
                            reason,
                        )
                    )
                    if len(first_errors) < 5:
                        first_errors.append(f"Row {idx}: {reason}")
                    continue
                if dip < 0.0 or dip > 90.0:
                    reason = f"Dip {dip} is outside the valid range [0, 90]"
                    errors += 1
                    self._fracture_exclusions.append(
                        self._importer._fracture_exclusion_record(
                            row,
                            idx,
                            Path(path).name,
                            bh_id,
                            "dip",
                            row.get("dip"),
                            reason,
                        )
                    )
                    if len(first_errors) < 5:
                        first_errors.append(f"Row {idx}: {reason}")
                    continue
                set_id = None
                raw_set = row.get("set_id")
                if raw_set is not None and not (isinstance(raw_set, float) and pd.isna(raw_set)):
                    try:
                        set_id = int(float(raw_set))
                    except (ValueError, TypeError):
                        errors += 1
                        reason = f"set_id '{raw_set}' is not a valid integer"
                        self._fracture_exclusions.append(
                            self._importer._fracture_exclusion_record(
                                row,
                                idx,
                                Path(path).name,
                                bh_id,
                                "set_id",
                                raw_set,
                                reason,
                            )
                        )
                        if len(first_errors) < 5:
                            first_errors.append(f"Row {idx}: {reason}")
                        continue
                aperture = float(row["aperture"]) if pd.notna(row.get("aperture")) else None
                signature = (bh_id, depth, dd, dip, set_id, aperture)
                if signature in existing:
                    continue
                obs = FractureObservation(
                    borehole_id=bh_id,
                    measured_depth=depth,
                    dip_direction=dd,
                    dip=dip,
                    aperture=aperture,
                    fracture_type=FractureType.JOINT,
                    confidence=float(row.get("confidence", 1.0)) if pd.notna(row.get("confidence")) else 1.0,
                    set_id=set_id,
                )
                bh_map[bh_id].fracture_observations.append(obs)
                existing.add(signature)
                imported += 1
            except (KeyError, TypeError, ValueError) as e:
                errors += 1
                if len(first_errors) < 5:
                    first_errors.append(f"Row {idx}: {type(e).__name__}: {e}")
        self._imported["fractures"] = {
            "raw": len(df),
            "imported": imported,
            "skipped": errors,
            "errors": errors,
        }
        self._processed_fracture_files.add(normalized_path)
        self._add_summary("fractures", self._imported["fractures"])
        self._status_label.setText(
            f"Fractures: {len(df)} raw, {imported} formally imported, " f"{errors} errors/excluded"
        )
        if first_errors:
            self._status_label.setText(self._status_label.text() + f"  First errors: {'; '.join(first_errors[:3])}")

    def _import_rqd(self, path, enc, delim, rename_map, sheet):
        """Import RQD intervals as DataFrame."""
        coll = self._collection or self._project.borehole_collection
        if coll is None:
            QMessageBox.warning(self, "No Collars", "Import collars first.")
            return
        self._collection = coll
        df = read_input_table(path, encoding=enc, delimiter=delim, sheet_name=sheet)
        mapped_df, err = self._apply_mapping_and_validate(df, "rqd", rename_map)
        if err:
            QMessageBox.critical(self, "Field Mapping Error", err)
            return
        self._rqd_df = mapped_df
        self._imported["rqd"] = {"imported": len(mapped_df), "skipped": 0, "errors": 0}
        self._add_summary("rqd", self._imported["rqd"])
        self._status_label.setText(f"RQD: {len(mapped_df)} rows imported")

    def _import_domain_intervals(self, path, enc, delim, rename_map, sheet):
        """Import domain intervals as DataFrame."""
        df = read_input_table(path, encoding=enc, delimiter=delim, sheet_name=sheet)
        mapped_df, err = self._apply_mapping_and_validate(df, "domain_intervals", rename_map)
        if err:
            QMessageBox.critical(self, "Field Mapping Error", err)
            return
        self._domain_df = mapped_df
        self._imported["domain_intervals"] = {"imported": len(mapped_df), "skipped": 0, "errors": 0}
        self._add_summary("domain_intervals", self._imported["domain_intervals"])
        self._status_label.setText(f"Domain intervals: {len(mapped_df)} rows imported")

    def _add_summary(self, dtype, counts):
        lines = []
        for key, label in DATA_TYPE_LABELS:
            info = self._imported.get(key, {})
            if info:
                lines.append(
                    f"  {label}: "
                    + (f"{info['raw']} raw, " if "raw" in info else "")
                    + f"{info.get('imported',0)} imported, "
                    f"{info.get('skipped',0)} skipped, {info.get('errors',0)} errors"
                )
            else:
                lines.append(f"  {label}: not yet imported")
        self._summary_text.setText("\n".join(lines))

    # ── Accept ────────────────────────────────────────────────────────

    def _on_accept(self):
        """Write all accumulated imports into the project."""
        self._committed_changes = any(info.get("imported", 0) > 0 for info in self._imported.values())
        if self._collection is not None and self._committed_changes:
            self._project.borehole_collection = self._collection

        # Raw surveys DataFrame (ALL rows, pre-cleaning)
        if hasattr(self, "_raw_surveys") and self._raw_surveys is not None:
            set_raw_surveys(self._project, self._raw_surveys)
        if self._raw_fractures is not None:
            set_raw_fractures(self._project, self._raw_fractures)
            existing_exclusions = {record.get("issue_id"): record for record in get_excluded_records(self._project)}
            for record in self._fracture_exclusions:
                existing_exclusions[record["issue_id"]] = record
            set_excluded_records(
                self._project,
                list(existing_exclusions.values()),
            )

        # Raw RQD DataFrame
        if self._rqd_df is not None:
            set_raw_rqd(self._project, self._rqd_df)

        # Raw domain intervals DataFrame
        if self._domain_df is not None:
            set_raw_domain_intervals(self._project, self._domain_df)
        self.accept()

    # ── Public accessors ──────────────────────────────────────────────

    def get_collection(self) -> Optional[BoreholeCollection]:
        return self._collection

    def get_import_summary(self) -> Dict[str, Any]:
        return dict(self._imported)

    def get_rqd_dataframe(self) -> Optional[pd.DataFrame]:
        return self._rqd_df

    def get_domain_dataframe(self) -> Optional[pd.DataFrame]:
        return self._domain_df

    @property
    def committed_changes(self) -> bool:
        """Whether OK committed at least one imported record."""
        return self._committed_changes
