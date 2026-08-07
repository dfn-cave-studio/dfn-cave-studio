"""
Borehole data import and validation service.

Handles CSV/Excel import of borehole data with:
  - Field mapping (user column names → standard names)
  - Unit checking and conversion
  - Missing value detection
  - Duplicate record detection
  - Angle validation (dip direction, dip, azimuth ranges)
  - Data quality report generation
  - Anomaly flagging and export

References:
  - SCIENTIFIC_SPEC.md Section 6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import pandas as pd

from dfn_cave_studio.models.borehole import (
    BoreholeCollection,
    Borehole,
    Collar,
    SurveyStation,
    BoreholeSurvey,
    FractureObservation,
    RQDInterval,
)
from dfn_cave_studio.models.enums import FractureType

# =============================================================================
# Import Result
# =============================================================================


@dataclass
class ImportResult:
    """Result of a borehole data import operation."""

    success: bool
    collection: Optional[BoreholeCollection] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rows_imported: int = 0
    rows_skipped: int = 0
    fracture_rows_raw: int = 0
    fracture_rows_imported: int = 0
    fracture_rows_excluded: int = 0
    fracture_exclusions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return len(self.errors) + len(self.warnings)


# =============================================================================
# Standard Field Names
# =============================================================================

STANDARD_COLLAR_FIELDS = {
    "borehole_id": ["borehole_id", "hole_id", "bh_id", "borehole", "hole", "id"],
    "collar_x": ["collar_x", "x", "easting", "east", "x_coord"],
    "collar_y": ["collar_y", "y", "northing", "north", "y_coord"],
    "collar_z": ["collar_z", "z", "elevation", "elev", "z_coord", "rl"],
    "azimuth": ["azimuth", "az", "bearing", "azm"],
    "dip": ["dip", "inclination", "incl", "dip_angle"],
    "final_depth": ["final_depth", "total_depth", "depth", "length", "eoh"],
}

STANDARD_SURVEY_FIELDS = {
    "borehole_id": ["borehole_id", "hole_id", "bh_id"],
    "measured_depth": ["measured_depth", "depth", "md", "downhole_depth"],
    "azimuth": ["azimuth", "az", "bearing"],
    "dip": ["dip", "inclination", "incl"],
}

STANDARD_FRACTURE_FIELDS = {
    "borehole_id": ["borehole_id", "hole_id", "bh_id"],
    "measured_depth": ["measured_depth", "depth", "md", "from"],
    "dip_direction": ["dip_direction", "dipdir", "dd", "dip_dir"],
    "dip": ["dip", "dip_angle"],
    "aperture": ["aperture", "aperture_mm", "opening"],
    "filling": ["filling", "infill", "fill", "filling_type"],
    "fracture_type": ["fracture_type", "type", "frac_type", "discontinuity_type"],
    "confidence": ["confidence", "certainty", "quality"],
    "set_id": ["set_id", "set", "joint_set_id", "family", "fracture_set_id"],
}

STANDARD_RQD_FIELDS = {
    "borehole_id": ["borehole_id", "hole_id", "bh_id"],
    "from_depth": ["from_depth", "from", "top", "start_depth"],
    "to_depth": ["to_depth", "to", "bottom", "end_depth"],
    "rqd": ["rqd", "rqd_value", "rqd_pct", "rqd_percent"],
    "core_recovery": ["core_recovery", "recovery", "rec", "cr"],
}


# =============================================================================
# Borehole Importer
# =============================================================================


class BoreholeImporter:
    """Import borehole data from CSV/Excel files with validation.

    Usage:
        importer = BoreholeImporter()
        result = importer.import_all(
            collar_path="collars.csv",
            survey_path="surveys.csv",       # optional
            fractures_path="fractures.csv",   # optional
            rqd_path="rqd.csv",               # optional
        )
        if result.success:
            collection = result.collection
    """

    def __init__(self):
        self._errors: List[str] = []
        self._warnings: List[str] = []

    # ── Main Import ──────────────────────────────────────────────────────

    def import_all(
        self,
        collar_path: str,
        survey_path: Optional[str] = None,
        fractures_path: Optional[str] = None,
        rqd_path: Optional[str] = None,
        field_map: Optional[Dict[str, str]] = None,
    ) -> ImportResult:
        """Import all borehole data from CSV/Excel files.

        Args:
            collar_path: Path to collar CSV/Excel.
            survey_path: Path to survey CSV/Excel (optional).
            fractures_path: Path to fracture observations (optional).
            rqd_path: Path to RQD intervals (optional).
            field_map: User-specified field name mapping.

        Returns:
            ImportResult with collection and any issues.
        """
        self._errors = []
        self._warnings = []
        self._rows_fractures_raw = 0
        self._rows_fractures_imported = 0
        self._rows_fractures_skipped = 0
        self._fracture_exclusions: List[Dict[str, Any]] = []

        try:
            collar_df = self._read_file(collar_path)
            if collar_df is None:
                return ImportResult(success=False, errors=self._errors, warnings=self._warnings)

            # Map field names
            collar_df = self._map_fields(collar_df, STANDARD_COLLAR_FIELDS, field_map)

            # Import collars
            boreholes = self._import_collars(collar_df)
            if not boreholes:
                return ImportResult(success=False, errors=self._errors, warnings=self._warnings)

            # Import survey data
            if survey_path:
                survey_df = self._read_file(survey_path)
                if survey_df is not None:
                    survey_df = self._map_fields(survey_df, STANDARD_SURVEY_FIELDS, field_map)
                    self._import_surveys(survey_df, boreholes)

            # Import fracture observations
            if fractures_path:
                frac_df = self._read_file(fractures_path)
                if frac_df is not None:
                    frac_df = self._map_fields(frac_df, STANDARD_FRACTURE_FIELDS, field_map)
                    self._import_fractures(frac_df, boreholes, Path(fractures_path).name)

            # Import RQD
            if rqd_path:
                rqd_df = self._read_file(rqd_path)
                if rqd_df is not None:
                    rqd_df = self._map_fields(rqd_df, STANDARD_RQD_FIELDS, field_map)
                    self._import_rqd(rqd_df, boreholes)

            # Build collection
            collection = BoreholeCollection(boreholes=boreholes)

            # Run validation
            validation_errors = collection.validate_all()
            for ve in validation_errors:
                if ve.severity == "error":
                    self._errors.append(str(ve))
                else:
                    self._warnings.append(str(ve))

            # Count fracture observation rows
            frac_imported = getattr(self, "_rows_fractures_imported", 0)
            frac_skipped = getattr(self, "_rows_fractures_skipped", 0)
            total_rows = len(boreholes) + frac_imported

            return ImportResult(
                success=len(self._errors) == 0,
                collection=collection,
                errors=self._errors,
                warnings=self._warnings,
                rows_imported=total_rows,
                rows_skipped=frac_skipped,
                fracture_rows_raw=self._rows_fractures_raw,
                fracture_rows_imported=frac_imported,
                fracture_rows_excluded=frac_skipped,
                fracture_exclusions=list(self._fracture_exclusions),
            )

        except Exception as e:
            self._errors.append(f"Import failed: {e}")
            return ImportResult(success=False, errors=self._errors, warnings=self._warnings)

    # ── File Reading ─────────────────────────────────────────────────────

    def _read_file(self, path: str) -> Optional[pd.DataFrame]:
        """Read CSV, Excel, or LAS file.

        LAS (Log ASCII Standard) is a common format in geotechnical and
        petroleum borehole logging. Basic LAS 2.0/3.0 support is provided
        via the `lasio` library (optional dependency).
        """
        path = Path(path)
        if not path.exists():
            self._errors.append(f"File not found: {path}")
            return None

        try:
            suffix = path.suffix.lower()
            if suffix in (".xlsx", ".xls"):
                return pd.read_excel(path)
            elif suffix == ".las":
                return self._read_las(path)
            else:
                return pd.read_csv(path)
        except Exception as e:
            self._errors.append(f"Failed to read {path.name}: {e}")
            return None

    def _read_las(self, path: Path) -> Optional[pd.DataFrame]:
        """Read LAS (Log ASCII Standard) borehole log file.

        Tries `lasio` first (full LAS 2.0/3.0 parser), falls back to a
        minimal built-in parser for basic ~V, ~W, ~A sections.

        Args:
            path: Path to .las file.

        Returns:
            DataFrame with columns from LAS curves, or None on failure.
        """
        # Try lasio (optional dependency)
        try:
            import lasio

            las = lasio.read(str(path))
            data = {}
            for curve in las.curves:
                data[curve.mnemonic] = las.curves[curve.mnemonic].data
            if "DEPT" in data or "DEPTH" in data:
                data.setdefault("measured_depth", data.pop("DEPT", data.get("DEPTH")))
            index_name = las.index_mnemonic if las.index_mnemonic else None
            if index_name and index_name in data:
                depth_data = data.pop(index_name)
                data["measured_depth"] = data.get("measured_depth", depth_data)
            return pd.DataFrame(data)
        except ImportError:
            pass  # Fall through to built-in parser
        except Exception as e:
            self._warnings.append(f"lasio failed for {path.name}: {e} — trying built-in parser")

        # Built-in minimal LAS parser
        try:
            return self._parse_las_minimal(path)
        except Exception as e:
            self._errors.append(f"LAS parse failed for {path.name}: {e}")
            return None

    def _parse_las_minimal(self, path: Path) -> pd.DataFrame:
        """Minimal built-in LAS parser for basic well log files.

        Handles ~V (version), ~W (well info), ~C (curve), ~A (ASCII data).
        """
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        section = None
        curves = []
        data_start = -1
        well_info = {}

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("~V"):
                section = "version"
            elif stripped.startswith("~W"):
                section = "well"
            elif stripped.startswith("~C"):
                section = "curve"
            elif stripped.startswith("~A") or stripped.startswith("~ASCII"):
                section = "data"
                data_start = i + 1
                break
            elif section == "well":
                if "." in stripped and not stripped.startswith("#"):
                    parts = stripped.split(".", 1)
                    if len(parts) == 2:
                        key = parts[0].strip().rstrip(".")
                        val = parts[1].strip().split(":")[0].strip()
                        well_info[key] = val
            elif section == "curve":
                if stripped and not stripped.startswith("#"):
                    parts = stripped.split(".", 1)
                    if len(parts) >= 1:
                        curves.append(parts[0].strip())

        if data_start < 0 or not curves:
            raise ValueError("LAS file missing ~A (data) section or ~C (curve) definitions")

        # Parse data rows
        rows = []
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split()
            if len(values) >= len(curves):
                rows.append(
                    [float(v) if v.replace(".", "").replace("-", "").isdigit() else v for v in values[: len(curves)]]
                )

        df = pd.DataFrame(rows, columns=curves)

        # Attach well info as metadata columns
        for key, val in well_info.items():
            df[key] = val

        return df

    # ── Field Mapping ────────────────────────────────────────────────────

    def _map_fields(
        self,
        df: pd.DataFrame,
        standard_fields: Dict[str, List[str]],
        user_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Map user column names to standard field names."""
        df = df.copy()

        # Apply user-provided mapping first
        if user_map:
            df = df.rename(columns={v: k for k, v in user_map.items() if v in df.columns})

        # Auto-detect standard fields
        for std_name, aliases in standard_fields.items():
            if std_name in df.columns:
                continue
            for alias in aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: std_name})
                    break

        # Check required fields
        missing = [f for f in standard_fields if f not in df.columns]
        if missing:
            self._warnings.append(f"Missing columns: {', '.join(missing)}")

        return df

    # ── Data Import ──────────────────────────────────────────────────────

    # Mandatory fields that must be present (no default values allowed).
    MANDATORY_COLLAR_FIELDS = ["borehole_id", "collar_x", "collar_y", "collar_z", "final_depth"]
    MANDATORY_SURVEY_FIELDS = ["borehole_id", "measured_depth"]
    MANDATORY_FRACTURE_FIELDS = ["borehole_id", "measured_depth", "dip_direction", "dip"]
    MANDATORY_RQD_FIELDS = ["borehole_id", "from_depth", "to_depth", "rqd"]

    def _check_mandatory_fields(self, row: pd.Series, mandatory: List[str], row_idx: int, context: str) -> List[str]:
        """Verify all mandatory fields are present and non-null in the row.

        Returns list of missing field names (empty = all present).
        """
        missing = []
        for field_name in mandatory:
            val = row.get(field_name)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                missing.append(field_name)
        if missing:
            self._errors.append(f"Row {row_idx} ({context}): missing mandatory field(s) — {', '.join(missing)}")
        return missing

    def _import_collars(self, df: pd.DataFrame) -> List[Borehole]:
        """Import borehole collar data with mandatory field enforcement."""
        boreholes = []
        for idx, row in df.iterrows():
            try:
                missing = self._check_mandatory_fields(row, self.MANDATORY_COLLAR_FIELDS, idx, "collar")
                if missing:
                    continue  # Skip rows with missing mandatory fields

                bh_id = str(row["borehole_id"])
                collar = Collar(
                    borehole_id=bh_id,
                    collar_x=float(row["collar_x"]),
                    collar_y=float(row["collar_y"]),
                    collar_z=float(row["collar_z"]),
                    azimuth=float(row.get("azimuth", 0)),
                    dip=float(row.get("dip", -90)),
                    final_depth=float(row["final_depth"]),
                )
                boreholes.append(Borehole(borehole_id=bh_id, collar=collar))
            except Exception as e:
                self._errors.append(f"Row {idx}: collar import error — {e}")

        self._check_duplicate_ids([bh.borehole_id for bh in boreholes])
        return boreholes

    def _import_surveys(self, df: pd.DataFrame, boreholes: List[Borehole]) -> None:
        """Import survey data with mandatory field enforcement."""
        bh_map = {bh.borehole_id: bh for bh in boreholes}
        stations_by_bh: Dict[str, List[SurveyStation]] = {}

        for idx, row in df.iterrows():
            try:
                missing = self._check_mandatory_fields(row, self.MANDATORY_SURVEY_FIELDS, idx, "survey")
                if missing:
                    continue

                bh_id = str(row["borehole_id"])
                station = SurveyStation(
                    measured_depth=float(row["measured_depth"]),
                    azimuth=float(row.get("azimuth", 0)),
                    dip=float(row.get("dip", -90)),
                )
                stations_by_bh.setdefault(bh_id, []).append(station)
            except Exception as e:
                self._warnings.append(f"Row {idx}: survey import — {e}")

        for bh_id, stations in stations_by_bh.items():
            if bh_id in bh_map:
                bh_map[bh_id].survey = BoreholeSurvey(stations=stations)

    def _import_fractures(
        self,
        df: pd.DataFrame,
        boreholes: List[Borehole],
        source_file: str = "fractures.csv",
    ) -> None:
        """Import fracture observations with mandatory field enforcement.

        Reads set_id from CSV, validates as integer, and attaches observations
        to the correct borehole.  Unknown borehole IDs are reported as errors
        (not silently dropped).
        """
        bh_map = {bh.borehole_id: bh for bh in boreholes}
        rows_imported = 0
        rows_skipped = 0
        exclusions: List[Dict[str, Any]] = []
        self._rows_fractures_raw = len(df)

        for idx, row in df.iterrows():
            try:
                missing = self._check_mandatory_fields(row, self.MANDATORY_FRACTURE_FIELDS, idx, "fracture")
                if missing:
                    rows_skipped += 1
                    continue

                bh_id = str(row["borehole_id"])
                if bh_id not in bh_map:
                    self._errors.append(
                        f"Row {idx}: unknown borehole ID '{bh_id}' — "
                        f"no matching collar record found. Observation skipped."
                    )
                    rows_skipped += 1
                    continue

                measured_depth = float(row["measured_depth"])
                if measured_depth > bh_map[bh_id].collar.final_depth:
                    reason = (
                        f"Fracture depth {measured_depth} exceeds borehole "
                        f"total depth {bh_map[bh_id].collar.final_depth}"
                    )
                    self._errors.append(f"Row {idx}: {reason}. Observation skipped.")
                    exclusions.append(
                        self._fracture_exclusion_record(
                            row,
                            idx,
                            source_file,
                            bh_id,
                            "measured_depth",
                            row["measured_depth"],
                            reason,
                        )
                    )
                    rows_skipped += 1
                    continue

                dip = float(row["dip"])
                if dip < 0.0 or dip > 90.0:
                    reason = f"Dip {dip} is outside the valid range [0, 90]"
                    self._errors.append(f"Row {idx}: {reason}. Observation skipped.")
                    exclusions.append(
                        self._fracture_exclusion_record(
                            row,
                            idx,
                            source_file,
                            bh_id,
                            "dip",
                            row["dip"],
                            reason,
                        )
                    )
                    rows_skipped += 1
                    continue

                # Parse and validate set_id
                set_id = None
                raw_set_id = row.get("set_id")
                if raw_set_id is not None and (not isinstance(raw_set_id, float) or not pd.isna(raw_set_id)):
                    try:
                        set_id_float = float(raw_set_id)
                        set_id_int = int(set_id_float)
                        if abs(set_id_float - set_id_int) > 1e-6:
                            self._errors.append(f"Row {idx}: set_id '{raw_set_id}' is not an integer")
                            exclusions.append(
                                self._fracture_exclusion_record(
                                    row,
                                    idx,
                                    source_file,
                                    bh_id,
                                    "set_id",
                                    raw_set_id,
                                    f"set_id '{raw_set_id}' is not an integer",
                                )
                            )
                            rows_skipped += 1
                            continue
                        set_id = set_id_int
                    except (ValueError, TypeError):
                        self._errors.append(f"Row {idx}: set_id '{raw_set_id}' is not a valid integer")
                        exclusions.append(
                            self._fracture_exclusion_record(
                                row,
                                idx,
                                source_file,
                                bh_id,
                                "set_id",
                                raw_set_id,
                                f"set_id '{raw_set_id}' is not a valid integer",
                            )
                        )
                        rows_skipped += 1
                        continue

                obs = FractureObservation(
                    borehole_id=bh_id,
                    measured_depth=measured_depth,
                    dip_direction=float(row["dip_direction"]),
                    dip=dip,
                    aperture=float(row["aperture"]) if pd.notna(row.get("aperture")) else None,
                    filling=str(row.get("filling", "")) if pd.notna(row.get("filling")) else None,
                    fracture_type=FractureType(str(row.get("fracture_type", "joint")).lower()),
                    confidence=float(row.get("confidence", 1.0)),
                    set_id=set_id,
                )
                bh_map[bh_id].fracture_observations.append(obs)
                rows_imported += 1
            except Exception as e:
                self._warnings.append(f"Row {idx}: fracture import — {e}")
                rows_skipped += 1

        self._rows_fractures_imported = rows_imported
        self._rows_fractures_skipped = rows_skipped
        self._fracture_exclusions = exclusions

    @staticmethod
    def _fracture_exclusion_record(
        row: pd.Series,
        source_row: Any,
        source_file: str,
        hole_id: str,
        field_name: str,
        original_value: Any,
        reason: str,
    ) -> Dict[str, Any]:
        """Build a serializable, source-traceable fracture exclusion."""
        row_number = int(source_row)
        raw_record = {str(key): (None if pd.isna(value) else value) for key, value in row.to_dict().items()}
        return {
            "issue_id": f"{source_file}:{row_number}:fracture-import",
            "source_file": source_file,
            "source_row": row_number,
            "data_row": row_number + 1,
            "file_line": row_number + 2,
            "hole_id": hole_id,
            "field": field_name,
            "original_value": original_value,
            "reason": reason,
            "applied_action": "excluded_during_import",
            "raw_record": raw_record,
        }

    def _import_rqd(self, df: pd.DataFrame, boreholes: List[Borehole]) -> None:
        """Import RQD intervals with mandatory field enforcement."""
        bh_map = {bh.borehole_id: bh for bh in boreholes}

        for idx, row in df.iterrows():
            try:
                missing = self._check_mandatory_fields(row, self.MANDATORY_RQD_FIELDS, idx, "RQD")
                if missing:
                    continue

                bh_id = str(row["borehole_id"])
                rqd = RQDInterval(
                    borehole_id=bh_id,
                    from_depth=float(row["from_depth"]),
                    to_depth=float(row["to_depth"]),
                    rqd_value=float(row["rqd"]),
                    core_recovery=float(row["core_recovery"]) if pd.notna(row.get("core_recovery")) else None,
                )
                if bh_id in bh_map:
                    bh_map[bh_id].rqd_intervals.append(rqd)
            except Exception as e:
                self._warnings.append(f"Row {idx}: RQD import — {e}")

    # ── Validation Helpers ───────────────────────────────────────────────

    def _check_duplicate_ids(self, ids: List[str]) -> None:
        seen = set()
        for bh_id in ids:
            if bh_id in seen:
                self._errors.append(f"Duplicate borehole ID: {bh_id}")
            seen.add(bh_id)
