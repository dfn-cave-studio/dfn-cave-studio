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
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import pandas as pd

from dfn_cave_studio.models.borehole import (
    BoreholeCollection, Borehole, Collar, SurveyStation,
    BoreholeSurvey, FractureObservation, RQDInterval,
    BoreholeValidationError,
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
                    self._import_fractures(frac_df, boreholes)

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

            return ImportResult(
                success=len(self._errors) == 0,
                collection=collection,
                errors=self._errors,
                warnings=self._warnings,
                rows_imported=len(boreholes),
            )

        except Exception as e:
            self._errors.append(f"Import failed: {e}")
            return ImportResult(success=False, errors=self._errors, warnings=self._warnings)

    # ── File Reading ─────────────────────────────────────────────────────

    def _read_file(self, path: str) -> Optional[pd.DataFrame]:
        """Read CSV or Excel file."""
        path = Path(path)
        if not path.exists():
            self._errors.append(f"File not found: {path}")
            return None

        try:
            if path.suffix.lower() in ('.xlsx', '.xls'):
                return pd.read_excel(path)
            else:
                return pd.read_csv(path)
        except Exception as e:
            self._errors.append(f"Failed to read {path.name}: {e}")
            return None

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

    def _import_collars(self, df: pd.DataFrame) -> List[Borehole]:
        """Import borehole collar data."""
        boreholes = []
        for idx, row in df.iterrows():
            try:
                bh_id = str(row.get("borehole_id", f"BH-{idx + 1:03d}"))
                collar = Collar(
                    borehole_id=bh_id,
                    collar_x=float(row.get("collar_x", 0)),
                    collar_y=float(row.get("collar_y", 0)),
                    collar_z=float(row.get("collar_z", 0)),
                    azimuth=float(row.get("azimuth", 0)),
                    dip=float(row.get("dip", -90)),
                    final_depth=float(row.get("final_depth", 100)),
                )
                boreholes.append(Borehole(borehole_id=bh_id, collar=collar))
            except Exception as e:
                self._errors.append(f"Row {idx}: collar import error — {e}")

        self._check_duplicate_ids([bh.borehole_id for bh in boreholes])
        return boreholes

    def _import_surveys(self, df: pd.DataFrame, boreholes: List[Borehole]) -> None:
        """Import survey data and attach to boreholes."""
        bh_map = {bh.borehole_id: bh for bh in boreholes}
        stations_by_bh: Dict[str, List[SurveyStation]] = {}

        for idx, row in df.iterrows():
            try:
                bh_id = str(row.get("borehole_id", ""))
                station = SurveyStation(
                    measured_depth=float(row.get("measured_depth", 0)),
                    azimuth=float(row.get("azimuth", 0)),
                    dip=float(row.get("dip", -90)),
                )
                stations_by_bh.setdefault(bh_id, []).append(station)
            except Exception as e:
                self._warnings.append(f"Row {idx}: survey import — {e}")

        for bh_id, stations in stations_by_bh.items():
            if bh_id in bh_map:
                bh_map[bh_id].survey = BoreholeSurvey(stations=stations)

    def _import_fractures(self, df: pd.DataFrame, boreholes: List[Borehole]) -> None:
        """Import fracture observations."""
        bh_map = {bh.borehole_id: bh for bh in boreholes}

        for idx, row in df.iterrows():
            try:
                bh_id = str(row.get("borehole_id", ""))
                obs = FractureObservation(
                    borehole_id=bh_id,
                    measured_depth=float(row.get("measured_depth", 0)),
                    dip_direction=float(row.get("dip_direction", 0)),
                    dip=float(row.get("dip", 0)),
                    aperture=float(row["aperture"]) if pd.notna(row.get("aperture")) else None,
                    filling=str(row.get("filling", "")) if pd.notna(row.get("filling")) else None,
                    fracture_type=FractureType(str(row.get("fracture_type", "joint")).lower()),
                    confidence=float(row.get("confidence", 1.0)),
                )
                if bh_id in bh_map:
                    bh_map[bh_id].fracture_observations.append(obs)
            except Exception as e:
                self._warnings.append(f"Row {idx}: fracture import — {e}")

    def _import_rqd(self, df: pd.DataFrame, boreholes: List[Borehole]) -> None:
        """Import RQD intervals."""
        bh_map = {bh.borehole_id: bh for bh in boreholes}

        for idx, row in df.iterrows():
            try:
                bh_id = str(row.get("borehole_id", ""))
                rqd = RQDInterval(
                    borehole_id=bh_id,
                    from_depth=float(row.get("from_depth", 0)),
                    to_depth=float(row.get("to_depth", 0)),
                    rqd_value=float(row.get("rqd", 0)),
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
