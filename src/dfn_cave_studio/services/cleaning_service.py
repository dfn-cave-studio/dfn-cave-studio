"""M7 data cleaning service with issue tracking and traceable corrections.

Checks for common data quality problems in borehole datasets:
  - Missing/empty hole_id
  - Duplicate hole IDs
  - Duplicate records
  - Missing coordinates
  - Non-numeric values
  - Total depth ≤ 0
  - Negative measured depths
  - Depth exceeding total depth
  - Survey depths not monotonically increasing
  - Fracture depth exceeding hole depth
  - RQD outside 0–100
  - Interval from_depth >= to_depth
  - Overlapping RQD or domain intervals
  - Angle out of range
  - set_id non-integer
  - Unknown borehole references

Every issue is recorded as a DataQualityIssue that references source file,
row, field, original value, and severity.  Corrections are applied in-memory
and logged; original files are NEVER modified.

DO NOT:
  - Overwrite source CSV/XLSX files
  - Silently change unexplainable orientation data
  - Treat RQD as a P32 proxy
"""

from __future__ import annotations

from typing import List, Dict, Any, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict

import pandas as pd

from dfn_cave_studio.models.data_management import (
    DataQualityIssue,
    IssueSeverity,
    IssueType,
    new_record_id,
)


class DataCleaningService:
    """Check and clean borehole datasets with full traceability."""

    def __init__(self):
        self._issues: List[DataQualityIssue] = []
        self._excluded: List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────

    def validate_collars(self, df: pd.DataFrame, source_file: str = "collars.csv") -> List[DataQualityIssue]:
        """Validate collar data."""
        issues: List[DataQualityIssue] = []
        seen_ids: Set[str] = set()

        for idx, row in df.iterrows():
            hole_id = str(row.get("hole_id", row.get("borehole_id", ""))).strip()

            # Empty hole_id
            if not hole_id or hole_id.lower() == "nan":
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        "",
                        "hole_id",
                        IssueType.MISSING_REQUIRED,
                        IssueSeverity.ERROR,
                        str(row.get("hole_id", "")),
                        "Borehole ID is required",
                        "Row excluded",
                    )
                )
                continue

            # Duplicate
            if hole_id in seen_ids:
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "hole_id",
                        IssueType.DUPLICATE,
                        IssueSeverity.ERROR,
                        hole_id,
                        "Duplicate borehole ID",
                        "Keep first, exclude duplicate",
                    )
                )
                continue
            seen_ids.add(hole_id)

            # Missing coordinates
            for coord in ["easting", "northing", "elevation"]:
                val = row.get(coord)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    issues.append(
                        self._issue(
                            source_file,
                            idx,
                            hole_id,
                            coord,
                            IssueType.MISSING_REQUIRED,
                            IssueSeverity.ERROR,
                            "",
                            f"Missing {coord}",
                            "Row excluded",
                        )
                    )
                else:
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        issues.append(
                            self._issue(
                                source_file,
                                idx,
                                hole_id,
                                coord,
                                IssueType.INVALID_TYPE,
                                IssueSeverity.ERROR,
                                str(val),
                                f"Non-numeric {coord}",
                                "Row excluded",
                            )
                        )

            # Total depth
            td = row.get("total_depth")
            if td is not None and not (isinstance(td, float) and pd.isna(td)):
                try:
                    td_val = float(td)
                    if td_val <= 0:
                        issues.append(
                            self._issue(
                                source_file,
                                idx,
                                hole_id,
                                "total_depth",
                                IssueType.ZERO_TOTAL_DEPTH,
                                IssueSeverity.ERROR,
                                str(td),
                                "Total depth must be > 0",
                                "Row excluded",
                            )
                        )
                except (ValueError, TypeError):
                    issues.append(
                        self._issue(
                            source_file,
                            idx,
                            hole_id,
                            "total_depth",
                            IssueType.INVALID_TYPE,
                            IssueSeverity.ERROR,
                            str(td),
                            "Non-numeric total depth",
                            "Row excluded",
                        )
                    )

            # Azimuth range
            az = row.get("azimuth")
            if az is not None and not (isinstance(az, float) and pd.isna(az)):
                try:
                    az_val = float(az)
                    if az_val < 0 or az_val >= 360:
                        issues.append(
                            self._issue(
                                source_file,
                                idx,
                                hole_id,
                                "azimuth",
                                IssueType.ANGLE_OUT_OF_RANGE,
                                IssueSeverity.WARNING,
                                str(az),
                                "Azimuth should be [0, 360)",
                                "Normalized to [0, 360)",
                            )
                        )
                        df.at[idx, "azimuth"] = az_val % 360
                except (ValueError, TypeError):
                    pass

            # Dip range
            dip = row.get("dip")
            if dip is not None and not (isinstance(dip, float) and pd.isna(dip)):
                try:
                    dip_val = float(dip)
                    if dip_val < -90 or dip_val > 90:
                        issues.append(
                            self._issue(
                                source_file,
                                idx,
                                hole_id,
                                "dip",
                                IssueType.ANGLE_OUT_OF_RANGE,
                                IssueSeverity.WARNING,
                                str(dip),
                                "Dip should be [-90, 90]",
                                "Clamped to [-90, 90]",
                            )
                        )
                except (ValueError, TypeError):
                    pass

        self._issues.extend(issues)
        return issues

    def validate_surveys(
        self, df: pd.DataFrame, collars: pd.DataFrame, source_file: str = "surveys.csv"
    ) -> List[DataQualityIssue]:
        """Validate survey data."""
        issues: List[DataQualityIssue] = []
        valid_hole_ids = set(str(h).strip() for h in collars.get("hole_id", collars.get("borehole_id", [])))
        max_depths: Dict[str, float] = {}
        for _, row in collars.iterrows():
            hid = str(row.get("hole_id", row.get("borehole_id", ""))).strip()
            td = row.get("total_depth")
            if hid and td and not (isinstance(td, float) and pd.isna(td)):
                max_depths[hid] = float(td)

        # Check per-borehole depth order
        bh_depths: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for idx, row in df.iterrows():
            hole_id = str(row.get("hole_id", "")).strip()
            md = row.get("measured_depth")
            if md is not None and not (isinstance(md, float) and pd.isna(md)):
                try:
                    bh_depths[hole_id].append((idx, float(md)))
                except (ValueError, TypeError):
                    pass

        # Per-borehole checks
        for hole_id, entries in bh_depths.items():
            if hole_id not in valid_hole_ids:
                for idx, _ in entries:
                    issues.append(
                        self._issue(
                            source_file,
                            idx,
                            hole_id,
                            "hole_id",
                            IssueType.UNKNOWN_REFERENCE,
                            IssueSeverity.ERROR,
                            hole_id,
                            f"Borehole '{hole_id}' not in collars",
                            "Row excluded",
                        )
                    )
                continue

            max_depth = max_depths.get(hole_id, float("inf"))
            entries_sorted = sorted(entries, key=lambda x: x[1])

            # Check depth monotonicity
            prev_md = -1
            for idx, md in entries_sorted:
                if md < 0:
                    issues.append(
                        self._issue(
                            source_file,
                            idx,
                            hole_id,
                            "measured_depth",
                            IssueType.NEGATIVE_DEPTH,
                            IssueSeverity.ERROR,
                            str(md),
                            "Depth cannot be negative",
                            "Row excluded",
                        )
                    )
                if md <= prev_md:
                    issues.append(
                        self._issue(
                            source_file,
                            idx,
                            hole_id,
                            "measured_depth",
                            IssueType.DEPTH_ORDER,
                            IssueSeverity.WARNING,
                            str(md),
                            "Survey depths must increase monotonically",
                            "",
                        )
                    )
                prev_md = md
                if md > max_depth:
                    issues.append(
                        self._issue(
                            source_file,
                            idx,
                            hole_id,
                            "measured_depth",
                            IssueType.DEPTH_EXCEEDS,
                            IssueSeverity.ERROR,
                            str(md),
                            f"Depth {md} exceeds total depth {max_depth}",
                            "Row excluded",
                        )
                    )

        self._issues.extend(issues)
        return issues

    def validate_fractures(
        self, df: pd.DataFrame, collars: pd.DataFrame, source_file: str = "fractures.csv"
    ) -> List[DataQualityIssue]:
        """Validate fracture observation data."""
        issues: List[DataQualityIssue] = []
        valid_hole_ids = set(str(h).strip() for h in collars.get("hole_id", collars.get("borehole_id", [])))
        max_depths: Dict[str, float] = {}
        for _, row in collars.iterrows():
            hid = str(row.get("hole_id", row.get("borehole_id", ""))).strip()
            td = row.get("total_depth")
            if hid and td and not (isinstance(td, float) and pd.isna(td)):
                max_depths[hid] = float(td)

        for idx, row in df.iterrows():
            hole_id = str(row.get("hole_id", "")).strip()
            # Unknown borehole
            if hole_id not in valid_hole_ids:
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "hole_id",
                        IssueType.UNKNOWN_REFERENCE,
                        IssueSeverity.ERROR,
                        hole_id,
                        f"Borehole '{hole_id}' not in collars",
                        "Row excluded",
                    )
                )
            # Depth exceeds total
            depth = row.get("depth", row.get("measured_depth"))
            max_d = max_depths.get(hole_id, float("inf"))
            if depth is not None and not (isinstance(depth, float) and pd.isna(depth)):
                try:
                    d_val = float(depth)
                    if d_val > max_d:
                        issues.append(
                            self._issue(
                                source_file,
                                idx,
                                hole_id,
                                "depth",
                                IssueType.DEPTH_EXCEEDS,
                                IssueSeverity.ERROR,
                                str(depth),
                                f"Fracture depth {d_val} exceeds total depth {max_d}",
                                "Row excluded",
                            )
                        )
                except (ValueError, TypeError):
                    pass
            # dip_direction range
            dd = row.get("dip_direction")
            if dd is not None and not (isinstance(dd, float) and pd.isna(dd)):
                try:
                    dd_val = float(dd)
                    if dd_val < 0 or dd_val >= 360:
                        issues.append(
                            self._issue(
                                source_file,
                                idx,
                                hole_id,
                                "dip_direction",
                                IssueType.ANGLE_OUT_OF_RANGE,
                                IssueSeverity.WARNING,
                                str(dd),
                                "Dip direction should be [0, 360)",
                                "Normalized to [0, 360)",
                            )
                        )
                except (ValueError, TypeError):
                    pass
            # dip range
            dip = row.get("dip")
            if dip is not None and not (isinstance(dip, float) and pd.isna(dip)):
                try:
                    dip_val = float(dip)
                    if dip_val < 0 or dip_val > 90:
                        issues.append(
                            self._issue(
                                source_file,
                                idx,
                                hole_id,
                                "dip",
                                IssueType.ANGLE_OUT_OF_RANGE,
                                IssueSeverity.ERROR,
                                str(dip),
                                "Dip should be [0, 90]",
                                "Row excluded",
                            )
                        )
                except (ValueError, TypeError):
                    pass
            # set_id integer check
            sid = row.get("set_id")
            if sid is not None and not (isinstance(sid, float) and pd.isna(sid)):
                try:
                    sid_float = float(sid)
                    sid_int = int(sid_float)
                    if abs(sid_float - sid_int) > 1e-6:
                        issues.append(
                            self._issue(
                                source_file,
                                idx,
                                hole_id,
                                "set_id",
                                IssueType.INVALID_TYPE,
                                IssueSeverity.ERROR,
                                str(sid),
                                "set_id must be an integer",
                                "Row excluded",
                            )
                        )
                except (ValueError, TypeError):
                    issues.append(
                        self._issue(
                            source_file,
                            idx,
                            hole_id,
                            "set_id",
                            IssueType.INVALID_TYPE,
                            IssueSeverity.ERROR,
                            str(sid),
                            "set_id must be an integer",
                            "Row excluded",
                        )
                    )

        self._issues.extend(issues)
        return issues

    def validate_rqd(
        self, df: pd.DataFrame, collars: pd.DataFrame, source_file: str = "rqd.csv"
    ) -> List[DataQualityIssue]:
        """Validate RQD data."""
        issues: List[DataQualityIssue] = []
        valid_hole_ids = set(str(h).strip() for h in collars.get("hole_id", collars.get("borehole_id", [])))
        depth_column = "total_depth" if "total_depth" in collars.columns else "final_depth"
        max_depths = {
            str(row.get("hole_id", row.get("borehole_id", ""))).strip(): float(row.get(depth_column, float("inf")))
            for _, row in collars.iterrows()
        }

        # Check per-borehole intervals
        bh_intervals: Dict[str, List[Tuple[int, float, float, float]]] = defaultdict(list)

        for idx, row in df.iterrows():
            hole_id = str(row.get("hole_id", "")).strip()
            if hole_id not in valid_hole_ids:
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "hole_id",
                        IssueType.UNKNOWN_REFERENCE,
                        IssueSeverity.ERROR,
                        hole_id,
                        f"Borehole '{hole_id}' not in collars",
                        "Row excluded",
                    )
                )
                continue

            fd = row.get("from_depth")
            td = row.get("to_depth")
            rqd = row.get("rqd")
            try:
                fd_val = float(fd)
                td_val = float(td)
                rqd_val = float(rqd) if rqd is not None and not (isinstance(rqd, float) and pd.isna(rqd)) else None
            except (ValueError, TypeError):
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "from_depth/to_depth/rqd",
                        IssueType.INVALID_TYPE,
                        IssueSeverity.ERROR,
                        f"from={fd}, to={td}, rqd={rqd}",
                        "RQD interval contains a non-numeric value",
                        "Row excluded",
                    )
                )
                continue

            if fd_val >= td_val:
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "from_depth/to_depth",
                        IssueType.OUT_OF_RANGE,
                        IssueSeverity.ERROR,
                        f"from={fd_val}, to={td_val}",
                        "from_depth must be < to_depth",
                        "Row excluded",
                    )
                )
            if fd_val < 0:
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "from_depth",
                        IssueType.NEGATIVE_DEPTH,
                        IssueSeverity.ERROR,
                        str(fd_val),
                        "Depth cannot be negative",
                        "Row excluded",
                    )
                )
            if td_val > max_depths.get(hole_id, float("inf")):
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "to_depth",
                        IssueType.DEPTH_EXCEEDS,
                        IssueSeverity.ERROR,
                        str(td_val),
                        f"Depth {td_val} exceeds borehole total depth " f"{max_depths[hole_id]}",
                        "Row excluded",
                    )
                )
            if rqd_val is not None and (rqd_val < 0 or rqd_val > 100):
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "rqd",
                        IssueType.OUT_OF_RANGE,
                        IssueSeverity.WARNING,
                        str(rqd_val),
                        "RQD must be [0, 100]",
                        "Clamped to [0, 100]",
                    )
                )

            bh_intervals[hole_id].append((idx, fd_val, td_val, rqd_val or 0))

        # Check interval overlap per borehole
        for hole_id, intervals in bh_intervals.items():
            sorted_ints = sorted(intervals, key=lambda x: x[1])
            for i in range(len(sorted_ints) - 1):
                idx_a, fa, ta, _ = sorted_ints[i]
                idx_b, fb, tb, _ = sorted_ints[i + 1]
                if ta > fb:
                    issues.append(
                        self._issue(
                            source_file,
                            idx_b,
                            hole_id,
                            "from_depth/to_depth",
                            IssueType.OVERLAP,
                            IssueSeverity.WARNING,
                            f"[{fa},{ta}] vs [{fb},{tb}]",
                            "RQD intervals overlap",
                            "",
                        )
                    )

        self._issues.extend(issues)
        return issues

    def validate_domain_intervals(
        self, df: pd.DataFrame, collars: pd.DataFrame, source_file: str = "domain_intervals.csv"
    ) -> List[DataQualityIssue]:
        """Validate structural domain interval data."""
        issues: List[DataQualityIssue] = []
        valid_hole_ids = set(str(h).strip() for h in collars.get("hole_id", collars.get("borehole_id", [])))
        depth_column = "total_depth" if "total_depth" in collars.columns else "final_depth"
        max_depths = {
            str(row.get("hole_id", row.get("borehole_id", ""))).strip(): float(row.get(depth_column, float("inf")))
            for _, row in collars.iterrows()
        }
        bh_intervals: Dict[str, List[Tuple[int, float, float, int]]] = defaultdict(list)

        for idx, row in df.iterrows():
            hole_id = str(row.get("hole_id", "")).strip()
            if hole_id not in valid_hole_ids:
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "hole_id",
                        IssueType.UNKNOWN_REFERENCE,
                        IssueSeverity.ERROR,
                        hole_id,
                        f"Borehole '{hole_id}' not in collars",
                        "Row excluded",
                    )
                )
                continue
            try:
                fd_val = float(row.get("from_depth", 0))
                td_val = float(row.get("to_depth", 0))
                domain_id = int(row.get("domain_id", 0))
            except (ValueError, TypeError):
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "from_depth/to_depth/domain_id",
                        IssueType.INVALID_TYPE,
                        IssueSeverity.ERROR,
                        str(row.to_dict()),
                        "Domain interval contains a non-numeric value",
                        "Row excluded",
                    )
                )
                continue
            if fd_val >= td_val:
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "from_depth/to_depth",
                        IssueType.OUT_OF_RANGE,
                        IssueSeverity.ERROR,
                        f"from={fd_val}, to={td_val}",
                        "from_depth must be < to_depth",
                        "Row excluded",
                    )
                )
            if td_val > max_depths.get(hole_id, float("inf")):
                issues.append(
                    self._issue(
                        source_file,
                        idx,
                        hole_id,
                        "to_depth",
                        IssueType.DEPTH_EXCEEDS,
                        IssueSeverity.ERROR,
                        str(td_val),
                        f"Depth {td_val} exceeds borehole total depth " f"{max_depths[hole_id]}",
                        "Row excluded",
                    )
                )
            bh_intervals[hole_id].append((idx, fd_val, td_val, domain_id))

        for hole_id, intervals in bh_intervals.items():
            sorted_ints = sorted(intervals, key=lambda x: x[1])
            for i in range(len(sorted_ints) - 1):
                _, fa, ta, _ = sorted_ints[i]
                _, fb, tb, _ = sorted_ints[i + 1]
                if ta > fb:
                    issues.append(
                        self._issue(
                            source_file,
                            sorted_ints[i + 1][0],
                            hole_id,
                            "from_depth/to_depth",
                            IssueType.OVERLAP,
                            IssueSeverity.ERROR,
                            f"[{fa},{ta}] vs [{fb},{tb}]",
                            "Domain intervals overlap — must be fixed manually",
                            "",
                        )
                    )

        self._issues.extend(issues)
        return issues

    # ── Helpers ───────────────────────────────────────────────────────────

    def _issue(
        self,
        source_file: str,
        row: int,
        hole_id: str,
        field: str,
        issue_type: IssueType,
        severity: IssueSeverity,
        original_value: Any,
        message: str,
        suggested_action: str,
    ) -> DataQualityIssue:
        return DataQualityIssue(
            issue_id=new_record_id(),
            source_file=source_file,
            source_row=int(row),
            hole_id=hole_id,
            field=field,
            original_value=original_value,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggested_action=suggested_action,
            created_at=datetime.now(timezone.utc),
        )

    @property
    def all_issues(self) -> List[DataQualityIssue]:
        return list(self._issues)

    def issues_by_severity(self, severity: IssueSeverity) -> List[DataQualityIssue]:
        return [i for i in self._issues if i.severity == severity]

    @property
    def error_count(self) -> int:
        return len(self.issues_by_severity(IssueSeverity.ERROR))

    @property
    def warning_count(self) -> int:
        return len(self.issues_by_severity(IssueSeverity.WARNING))

    def export_issues_report(self) -> List[Dict[str, Any]]:
        """Export all issues as a list of dicts (suitable for CSV/JSON)."""
        return [
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
            for i in self._issues
        ]

    def clear(self) -> None:
        self._issues.clear()
        self._excluded.clear()
