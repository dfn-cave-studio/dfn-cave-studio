"""M7 data management models: import/cleaning/provenance/holdout.

Defines the data structures for the complete M7 pre-processing pipeline:
  - FieldMapping: user-saved field mapping templates
  - DataQualityIssue: individual data problem tracking
  - CleaningRecord: per-field modification history
  - ValidationHoldout: calibration/validation borehole split
  - StructuralDomainInterval: per-borehole depth-bounded domain assignment
  - ProcessedDataExport: output format for cleaned/processed data

DO NOT:
  - Use mutable DataFrame row numbers as record identity
  - Silence uncorrectable data without logging
  - Use RQD as a proxy for P32
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Record identity
# ═══════════════════════════════════════════════════════════════════════════

def new_record_id() -> str:
    """Stable record identifier — survives DataFrame row reordering."""
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════
# Field mapping
# ═══════════════════════════════════════════════════════════════════════════

class FieldMapping(BaseModel):
    """A saved field-mapping template for CSV/XLSX import.

    Users can save and reuse mappings so they don't have to re-map
    columns every time they import similar files.
    """

    name: str = "Default Mapping"
    description: str = ""
    data_type: str = "collars"  # collars | surveys | fractures | rqd | domain_intervals
    file_format: str = "csv"    # csv | xlsx
    encoding: str = "utf-8"
    delimiter: str = ","
    header_row: int = 0
    skip_rows: int = 0

    # source_column_name → standard_field_name
    column_map: Dict[str, str] = Field(default_factory=dict)

    # Unit / angle conventions
    depth_unit: str = "m"
    angle_convention: str = "dip_direction_dip"  # dip_direction_dip | strike_dip | alpha_beta
    coord_system_note: str = ""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════════════════
# Data quality issue
# ═══════════════════════════════════════════════════════════════════════════

class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueType(str, Enum):
    MISSING_REQUIRED = "missing_required"
    MISSING_OPTIONAL = "missing_optional"
    INVALID_TYPE = "invalid_type"
    OUT_OF_RANGE = "out_of_range"
    DUPLICATE = "duplicate"
    OVERLAP = "overlap"
    UNKNOWN_REFERENCE = "unknown_reference"
    DEPTH_ORDER = "depth_order"
    DEPTH_EXCEEDS = "depth_exceeds"
    ANGLE_OUT_OF_RANGE = "angle_out_of_range"
    NEGATIVE_DEPTH = "negative_depth"
    ZERO_TOTAL_DEPTH = "zero_total_depth"


class DataQualityIssue(BaseModel):
    """A single data quality problem found during import or cleaning.

    Every issue is traceable back to its source file, row, and field.
    """

    issue_id: str = Field(default_factory=new_record_id)
    source_file: str = ""
    source_row: int = -1
    hole_id: str = ""
    field: str = ""
    original_value: Any = None
    issue_type: IssueType = IssueType.MISSING_REQUIRED
    severity: IssueSeverity = IssueSeverity.WARNING
    message: str = ""
    suggested_action: str = ""
    applied_action: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════════════════
# Data provenance per record
# ═══════════════════════════════════════════════════════════════════════════

class RecordProvenance(BaseModel):
    """Tracks the history of a single data record through the pipeline."""

    record_id: str = Field(default_factory=new_record_id)
    source_file: str = ""
    source_row: int = -1
    imported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_values: Dict[str, Any] = Field(default_factory=dict)
    current_values: Dict[str, Any] = Field(default_factory=dict)
    modifications: List[Dict[str, Any]] = Field(default_factory=list)  # [{field, old, new, reason, at}]
    excluded: bool = False
    exclusion_reason: str = ""
    role: Literal["calibration", "validation", "unassigned"] = "unassigned"
    domain_id: Optional[int] = None
    set_id: Optional[int] = None
    set_id_source: Literal["imported", "automatic", "manual", ""] = ""


# ═══════════════════════════════════════════════════════════════════════════
# Validation holdout
# ═══════════════════════════════════════════════════════════════════════════

class HoldoutRole(str, Enum):
    CALIBRATION = "calibration"
    VALIDATION = "validation"


class ValidationHoldout(BaseModel):
    """Records the calibration/validation split for boreholes.

    Validation boreholes are held out BEFORE any modelling step.
    They must not participate in: domain fitting, joint set clustering,
    Fisher statistics, density estimation, or DFN parameter assignment.
    """

    holdout_id: str = Field(default_factory=new_record_id)
    hole_id: str
    role: HoldoutRole = HoldoutRole.CALIBRATION
    selection_method: str = "manual"  # manual | random | stratified
    random_seed: Optional[int] = None
    domain_stratum: Optional[int] = None  # if stratified by domain
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    locked: bool = False
    locked_at: Optional[datetime] = None
    unlocked_at: Optional[datetime] = None


class HoldoutConfig(BaseModel):
    """Configuration for validation holdout selection."""

    method: str = "manual"  # manual | random | stratified
    validation_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    random_seed: Optional[int] = 42
    stratify_by_domain: bool = False
    min_validation_count: int = Field(default=1, ge=0)
    locked: bool = False
    holdouts: List[ValidationHoldout] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════════════════
# Structural domain intervals (borehole depth-bounded)
# ═══════════════════════════════════════════════════════════════════════════

class DomainInterval(BaseModel):
    """A depth-bounded structural domain assignment for one borehole.

    One borehole can pass through multiple structural domains.
    This is NOT a claim that the entire 3D volume is partitioned —
    it is a constraint at the borehole location only.
    """

    interval_id: str = Field(default_factory=new_record_id)
    hole_id: str
    from_depth: float = Field(ge=0.0)
    to_depth: float = Field(gt=0.0)
    domain_id: int
    domain_name: str = ""
    assignment_method: str = "imported"  # imported | manual | inferred
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def interval_length(self) -> float:
        return self.to_depth - self.from_depth

    def overlaps(self, other: "DomainInterval") -> bool:
        """Check if two intervals overlap."""
        return self.from_depth < other.to_depth and other.from_depth < self.to_depth


# ═══════════════════════════════════════════════════════════════════════════
# Output / export
# ═══════════════════════════════════════════════════════════════════════════

class ProcessedDataExport(BaseModel):
    """Metadata for a processed-data export."""

    export_id: str = Field(default_factory=new_record_id)
    project_name: str = ""
    exported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    records_total: int = 0
    records_calibration: int = 0
    records_validation: int = 0
    records_excluded: int = 0
    domains_count: int = 0
    joint_sets_count: int = 0
    random_seed: int = 0
    files: List[str] = Field(default_factory=list)  # list of output file paths
