"""Canonical project-level borehole database models for M8."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class BoreholeDataType(StrEnum):
    """Built-in borehole data tables."""

    COLLARS = "collars"
    SURVEYS = "surveys"
    FRACTURES = "fractures"
    RQD = "rqd"
    DOMAIN_INTERVALS = "domain_intervals"


class RecordState(StrEnum):
    """Downstream disposition of an immutable raw record."""

    FORMAL = "formal"
    EXCLUDED = "excluded"
    PENDING = "pending"


class QualitySeverity(StrEnum):
    """Severity used by the M8 quality workflow."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class QualityIssueStatus(StrEnum):
    """Resolution state of a persisted quality issue."""

    OPEN = "open"
    APPLIED = "applied"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"


class ModificationEvent(BaseModel):
    """Auditable change applied after import."""

    changed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str
    action: str
    issue_id: str | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


class BoreholeRecord(BaseModel):
    """One raw record and its cleaned/formal disposition.

    ``original_values`` is never edited. User corrections are stored in
    ``values`` and every change is appended to ``modification_history``.
    """

    record_id: str = Field(default_factory=lambda: str(uuid4()))
    data_type: str
    hole_id: str = ""
    source_file: str
    source_row: int = Field(ge=0)
    imported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    original_values: dict[str, Any]
    values: dict[str, Any]
    source_field_mapping: dict[str, str] = Field(default_factory=dict)
    normalized_source_headers: dict[str, str] = Field(default_factory=dict)
    state: RecordState
    exclusion_reason: str | None = None
    modification_source: str = "import"
    import_batch_id: str = Field(default_factory=lambda: str(uuid4()))
    duplicate_of: str | None = None
    modification_history: list[ModificationEvent] = Field(default_factory=list)
    exclusion_confirmed_at: datetime | None = None


class BoreholeQualityIssue(BaseModel):
    """Persisted, idempotent quality finding linked to one database record."""

    issue_id: str = Field(default_factory=lambda: str(uuid4()))
    issue_key: str
    record_id: str
    data_type: str
    hole_id: str = ""
    source_file: str = ""
    source_row: int = Field(default=0, ge=0)
    field: str = ""
    code: str
    severity: QualitySeverity
    status: QualityIssueStatus = QualityIssueStatus.OPEN
    original_value: Any = None
    current_value: Any = None
    suggested_value: Any = None
    reason: str
    auto_fix_available: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    resolution_note: str = ""
    orientation_completeness: str | None = None


class BoreholeDatabase(BaseModel):
    """Canonical data store for all borehole-related tables."""

    schema_version: int = 2
    records: list[BoreholeRecord] = Field(default_factory=list)
    quality_issues: list[BoreholeQualityIssue] = Field(default_factory=list)
    quality_confirmed_at: datetime | None = None
    quality_confirmation_note: str = ""

    def query(
        self,
        data_type: str | None = None,
        state: RecordState | str | None = None,
        hole_id: str | None = None,
        raw: bool = False,
    ) -> list[BoreholeRecord]:
        """Return records matching table, disposition, and hole filters."""
        result = self.records
        if data_type is not None:
            result = [record for record in result if record.data_type == str(data_type)]
        if not raw and state is not None:
            state_value = state.value if isinstance(state, RecordState) else str(state)
            result = [record for record in result if record.state.value == state_value]
        if hole_id is not None:
            result = [record for record in result if record.hole_id == hole_id]
        return list(result)

    def counts(self, data_type: str | None = None) -> dict[str, int]:
        """Return raw/formal/excluded/pending counts."""
        records = self.query(data_type=data_type, raw=True)
        return {
            "raw": len(records),
            "formal": sum(record.state == RecordState.FORMAL for record in records),
            "excluded": sum(record.state == RecordState.EXCLUDED for record in records),
            "pending": sum(record.state == RecordState.PENDING for record in records),
        }

    def orientation_counts(self, state: RecordState | str | None = RecordState.FORMAL) -> dict[str, int]:
        """Count fracture records by scientifically meaningful orientation completeness."""
        records = self.query(BoreholeDataType.FRACTURES, state=state, raw=state is None)
        return {
            "full_orientation": sum(
                record.values.get("orientation_completeness") == "full_orientation" for record in records
            ),
            "dip_only": sum(record.values.get("orientation_completeness") == "dip_only" for record in records),
        }

    @property
    def unresolved_error_count(self) -> int:
        """Return the number of open ERROR-level quality findings."""
        return sum(
            issue.severity == QualitySeverity.ERROR and issue.status == QualityIssueStatus.OPEN
            for issue in self.quality_issues
        )
