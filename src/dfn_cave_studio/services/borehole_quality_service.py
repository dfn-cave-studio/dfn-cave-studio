"""M8 data-quality checks and audited resolution operations."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from dfn_cave_studio.models.borehole_database import (
    BoreholeDataType,
    BoreholeQualityIssue,
    BoreholeRecord,
    FractureObservationMode,
    QualityIssueStatus,
    QualitySeverity,
    RecordState,
)
from dfn_cave_studio.services.borehole_repository import BoreholeRepository


class BoreholeQualityService:
    """Run idempotent checks against the canonical M8 borehole database."""

    def __init__(self, project: Any):
        self.project = project
        self.repository = BoreholeRepository(project)
        self.last_auto_fix_record_count = 0
        self.last_auto_fix_field_count = 0

    @property
    def issues(self) -> list[BoreholeQualityIssue]:
        """Return all persisted findings, including resolved history."""
        return [issue.model_copy(deep=True) for issue in self.repository.database.quality_issues]

    @property
    def unresolved_error_count(self) -> int:
        """Return open ERROR findings."""
        return self.repository.database.unresolved_error_count

    @property
    def can_complete(self) -> bool:
        """Whether quality confirmation is currently permitted."""
        counts = self.repository.database.counts()
        return counts["raw"] > 0 and counts["pending"] == 0 and self.unresolved_error_count == 0

    def run_checks(self) -> list[BoreholeQualityIssue]:
        """Re-run all checks and merge findings by stable issue key."""
        detected: dict[str, BoreholeQualityIssue] = {}
        records = self.repository.database.records
        formal_collars = {
            record.hole_id
            for record in records
            if record.data_type == BoreholeDataType.COLLARS and record.state == RecordState.FORMAL
        }

        for record in records:
            if record.state == RecordState.EXCLUDED:
                issue = self._issue(
                    record,
                    code="excluded_record",
                    field="*",
                    severity=QualitySeverity.ERROR,
                    reason=record.exclusion_reason or "Excluded record requires a documented reason",
                )
                if record.exclusion_confirmed_at is not None and record.exclusion_reason:
                    issue.status = QualityIssueStatus.CONFIRMED
                    issue.resolved_at = record.exclusion_confirmed_at
                    issue.resolution_note = record.exclusion_reason
                detected[issue.issue_key] = issue
                continue

            if record.state == RecordState.PENDING or (
                record.data_type not in {BoreholeDataType.COLLARS, BoreholeDataType.ORIENTATION_POINTS}
                and record.hole_id not in formal_collars
            ):
                issue = self._issue(
                    record,
                    code="unknown_collar",
                    field="hole_id",
                    severity=QualitySeverity.ERROR,
                    reason=f"Collar '{record.hole_id}' is not available",
                )
                detected[issue.issue_key] = issue
                continue

            if record.data_type == BoreholeDataType.COLLARS:
                self._check_collar(record, detected)
            elif record.data_type == BoreholeDataType.SURVEYS:
                self._check_survey(record, detected)
            elif record.data_type == BoreholeDataType.FRACTURES:
                self._check_fracture(record, detected)
            elif record.data_type == BoreholeDataType.RQD:
                self._check_rqd(record, detected)
            elif record.data_type == BoreholeDataType.RMR:
                self._check_rmr(record, detected)
            elif record.data_type == BoreholeDataType.DOMAIN_INTERVALS:
                self._check_domain(record, detected)
            elif record.data_type == BoreholeDataType.ORIENTATION_POINTS:
                self._check_orientation_point(record, detected)

        self._check_overlaps(BoreholeDataType.RQD, detected)
        self._check_overlaps(BoreholeDataType.RMR, detected)
        self._check_overlaps(BoreholeDataType.DOMAIN_INTERVALS, detected)
        self._merge_findings(detected)
        if self.repository.database.counts()["pending"] or self.unresolved_error_count:
            self.repository.set_quality_confirmation(None)
        return self.issues

    def apply_auto_fixes(self, issue_ids: list[str] | None = None) -> int:
        """Apply open safe fixes grouped by record in one transaction."""
        selected = set(issue_ids) if issue_ids is not None else None
        issues = [issue.model_copy(deep=True) for issue in self.repository.database.quality_issues]
        grouped: dict[str, list[BoreholeQualityIssue]] = defaultdict(list)
        for issue in issues:
            if (
                issue.status != QualityIssueStatus.OPEN
                or not issue.auto_fix_available
                or (selected is not None and issue.issue_id not in selected)
            ):
                continue
            record = self.repository.get_record(issue.record_id)
            if record.values.get(issue.field) != issue.suggested_value:
                grouped[issue.record_id].append(issue)

        self.last_auto_fix_record_count = 0
        self.last_auto_fix_field_count = 0
        if not grouped:
            return 0
        selected_issue_ids = {issue.issue_id for record_issues in grouped.values() for issue in record_issues}
        try:
            with self.repository.transaction():
                record_count, field_count = self.repository.apply_grouped_quality_fixes(grouped)
                resolved_at = datetime.now(UTC)
                for issue in issues:
                    if issue.issue_id not in selected_issue_ids:
                        continue
                    issue.status = QualityIssueStatus.RESOLVED
                    issue.current_value = issue.suggested_value
                    issue.resolved_at = resolved_at
                    issue.resolution_note = f"Applied suggested {issue.field}={issue.suggested_value}"
                self.repository.set_quality_issues(issues)
                self.run_checks()
        except Exception:
            self.last_auto_fix_record_count = 0
            self.last_auto_fix_field_count = 0
            raise
        self.last_auto_fix_record_count = record_count
        self.last_auto_fix_field_count = field_count
        return field_count

    def edit_record(self, record_id: str, values: dict[str, Any]) -> BoreholeRecord:
        """Apply an audited manual correction."""
        record = self.repository.edit_record(record_id, values, modification_source="m8_quality:manual_edit")
        self.run_checks()
        return record

    def exclude_record(self, record_id: str, reason: str) -> BoreholeRecord:
        """Exclude an unfixable record with a mandatory reason."""
        if not reason.strip():
            raise ValueError("An exclusion reason is required")
        record = self.repository.exclude_record(record_id, reason.strip(), modification_source="m8_quality:exclude")
        self.run_checks()
        return record

    def retry_record(self, record_id: str) -> BoreholeRecord:
        """Attempt to restore/reclassify an Excluded or Pending record."""
        record = self.repository.reclassify_record(record_id, modification_source="m8_quality:retry")
        self.run_checks()
        return record

    def confirm_exclusions(self, record_ids: list[str] | None = None) -> int:
        """Confirm selected or all documented exclusions."""
        confirmed = self.repository.confirm_exclusions(record_ids)
        if confirmed:
            self.run_checks()
        return confirmed

    def confirm_complete(self, workflow: Any, note: str = "") -> None:
        """Confirm data quality and advance only the M8 workflow states."""
        self.run_checks()
        counts = self.repository.database.counts()
        if counts["raw"] == 0:
            raise ValueError("At least one database record is required")
        if counts["pending"]:
            raise ValueError(f"{counts['pending']} Pending records must be resolved first")
        if self.unresolved_error_count:
            raise ValueError(f"{self.unresolved_error_count} unresolved ERROR issues remain")
        now = datetime.now(UTC)
        self.repository.set_quality_confirmation(now, note)
        import_step = workflow.get_step("import")
        clean_step = workflow.get_step("clean")
        if import_step is not None:
            import_step.set_metadata("raw_count", counts["raw"])
            import_step.set_metadata("excluded_count", counts["excluded"])
        if clean_step is not None:
            clean_step.set_metadata("excluded_count", counts["excluded"])
            clean_step.set_metadata("warning_count", self._warning_count())
            clean_step.set_metadata("confirmed_at", now.isoformat())
        workflow.complete_step("import")
        workflow.complete_step("clean")
        if self.repository.query(BoreholeDataType.COLLARS, RecordState.FORMAL):
            workflow.mark_ready("holdout")

    def export_json(self, path: Path) -> Path:
        """Export the complete persisted quality report as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.report()
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return target

    def export_csv(self, path: Path) -> Path:
        """Export quality findings as CSV."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = [self._issue_row(issue) for issue in self.repository.database.quality_issues]
        fieldnames = list(rows[0]) if rows else list(self._issue_row(None))
        with target.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return target

    def report(self) -> dict[str, Any]:
        """Return counts, confirmation state, and every finding."""
        database = self.repository.database
        return {
            "counts": database.counts(),
            "orientation_counts": database.orientation_counts(),
            "observation_mode_counts": database.observation_mode_counts(),
            "unresolved_error_count": self.unresolved_error_count,
            "quality_confirmed_at": database.quality_confirmed_at,
            "quality_confirmation_note": database.quality_confirmation_note,
            "issues": [self._issue_row(issue) for issue in database.quality_issues],
        }

    def _check_collar(self, record: BoreholeRecord, detected: dict[str, BoreholeQualityIssue]) -> None:
        for field in ("collar_x", "collar_y", "collar_z", "final_depth"):
            if not self.repository.is_number(record.values.get(field)):
                self._add(record, detected, "missing_or_non_numeric", field, f"{field} is required and must be numeric")
        if self.repository.is_number(record.values.get("final_depth")) and float(record.values["final_depth"]) <= 0:
            self._add(record, detected, "invalid_total_depth", "final_depth", "Total depth must be greater than zero")

    def _check_survey(self, record: BoreholeRecord, detected: dict[str, BoreholeQualityIssue]) -> None:
        values = record.values
        for field in ("measured_depth", "azimuth", "dip"):
            if not self.repository.is_number(values.get(field)):
                self._add(record, detected, "survey_non_numeric", field, f"Survey {field} must be numeric")
                return
        azimuth = float(values["azimuth"])
        dip = float(values["dip"])
        if not 0 <= azimuth < 360:
            self._add(
                record,
                detected,
                "survey_azimuth_range",
                "azimuth",
                "Survey azimuth must be in [0, 360)",
                suggested=azimuth % 360.0,
                auto=True,
            )
        if not -90 <= dip <= 90:
            self._add(
                record,
                detected,
                "survey_dip_range",
                "dip",
                "Survey dip must be in [-90, 90]",
                suggested=max(-90.0, min(90.0, dip)),
                auto=True,
            )

    def _check_fracture(self, record: BoreholeRecord, detected: dict[str, BoreholeQualityIssue]) -> None:
        values = record.values
        mode = values.get("observation_mode")
        if mode == FractureObservationMode.INTERVAL_SPACING:
            for field in ("from_depth", "to_depth", "fracture_spacing"):
                if not self.repository.is_number(values.get(field)):
                    self._add(record, detected, "spacing_non_numeric", field, f"Spacing {field} must be numeric")
                    return
            if float(values["fracture_spacing"]) <= 0:
                self._add(record, detected, "spacing_positive", "fracture_spacing", "fracture_spacing must be > 0")
            self._check_interval(record, detected, "spacing")
            return
        if mode == FractureObservationMode.AXIS_PLANE_ANGLE:
            for field in ("depth", "axis_plane_angle"):
                if not self.repository.is_number(values.get(field)):
                    self._add(record, detected, "axis_angle_non_numeric", field, f"{field} must be numeric")
                    return
            if not 0 <= float(values["axis_plane_angle"]) <= 90:
                self._add(record, detected, "axis_angle_range", "axis_plane_angle", "axis_plane_angle must be in [0, 90]")
            self._add(
                record,
                detected,
                "borehole_relative_orientation",
                "axis_plane_angle",
                "Borehole-relative angle is retained but direction completion is not implemented.",
                severity=QualitySeverity.INFO,
            )
            return
        for field in ("depth", "dip"):
            if not self.repository.is_number(values.get(field)):
                self._add(record, detected, "fracture_non_numeric", field, f"Fracture {field} must be numeric")
                return
        dip = float(values["dip"])
        dip_direction = values.get("dip_direction")
        if dip_direction is None:
            self._add(
                record,
                detected,
                "fracture_dip_only",
                "dip_direction",
                "Orientation incomplete: dip-only fracture record. This record is retained but is not eligible for full 3D orientation analysis.",
                severity=QualitySeverity.INFO,
            )
        elif not self.repository.is_number(dip_direction):
            self._add(
                record,
                detected,
                "fracture_direction_non_numeric",
                "dip_direction",
                "Dip direction must be numeric when provided",
            )
        else:
            numeric_direction = float(dip_direction)
            if numeric_direction == 360.0:
                self._add(
                    record,
                    detected,
                    "fracture_direction_normalize",
                    "dip_direction",
                    "Dip direction 360 degrees is equivalent to 0 degrees",
                    suggested=0.0,
                    auto=True,
                )
            elif not 0 <= numeric_direction < 360:
                self._add(
                    record,
                    detected,
                    "fracture_direction_range",
                    "dip_direction",
                    "Dip direction must be in [0, 360]",
                )
        if not 0 <= dip <= 90:
            self._add(record, detected, "fracture_dip_range", "dip", "Fracture dip must be in [0, 90]")
        set_id = values.get("set_id")
        if set_id not in (None, "") and self.repository.optional_int(set_id) is None:
            self._add(record, detected, "fracture_set_id", "set_id", "set_id must be an integer")

    def _check_rqd(self, record: BoreholeRecord, detected: dict[str, BoreholeQualityIssue]) -> None:
        values = record.values
        for field in ("from_depth", "to_depth", "rqd"):
            if not self.repository.is_number(values.get(field)):
                self._add(record, detected, "rqd_non_numeric", field, f"RQD {field} must be numeric")
                return
        if not 0 <= float(values["rqd"]) <= 100:
            self._add(record, detected, "rqd_range", "rqd", "RQD must be in [0, 100]")
        self._check_interval(record, detected, "rqd")

    def _check_rmr(self, record: BoreholeRecord, detected: dict[str, BoreholeQualityIssue]) -> None:
        values = record.values
        for field in ("from_depth", "to_depth", "rmr"):
            if not self.repository.is_number(values.get(field)):
                self._add(record, detected, "rmr_non_numeric", field, f"RMR {field} must be numeric")
                return
        if not 0 <= float(values["rmr"]) <= 100:
            self._add(record, detected, "rmr_range", "rmr", "RMR must be in [0, 100]")
        self._check_interval(record, detected, "rmr")

    def _check_orientation_point(
        self, record: BoreholeRecord, detected: dict[str, BoreholeQualityIssue]
    ) -> None:
        values = record.values
        if not str(values.get("observation_id", "")).strip():
            self._add(record, detected, "orientation_observation_id", "observation_id", "observation_id is required")
        if not str(values.get("point_id", "")).strip():
            self._add(record, detected, "orientation_point_id", "point_id", "point_id is required")
        for field in ("x", "y", "z"):
            if not self.repository.is_number(values.get(field)):
                self._add(record, detected, "orientation_point_non_numeric", field, f"{field} must be numeric")
                return
        state, reason = self.repository._classify(BoreholeDataType.ORIENTATION_POINTS, values)
        if state == RecordState.EXCLUDED and reason:
            field = reason.split("field ", 1)[1].split(" ", 1)[0] if "field " in reason else "*"
            self._add(record, detected, "orientation_point_contract", field, reason)

    def _check_domain(self, record: BoreholeRecord, detected: dict[str, BoreholeQualityIssue]) -> None:
        values = record.values
        for field in ("from_depth", "to_depth", "domain_id"):
            if not self.repository.is_number(values.get(field)):
                self._add(record, detected, "domain_non_numeric", field, f"Domain {field} must be numeric")
                return
        self._check_interval(record, detected, "domain")

    def _check_interval(
        self,
        record: BoreholeRecord,
        detected: dict[str, BoreholeQualityIssue],
        prefix: str,
    ) -> None:
        from_depth = float(record.values["from_depth"])
        to_depth = float(record.values["to_depth"])
        if from_depth < 0 or from_depth >= to_depth:
            self._add(
                record,
                detected,
                f"{prefix}_interval_order",
                "from_depth/to_depth",
                "Interval requires 0 <= from_depth < to_depth",
            )

    def _check_overlaps(self, data_type: BoreholeDataType, detected: dict[str, BoreholeQualityIssue]) -> None:
        grouped: dict[str, list[BoreholeRecord]] = defaultdict(list)
        for record in self.repository.query(data_type, RecordState.FORMAL):
            if self.repository.is_number(record.values.get("from_depth")) and self.repository.is_number(
                record.values.get("to_depth")
            ):
                grouped[record.hole_id].append(record)
        for records in grouped.values():
            ordered = sorted(records, key=lambda record: (float(record.values["from_depth"]), record.record_id))
            if not ordered:
                continue
            furthest = ordered[0]
            for current in ordered[1:]:
                if float(furthest.values["to_depth"]) > float(current.values["from_depth"]):
                    code = f"{data_type.value}_overlap:{furthest.record_id}"
                    self._add(
                        current,
                        detected,
                        code,
                        "from_depth/to_depth",
                        f"Interval overlaps record {furthest.record_id}",
                        severity=QualitySeverity.WARNING,
                    )
                if float(current.values["to_depth"]) > float(furthest.values["to_depth"]):
                    furthest = current

    def _add(
        self,
        record: BoreholeRecord,
        detected: dict[str, BoreholeQualityIssue],
        code: str,
        field: str,
        reason: str,
        suggested: Any = None,
        auto: bool = False,
        severity: QualitySeverity = QualitySeverity.ERROR,
    ) -> None:
        issue = self._issue(
            record,
            code=code,
            field=field,
            severity=severity,
            reason=reason,
            suggested=suggested,
            auto=auto,
        )
        detected[issue.issue_key] = issue

    def _issue(
        self,
        record: BoreholeRecord,
        code: str,
        field: str,
        severity: QualitySeverity,
        reason: str,
        suggested: Any = None,
        auto: bool = False,
    ) -> BoreholeQualityIssue:
        key_text = f"{record.record_id}|{code}|{field}"
        issue_key = sha256(key_text.encode("utf-8")).hexdigest()
        current = record.values.get(field) if "/" not in field and field != "*" else None
        original = record.original_values.get(field) if "/" not in field and field != "*" else None
        return BoreholeQualityIssue(
            issue_key=issue_key,
            record_id=record.record_id,
            data_type=record.data_type,
            hole_id=record.hole_id,
            source_file=record.source_file,
            source_row=record.source_row,
            field=field,
            code=code,
            severity=severity,
            original_value=original,
            current_value=current,
            suggested_value=suggested,
            reason=reason,
            auto_fix_available=auto,
            orientation_completeness=record.values.get("orientation_completeness"),
        )

    def _merge_findings(self, detected: dict[str, BoreholeQualityIssue]) -> None:
        database = self.repository.database
        issues = [issue.model_copy(deep=True) for issue in database.quality_issues]
        existing = {issue.issue_key: issue for issue in issues}
        for key, finding in detected.items():
            prior = existing.get(key)
            if prior is None:
                issues.append(finding)
                continue
            prior.current_value = finding.current_value
            prior.suggested_value = finding.suggested_value
            prior.reason = finding.reason
            prior.auto_fix_available = finding.auto_fix_available
            prior.orientation_completeness = finding.orientation_completeness
            if finding.status == QualityIssueStatus.CONFIRMED:
                prior.status = QualityIssueStatus.CONFIRMED
                prior.resolved_at = finding.resolved_at
                prior.resolution_note = finding.resolution_note
            elif prior.status in {
                QualityIssueStatus.RESOLVED,
                QualityIssueStatus.APPLIED,
                QualityIssueStatus.CONFIRMED,
            }:
                prior.status = QualityIssueStatus.OPEN
                prior.resolved_at = None
                prior.resolution_note = ""
        detected_keys = set(detected)
        for issue in issues:
            if issue.issue_key not in detected_keys and issue.status == QualityIssueStatus.OPEN:
                issue.status = QualityIssueStatus.RESOLVED
                issue.resolved_at = datetime.now(UTC)
                issue.resolution_note = "Condition no longer present"
        self.repository.set_quality_issues(issues)

    def _warning_count(self) -> int:
        return sum(
            issue.severity == QualitySeverity.WARNING and issue.status == QualityIssueStatus.OPEN
            for issue in self.repository.database.quality_issues
        )

    @staticmethod
    def _issue_row(issue: BoreholeQualityIssue | None) -> dict[str, Any]:
        fields = {
            "issue_id": "",
            "record_id": "",
            "data_type": "",
            "hole_id": "",
            "source_file": "",
            "source_row": "",
            "field": "",
            "code": "",
            "severity": "",
            "status": "",
            "original_value": "",
            "current_value": "",
            "suggested_value": "",
            "reason": "",
            "resolution_note": "",
            "orientation_completeness": "",
        }
        if issue is None:
            return fields
        fields.update(
            {
                "issue_id": issue.issue_id,
                "record_id": issue.record_id,
                "data_type": issue.data_type,
                "hole_id": issue.hole_id,
                "source_file": issue.source_file,
                "source_row": issue.source_row,
                "field": issue.field,
                "code": issue.code,
                "severity": issue.severity.value,
                "status": issue.status.value,
                "original_value": issue.original_value,
                "current_value": issue.current_value,
                "suggested_value": issue.suggested_value,
                "reason": issue.reason,
                "resolution_note": issue.resolution_note,
                "orientation_completeness": issue.orientation_completeness or "",
            }
        )
        return fields
