"""Repository/service layer for the canonical M8 borehole database."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from dfn_cave_studio.models.borehole import (
    Borehole,
    BoreholeCollection,
    Collar,
    FractureObservation,
    OrientationCompleteness,
    RQDInterval,
    SurveyStation,
)
from dfn_cave_studio.models.borehole_database import (
    BoreholeDatabase,
    BoreholeDataType,
    BoreholeQualityIssue,
    BoreholeRecord,
    ModificationEvent,
    RecordState,
)
from dfn_cave_studio.models.data_management import DomainInterval
from dfn_cave_studio.services.field_mapping import (
    COLLAR_FIELD_ALIASES,
    detect_field_mapping,
    normalize_header,
    validate_field_mapping,
)

_ALIASES: dict[str, dict[str, str]] = {
    BoreholeDataType.COLLARS: {
        "hole_id": "borehole_id",
        "easting": "collar_x",
        "northing": "collar_y",
        "elevation": "collar_z",
        "total_depth": "final_depth",
    },
    BoreholeDataType.SURVEYS: {"borehole_id": "hole_id", "depth": "measured_depth"},
    BoreholeDataType.FRACTURES: {"borehole_id": "hole_id", "measured_depth": "depth"},
    BoreholeDataType.RQD: {"borehole_id": "hole_id", "rqd_value": "rqd"},
    BoreholeDataType.DOMAIN_INTERVALS: {"borehole_id": "hole_id"},
}


class ImportBatchResult(dict):
    """Import counts with dictionary compatibility."""

    @property
    def changed(self) -> bool:
        """Whether the repository received new raw records."""
        return int(self.get("raw", 0)) > 0


class BoreholeRepository:
    """The only service that mutates the project borehole database.

    The canonical store is ``project.borehole_database``. The legacy
    ``project.borehole_collection`` is rebuilt as a compatibility projection
    containing only formal records.
    """

    def __init__(self, project: Any):
        self.project = project
        if getattr(project, "borehole_database", None) is None:
            project.borehole_database = BoreholeDatabase()

    @property
    def database(self) -> BoreholeDatabase:
        """Return the canonical database."""
        return self.project.borehole_database

    def snapshot(self) -> BoreholeDatabase:
        """Return a deep copy suitable for dialog rollback."""
        return self.database.model_copy(deep=True)

    def restore(self, snapshot: BoreholeDatabase) -> None:
        """Restore a previously captured database snapshot."""
        self.project.borehole_database = snapshot.model_copy(deep=True)
        self.rebuild_formal_collection()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Rollback database, formal projection, legacy domains, and workflow on failure."""
        database_snapshot = self.snapshot()
        collection_snapshot = self.project.borehole_collection.model_copy(deep=True)
        m7_data = getattr(self.project, "_m7_data", None)
        domain_snapshot = deepcopy((m7_data or {}).get("domain_intervals", []))
        workflow = (m7_data or {}).get("workflow")
        workflow_snapshot = workflow.to_dict() if workflow is not None and hasattr(workflow, "to_dict") else None
        try:
            yield
        except Exception:
            self.project.borehole_database = database_snapshot
            self.project.borehole_collection = collection_snapshot
            if m7_data is not None:
                m7_data["domain_intervals"] = domain_snapshot
            if workflow_snapshot is not None and hasattr(workflow, "from_dict"):
                workflow.from_dict(workflow_snapshot)
            raise

    def query(
        self,
        data_type: str | None = None,
        state: RecordState | str | None = None,
        hole_id: str | None = None,
        raw: bool = False,
    ) -> list[BoreholeRecord]:
        """Query canonical records."""
        return self.database.query(data_type=data_type, state=state, hole_id=hole_id, raw=raw)

    def import_dataframe(
        self,
        data_type: str | BoreholeDataType,
        dataframe: pd.DataFrame,
        source_file: str,
        mode: str = "append",
        modification_source: str = "import",
        field_mapping: Mapping[str, str] | None = None,
    ) -> ImportBatchResult:
        """Import one table independently and atomically.

        Args:
            data_type: Built-in table name or a future extension table.
            dataframe: Mapped source rows. The caller retains ownership.
            source_file: Original source path/name.
            mode: ``append``, ``replace``, or ``cancel``.
            modification_source: Audit label for the change.

        Returns:
            Raw/formal/excluded/pending/duplicate counts for this request.
        """
        dtype = data_type.value if isinstance(data_type, BoreholeDataType) else str(data_type)
        if mode not in {"append", "replace", "cancel"}:
            raise ValueError("mode must be append, replace, or cancel")
        if mode == "cancel":
            return ImportBatchResult(raw=0, formal=0, excluded=0, pending=0, duplicates=0, cancelled=True)

        if dtype == BoreholeDataType.COLLARS:
            if field_mapping is None:
                source_field_mapping = detect_field_mapping(
                    dataframe.columns, COLLAR_FIELD_ALIASES
                ).require_unambiguous()
            else:
                source_field_mapping = validate_field_mapping(
                    dataframe.columns, field_mapping, COLLAR_FIELD_ALIASES
                )
        else:
            source_field_mapping = dict(field_mapping or {})
        normalized_headers = {str(column): normalize_header(column) for column in dataframe.columns}

        batch_id = str(uuid4())
        pending_records: list[BoreholeRecord] = []
        existing_signatures = {
            self._signature(record.data_type, record.original_values): record.record_id
            for record in self.database.records
        }
        duplicates = 0
        for row_number, (_, row) in enumerate(dataframe.iterrows()):
            original = self._json_values(row.to_dict())
            mapped_values = dict(original)
            for source, standard in source_field_mapping.items():
                mapped_values[standard] = original.get(source)
            values = self._standardize(dtype, mapped_values)
            signature = self._signature(dtype, original)
            if signature in existing_signatures:
                duplicates += 1
                continue
            hole_id = self._hole_id(dtype, values)
            state, reason = self._classify(dtype, values)
            record = BoreholeRecord(
                data_type=dtype,
                hole_id=hole_id,
                source_file=str(source_file),
                source_row=row_number,
                original_values=original,
                values=values,
                source_field_mapping=dict(source_field_mapping),
                normalized_source_headers=dict(normalized_headers),
                state=state,
                exclusion_reason=reason,
                modification_source=modification_source,
                import_batch_id=batch_id,
            )
            pending_records.append(record)
            existing_signatures[signature] = record.record_id

        if mode == "replace" and pending_records:
            now = datetime.now(UTC)
            for record in self.database.records:
                if record.data_type == dtype and record.state != RecordState.EXCLUDED:
                    before = record.values.copy()
                    record.state = RecordState.EXCLUDED
                    record.exclusion_reason = f"Superseded by replacement batch {batch_id}"
                    record.modification_source = modification_source
                    record.modification_history.append(
                        ModificationEvent(
                            changed_at=now,
                            source=modification_source,
                            action="replace",
                            before=before,
                            after=None,
                        )
                    )

        self.database.records.extend(pending_records)
        if dtype == BoreholeDataType.COLLARS:
            self.reconcile_pending()
        else:
            self._reclassify_records(pending_records)
            self.rebuild_formal_collection()
        if pending_records and modification_source not in {"m7_migration", "staged_import"}:
            self.invalidate_dependent_results(dtype)
        if pending_records:
            self._mark_quality_stale()
        counts = {
            "raw": len(pending_records),
            "formal": sum(record.state == RecordState.FORMAL for record in pending_records),
            "excluded": sum(record.state == RecordState.EXCLUDED for record in pending_records),
            "pending": sum(record.state == RecordState.PENDING for record in pending_records),
            "duplicates": duplicates,
            "cancelled": False,
            "data_type": dtype,
        }
        if dtype == BoreholeDataType.FRACTURES:
            counts["total_fracture_rows"] = len(dataframe)
            counts["full_orientation"] = sum(
                record.values.get("orientation_completeness") == OrientationCompleteness.FULL_ORIENTATION.value
                and record.state != RecordState.EXCLUDED
                for record in pending_records
            )
            counts["dip_only"] = sum(
                record.values.get("orientation_completeness") == OrientationCompleteness.DIP_ONLY.value
                and record.state != RecordState.EXCLUDED
                for record in pending_records
            )
            counts["excluded_error"] = counts["excluded"]
        return ImportBatchResult(counts)

    def edit_record(
        self,
        record_id: str,
        values: Mapping[str, Any],
        modification_source: str = "user_edit",
    ) -> BoreholeRecord:
        """Edit cleaned values while retaining raw values and change history."""
        record = self._get(record_id)
        before = deepcopy(record.values)
        record.values = self._standardize(record.data_type, self._json_values(dict(values)))
        record.hole_id = self._hole_id(record.data_type, record.values)
        record.state = RecordState.PENDING
        record.state, record.exclusion_reason = self._classify(record.data_type, record.values)
        record.modification_source = modification_source
        record.modification_history.append(
            ModificationEvent(source=modification_source, action="edit", before=before, after=deepcopy(record.values))
        )
        if record.data_type == BoreholeDataType.COLLARS:
            self.reconcile_pending()
        else:
            self.rebuild_formal_collection()
        self.invalidate_dependent_results(record.data_type)
        self._mark_quality_stale()
        return record

    def apply_joint_set_assignments(
        self,
        assignments: Mapping[str, int],
        *,
        modification_source: str = "m8_joint_set:auto_assignment",
    ) -> int:
        """Commit Formal-fracture joint-set assignments in one audited batch.

        Assignment keys use the ``hole_id:observation_index`` references
        produced from the current Formal ``BoreholeCollection``.  They are
        resolved to canonical ``BoreholeRecord`` instances before any record
        is mutated, then the Formal projection is rebuilt exactly once.

        Args:
            assignments: Formal observation reference to positive set ID.
            modification_source: Audit source stored in ModificationEvent.

        Returns:
            Number of fracture records whose cleaned ``set_id`` changed.
        """
        if not assignments:
            return 0
        by_hole: dict[str, list[BoreholeRecord]] = {}
        for record in self.query(BoreholeDataType.FRACTURES, RecordState.FORMAL):
            by_hole.setdefault(record.hole_id, []).append(record)

        resolved: list[tuple[BoreholeRecord, int]] = []
        seen_records: set[str] = set()
        for reference, raw_set_id in assignments.items():
            try:
                hole_id, raw_index = str(reference).rsplit(":", 1)
                observation_index = int(raw_index)
                set_id = int(raw_set_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid joint-set assignment reference: {reference!r}") from exc
            if set_id <= 0:
                raise ValueError(f"Joint-set assignment for {reference} must use a positive set_id")
            records = by_hole.get(hole_id, [])
            if observation_index < 0 or observation_index >= len(records):
                raise KeyError(f"Formal fracture observation not found for assignment: {reference}")
            record = records[observation_index]
            if record.record_id in seen_records:
                raise ValueError(f"Formal fracture observation assigned more than once: {reference}")
            seen_records.add(record.record_id)
            resolved.append((record, set_id))

        changed = 0
        with self.transaction():
            for record, set_id in resolved:
                current = self._optional_int(record.values.get("set_id"))
                if current == set_id:
                    continue
                before = deepcopy(record.values)
                record.values = deepcopy(record.values)
                record.values["set_id"] = set_id
                record.modification_source = modification_source
                record.modification_history.append(
                    ModificationEvent(
                        source=modification_source,
                        action="assign_joint_set",
                        before=before,
                        after=deepcopy(record.values),
                    )
                )
                changed += 1
            if changed:
                self.rebuild_formal_collection()
                workflow = (getattr(self.project, "_m7_data", {}) or {}).get("workflow")
                if workflow is not None and hasattr(workflow, "invalidate_steps"):
                    workflow.invalidate_steps(
                        ["density", "size", "parameter_field", "validation", "explicit_dfn", "second_voxelization"]
                    )
        return changed

    def apply_grouped_quality_fixes(
        self,
        fixes_by_record: Mapping[str, Iterable[BoreholeQualityIssue]],
    ) -> tuple[int, int]:
        """Apply grouped quality fixes with one classification/rebuild cycle.

        Every issue produces its own audit event, while all fields belonging
        to one record are merged into a single cleaned-value update.
        """
        changed_records: list[BoreholeRecord] = []
        changed_types: set[str] = set()
        field_count = 0
        for record_id, raw_issues in fixes_by_record.items():
            record = self._get(record_id)
            before = deepcopy(record.values)
            after = deepcopy(record.values)
            applied_issues: list[BoreholeQualityIssue] = []
            for issue in raw_issues:
                if (
                    issue.record_id != record_id
                    or issue.status != "open"
                    or not issue.auto_fix_available
                    or not issue.field
                ):
                    continue
                if after.get(issue.field) == issue.suggested_value:
                    continue
                after[issue.field] = issue.suggested_value
                applied_issues.append(issue)
            if not applied_issues:
                continue
            standardized = self._standardize(record.data_type, self._json_values(after))
            record.values = standardized
            record.hole_id = self._hole_id(record.data_type, standardized)
            record.modification_source = "m8_quality:auto_fix_batch"
            final_values = deepcopy(standardized)
            for issue in applied_issues:
                record.modification_history.append(
                    ModificationEvent(
                        source=f"m8_quality:auto_fix:{issue.code}",
                        action="auto_fix",
                        issue_id=issue.issue_id,
                        before=deepcopy(before),
                        after=deepcopy(final_values),
                    )
                )
            changed_records.append(record)
            changed_types.add(record.data_type)
            field_count += len(applied_issues)

        if not changed_records:
            return 0, 0
        self._reclassify_records(changed_records)
        self.rebuild_formal_collection()
        self.invalidate_dependent_results_batch(changed_types)
        self._mark_quality_stale()
        return len(changed_records), field_count

    def delete_record(self, record_id: str, modification_source: str = "user_delete") -> BoreholeRecord:
        """Soft-delete a record into Excluded state for traceability."""
        record = self._get(record_id)
        before = deepcopy(record.values)
        record.state = RecordState.EXCLUDED
        record.exclusion_reason = "Deleted by user"
        record.modification_source = modification_source
        record.modification_history.append(
            ModificationEvent(source=modification_source, action="delete", before=before, after=None)
        )
        if record.data_type == BoreholeDataType.COLLARS:
            self.reconcile_pending()
        else:
            self.rebuild_formal_collection()
        self.invalidate_dependent_results(record.data_type)
        self._mark_quality_stale()
        return record

    def exclude_record(
        self,
        record_id: str,
        reason: str,
        modification_source: str = "user_exclude",
    ) -> BoreholeRecord:
        """Exclude a record with a mandatory audited reason."""
        if not reason.strip():
            raise ValueError("An exclusion reason is required")
        record = self._get(record_id)
        before = deepcopy(record.values)
        record.state = RecordState.EXCLUDED
        record.exclusion_reason = reason.strip()
        record.exclusion_confirmed_at = None
        record.modification_source = modification_source
        record.modification_history.append(
            ModificationEvent(
                source=modification_source, action="exclude", before=before, after=deepcopy(record.values)
            )
        )
        self.rebuild_formal_collection()
        self.invalidate_dependent_results(record.data_type)
        self._mark_quality_stale()
        return record

    def reclassify_record(
        self,
        record_id: str,
        modification_source: str = "user_retry",
    ) -> BoreholeRecord:
        """Retry classification of an Excluded or Pending record."""
        record = self._get(record_id)
        before_state = record.state
        record.state = RecordState.PENDING
        record.exclusion_reason = None
        record.exclusion_confirmed_at = None
        record.state, record.exclusion_reason = self._classify(record.data_type, record.values)
        record.modification_source = modification_source
        record.modification_history.append(
            ModificationEvent(
                source=modification_source,
                action=f"reclassify:{before_state.value}_to_{record.state.value}",
                before=deepcopy(record.values),
                after=deepcopy(record.values),
            )
        )
        if record.data_type == BoreholeDataType.COLLARS:
            self.reconcile_pending()
        else:
            self.rebuild_formal_collection()
        self.invalidate_dependent_results(record.data_type)
        self._mark_quality_stale()
        return record

    def get_record(self, record_id: str) -> BoreholeRecord:
        """Return one record by ID."""
        return self._get(record_id)

    @staticmethod
    def is_number(value: Any) -> bool:
        """Return whether a value is a finite number."""
        return BoreholeRepository._is_number(value)

    @staticmethod
    def optional_int(value: Any) -> int | None:
        """Parse an optional integer without coercing non-integral values."""
        return BoreholeRepository._optional_int(value)

    def set_quality_issues(self, issues: Iterable[BoreholeQualityIssue]) -> None:
        """Replace the persisted quality-issue projection through the write boundary."""
        self.database.quality_issues = [issue.model_copy(deep=True) for issue in issues]

    def set_quality_confirmation(self, confirmed_at: datetime | None, note: str = "") -> None:
        """Set or clear the project-level quality gate confirmation."""
        self.database.quality_confirmed_at = confirmed_at
        self.database.quality_confirmation_note = note

    def confirm_exclusions(
        self,
        record_ids: Iterable[str] | None = None,
        modification_source: str = "m8_quality:confirm_exclusion",
    ) -> int:
        """Confirm documented exclusions and append one idempotent audit event."""
        selected = set(record_ids) if record_ids is not None else None
        now = datetime.now(UTC)
        confirmed = 0
        for record in self.database.records:
            if (
                record.state != RecordState.EXCLUDED
                or not record.exclusion_reason
                or (selected is not None and record.record_id not in selected)
                or record.exclusion_confirmed_at is not None
            ):
                continue
            record.exclusion_confirmed_at = now
            record.modification_history.append(
                ModificationEvent(
                    action="confirm_exclusion",
                    before={"exclusion_confirmed_at": None, "reason": record.exclusion_reason},
                    after={"exclusion_confirmed_at": now, "reason": record.exclusion_reason},
                    source=modification_source,
                )
            )
            confirmed += 1
        return confirmed

    def invalidate_dependent_results(self, data_type: str) -> None:
        """Mark only workflow results that depend on the changed table stale."""
        self.invalidate_dependent_results_batch([data_type])

    def invalidate_dependent_results_batch(self, data_types: Iterable[str]) -> None:
        """Invalidate the union of dependent results in one workflow update."""
        workflow = (getattr(self.project, "_m7_data", {}) or {}).get("workflow")
        if workflow is None or not hasattr(workflow, "invalidate_steps"):
            return
        dependencies = {
            BoreholeDataType.COLLARS: ["clean", "holdout", "domains", "joint_sets", "bounds", "voxel_grid"],
            BoreholeDataType.SURVEYS: ["clean", "bounds", "voxel_grid"],
            BoreholeDataType.FRACTURES: ["clean", "joint_sets", "bounds", "voxel_grid"],
            BoreholeDataType.RQD: ["clean"],
            BoreholeDataType.DOMAIN_INTERVALS: ["clean", "domains"],
        }
        affected = {step_id for data_type in data_types for step_id in dependencies.get(data_type, ["clean"])}
        workflow.invalidate_steps(sorted(affected))

    def reconcile_pending(self) -> int:
        """Re-evaluate related rows after collars are added, edited, or removed."""
        changed = 0
        for record in self.database.records:
            if record.data_type == BoreholeDataType.COLLARS or record.state == RecordState.EXCLUDED:
                continue
            previous = record.state
            state, reason = self._classify(record.data_type, record.values)
            record.state = state
            record.exclusion_reason = reason
            if state != previous:
                changed += 1
                record.modification_history.append(
                    ModificationEvent(source="automatic_relink", action=f"{previous.value}_to_{state.value}")
                )
                record.modification_source = "automatic_relink"
        self.rebuild_formal_collection()
        return changed

    def rebuild_formal_collection(self) -> BoreholeCollection:
        """Rebuild downstream boreholes exclusively from formal records."""
        collection = BoreholeCollection()
        collar_records = self.query(BoreholeDataType.COLLARS, RecordState.FORMAL)
        boreholes: dict[str, Borehole] = {}
        for record in collar_records:
            values = record.values
            try:
                collar = Collar(
                    borehole_id=record.hole_id,
                    collar_x=float(values["collar_x"]),
                    collar_y=float(values["collar_y"]),
                    collar_z=float(values["collar_z"]),
                    azimuth=float(values.get("azimuth", 0.0) or 0.0),
                    dip=float(values.get("dip", -90.0) if values.get("dip") is not None else -90.0),
                    final_depth=float(values["final_depth"]),
                )
                boreholes[record.hole_id] = Borehole(borehole_id=record.hole_id, collar=collar)
            except (KeyError, TypeError, ValueError):
                continue

        for record in self.query(BoreholeDataType.SURVEYS, RecordState.FORMAL):
            if record.hole_id not in boreholes:
                continue
            values = record.values
            try:
                station = SurveyStation(
                    measured_depth=float(values["measured_depth"]),
                    azimuth=float(values["azimuth"]),
                    dip=float(values["dip"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            boreholes[record.hole_id].survey.stations.append(station)
        for borehole in boreholes.values():
            borehole.survey.stations.sort(key=lambda station: station.measured_depth)

        for record in self.query(BoreholeDataType.FRACTURES, RecordState.FORMAL):
            if record.hole_id not in boreholes:
                continue
            values = record.values
            try:
                observation = FractureObservation(
                    borehole_id=record.hole_id,
                    measured_depth=float(values["depth"]),
                    dip_direction=self._optional_float(values.get("dip_direction")),
                    dip=float(values["dip"]),
                    aperture=self._optional_float(values.get("aperture")),
                    set_id=self._optional_int(values.get("set_id")),
                )
            except (KeyError, TypeError, ValueError):
                continue
            boreholes[record.hole_id].fracture_observations.append(observation)

        for record in self.query(BoreholeDataType.RQD, RecordState.FORMAL):
            if record.hole_id not in boreholes:
                continue
            values = record.values
            boreholes[record.hole_id].rqd_intervals.append(
                RQDInterval(
                    borehole_id=record.hole_id,
                    from_depth=float(values["from_depth"]),
                    to_depth=float(values["to_depth"]),
                    rqd_value=float(values["rqd"]),
                    core_recovery=self._optional_float(values.get("core_recovery")),
                )
            )
        for borehole in boreholes.values():
            collection.add(borehole)
        self.project.borehole_collection = collection

        domain_records = self.query(BoreholeDataType.DOMAIN_INTERVALS, raw=True)
        domains = []
        for record in self.query(BoreholeDataType.DOMAIN_INTERVALS, RecordState.FORMAL):
            values = record.values
            domains.append(
                DomainInterval(
                    hole_id=record.hole_id,
                    from_depth=float(values["from_depth"]),
                    to_depth=float(values["to_depth"]),
                    domain_id=int(float(values["domain_id"])),
                    source_file=record.source_file,
                    source_row=record.source_row,
                )
            )
        m7_data = getattr(self.project, "_m7_data", None)
        if m7_data is None:
            self.project._m7_data = {}
            m7_data = self.project._m7_data
        if domain_records:
            m7_data["domain_intervals"] = domains
        return collection

    def migrate_m7(self) -> BoreholeDatabase:
        """Populate an empty M8 database from a loaded M7 project."""
        if self.database.records:
            return self.database
        m7 = getattr(self.project, "_m7_data", {}) or {}
        legacy_domain_intervals = list(m7.get("domain_intervals", []))
        collection = getattr(self.project, "borehole_collection", BoreholeCollection())

        collar_rows = []
        for borehole in collection:
            collar_rows.append(
                {
                    "borehole_id": borehole.borehole_id,
                    "collar_x": borehole.collar.collar_x,
                    "collar_y": borehole.collar.collar_y,
                    "collar_z": borehole.collar.collar_z,
                    "azimuth": borehole.collar.azimuth,
                    "dip": borehole.collar.dip,
                    "final_depth": borehole.collar.final_depth,
                }
            )
        if collar_rows:
            self.import_dataframe(
                BoreholeDataType.COLLARS,
                pd.DataFrame(collar_rows),
                "m7:inputs/boreholes.json",
                modification_source="m7_migration",
            )

        raw_tables = {
            BoreholeDataType.SURVEYS: m7.get("raw_surveys"),
            BoreholeDataType.FRACTURES: m7.get("raw_fractures"),
            BoreholeDataType.RQD: m7.get("raw_rqd"),
            BoreholeDataType.DOMAIN_INTERVALS: m7.get("raw_domain_intervals"),
        }
        for dtype, dataframe in raw_tables.items():
            if dataframe is not None and len(dataframe):
                self.import_dataframe(
                    dtype,
                    dataframe,
                    f"m7:{dtype.value}.csv",
                    modification_source="m7_migration",
                )

        # M7's formal projection already contains its accepted survey cleaning.
        # Preserve those values explicitly in the canonical database and record
        # the migration event rather than reintroducing the former silent
        # trajectory-time normalization.
        legacy_surveys = {
            (borehole.borehole_id, float(station.measured_depth)): station
            for borehole in collection
            for station in borehole.survey.stations
        }
        for record in self.query(BoreholeDataType.SURVEYS, RecordState.FORMAL):
            if not self._is_number(record.values.get("measured_depth")):
                continue
            station = legacy_surveys.get((record.hole_id, float(record.values["measured_depth"])))
            if station is None:
                continue
            before = deepcopy(record.values)
            record.values["azimuth"] = float(station.azimuth)
            record.values["dip"] = float(station.dip)
            if record.values != before:
                record.modification_source = "m7_migration:formal_survey"
                record.modification_history.append(
                    ModificationEvent(
                        source=record.modification_source,
                        action="migrate_formal_values",
                        before=before,
                        after=deepcopy(record.values),
                    )
                )

        # If a legacy file lacks raw tables, reconstruct the formal view.
        if raw_tables[BoreholeDataType.SURVEYS] is None:
            rows = [
                {
                    "hole_id": borehole.borehole_id,
                    "measured_depth": station.measured_depth,
                    "azimuth": station.azimuth,
                    "dip": station.dip,
                }
                for borehole in collection
                for station in borehole.survey.stations
            ]
            if rows:
                self.import_dataframe(
                    BoreholeDataType.SURVEYS,
                    pd.DataFrame(rows),
                    "m7:formal-surveys",
                    modification_source="m7_migration",
                )
        if raw_tables[BoreholeDataType.FRACTURES] is None:
            rows = [
                {
                    "hole_id": borehole.borehole_id,
                    "depth": observation.measured_depth,
                    "dip_direction": observation.dip_direction,
                    "dip": observation.dip,
                    "set_id": observation.set_id,
                    "aperture": observation.aperture,
                }
                for borehole in collection
                for observation in borehole.fracture_observations
            ]
            if rows:
                self.import_dataframe(
                    BoreholeDataType.FRACTURES,
                    pd.DataFrame(rows),
                    "m7:formal-fractures",
                    modification_source="m7_migration",
                )
        if raw_tables[BoreholeDataType.DOMAIN_INTERVALS] is None and legacy_domain_intervals:
            rows = [
                {
                    "hole_id": interval.hole_id,
                    "from_depth": interval.from_depth,
                    "to_depth": interval.to_depth,
                    "domain_id": interval.domain_id,
                    "domain_name": interval.domain_name,
                }
                for interval in legacy_domain_intervals
            ]
            self.import_dataframe(
                BoreholeDataType.DOMAIN_INTERVALS,
                pd.DataFrame(rows),
                "m7:formal-domains",
                modification_source="m7_migration",
            )
        self.rebuild_formal_collection()
        return self.database

    def migrate_orientation_completeness(self) -> int:
        """Migrate pre-v0.9.1 fracture records without changing a real legacy zero."""
        changed = 0
        for record in self.query(BoreholeDataType.FRACTURES, raw=True):
            if "orientation_completeness" in record.values:
                continue
            before = deepcopy(record.values)
            direction = record.values.get("dip_direction")
            if direction is None or (
                isinstance(direction, str) and direction.strip().lower() in {"", "na", "n/a", "null", "none"}
            ):
                record.values["dip_direction"] = None
                record.values["orientation_completeness"] = OrientationCompleteness.DIP_ONLY.value
                action = "migrate_dip_only_orientation"
            else:
                record.values["orientation_completeness"] = OrientationCompleteness.FULL_ORIENTATION.value
                action = "migrate_full_orientation"
                if self._is_number(direction) and float(direction) == 0.0:
                    action = "ambiguous_legacy_orientation_preserved_zero"
            record.modification_history.append(
                ModificationEvent(
                    source="v0.9.1_migration",
                    action=action,
                    before=before,
                    after=deepcopy(record.values),
                )
            )
            changed += 1
        self.database.schema_version = 2
        return changed

    def _classify(self, data_type: str, values: Mapping[str, Any]) -> tuple[RecordState, str | None]:
        if data_type not in {member.value for member in BoreholeDataType}:
            return RecordState.FORMAL, None
        hole_id = self._hole_id(data_type, values)
        if not hole_id:
            return RecordState.EXCLUDED, "Missing required hole_id"
        if data_type == BoreholeDataType.COLLARS:
            required = ("collar_x", "collar_y", "collar_z", "final_depth")
            if any(not self._is_number(values.get(field)) for field in required):
                return RecordState.EXCLUDED, "Missing or non-numeric collar coordinate/total depth"
            if float(values["final_depth"]) <= 0:
                return RecordState.EXCLUDED, "Total depth must be greater than zero"
            return RecordState.FORMAL, None

        collar = self._formal_collar(hole_id)
        if collar is None:
            return RecordState.PENDING, f"Collar '{hole_id}' is not available"
        total_depth = float(collar.values["final_depth"])
        if data_type == BoreholeDataType.SURVEYS:
            required = ("measured_depth", "azimuth", "dip")
            if any(not self._is_number(values.get(field)) for field in required):
                return RecordState.EXCLUDED, "Survey depth/orientation must be numeric"
            depth = float(values["measured_depth"])
            if depth < 0 or depth > total_depth:
                return RecordState.EXCLUDED, f"Survey depth {depth} exceeds valid range 0-{total_depth}"
            if any(
                record.data_type == BoreholeDataType.SURVEYS
                and record.hole_id == hole_id
                and record.state == RecordState.FORMAL
                and self._is_number(record.values.get("measured_depth"))
                and abs(float(record.values["measured_depth"]) - depth) < 1e-9
                for record in self.database.records
            ):
                return RecordState.EXCLUDED, f"Duplicate survey station at measured depth {depth}"
        elif data_type == BoreholeDataType.FRACTURES:
            required = ("depth", "dip")
            if any(not self._is_number(values.get(field)) for field in required):
                return RecordState.EXCLUDED, "Fracture depth/dip must be numeric"
            depth = float(values["depth"])
            dip = float(values["dip"])
            if depth < 0 or depth > total_depth:
                return RecordState.EXCLUDED, f"Fracture depth {depth} exceeds borehole total depth {total_depth}"
            if dip < 0 or dip > 90:
                return RecordState.EXCLUDED, f"Dip {dip} is outside [0, 90]"
            dip_direction = values.get("dip_direction")
            if dip_direction is not None:
                if not self._is_number(dip_direction):
                    return RecordState.EXCLUDED, "dip_direction must be numeric when provided"
                if not 0 <= float(dip_direction) <= 360:
                    return RecordState.EXCLUDED, f"Dip direction {dip_direction} is outside [0, 360]"
            set_id = values.get("set_id")
            if set_id is not None and set_id != "" and self._optional_int(set_id) is None:
                return RecordState.EXCLUDED, f"set_id '{set_id}' is not an integer"
        elif data_type in {BoreholeDataType.RQD, BoreholeDataType.DOMAIN_INTERVALS}:
            required = ("from_depth", "to_depth")
            if any(not self._is_number(values.get(field)) for field in required):
                return RecordState.EXCLUDED, "Interval depths must be numeric"
            from_depth = float(values["from_depth"])
            to_depth = float(values["to_depth"])
            if from_depth < 0 or from_depth >= to_depth or to_depth > total_depth:
                return RecordState.EXCLUDED, f"Interval [{from_depth}, {to_depth}] is outside borehole depth"
            if data_type == BoreholeDataType.RQD:
                if not self._is_number(values.get("rqd")) or not 0 <= float(values["rqd"]) <= 100:
                    return RecordState.EXCLUDED, "RQD must be numeric in [0, 100]"
            elif not self._is_number(values.get("domain_id")):
                return RecordState.EXCLUDED, "domain_id must be numeric"
        return RecordState.FORMAL, None

    def _formal_collar(self, hole_id: str) -> BoreholeRecord | None:
        for record in reversed(self.database.records):
            if (
                record.data_type == BoreholeDataType.COLLARS
                and record.hole_id == hole_id
                and record.state == RecordState.FORMAL
            ):
                return record
        return None

    @staticmethod
    def _standardize(data_type: str, values: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(values)
        for source, target in _ALIASES.get(data_type, {}).items():
            if target not in result and source in result:
                result[target] = result[source]
        if data_type == BoreholeDataType.FRACTURES:
            direction = result.get("dip_direction")
            if isinstance(direction, str) and direction.strip().lower() in {"", "na", "n/a", "null", "none"}:
                direction = None
            if direction is not None and BoreholeRepository._is_number(direction):
                direction = float(direction)
            result["dip_direction"] = direction
            result["orientation_completeness"] = (
                OrientationCompleteness.FULL_ORIENTATION.value
                if direction is not None
                else OrientationCompleteness.DIP_ONLY.value
            )
        return result

    @staticmethod
    def _hole_id(data_type: str, values: Mapping[str, Any]) -> str:
        key = "borehole_id" if data_type == BoreholeDataType.COLLARS else "hole_id"
        value = values.get(key, values.get("hole_id", values.get("borehole_id", "")))
        return "" if value is None else str(value).strip()

    @staticmethod
    def _json_values(values: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values.items():
            if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
                result[str(key)] = None
            elif hasattr(value, "item"):
                result[str(key)] = value.item()
            elif isinstance(value, (datetime, Path)):
                result[str(key)] = str(value)
            else:
                result[str(key)] = value
        return result

    @staticmethod
    def _signature(data_type: str, values: Mapping[str, Any]) -> str:
        return f"{data_type}:{json.dumps(values, sort_keys=True, default=str, separators=(',', ':'))}"

    @staticmethod
    def _is_number(value: Any) -> bool:
        try:
            return value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            numeric = float(value)
            integer = int(numeric)
            return integer if math.isfinite(numeric) and abs(numeric - integer) < 1e-9 else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _get(self, record_id: str) -> BoreholeRecord:
        for record in self.database.records:
            if record.record_id == record_id:
                return record
        raise KeyError(f"Unknown borehole record_id: {record_id}")

    def _reclassify_records(self, records: Iterable[BoreholeRecord]) -> None:
        """Classify a batch sequentially so intra-batch duplicates are visible."""
        records = list(records)
        for record in records:
            record.state = RecordState.PENDING
        for record in records:
            record.state, record.exclusion_reason = self._classify(record.data_type, record.values)

    def _mark_quality_stale(self) -> None:
        self.set_quality_confirmation(None)
