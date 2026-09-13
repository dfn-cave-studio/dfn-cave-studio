"""Standard read access for multi-source observations stored in the M8 database."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from dfn_cave_studio.models.borehole_database import (
    BoreholeDataType,
    FractureObservationMode,
    RecordState,
)
from dfn_cave_studio.models.m9 import ScalarParameterSample
from dfn_cave_studio.models.observations import (
    AxisPlaneAngleObservation,
    DomainAssociationSegment,
    FractureSpacingObservation,
    OrientationPointObservation,
    OrientationPointSummary,
)
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.m7_state import get_holdout


class ObservationService:
    """Expose typed observations without projecting incomplete directions as fractures."""

    def __init__(self, project: Any):
        self.project = project
        self.repository = BoreholeRepository(project)

    def spacing_observations(self) -> list[FractureSpacingObservation]:
        """Return formal interval-spacing observations and their explicit domain overlaps."""
        output: list[FractureSpacingObservation] = []
        for record in self.repository.query(BoreholeDataType.FRACTURES, RecordState.FORMAL):
            values = record.values
            if values.get("observation_mode") != FractureObservationMode.INTERVAL_SPACING:
                continue
            output.append(
                FractureSpacingObservation(
                    observation_id=f"FS-{record.record_id}",
                    record_id=record.record_id,
                    hole_id=record.hole_id,
                    from_depth=float(values["from_depth"]),
                    to_depth=float(values["to_depth"]),
                    fracture_spacing=float(values["fracture_spacing"]),
                    spacing_unit=str(values["spacing_unit"]),
                    measurement_basis=str(values.get("measurement_basis", "BOREHOLE_ALONG_HOLE")),
                    derived_p10=float(values["derived_p10"]),
                    set_id=self.repository.optional_int(values.get("set_id")),
                    domain_segments=self.domain_associations(
                        record.hole_id, float(values["from_depth"]), float(values["to_depth"])
                    ),
                )
            )
        return output

    def axis_plane_angle_observations(self) -> list[AxisPlaneAngleObservation]:
        """Return relative-angle observations located on the measured borehole trajectory."""
        output: list[AxisPlaneAngleObservation] = []
        for record in self.repository.query(BoreholeDataType.FRACTURES, RecordState.FORMAL):
            values = record.values
            if values.get("observation_mode") != FractureObservationMode.AXIS_PLANE_ANGLE:
                continue
            xyz = self.position_at_depth(record.hole_id, float(values["depth"]))
            output.append(
                AxisPlaneAngleObservation(
                    observation_id=f"AA-{record.record_id}",
                    record_id=record.record_id,
                    hole_id=record.hole_id,
                    measured_depth=float(values["depth"]),
                    axis_plane_angle=float(values["axis_plane_angle"]),
                    x=float(xyz[0]),
                    y=float(xyz[1]),
                    z=float(xyz[2]),
                    set_id=self.repository.optional_int(values.get("set_id")),
                )
            )
        return output

    def orientation_points(self) -> list[OrientationPointObservation]:
        """Return point observations without assigning them to M9 calibration."""
        output: list[OrientationPointObservation] = []
        for record in self.repository.query(BoreholeDataType.ORIENTATION_POINTS, RecordState.FORMAL):
            values = record.values
            source_kind = values.get("source_kind")
            if source_kind not in {"POINT_CLOUD", "BOREHOLE_CAMERA"}:
                source_kind = "BOREHOLE_CAMERA" if str(values.get("observation_id", "")).upper().startswith("Z") else "POINT_CLOUD"
            local_set_id = str(values.get("local_set_id") or values.get("set_id") or "LEGACY")
            is_random = local_set_id.upper() == "RANDOM"
            output.append(
                OrientationPointObservation(
                    observation_id=str(values["observation_id"]),
                    point_id=str(values["point_id"]),
                    point_key=str(values.get("point_key") or f"{source_kind}:{values['point_id']}"),
                    record_id=record.record_id,
                    source_kind=str(source_kind),
                    x=float(values["x"]),
                    y=float(values["y"]),
                    z=float(values["z"]),
                    dip=self.repository.optional_float(values.get("dip")),
                    dip_direction=self.repository.optional_float(values.get("dip_direction")),
                    local_set_id=local_set_id,
                    component_type=str(values.get("component_type", "RANDOM_BACKGROUND" if is_random else "LOCAL_DOMINANT_SET")),
                    orientation_status=str(values.get("orientation_status", "MISSING" if is_random else "COMPLETE")),
                    joint_spacing_m=self.repository.optional_float(values.get("joint_spacing_m")),
                    measurement_basis=self._optional_text(values.get("measurement_basis")),
                    joint_num=self.repository.optional_int(values.get("joint_num")),
                    domain_id=self.repository.optional_int(values.get("domain_id")),
                    set_id=self.repository.optional_int(values.get("set_id")),
                    site_id=self._optional_text(values.get("site_id")),
                    source=self._optional_text(values.get("source")),
                    quality=self._optional_text(values.get("quality")),
                    observation_kind=str(values.get("observation_kind", "MEASURED")).upper(),
                    source_file=record.source_file,
                    source_row=record.source_row,
                    import_batch_id=record.import_batch_id,
                    audit_metadata={
                        "record_id": record.record_id,
                        "modification_source": record.modification_source,
                        "import_metadata": dict(record.import_metadata),
                    },
                )
            )
        return output

    def orientation_point_summaries(self) -> list[OrientationPointSummary]:
        """Return persisted P-only Jv/RQD audit summaries."""
        return [item.model_copy(deep=True) for item in self.repository.database.orientation_point_summaries]

    def domain_associations(self, hole_id: str, start: float, end: float) -> list[DomainAssociationSegment]:
        """Split an interval at domain boundaries without creating new observations."""
        records = self.repository.query(BoreholeDataType.DOMAIN_INTERVALS, RecordState.FORMAL, hole_id=hole_id)
        collar = self.repository.formal_collar(hole_id)
        fallback = self.repository.optional_int(collar.values.get("domain_id")) if collar is not None else None
        intervals = [
            (
                max(start, float(record.values["from_depth"])),
                min(end, float(record.values["to_depth"])),
                int(float(record.values["domain_id"])),
            )
            for record in records
            if max(start, float(record.values["from_depth"])) < min(end, float(record.values["to_depth"]))
        ]
        boundaries = sorted({start, end, *(value for left, right, _domain in intervals for value in (left, right))})
        segments: list[DomainAssociationSegment] = []
        for left, right in zip(boundaries, boundaries[1:]):
            midpoint = (left + right) / 2.0
            active_domains = {domain for item_left, item_right, domain in intervals if item_left <= midpoint < item_right}
            if len(active_domains) == 1:
                domain_id = next(iter(active_domains))
                source = "domain_intervals"
            elif len(active_domains) > 1:
                domain_id = None
                source = "overlap_conflict"
            else:
                domain_id = fallback
                source = "collar_fallback" if fallback is not None else "unassigned"
            segments.append(
                DomainAssociationSegment(
                    from_depth=left,
                    to_depth=right,
                    domain_id=domain_id,
                    assignment_source=source,
                )
            )
        return segments

    def database_scalar_samples(self) -> list[ScalarParameterSample]:
        """Project formal RQD/RMR intervals through the existing generic scalar-field contract."""
        samples: list[ScalarParameterSample] = []
        holdout = get_holdout(self.project)
        for data_type, parameter, unit, field in (
            (BoreholeDataType.RQD, "rqd", "%", "rqd"),
            (BoreholeDataType.RMR, "rmr", "score", "rmr"),
        ):
            for record in self.repository.query(data_type, RecordState.FORMAL):
                values = record.values
                start, end = float(values["from_depth"]), float(values["to_depth"])
                xyz = self.position_at_depth(record.hole_id, (start + end) / 2.0)
                segments = self.domain_associations(record.hole_id, start, end)
                domain_ids = {segment.domain_id for segment in segments}
                samples.append(
                    ScalarParameterSample(
                        sample_id=f"db:{record.record_id}",
                        borehole_id=record.hole_id,
                        from_depth=start,
                        to_depth=end,
                        parameter_name=parameter,
                        value=float(values[field]),
                        unit=unit,
                        source_dataset=record.source_file,
                        quality_flag=str(values.get("quality_flag", "")),
                        midpoint_x=float(xyz[0]),
                        midpoint_y=float(xyz[1]),
                        midpoint_z=float(xyz[2]),
                        domain_id=next(iter(domain_ids)) if len(domain_ids) == 1 else None,
                        role=(
                            "validation"
                            if holdout is not None and holdout.is_validation(record.hole_id)
                            else "calibration"
                        ),
                        source_record_id=record.record_id,
                        domain_segments=[segment.model_dump() for segment in segments],
                        domain_assignment_method="explicit_segments",
                    )
                )
        return samples

    def synchronize_scalar_samples(self) -> int:
        """Synchronize Formal RQD/RMR rows and invalidate only changed fields.

        Database projections are compared by their full scientific payload, so
        rebuilding an unchanged Formal collection is a no-op.  A changed RQD or
        RMR projection removes only that parameter's stored field (including its
        validation results); unrelated scalar fields and M10/M11 remain intact.
        """
        previous = [item for item in self.project.m9_state.scalar_samples if item.sample_id.startswith("db:")]
        retained = [item for item in self.project.m9_state.scalar_samples if not item.sample_id.startswith("db:")]
        projected = self.database_scalar_samples()
        previous_by_id = {item.sample_id: self._sample_payload(item) for item in previous}
        projected_by_id = {item.sample_id: self._sample_payload(item) for item in projected}
        changed_parameters = {
            payload["parameter_name"]
            for sample_id, payload in previous_by_id.items()
            if projected_by_id.get(sample_id) != payload
        }
        changed_parameters.update(
            payload["parameter_name"]
            for sample_id, payload in projected_by_id.items()
            if previous_by_id.get(sample_id) != payload
        )
        self.project.m9_state.scalar_samples = retained + projected
        if changed_parameters:
            from dfn_cave_studio.services.scalar_field_service import canonical_parameter_name

            changed = {canonical_parameter_name(name) for name in changed_parameters}
            self.project.m9_state.scalar_fields = [
                field
                for field in self.project.m9_state.scalar_fields
                if canonical_parameter_name(field.metadata.parameter_name) not in changed
            ]
        return len(projected)

    @staticmethod
    def _sample_payload(sample: ScalarParameterSample) -> dict[str, Any]:
        """Return the stable scientific payload used for sync change detection."""
        payload = sample.model_dump(mode="json")
        # Normalize nested mappings to make ordering irrelevant.
        return json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def position_at_depth(self, hole_id: str, depth: float) -> np.ndarray:
        """Locate measured depth using the project's existing 3-D trajectory."""
        hole = self.project.borehole_collection[hole_id]
        points, measured_depths = hole.compute_trajectory()
        return np.asarray([np.interp(depth, measured_depths, points[:, axis]) for axis in range(3)], dtype=float)

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        text = "" if value is None else str(value).strip()
        return text or None
