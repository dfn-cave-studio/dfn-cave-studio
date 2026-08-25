"""
Borehole and geological data models for DFN Cave Studio.

Defines:
  - Collar: Borehole starting point (x, y, z, azimuth, dip)
  - BoreholeSurvey: Downhole trajectory measurements
  - FractureObservation: Fracture intersected in borehole
  - RQDInterval: Rock Quality Designation measurement
  - Borehole: Complete borehole with collar, survey, fractures, RQD
  - BoreholeCollection: Multi-borehole management and validation

References:
  - Priest, S.D. (1993). Discontinuity Analysis for Rock Engineering.
  - SCIENTIFIC_SPEC.md Section 6.
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator

from dfn_cave_studio.models.enums import FractureType


# =============================================================================
# Trajectory Computation Method
# =============================================================================

class TrajectoryMethod(str, Enum):
    """Method for computing 3D borehole trajectory from survey data."""
    TANGENTIAL = "tangential"          # Simple straight segments
    MINIMUM_CURVATURE = "min_curvature"  # Industry standard
    RADIUS_OF_CURVATURE = "radius_of_curvature"
    BALANCED_TANGENTIAL = "balanced_tangential"


# =============================================================================
# Collar
# =============================================================================

class Collar(BaseModel):
    """Borehole collar (starting point) definition.

    All coordinates in meters, angles in degrees.
    """

    borehole_id: str = Field(..., description="Unique borehole identifier")
    collar_x: float = Field(..., description="Collar X coordinate (easting, m)")
    collar_y: float = Field(..., description="Collar Y coordinate (northing, m)")
    collar_z: float = Field(..., description="Collar Z coordinate (elevation, m)")
    azimuth: float = Field(default=0.0, ge=0.0, lt=360.0, description="Borehole azimuth (°, from north)")
    dip: float = Field(default=-90.0, ge=-90.0, le=90.0, description="Borehole dip (°, negative = downward)")
    final_depth: float = Field(default=100.0, gt=0.0, description="Final measured depth (m)")

    @property
    def position(self) -> NDArray[np.float64]:
        """Collar position as numpy array (x, y, z)."""
        return np.array([self.collar_x, self.collar_y, self.collar_z], dtype=np.float64)


# =============================================================================
# Survey Station & Trajectory
# =============================================================================

class SurveyStation(BaseModel):
    """A single downhole survey measurement."""

    measured_depth: float = Field(..., ge=0.0, description="Measured depth from collar (m)")
    azimuth: float = Field(default=0.0, ge=0.0, lt=360.0, description="Hole azimuth at this depth (°)")
    dip: float = Field(default=-90.0, ge=-90.0, le=90.0, description="Hole dip at this depth (°)")

    @model_validator(mode="after")
    def check_depth_positive(self) -> "SurveyStation":
        if self.measured_depth < 0:
            raise ValueError(f"measured_depth must be >= 0, got {self.measured_depth}")
        return self


class BoreholeSurvey(BaseModel):
    """Collection of survey stations defining the borehole trajectory.

    The survey is processed sequentially from collar to final depth.
    The first station is always at measured_depth=0 (the collar).
    """

    stations: List[SurveyStation] = Field(default_factory=list)
    trajectory_method: TrajectoryMethod = TrajectoryMethod.MINIMUM_CURVATURE

    def compute_trajectory_3d(
        self,
        collar: Collar,
        step_length: float = 1.0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the 3D trajectory as arrays of points.

        Uses the minimum curvature method (industry standard) for smooth
        interpolation between survey stations.

        Args:
            collar: Borehole collar definition.
            step_length: Interpolation step length in meters.

        Returns:
            Tuple of (points (N,3), measured_depths (N,)) in meters.
        """
        if not self.stations:
            # No survey: straight hole from collar
            return self._straight_trajectory(collar)

        sorted_stations = sorted(self.stations, key=lambda s: s.measured_depth)

        # Ensure station at depth 0 exists
        if sorted_stations[0].measured_depth > 1e-6:
            sorted_stations.insert(0, SurveyStation(
                measured_depth=0.0,
                azimuth=collar.azimuth,
                dip=collar.dip,
            ))
        if sorted_stations[-1].measured_depth < collar.final_depth - 1e-6:
            last = sorted_stations[-1]
            sorted_stations.append(
                SurveyStation(
                    measured_depth=collar.final_depth,
                    azimuth=last.azimuth,
                    dip=last.dip,
                )
            )

        points = [collar.position]
        mds = [0.0]

        for i in range(len(sorted_stations) - 1):
            s1 = sorted_stations[i]
            s2 = sorted_stations[i + 1]

            seg_length = s2.measured_depth - s1.measured_depth
            if seg_length <= 1e-10:
                continue

            # P1 is the 3D position at station s1 (segment start)
            P1 = points[-1].copy()

            n_steps = max(1, int(math.ceil(seg_length / step_length)))
            actual_step = seg_length / n_steps

            for step in range(1, n_steps + 1):
                frac = step / n_steps

                if self.trajectory_method == TrajectoryMethod.MINIMUM_CURVATURE:
                    # Compute absolute position from segment start P1
                    # using the minimum curvature displacement formula
                    pt = self._min_curvature_interpolate(
                        P1, s1, s2, frac
                    )
                else:
                    # Tangential: use incremental step from last point
                    pt = self._tangential_interpolate(
                        points[-1], s1, s2, actual_step
                    )

                points.append(pt)
                mds.append(mds[-1] + actual_step)

        return np.array(points, dtype=np.float64), np.array(mds, dtype=np.float64)

    def _straight_trajectory(
        self, collar: Collar, step: float = 1.0
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute straight trajectory from collar orientation."""
        az_rad = math.radians(collar.azimuth)
        dip_rad = math.radians(collar.dip)

        # Direction vector: azimuth from north (Y), dip from horizontal
        dx = math.sin(az_rad) * math.cos(dip_rad)
        dy = math.cos(az_rad) * math.cos(dip_rad)
        dz = math.sin(dip_rad)

        n_points = max(1, int(math.ceil(collar.final_depth / step))) + 1
        mds = np.linspace(0, collar.final_depth, n_points)
        points = np.column_stack([
            collar.collar_x + mds * dx,
            collar.collar_y + mds * dy,
            collar.collar_z + mds * dz,
        ])

        return points.astype(np.float64), mds.astype(np.float64)

    def _min_curvature_interpolate(
        self,
        P1: NDArray[np.float64],
        s1: SurveyStation,
        s2: SurveyStation,
        frac: float,
    ) -> NDArray[np.float64]:
        """Minimum curvature interpolation between two survey stations.

        Computes the absolute 3D position at fraction `frac` along the
        circular arc from station s1 to s2.

        The minimum curvature method (industry standard) assumes a circular

        Args:
            P1: 3D position (x,y,z) at station s1 (segment start).
            s1: Survey station at the start of the segment.
            s2: Survey station at the end of the segment.
            frac: Fraction along the segment [0, 1].

        Returns:
            3D position at the interpolated point.
        """
        az1 = math.radians(s1.azimuth)
        dip1 = math.radians(s1.dip)
        az2 = math.radians(s2.azimuth)
        dip2 = math.radians(s2.dip)

        seg_length = s2.measured_depth - s1.measured_depth

        # Unit vectors along borehole direction at each station
        v1 = np.array([
            math.sin(az1) * math.cos(dip1),
            math.cos(az1) * math.cos(dip1),
            math.sin(dip1),
        ])
        v2 = np.array([
            math.sin(az2) * math.cos(dip2),
            math.cos(az2) * math.cos(dip2),
            math.sin(dip2),
        ])

        # Dogleg angle (angle between the two borehole direction vectors)
        cos_dogleg = np.clip(np.dot(v1, v2), -1.0, 1.0)
        dogleg = math.acos(cos_dogleg)

        if dogleg < 1e-10:
            # Straight segment — simple linear interpolation from P1
            return P1 + v1 * (frac * seg_length)

        # ── Circular Arc Parameterization ──────────────────────────────
        # The borehole follows a circular arc of radius R = ΔMD / γ.
        #
        # Construct an orthonormal basis in the plane of the arc:
        #   u = v1                   (tangent at start of segment)
        #   w = (v2 - cos(γ)·v1) / sin(γ)   (perpendicular to v1 in arc plane)
        #
        # v(f) = cos(f·γ)·u + sin(f·γ)·w   (unit direction at fraction f)
        # P(f) = P1 + R·[sin(f·γ)·u + (1 - cos(f·γ))·w]
        #
        # This places points ON the circular arc, preserving the arc length.
        R = seg_length / dogleg  # radius of curvature (m)

        sin_gamma = math.sin(dogleg)
        # Perpendicular unit vector in the arc plane
        w = (v2 - math.cos(dogleg) * v1) / sin_gamma

        angle = frac * dogleg  # angle along the arc
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)

        return P1 + R * (sin_angle * v1 + (1.0 - cos_angle) * w)

    def _tangential_interpolate(
        self,
        prev_point: NDArray[np.float64],
        s1: SurveyStation,
        s2: SurveyStation,
        step: float,
    ) -> NDArray[np.float64]:
        """Tangential interpolation (uses s1 direction for the whole segment)."""
        az = math.radians(s1.azimuth)
        dip = math.radians(s1.dip)
        direction = np.array([
            math.sin(az) * math.cos(dip),
            math.cos(az) * math.cos(dip),
            math.sin(dip),
        ])
        return prev_point + direction * step


# =============================================================================
# Fracture Observation
# =============================================================================

class OrientationCompleteness(str, Enum):
    """Scientific completeness of an observed fracture orientation."""

    FULL_ORIENTATION = "full_orientation"
    DIP_ONLY = "dip_only"


class FractureObservation(BaseModel):
    """A fracture observed in a borehole at a specific depth.

    Fields follow standard geotechnical logging conventions.
    """

    borehole_id: str = ""
    measured_depth: float = Field(..., ge=0.0, description="Measured depth of observation (m)")
    dip_direction: Optional[float] = Field(default=None, ge=0.0, le=360.0, description="Dip direction (°)")
    dip: float = Field(..., ge=0.0, le=90.0, description="Dip angle (°)")
    orientation_completeness: OrientationCompleteness = OrientationCompleteness.DIP_ONLY
    aperture: Optional[float] = Field(default=None, ge=0.0, description="Fracture aperture (mm)")
    filling: Optional[str] = Field(default=None, description="Filling material description")
    fracture_type: FractureType = FractureType.JOINT
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Observation confidence (0-1)")
    set_id: Optional[int] = Field(default=None, description="Assigned fracture set ID for DFN parameter inference")

    @field_validator("dip_direction", mode="before")
    @classmethod
    def parse_optional_dip_direction(cls, value: Any) -> Any:
        """Parse missing markers while preserving numeric zero as a real direction."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "na", "n/a", "null", "none"}:
            return None
        return value

    @model_validator(mode="after")
    def validate_angles(self) -> "FractureObservation":
        if self.dip_direction is not None and (self.dip_direction < 0 or self.dip_direction > 360):
            raise ValueError(f"dip_direction must be in [0, 360], got {self.dip_direction}")
        if self.dip < 0 or self.dip > 90:
            raise ValueError(f"dip must be in [0, 90], got {self.dip}")
        self.orientation_completeness = (
            OrientationCompleteness.FULL_ORIENTATION
            if self.dip_direction is not None
            else OrientationCompleteness.DIP_ONLY
        )
        return self

    @property
    def has_full_orientation(self) -> bool:
        """Return whether the record is eligible for 3D orientation analysis."""
        return self.orientation_completeness == OrientationCompleteness.FULL_ORIENTATION


# =============================================================================
# RQD Interval
# =============================================================================

class RQDInterval(BaseModel):
    """Rock Quality Designation (RQD) measurement interval.

    RQD = sum of core pieces > 10cm / total interval length × 100%
    """

    borehole_id: str = ""
    from_depth: float = Field(..., ge=0.0, description="Start measured depth (m)")
    to_depth: float = Field(..., gt=0.0, description="End measured depth (m)")
    rqd_value: float = Field(default=0.0, ge=0.0, le=100.0, description="RQD value (%)")
    core_recovery: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Core recovery (%)")

    @model_validator(mode="after")
    def validate_interval(self) -> "RQDInterval":
        if self.from_depth >= self.to_depth:
            raise ValueError(
                f"from_depth ({self.from_depth}) must be < to_depth ({self.to_depth})"
            )
        return self

    @property
    def interval_length(self) -> float:
        """Length of the RQD interval (m)."""
        return self.to_depth - self.from_depth


# =============================================================================
# Borehole
# =============================================================================

class Borehole(BaseModel):
    """Complete borehole with collar, survey, fracture observations, and RQD."""

    borehole_id: str = Field(..., description="Unique borehole identifier")
    name: str = ""
    collar: Collar = Field(...)
    survey: BoreholeSurvey = Field(default_factory=BoreholeSurvey)
    fracture_observations: List[FractureObservation] = Field(default_factory=list)
    rqd_intervals: List[RQDInterval] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Version tracking
    model_version: int = 1

    def __init__(self, **data):
        if 'borehole_id' not in data and 'collar' in data:
            data['borehole_id'] = data['collar'].borehole_id
        super().__init__(**data)
        if self.borehole_id != self.collar.borehole_id and self.collar.borehole_id:
            self.borehole_id = self.collar.borehole_id
        # Sync observation borehole_ids
        for obs in self.fracture_observations:
            if not obs.borehole_id:
                obs.borehole_id = self.borehole_id
        for rqd in self.rqd_intervals:
            if not rqd.borehole_id:
                rqd.borehole_id = self.borehole_id

    def compute_trajectory(self, step_length: float = 1.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the 3D trajectory of this borehole.

        Returns:
            Tuple of (points (N,3), measured_depths (N,)).
        """
        return self.survey.compute_trajectory_3d(self.collar, step_length)

    def locate_observation(
        self, observation: FractureObservation
    ) -> Optional[NDArray[np.float64]]:
        """Compute the 3D position of a fracture observation.

        Args:
            observation: Fracture observation to locate.

        Returns:
            3D position (x, y, z) or None if outside trajectory.
        """
        points, mds = self.compute_trajectory()
        # Find the closest trajectory point to the observation depth
        idx = np.argmin(np.abs(mds - observation.measured_depth))
        if idx < 0 or idx >= len(points):
            return None
        return points[idx]

    @property
    def observed_fracture_count(self) -> int:
        """Number of fracture observations in this borehole."""
        return len(self.fracture_observations)


# =============================================================================
# Borehole Collection
# =============================================================================

class BoreholeValidationError:
    """Record of a validation issue found in borehole data."""

    def __init__(self, borehole_id: str, field: str, issue: str, severity: str = "error"):
        self.borehole_id = borehole_id
        self.field = field
        self.issue = issue
        self.severity = severity  # "error", "warning", "info"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.borehole_id}/{self.field}: {self.issue}"


class BoreholeCollection(BaseModel):
    """Collection of boreholes with validation and query methods."""

    boreholes: List[Borehole] = Field(default_factory=list)
    name: str = "Borehole Collection"
    model_version: int = 1

    # ── Access ──────────────────────────────────────────────────────────

    def __getitem__(self, borehole_id: str) -> Borehole:
        for bh in self.boreholes:
            if bh.borehole_id == borehole_id:
                return bh
        raise KeyError(f"Borehole '{borehole_id}' not found")

    def __iter__(self):
        return iter(self.boreholes)

    def __len__(self) -> int:
        return len(self.boreholes)

    def add(self, borehole: Borehole) -> None:
        """Add a borehole, checking for duplicates."""
        if any(bh.borehole_id == borehole.borehole_id for bh in self.boreholes):
            raise ValueError(f"Duplicate borehole_id: {borehole.borehole_id}")
        self.boreholes.append(borehole)

    # ── Validation ──────────────────────────────────────────────────────

    def validate_all(self) -> List[BoreholeValidationError]:
        """Run all validation checks and return list of issues."""
        errors: List[BoreholeValidationError] = []
        errors.extend(self._check_duplicate_ids())
        errors.extend(self._check_missing_values())
        errors.extend(self._check_angle_ranges())
        errors.extend(self._check_depths())
        return errors

    def _check_duplicate_ids(self) -> List[BoreholeValidationError]:
        seen = set()
        errors = []
        for bh in self.boreholes:
            if bh.borehole_id in seen:
                errors.append(BoreholeValidationError(
                    bh.borehole_id, "borehole_id",
                    "Duplicate borehole ID", "error"
                ))
            seen.add(bh.borehole_id)
        return errors

    def _check_missing_values(self) -> List[BoreholeValidationError]:
        errors = []
        for bh in self.boreholes:
            if not bh.borehole_id:
                errors.append(BoreholeValidationError(
                    "", "borehole_id", "Missing borehole ID", "error"
                ))
            if bh.collar.final_depth is None or bh.collar.final_depth <= 0:
                errors.append(BoreholeValidationError(
                    bh.borehole_id, "final_depth",
                    "Missing or invalid final depth", "error"
                ))
        return errors

    def _check_angle_ranges(self) -> List[BoreholeValidationError]:
        errors = []
        for bh in self.boreholes:
            c = bh.collar
            if c.azimuth < 0 or c.azimuth >= 360:
                errors.append(BoreholeValidationError(
                    bh.borehole_id, "azimuth",
                    f"Azimuth {c.azimuth}° outside [0, 360)", "error"
                ))
            if c.dip < -90 or c.dip > 90:
                errors.append(BoreholeValidationError(
                    bh.borehole_id, "dip",
                    f"Dip {c.dip}° outside [-90, 90]", "error"
                ))
            for i, obs in enumerate(bh.fracture_observations):
                if obs.dip_direction is not None and (obs.dip_direction < 0 or obs.dip_direction > 360):
                    errors.append(BoreholeValidationError(
                        bh.borehole_id, f"fracture[{i}].dip_direction",
                        f"Dip direction {obs.dip_direction}° outside [0, 360)", "error"
                    ))
                if obs.dip < 0 or obs.dip > 90:
                    errors.append(BoreholeValidationError(
                        bh.borehole_id, f"fracture[{i}].dip",
                        f"Dip {obs.dip}° outside [0, 90]", "error"
                    ))
            for station in bh.survey.stations:
                if station.azimuth < 0 or station.azimuth >= 360:
                    errors.append(BoreholeValidationError(
                        bh.borehole_id, "survey.azimuth",
                        f"Survey azimuth {station.azimuth}° outside [0, 360)", "warning"
                    ))
        return errors

    def _check_depths(self) -> List[BoreholeValidationError]:
        errors = []
        for bh in self.boreholes:
            for i, obs in enumerate(bh.fracture_observations):
                if obs.measured_depth > bh.collar.final_depth:
                    errors.append(BoreholeValidationError(
                        bh.borehole_id, f"fracture[{i}].measured_depth",
                        f"Depth {obs.measured_depth}m exceeds final depth {bh.collar.final_depth}m",
                        "warning"
                    ))
            for i, rqd in enumerate(bh.rqd_intervals):
                if rqd.to_depth > bh.collar.final_depth:
                    errors.append(BoreholeValidationError(
                        bh.borehole_id, f"rqd[{i}].to_depth",
                        f"RQD end {rqd.to_depth}m exceeds final depth {bh.collar.final_depth}m",
                        "warning"
                    ))
        return errors

    def quality_report(self) -> str:
        """Generate a data quality report as formatted text."""
        errors = self.validate_all()
        if not errors:
            return f"Data Quality Report - {self.name}\nAll checks passed ✅\n"
        lines = [f"Data Quality Report - {self.name}",
                 f"{len(errors)} issue(s) found:",
                 ""]
        severity_order = {"error": 0, "warning": 1, "info": 2}
        sorted_errors = sorted(errors, key=lambda e: severity_order.get(e.severity, 99))
        for e in sorted_errors:
            lines.append(f"  {e}")
        return "\n".join(lines)

    # ── Import Helpers ──────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        collar_path: str,
        survey_path: Optional[str] = None,
        fractures_path: Optional[str] = None,
        rqd_path: Optional[str] = None,
        field_map: Optional[Dict[str, str]] = None,
    ) -> "BoreholeCollection":
        """Create BoreholeCollection from CSV files.

        Args:
            collar_path: Path to collar CSV (columns: borehole_id, collar_x, collar_y,
                        collar_z, azimuth, dip, final_depth).
            survey_path: Path to survey CSV (columns: borehole_id, measured_depth,
                        azimuth, dip).
            fractures_path: Path to fracture observations CSV.
            rqd_path: Path to RQD intervals CSV.
            field_map: Optional column name mapping.

        Returns:
            BoreholeCollection instance.
        """
        import pandas as pd

        # Read collar data
        collar_df = pd.read_csv(collar_path)
        if field_map:
            collar_df = collar_df.rename(columns=field_map)

        boreholes: List[Borehole] = []

        for _, row in collar_df.iterrows():
            collar = Collar(
                borehole_id=str(row.get("borehole_id", "")),
                collar_x=float(row.get("collar_x", 0)),
                collar_y=float(row.get("collar_y", 0)),
                collar_z=float(row.get("collar_z", 0)),
                azimuth=float(row.get("azimuth", 0)),
                dip=float(row.get("dip", -90)),
                final_depth=float(row.get("final_depth", 100)),
            )

            borehole = Borehole(
                borehole_id=collar.borehole_id,
                collar=collar,
            )
            boreholes.append(borehole)

        # Read survey data
        if survey_path:
            survey_df = pd.read_csv(survey_path)
            survey_by_bh: Dict[str, List[SurveyStation]] = {}
            for _, row in survey_df.iterrows():
                bh_id = str(row.get("borehole_id", ""))
                station = SurveyStation(
                    measured_depth=float(row.get("measured_depth", 0)),
                    azimuth=float(row.get("azimuth", 0)),
                    dip=float(row.get("dip", -90)),
                )
                survey_by_bh.setdefault(bh_id, []).append(station)

            for bh in boreholes:
                if bh.borehole_id in survey_by_bh:
                    bh.survey = BoreholeSurvey(stations=survey_by_bh[bh.borehole_id])

        # Read fracture observations
        if fractures_path:
            frac_df = pd.read_csv(fractures_path)
            for _, row in frac_df.iterrows():
                bh_id = str(row.get("borehole_id", ""))
                raw_direction = row.get("dip_direction")
                obs = FractureObservation(
                    borehole_id=bh_id,
                    measured_depth=float(row.get("measured_depth", 0)),
                    dip_direction=(
                        None
                        if raw_direction is None
                        or pd.isna(raw_direction)
                        or str(raw_direction).strip().lower() in {"", "na", "n/a", "null", "none"}
                        else float(raw_direction)
                    ),
                    dip=float(row.get("dip", 0)),
                    aperture=float(row["aperture"]) if pd.notna(row.get("aperture")) else None,
                    filling=str(row.get("filling", "")) if pd.notna(row.get("filling")) else None,
                    fracture_type=FractureType(str(row.get("fracture_type", "joint"))),
                    confidence=float(row.get("confidence", 1.0)),
                )
                for bh in boreholes:
                    if bh.borehole_id == bh_id:
                        bh.fracture_observations.append(obs)
                        break

        # Read RQD data
        if rqd_path:
            rqd_df = pd.read_csv(rqd_path)
            for _, row in rqd_df.iterrows():
                bh_id = str(row.get("borehole_id", ""))
                rqd = RQDInterval(
                    borehole_id=bh_id,
                    from_depth=float(row.get("from_depth", 0)),
                    to_depth=float(row.get("to_depth", 0)),
                    rqd_value=float(row.get("rqd", 0)),
                    core_recovery=float(row["core_recovery"]) if pd.notna(row.get("core_recovery")) else None,
                )
                for bh in boreholes:
                    if bh.borehole_id == bh_id:
                        bh.rqd_intervals.append(rqd)
                        break

        return cls(boreholes=boreholes)
