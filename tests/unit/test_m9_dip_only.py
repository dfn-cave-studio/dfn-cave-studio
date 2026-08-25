"""v0.9.1-M9 scientific regressions for dip-only fracture observations."""

import pandas as pd
import pytest

from dfn_cave_studio.borehole.orientation_statistics import OrientationStatisticsCalculator
from dfn_cave_studio.dfn.intensity import build_p10_intervals, estimate_p32
from dfn_cave_studio.geometry.coordinate import dip_dir_dip_to_normal
from dfn_cave_studio.models.borehole import (
    Borehole,
    BoreholeCollection,
    Collar,
    FractureObservation,
    OrientationCompleteness,
)
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.m9 import ObservabilityState
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.joint_set_service import JointSetService


def _collection() -> BoreholeCollection:
    calibration = Borehole(
        borehole_id="CAL",
        collar=Collar(borehole_id="CAL", collar_x=0, collar_y=0, collar_z=100, final_depth=100),
        fracture_observations=[
            FractureObservation(measured_depth=10, dip_direction=40, dip=50, set_id=1),
            FractureObservation(measured_depth=20, dip_direction=45, dip=55, set_id=1),
            FractureObservation(measured_depth=30, dip_direction=50, dip=60, set_id=1),
            FractureObservation(measured_depth=40, dip_direction=None, dip=65, set_id=1),
            FractureObservation(measured_depth=50, dip_direction=None, dip=70, set_id=1),
            FractureObservation(measured_depth=60, dip_direction=None, dip=30, set_id=None),
        ],
    )
    validation = Borehole(
        borehole_id="VAL",
        collar=Collar(borehole_id="VAL", collar_x=10, collar_y=0, collar_z=100, final_depth=100),
        fracture_observations=[
            FractureObservation(measured_depth=10, dip_direction=45, dip=55, set_id=1),
            FractureObservation(measured_depth=20, dip_direction=None, dip=65, set_id=1),
        ],
    )
    return BoreholeCollection(boreholes=[calibration, validation])


def test_model_preserves_missing_and_real_zero_and_rejects_none_normal() -> None:
    dip_only = FractureObservation(measured_depth=1, dip=35, dip_direction="N/A")
    north = FractureObservation(measured_depth=2, dip=35, dip_direction=0)
    assert dip_only.dip_direction is None
    assert dip_only.orientation_completeness == OrientationCompleteness.DIP_ONLY
    assert north.dip_direction == 0
    assert north.orientation_completeness == OrientationCompleteness.FULL_ORIENTATION
    with pytest.raises(ValueError, match="dip-only fracture"):
        dip_dir_dip_to_normal(None, 35)
    with pytest.raises(ValueError, match="dip"):
        FractureObservation(measured_depth=3)


def test_repository_quality_retains_dip_only_and_audits_360_fix() -> None:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame([{"borehole_id": "BH", "collar_x": 0, "collar_y": 0, "collar_z": 0, "final_depth": 100}]),
        "collars.csv",
    )
    result = repository.import_dataframe(
        "fractures",
        pd.DataFrame(
            [
                {"hole_id": "BH", "depth": 10, "dip": 40, "dip_direction": None, "set_id": 1},
                {"hole_id": "BH", "depth": 20, "dip": 50, "dip_direction": 0, "set_id": 1},
                {"hole_id": "BH", "depth": 30, "dip": 60, "dip_direction": 360, "set_id": 1},
            ]
        ),
        "fractures.csv",
    )
    assert result == {
        "raw": 3,
        "formal": 3,
        "excluded": 0,
        "pending": 0,
        "duplicates": 0,
        "cancelled": False,
        "data_type": "fractures",
        "total_fracture_rows": 3,
        "full_orientation": 2,
        "dip_only": 1,
        "excluded_error": 0,
    }
    quality = BoreholeQualityService(project)
    quality.run_checks()
    assert quality.unresolved_error_count == 1
    dip_issue = next(issue for issue in quality.issues if issue.code == "fracture_dip_only")
    assert dip_issue.severity.value == "info"
    assert dip_issue.auto_fix_available is False
    quality.apply_auto_fixes()
    assert quality.unresolved_error_count == 0
    formal = repository.query("fractures", "formal")
    assert [record.values["dip_direction"] for record in formal] == [None, 0.0, 0.0]
    fixed = formal[2]
    assert fixed.modification_history[-1].issue_id is not None
    assert fixed.modification_history[-1].before["dip_direction"] == 360.0
    assert fixed.modification_history[-1].after["dip_direction"] == 0.0
    assert sum(observation.dip_direction is None for observation in project.borehole_collection["BH"].fracture_observations) == 1


def test_mode_a_counts_dip_only_but_mode_b_clusters_full_orientation_only() -> None:
    collection = _collection()
    service = JointSetService(random_seed=42)
    imported = service.identify_from_imported(collection, {"CAL"}, {"VAL"})
    assert imported.set_counts[1] == {"total": 5, "full_orientation": 3, "dip_only": 2}
    assert imported.calibration_count == 5
    assert imported.full_orientation_count == 3
    assert imported.dip_only_count == 2
    assert imported.sets[1].provenance["dip_only_count"] == 2

    automatic = JointSetService(random_seed=42).identify_auto(
        collection, {"CAL"}, {"VAL"}, n_clusters=1, random_seed=42
    )
    assert automatic.calibration_count == 3
    assert automatic.full_orientation_count == 3
    assert automatic.dip_only_count == 3
    assert len(automatic.assignments) == 3
    assert automatic.validation_count == 2
    assert automatic.validation_dip_only_count == 1


def test_fisher_uses_full_orientation_and_reports_dip_statistics() -> None:
    stats = OrientationStatisticsCalculator().compute_detailed_by_set(_collection())[1]
    assert stats.total_count == 7
    assert stats.full_orientation_count == 4
    assert stats.dip_only_count == 3
    assert stats.orientation_fit_eligible is True
    assert stats.fisher_orientation is not None
    assert stats.mean_dip == pytest.approx(60.0)


def test_p10_counts_assigned_dip_only_without_fabricating_unassigned_set() -> None:
    collection = _collection()
    intervals = build_p10_intervals(collection, {"CAL": "calibration", "VAL": "validation"}, [], interval_length=100)
    calibration = next(row for row in intervals if row.hole_id == "CAL" and row.set_id == 1)
    assert calibration.observation_count == 5
    assert calibration.full_orientation_count == 3
    assert calibration.dip_only_count == 2
    assert calibration.unassigned_dip_only_count == 1
    assert calibration.provenance["unassigned_dip_only_not_in_set_specific_p10"] == 1


def test_p32_uses_domain_set_model_or_returns_explicit_unavailable_state() -> None:
    intervals = build_p10_intervals(_collection(), {"CAL": "calibration", "VAL": "validation"}, [], interval_length=100)
    joint_set = JointSetConfig(
        set_id=1,
        orientation=OrientationDistribution(mean_dip_direction=45, mean_dip=55, kappa=30),
    )
    available = estimate_p32(
        intervals,
        [joint_set],
        random_seed=42,
        sample_count=100,
        joint_sets_by_domain={(None, 1): joint_set},
    )[0]
    assert available.fracture_count == 5
    assert available.full_orientation_count == 3
    assert available.dip_only_count == 2
    assert available.orientation_model_source == "domain_set_model"
    assert available.provenance["individual_orientation"] == "missing"
    assert available.p32 is not None

    unavailable = estimate_p32(
        intervals,
        [joint_set],
        random_seed=42,
        sample_count=100,
        joint_sets_by_domain={},
    )[0]
    assert unavailable.p32 is None
    assert unavailable.observability == ObservabilityState.INSUFFICIENT_ORIENTATION_DATA
    assert unavailable.eligibility_status == "INSUFFICIENT_ORIENTATION_DATA"


def test_orientation_fit_requires_three_full_records() -> None:
    collection = _collection()
    collection["CAL"].fracture_observations = collection["CAL"].fracture_observations[2:]
    stats = OrientationStatisticsCalculator().compute_detailed_by_set(collection)[1]
    assert stats.orientation_fit_eligible is False
    assert stats.fisher_orientation is None
    assert stats.exclusion_reason == "INSUFFICIENT_ORIENTATION_DATA"


def test_mode_a_preserves_dip_only_only_set_identity_without_fake_fisher() -> None:
    collection = _collection()
    collection["CAL"].fracture_observations.append(
        FractureObservation(measured_depth=80, dip_direction=None, dip=25, set_id=9)
    )
    result = JointSetService().identify_from_imported(collection, {"CAL"}, {"VAL"})
    assert result.set_counts[9] == {"total": 1, "full_orientation": 0, "dip_only": 1}
    assert result.assignments["CAL:6"] == 9
    assert 9 not in result.sets
