"""Known-result scientific tests for M9 borehole intensity."""

import numpy as np

from dfn_cave_studio.dfn.intensity import build_p10_intervals, estimate_p32, expected_orientation_exposure
from dfn_cave_studio.models.borehole import Borehole, BoreholeCollection, Collar, FractureObservation
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.data_management import DomainInterval


def _vertical_hole() -> Borehole:
    return Borehole(
        borehole_id="BH-1",
        collar=Collar(borehole_id="BH-1", collar_x=0, collar_y=0, collar_z=20, dip=-90, final_depth=20),
        fracture_observations=[
            FractureObservation(borehole_id="BH-1", measured_depth=5, dip_direction=0, dip=0, set_id=1),
            FractureObservation(borehole_id="BH-1", measured_depth=10, dip_direction=0, dip=0, set_id=1),
            FractureObservation(borehole_id="BH-1", measured_depth=20, dip_direction=0, dip=0, set_id=1),
        ],
    )


def test_p10_known_result_and_boundary_is_counted_once():
    intervals = build_p10_intervals(BoreholeCollection(boreholes=[_vertical_hole()]), {"BH-1": "calibration"}, [], interval_length=10)
    assert [row.observation_count for row in intervals] == [1, 2]
    assert sum(row.observation_count for row in intervals) == 3
    assert np.allclose([row.p10 for row in intervals], [0.1, 0.2])


def test_direction_corrected_p32_is_seeded_and_finite():
    joint_set = JointSetConfig(
        set_id=1,
        orientation=OrientationDistribution(mean_dip_direction=0, mean_dip=0, kappa=1000),
    )
    exposure = expected_orientation_exposure(joint_set, (0, 0, -1), random_seed=7, sample_count=5000)
    assert 0.99 < exposure <= 1.0
    intervals = build_p10_intervals(BoreholeCollection(boreholes=[_vertical_hole()]), {"BH-1": "calibration"}, [], interval_length=20)
    first = estimate_p32(intervals, [joint_set], random_seed=7, sample_count=5000)
    second = estimate_p32(intervals, [joint_set], random_seed=7, sample_count=5000)
    assert first == second
    assert np.isfinite(first[0].p32)
    assert abs(first[0].p32 - 0.15) < 0.005


def test_validation_intervals_are_never_used_in_fit():
    calibration = _vertical_hole()
    validation = _vertical_hole().model_copy(deep=True)
    validation.borehole_id = "BH-V"
    validation.collar.borehole_id = "BH-V"
    validation.fracture_observations *= 5
    for item in validation.fracture_observations:
        item.borehole_id = "BH-V"
    collection = BoreholeCollection(boreholes=[calibration, validation])
    rows = build_p10_intervals(collection, {"BH-1": "calibration", "BH-V": "validation"}, [], interval_length=20)
    joint_set = JointSetConfig(set_id=1, orientation=OrientationDistribution(mean_dip_direction=0, mean_dip=0, kappa=1000))
    estimate = estimate_p32(rows, [joint_set], random_seed=4, sample_count=1000)[0]
    assert estimate.fracture_count == 3
    assert estimate.calibration_holes == ["BH-1"]


def test_low_observability_does_not_emit_extreme_p32():
    rows = build_p10_intervals(
        BoreholeCollection(boreholes=[_vertical_hole()]), {"BH-1": "calibration"}, [], interval_length=20
    )
    vertical_planes = JointSetConfig(
        set_id=1,
        orientation=OrientationDistribution(mean_dip_direction=90, mean_dip=90, kappa=1000),
    )
    estimate = estimate_p32(
        rows,
        [vertical_planes],
        random_seed=2,
        sample_count=5000,
        low_observability_threshold=0.05,
    )[0]
    assert estimate.observability.value == "low_observability"
    assert estimate.p32 is None


def test_fixed_p10_interval_splits_at_domain_boundary_and_counts_boundary_once():
    hole = _vertical_hole().model_copy(deep=True)
    hole.collar.final_depth = 30
    hole.fracture_observations = [
        FractureObservation(borehole_id="BH-1", measured_depth=10, dip_direction=0, dip=0, set_id=1),
        FractureObservation(borehole_id="BH-1", measured_depth=15, dip_direction=0, dip=0, set_id=1),
        FractureObservation(borehole_id="BH-1", measured_depth=25, dip_direction=0, dip=0, set_id=1),
    ]
    domains = [
        DomainInterval(hole_id="BH-1", from_depth=0, to_depth=10, domain_id=1),
        DomainInterval(hole_id="BH-1", from_depth=10, to_depth=30, domain_id=2),
    ]

    rows = build_p10_intervals(
        BoreholeCollection(boreholes=[hole]), {"BH-1": "calibration"}, domains, interval_length=25
    )

    assert [(row.from_depth, row.to_depth, row.domain_id) for row in rows] == [
        (0.0, 10.0, 1),
        (10.0, 25.0, 2),
        (25.0, 30.0, 2),
    ]
    assert [row.observation_count for row in rows] == [0, 2, 1]
    assert sum(row.observation_count for row in rows) == 3
    assert rows[0].provenance["split_by_domain_boundary"] is True
    assert rows[1].provenance["split_by_domain_boundary"] is True


def test_fixed_p10_multiple_domain_splits_conserve_trajectory_length():
    hole = _vertical_hole().model_copy(deep=True)
    hole.collar.final_depth = 40
    hole.fracture_observations = [
        FractureObservation(borehole_id="BH-1", measured_depth=depth, dip_direction=0, dip=0, set_id=1)
        for depth in (5, 10, 20, 30, 40)
    ]
    domains = [
        DomainInterval(hole_id="BH-1", from_depth=0, to_depth=10, domain_id=1),
        DomainInterval(hole_id="BH-1", from_depth=10, to_depth=20, domain_id=2),
        DomainInterval(hole_id="BH-1", from_depth=20, to_depth=30, domain_id=3),
        DomainInterval(hole_id="BH-1", from_depth=30, to_depth=40, domain_id=4),
    ]

    rows = build_p10_intervals(
        BoreholeCollection(boreholes=[hole]), {"BH-1": "validation"}, domains, interval_length=25
    )

    assert [(row.from_depth, row.to_depth) for row in rows] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 25.0),
        (25.0, 30.0),
        (30.0, 40.0),
    ]
    assert [row.domain_id for row in rows] == [1, 2, 3, 3, 4]
    assert np.isclose(sum(row.sample_length for row in rows), 40.0)
    assert sum(row.observation_count for row in rows) == 5
    assert all(row.role == "validation" for row in rows)
