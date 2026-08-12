"""Regression tests for the requested Mode B joint-set count."""

from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from dfn_cave_studio.models.borehole import Borehole, BoreholeCollection, Collar, FractureObservation
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.joint_set_service import INSUFFICIENT_ORIENTATIONS_MESSAGE, JointSetService


DEMO = Path(__file__).parents[2] / "examples" / "m7_demo"
CALIBRATION_HOLES = {f"BH-{index:02d}" for index in range(1, 7)}
VALIDATION_HOLES = {"BH-07", "BH-08"}


@pytest.fixture(scope="module")
def demo_collection() -> BoreholeCollection:
    """Load the cleaned Formal collection from all real demo tables."""
    project = Project()
    repository = BoreholeRepository(project)
    for name in ("surveys", "collars", "fractures", "rqd", "domain_intervals"):
        repository.import_dataframe(name, pd.read_csv(DEMO / f"{name}.csv"), str(DEMO / f"{name}.csv"))
    quality = BoreholeQualityService(project)
    quality.run_checks()
    quality.apply_auto_fixes()
    quality.confirm_exclusions()
    return project.borehole_collection


@pytest.mark.parametrize("requested_k", range(1, 7))
def test_demo_returns_requested_nonempty_set_count(demo_collection: BoreholeCollection, requested_k: int) -> None:
    """Every supported demo K produces exactly K non-empty groups covering calibration."""
    result = JointSetService(random_seed=42).identify_auto(
        demo_collection,
        CALIBRATION_HOLES,
        VALIDATION_HOLES,
        n_clusters=requested_k,
        random_seed=42,
    )
    counts = [sum(label == set_id for label in result.assignments.values()) for set_id in sorted(result.sets)]

    assert len(result.sets) == requested_k
    assert all(count > 0 for count in counts)
    assert sum(counts) == result.calibration_count == 60
    assert len(result.assignments) == len(set(result.assignments)) == 60
    assert result.validation_count == 20
    assert not any(record_id.split(":", 1)[0] in VALIDATION_HOLES for record_id in result.assignments)


def test_k3_seed42_baseline_is_unchanged(demo_collection: BoreholeCollection) -> None:
    """The published K=3/seed=42 assignment counts remain unchanged."""
    result = JointSetService(random_seed=42).identify_auto(
        demo_collection, CALIBRATION_HOLES, VALIDATION_HOLES, n_clusters=3, random_seed=42
    )
    counts = [sum(label == set_id for label in result.assignments.values()) for set_id in sorted(result.sets)]
    assert counts == [24, 18, 18]


def test_same_seed_repeats_labels_and_counts(demo_collection: BoreholeCollection) -> None:
    """A fixed seed reproduces the complete assignment map, not only set count."""
    first = JointSetService().identify_auto(
        demo_collection, CALIBRATION_HOLES, VALIDATION_HOLES, n_clusters=6, random_seed=42
    )
    second = JointSetService().identify_auto(
        demo_collection, CALIBRATION_HOLES, VALIDATION_HOLES, n_clusters=6, random_seed=42
    )
    assert first.assignments == second.assignments
    assert [sum(value == key for value in first.assignments.values()) for key in sorted(first.sets)] == [
        sum(value == key for value in second.assignments.values()) for key in sorted(second.sets)
    ]


@pytest.mark.parametrize("seed", [1, 7, 99])
def test_different_seeds_still_return_requested_nonempty_count(
    demo_collection: BoreholeCollection, seed: int
) -> None:
    """Changing the seed may change assignments but cannot silently remove a group."""
    result = JointSetService().identify_auto(
        demo_collection, CALIBRATION_HOLES, VALIDATION_HOLES, n_clusters=6, random_seed=seed
    )
    counts = [sum(value == key for value in result.assignments.values()) for key in result.sets]
    assert len(result.sets) == 6
    assert len(result.assignments) == 60
    assert sum(counts) == 60
    assert all(count > 0 for count in counts)


def test_insufficient_distinct_axial_directions_raises_clear_error() -> None:
    """Repeated n/-n-equivalent observations do not count as distinct centers."""
    hole = Borehole(
        borehole_id="BH-01",
        collar=Collar(borehole_id="BH-01", collar_x=0, collar_y=0, collar_z=0, final_depth=10),
        fracture_observations=[
            FractureObservation(measured_depth=1, dip_direction=45, dip=60),
            FractureObservation(measured_depth=2, dip_direction=45, dip=60),
            FractureObservation(measured_depth=3, dip_direction=45, dip=60),
        ],
    )
    with pytest.raises(ValueError, match="Cannot identify K non-empty joint sets") as error:
        JointSetService().identify_auto(BoreholeCollection(boreholes=[hole]), {"BH-01"}, set(), n_clusters=2)
    assert str(error.value) == INSUFFICIENT_ORIENTATIONS_MESSAGE


def test_empty_cluster_assignment_is_repaired_deterministically() -> None:
    """Duplicate centers cannot leave a requested cluster empty."""
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.045, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.999, 0.045],
        ]
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    centroids = np.array([normals[0], normals[0], normals[2]])

    first = JointSetService._assign_nonempty(normals, centroids)
    second = JointSetService._assign_nonempty(normals, centroids)

    assert np.array_equal(first, second)
    assert np.all(np.bincount(first, minlength=3) > 0)
