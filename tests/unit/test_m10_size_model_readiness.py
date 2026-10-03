"""Synthetic regressions for explicit fracture-size provenance and M10 readiness."""

from __future__ import annotations

import numpy as np
import pytest

from dfn_cave_studio.dfn.m10_multiscale import SizeClass, calculate_thresholds, classify_radius
from dfn_cave_studio.dfn.size_models import assumed_size_model, size_moments
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.borehole_fracture_realization import BoreholeFractureGenerationConfig
from dfn_cave_studio.models.m10 import M10GenerationConfig
from dfn_cave_studio.models.m9 import SizeModel, SizeModelSource, SizeModelStatus
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.m10_service import M10Service
from tests.integration.test_m10_persistence_export import make_m10_project
from tests.unit.test_borehole_fracture_phase2a import _synthetic_project
from tests.unit.test_m10_generator import _input


def _unresolved_model() -> SizeModel:
    return SizeModel(
        domain_id=1,
        set_id=1,
        distribution_type="fixed",
        parameters={"radius": 1.0},
        min_radius=1.0,
        max_radius=1.0,
        mean_radius=1.0,
        mean_squared_radius=1.0,
        source=SizeModelSource.ASSUMED,
        provenance={"size_status": "UNRESOLVED", "reason": "no independent size observations"},
    )


def test_unresolved_positive_p32_is_blocked_before_estimate_or_generation() -> None:
    project = make_m10_project()
    project.joint_sets = [JointSetConfig(set_id=1, name="Synthetic Set")]
    project.m9_state.size_models = [_unresolved_model()]

    errors = M10Service(project).validate_readiness(
        M10GenerationConfig(validation_warning_acknowledged=True, condition_calibration_observations=False)
    )
    assert any("Domain 1 / Set 1 (Synthetic Set): UNRESOLVED" in item for item in errors)
    with pytest.raises(ValueError, match="unresolved fracture size"):
        M10Service(project).estimate_details(
            M10GenerationConfig(validation_warning_acknowledged=True, condition_calibration_observations=False)
        )


def test_phase2a_preview_has_no_scientific_radius_array_or_size_model() -> None:
    project = _synthetic_project()
    realization_state = project.borehole_fracture_state
    assert realization_state.realizations == []
    assert project.m9_state.size_models == []
    # The Phase 2A column contract stores points and orientations, not disc radii.
    from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService

    fit = BoreholeFractureService(project).fit_global_sets(2, 42)
    project.borehole_fracture_state.global_fit = fit
    candidate = BoreholeFractureService(project).build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=2, master_seed=42),
        fit_override=fit,
    )
    assert candidate.realizations
    assert "radius" not in candidate.realizations[0].arrays
    assert project.m9_state.size_models == []


def test_explicit_manual_fixed_one_metre_is_audited_and_generates_one_metre_radii() -> None:
    model = assumed_size_model(
        domain_id=1,
        set_id=1,
        distribution_type="fixed",
        parameters={"radius": 1.0},
        lower=1.0,
        upper=1.0,
        user_defined=True,
    )
    result = _input(p32=0.2, size_model=model, seed=42).generate(0)

    assert model.source == SizeModelSource.MANUAL_FIXED
    assert model.scientific_status == SizeModelStatus.MANUAL_FIXED
    assert model.provenance["radius_unit"] == "m"
    assert result.fracture_count > 0
    np.testing.assert_array_equal(result.geometry_arrays["radius"], 1.0)
    assert result.provenance["size_source_codes"]["3"] == "manual_fixed"


def test_fixed_one_metre_radii_fall_only_in_half_open_medium_class() -> None:
    model = assumed_size_model(
        domain_id=1,
        set_id=1,
        distribution_type="fixed",
        parameters={"radius": 1.0},
        lower=1.0,
        upper=1.0,
        user_defined=True,
    )
    generator = _input(p32=0.2, size_model=model, seed=42)
    generator.config = M10GenerationConfig(
        base_seed=42,
        validation_warning_acknowledged=True,
        condition_calibration_observations=False,
        size_threshold_mode="manual",
        manual_small_medium_radius=0.99,
        manual_medium_large_radius=1.01,
        enabled_size_classes=["SMALL", "MEDIUM", "LARGE"],
    )
    result = generator.generate(0)

    np.testing.assert_array_equal(result.geometry_arrays["radius"], 1.0)
    np.testing.assert_array_equal(result.geometry_arrays["size_class"], int(SizeClass.MEDIUM))


def test_legacy_assumed_source_code_is_preserved_for_old_scenario_path() -> None:
    model = assumed_size_model(
        domain_id=1,
        set_id=1,
        distribution_type="fixed",
        parameters={"radius": 1.0},
        lower=1.0,
        upper=1.0,
        user_defined=False,
    )
    assert model.source == SizeModelSource.ASSUMED
    assert model.scientific_status == SizeModelStatus.ASSUMED_SCENARIO


def test_truncated_lognormal_samples_are_bounded_nonconstant_and_reproducible() -> None:
    parameters = {"mu": 0.0, "sigma": 0.35}
    mean, mean2 = size_moments("truncated_lognormal", parameters, 0.5, 2.0)
    model = SizeModel(
        domain_id=1,
        set_id=1,
        distribution_type="truncated_lognormal",
        parameters=parameters,
        min_radius=0.5,
        max_radius=2.0,
        mean_radius=mean,
        mean_squared_radius=mean2,
        source=SizeModelSource.MANUAL_DISTRIBUTION,
        provenance={"size_status": "MANUAL_DISTRIBUTION", "radius_unit": "m"},
    )
    first = _input(p32=1.0, size_model=model, seed=77).generate(0).geometry_arrays["radius"]
    second = _input(p32=1.0, size_model=model, seed=77).generate(0).geometry_arrays["radius"]

    assert len(first) > 10
    assert np.unique(first).size > 1
    assert float(first.min()) >= 0.5
    assert float(first.max()) <= 2.0
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("radius", "expected"),
    [(0.98, SizeClass.SMALL), (0.99, SizeClass.MEDIUM), (1.0, SizeClass.MEDIUM), (1.01, SizeClass.LARGE)],
)
def test_radius_classes_use_unique_half_open_boundaries(radius: float, expected: SizeClass) -> None:
    model = assumed_size_model(
        domain_id=1,
        set_id=1,
        distribution_type="uniform",
        parameters={},
        lower=0.5,
        upper=2.0,
    )
    thresholds = calculate_thresholds(
        model,
        mode="manual",
        manual_small_medium_radius=0.99,
        manual_medium_large_radius=1.01,
    )
    assert classify_radius(radius, thresholds) == expected


def test_size_status_source_and_parameters_survive_save_reopen(tmp_path) -> None:
    project = make_m10_project()
    project.m9_state.size_models = [_unresolved_model()]
    path = tmp_path / "synthetic-unresolved-size.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)

    restored = reopened.m9_state.size_models[0]
    assert restored.scientific_status == SizeModelStatus.UNRESOLVED
    assert restored.source == SizeModelSource.ASSUMED
    assert restored.parameters == {"radius": 1.0}
    assert not restored.is_usable_for_explicit_dfn


def test_manual_fixed_status_source_unit_and_radius_survive_save_reopen(tmp_path) -> None:
    project = make_m10_project()
    project.m9_state.size_models = [
        assumed_size_model(
            domain_id=1,
            set_id=1,
            distribution_type="fixed",
            parameters={"radius": 1.0},
            lower=1.0,
            upper=1.0,
            user_defined=True,
        )
    ]
    path = tmp_path / "synthetic-manual-fixed-size.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path).m9_state.size_models[0]

    assert restored.scientific_status == SizeModelStatus.MANUAL_FIXED
    assert restored.source == SizeModelSource.MANUAL_FIXED
    assert restored.provenance["radius_unit"] == "m"
    assert restored.parameters == {"radius": 1.0}
    assert restored.min_radius == restored.mean_radius == restored.max_radius == 1.0
