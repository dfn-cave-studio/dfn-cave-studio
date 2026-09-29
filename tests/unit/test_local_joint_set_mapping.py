"""Scientific contracts for explicit local-component to global-set mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfn_cave_studio.models.borehole_database import BoreholeDataType
from dfn_cave_studio.models.borehole_fracture_realization import BoreholeFractureGenerationConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService, StableIDW
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from tests.unit.test_borehole_fracture_phase2a import _synthetic_project


def _project_with_fifteen_representatives():
    project = _synthetic_project()
    rows = []
    for index in range(12):
        rows.append(
            {
                "observation_id": f"P-X{index:02d}-1",
                "point_id": f"X{index:02d}",
                "x": 100.0 + index * 10.0,
                "y": 20.0 + index,
                "z": 80.0 - index,
                "dip": 40.0 + index,
                "dip_direction": (359.0 if index == 0 else 1.0 if index == 1 else 30.0 * index) % 360.0,
                "local_set_id": "REUSED" if index < 2 else f"LOCAL-{index}",
                "joint_spacing_m": 0.5 + index * 0.05,
                "joint_num": index + 1,
            }
        )
    result = BoreholeRepository(project).import_dataframe(
        BoreholeDataType.ORIENTATION_POINTS,
        pd.DataFrame(rows),
        "synthetic-local-components.csv",
    )
    assert result["formal"] == 12
    return project


def _confirmed_sets(fit) -> list[JointSetConfig]:
    return [
        JointSetConfig(
            set_id=model.global_set_id,
            name=f"Set {model.global_set_id}",
            orientation=OrientationDistribution(
                mean_dip_direction=model.mean_dip_direction,
                mean_dip=model.mean_dip,
                kappa=model.kappa,
            ),
        )
        for model in fit.sets
    ]


def test_fifteen_representatives_default_to_fifteen_then_explicitly_merge_to_thirteen() -> None:
    project = _project_with_fifteen_representatives()
    service = BoreholeFractureService(project)
    identity = service.build_local_component_mapping()
    assert identity.number_of_sets == 15
    assert len(identity.mappings) == 15
    assert len(identity.local_components) == 16  # 15 oriented representatives plus RANDOM.
    assert len({item.global_set_id for item in identity.mappings}) == 15

    assignments = {item.component_id: item.global_set_id for item in identity.mappings}
    components = sorted(assignments)
    assignments[components[1]] = assignments[components[0]]
    assignments[components[3]] = assignments[components[2]]
    merged = service.build_local_component_mapping(assignments, user_confirmed=True)
    assert merged.number_of_sets == 13
    assert len(merged.mappings) == 15
    assert len(merged.local_components) == 16
    assert merged.provenance["pre_merge_group_count"] == 15
    assert merged.provenance["post_merge_group_count"] == 13
    assert merged.provenance["confirmation_status"] == "USER_CONFIRMED"


def test_same_local_label_is_independent_and_axial_boundary_suggestion_is_advisory() -> None:
    service = BoreholeFractureService(_project_with_fifteen_representatives())
    fit = service.build_local_component_mapping()
    reused = [item for item in fit.mappings if item.local_set_id == "REUSED"]
    assert len(reused) == 2
    assert reused[0].global_set_id != reused[1].global_set_id

    boundary = next(
        item
        for item in service.local_mapping_suggestions()
        if {item.point_a, item.point_b} == {"X00", "X01"}
    )
    assert boundary.axial_angle_deg < 2.0
    unchanged = service.build_local_component_mapping()
    assert unchanged.number_of_sets == 15


def test_noncontinuous_mapping_preserves_pz_components_and_z_only_is_not_zero_intensity() -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    identity = service.build_local_component_mapping()
    by_observation = {item.observation_id: item.component_id for item in identity.mappings}
    assignments = {
        by_observation["P-A-1"]: 4,
        by_observation["Z-C-1"]: 4,
        by_observation["P-B-1"]: 9,
    }
    fit = service.build_local_component_mapping(assignments, user_confirmed=True)
    assert {item.global_set_id for item in fit.sets} == {4, 9}
    assert {item.observation_id for item in fit.mappings} == {"P-A-1", "P-B-1", "Z-C-1"}
    assert all(item.observation_id != "P-A-R" for item in fit.mappings)
    assert fit.provenance["intensity_support_by_global_set"] == {
        "4": "P_SPACING_SUPPORTED",
        "9": "P_SPACING_SUPPORTED",
    }

    z_only = service.build_local_component_mapping(user_confirmed=True)
    z_mapping = next(item for item in z_only.mappings if item.observation_id == "Z-C-1")
    assert z_only.provenance["intensity_support_by_global_set"][str(z_mapping.global_set_id)] == (
        "Z_DIRECTION_ONLY_NO_P_INTENSITY"
    )

    constraints = service._intensity_constraints(fit)
    idw = StableIDW(power=2.0, search_radius=1_000.0, max_neighbors=12, min_neighbors=1)
    near_a, valid_a = service._directions_at(idw, constraints, 4, np.asarray([[10.0, 10.0, 90.0]]), fit)
    near_z, valid_z = service._directions_at(idw, constraints, 4, np.asarray([[20.0, 10.0, 80.0]]), fit)
    assert valid_a.tolist() == [True]
    assert valid_z.tolist() == [True]
    assert not np.allclose(near_a, near_z)

    probabilities = service._local_probabilities(idw, constraints, np.asarray([10.0, 10.0, 90.0]), fit)
    assert 4 in probabilities["global_set_probabilities"]
    z_index = next(
        index for index, item in enumerate(fit.local_components) if item.observation_id == "Z-C-1"
    )
    assert z_index not in probabilities["local_probabilities"]


def test_user_mapping_requires_spatial_local_influence_without_changing_legacy_modes() -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    fit = service.build_local_component_mapping(user_confirmed=True)
    project.joint_sets = _confirmed_sets(fit)
    project.borehole_fracture_state = project.borehole_fracture_state.model_copy(update={"global_fit": fit})
    with pytest.raises(ValueError, match="SPATIALLY_FITTED_WITHIN_GLOBAL_SET"):
        service.build_candidate(
            BoreholeFractureGenerationConfig(
                number_of_sets=len(fit.sets),
                use_confirmed_global_fit=True,
                direction_mode="FIXED_GLOBAL_SET_MEAN",
            )
        )

    legacy = service.fit_global_sets(2, 42)
    legacy_state = service.build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=2, master_seed=42),
        fit_override=legacy,
    )
    assert legacy_state.realizations


def test_confirmed_mapping_round_trips_without_automatic_remapping(tmp_path) -> None:
    project = _synthetic_project()
    service = BoreholeFractureService(project)
    identity = service.build_local_component_mapping()
    assignments = {item.component_id: (4 if index < 2 else 11) for index, item in enumerate(identity.mappings)}
    fit = service.build_local_component_mapping(assignments, user_confirmed=True)
    project.joint_sets = _confirmed_sets(fit)
    project.borehole_fracture_state = project.borehole_fracture_state.model_copy(update={"global_fit": fit})

    target = tmp_path / "synthetic-local-mapping.dfnproj"
    ZipProjectStore().save(project, target)
    reopened = ZipProjectStore().load(target)
    assert reopened.borehole_fracture_state.global_fit == fit
    assert BoreholeFractureService(reopened).authoritative_confirmed_fit() == fit
