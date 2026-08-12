"""End-to-end M8 project to M9 parameter-field workflow."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

from dfn_cave_studio.models.bounds import VoxelConfig
from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.joint_set_service import JointSetService
from dfn_cave_studio.services.m7_state import get_workflow, set_holdout, set_workflow
from dfn_cave_studio.services.m9_service import M9Service
from dfn_cave_studio.services.spatial_domain_service import SpatialDomainService
from dfn_cave_studio.services.workflow_controller import WorkflowController


DEMO = Path(__file__).parents[2] / "examples" / "m7_demo"


def test_complete_m9_demo_pipeline_save_reopen_and_export(tmp_path):
    project = Project()
    workflow = WorkflowController()
    set_workflow(project, workflow)
    repository = BoreholeRepository(project)
    for name in ("surveys", "collars", "fractures", "rqd", "domain_intervals"):
        repository.import_dataframe(name, pd.read_csv(DEMO / f"{name}.csv"), str(DEMO / f"{name}.csv"))
    quality = BoreholeQualityService(project)
    quality.run_checks()
    quality.apply_auto_fixes()
    quality.confirm_exclusions()
    quality.confirm_complete(workflow)
    assert repository.database.counts() == {"raw": 284, "formal": 272, "excluded": 12, "pending": 0}

    hole_ids = sorted(hole.borehole_id for hole in project.borehole_collection)
    holdout = HoldoutService()
    holdout.select_manual(hole_ids, ["BH-07", "BH-08"])
    holdout.lock()
    set_holdout(project, holdout)
    for step_id in ("import", "clean", "holdout", "domains", "joint_sets", "bounds", "voxel_grid"):
        workflow.complete_step(step_id)
    identified = JointSetService(random_seed=42).identify_from_imported(
        project.borehole_collection, set(holdout.calibration_holes), set(holdout.validation_holes)
    )
    project.joint_sets = list(identified.sets.values())
    assert identified.calibration_count == 60
    assert identified.validation_count == 20

    bounds = SpatialDomainService.automatic_bounds(project.borehole_collection, outward_margin=1)
    project.voxel_config = VoxelConfig(cell_size_x=50, cell_size_y=50, cell_size_z=50)
    generation = SpatialDomainService.generation_domain(bounds, project.voxel_config, 5, 1)
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=bounds, generation_domain=generation)

    service = M9Service(project)
    service.calculate_density(
        DensitySettings(
            interval_length=25,
            method=DensityMethod.IDW,
            min_neighbors=1,
            max_neighbors=8,
            global_fallback=True,
            monte_carlo_samples=1000,
        )
    )
    workflow.complete_step("density")
    assert all(hole not in project.m9_state.provenance["density_fit_holes"] for hole in holdout.validation_holes)
    assert {row.role for row in project.m9_state.p10_intervals} == {"calibration", "validation"}
    service.set_assumed_sizes("fixed", {"radius": 2.0}, 2.0, 2.000001, user_defined=False)
    workflow.complete_step("size")
    service.build_parameter_field()
    workflow.complete_step("parameter_field")
    summary = service.validate()
    workflow.complete_step("validation")
    assert summary.valid_interval_count + summary.no_data_interval_count > 0
    arrays_before = {name: values.copy() for name, values in project.m9_state.parameter_field_arrays.items()}

    exported = service.export(tmp_path / "export")
    assert {path.name for path in exported} >= {
        "p10_intervals.csv",
        "p32_estimates.csv",
        "density_model.json",
        "fracture_size_models.json",
        "voxel_parameter_field.npz",
        "voxel_parameter_field.vti",
        "validation_results.csv",
        "validation_summary.json",
    }
    with np.load(tmp_path / "export" / "voxel_parameter_field.npz") as archive:
        np.testing.assert_equal(archive["cell_state"], arrays_before["cell_state"])
    vti = pv.read(tmp_path / "export" / "voxel_parameter_field.vti")
    metadata = project.m9_state.parameter_field_metadata
    assert metadata is not None
    assert vti.dimensions == tuple(value + 1 for value in metadata.shape)
    assert vti.origin == metadata.origin
    assert vti.spacing == metadata.spacing
    assert vti.n_cells == int(np.prod(metadata.shape))
    assert {"cell_state", "p32_total"}.issubset(vti.cell_data)

    path = tmp_path / "m9_demo.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)
    assert restored.m9_state.p10_intervals == project.m9_state.p10_intervals
    assert restored.m9_state.p32_estimates == project.m9_state.p32_estimates
    assert restored.m9_state.density_settings == project.m9_state.density_settings
    assert restored.m9_state.size_models == project.m9_state.size_models
    assert restored.m9_state.parameter_field_metadata == project.m9_state.parameter_field_metadata
    assert restored.m9_state.validation_summary == project.m9_state.validation_summary
    assert restored.m9_state.validation_results == project.m9_state.validation_results
    assert get_workflow(restored).to_dict() == workflow.to_dict()
    for name, before in arrays_before.items():
        np.testing.assert_equal(restored.m9_state.parameter_field_arrays[name], before)

    second_path = tmp_path / "m9_demo_resaved.dfnproj"
    ZipProjectStore().save(restored, second_path)
    reopened_again = ZipProjectStore().load(second_path)
    assert reopened_again.m9_state.p10_intervals == restored.m9_state.p10_intervals
    assert reopened_again.m9_state.p32_estimates == restored.m9_state.p32_estimates
    assert reopened_again.m9_state.size_models == restored.m9_state.size_models
    assert reopened_again.m9_state.validation_results == restored.m9_state.validation_results
    assert get_workflow(reopened_again).to_dict() == get_workflow(restored).to_dict()
    for name, before in arrays_before.items():
        np.testing.assert_equal(reopened_again.m9_state.parameter_field_arrays[name], before)
