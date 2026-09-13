from __future__ import annotations

import pandas as pd

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.observation_service import ObservationService
from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController


def test_multisource_records_audit_and_scalar_projection_survive_reopen(tmp_path) -> None:
    project = Project()
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame(
            [{"borehole_id": "SYN-1", "collar_x": 0, "collar_y": 0, "collar_z": 20, "final_depth": 10}]
        ),
        "synthetic-collars.csv",
    )
    repository.import_dataframe(
        "fractures",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "fracture_spacing": 10}]),
        "synthetic-spacing.csv",
        import_metadata={"observation_mode": "interval_spacing", "spacing_unit": "cm"},
    )
    repository.import_dataframe(
        "rmr",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "rmr": 50}]),
        "synthetic-rmr.csv",
    )
    repository.import_dataframe(
        "orientation_points",
        pd.DataFrame(
            [
                {
                    "observation_id": "P001-A",
                    "point_id": "001",
                    "x": 1,
                    "y": 2,
                    "z": 3,
                    "dip": 40,
                    "dip_direction": 120,
                    "local_set_id": "A",
                    "joint_spacing_m": 0.5,
                    "joint_num": 2,
                }
            ]
        ),
        "synthetic-points.csv",
    )
    ObservationService(project).synchronize_scalar_samples()
    path = tmp_path / "multisource.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)
    assert reopened.borehole_database.counts("fractures") == project.borehole_database.counts("fractures")
    assert reopened.borehole_database.counts("rmr") == project.borehole_database.counts("rmr")
    assert reopened.borehole_database.counts("orientation_points") == project.borehole_database.counts(
        "orientation_points"
    )
    assert reopened.borehole_database.query("fractures", "formal")[0].original_values["fracture_spacing"] == 10
    assert ObservationService(reopened).spacing_observations()[0].derived_p10 == 10.0
    assert reopened.m9_state.scalar_samples[0].source_record_id is not None
    assert reopened.borehole_database.orientation_point_summaries == project.borehole_database.orientation_point_summaries
    assert ObservationService(reopened).orientation_points()[0].model_dump(mode="json") == (
        ObservationService(project).orientation_points()[0].model_dump(mode="json")
    )


def test_auxiliary_rmr_update_does_not_invalidate_m10_or_m11_steps() -> None:
    project = Project()
    workflow = WorkflowController()
    for step in workflow.get_steps():
        workflow.complete_step(step.step_id)
    project._m7_data = {"workflow": workflow}
    repository = BoreholeRepository(project)
    repository.import_dataframe(
        "collars",
        pd.DataFrame(
            [{"borehole_id": "SYN-1", "collar_x": 0, "collar_y": 0, "collar_z": 10, "final_depth": 10}]
        ),
        "synthetic-collars.csv",
        modification_source="staged_import",
    )
    repository.import_dataframe(
        "rmr",
        pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "rmr": 60}]),
        "synthetic-rmr.csv",
    )
    assert workflow.get_step("explicit_dfn").status == StepStatus.COMPLETED
    assert workflow.get_step("second_voxelization").status == StepStatus.COMPLETED
    assert project.m9_state.scalar_samples[0].parameter_name == "rmr"
    record = repository.query("rmr", "formal")[0]
    repository.edit_record(record.record_id, {**record.values, "rmr": 70})
    assert workflow.get_step("explicit_dfn").status == StepStatus.COMPLETED
    assert workflow.get_step("second_voxelization").status == StepStatus.COMPLETED
    repository.delete_record(record.record_id)
    assert workflow.get_step("explicit_dfn").status == StepStatus.COMPLETED
    assert workflow.get_step("second_voxelization").status == StepStatus.COMPLETED
