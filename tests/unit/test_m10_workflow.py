"""M10 dependency and v0.9.1 workflow migration tests."""

from dfn_cave_studio.services.workflow_controller import StepStatus, WorkflowController


def test_explicit_dfn_depends_on_scientific_m9_inputs_not_validation():
    workflow = WorkflowController()
    step = workflow.get_step("explicit_dfn")
    assert step.depends_on == ["density", "size", "parameter_field", "voxel_grid"]
    workflow.complete_step("parameter_field")
    workflow.complete_step("explicit_dfn")
    workflow.invalidate_from("parameter_field")
    assert workflow.get_step("explicit_dfn").status == StepStatus.STALE


def test_old_m9_workflow_restore_adds_m10_step_without_changing_old_states():
    old = WorkflowController()
    old.complete_step("parameter_field")
    payload = old.to_dict()
    payload["steps"].pop("explicit_dfn")
    payload["enabled_steps"].remove("explicit_dfn")
    restored = WorkflowController()
    restored.from_dict(payload)
    assert restored.get_step("parameter_field").status == StepStatus.COMPLETED
    assert restored.get_step("explicit_dfn").status == StepStatus.NOT_STARTED
    assert restored.is_enabled("explicit_dfn")
