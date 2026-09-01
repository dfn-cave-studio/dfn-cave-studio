"""GUI regressions for scrollable M10/M11 dialogs with fixed action areas."""

from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.ui.dialogs.m10_dialog import M10ExplicitDFNDialog
from dfn_cave_studio.ui.dialogs.m11_dialog import M11SecondVoxelizationDialog
from dfn_cave_studio.ui.qt_adapter import QScrollArea
from tests.integration.test_m10_persistence_export import make_m10_project


def _workflow() -> WorkflowController:
    workflow = WorkflowController()
    for step in ("voxel_grid", "density", "size", "parameter_field", "explicit_dfn"):
        workflow.complete_step(step)
    workflow.mark_ready("second_voxelization")
    return workflow


def _assert_responsive(dialog, fixed_buttons) -> None:
    dialog.resize(640, 440)
    dialog.show()
    assert dialog.minimumWidth() <= 640
    assert dialog.minimumHeight() <= 440
    scroll = dialog.findChild(QScrollArea)
    assert scroll is not None and scroll.widgetResizable()
    for button in fixed_buttons:
        assert button.isVisible()
        assert dialog.rect().intersects(button.geometry())


def test_m10_dialog_scrolls_while_generate_and_commit_buttons_stay_accessible(qtbot) -> None:
    dialog = M10ExplicitDFNDialog(make_m10_project(), _workflow())
    qtbot.addWidget(dialog)
    _assert_responsive(dialog, [dialog.generate_button, dialog.buttons])


def test_m11_dialog_scrolls_while_compute_cancel_and_commit_stay_accessible(qtbot) -> None:
    dialog = M11SecondVoxelizationDialog(make_m10_project(), _workflow())
    qtbot.addWidget(dialog)
    _assert_responsive(dialog, [dialog.compute_button, dialog.cancel_button, dialog.buttons])
