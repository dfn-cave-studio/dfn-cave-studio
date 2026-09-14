"""GUI contracts for explicit, transactional M9 density source selection."""

from __future__ import annotations

from dfn_cave_studio.models.m9 import M9DensityInputMode
from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.ui.qt_adapter import QDialog, QDialogButtonBox, Qt
from dfn_cave_studio.ui.dialogs.m9_dialogs import M9DensityDialog
from tests.unit.test_m9_phase2a_adapter import _project_with_realizations


def test_density_dialog_never_auto_selects_a_phase2a_realization(qtbot) -> None:
    project = _project_with_realizations(2)
    dialog = M9DensityDialog(project, WorkflowController())
    qtbot.addWidget(dialog)
    assert dialog.input_mode.currentData() == M9DensityInputMode.FORMAL_OBSERVATIONS.value
    dialog.input_mode.setCurrentIndex(dialog.input_mode.findData(M9DensityInputMode.PHASE2A_REALIZATION.value))
    assert dialog.realization.isEnabled()
    assert dialog.realization.currentData() is None


def test_cancelled_late_density_result_is_discarded_and_next_run_can_commit(qtbot) -> None:
    project = _project_with_realizations(1)
    dialog = M9DensityDialog(project, WorkflowController())
    qtbot.addWidget(dialog)
    original = project.m9_state

    class Worker:
        def cancel(self):
            self.cancelled = True

    first = Worker()
    dialog._worker = first
    dialog._cancel_computation()
    candidate = original.model_copy(update={"density_input_realization_id": "late"})
    dialog._calculation_done(candidate, first)
    assert project.m9_state == original
    second = Worker()
    dialog._worker = second
    accepted = original.model_copy(update={"density_input_realization_id": "accepted"})
    dialog._calculation_done(accepted, second)
    assert project.m9_state.density_input_realization_id == "accepted"


def test_reject_while_running_prevents_late_commit(qtbot) -> None:
    project = _project_with_realizations(1)
    dialog = M9DensityDialog(project, WorkflowController())
    qtbot.addWidget(dialog)
    original = project.m9_state

    class Worker:
        def cancel(self):
            self.cancelled = True

    worker = Worker()
    dialog._worker = worker
    dialog.reject()
    late = original.model_copy(update={"density_input_realization_id": "late"})
    dialog._calculation_done(late, worker)
    assert project.m9_state == original


def test_ok_button_and_direct_accept_cannot_close_while_density_worker_runs(qtbot, monkeypatch) -> None:
    project = _project_with_realizations(1)
    workflow = WorkflowController()
    dialog = M9DensityDialog(project, workflow)
    qtbot.addWidget(dialog)
    dialog.show()
    original = project.m9_state.model_copy(deep=True)

    class Worker:
        def cancel(self):
            self.cancelled = True

    worker = Worker()
    dialog._worker = worker
    dialog._set_calculation_running(True)
    ok_button = dialog.dialog_buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button is not None and not ok_button.isEnabled()
    qtbot.mouseClick(ok_button, Qt.MouseButton.LeftButton)
    assert dialog.isVisible()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert project.m9_state == original

    failures: list[str] = []
    monkeypatch.setattr(dialog, "_fail", failures.append)
    dialog.accept()
    assert failures == ["Wait for density calculation to finish or cancel the dialog"]
    assert dialog.isVisible()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert project.m9_state == original

    dialog._cancel_computation()
    late = original.model_copy(update={"density_input_realization_id": "late"})
    dialog._calculation_done(late, worker)
    assert project.m9_state == original
