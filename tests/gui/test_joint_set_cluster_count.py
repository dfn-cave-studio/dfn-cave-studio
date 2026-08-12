"""Real-button GUI regression tests for Mode B K=4, 5, and 6."""

from pathlib import Path

import pandas as pd
import pytest

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.ui.dialogs.m7_joint_set_dialog import M7JointSetDialog
from dfn_cave_studio.ui.qt_adapter import Qt


DEMO = Path(__file__).parents[2] / "examples" / "m7_demo"


@pytest.fixture(scope="module")
def joint_set_project() -> Project:
    """Build a real cleaned demo project with a locked 60/20 holdout."""
    project = Project()
    repository = BoreholeRepository(project)
    for name in ("surveys", "collars", "fractures", "rqd", "domain_intervals"):
        repository.import_dataframe(name, pd.read_csv(DEMO / f"{name}.csv"), str(DEMO / f"{name}.csv"))
    quality = BoreholeQualityService(project)
    quality.run_checks()
    quality.apply_auto_fixes()
    quality.confirm_exclusions()

    hole_ids = sorted(hole.borehole_id for hole in project.borehole_collection)
    holdout = HoldoutService()
    holdout.select_manual(hole_ids, ["BH-07", "BH-08"])
    holdout.lock()
    set_holdout(project, holdout)
    return project


@pytest.mark.parametrize("requested_k", [4, 5, 6])
def test_identify_button_displays_exact_requested_rows(
    joint_set_project: Project, qtbot, requested_k: int
) -> None:
    """The production button path renders K non-empty rows with all 60 assignments."""
    dialog = M7JointSetDialog(joint_set_project, WorkflowController())
    qtbot.addWidget(dialog)
    dialog.show()
    dialog._mode_combo.setCurrentIndex(dialog._mode_combo.findData("automatic"))
    dialog._n_clusters_spin.setValue(requested_k)
    dialog._seed_spin.setValue(42)

    qtbot.mouseClick(dialog._identify_btn, Qt.MouseButton.LeftButton)

    counts = [int(dialog._result_table.item(row, 5).text()) for row in range(dialog._result_table.rowCount())]
    assert dialog._result_table.rowCount() == requested_k
    assert all(count > 0 for count in counts)
    assert sum(counts) == 60
    assert "Calibration fractures used: 60" in dialog._stats_label.text()
    assert "Validation fractures excluded: 20" in dialog._stats_label.text()
