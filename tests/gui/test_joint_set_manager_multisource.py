"""GUI regressions for joint-set dialogs with incomplete multi-source observations."""

from __future__ import annotations

import pandas as pd
import pytest

from dfn_cave_studio.models.borehole_database import BoreholeDataType
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService
from dfn_cave_studio.ui.dialogs.borehole_fracture_dialog import BoreholeFractureDialog
from dfn_cave_studio.ui.dialogs.joint_set_dialog import JointSetManagerDialog
from dfn_cave_studio.ui.main_window import MainWindow
from dfn_cave_studio.ui.qt_adapter import QDialog, Qt
from dfn_cave_studio.services.joint_set_service import joint_set_color
from tests.unit.test_borehole_fracture_phase2a import _synthetic_project
from tests.unit.test_multisource_observations import _project as _base_borehole_project


def _joint_set(set_id: int, dip_direction: float, dip: float) -> JointSetConfig:
    return JointSetConfig(
        set_id=set_id,
        name=f"Synthetic set {set_id}",
        color="#1976d2" if set_id == 4 else "#d32f2f",
        orientation=OrientationDistribution(mean_dip_direction=dip_direction, mean_dip=dip, kappa=25.0),
    )


def _spacing_or_axis_project(mode: str) -> Project:
    project, repository = _base_borehole_project()
    if mode == "spacing":
        row = {"hole_id": "SYN-1", "from_depth": 0.0, "to_depth": 10.0, "fracture_spacing": 0.5}
        metadata = {"observation_mode": "interval_spacing", "spacing_unit": "m"}
    else:
        row = {"hole_id": "SYN-1", "depth": 5.0, "axis_plane_angle": 35.0}
        metadata = {"observation_mode": "axis_plane_angle"}
    repository.import_dataframe(
        BoreholeDataType.FRACTURES,
        pd.DataFrame([row]),
        f"synthetic-{mode}.csv",
        import_metadata=metadata,
    )
    return project


@pytest.mark.parametrize(
    ("project", "representative_count", "random_count"),
    [
        pytest.param(Project(), 0, 0, id="no-direction-data"),
        pytest.param(_spacing_or_axis_project("spacing"), 0, 0, id="spacing-only"),
        pytest.param(_spacing_or_axis_project("axis"), 0, 0, id="axis-angle-only"),
        pytest.param(_synthetic_project(), 3, 1, id="pz-representatives-only"),
    ],
)
def test_joint_set_manager_opens_and_closes_without_legacy_orientations(
    qtbot,
    project: Project,
    representative_count: int,
    random_count: int,
) -> None:
    dialog = JointSetManagerDialog(
        joint_sets=project.joint_sets,
        model_volume=project.model_bounds.volume,
        borehole_collection=project.borehole_collection,
        project=project,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog._list.count() == 0
    assert not dialog._import_from_obs_btn.isEnabled()
    assert "No confirmed joint sets" in dialog._data_status.text()
    assert dialog._representative_table.rowCount() == representative_count
    assert dialog._random_table.rowCount() == random_count
    dialog.reject()


def test_joint_set_manager_preserves_existing_confirmed_set_ids(qtbot) -> None:
    project = _synthetic_project()
    project.joint_sets = [_joint_set(4, 20.0, 35.0), _joint_set(9, 210.0, 65.0)]
    dialog = JointSetManagerDialog(
        joint_sets=project.joint_sets,
        model_volume=project.model_bounds.volume,
        borehole_collection=project.borehole_collection,
        project=project,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog._list.count() == 2
    assert "Existing confirmed project joint sets" in dialog._data_status.text()
    dialog._on_accept()
    assert [item.set_id for item in dialog.get_joint_sets()] == [4, 9]


def test_joint_set_manager_recognizes_complete_legacy_orientation(qtbot) -> None:
    project, repository = _base_borehole_project()
    repository.import_dataframe(
        BoreholeDataType.FRACTURES,
        pd.DataFrame(
            [{"hole_id": "SYN-1", "depth": 5.0, "dip_direction": 120.0, "dip": 40.0, "set_id": 1}]
        ),
        "synthetic-full-orientation.csv",
        import_metadata={"observation_mode": "full_orientation"},
    )
    dialog = JointSetManagerDialog(
        joint_sets=project.joint_sets,
        model_volume=project.model_bounds.volume,
        borehole_collection=project.borehole_collection,
        project=project,
    )
    qtbot.addWidget(dialog)
    assert dialog._complete_borehole_orientation_count() == 1
    assert dialog._import_from_obs_btn.isEnabled()
    dialog.reject()


def test_main_window_joint_set_entry_constructs_safely(monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    project = _synthetic_project()
    window._project_store._current_project = project
    monkeypatch.setattr(JointSetManagerDialog, "exec", lambda _dialog: QDialog.DialogCode.Rejected)
    window._on_joint_set_manager()
    assert project.joint_sets == []


def test_imported_representatives_fit_confirm_persist_and_cancel(monkeypatch, qtbot, tmp_path) -> None:
    project = _synthetic_project()
    original_sets = list(project.joint_sets)
    original_fit = project.borehole_fracture_state.global_fit
    preview = JointSetManagerDialog(
        joint_sets=project.joint_sets,
        model_volume=project.model_bounds.volume,
        borehole_collection=project.borehole_collection,
        project=project,
    )
    qtbot.addWidget(preview)
    preview.show()
    preview._global_k_spin.setValue(2)
    qtbot.mouseClick(preview._fit_imported_button, Qt.MouseButton.LeftButton)
    assert preview._list.count() == 2
    assert preview._mapping_table.rowCount() == 3
    assert preview._random_table.rowCount() == 1
    assert "Imported Local Representatives: 3" in preview._count_summary.text()
    assert "strictly selected K=2" in preview._count_summary.text()
    assert len(preview.get_joint_sets()) == preview._global_k_spin.value()
    assert all(item.provenance["size_status"] == "UNRESOLVED" for item in preview.get_joint_sets())
    assert all(item.provenance["p32_status"] == "DERIVED_LATER_BY_M9" for item in preview.get_joint_sets())
    assert all(item.target_p32 == pytest.approx(0.5) for item in preview.get_joint_sets())
    assert not preview._p32_spin.isVisible()
    assert not preview._dist_combo.isVisible()
    assert "Derived later by M9" in preview._intensity_status_label.text()
    assert "UNRESOLVED" in preview._size_status_label.text()
    assert [item.color for item in preview.get_joint_sets()] == [joint_set_color(1), joint_set_color(2)]
    assert not preview._p32_spin.isVisible()
    assert not preview._dist_combo.isVisible()
    first_model = preview.get_global_fit().sets[0]
    assert preview.get_joint_sets()[0].provenance["kappa_status"] == first_model.kappa_status
    fitted_orientations = [item.orientation.model_copy(deep=True) for item in preview.get_joint_sets()]
    preview._on_accept()
    assert [item.orientation for item in preview.get_joint_sets()] == fitted_orientations
    preview.reject()
    assert project.joint_sets == original_sets
    assert project.borehole_fracture_state.global_fit == original_fit

    window = MainWindow()
    qtbot.addWidget(window)
    window._project_store._current_project = project

    def confirm(dialog):
        dialog._global_k_spin.setValue(2)
        dialog._fit_imported_representatives()
        dialog._on_accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(JointSetManagerDialog, "exec", confirm)
    window._on_joint_set_manager()
    assert len(project.joint_sets) == 2
    fit = project.borehole_fracture_state.global_fit
    assert fit is not None
    assert len(fit.mappings) == 3
    assert all(item.observation_id != "P-A-R" for item in fit.mappings)
    path = tmp_path / "confirmed-mapping.dfnproj"
    ZipProjectStore().save(project, path)
    reopened = ZipProjectStore().load(path)
    assert reopened.joint_sets == project.joint_sets
    assert reopened.borehole_fracture_state.global_fit == fit

    reopened_dialog = JointSetManagerDialog(
        joint_sets=reopened.joint_sets,
        model_volume=reopened.model_bounds.volume,
        borehole_collection=reopened.borehole_collection,
        project=reopened,
    )
    qtbot.addWidget(reopened_dialog)
    assert reopened_dialog.get_global_fit() == fit
    reopened_dialog._on_accept()
    assert reopened_dialog.get_global_fit() == fit

    unchanged_window = MainWindow()
    qtbot.addWidget(unchanged_window)
    unchanged_window._project_store._current_project = reopened
    state_before = reopened.borehole_fracture_state.model_copy(deep=True)
    workflow_before = unchanged_window._workflow.to_dict()

    def accept_without_changes(dialog):
        dialog._on_accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(JointSetManagerDialog, "exec", accept_without_changes)
    unchanged_window._on_joint_set_manager()
    assert reopened.borehole_fracture_state == state_before
    assert unchanged_window._workflow.to_dict() == workflow_before
    assert not unchanged_window._project_store.is_dirty


def test_phase2a_uses_only_confirmed_read_only_k_and_preserves_point_local_ids(qtbot) -> None:
    project = _synthetic_project()
    project.joint_sets = [_joint_set(4, 20.0, 35.0), _joint_set(9, 210.0, 65.0)]
    fit = BoreholeFractureService(project).use_confirmed_project_sets()
    project.borehole_fracture_state = project.borehole_fracture_state.model_copy(update={"global_fit": fit})
    dialog = BoreholeFractureDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()

    assert not hasattr(dialog, "fit_source_combo")
    assert not hasattr(dialog, "k_spin")
    assert dialog.confirmed_groups_label.text() == "K=2"
    assert fit.algorithm in dialog.confirmed_fit_label.text()
    assert "mapping=CONFIRMED" in dialog.confirmed_fit_label.text()
    assert dialog.generate_button.isEnabled()
    confirmed = dialog.service.use_confirmed_project_sets()
    assert {item.global_set_id for item in confirmed.sets} == {4, 9}
    local_red = [item for item in confirmed.mappings if item.local_set_id == "LOCAL-RED"]
    assert len(local_red) == 2
    assert {item.point_key for item in local_red} == {"POINT_CLOUD:A", "POINT_CLOUD:B"}
    assert {item.global_set_id for item in local_red} == {4, 9}

    dialog.reject()


def test_phase2a_without_confirmed_sets_disables_generation_and_only_opens_joint_set_management(qtbot) -> None:
    dialog = BoreholeFractureDialog(_synthetic_project())
    qtbot.addWidget(dialog)
    assert dialog.confirmed_groups_label.text() == "Not confirmed"
    assert "mapping=NOT CONFIRMED" in dialog.confirmed_fit_label.text()
    assert not dialog.generate_button.isEnabled()
    assert dialog.open_joint_sets_button.isEnabled()
    dialog.reject()


def test_phase2a_joint_set_button_refreshes_only_after_authoritative_confirmation(monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    project = _synthetic_project()
    window._project_store._current_project = project
    dialog = BoreholeFractureDialog(project, parent=window)
    qtbot.addWidget(dialog)

    monkeypatch.setattr(window, "_on_joint_set_manager", lambda: None)
    qtbot.mouseClick(dialog.open_joint_sets_button, Qt.MouseButton.LeftButton)
    assert project.joint_sets == []
    assert not dialog.generate_button.isEnabled()

    def confirm() -> None:
        service = BoreholeFractureService(project)
        fit = service.fit_global_sets(2, 42)
        project.joint_sets = [
            JointSetConfig(
                set_id=item.global_set_id,
                orientation=OrientationDistribution(
                    mean_dip_direction=item.mean_dip_direction,
                    mean_dip=item.mean_dip,
                    kappa=item.kappa,
                ),
            )
            for item in fit.sets
        ]
        project.borehole_fracture_state = project.borehole_fracture_state.model_copy(update={"global_fit": fit})

    monkeypatch.setattr(window, "_on_joint_set_manager", confirm)
    qtbot.mouseClick(dialog.open_joint_sets_button, Qt.MouseButton.LeftButton)
    assert dialog.confirmed_groups_label.text() == "K=2"
    assert "mapping=CONFIRMED" in dialog.confirmed_fit_label.text()
    assert dialog.generate_button.isEnabled()


def test_joint_set_confirmation_requires_fitting_available_candidates(qtbot) -> None:
    dialog = JointSetManagerDialog(project=_synthetic_project())
    qtbot.addWidget(dialog)
    dialog._on_accept()
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert "Fit and review" in dialog._fit_status.text()
    assert dialog.get_joint_sets() == []
