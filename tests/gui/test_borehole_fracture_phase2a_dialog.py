"""GUI regression tests for the transactional Phase 2A dialog."""

from __future__ import annotations

from dfn_cave_studio.models.borehole_fracture_realization import BoreholeFractureGenerationConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService
from dfn_cave_studio.ui.dialogs.borehole_fracture_dialog import BoreholeFractureDialog
from dfn_cave_studio.ui.main_window import MainWindow
from dfn_cave_studio.ui.qt_adapter import QDialogButtonBox, Qt
from dfn_cave_studio.visualization.borehole_fracture_renderer import BoreholeFractureRenderer
from tests.unit.test_borehole_fracture_phase2a import _synthetic_project


def _confirm_global_sets(project, count: int = 2):
    service = BoreholeFractureService(project)
    fit = service.fit_global_sets(count, 42)
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
    project.borehole_fracture_state = project.borehole_fracture_state.model_copy(
        update={
            "config": BoreholeFractureGenerationConfig(
                number_of_sets=count,
                use_confirmed_global_fit=True,
            ),
            "global_fit": fit,
        }
    )
    return fit


def test_generate_preview_and_commit_use_summary_not_fracture_rows(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    project = _synthetic_project()
    _confirm_global_sets(project)
    renderer = BoreholeFractureRenderer(window._plotter)
    dialog = BoreholeFractureDialog(project, renderer, parent=window)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.realization_count.setValue(2)
    qtbot.mouseClick(dialog.generate_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=5000)
    assert project.borehole_fracture_state.realizations == []
    assert dialog.realization_table.rowCount() == 2
    qtbot.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    assert 1 <= len(renderer._actor_names) <= len(project.joint_sets) + 1
    actor_count = len(window._plotter._actors_by_name)
    qtbot.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    assert len(window._plotter._actors_by_name) == actor_count
    assert all(not actor.GetPickable() for actor in renderer._actors.values())
    before = project.model_dump(mode="json")
    dirty_before = window._project_store.is_dirty
    workflow_before = window._workflow.to_dict()
    qtbot.mouseClick(dialog.hide_preview_button, Qt.MouseButton.LeftButton)
    assert all(not actor.GetVisibility() for actor in renderer._actors.values())
    assert len(project.borehole_fracture_state.realizations) == 0
    qtbot.mouseClick(dialog.show_preview_check, Qt.MouseButton.LeftButton)
    assert all(actor.GetVisibility() for actor in renderer._actors.values())
    first_names = renderer.actor_names
    dialog.realization_combo.setCurrentIndex(1)
    assert renderer.current_realization_id == dialog.realization_combo.currentData()
    assert renderer.actor_names != first_names
    actor_count = len([name for name in window._plotter._actors_by_name if name.startswith(renderer.PREFIX)])
    for _ in range(5):
        qtbot.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    assert len([name for name in window._plotter._actors_by_name if name.startswith(renderer.PREFIX)]) == actor_count
    assert project.model_dump(mode="json") == before
    assert window._project_store.is_dirty == dirty_before
    assert window._workflow.to_dict() == workflow_before
    qtbot.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
    assert dialog.committed_changes
    assert len(project.borehole_fracture_state.realizations) == 2


def test_cancel_flag_discards_late_candidate_and_next_run_can_commit(qtbot) -> None:
    class _LateWorker:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    project = _synthetic_project()
    fit = _confirm_global_sets(project)
    dialog = BoreholeFractureDialog(project)
    qtbot.addWidget(dialog)
    candidate = BoreholeFractureService(project).build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=2, use_confirmed_global_fit=True),
        fit_override=fit,
    )
    worker = _LateWorker()
    dialog._worker = worker
    dialog._cancel_computation()
    assert worker.cancelled and dialog._discard_worker_result
    dialog._on_generated(worker, candidate)
    assert dialog._pending_state is None
    assert project.borehole_fracture_state.realizations == []

    qtbot.mouseClick(dialog.generate_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=5000)
    assert dialog._pending_state is not None
    dialog.accept()
    assert dialog.committed_changes


def test_running_accept_is_blocked_and_reject_rolls_back_random_status(qtbot) -> None:
    class _RunningWorker:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    project = _synthetic_project(include_random=False)
    _confirm_global_sets(project)
    original = project.borehole_database.model_copy(deep=True)
    dialog = BoreholeFractureDialog(project)
    qtbot.addWidget(dialog)
    row = next(
        row for row in range(dialog.site_table.rowCount()) if dialog.site_table.item(row, 0).text() == "POINT_CLOUD:A"
    )
    dialog.site_table.selectRow(row)
    qtbot.mouseClick(dialog.mark_absent_button, Qt.MouseButton.LeftButton)
    assert project.borehole_database != original
    worker = _RunningWorker()
    dialog._worker = worker
    dialog.accept()
    assert not dialog.committed_changes
    dialog.reject()
    assert worker.cancelled
    assert project.borehole_database == original


def test_main_window_entry_commits_without_starting_m9(monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    project = _synthetic_project()
    fit = _confirm_global_sets(project)
    window._project_store._current_project = project
    m9_before = project.m9_state.model_copy(deep=True)

    def _exec(dialog):
        dialog._pending_state = BoreholeFractureService(project).build_candidate(
            BoreholeFractureGenerationConfig(number_of_sets=2, use_confirmed_global_fit=True),
            fit_override=fit,
        )
        dialog.accept()
        return dialog.DialogCode.Accepted

    monkeypatch.setattr(BoreholeFractureDialog, "exec", _exec)
    window._on_borehole_fracture_realizations()
    assert project.borehole_fracture_state.realizations
    assert project.m9_state == m9_before
    assert window._project_store.is_dirty


def test_preview_replaces_only_phase2a_namespace(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    project = _synthetic_project()
    fit = _confirm_global_sets(project)
    realization = BoreholeFractureService(project).build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=2, use_confirmed_global_fit=True),
        fit_override=fit,
    ).realizations[0]
    window._plotter.add_points([[0, 0, 0]], name="borehole:reference")
    renderer = BoreholeFractureRenderer(window._plotter)
    first = renderer.render(realization)
    renderer.render(realization)
    assert first > 0
    assert len([name for name in window._plotter._actors_by_name if name.startswith(renderer.PREFIX)]) <= (
        len(project.joint_sets) + 1
    )
    assert all(not actor.GetPickable() for actor in renderer._actors.values())
    renderer.clear()
    assert "borehole:reference" in window._plotter._actors_by_name


def test_group_filters_use_authoritative_colours_and_preserve_borehole_actor(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    project = _synthetic_project()
    fit = _confirm_global_sets(project)
    project.joint_sets[0].color = "#123456"
    project.joint_sets[1].color = "#abcdef"
    state = BoreholeFractureService(project).build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=2, use_confirmed_global_fit=True, master_seed=9),
        fit_override=fit,
    )
    project.borehole_fracture_state = state
    borehole_actor = window._plotter.add_points([[0, 0, 0]], name="borehole:green", color="#00ff00")
    renderer = BoreholeFractureRenderer(window._plotter)
    dialog = BoreholeFractureDialog(project, renderer, parent=window)
    qtbot.addWidget(dialog)
    dialog.show()

    qtbot.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    set_two_row = next(
        row
        for row in range(dialog.visible_sets_table.rowCount())
        if dialog.visible_sets_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == 2
    )
    dialog.visible_sets_table.item(set_two_row, 0).setCheckState(Qt.CheckState.Unchecked)
    assert set(renderer.actor_names) <= {
        f"{renderer.PREFIX}{state.realizations[0].realization_id}:global_set:1",
        f"{renderer.PREFIX}{state.realizations[0].realization_id}:random_background",
    }
    set_actor = renderer._actors.get(f"{renderer.PREFIX}{state.realizations[0].realization_id}:global_set:1")
    if set_actor is not None:
        assert set_actor.color == "#123456"
    random_actor = renderer._actors.get(f"{renderer.PREFIX}{state.realizations[0].realization_id}:random_background")
    if random_actor is not None:
        assert random_actor.color == renderer.RANDOM_COLOR
    assert renderer._legend_actor is not None
    assert ("Random Background", renderer.RANDOM_COLOR) in renderer._legend_actor.entries
    assert borehole_actor.color == "#00ff00"
    for _ in range(5):
        qtbot.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    assert renderer.LEGEND_NAME in window._plotter._actors_by_name
    qtbot.mouseClick(dialog.random_visible_check, Qt.MouseButton.LeftButton)
    assert not dialog.random_visible_check.isChecked()
    assert not any(name.endswith(":random_background") for name in renderer.actor_names)
    before = project.model_dump(mode="json")
    qtbot.mouseClick(dialog.hide_all_groups_button, Qt.MouseButton.LeftButton)
    assert renderer.actor_names == set()
    assert renderer._legend_actor is None
    assert project.model_dump(mode="json") == before

    qtbot.mouseClick(dialog.show_all_groups_button, Qt.MouseButton.LeftButton)
    expected_colours = {item.set_id: item.color for item in project.joint_sets}
    for set_id, colour in expected_colours.items():
        actor = renderer._actors.get(f"{renderer.PREFIX}{state.realizations[0].realization_id}:global_set:{set_id}")
        if actor is not None:
            assert actor.color == colour
            assert not actor.GetPickable()
    assert borehole_actor.color == "#00ff00"


def test_group_checkboxes_random_zero_and_realization_switch_stay_in_sync(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    project = _synthetic_project()
    fit = _confirm_global_sets(project)
    state = BoreholeFractureService(project).build_candidate(
        BoreholeFractureGenerationConfig(
            number_of_sets=2,
            realization_count=2,
            use_confirmed_global_fit=True,
            master_seed=42,
        ),
        fit_override=fit,
    )
    project.borehole_fracture_state = state
    renderer = BoreholeFractureRenderer(window._plotter)
    dialog = BoreholeFractureDialog(project, renderer, parent=window)
    qtbot.addWidget(dialog)
    dialog.show()
    before = project.model_dump(mode="json")
    workflow_before = window._workflow.to_dict()
    dirty_before = window._project_store.is_dirty

    qtbot.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    assert dialog.visible_sets_table.rowCount() == len(project.joint_sets)
    for row, joint_set in enumerate(sorted(project.joint_sets, key=lambda item: item.set_id)):
        assert dialog.visible_sets_table.item(row, 2).text() == str(joint_set.set_id)
        assert dialog.visible_sets_table.item(row, 3).text() == joint_set.name
        assert int(dialog.visible_sets_table.item(row, 4).text()) == state.realizations[0].global_set_counts.get(
            joint_set.set_id, 0
        )

    first = dialog.visible_sets_table.item(0, 0)
    first_id = first.data(Qt.ItemDataRole.UserRole)
    first.setCheckState(Qt.CheckState.Unchecked)
    assert not any(name.endswith(f":global_set:{first_id}") for name in renderer.actor_names)
    assert all(not actor.GetPickable() for actor in renderer._actors.values())

    random_count = state.realizations[0].random_background_count
    assert random_count > 0
    assert dialog.random_visible_check.text() == f"Random Background: {random_count:,}"
    if random_count:
        assert dialog.random_visible_check.isEnabled()
        qtbot.mouseClick(dialog.random_visible_check, Qt.MouseButton.LeftButton)
        assert not any(name.endswith(":random_background") for name in renderer.actor_names)
        qtbot.mouseClick(dialog.random_visible_check, Qt.MouseButton.LeftButton)
        assert any(name.endswith(":random_background") for name in renderer.actor_names)

    qtbot.mouseClick(dialog.hide_all_groups_button, Qt.MouseButton.LeftButton)
    assert renderer.actor_names == set()
    qtbot.mouseClick(dialog.show_all_groups_button, Qt.MouseButton.LeftButton)
    assert all(dialog.visible_sets_table.item(row, 0).checkState() == Qt.CheckState.Checked for row in range(2))
    actor_count = len(renderer.actor_names)
    dialog.realization_combo.setCurrentIndex(1)
    assert renderer.current_realization_id == state.realizations[1].realization_id
    assert len(renderer.actor_names) <= actor_count
    for _ in range(10):
        qtbot.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    assert len([name for name in window._plotter._actors_by_name if name.startswith(renderer.PREFIX)]) == len(
        renderer.actor_names
    )
    assert project.model_dump(mode="json") == before
    assert window._workflow.to_dict() == workflow_before
    assert window._project_store.is_dirty == dirty_before


def test_random_zero_disables_filter_and_creates_no_empty_actor(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    project = _synthetic_project(include_random=False)
    fit = _confirm_global_sets(project)
    state = BoreholeFractureService(project).build_candidate(
        BoreholeFractureGenerationConfig(number_of_sets=2, use_confirmed_global_fit=True),
        fit_override=fit,
    )
    project.borehole_fracture_state = state
    renderer = BoreholeFractureRenderer(window._plotter)
    dialog = BoreholeFractureDialog(project, renderer, parent=window)
    qtbot.addWidget(dialog)
    qtbot.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    assert state.realizations[0].random_background_count == 0
    assert dialog.random_visible_check.text() == "Random Background: 0"
    assert not dialog.random_visible_check.isEnabled()
    assert not any(name.endswith(":random_background") for name in renderer.actor_names)
