"""GUI regression tests for session-safe Chinese/English translation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.ui.dialogs.m10_dialog import M10ExplicitDFNDialog
from dfn_cave_studio.ui.dialogs.m11_dialog import M11SecondVoxelizationDialog
from dfn_cave_studio.ui.i18n import (
    LANGUAGE_CHINESE,
    LANGUAGE_ENGLISH,
    LanguageManager,
    language_manager,
    system_default_language,
    translation_directory,
)
from dfn_cave_studio.ui.main_window import MainWindow
from dfn_cave_studio.ui.qt_adapter import QApplication, QLabel, QLocale, QSettings
from tests.integration.test_m10_persistence_export import make_m10_project


def _workflow() -> WorkflowController:
    workflow = WorkflowController()
    for step in ("bounds", "voxel_grid", "density", "size", "parameter_field"):
        workflow.complete_step(step)
    workflow.mark_ready("explicit_dfn")
    return workflow


def _restore_language(manager, language: str) -> None:
    manager.set_busy("test", False)
    manager.set_language(language, persist=False, force=True)


def test_system_default_language_selection() -> None:
    assert system_default_language(QLocale("zh_CN")) == LANGUAGE_CHINESE
    assert system_default_language(QLocale("en_US")) == LANGUAGE_ENGLISH


def test_preference_survives_manager_restart_and_missing_catalog_falls_back(qtbot, tmp_path: Path) -> None:
    app = QApplication.instance()
    settings_path = tmp_path / "preferences.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    manager = LanguageManager(app, settings=settings, resource_dir=translation_directory())
    assert manager.set_language(LANGUAGE_CHINESE)
    restarted = LanguageManager(
        app,
        settings=QSettings(str(settings_path), QSettings.Format.IniFormat),
        resource_dir=translation_directory(),
    )
    assert restarted.language == LANGUAGE_CHINESE
    assert restarted.initialize()
    assert restarted.set_language(LANGUAGE_ENGLISH, persist=False, force=True)
    assert manager.set_language(LANGUAGE_ENGLISH, persist=False, force=True)
    missing = LanguageManager(
        app,
        settings=QSettings(str(tmp_path / "missing.ini"), QSettings.Format.IniFormat),
        resource_dir=tmp_path / "missing",
    )
    assert not missing.set_language(LANGUAGE_CHINESE, force=True)
    assert missing.language == LANGUAGE_ENGLISH
    label = QLabel("Ready")
    qtbot.addWidget(label)
    label.show()
    assert label.text() == "Ready"
    missing.dispose()
    restarted.dispose()
    manager.dispose()


def test_main_window_language_menu_switches_without_project_mutation(qtbot) -> None:
    manager = language_manager()
    original_language = manager.language
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_new_project()
    project = window._project_store.current_project
    state_before = deepcopy(project.model_dump())
    workflow_before = deepcopy(window._workflow.to_dict())
    dirty_before = window._project_store.is_dirty
    actors_before = set(window._plotter.renderer.actors)
    camera_before = list(window._plotter._camera_actions)
    try:
        assert manager.set_language(LANGUAGE_CHINESE, persist=False, force=True)
        assert window._settings_menu.title() == "设置(&S)"
        assert window._language_menu.title() == "语言(&L)"
        assert manager.set_language(LANGUAGE_ENGLISH, persist=False, force=True)
        assert window._settings_menu.title() == "&Settings"
        assert project.model_dump() == state_before
        assert window._workflow.to_dict() == workflow_before
        assert window._project_store.is_dirty == dirty_before
        assert set(window._plotter.renderer.actors) == actors_before
        assert window._plotter._camera_actions == camera_before
    finally:
        _restore_language(manager, original_language)


def test_m10_m11_stable_combo_values_and_science_survive_translation(qtbot) -> None:
    manager = language_manager()
    original_language = manager.language
    project = make_m10_project()
    workflow = _workflow()
    m10 = M10ExplicitDFNDialog(project, workflow)
    m11 = M11SecondVoxelizationDialog(project, workflow)
    qtbot.addWidget(m10)
    qtbot.addWidget(m11)
    config_before = project.m10_state.config.model_dump()
    m9_arrays_before = {name: value.copy() for name, value in project.m9_state.parameter_field_arrays.items()}
    m11_arrays_before = {
        name: value.copy()
        for result in project.m11_state.results
        for name, value in result.arrays.items()
    }
    workflow_before = deepcopy(workflow.to_dict())
    pending_before = m10._pending_realizations
    stable_values = (
        m10.threshold_mode.currentData(),
        m10.color_by.currentData(),
        m10.display_size_class.currentData(),
        m11.display_mode_combo.currentData(),
        m11.field_combo.currentData(),
        m11.axis_combo.currentData(),
    )
    try:
        assert manager.set_language(LANGUAGE_CHINESE, persist=False, force=True)
        assert m10.windowTitle() == "M10 显式DFN生成"
        assert m10.threshold_mode.currentText() == "自动"
        assert m11.windowTitle() == "M11.1 精确第二次体素化"
        assert manager.set_language(LANGUAGE_ENGLISH, persist=False, force=True)
        assert stable_values == (
            m10.threshold_mode.currentData(),
            m10.color_by.currentData(),
            m10.display_size_class.currentData(),
            m11.display_mode_combo.currentData(),
            m11.field_combo.currentData(),
            m11.axis_combo.currentData(),
        )
        assert project.m10_state.config.model_dump() == config_before
        assert project.m10_state.config.__class__.model_validate(config_before)
        for name, expected in m9_arrays_before.items():
            assert (project.m9_state.parameter_field_arrays[name] == expected).all()
        for result in project.m11_state.results:
            for name, value in result.arrays.items():
                assert (value == m11_arrays_before[name]).all()
        assert workflow.to_dict() == workflow_before
        assert m10._pending_realizations is pending_before
    finally:
        _restore_language(manager, original_language)


def test_language_switching_is_disabled_while_busy(qtbot) -> None:
    manager = language_manager()
    original_language = manager.language
    window = MainWindow()
    qtbot.addWidget(window)
    manager.set_busy("synthetic-save", True)
    try:
        window._refresh_language_actions()
        assert all(not action.isEnabled() for action in window._language_actions.values())
        requested = LANGUAGE_ENGLISH if manager.language == LANGUAGE_CHINESE else LANGUAGE_CHINESE
        assert not manager.set_language(requested)
        assert manager.language == original_language
    finally:
        manager.set_busy("synthetic-save", False)
        _restore_language(manager, original_language)


def test_translation_resources_are_packaged() -> None:
    directory = translation_directory()
    assert (directory / "dfn_cave_studio_zh_CN.ts").is_file()
    assert (directory / "dfn_cave_studio_zh_CN.qm").is_file()


def test_language_is_not_serialized_and_save_reopen_is_identical(qtbot, tmp_path: Path) -> None:
    manager = language_manager()
    original_language = manager.language
    project = make_m10_project()
    path = tmp_path / "language-independent.dfnproj"
    before = project.m10_state.model_dump(exclude={"realizations": {"__all__": {"arrays"}}})
    arrays_before = {
        name: array.copy()
        for realization in project.m10_state.realizations
        for name, array in realization.arrays.items()
    }
    try:
        assert manager.set_language(LANGUAGE_CHINESE, persist=False, force=True)
        ZipProjectStore().save(project, path)
        reopened = ZipProjectStore().load(path)
        assert reopened.m10_state.model_dump(exclude={"realizations": {"__all__": {"arrays"}}}) == before
        for realization in reopened.m10_state.realizations:
            for name, array in realization.arrays.items():
                assert (array == arrays_before[name]).all()
        assert "language" not in reopened.model_dump_json()
    finally:
        _restore_language(manager, original_language)
