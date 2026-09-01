"""GUI regressions for validated M10 configuration submission and wheel safety."""

from __future__ import annotations

import pytest

from dfn_cave_studio.ui.dialogs.m10_dialog import M10ExplicitDFNDialog
from dfn_cave_studio.ui.i18n import language_manager
from dfn_cave_studio.ui.qt_adapter import QApplication, QDialogButtonBox, Qt, QtCore, QtGui
from dfn_cave_studio.visualization.dfn_layer_manager import DFNLayerManager
from tests.gui.test_m10_dialog import FakePlotter, _workflow
from tests.integration.test_m10_persistence_export import make_m10_project


@pytest.mark.parametrize("language", ["en", "zh_CN"])
def test_invalid_auto_config_cannot_commit_in_either_language(qtbot, language: str) -> None:
    manager = language_manager()
    previous = manager.language
    manager.set_language(language, persist=False, force=True)
    try:
        project = make_m10_project()
        before = project.m10_state.model_dump(mode="python")
        dialog = M10ExplicitDFNDialog(project, _workflow(), DFNLayerManager(FakePlotter()))
        qtbot.addWidget(dialog)
        dialog.threshold_mode.setCurrentIndex(dialog.threshold_mode.findData("auto"))
        dialog.small_area_share.setValue(0.998)
        dialog.medium_large_share.setValue(0.7)
        qtbot.mouseClick(
            dialog.buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton
        )
        assert project.m10_state.model_dump(mode="python") == before
        assert not dialog.committed_changes
        assert dialog.small_area_share.value() == pytest.approx(0.998)
        assert dialog.medium_large_share.value() == pytest.approx(0.7)
        assert "invalid" in dialog.status_label.text().lower()
    finally:
        manager.set_language(previous, persist=False, force=True)


def test_manual_mode_ignores_unused_auto_relation_but_switching_back_revalidates(qtbot) -> None:
    project = make_m10_project()
    dialog = M10ExplicitDFNDialog(project, _workflow(), DFNLayerManager(FakePlotter()))
    qtbot.addWidget(dialog)
    dialog.small_area_share.setValue(0.998)
    dialog.medium_large_share.setValue(0.7)
    dialog.manual_sm.setValue(0.5)
    dialog.manual_ml.setValue(2.0)
    dialog.threshold_mode.setCurrentIndex(dialog.threshold_mode.findData("manual"))
    assert dialog._config().size_threshold_mode == "manual"
    dialog.threshold_mode.setCurrentIndex(dialog.threshold_mode.findData("auto"))
    with pytest.raises(ValueError, match="Auto shares"):
        dialog._config()


def test_wheel_event_does_not_change_scientific_spinbox_or_combo(qtbot) -> None:
    dialog = M10ExplicitDFNDialog(
        make_m10_project(), _workflow(), DFNLayerManager(FakePlotter())
    )
    qtbot.addWidget(dialog)
    dialog.show()
    share_before = dialog.small_area_share.value()
    mode_before = dialog.threshold_mode.currentData()
    event = QtGui.QWheelEvent(
        QtCore.QPointF(5.0, 5.0),
        QtCore.QPointF(5.0, 5.0),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(dialog.small_area_share, event)
    QApplication.sendEvent(dialog.threshold_mode, event)
    assert dialog.small_area_share.value() == share_before
    assert dialog.threshold_mode.currentData() == mode_before
