"""Conftest for GUI tests — ensures offscreen mode, injects FakePlotter,
and mocks all modal dialogs so no test ever blocks on user interaction.

The FakePlotter avoids real VTK/OpenGL initialisation which crashes
headless CI runners.  Modal-dialog mocks prevent hangs on QMessageBox,
QFileDialog, QColorDialog, and QInputDialog.
"""

import os


def pytest_configure(config):
    """Force offscreen rendering for all GUI tests."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import sys
    from pathlib import Path
    src_dir = str(Path(__file__).parent.parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # ── Inject FakePlotter factory ──────────────────────────────────────
    _inject_fake_plotter_factory()

    # ── Mock modal dialogs so they never block tests ────────────────────
    _install_dialog_mocks()


# ═══════════════════════════════════════════════════════════════════════════
# FakePlotter injection
# ═══════════════════════════════════════════════════════════════════════════

def _inject_fake_plotter_factory():
    import sys
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
    from PySide6.QtCore import Qt as _Qt

    class _FakePlotter(QWidget):
        """Minimal fake 3D plotter — no VTK/OpenGL."""

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            label = QLabel("[Fake 3D View]")
            label.setStyleSheet("color: #888; font-size: 14px;")
            label.setAlignment(_Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self._actors = []
            self._camera_actions = []

        def add_mesh(self, *args, **kwargs):
            self._actors.append((args, kwargs))

        def reset_camera(self):
            self._camera_actions.append("reset_camera")

        def show_axes(self):
            self._camera_actions.append("show_axes")

        def view_xy(self):
            self._camera_actions.append("view_xy")

        def view_xz(self):
            self._camera_actions.append("view_xz")

        def view_yz(self):
            self._camera_actions.append("view_yz")

        def render(self):
            self._camera_actions.append("render")

        def clear(self):
            self._actors.clear()

        def close(self):
            super().close()

        def __bool__(self):
            return True

    import dfn_cave_studio.ui.main_window as _mw

    _mw.MainWindow._create_plotter = staticmethod(lambda parent: _FakePlotter(parent))
    sys.modules[__name__].FakePlotter = _FakePlotter


# ═══════════════════════════════════════════════════════════════════════════
# Modal dialog mocks — installed once at session start
# ═══════════════════════════════════════════════════════════════════════════

def _install_dialog_mocks():
    """Globally replace QMessageBox, QFileDialog static methods so no
    test ever blocks waiting for a modal dialog in offscreen mode.

    All dialogs return a safe default:
      - QMessageBox.question → Discard
      - QMessageBox.warning / critical / information → Ok / ignore
      - QFileDialog.getOpenFileName → ("", "")
      - QFileDialog.getSaveFileName → ("", "")
    """
    from unittest.mock import patch, MagicMock

    # We use a module-level patch that is active for the entire test session.
    # These patches are intentionally NEVER stopped so they cover every test.
    import PySide6.QtWidgets as _qtw
    from PySide6.QtWidgets import QMessageBox, QFileDialog

    # QMessageBox static helpers
    _qtw.QMessageBox.question = MagicMock(
        return_value=QMessageBox.StandardButton.Discard,
        side_effect=lambda *a, **kw: QMessageBox.StandardButton.Discard,
    )
    _qtw.QMessageBox.warning = MagicMock(return_value=None)
    _qtw.QMessageBox.critical = MagicMock(return_value=None)
    _qtw.QMessageBox.information = MagicMock(return_value=None)
    _qtw.QMessageBox.about = MagicMock(return_value=None)
    _qtw.QMessageBox.StandardButton = QMessageBox.StandardButton  # preserve enum

    # QFileDialog static helpers
    _qtw.QFileDialog.getOpenFileName = MagicMock(return_value=("", ""))
    _qtw.QFileDialog.getSaveFileName = MagicMock(return_value=("", ""))
    _qtw.QFileDialog.getExistingDirectory = MagicMock(return_value="")

    # QColorDialog (used in joint_set_dialog)
    try:
        from PySide6.QtWidgets import QColorDialog
        _qtw.QColorDialog.getColor = MagicMock(return_value=None)
    except ImportError:
        pass

    # QInputDialog
    try:
        from PySide6.QtWidgets import QInputDialog
        _qtw.QInputDialog.getText = MagicMock(return_value=("", False))
        _qtw.QInputDialog.getInt = MagicMock(return_value=(0, False))
        _qtw.QInputDialog.getDouble = MagicMock(return_value=(0.0, False))
    except ImportError:
        pass
