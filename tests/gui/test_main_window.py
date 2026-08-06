"""Tests for main window creation and basic functionality.

All MainWindow instances are managed by qtbot.addWidget() for proper
Qt resource cleanup.  FakePlotter is injected by conftest so no real
VTK/OpenGL initialisation occurs.
"""

import sys
import pytest

from PySide6.QtCore import Qt, QTimer


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_window(qtbot):
    """Create a MainWindow managed by qtbot for proper cleanup."""
    from dfn_cave_studio.ui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _cleanup_window(window, qtbot):
    """Teardown: stop timer, clear dirty, close, process events.

    *Must* clear dirty *before* close() because closeEvent checks
    is_dirty and shows a modal QMessageBox.question otherwise.

    Do NOT call deleteLater() — qtbot.addWidget() handles that
    in its own teardown; calling it here causes "Internal C++ object
    already deleted" errors.
    """
    window._auto_save_timer.stop()
    window._project_store._dirty = False
    window.close()
    qtbot.wait(50)


# ── Tests ────────────────────────────────────────────────────────────────


class TestMainWindowCreation:
    """Test that the main window can be created and destroyed."""

    def test_window_creation(self, qtbot):
        window = _make_window(qtbot)
        assert window is not None
        assert "DFN Cave Studio" in window.windowTitle()
        _cleanup_window(window, qtbot)

    def test_window_size(self, qtbot):
        window = _make_window(qtbot)
        size = window.size()
        assert size.width() >= 1024
        assert size.height() >= 600
        _cleanup_window(window, qtbot)


class TestMainWindowMenus:
    """Test menu bar functionality."""

    def test_menu_bar_exists(self, qtbot):
        window = _make_window(qtbot)
        menu_bar = window.menuBar()
        assert menu_bar is not None
        assert len(menu_bar.actions()) > 0
        _cleanup_window(window, qtbot)

    def test_file_menu(self, qtbot):
        window = _make_window(qtbot)
        menu_bar = window.menuBar()
        file_action = None
        for action in menu_bar.actions():
            if "&File" in action.text():
                file_action = action
                break
        assert file_action is not None, "File menu not found"
        _cleanup_window(window, qtbot)

    def test_help_about(self, qtbot):
        window = _make_window(qtbot)
        menu_bar = window.menuBar()
        for action in menu_bar.actions():
            if "&Help" in action.text():
                help_menu = action.menu()
                texts = [a.text() for a in help_menu.actions()]
                assert any("About" in t for t in texts)
                break
        _cleanup_window(window, qtbot)


class TestMainWindowDocks:
    """Test dock widget functionality."""

    def test_project_dock_exists(self, qtbot):
        window = _make_window(qtbot)
        dock_found = False
        for child in window.children():
            type_name = type(child).__name__
            if "Dock" in type_name or hasattr(child, 'toggleViewAction'):
                dock_found = True
                break
        assert dock_found, "No dock widget found"
        _cleanup_window(window, qtbot)

    def test_log_widget(self, qtbot):
        window = _make_window(qtbot)
        window.log_message("Test message")
        window.log_warning("Test warning")
        window.log_error("Test error")
        assert True  # Should not raise
        _cleanup_window(window, qtbot)


class TestMainWindowWorkflows:
    """Test application workflow scenarios."""

    def test_non_blocking_callbacks(self, qtbot):
        """Menu callbacks that don't open modal dialogs should not crash."""
        window = _make_window(qtbot)
        # Only callbacks that are safe without event loop or project:
        window._on_domain_manager()   # logs "not yet implemented"
        window._on_documentation()    # logs "Documentation requested"
        window.set_status("Testing status")
        window.log_message("Test log entry")
        _cleanup_window(window, qtbot)

    def test_view_commands(self, qtbot):
        """View menu commands should handle missing plotter gracefully."""
        window = _make_window(qtbot)
        window._on_reset_view()
        window._on_top_view()
        window._on_front_view()
        window._on_left_view()
        _cleanup_window(window, qtbot)

    def test_export_callbacks(self, qtbot):
        """Export menu callbacks handle missing project gracefully."""
        window = _make_window(qtbot)
        window._on_settings()
        window._on_documentation()
        _cleanup_window(window, qtbot)

    def test_status_and_progress(self, qtbot):
        """Status bar and progress bar should work."""
        window = _make_window(qtbot)
        window.set_status("Testing...")
        window.show_progress(50, 100)
        window.hide_progress()
        _cleanup_window(window, qtbot)

    def test_project_tree(self, qtbot):
        """Project tree should have expected structure."""
        window = _make_window(qtbot)
        tree = window._project_tree
        assert tree.topLevelItemCount() > 0
        root = tree.topLevelItem(0)
        assert "DFN Cave Studio" in root.text(0)
        _cleanup_window(window, qtbot)
