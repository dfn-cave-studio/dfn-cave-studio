"""Tests for main window creation and basic functionality."""

import sys
import pytest

from dfn_cave_studio.ui.qt_adapter import (
    QApplication, Qt, QMessageBox,
    PyVistaQtInteractor, HAS_PYVISTAQT,
)

from dfn_cave_studio.ui.main_window import MainWindow


@pytest.fixture
def qapp():
    """Create a QApplication for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


class TestMainWindowCreation:
    """Test that the main window can be created and destroyed."""

    def test_window_creation(self, qapp):
        """Main window should create without errors."""
        window = MainWindow()
        assert window is not None
        assert "DFN Cave Studio" in window.windowTitle()
        window.close()

    def test_window_size(self, qapp):
        """Window should have reasonable default size."""
        window = MainWindow()
        size = window.size()
        assert size.width() >= 1024
        assert size.height() >= 600
        window.close()


class TestMainWindowMenus:
    """Test menu bar functionality."""

    def test_menu_bar_exists(self, qapp):
        """Main window should have a menu bar."""
        window = MainWindow()
        menu_bar = window.menuBar()
        assert menu_bar is not None
        assert len(menu_bar.actions()) > 0
        window.close()

    def test_file_menu(self, qapp):
        """File menu should exist with expected actions."""
        window = MainWindow()
        menu_bar = window.menuBar()
        file_action = None
        for action in menu_bar.actions():
            if "&File" in action.text():
                file_action = action
                break
        assert file_action is not None, "File menu not found"
        window.close()

    def test_help_about(self, qapp):
        """Help menu should contain About action."""
        window = MainWindow()
        menu_bar = window.menuBar()
        for action in menu_bar.actions():
            if "&Help" in action.text():
                help_menu = action.menu()
                texts = [a.text() for a in help_menu.actions()]
                assert any("About" in t for t in texts)
                break
        window.close()


class TestMainWindowDocks:
    """Test dock widget functionality."""

    def test_project_dock_exists(self, qapp):
        """Project dock should exist."""
        window = MainWindow()
        dock_found = False
        for child in window.children():
            type_name = type(child).__name__
            if "Dock" in type_name or hasattr(child, 'toggleViewAction'):
                dock_found = True
                break
        assert dock_found, "No dock widget found"
        window.close()

    def test_log_widget(self, qapp):
        """Log widget should accept messages."""
        window = MainWindow()
        window.log_message("Test message")
        window.log_warning("Test warning")
        window.log_error("Test error")
        assert True  # Should not raise
        window.close()


class TestMainWindowWorkflows:
    """Test application workflow scenarios."""

    def test_non_blocking_callbacks(self, qapp):
        """Menu callbacks that don't open modal dialogs should not crash."""
        window = MainWindow()
        # These are safe - they either show info boxes or are no-ops
        window._on_save_project_as()
        window._on_voxel_settings()
        window._on_domain_manager()
        window._on_connectivity()
        window._on_fragmentation()
        window._on_documentation()
        window.close()

    def test_view_commands(self, qapp):
        """View menu commands should handle missing plotter gracefully."""
        window = MainWindow()
        window._on_reset_view()  # Handles None plotter
        window._on_top_view()
        window._on_front_view()
        window._on_left_view()
        window.close()

    def test_export_callbacks(self, qapp):
        """Export menu callbacks should not crash."""
        window = MainWindow()
        window._on_export_3dec()
        window._on_export_flac3d()
        window._on_export_vtk()
        window._on_settings()
        window.close()

    def test_status_and_progress(self, qapp):
        """Status bar and progress bar should work."""
        window = MainWindow()
        window.set_status("Testing...")
        window.show_progress(50, 100)
        window.hide_progress()
        window.close()

    def test_project_tree(self, qapp):
        """Project tree should have expected structure."""
        window = MainWindow()
        tree = window._project_tree
        assert tree.topLevelItemCount() > 0
        root = tree.topLevelItem(0)
        assert "DFN Cave Studio" in root.text(0)
        window.close()
