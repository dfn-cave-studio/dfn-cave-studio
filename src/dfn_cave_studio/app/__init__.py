"""
Application entry point for DFN Cave Studio.
"""

import sys
import os
from pathlib import Path


def setup_environment() -> None:
    """Configure environment before application startup."""
    # Ensure src is on path
    src_dir = Path(__file__).resolve().parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    """Main entry point for DFN Cave Studio."""
    setup_environment()

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt, QCoreApplication

    # Application metadata
    QCoreApplication.setApplicationName("DFN Cave Studio")
    QCoreApplication.setApplicationVersion("0.7.0")
    QCoreApplication.setOrganizationName("DFNCaveStudio")

    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # Apply stylesheet
    _apply_stylesheet(app)

    # Create and show main window
    from dfn_cave_studio.ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


def _apply_stylesheet(app: "QApplication") -> None:
    """Apply application-wide stylesheet."""
    style_path = Path(__file__).resolve().parent.parent.parent / "resources" / "styles" / "app.qss"
    if style_path.exists():
        with open(style_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())


if __name__ == "__main__":
    main()
