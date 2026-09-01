"""Logical-pixel helpers for keeping large dialogs inside their active screen."""

from __future__ import annotations

from dfn_cave_studio.ui.qt_adapter import QApplication, QDialog, QPoint, QRect, QSize


def fitted_dialog_rect(
    available: QRect,
    preferred: QSize,
    *,
    width_fraction: float = 0.92,
    height_fraction: float = 0.92,
) -> QRect:
    """Return a centred logical-pixel rectangle contained in available geometry."""
    if available.width() <= 0 or available.height() <= 0:
        raise ValueError("Screen available geometry must be positive")
    maximum_width = max(1, int(available.width() * width_fraction))
    maximum_height = max(1, int(available.height() * height_fraction))
    width = min(max(1, preferred.width()), maximum_width)
    height = min(max(1, preferred.height()), maximum_height)
    x = available.x() + (available.width() - width) // 2
    y = available.y() + (available.height() - height) // 2
    return QRect(QPoint(x, y), QSize(width, height))


def fit_dialog_to_screen(dialog: QDialog, preferred: QSize) -> QRect:
    """Fit and centre a dialog on its current/parent screen without applying DPI twice."""
    screen = dialog.screen()
    if screen is None and dialog.parentWidget() is not None:
        screen = dialog.parentWidget().screen()
    if screen is None:
        screen = QApplication.screenAt(dialog.frameGeometry().center())
    if screen is None:
        screen = QApplication.primaryScreen()
    if screen is None:
        return dialog.geometry()
    target = fitted_dialog_rect(screen.availableGeometry(), preferred)
    dialog.setGeometry(target)
    return target
