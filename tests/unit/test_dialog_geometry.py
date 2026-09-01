"""Logical-pixel dialog fitting tests for common Windows screen layouts."""

from dfn_cave_studio.ui.dialog_geometry import fitted_dialog_rect
from dfn_cave_studio.ui.qt_adapter import QRect, QSize


def test_fits_1366_by_768_available_geometry() -> None:
    available = QRect(0, 0, 1366, 728)
    target = fitted_dialog_rect(available, QSize(1050, 900))
    assert available.contains(target)
    assert target.height() <= int(available.height() * 0.92)


def test_fits_1920_by_1080_available_geometry() -> None:
    available = QRect(0, 0, 1920, 1040)
    target = fitted_dialog_rect(available, QSize(1050, 900))
    assert available.contains(target)
    assert target.size() == QSize(1050, 900)


def test_centres_inside_nonzero_and_negative_monitor_coordinates() -> None:
    for available in (QRect(1920, 40, 1600, 860), QRect(-1600, -120, 1600, 860)):
        target = fitted_dialog_rect(available, QSize(1200, 900))
        assert available.contains(target)
        assert target.center().x() == available.center().x()
        assert abs(target.center().y() - available.center().y()) <= 1
