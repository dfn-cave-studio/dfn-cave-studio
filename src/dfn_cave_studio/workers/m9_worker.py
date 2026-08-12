"""Cancelable background worker for M9 scientific operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dfn_cave_studio.ui.qt_adapter import QObject, QRunnable, Signal


class M9WorkerSignals(QObject):
    """Signals emitted by an M9 worker."""

    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class M9Worker(QRunnable):
    """Execute a supplied service operation outside the Qt GUI thread."""

    def __init__(self, operation: Callable[..., Any]) -> None:
        super().__init__()
        self.operation = operation
        self.signals = M9WorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        self._cancelled = True

    def run(self) -> None:
        """Run the operation and surface every exception to the UI."""
        try:
            result = self.operation(
                progress=lambda current, total: self.signals.progress.emit(current, total),
                cancelled=lambda: self._cancelled,
            )
            if self._cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(result)
        except InterruptedError:
            self.signals.cancelled.emit()
        except Exception as exc:
            self.signals.failed.emit(str(exc))
