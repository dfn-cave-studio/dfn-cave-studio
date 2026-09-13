"""Background worker for transactional Phase 2A borehole generation."""

from __future__ import annotations

from dfn_cave_studio.models.borehole_fracture_realization import BoreholeFractureGenerationConfig
from dfn_cave_studio.services.borehole_fracture_service import (
    BoreholeFractureService,
    BoreholeGenerationCancelled,
)
from dfn_cave_studio.ui.qt_adapter import QObject, QRunnable, Signal


class BoreholeFractureWorkerSignals(QObject):
    """Signals emitted by one Phase 2A calculation."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)


class BoreholeFractureWorker(QRunnable):
    """Build a complete candidate state without mutating the project."""

    def __init__(self, project, config: BoreholeFractureGenerationConfig, fit_override=None):
        super().__init__()
        self.project = project
        self.config = config
        self.fit_override = fit_override
        self.signals = BoreholeFractureWorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        self._cancelled = True

    def run(self) -> None:
        """Build and emit a complete candidate, or a terminal error/cancel event."""
        try:
            candidate = BoreholeFractureService(self.project).build_candidate(
                self.config,
                fit_override=self.fit_override,
                cancelled=lambda: self._cancelled,
                progress=self.signals.progress.emit,
            )
            if self._cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(candidate)
        except BoreholeGenerationCancelled:
            self.signals.cancelled.emit()
        except Exception as exc:
            self.signals.failed.emit(str(exc))
