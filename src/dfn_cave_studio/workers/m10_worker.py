"""Cancelable background worker for transactional M10 batch generation."""

from __future__ import annotations

from typing import Any

from dfn_cave_studio.models.m10 import DeterministicStructure, M10GenerationConfig
from dfn_cave_studio.services.m10_service import M10Service
from dfn_cave_studio.ui.qt_adapter import QObject, QRunnable, Signal


class M10WorkerSignals(QObject):
    """Signals emitted by M10 generation."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class M10GenerationWorker(QRunnable):
    """Run a complete M10 batch off the GUI thread with cooperative cancellation."""

    def __init__(
        self,
        project: Any,
        config: M10GenerationConfig,
        deterministic_structures: list[DeterministicStructure] | None = None,
    ) -> None:
        super().__init__()
        self.project = project
        self.config = config
        self.deterministic_structures = list(deterministic_structures or [])
        self.signals = M10WorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation; no partial realization will be committed."""
        self._cancelled = True

    def run(self) -> None:
        """Generate and surface completion, cancellation, or an explicit error."""
        try:
            results = M10Service(self.project).generate_batch(
                self.config,
                progress=lambda current, total, message: self.signals.progress.emit(current, total, message),
                cancelled=lambda: self._cancelled,
                commit=False,
                deterministic_structures=self.deterministic_structures,
            )
            if self._cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(results)
        except InterruptedError:
            self.signals.cancelled.emit()
        except Exception as exc:
            self.signals.failed.emit(str(exc))
