"""Cancelable background worker for M11.1 second voxelization."""

from __future__ import annotations

import threading
from typing import Any

from dfn_cave_studio.services.m11_service import M11SecondVoxelizationService
from dfn_cave_studio.ui.qt_adapter import QObject, QRunnable, Signal
from dfn_cave_studio.voxel.second_voxelization import SecondVoxelizationConfig


class M11WorkerSignals(QObject):
    """Signals emitted by an M11.1 background calculation."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class M11SecondVoxelizationWorker(QRunnable):
    """Compute a complete M11.1 result without mutating the project in the worker."""

    def __init__(self, project: Any, realization_id: str, config: SecondVoxelizationConfig) -> None:
        super().__init__()
        self.project = project
        self.realization_id = realization_id
        self.config = config
        self.signals = M11WorkerSignals()
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation inside fracture and candidate-voxel loops."""
        self._cancelled.set()

    def run(self) -> None:
        """Run and emit exactly one terminal signal."""
        try:
            result = M11SecondVoxelizationService(self.project).compute(
                self.realization_id,
                config=self.config,
                progress=lambda current, total, message: self.signals.progress.emit(current, total, message),
                cancelled=self._cancelled.is_set,
                commit=False,
            )
            if self._cancelled.is_set():
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(result)
        except InterruptedError:
            self.signals.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - worker boundary must report failures to Qt
            self.signals.failed.emit(str(exc))
