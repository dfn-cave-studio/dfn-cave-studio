"""
Background worker for DFN generation.

Runs DFN generation in a separate thread via QRunnable + QThreadPool.
Communicates progress and completion via Qt signals.

References:
  - AGENTS.md: "All long-running tasks MUST run in background threads."
"""

from __future__ import annotations

from typing import Optional, List, Callable

from dfn_cave_studio.ui.qt_adapter import (
    QObject, Signal, Slot, QRunnable,
)

from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.dfn_realization import (
    DFNGenerationConfig, DFNRealization,
)
from dfn_cave_studio.dfn.generator import DFNGenerator


# =============================================================================
# Signal Emitter for Workers
# =============================================================================

class DFNWorkerSignals(QObject):
    """Signals emitted by DFN generation workers.

    These are cross-thread signals — connected slots run in the main thread.
    """

    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(object)          # DFNRealization
    error = Signal(str)                # Error message
    cancelled = Signal()               # Generation was cancelled by user


# =============================================================================
# DFN Generation Worker
# =============================================================================

class DFNGenerationWorker(QRunnable):
    """Background worker that generates a DFN realization.

    Usage:
        worker = DFNGenerationWorker(joint_sets, bounds, seed)
        worker.signals.progress.connect(on_progress)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(
        self,
        joint_sets: List[JointSetConfig],
        bounds: ModelBounds,
        master_seed: int = 42,
        realization_number: int = 0,
        max_fractures_per_set: int = 100000,
        boundary_buffer: float = 1.0,
    ):
        super().__init__()
        self._joint_sets = joint_sets
        self._bounds = bounds
        self._master_seed = master_seed
        self._realization_number = realization_number
        self._max_fractures = max_fractures_per_set
        self._buffer = boundary_buffer

        self.signals = DFNWorkerSignals()
        self._generator: Optional[DFNGenerator] = None
        self._cancelled = False

    def run(self) -> None:
        """Execute DFN generation (runs in thread pool)."""
        try:
            config = DFNGenerationConfig(
                master_seed=self._master_seed,
                joint_sets=self._joint_sets,
                model_volume=self._bounds.volume,
                max_fractures_per_set=self._max_fractures,
                boundary_buffer=self._buffer,
            )

            self._generator = DFNGenerator(config, self._bounds)

            # Connect progress callback
            def on_progress(current: int, total: int, message: str) -> None:
                if self._cancelled:
                    self._generator.cancel()
                self.signals.progress.emit(current, total, message)

            self._generator.set_progress_callback(on_progress)

            # Check for pre-cancellation
            if self._cancelled:
                self.signals.cancelled.emit()
                return

            # Generate
            realization = self._generator.generate(self._realization_number)

            if self._cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(realization)

        except InterruptedError:
            self.signals.cancelled.emit()
        except Exception as e:
            self.signals.error.emit(str(e))

    def cancel(self) -> None:
        """Request cancellation of the generation."""
        self._cancelled = True
        if self._generator:
            self._generator.cancel()
