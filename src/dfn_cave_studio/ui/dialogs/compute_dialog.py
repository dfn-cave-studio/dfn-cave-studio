"""Computation pipeline dialog — DFN → Voxel → Connectivity with progress."""

from typing import Dict, Any

from dfn_cave_studio.ui.qt_adapter import (
    Qt, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit,
    QGroupBox, QThreadPool,
)
from dfn_cave_studio.workers.pipeline_worker import PipelineWorker, PipelineWorkerSignals


class ComputePipelineDialog(QDialog):
    """Dialog showing progress of the full computation pipeline.

    Runs DFN Generation → Voxel Intersection → Connectivity
    sequentially in a background thread with progress bars.
    """

    def __init__(self, worker: PipelineWorker, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate & Compute")
        self.resize(550, 450)
        self.setModal(True)
        self._worker = worker
        self._results: Dict[str, Any] = {}
        self._running = False
        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Step label
        self._step_lbl = QLabel("Ready to start...")
        self._step_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self._step_lbl)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        layout.addWidget(self._log)

        # Results group
        self._results_group = QGroupBox("Results")
        self._results_group.setVisible(False)
        rl = QVBoxLayout(self._results_group)
        self._result_lbl = QLabel("")
        rl.addWidget(self._result_lbl)
        layout.addWidget(self._results_group)

        # Buttons
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Computation")
        self._start_btn.clicked.connect(self._start)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel)
        self._cancel_btn.setEnabled(False)
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        self._close_btn.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

    def _connect_signals(self) -> None:
        self._worker.signals.step_progress.connect(self._on_progress)
        self._worker.signals.step_finished.connect(self._on_step_done)
        self._worker.signals.all_finished.connect(self._on_all_done)
        self._worker.signals.error.connect(self._on_error)
        self._worker.signals.cancelled.connect(self._on_cancelled)

    def _start(self) -> None:
        self._running = True
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._log.clear()
        self._log_message("Starting computation pipeline...")
        QThreadPool.globalInstance().start(self._worker)

    def _cancel(self) -> None:
        self._worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._log_message("Cancelling...")

    def _log_message(self, msg: str) -> None:
        self._log.append(msg)

    def _on_progress(self, step: str, current: int, total: int) -> None:
        self._step_lbl.setText(f"Running: {step}")
        if total > 0:
            self._progress.setValue(int(current * 100 / total))
        else:
            self._progress.setValue(0)

    def _on_step_done(self, step: str, result: Any) -> None:
        if step == "DFN Generation":
            self._log_message(f"  ✓ DFN: {result.get('fractures', '?')} fractures, "
                            f"P32={result.get('p32', 0):.4f}")
        elif step == "Voxel Intersection":
            self._log_message(f"  ✓ Voxel: {result.get('pairs', '?')} fracture-voxel pairs, "
                            f"{result.get('active_voxels', '?')} active voxels")
        elif step == "Connectivity":
            self._log_message(f"  ✓ Connectivity: {result.get('n_edges', '?')} edges, "
                            f"{result.get('n_components', '?')} clusters, "
                            f"largest={result.get('largest_component_size', '?')}")

    def _on_all_done(self, results: Dict[str, Any]) -> None:
        self._running = False
        self._results = results
        self._progress.setValue(100)
        self._step_lbl.setText("Computation complete!")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

        # Show results
        stats = {}
        if "connectivity" in results:
            stats = results["connectivity"].statistics()
        dfn_info = ""
        if "dfn" in results:
            r = results["dfn"].generation_result
            dfn_info = f"  Fractures: {r.total_fractures}\n  P32: {r.achieved_p32:.4f} (target {r.target_p32:.2f})\n"
        self._result_lbl.setText(
            f"DFN Generation:\n{dfn_info}\n"
            f"Connectivity:\n"
            f"  Edges: {stats.get('n_edges', '?')}\n"
            f"  Components: {stats.get('n_components', '?')}\n"
            f"  Largest: {stats.get('largest_component_size', '?')} fractures\n"
            f"  Isolated: {stats.get('isolated_fractures', '?')}\n"
        )
        self._results_group.setVisible(True)

    def _on_error(self, step: str, message: str) -> None:
        self._running = False
        self._log_message(f"  ✗ ERROR [{step}]: {message}")
        self._step_lbl.setText(f"Error in {step}")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def _on_cancelled(self) -> None:
        self._running = False
        self._log_message("  ⚠ Computation cancelled by user")
        self._step_lbl.setText("Cancelled")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def get_results(self) -> Dict[str, Any]:
        return self._results
