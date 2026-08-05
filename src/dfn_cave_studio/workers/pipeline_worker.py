"""
Pipeline worker — runs DFN generation → voxel intersection → connectivity
as a sequential background pipeline via QRunnable + QThreadPool.

Each step reports progress independently. The pipeline can be cancelled
at any point, and errors propagate to the UI.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from dfn_cave_studio.ui.qt_adapter import QObject, Signal, QRunnable

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.dfn_realization import DFNGenerationConfig, DFNRealization
from dfn_cave_studio.dfn.generator import DFNGenerator
from dfn_cave_studio.voxel.voxel_grid import VoxelGrid
from dfn_cave_studio.voxel.dfn_voxel_intersection import DFNVoxelIntersectionEngine
from dfn_cave_studio.connectivity.connectivity_graph import ConnectivityGraph


class PipelineWorkerSignals(QObject):
    """Signals for the multi-step pipeline worker."""

    step_progress = Signal(str, int, int)   # step_name, current, total
    step_finished = Signal(str, object)     # step_name, result
    all_finished = Signal(dict)             # {"dfn": realization, "voxel": grid, "connectivity": graph}
    error = Signal(str, str)                # step_name, error_message
    cancelled = Signal()


class PipelineWorker(QRunnable):
    """Background worker that executes the full DFN computation pipeline.

    Steps:
      1. DFN Generation — DFNGenerator in thread pool
      2. Voxel Intersection — DFNVoxelIntersectionEngine
      3. Connectivity — ConnectivityGraph

    Usage:
        worker = PipelineWorker(joint_sets, bounds, voxel_config, seed)
        worker.signals.step_progress.connect(on_progress)
        worker.signals.all_finished.connect(on_done)
        worker.signals.error.connect(on_error)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(
        self,
        joint_sets: List[JointSetConfig],
        bounds: ModelBounds,
        voxel_config: VoxelConfig,
        master_seed: int = 42,
        realization_number: int = 0,
        max_fractures_per_set: int = 100000,
    ):
        super().__init__()
        self._joint_sets = joint_sets
        self._bounds = bounds
        self._voxel_config = voxel_config
        self._master_seed = master_seed
        self._realization_number = realization_number
        self._max_fractures = max_fractures_per_set

        self.signals = PipelineWorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the pipeline."""
        self._cancelled = True
        if hasattr(self, '_generator') and self._generator:
            self._generator.cancel()
        if hasattr(self, '_engine') and self._engine:
            self._engine.cancel()

    def run(self) -> None:
        """Execute the pipeline (runs in thread pool)."""
        results = {}

        try:
            # ── Step 1: DFN Generation ──────────────────────────────
            if self._cancelled:
                self.signals.cancelled.emit()
                return

            self.signals.step_progress.emit("DFN Generation", 0, 100)
            config = DFNGenerationConfig(
                master_seed=self._master_seed,
                joint_sets=self._joint_sets,
                model_volume=self._bounds.volume,
                max_fractures_per_set=self._max_fractures,
                boundary_buffer=1.0,
            )
            self._generator = DFNGenerator(config, self._bounds)

            def on_dfn_progress(current: int, total: int, message: str) -> None:
                if self._cancelled:
                    self._generator.cancel()
                self.signals.step_progress.emit("DFN Generation", current, total)

            self._generator.set_progress_callback(on_dfn_progress)
            realization = self._generator.generate(self._realization_number)

            if self._cancelled:
                self.signals.cancelled.emit()
                return

            results["dfn"] = realization
            self.signals.step_finished.emit(
                "DFN Generation",
                {"fractures": realization.generation_result.total_fractures,
                 "p32": realization.generation_result.achieved_p32}
            )

            # ── Step 2: Voxel Intersection ──────────────────────────
            if self._cancelled:
                self.signals.cancelled.emit()
                return

            self.signals.step_progress.emit("Voxel Intersection", 0, 100)
            grid = VoxelGrid.from_bounds(self._bounds, self._voxel_config)

            # Apply simple box mask (full volume)
            from dfn_cave_studio.models.rock_mask import RockMask
            from dfn_cave_studio.models.enums import MaskType
            mask = RockMask(
                mask_type=MaskType.BOX,
                x_min=self._bounds.x_min, x_max=self._bounds.x_max,
                y_min=self._bounds.y_min, y_max=self._bounds.y_max,
                z_min=self._bounds.z_min, z_max=self._bounds.z_max,
            )
            grid.apply_mask(mask)

            self._engine = DFNVoxelIntersectionEngine(grid, realization)

            def on_voxel_progress(current: int, total: int, message: str) -> None:
                if self._cancelled:
                    self._engine.cancel()
                self.signals.step_progress.emit("Voxel Intersection", current, total)

            self._engine.set_progress_callback(on_voxel_progress)
            n_pairs = self._engine.compute_intersections()

            if self._cancelled:
                self.signals.cancelled.emit()
                return

            results["voxel"] = grid
            self.signals.step_finished.emit(
                "Voxel Intersection",
                {"pairs": n_pairs, "active_voxels": grid.active_voxel_count}
            )

            # ── Step 3: Connectivity ────────────────────────────────
            if self._cancelled:
                self.signals.cancelled.emit()
                return

            self.signals.step_progress.emit("Connectivity", 0, 100)
            graph = ConnectivityGraph(realization)

            def on_conn_progress(current: int, total: int, message: str) -> None:
                self.signals.step_progress.emit("Connectivity", current, total)

            graph.compute_edges(progress_callback=on_conn_progress)
            graph.find_components()

            if self._cancelled:
                self.signals.cancelled.emit()
                return

            results["connectivity"] = graph
            stats = graph.statistics()
            self.signals.step_finished.emit("Connectivity", stats)

            # ── Done ─────────────────────────────────────────────────
            self.signals.all_finished.emit(results)

        except InterruptedError:
            self.signals.cancelled.emit()
        except Exception as e:
            step = "Pipeline"
            self.signals.error.emit(step, str(e))
