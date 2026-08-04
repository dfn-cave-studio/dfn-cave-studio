"""Tests for DFN renderer (non-rendering logic only)."""

import math
import numpy as np
import pytest

from dfn_cave_studio.models.fracture import (
    StochasticFracture, FractureGeometry, create_fracture_from_dip,
)
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.dfn_realization import DFNRealization, DFNGenerationResult


class TestDFNRendererLogic:
    """Test DFNRenderer logic without requiring a 3D display."""

    def test_default_colors(self):
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        r = DFNRenderer()
        assert len(r.DEFAULT_COLORS) == 10

    def test_build_color_map(self):
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        r = DFNRenderer()
        sets = [
            JointSetConfig(set_id=1, name="Set1", color="#ff0000"),
            JointSetConfig(set_id=2, name="Set2", color="#00ff00"),
        ]
        color_map = r._build_color_map(sets)
        assert color_map[1] == "#ff0000"
        assert color_map[2] == "#00ff00"

    def test_group_by_set(self):
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        r = DFNRenderer()

        f1 = create_fracture_from_dip((0, 0, 0), 45, 60, 3.0, set_id=1, realization_id=0)
        f2 = create_fracture_from_dip((10, 10, 10), 90, 45, 3.0, set_id=1, realization_id=0)
        f3 = create_fracture_from_dip((20, 20, 20), 135, 30, 3.0, set_id=2, realization_id=0)

        real = DFNRealization(
            realization_number=0,
            stochastic_fractures=[f1, f2, f3],
        )

        groups = r._group_by_set(real)
        assert len(groups[1]) == 2
        assert len(groups[2]) == 1
        assert 3 not in groups

    def test_fracture_to_disk_mesh_shape(self):
        """Disk mesh should have correct number of vertices and faces."""
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer

        center = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        disk = DFNRenderer.fracture_to_disk_mesh(center, normal, 5.0, n_sides=8)

        # Should have center + n_sides vertices = 9 vertices
        assert disk.n_points == 9
        # Should have n_sides triangular faces
        assert disk.n_cells == 8

    def test_fracture_to_disk_mesh_non_vertical(self):
        """Disk mesh for a dipping fracture."""
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer

        center = np.array([10.0, 20.0, 30.0])
        normal = np.array([0.5, 0.5, 0.7071])  # ~45° dip
        normal = normal / np.linalg.norm(normal)
        disk = DFNRenderer.fracture_to_disk_mesh(center, normal, 3.0, n_sides=8)

        assert disk.n_points > 0
        assert disk.n_cells > 0
        # Center should be approximately at input center
        assert np.linalg.norm(disk.points[0] - center) < 1e-10

    def test_empty_fractures_to_multiblock(self):
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        result = DFNRenderer.fractures_to_multiblock([], "#ff0000")
        assert result is None

    def test_get_statistics_empty(self):
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        r = DFNRenderer()
        stats = r.get_statistics()
        assert len(stats) == 0


class TestDFNWorker:
    """Test DFN generation worker logic."""

    def test_worker_signals_exist(self):
        from dfn_cave_studio.workers.dfn_worker import DFNGenerationWorker, DFNWorkerSignals
        signals = DFNWorkerSignals()
        assert hasattr(signals, 'progress')
        assert hasattr(signals, 'finished')
        assert hasattr(signals, 'error')
        assert hasattr(signals, 'cancelled')

    def test_worker_creation(self):
        from dfn_cave_studio.workers.dfn_worker import DFNGenerationWorker
        from dfn_cave_studio.models.bounds import ModelBounds
        from dfn_cave_studio.models.fracture_set import JointSetConfig

        bounds = ModelBounds(x_min=0, x_max=50, y_min=0, y_max=50, z_min=0, z_max=50)
        sets = [JointSetConfig(set_id=1, name="Test", target_p32=1.0)]

        worker = DFNGenerationWorker(
            joint_sets=sets,
            bounds=bounds,
            master_seed=42,
        )
        assert worker is not None
        assert worker._master_seed == 42

    def test_worker_cancel_before_run(self):
        from dfn_cave_studio.workers.dfn_worker import DFNGenerationWorker
        from dfn_cave_studio.models.bounds import ModelBounds
        from dfn_cave_studio.models.fracture_set import JointSetConfig

        bounds = ModelBounds(x_min=0, x_max=10, y_min=0, y_max=10, z_min=0, z_max=10)
        sets = [JointSetConfig(set_id=1, name="Test", target_p32=1.0)]

        worker = DFNGenerationWorker(joint_sets=sets, bounds=bounds, master_seed=42)
        worker.cancel()
        assert worker._cancelled
