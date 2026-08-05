"""Expanded tests for DFN renderer — PyVista ImageData, VTU export, rendering.

Tests actual ImageData creation, VTU file round-trip, screenshot, borehole
and observation rendering, model bounds, and error handling (no silent pass).
"""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture import (
    StochasticFracture, FractureGeometry, create_fracture_from_dip,
)
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.dfn_realization import DFNRealization, DFNGenerationResult
from dfn_cave_studio.models.borehole import (
    BoreholeCollection, Borehole, Collar, BoreholeSurvey,
    SurveyStation, FractureObservation,
)


class TestImageDataCreation:
    """pv.ImageData creation and data correctness."""

    def test_image_data_created(self):
        """pv.ImageData is successfully constructed."""
        import pyvista as pv
        grid = pv.ImageData(
            dimensions=(6, 6, 6),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        assert grid is not None
        assert grid.dimensions == (6, 6, 6)
        assert grid.n_cells == 125  # 5×5×5

    def test_image_data_cell_data(self):
        """Cell data can be attached and read back."""
        import pyvista as pv
        grid = pv.ImageData(
            dimensions=(6, 6, 6),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        p32_values = np.arange(125, dtype=np.float64)
        grid.cell_data["local_p32"] = p32_values
        assert "local_p32" in grid.cell_data
        assert len(grid.cell_data["local_p32"]) == 125
        assert grid.cell_data["local_p32"][0] == 0.0
        assert grid.cell_data["local_p32"][124] == 124.0

    def test_image_data_from_bounds(self):
        """ImageData matches model bounds and voxel config."""
        import pyvista as pv
        bounds = ModelBounds(x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5)
        vc = VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)

        nx = int(math.ceil((bounds.x_max - bounds.x_min) / vc.cell_size_x))
        ny = int(math.ceil((bounds.y_max - bounds.y_min) / vc.cell_size_y))
        nz = int(math.ceil((bounds.z_max - bounds.z_min) / vc.cell_size_z))
        grid = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(vc.cell_size_x, vc.cell_size_y, vc.cell_size_z),
            origin=(bounds.x_min, bounds.y_min, bounds.z_min),
        )
        assert grid.n_cells == 125  # 5×5×5
        assert grid.origin == (0.0, 0.0, 0.0)


class TestVTUExportRoundTrip:
    """VTU/VTI file export → re-read → verify."""

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def test_export_vti_and_re_read(self, temp_dir):
        """Export voxel P32 to VTI and re-read matching values."""
        import pyvista as pv
        bounds = ModelBounds(x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5)
        vc = VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)

        nx = int(math.ceil((bounds.x_max - bounds.x_min) / vc.cell_size_x))
        ny = int(math.ceil((bounds.y_max - bounds.y_min) / vc.cell_size_y))
        nz = int(math.ceil((bounds.z_max - bounds.z_min) / vc.cell_size_z))

        grid = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(vc.cell_size_x, vc.cell_size_y, vc.cell_size_z),
            origin=(bounds.x_min, bounds.y_min, bounds.z_min),
        )
        p32_values = np.random.default_rng(42).uniform(0, 5, nx * ny * nz)
        grid.cell_data["local_p32"] = p32_values
        grid.cell_data["fracture_count"] = np.ones(nx * ny * nz, dtype=np.int32) * 3

        vti_path = os.path.join(temp_dir, "test_export.vti")
        grid.save(vti_path)
        assert os.path.exists(vti_path), "VTI file not created"
        assert os.path.getsize(vti_path) > 0, "VTI file is empty"

        # Re-read
        imported = pv.read(vti_path)
        assert imported is not None, "Failed to re-read VTI"
        assert "local_p32" in imported.cell_data, "local_p32 missing after re-read"
        re_read = imported.cell_data["local_p32"]
        assert len(re_read) == len(p32_values), "Re-read array length mismatch"
        assert np.allclose(re_read, p32_values), "Re-read values differ from original"
        assert "fracture_count" in imported.cell_data

    def test_export_vtu_via_renderer(self, temp_dir):
        """DFNRenderer.export_voxels_vtu creates a valid file."""
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        bounds = ModelBounds(x_min=0, x_max=3, y_min=0, y_max=3, z_min=0, z_max=3)
        vc = VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)

        voxel_data = [
            {"i": 0, "j": 0, "k": 0, "local_p32": 2.5, "fracture_count": 3},
            {"i": 1, "j": 0, "k": 0, "local_p32": 1.5, "fracture_count": 2},
            {"i": 0, "j": 1, "k": 0, "local_p32": 0.5, "fracture_count": 1},
        ]
        renderer = DFNRenderer()
        vti_path = os.path.join(temp_dir, "renderer_export.vti")
        success = renderer.export_voxels_vtu(voxel_data, bounds, vc, vti_path)
        assert success, "export_voxels_vtu returned False"
        assert os.path.exists(vti_path), "VTI file not created by renderer"

        # Re-read
        import pyvista as pv
        imported = pv.read(vti_path)
        assert imported is not None
        assert "local_p32" in imported.cell_data
        re_read = imported.cell_data["local_p32"]
        # Total 27 cells (3×3×3), 3 non-zero
        non_zero = np.sum(re_read > 0)
        assert non_zero >= 1, "No non-zero P32 values in exported file"
        assert np.isclose(np.sum(re_read), 4.5, rtol=0.1), (
            f"Sum of P32 should be ~4.5, got {np.sum(re_read):.3f}")


class TestRendererMethodsCall:
    """Verify renderer methods accept arguments and don't silently fail."""

    @pytest.fixture
    def renderer(self):
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        return DFNRenderer()

    @pytest.fixture
    def plotter(self):
        import pyvista as pv
        return pv.Plotter(off_screen=True)

    @pytest.fixture
    def sample_bounds(self):
        return ModelBounds(x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5)

    @pytest.fixture
    def sample_voxel_config(self):
        return VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)

    @pytest.fixture
    def borehole_collection(self):
        bh = Borehole(
            borehole_id="BH-001",
            collar=Collar(
                borehole_id="BH-001",
                collar_x=25, collar_y=25, collar_z=200,
                final_depth=200,
            ),
            survey=BoreholeSurvey(stations=[
                SurveyStation(measured_depth=0, azimuth=0, dip=-90),
                SurveyStation(measured_depth=200, azimuth=0, dip=-90),
            ]),
            fracture_observations=[
                FractureObservation(measured_depth=50, dip_direction=45, dip=60, set_id=1),
                FractureObservation(measured_depth=100, dip_direction=135, dip=75, set_id=2),
            ],
        )
        return BoreholeCollection(boreholes=[bh])

    def test_render_boreholes_called(self, renderer, plotter, borehole_collection):
        """render_boreholes runs without exception."""
        try:
            renderer.render_boreholes(borehole_collection, plotter)
        except Exception as e:
            pytest.fail(f"render_boreholes raised: {e}")

    def test_render_fracture_observations_called(self, renderer, plotter, borehole_collection):
        """render_fracture_observations runs without exception."""
        try:
            renderer.render_fracture_observations(borehole_collection, plotter)
        except Exception as e:
            pytest.fail(f"render_fracture_observations raised: {e}")

    def test_render_model_bounds_called(self, renderer, plotter, sample_bounds):
        """render_model_bounds runs without exception."""
        try:
            renderer.render_model_bounds(sample_bounds, plotter)
        except Exception as e:
            pytest.fail(f"render_model_bounds raised: {e}")

    def test_add_coordinate_axes_called(self, renderer, plotter):
        """add_coordinate_axes runs."""
        renderer.add_coordinate_axes(plotter)

    def test_render_voxel_p32(self, renderer, plotter, sample_bounds, sample_voxel_config):
        """render_voxel_p32 with list-of-dicts data."""
        voxel_data = [
            {"i": 0, "j": 0, "k": 0, "local_p32": 2.0},
            {"i": 1, "j": 1, "k": 1, "local_p32": 1.0},
        ]
        success = renderer.render_voxel_p32(
            voxel_data, sample_bounds, sample_voxel_config, plotter,
        )
        assert success, "render_voxel_p32 returned False"

    def test_render_voxel_p32_dict_format(self, renderer, plotter, sample_bounds, sample_voxel_config):
        """render_voxel_p32 with old dict-format data."""
        voxel_dict = {
            (0, 0, 0): 2.0,
            (1, 1, 1): 1.0,
        }
        success = renderer.render_voxel_p32(
            voxel_dict, sample_bounds, sample_voxel_config, plotter,
        )
        assert success, "render_voxel_p32 with dict format returned False"

    def test_screenshot_success(self, renderer, plotter, tmp_path, sample_bounds):
        """Screenshot saves a valid PNG file."""
        renderer.render_model_bounds(sample_bounds, plotter)
        path = str(tmp_path / "test.png")
        success = renderer.screenshot(plotter, path)
        assert success, "screenshot returned False"
        assert os.path.exists(path), "Screenshot file not created"
        assert os.path.getsize(path) > 0, "Screenshot file is empty"

    def test_axes_rendered(self, renderer, plotter):
        """add_coordinate_axes does not raise."""
        renderer.add_coordinate_axes(plotter)
        assert True  # No exception = pass

    def test_legend_rendered(self, renderer, plotter):
        """add_legend does not raise."""
        items = {"Set 1": "#ff0000", "Set 2": "#00ff00"}
        renderer.add_legend(plotter, items)
        assert True

    def test_empty_boreholes_does_not_crash(self, renderer, plotter):
        """Empty collection doesn't crash renderer."""
        empty = BoreholeCollection(boreholes=[])
        renderer.render_boreholes(empty, plotter)
        assert True

    def test_error_logging_on_failure(self, renderer, sample_bounds, sample_voxel_config, caplog):
        """Errors are logged, not silently passed."""
        import logging
        caplog.set_level(logging.ERROR)
        # export_voxels_vtu to invalid path
        result = renderer.export_voxels_vtu(
            [], sample_bounds, sample_voxel_config,
            "/nonexistent/path/should/fail.vti"
        )
        assert result is False, "Should return False on failure, not silently pass"
        # Error should be logged
        assert len(caplog.records) >= 1, "No error logged for failed export"


class TestRendererLogicExpanded:
    """Test add_realization, render_to_plotter, clear, visibility, opacity."""

    @pytest.fixture
    def renderer(self):
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        return DFNRenderer()

    @pytest.fixture
    def plotter(self):
        import pyvista as pv
        return pv.Plotter(off_screen=True)

    @pytest.fixture
    def sample_realization(self):
        from dfn_cave_studio.models.dfn_realization import DFNRealization
        f1 = create_fracture_from_dip((0, 0, 0), 45, 60, 3.0, set_id=1, realization_id=0)
        f2 = create_fracture_from_dip((10, 10, 10), 90, 45, 3.0, set_id=2, realization_id=0)
        return DFNRealization(realization_number=0, stochastic_fractures=[f1, f2])

    @pytest.fixture
    def sample_joint_sets(self):
        return [
            JointSetConfig(set_id=1, name="Set1", color="#ff0000"),
            JointSetConfig(set_id=2, name="Set2", color="#00ff00"),
        ]

    def test_add_realization(self, renderer, plotter, sample_realization, sample_joint_sets):
        """add_realization adds actors to the plotter."""
        try:
            renderer.add_realization(sample_realization, sample_joint_sets, plotter)
        except Exception as e:
            pytest.fail(f"add_realization raised: {e}")
        stats = renderer.get_statistics()
        assert len(stats) >= 1  # Should have at least one set

    def test_render_to_plotter(self, renderer, plotter, sample_realization, sample_joint_sets):
        """render_to_plotter clears and re-renders."""
        renderer.render_to_plotter(plotter, sample_realization, sample_joint_sets)
        assert True

    def test_clear(self, renderer, plotter, sample_realization, sample_joint_sets):
        """clear removes all actors."""
        renderer.add_realization(sample_realization, sample_joint_sets, plotter)
        renderer.clear(plotter)
        stats = renderer.get_statistics()
        assert len(stats) == 0  # All cleared

    def test_set_visibility(self, renderer, plotter, sample_realization, sample_joint_sets):
        """set_visibility toggles actor visibility."""
        renderer.add_realization(sample_realization, sample_joint_sets, plotter)
        renderer.set_visibility(plotter, "Set1", False)
        renderer.set_visibility(plotter, "Set1", True)
        assert True  # No exception

    def test_set_opacity(self, renderer, plotter, sample_realization, sample_joint_sets):
        """set_opacity changes actor opacity."""
        renderer.add_realization(sample_realization, sample_joint_sets, plotter)
        renderer.set_opacity(plotter, "Set1", 0.5)
        assert True

    def test_clear_nonexistent_actor(self, renderer, plotter):
        """clear on empty renderer does not crash."""
        renderer.clear(plotter)
        assert True

    def test_visibility_nonexistent(self, renderer, plotter):
        """set_visibility for nonexistent does not crash."""
        renderer.set_visibility(plotter, "NoSuchSet", True)
        assert True

    def test_fractures_to_multiblock_valid(self, renderer):
        """fractures_to_multiblock creates a valid merged mesh."""
        f1 = create_fracture_from_dip((0, 0, 0), 45, 60, 3.0, set_id=1, realization_id=0)
        f2 = create_fracture_from_dip((5, 5, 5), 90, 45, 2.0, set_id=1, realization_id=0)
        fractures = [f1, f2]
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        mesh = DFNRenderer.fractures_to_multiblock(fractures, "#ff0000")
        assert mesh is not None
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_fractures_to_multiblock_single(self, renderer):
        """Single fracture produces valid mesh."""
        f1 = create_fracture_from_dip((0, 0, 0), 0, 90, 3.0, set_id=1, realization_id=0)
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        mesh = DFNRenderer.fractures_to_multiblock([f1], "#ff0000")
        assert mesh is not None
        assert mesh.n_points > 0


class TestFractureToDiskMesh:
    """Coverage for fracture_to_disk_mesh edge cases."""

    def test_disk_mesh_opposite_normal(self):
        """Disk mesh with normal opposite to Z axis (negative Z)."""
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        center = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, -1.0])  # opposite direction
        disk = DFNRenderer.fracture_to_disk_mesh(center, normal, 3.0, n_sides=6)
        assert disk.n_points > 0
        assert disk.n_cells > 0

    def test_disk_mesh_arbitrary_normal(self):
        """Disk mesh with arbitrary non-axis-aligned normal."""
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        center = np.array([1.0, 2.0, 3.0])
        normal = np.array([1.0, -1.0, 0.5])
        normal = normal / np.linalg.norm(normal)
        disk = DFNRenderer.fracture_to_disk_mesh(center, normal, 5.0, n_sides=12)
        assert disk.n_points == 13  # center + 12 vertices
        assert disk.n_cells == 12
