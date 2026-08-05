"""GUI smoke tests using pytest-qt and offscreen rendering.

Verifies: QApplication creation, MainWindow creation, project lifecycle
(new/open/save .dfnproj), data import, computation, and UI close.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

# Force offscreen before any Qt imports
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYTHONPATH"] = os.pathsep.join([
    os.environ.get("PYTHONPATH", ""),
    str(Path(__file__).parent.parent.parent / "src"),
])

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt, QTimer


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    # Don't quit — other tests may need it


class TestMainWindowCreation:
    """MainWindow can be created and destroyed without crash."""

    @pytest.fixture
    def window(self, qapp):
        from dfn_cave_studio.ui.main_window import MainWindow
        w = MainWindow()
        yield w
        w.close()

    def test_main_window_created(self, window):
        """MainWindow instantiates successfully."""
        assert window is not None
        assert window.windowTitle() == "DFN Cave Studio"

    def test_status_bar_exists(self, window):
        """Status bar has the expected label."""
        assert window._status_label is not None

    def test_menu_bar_exists(self, window):
        """Menu bar has File menu."""
        menu_bar = window.menuBar()
        assert menu_bar is not None
        # Find File menu action
        file_actions = [a.text() for a in menu_bar.actions()]
        assert any("File" in a for a in file_actions)

    def test_project_tree_exists(self, window):
        """Project tree widget exists."""
        assert window._project_tree is not None
        assert window._project_tree.topLevelItemCount() > 0


class TestProjectLifecycle:
    """New → save .dfnproj → open → verify."""

    @pytest.fixture
    def window(self, qapp):
        from dfn_cave_studio.ui.main_window import MainWindow
        w = MainWindow()
        yield w
        w.close()

    def test_new_project_creates_state(self, window):
        """New Project sets up project state."""
        window._on_new_project()
        assert window._project_store.has_project
        project = window._project_store.current_project
        assert project is not None
        assert project.metadata.name == "New Project"

    def test_project_has_bounds(self, window):
        """New project has default model bounds."""
        window._on_new_project()
        project = window._project_store.current_project
        assert project.model_bounds is not None
        assert project.model_bounds.volume > 0

    def test_save_and_reopen_dfproj(self, window, tmp_path, qapp):
        """Save project as .dfnproj and reopen."""
        window._on_new_project()
        project = window._project_store.current_project

        # Set up minimal data
        from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
        project.model_bounds = ModelBounds(x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5)
        project.voxel_config = VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)

        # Save
        save_path = tmp_path / "test_project.dfnproj"
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        zps = ZipProjectStore()
        zps.save(project, save_path)
        assert save_path.exists()

        # Reopen
        from dfn_cave_studio.models.project import Project
        reopened = zps.load(save_path)
        assert reopened is not None
        assert reopened.metadata.name == "New Project"
        assert reopened.model_bounds.x_max == 5.0

    def test_save_and_reopen_with_data(self, window, tmp_path, qapp):
        """Save project with borehole + DFN data and reopen."""
        window._on_new_project()
        project = window._project_store.current_project

        from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
        from dfn_cave_studio.models.borehole import (
            BoreholeCollection, Borehole, Collar, BoreholeSurvey,
            SurveyStation, FractureObservation,
        )
        from dfn_cave_studio.models.fracture_set import (
            JointSetConfig, OrientationDistribution, SizeDistribution,
        )
        from dfn_cave_studio.models.enums import SizeDistributionType

        project.model_bounds = ModelBounds(x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5)
        project.voxel_config = VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)
        project.config.master_seed = 42

        # Add borehole with observations
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
                FractureObservation(measured_depth=150, dip_direction=270, dip=30, set_id=3),
            ],
        )
        project.borehole_collection = BoreholeCollection(boreholes=[bh])

        # Add joint sets with provenance
        js = JointSetConfig(
            set_id=1, name="Set 1",
            orientation=OrientationDistribution(mean_dip_direction=45, mean_dip=60, kappa=30),
            size=SizeDistribution(distribution_type=SizeDistributionType.FIXED,
                                  min_radius=0.99, max_radius=1.01),
            target_p32=2.0,
            provenance={"orientation": "borehole", "size": "user", "p32": "user"},
        )
        project.joint_sets = [js]

        # Add connectivity data
        project.connectivity_results = {"n_edges": 42, "n_components": 3}
        project.connectivity_clusters = [1, 1, 2, 0, 0, 2]

        # Add voxel P32 results
        project.voxel_p32_results = [
            {"i": 0, "j": 0, "k": 0, "local_p32": 1.5, "fracture_count": 2,
             "fracture_area": 3.14, "connectivity_cluster": 0,
             "x": 0.5, "y": 0.5, "z": 0.5,
             "dx": 1.0, "dy": 1.0, "dz": 1.0},
        ]

        # Save
        save_path = tmp_path / "smoke_test.dfnproj"
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        zps = ZipProjectStore()
        zps.save(project, save_path)

        # Reopen
        reopened = zps.load(save_path)
        assert reopened.borehole_collection is not None
        assert len(reopened.borehole_collection) == 1
        reopened_bh = reopened.borehole_collection["BH-001"]
        assert len(reopened_bh.fracture_observations) == 3
        # set_ids preserved
        obs_set_ids = {o.set_id for o in reopened_bh.fracture_observations}
        assert obs_set_ids == {1, 2, 3}

        # Joint sets preserved
        assert len(reopened.joint_sets) == 1
        assert reopened.joint_sets[0].provenance.get("orientation") == "borehole"

        # Connectivity preserved
        assert reopened.connectivity_results is not None
        assert reopened.connectivity_results["n_edges"] == 42
        assert reopened.connectivity_clusters is not None
        assert len(reopened.connectivity_clusters) == 6

        # Voxel P32 preserved
        assert reopened.voxel_p32_results is not None
        assert len(reopened.voxel_p32_results) > 0
        assert any(isinstance(v, dict) and v.get("local_p32", 0) > 0
                   for v in reopened.voxel_p32_results)


class TestWindowClose:
    """MainWindow close event does not crash."""

    def test_window_closes_cleanly(self, qapp):
        from dfn_cave_studio.ui.main_window import MainWindow
        w = MainWindow()
        w._project_store._dirty = False  # Avoid save prompt
        # Close should not raise
        try:
            w.close()
            assert True
        except Exception as e:
            pytest.fail(f"MainWindow.close() raised: {e}")


class TestDataImport:
    """Import demo CSV data through UI path."""

    @pytest.fixture
    def demo_dir(self):
        return Path(__file__).parent.parent.parent / "examples" / "end_to_end_demo"

    def test_import_demo_data(self, qapp, demo_dir):
        """Import all three CSV files and verify state."""
        from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter
        importer = BoreholeImporter()
        result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            survey_path=str(demo_dir / "surveys.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        assert result.success, f"Import errors: {result.errors}"
        collection = result.collection
        assert len(collection) == 5

        # Verify set_ids from CSV
        set_ids = set()
        for bh in collection:
            for obs in bh.fracture_observations:
                set_ids.add(obs.set_id)
        assert 1 in set_ids and 2 in set_ids and 3 in set_ids
        assert None not in set_ids

        # Compute trajectories
        for bh in collection:
            points, mds = bh.compute_trajectory(step_length=5.0)
            assert len(points) >= 2


class TestRenderCalls:
    """Verify render functions are callable from MainWindow dependencies."""

    @pytest.fixture
    def window(self, qapp):
        from dfn_cave_studio.ui.main_window import MainWindow
        w = MainWindow()
        yield w
        w.close()

    def test_render_bounds_box_available(self, window):
        """_render_bounds_box is a callable method."""
        assert hasattr(window, '_render_bounds_box')
        assert callable(window._render_bounds_box)

    def test_load_project_method_exists(self, window):
        """_load_project method exists."""
        assert hasattr(window, '_load_project')
        assert callable(window._load_project)

    def test_save_project_to_exists(self, window):
        """_save_project_to method exists."""
        assert hasattr(window, '_save_project_to')
        assert callable(window._save_project_to)

    def test_restore_project_to_ui_exists(self, window):
        """_restore_project_to_ui method exists."""
        assert hasattr(window, '_restore_project_to_ui')
        assert callable(window._restore_project_to_ui)
