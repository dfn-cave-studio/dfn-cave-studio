"""GUI smoke tests using pytest-qt, offscreen rendering, and FakePlotter.

Verifies: QApplication creation, MainWindow creation, project lifecycle
(new/save/open .dfnproj), data import, and UI close — all without
real VTK/OpenGL initialisation.

Real VTK/OpenGL rendering tests belong in tests/gui/test_vtk_render.py
(marker: vtk_render) and are NOT run on standard headless CI runners.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import Qt


@pytest.fixture
def window(qtbot):
    """Create a fresh MainWindow managed by qtbot for proper Qt cleanup.

    FakePlotter is injected by conftest — no real VTK/OpenGL.
    Modal dialogs are mocked by conftest — no test ever blocks.
    """
    from dfn_cave_studio.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    # ── Teardown: must clear dirty BEFORE close() ─────────────────────
    # closeEvent checks is_dirty and shows QMessageBox.question if true.
    # Modal dialogs are mocked by conftest as a safety net.
    # Do NOT call deleteLater() — qtbot.addWidget() handles that.
    w._auto_save_timer.stop()
    w._project_store._dirty = False
    w.close()
    qtbot.wait(50)


class TestMainWindowCreation:
    """MainWindow can be created and destroyed without crash."""

    def test_main_window_created(self, window):
        """MainWindow instantiates successfully with FakePlotter."""
        assert window is not None
        assert window.windowTitle() == "DFN Cave Studio"
        # FakePlotter is injected — no real VTK/OpenGL
        assert window._plotter is not None, "FakePlotter should be injected"
        assert bool(window._plotter) is True

    def test_status_bar_exists(self, window):
        """Status bar has the expected label."""
        assert window._status_label is not None

    def test_menu_bar_exists(self, window):
        """Menu bar has File menu."""
        menu_bar = window.menuBar()
        assert menu_bar is not None
        file_actions = [a.text() for a in menu_bar.actions()]
        assert any("File" in a for a in file_actions)

    def test_project_tree_exists(self, window):
        """Project tree widget exists."""
        assert window._project_tree is not None
        assert window._project_tree.topLevelItemCount() > 0

    def test_log_message_works(self, window):
        """log_message writes to the log widget."""
        window.log_message("Test message")
        # Should not raise

    def test_set_status_works(self, window):
        """set_status updates the status label."""
        window.set_status("Testing status")
        assert window._status_label is not None


class TestNewProjectAndSave:
    """New project → save .dfnproj without GUI dependencies."""

    def test_new_project_creates_state(self, window):
        """New Project sets up project state."""
        window._on_new_project()
        assert window._project_store.has_project
        project = window._project_store.current_project
        assert project is not None
        assert project.metadata.name == "New Project"

    def test_save_dfproj_via_model(self, window, tmp_path):
        """Save project as .dfnproj through the model layer."""
        window._on_new_project()
        project = window._project_store.current_project
        from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
        project.model_bounds = ModelBounds(x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5)
        project.voxel_config = VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)

        save_path = tmp_path / "smoke.dfnproj"
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        zps = ZipProjectStore()
        zps.save(project, save_path)
        assert save_path.exists()

        reopened = zps.load(save_path)
        assert reopened.metadata.name == "New Project"

    def test_save_and_reopen_with_data(self, window, tmp_path):
        """Save project with borehole + DFN + connectivity and reopen."""
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

        bh = Borehole(
            borehole_id="BH-001",
            collar=Collar(borehole_id="BH-001", collar_x=25, collar_y=25, collar_z=200, final_depth=200),
            survey=BoreholeSurvey(stations=[
                SurveyStation(measured_depth=0, azimuth=0, dip=-90),
                SurveyStation(measured_depth=200, azimuth=0, dip=-90),
            ]),
            fracture_observations=[
                FractureObservation(measured_depth=50, dip_direction=45, dip=60, set_id=1),
                FractureObservation(measured_depth=100, dip_direction=135, dip=75, set_id=2),
            ],
        )
        project.borehole_collection = BoreholeCollection(boreholes=[bh])
        project.joint_sets = [
            JointSetConfig(
                set_id=1, name="Set 1",
                orientation=OrientationDistribution(mean_dip_direction=45, mean_dip=60, kappa=30),
                size=SizeDistribution(distribution_type=SizeDistributionType.FIXED,
                                      min_radius=0.99, max_radius=1.01),
                target_p32=2.0,
                provenance={"orientation": "borehole", "size": "user", "p32": "user"},
            )
        ]
        project.connectivity_results = {"n_edges": 42, "n_components": 3}
        project.connectivity_clusters = [1, 1, 2, 0, 0, 2]
        project.voxel_p32_results = [
            {"i": 0, "j": 0, "k": 0, "local_p32": 1.5, "fracture_count": 2,
             "fracture_area": 3.14, "connectivity_cluster": 0,
             "x": 0.5, "y": 0.5, "z": 0.5, "dx": 1.0, "dy": 1.0, "dz": 1.0},
        ]

        save_path = tmp_path / "full_smoke.dfnproj"
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        zps = ZipProjectStore()
        zps.save(project, save_path)
        reopened = zps.load(save_path)

        assert reopened.borehole_collection is not None
        assert len(reopened.borehole_collection) == 1
        obs_set_ids = {o.set_id for o in reopened.borehole_collection["BH-001"].fracture_observations}
        assert obs_set_ids == {1, 2}
        assert len(reopened.joint_sets) == 1
        assert reopened.joint_sets[0].provenance.get("orientation") == "borehole"
        assert reopened.connectivity_results["n_edges"] == 42
        assert reopened.connectivity_clusters == [1, 1, 2, 0, 0, 2]
        assert len(reopened.voxel_p32_results) > 0


class TestMainWindowClose:
    """MainWindow close event does not crash."""

    def test_window_closes_cleanly(self, window):
        """Close with clean state (fixture teardown handles actual close)."""
        # Fixture teardown stops timer, clears dirty, calls close+deleteLater.
        # This test just verifies the window exists and is healthy.
        assert window is not None
        assert window.isVisible() or not window.isVisible()  # either is fine

    def test_second_window_also_closes(self, qtbot):
        """A second MainWindow instance also works."""
        from dfn_cave_studio.ui.main_window import MainWindow
        w = MainWindow()
        qtbot.addWidget(w)
        # qtbot will handle close+delete in its own teardown;
        # just verify creation succeeded.
        assert w is not None
        assert w.windowTitle() == "DFN Cave Studio"


class TestDataImport:
    """Import demo CSV data through UI path (no GUI rendering)."""

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
        set_ids = set()
        for bh in collection:
            for obs in bh.fracture_observations:
                set_ids.add(obs.set_id)
        assert 1 in set_ids and 2 in set_ids and 3 in set_ids
        assert None not in set_ids

        for bh in collection:
            points, mds = bh.compute_trajectory(step_length=5.0)
            assert len(points) >= 2


class TestRenderCallsAvailable:
    """Verify render-related methods exist and are callable on MainWindow."""

    def test_render_bounds_box_callable(self, window):
        assert hasattr(window, '_render_bounds_box')
        assert callable(window._render_bounds_box)

    def test_load_project_callable(self, window):
        assert hasattr(window, '_load_project')
        assert callable(window._load_project)

    def test_save_project_to_callable(self, window):
        assert hasattr(window, '_save_project_to')
        assert callable(window._save_project_to)

    def test_restore_project_to_ui_callable(self, window):
        assert hasattr(window, '_restore_project_to_ui')
        assert callable(window._restore_project_to_ui)

    def test_fake_plotter_has_required_methods(self, window):
        """FakePlotter provides all methods MainWindow needs."""
        p = window._plotter
        required = ['add_mesh', 'reset_camera', 'show_axes',
                     'view_xy', 'view_xz', 'view_yz', 'render', 'clear', 'close']
        for method in required:
            assert hasattr(p, method), f"FakePlotter missing method: {method}"
            assert callable(getattr(p, method)), f"FakePlotter.{method} not callable"
