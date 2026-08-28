"""
Main window for DFN Cave Studio.
"""

from pathlib import Path

from dfn_cave_studio.ui.qt_adapter import (
    Qt,
    QMainWindow,
    QWidget,
    QSplitter,
    QLabel,
    QStatusBar,
    QMenu,
    QAction,
    QKeySequence,
    QDialog,
    QMessageBox,
    QDockWidget,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QTabWidget,
    QProgressBar,
    QFileDialog,
    QSettings,
    QTimer,
    PyVistaQtInteractor,
    HAS_PYVISTAQT,
)

from dfn_cave_studio.core import get_config
from dfn_cave_studio.persistence.project_store import ProjectStore, RecentProjectsManager


class MainWindow(QMainWindow):
    """
    Main application window for DFN Cave Studio.

    Layout:
        Left: Project tree / data management dock
        Center: 3D visualization (PyVistaQt) or welcome page
        Right: Properties / parameters dock
        Bottom: Log output / progress dock
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = get_config()

        # Initialize project management
        self._project_store = ProjectStore()
        self._recent_manager = RecentProjectsManager()
        self._project_store.set_on_dirty_changed(self._on_project_dirty_changed)
        self._dfn_renderer = None  # Lazy-loaded (imports pyvista)
        self._m9_layer_manager = None
        self._dfn_layer_manager = None
        self._database_panel = None

        # M7 workflow controller
        from dfn_cave_studio.services.workflow_controller import WorkflowController

        self._workflow = WorkflowController()

        # Auto-save timer
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.timeout.connect(self._on_auto_save_tick)
        self._auto_save_timer.start(60000)  # Check every 60 seconds

        self._init_window()
        self._init_menu_bar()
        self._init_tool_bar()
        self._init_status_bar()
        self._init_central_widget()
        self._init_docks()
        self._init_connections()
        self._restore_state()
        self._log_startup_info()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_window(self) -> None:
        """Configure the main window."""
        self.setWindowTitle("DFN Cave Studio")
        self.resize(1400, 900)
        self.setMinimumSize(1024, 600)

        # Center on screen
        screen = self.screen()
        if screen:
            center = screen.availableGeometry().center()
            frame = self.frameGeometry()
            frame.moveCenter(center)
            self.move(frame.topLeft())

    def _init_menu_bar(self) -> None:
        """Create the main menu bar."""
        menu_bar = self.menuBar()

        # === File Menu ===
        self._file_menu = menu_bar.addMenu("&File")
        self._add_menu_action(
            self._file_menu, "&New Project", "Ctrl+N", self._on_new_project, "Create a new DFN Cave Studio project"
        )
        self._add_menu_action(
            self._file_menu, "&Open Project...", "Ctrl+O", self._on_open_project, "Open an existing project"
        )
        self._file_menu.addSeparator()
        self._add_menu_action(self._file_menu, "&Save Project", "Ctrl+S", self._on_save_project, "Save current project")
        self._add_menu_action(
            self._file_menu,
            "Save Project &As...",
            "Ctrl+Shift+S",
            self._on_save_project_as,
            "Save project to a new location",
        )
        self._file_menu.addSeparator()
        self._add_menu_action(self._file_menu, "&Import", None, None, "Import data")
        self._add_menu_action(self._file_menu, "&Export", None, None, "Export data")
        self._file_menu.addSeparator()
        self._recent_menu = self._file_menu.addMenu("&Recent Projects")
        self._file_menu.addSeparator()
        self._add_menu_action(self._file_menu, "E&xit", "Alt+F4", self.close, "Exit DFN Cave Studio")

        # === Data Menu ===
        self._data_menu = menu_bar.addMenu("&Data")
        self._add_menu_action(
            self._data_menu,
            "&Borehole Database...",
            None,
            self._on_borehole_manager,
            "View, import, and maintain the project borehole database",
        )

        # === Voxel Menu ===
        self._voxel_menu = menu_bar.addMenu("&Voxel")
        self._add_menu_action(
            self._voxel_menu, "Voxel &Settings...", None, self._on_voxel_settings, "Configure voxel grid"
        )

        # === DFN Menu ===
        self._dfn_menu = menu_bar.addMenu("D&FN")
        self._add_menu_action(
            self._dfn_menu, "&Joint Set Manager...", None, self._on_joint_set_manager, "Manage fracture sets"
        )
        self._add_menu_action(
            self._dfn_menu, "&Explicit DFN Generation...", None, self._on_generate_dfn, "Generate M10 conditional explicit DFN"
        )

        # === Domains Menu ===
        self._domains_menu = menu_bar.addMenu("D&omains")
        self._add_menu_action(
            self._domains_menu, "&Domain Manager...", None, self._on_domain_manager, "Manage structural domains"
        )

        # === Analysis Menu ===
        self._analysis_menu = menu_bar.addMenu("&Analysis")
        self._add_menu_action(
            self._analysis_menu, "&Connectivity...", None, self._on_connectivity, "Analyze fracture connectivity"
        )
        self._add_menu_action(
            self._analysis_menu, "&Fragmentation...", None, self._on_fragmentation, "Analyze block fragmentation"
        )

        # === Visualization Menu ===
        self._vis_menu = menu_bar.addMenu("&Visualization")
        self._add_menu_action(self._vis_menu, "&Reset View", "R", self._on_reset_view, "Reset 3D camera view")
        self._add_menu_action(self._vis_menu, "Top &View", "T", self._on_top_view, "Switch to top-down view")
        self._add_menu_action(self._vis_menu, "&Front View", "F", self._on_front_view, "Switch to front view")
        self._add_menu_action(self._vis_menu, "&Left View", "L", self._on_left_view, "Switch to left view")

        # === Export Menu ===
        self._export_menu = menu_bar.addMenu("E&xport")
        self._add_menu_action(self._export_menu, "Export &3DEC...", None, self._on_export_3dec, "Export for 3DEC")
        self._add_menu_action(self._export_menu, "Export &FLAC3D...", None, self._on_export_flac3d, "Export for FLAC3D")
        self._add_menu_action(self._export_menu, "Export &VTK...", None, self._on_export_vtk, "Export to VTK format")

        # === Tools Menu ===
        self._tools_menu = menu_bar.addMenu("&Tools")
        self._add_menu_action(self._tools_menu, "&Settings...", "Ctrl+,", self._on_settings, "Application settings")

        # === Help Menu ===
        self._help_menu = menu_bar.addMenu("&Help")
        self._add_menu_action(self._help_menu, "&About", None, self._on_about, "About DFN Cave Studio")
        self._add_menu_action(self._help_menu, "&Documentation", "F1", self._on_documentation, "Open documentation")

    def _init_tool_bar(self) -> None:
        """Create the main toolbar."""
        self._toolbar = QToolBar("Main Toolbar", self)
        self._toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        self._add_toolbar_action("New Project", self._on_new_project)
        self._add_toolbar_action("Open Project", self._on_open_project)
        self._add_toolbar_action("Save Project", self._on_save_project)
        self._toolbar.addSeparator()
        self._add_toolbar_action("Explicit DFN", self._on_generate_dfn)

    def _init_status_bar(self) -> None:
        """Create the status bar."""
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setMaximumHeight(16)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)

        self._coords_label = QLabel("X: --  Y: --  Z: --")
        self._status_bar.addPermanentWidget(self._coords_label)

    # ── Plotter factory (overridable for testing) ──────────────────────────

    @staticmethod
    def _create_plotter(parent: QWidget):
        """Create the 3D plotter widget.

        Override or monkeypatch this in tests to inject a FakePlotter
        that does not initialise real VTK/OpenGL.
        """
        if HAS_PYVISTAQT:
            return PyVistaQtInteractor(parent)
        return None

    def _init_central_widget(self) -> None:
        """Create the central widget with 3D view or welcome page."""
        self._central_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(self._central_splitter)

        # 3D view area
        self._plotter = self._create_plotter(self)
        if self._plotter is not None:
            self._central_splitter.addWidget(self._plotter)
        else:
            # Fallback: show welcome/label
            welcome = QLabel(
                "<h1>DFN Cave Studio</h1>"
                "<p>Discrete Fracture Network Modeling for Block Cave Mining</p>"
                "<p>Version 0.10.1-M10</p>"
                "<hr>"
                "<p>PyVistaQt not available. 3D visualization disabled.</p>"
                "<p>Create or open a project to begin.</p>"
            )
            welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._central_splitter.addWidget(welcome)

    def _init_docks(self) -> None:
        """Create dock widgets."""
        # Left dock: Project tree
        self._project_dock = QDockWidget("Project Explorer", self)
        self._project_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._project_tree = QTreeWidget()
        self._project_tree.setHeaderLabel("Project")
        self._project_dock.setWidget(self._project_tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._project_dock)
        self._init_project_tree()

        # Right dock: Properties
        self._properties_dock = QDockWidget("Properties", self)
        self._properties_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._properties_tabs = QTabWidget()
        self._properties_tabs.addTab(QLabel("Select an object to view properties"), "Properties")
        self._properties_dock.setWidget(self._properties_tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._properties_dock)

        # Bottom dock: Log output
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.document().setMaximumBlockCount(1000)
        self._log_dock.setWidget(self._log_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)

        # M7 workflow panel (left dock, below project explorer)
        from dfn_cave_studio.ui.panels.m7_workflow_panel import M7WorkflowPanel

        self._workflow_panel = M7WorkflowPanel(self._workflow, main_window=self, parent=self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._workflow_panel)
        self.tabifyDockWidget(self._project_dock, self._workflow_panel)
        self._project_dock.raise_()

    def _init_project_tree(self) -> None:
        """Initialize the project tree with default structure."""
        tree = self._project_tree
        tree.clear()

        self._tree_root = QTreeWidgetItem(tree, ["DFN Cave Studio Project"])
        self._tree_root.setExpanded(True)

        self._tree_model = QTreeWidgetItem(self._tree_root, ["Model"])
        QTreeWidgetItem(self._tree_model, ["Model Bounds"])
        QTreeWidgetItem(self._tree_model, ["Surface Model"])

        self._tree_data = QTreeWidgetItem(self._tree_root, ["Data"])
        QTreeWidgetItem(self._tree_data, ["Boreholes"])
        QTreeWidgetItem(self._tree_data, ["Deterministic Fractures"])

        self._tree_dfn = QTreeWidgetItem(self._tree_root, ["DFN"])
        QTreeWidgetItem(self._tree_dfn, ["Joint Sets"])
        QTreeWidgetItem(self._tree_dfn, ["Realizations"])

        self._tree_voxel = QTreeWidgetItem(self._tree_root, ["Voxel Grid"])
        QTreeWidgetItem(self._tree_voxel, ["Voxel Properties"])

        self._tree_domains = QTreeWidgetItem(self._tree_root, ["Structural Domains"])

        self._tree_analysis = QTreeWidgetItem(self._tree_root, ["Analysis"])
        QTreeWidgetItem(self._tree_analysis, ["Connectivity"])
        QTreeWidgetItem(self._tree_analysis, ["Fragmentation"])

        for i in range(self._tree_root.childCount()):
            self._tree_root.child(i).setExpanded(True)

    def _init_connections(self) -> None:
        """Set up signal/slot connections."""
        # Project tree selection
        self._project_tree.itemClicked.connect(self._on_tree_item_clicked)

    def _restore_state(self) -> None:
        """Restore window state from settings."""
        settings = QSettings("DFNCaveStudio", "MainWindow")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)

    def _log_startup_info(self) -> None:
        """Log startup information."""
        self.log_message("DFN Cave Studio v0.10.1-M10 started")
        self.log_message(
            f"Python: {__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"
        )
        if HAS_PYVISTAQT:
            self.log_message("3D Visualization: Available (PyVistaQt)")
        else:
            self.log_message("3D Visualization: Not available (PyVistaQt not found)")

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _add_menu_action(
        self,
        menu: QMenu,
        text: str,
        shortcut: str | None,
        callback,
        tooltip: str = "",
    ) -> QAction:
        """Add an action to a menu."""
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if callback:
            action.triggered.connect(callback)
        if tooltip:
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
        menu.addAction(action)
        return action

    def _add_toolbar_action(self, text: str, callback) -> QAction:
        """Add an action to the toolbar."""
        action = QAction(text, self)
        action.triggered.connect(callback)
        self._toolbar.addAction(action)
        return action

    def log_message(self, message: str) -> None:
        """Log a message to the log dock."""
        self._log_widget.append(f"[INFO] {message}")

    def log_error(self, message: str) -> None:
        """Log an error to the log dock."""
        self._log_widget.append(f"[ERROR] {message}")

    def log_warning(self, message: str) -> None:
        """Log a warning to the log dock."""
        self._log_widget.append(f"[WARN] {message}")

    def set_status(self, message: str) -> None:
        """Set the status bar message."""
        if self._status_label:
            self._status_label.setText(message)

    def show_progress(self, value: int, maximum: int = 100) -> None:
        """Show/hide the progress bar."""
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(maximum)
        self._progress_bar.setValue(value)

    def hide_progress(self) -> None:
        """Hide the progress bar."""
        self._progress_bar.setVisible(False)

    # ------------------------------------------------------------------
    # Menu Callbacks
    # ------------------------------------------------------------------

    def _on_new_project(self) -> None:
        """Create a new project — clears all previous state."""
        self.log_message("Creating new project...")
        self._clear_renderer_state()
        # Clear project tree
        self._init_project_tree()
        # Reset workflow
        from dfn_cave_studio.services.workflow_controller import WorkflowController

        self._workflow = WorkflowController()
        self._workflow_panel._controller = self._workflow
        self._workflow.add_change_listener(self._workflow_panel._refresh)
        self._workflow_panel._refresh()
        try:
            project = self._project_store.new_project("New Project")
            # Bind workflow to project immediately
            from dfn_cave_studio.services.m7_state import set_workflow

            set_workflow(project, self._workflow)
            self.setWindowTitle("DFN Cave Studio — New Project [unsaved]")
            self.set_status("New project created")
            self._update_project_tree_from_project(project)
            self._ensure_database_panel(project)
            self.log_message(f"New project '{project.metadata.name}' created")
        except (RuntimeError, TypeError, ValueError) as e:
            self.log_error(f"Failed to create project: {e}")

    def _clear_renderer_state(self) -> None:
        """Clear renderer bookkeeping and every actor in the active plotter."""
        if self._plotter is None:
            return
        self._clear_m9_layers()
        self._clear_dfn_layers()
        if self._dfn_renderer is not None:
            try:
                self._dfn_renderer.clear(self._plotter)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self.log_warning("Failed to clear 3D renderer during new project")
        try:
            self._plotter.clear()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self.log_warning("Failed to clear plotter actors")

    def _get_m9_layer_manager(self):
        """Return the main-window-owned, session-only M9 slice registry."""
        if self._plotter is None:
            return None
        if self._m9_layer_manager is None or self._m9_layer_manager.plotter is not self._plotter:
            from dfn_cave_studio.visualization.m9_layer_manager import M9LayerManager

            self._m9_layer_manager = M9LayerManager(self._plotter)
        return self._m9_layer_manager

    def _clear_m9_layers(self) -> None:
        """Remove M9 slice actors without touching other scene namespaces."""
        if self._m9_layer_manager is not None:
            self._m9_layer_manager.clear_m9_layers()

    def _get_dfn_layer_manager(self):
        """Return the main-window-owned, session-only M10 DFN registry."""
        if self._plotter is None:
            return None
        if self._dfn_layer_manager is None or self._dfn_layer_manager.plotter is not self._plotter:
            from dfn_cave_studio.visualization.dfn_layer_manager import DFNLayerManager

            self._dfn_layer_manager = DFNLayerManager(self._plotter)
        return self._dfn_layer_manager

    def _clear_dfn_layers(self) -> None:
        """Remove M10 DFN actors without touching M9 or reference actors."""
        if self._dfn_layer_manager is not None:
            self._dfn_layer_manager.clear_dfn_layers()

    def _on_open_project(self) -> None:
        """Open an existing project (.dfnproj or legacy .dfncs)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "DFN Cave Studio Projects (*.dfnproj *.dfncs);;"
            "ZIP Projects (*.dfnproj);;"
            "Legacy JSON Projects (*.dfncs);;"
            "All Files (*)",
        )
        if not path:
            return

        self.log_message(f"Opening project: {path}")
        try:
            project = self._load_project(Path(path))
            self._recent_manager.add(Path(path), project.metadata.name)
            self.setWindowTitle(f"DFN Cave Studio — {project.metadata.name}")
            self.set_status(f"Loaded: {Path(path).name}")
            self._update_project_tree_from_project(project)
            self._update_recent_menu()
            self._restore_project_to_ui(project)
            self._ensure_database_panel(project)
            self.log_message(f"Project '{project.metadata.name}' loaded ({project.model_volume:.0f} m³)")
        except Exception as e:
            self.log_error(f"Failed to open project: {e}")
            QMessageBox.critical(self, "Open Project Error", str(e))

    def _on_save_project(self) -> None:
        """Save the current project."""
        if not self._project_store.has_project:
            self._on_save_project_as()
            return
        if self._project_store.current_path is None:
            self._on_save_project_as()
            return

        try:
            self.set_status("Saving project…")
            path = self._save_project_to(self._project_store.current_path)
            self._recent_manager.add(path, self._project_store.current_project.metadata.name)
            self.setWindowTitle(f"DFN Cave Studio — {self._project_store.current_project.metadata.name}")
            self.set_status(f"Saved: {path.name}")
            self.log_message(f"Project saved to {path}")
            self._update_recent_menu()
        except Exception as e:
            self.log_error(f"Failed to save project: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_save_project_as(self) -> None:
        """Save project to a new location (.dfnproj or .dfncs)."""
        if not self._project_store.has_project:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "untitled.dfnproj",
            "DFN Cave Studio ZIP Projects (*.dfnproj);;" "Legacy JSON Projects (*.dfncs);;" "All Files (*)",
        )
        if not path:
            return

        try:
            self.set_status("Saving project…")
            saved_path = self._save_project_to(Path(path))
            self._recent_manager.add(saved_path, self._project_store.current_project.metadata.name)
            self.setWindowTitle(f"DFN Cave Studio — {self._project_store.current_project.metadata.name}")
            self.set_status(f"Saved: {saved_path.name}")
            self.log_message(f"Project saved to {saved_path}")
            self._update_recent_menu()
        except Exception as e:
            self.log_error(f"Failed to save project: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_borehole_manager(self) -> None:
        """Show the canonical M8 borehole database."""
        if not self._project_store.has_project:
            self._on_new_project()
        self._ensure_database_panel(self._project_store.current_project)
        self._database_panel.show()
        self._database_panel.raise_()

    def _render_boreholes(self, collection) -> None:
        """Render borehole trajectories in 3D view."""
        if not self._plotter:
            return
        try:
            import pyvista as pv

            for bh in collection:
                points, _ = bh.compute_trajectory(step_length=2.0)
                if len(points) >= 2:
                    line = pv.PolyData(points)
                    tube = line.tube(radius=0.3)
                    self._plotter.add_mesh(tube, color="cyan", name=f"BH-{bh.borehole_id}")
            self._plotter.reset_camera()
        except Exception as e:
            self.log_warning(f"Borehole rendering: {e}")

    def _on_voxel_settings(self) -> None:
        """Open the M8 voxel-grid preview and confirmation step."""
        self._open_spatial_settings("voxel")

    def _open_spatial_settings(self, mode: str) -> None:
        """Open one explicit M8 spatial-workflow mode."""
        if not self._project_store.has_project:
            self._on_new_project()
        from dfn_cave_studio.ui.dialogs.m8_spatial_grid_dialog import M8SpatialGridDialog

        project = self._project_store.current_project
        scientific_before = self._spatial_science_snapshot(project)
        m10_generation_before = (
            project.spatial_grid_config.generation_domain.model_dump(mode="json")
            if project.spatial_grid_config is not None
            else None
        )
        persisted_before = {
            "spatial_grid_config": (
                project.spatial_grid_config.model_dump(mode="json") if project.spatial_grid_config is not None else None
            ),
            "model_bounds": project.model_bounds.model_dump(mode="json"),
            "voxel_config": project.voxel_config.model_dump(mode="json"),
        }
        workflow_before = self._workflow.to_dict()
        m8_dialog = M8SpatialGridDialog(project, plotter=self._plotter, mode=mode, parent=self)
        if m8_dialog.exec() == QDialog.DialogCode.Accepted:
            config = m8_dialog.get_config()
            project.spatial_grid_config = config
            project.model_bounds = config.analysis_domain
            if mode == "bounds":
                analysis_changed = scientific_before["analysis_domain"] != config.analysis_domain.model_dump(mode="json")
                if not self._workflow.is_step_done("bounds"):
                    self._workflow.complete_step("bounds")
                if analysis_changed or not self._workflow.is_step_done("voxel_grid"):
                    self._workflow.mark_ready("voxel_grid")
                message = "M8 voxel analysis boundary confirmed"
            else:
                project.voxel_config = m8_dialog.get_voxel_config()
                if not self._workflow.is_step_done("voxel_grid"):
                    self._workflow.complete_step("voxel_grid")
                self._refresh_m10_readiness()
                message = "M8 voxel grid and DFN generation domain confirmed"

            scientific_after = self._spatial_science_snapshot(project)
            if any(scientific_before[key] != scientific_after[key] for key in scientific_before):
                self._invalidate_m9_spatial_results()
                self._refresh_m9_readiness()
            m10_generation_after = config.generation_domain.model_dump(mode="json")
            if m10_generation_before != m10_generation_after:
                self._workflow.invalidate_steps(["explicit_dfn"])
            persisted_after = {
                "spatial_grid_config": config.model_dump(mode="json"),
                "model_bounds": project.model_bounds.model_dump(mode="json"),
                "voxel_config": project.voxel_config.model_dump(mode="json"),
            }
            if persisted_before != persisted_after or workflow_before != self._workflow.to_dict():
                self._project_store.mark_dirty()
                self._update_project_tree_from_project(project)
                self.log_message(message)

    @staticmethod
    def _spatial_science_snapshot(project) -> dict[str, dict]:
        """Capture only M9-relevant spatial inputs, excluding preview and M10 buffer settings."""
        analysis = (
            project.spatial_grid_config.analysis_domain
            if project.spatial_grid_config is not None
            else project.model_bounds
        )
        return {
            "analysis_domain": analysis.model_dump(mode="json"),
            "voxel_config": project.voxel_config.model_dump(mode="json"),
            "rock_mask": project.rock_mask.model_dump(mode="json"),
            "excavation_mask": project.excavation_mask.model_dump(mode="json"),
        }

    def _on_joint_set_manager(self) -> None:
        """Open joint set manager dialog."""
        if not self._project_store.has_project:
            QMessageBox.warning(self, "No Project", "Please create or open a project first.")
            return
        project = self._project_store.current_project
        from dfn_cave_studio.ui.dialogs.joint_set_dialog import JointSetManagerDialog

        dlg = JointSetManagerDialog(
            joint_sets=project.joint_sets,
            model_volume=project.model_bounds.volume,
            parent=self,
            borehole_collection=project.borehole_collection,
        )
        if dlg.exec() == JointSetManagerDialog.DialogCode.Accepted:
            project.joint_sets = dlg.get_joint_sets()
            self._project_store.mark_dirty()
            self.log_message(f"Updated {len(project.joint_sets)} joint sets")
            self._update_project_tree_from_project(project)

    def _on_generate_dfn(self) -> None:
        """Open the M10 explicit DFN workflow entry."""
        self._m7_explicit_dfn()

    def _on_generate_dfn_legacy(self) -> None:
        """Run full computation pipeline: DFN → Voxel → Connectivity."""
        if not self._project_store.has_project:
            QMessageBox.warning(self, "No Project", "Please create or open a project first.")
            return

        project = self._project_store.current_project
        if not project.joint_sets:
            QMessageBox.warning(self, "No Joint Sets", "Please add at least one joint set.")
            return

        # Create pipeline worker
        from dfn_cave_studio.workers.pipeline_worker import PipelineWorker
        from dfn_cave_studio.ui.dialogs.compute_dialog import ComputePipelineDialog

        worker = PipelineWorker(
            joint_sets=project.joint_sets,
            bounds=project.model_bounds,
            voxel_config=project.voxel_config,
            master_seed=project.config.master_seed,
            realization_number=len(project.dfn_realizations),
        )

        dlg = ComputePipelineDialog(worker, self)
        if dlg.exec() == ComputePipelineDialog.DialogCode.Accepted:
            results = dlg.get_results()
            if "dfn" in results:
                realization = results["dfn"]
                project.dfn_realizations.append(realization)
                self._project_store.mark_dirty()

                self.log_message(
                    f"DFN: {realization.generation_result.total_fractures} fractures, "
                    f"P32={realization.generation_result.achieved_p32:.3f}"
                )
                self.set_status(f"DFN: {realization.generation_result.total_fractures} fractures")

                # Render DFN in 3D
                if self._plotter:
                    try:
                        if self._dfn_renderer is None:
                            from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer

                            self._dfn_renderer = DFNRenderer()
                        self._dfn_renderer.render_to_plotter(self._plotter, realization, project.joint_sets)
                        # Add model bounds box
                        self._render_bounds_box()
                        self._plotter.show_axes()
                        self._plotter.reset_camera()
                    except Exception as e:
                        self.log_error(f"3D rendering failed: {e}")

            if "connectivity" in results:
                conn = results["connectivity"]
                stats = conn.statistics()
                project.connectivity_results = stats
                project.connectivity_clusters = conn.component_labels().tolist()
                self.log_message(
                    f"Connectivity: {stats.get('n_edges', '?')} edges, "
                    f"{stats.get('n_components', '?')} clusters, "
                    f"largest={stats.get('largest_component_size', '?')}"
                )
                # Percolation detail
                b = project.model_bounds
                percolation = conn.percolation_detail((b.x_min, b.x_max), (b.y_min, b.y_max), (b.z_min, b.z_max))
                if percolation:
                    project.connectivity_results["percolation"] = percolation

            if "voxel_grid" in results and "dfn" in results:
                from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore

                try:
                    voxel_data = ZipProjectStore.extract_voxel_p32_results(
                        results["voxel_grid"],
                        project.connectivity_clusters,
                    )
                    project.voxel_p32_results = voxel_data
                    self.log_message(f"Voxel P32: {len(voxel_data)} active cells")
                except Exception as e:
                    self.log_warning(f"Voxel data extraction: {e}")

                # Render voxel P32
                if self._plotter and project.voxel_p32_results:
                    try:
                        if self._dfn_renderer is None:
                            from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer

                            self._dfn_renderer = DFNRenderer()
                        self._dfn_renderer.render_voxel_p32(
                            project.voxel_p32_results,
                            project.model_bounds,
                            project.voxel_config,
                            self._plotter,
                        )
                    except Exception as e:
                        self.log_error(f"Voxel rendering failed: {e}")

            # Render boreholes and observations
            if self._plotter and project.borehole_collection:
                try:
                    if self._dfn_renderer is None:
                        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer

                        self._dfn_renderer = DFNRenderer()
                    self._dfn_renderer.render_boreholes(project.borehole_collection, self._plotter)
                    self._dfn_renderer.render_fracture_observations(project.borehole_collection, self._plotter)
                except Exception as e:
                    self.log_error(f"Borehole rendering failed: {e}")

            # Render model bounds
            if self._plotter:
                self._render_bounds_box()

            self._update_project_tree_from_project(project)

    def _render_bounds_box(self) -> None:
        """Render model boundary wireframe box using DFNRenderer."""
        if not self._plotter or not self._project_store.has_project:
            return
        try:
            if self._dfn_renderer is None:
                from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer

                self._dfn_renderer = DFNRenderer()
            b = self._project_store.current_project.model_bounds
            self._dfn_renderer.render_model_bounds(b, self._plotter)
            self._dfn_renderer.add_coordinate_axes(self._plotter)
        except Exception as e:
            self.log_error(f"Bounds rendering failed: {e}")

    # ── M7 workflow handlers ──────────────────────────────────────────

    def _m7_import(self) -> None:
        """Show the M8 database, where independent imports are started."""
        self._on_borehole_manager()

    def _m7_clean(self) -> None:
        """Open the transactional M8 data-quality production interface."""
        if not self._project_store.has_project:
            self._on_new_project()
        from dfn_cave_studio.ui.dialogs.m8_quality_dialog import M8QualityDialog

        project = self._project_store.current_project
        dialog = M8QualityDialog(project, self._workflow, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.committed_changes:
            self._project_store.mark_dirty()
            self._workflow_panel._refresh()
            if self._database_panel is not None:
                self._database_panel.refresh()
            self._update_project_tree_from_project(project)

    def _m7_bounds(self) -> None:
        """Open M8 model-boundary configuration."""
        self._open_spatial_settings("bounds")

    def _m7_voxel_grid(self) -> None:
        """Open M8 voxel-grid preview and confirmation."""
        self._on_voxel_settings()

    def _m7_holdout(self) -> None:
        """Open validation holdout dialog."""
        from dfn_cave_studio.ui.dialogs.m7_holdout_dialog import M7HoldoutDialog

        if not self._project_store.has_project:
            QMessageBox.warning(self, "No Project", "Import data first.")
            return
        project = self._project_store.current_project
        dlg = M7HoldoutDialog(project, self._workflow, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.committed_changes:
            self._project_store.mark_dirty()
            self._invalidate_m9_state()
            self._refresh_m9_readiness()

    def _m7_domains(self) -> None:
        """Open structural domain editor."""
        from dfn_cave_studio.ui.dialogs.m7_domain_dialog import M7DomainDialog

        if not self._project_store.has_project:
            QMessageBox.warning(self, "No Project", "Load data first.")
            return
        project = self._project_store.current_project
        dlg = M7DomainDialog(project, self._workflow, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.committed_changes:
            self._project_store.mark_dirty()
            self._invalidate_m9_state()
            self._refresh_m9_readiness()

    def _m7_joint_sets(self) -> None:
        """Open joint set identification dialog (wired to M7 services)."""
        from dfn_cave_studio.ui.dialogs.m7_joint_set_dialog import M7JointSetDialog

        if not self._project_store.has_project:
            QMessageBox.warning(self, "No Project", "Load data and define domains first.")
            return
        project = self._project_store.current_project
        dlg = M7JointSetDialog(project, self._workflow, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.committed_changes:
            self._project_store.mark_dirty()
            self._invalidate_m9_state()
            self._refresh_m9_readiness()

    def _invalidate_m9_state(self) -> None:
        """Clear only M9 results that depend on changed M8 inputs."""
        if not self._project_store.has_project:
            return
        from dfn_cave_studio.models.m9 import M9State

        project = self._project_store.current_project
        seed = project.m9_state.random_seed
        settings = project.m9_state.density_settings
        project.m9_state = M9State(random_seed=seed, density_settings=settings)
        self._workflow.invalidate_steps(["density", "size", "parameter_field", "validation", "explicit_dfn"])

    def _invalidate_m9_spatial_results(self) -> None:
        """Mark spatial M9 products stale while retaining their auditable values."""
        self._workflow.invalidate_steps(["parameter_field", "validation", "explicit_dfn"])

    def _refresh_m9_readiness(self) -> None:
        """Expose density modelling only when every M8 scientific dependency is complete."""
        from dfn_cave_studio.services.workflow_controller import StepStatus

        density = self._workflow.get_step("density")
        prerequisites = ("clean", "holdout", "domains", "joint_sets")
        if density is not None and density.status in (StepStatus.NOT_STARTED, StepStatus.READY):
            if all(self._workflow.is_step_done(step_id) for step_id in prerequisites):
                self._workflow.mark_ready("density")

    def _refresh_m10_readiness(self) -> None:
        """Expose M10 only when its real M9 and spatial prerequisites exist."""
        from dfn_cave_studio.services.workflow_controller import StepStatus

        step = self._workflow.get_step("explicit_dfn")
        prerequisites = ("density", "size", "parameter_field", "voxel_grid")
        project = self._project_store.current_project if self._project_store.has_project else None
        if (
            step is not None
            and step.status in (StepStatus.NOT_STARTED, StepStatus.READY)
            and project is not None
            and project.spatial_grid_config is not None
            and all(self._workflow.is_step_done(step_id) for step_id in prerequisites)
        ):
            self._workflow.mark_ready("explicit_dfn")

    def _open_m9_dialog(self, dialog_type, completed_step: str) -> None:
        """Open one transactional M9 workflow dialog and apply lifecycle effects."""
        if not self._project_store.has_project:
            QMessageBox.warning(self, "No Project", "Open or create a project first.")
            return
        project = self._project_store.current_project
        dialog = dialog_type(project, self._workflow, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.committed_changes:
            if completed_step in {"density", "size", "parameter_field"}:
                self._workflow.invalidate_steps(["explicit_dfn"])
            self._project_store.mark_dirty()
            self._workflow_panel._refresh()
            self._update_project_tree_from_project(project)
            self.log_message(f"M9 workflow step completed: {completed_step}")
            self._refresh_m10_readiness()

    def _m7_density(self) -> None:
        """Open M9 P10/P32 density modelling."""
        from dfn_cave_studio.ui.dialogs.m9_dialogs import M9DensityDialog

        self._open_m9_dialog(M9DensityDialog, "density")

    def _m7_size(self) -> None:
        """Open M9 fracture-size modelling."""
        from dfn_cave_studio.ui.dialogs.m9_dialogs import M9SizeDialog

        self._open_m9_dialog(M9SizeDialog, "size")

    def _m7_parameter_field(self) -> None:
        """Open M9 first-voxelization generation and preview."""
        from dfn_cave_studio.ui.dialogs.m9_dialogs import M9ParameterFieldDialog

        self._open_m9_dialog(M9ParameterFieldDialog, "parameter_field")

    def _m7_validation(self) -> None:
        """Open independent M9 held-out validation."""
        from dfn_cave_studio.ui.dialogs.m9_dialogs import M9ValidationDialog

        self._open_m9_dialog(M9ValidationDialog, "validation")

    def _m7_explicit_dfn(self) -> None:
        """Open transactional M10 generation and session layer management."""
        if not self._project_store.has_project:
            QMessageBox.warning(self, "No Project", "Open an M9 project first.")
            return
        from dfn_cave_studio.ui.dialogs.m10_dialog import M10ExplicitDFNDialog

        project = self._project_store.current_project
        dialog = M10ExplicitDFNDialog(project, self._workflow, self._get_dfn_layer_manager(), self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.committed_changes:
            self._project_store.mark_dirty()
            self._workflow_panel._refresh()
            self._update_project_tree_from_project(project)
            self.log_message("M10 explicit DFN generation committed")

    def _on_domain_manager(self) -> None:
        """Open domain manager."""
        self.log_message("Domain Manager: not yet implemented (planned for M9)")

    def _on_connectivity(self) -> None:
        """Show connectivity results if available."""
        project = self._project_store.current_project if self._project_store.has_project else None
        if project and project.dfn_realizations:
            from dfn_cave_studio.connectivity.connectivity_graph import ConnectivityGraph

            graph = ConnectivityGraph(project.dfn_realizations[-1])
            graph.compute_edges()
            graph.find_components()
            stats = graph.statistics()
            msg = (
                f"Connectivity Analysis\n\n"
                f"Fractures: {stats['n_fractures']}\n"
                f"Edges: {stats['n_edges']}\n"
                f"Components: {stats['n_components']}\n"
                f"Largest cluster: {stats['largest_component_size']}\n"
                f"Isolated: {stats['isolated_fractures']}\n"
                f"Avg degree: {stats['average_degree']:.2f}"
            )
            QMessageBox.information(self, "Connectivity Analysis", msg)
            self.log_message(f"Connectivity: {stats['n_edges']} edges, {stats['n_components']} components")
        else:
            QMessageBox.warning(self, "No Data", "Generate a DFN first.")

    def _on_fragmentation(self) -> None:
        """Run fragmentation analysis."""
        self.log_message("Fragmentation Analysis: not yet implemented (fixed roadmap: M12)")

    def _on_reset_view(self) -> None:
        """Reset the 3D view."""
        if self._plotter:
            self._plotter.reset_camera()
        self.log_message("View reset")

    def _on_top_view(self) -> None:
        """Switch to top view."""
        if self._plotter:
            self._plotter.view_xy()
        self.log_message("Top view")

    def _on_front_view(self) -> None:
        """Switch to front view."""
        if self._plotter:
            self._plotter.view_xz()
        self.log_message("Front view")

    def _on_left_view(self) -> None:
        """Switch to left view."""
        if self._plotter:
            self._plotter.view_yz()
        self.log_message("Left view")

    def _on_export_3dec(self) -> None:
        """Export for 3DEC."""
        self._do_export("3DEC")

    def _on_export_flac3d(self) -> None:
        """Export for FLAC3D."""
        self._do_export("FLAC3D")

    def _on_export_vtk(self) -> None:
        """Export to VTK."""
        self._do_export("VTK")

    def _do_export(self, fmt: str) -> None:
        """Export current results to the specified format."""
        if not self._project_store.has_project:
            QMessageBox.warning(self, "No Project", "No project to export.")
            return
        project = self._project_store.current_project
        if not project.dfn_realizations:
            QMessageBox.warning(self, "No Data", "Generate a DFN before exporting.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {fmt}",
            f"export_{fmt.lower()}",
            "All Files (*)",
        )
        if not path:
            return

        try:
            out = Path(path)
            if fmt == "VTK":

                out.mkdir(parents=True, exist_ok=True)
                realization = project.dfn_realizations[-1]
                from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer

                meshes = []
                for f in realization.stochastic_fractures[:1000]:
                    g = f.geometry
                    disk = DFNRenderer.fracture_to_disk_mesh(
                        g.center, g.normal, f.radius if f.radius > 0 else (g.radius or 1.0)
                    )
                    meshes.append(disk)
                if meshes:
                    merged = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
                    merged.save(str(out / "fractures.vtp"))
                self.log_message(f"Exported {len(meshes)} fractures to {out}/fractures.vtp")
            else:
                # Generic CSV export
                import csv

                with open(out, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["fracture_id", "set_id", "center_x", "center_y", "center_z", "radius"])
                    for frac in project.dfn_realizations[-1].stochastic_fractures:
                        g = frac.geometry
                        writer.writerow(
                            [str(frac.fracture_id), frac.set_id, g.center_x, g.center_y, g.center_z, frac.radius]
                        )
                self.log_message(f"Exported to {out}")

            self.set_status(f"Export complete: {out}")
        except Exception as e:
            self.log_error(f"Export failed: {e}")
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_settings(self) -> None:
        """Open settings dialog."""
        self.log_message("Settings requested")

    def _on_about(self) -> None:
        """Show about dialog."""
        import sys

        QMessageBox.about(
            self,
            "About DFN Cave Studio",
            "<h2>DFN Cave Studio</h2>"
            "<p>Version 0.10.1-M10</p>"
            "<p>Discrete Fracture Network Modeling<br>"
            "for Underground Block Cave Mining Research</p>"
            f"<p>Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}</p>"
            "<p>License: MIT</p>",
        )

    def _on_documentation(self) -> None:
        """Open documentation."""
        self.log_message("Documentation requested")

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle project tree item clicks."""
        self.log_message(f"Selected: {item.text(column)}")

    # ------------------------------------------------------------------
    # Window Events
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Save state before closing."""
        # Check for unsaved changes
        if self._project_store.is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "The project has unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_project()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        self._auto_save_timer.stop()
        settings = QSettings("DFNCaveStudio", "MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        self.log_message("DFN Cave Studio closing")
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Project Management Helpers
    # ------------------------------------------------------------------

    def _on_project_dirty_changed(self, dirty: bool) -> None:
        """Handle project dirty state change."""
        if self._project_store.has_project:
            title = f"DFN Cave Studio — {self._project_store.current_project.metadata.name}"
            if dirty:
                title += " *"
            self.setWindowTitle(title)

    def _on_auto_save_tick(self) -> None:
        """Periodic auto-save check."""
        if self._project_store.tick_auto_save():
            self.log_message("Auto-saved project")
            self.set_status("Auto-saved")

    def _load_project(self, path: Path):
        """Load project, auto-detecting .dfnproj vs .dfncs format."""
        from dfn_cave_studio.services.m7_state import set_workflow

        self._clear_m9_layers()
        self._clear_dfn_layers()
        suffix = path.suffix.lower()
        if suffix == ".dfnproj":
            from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore

            zps = ZipProjectStore()
            project = zps.load(path)
            # Ensure workflow is bound to project
            wf = self._workflow
            m7 = getattr(project, "_m7_data", {}) or {}
            saved_wf = m7.get("workflow")
            if saved_wf is not None:
                wf = saved_wf
            set_workflow(project, wf)
            # Update ProjectStore via public API
            self._project_store.adopt_project(project, Path(path))
            return project
        else:
            return self._project_store.open(path)

    def _save_project_to(self, path: Path) -> Path:
        """Save project, auto-detecting .dfnproj vs .dfncs format."""
        from dfn_cave_studio.services.m7_state import set_workflow

        project = self._project_store.current_project
        # Persist current workflow into project before saving
        set_workflow(project, self._workflow)
        return self._project_store.save_as(path)

    def _restore_project_to_ui(self, project) -> None:
        """Restore project state to UI: workflow, renderer, project tree."""
        self._clear_renderer_state()
        # Restore M7 workflow from saved state
        m7 = getattr(project, "_m7_data", {}) or {}
        saved_wf = m7.get("workflow")
        if saved_wf is not None:
            self._workflow = saved_wf
            self._workflow_panel._controller = saved_wf
            saved_wf.add_change_listener(self._workflow_panel._refresh)
            self._workflow_panel._refresh()
        # Restore holdout
        saved_ho = m7.get("holdout")
        if saved_ho is not None:
            from dfn_cave_studio.services.holdout_service import HoldoutService

            if not isinstance(saved_ho, HoldoutService):
                m7["holdout"] = HoldoutService.from_dict(saved_ho.to_dict() if hasattr(saved_ho, "to_dict") else {})

        self._refresh_m10_readiness()

        if not self._plotter:
            self._update_project_tree_from_project(project)
            return
        try:
            if self._dfn_renderer is None:
                from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer

                self._dfn_renderer = DFNRenderer()

            # Render boreholes
            if project.borehole_collection:
                self._dfn_renderer.render_boreholes(project.borehole_collection, self._plotter)
                self._dfn_renderer.render_fracture_observations(project.borehole_collection, self._plotter)

            # Render DFN
            if project.dfn_realizations:
                last = project.dfn_realizations[-1]
                self._dfn_renderer.render_to_plotter(self._plotter, last, project.joint_sets)

            # Render voxel P32
            if project.voxel_p32_results:
                self._dfn_renderer.render_voxel_p32(
                    project.voxel_p32_results,
                    project.model_bounds,
                    project.voxel_config,
                    self._plotter,
                )

            # Render model bounds and axes
            self._dfn_renderer.render_model_bounds(project.model_bounds, self._plotter)
            self._dfn_renderer.add_coordinate_axes(self._plotter)
            self._plotter.reset_camera()
        except Exception as e:
            self.log_error(f"Failed to restore project to UI: {e}")

    def _update_project_tree_from_project(self, project) -> None:
        """Refresh the project tree to reflect the current project."""
        tree = self._project_tree
        tree.clear()

        root = QTreeWidgetItem(tree, [project.metadata.name])
        root.setExpanded(True)

        # Model section
        model = QTreeWidgetItem(root, ["Model"])
        QTreeWidgetItem(
            model,
            [
                f"Bounds: {project.model_bounds.width:.0f}x{project.model_bounds.depth:.0f}x{project.model_bounds.height:.0f} m"
            ],
        )
        if project.spatial_grid_config is not None:
            generation = project.spatial_grid_config.generation_domain
            QTreeWidgetItem(
                model,
                [f"DFN generation: {generation.width:.0f}x" f"{generation.depth:.0f}x{generation.height:.0f} m"],
            )
        QTreeWidgetItem(
            model,
            [
                f"Voxel: {project.voxel_config.cell_size_x:.1f}x{project.voxel_config.cell_size_y:.1f}x{project.voxel_config.cell_size_z:.1f} m"
            ],
        )
        if project.surface_model:
            QTreeWidgetItem(model, [f"Surface: {project.surface_model.name}"])

        # Data section
        data = QTreeWidgetItem(root, ["Data"])
        n_boreholes = len(project.borehole_collection)
        QTreeWidgetItem(data, [f"Boreholes ({n_boreholes})"])
        database_counts = project.borehole_database.counts()
        QTreeWidgetItem(
            data,
            [
                f"Database: raw {database_counts['raw']}, formal {database_counts['formal']}, "
                f"excluded {database_counts['excluded']}, pending {database_counts['pending']}"
            ],
        )
        QTreeWidgetItem(data, [f"Deterministic Fractures ({len(project.deterministic_fractures)})"])

        # DFN section
        dfn_node = QTreeWidgetItem(root, ["DFN"])
        QTreeWidgetItem(dfn_node, [f"Joint Sets ({len(project.joint_sets)})"])
        for js in project.joint_sets:
            QTreeWidgetItem(dfn_node, [f"  {js.name} (P32={js.target_p32})"])
        QTreeWidgetItem(dfn_node, [f"Realizations ({len(project.dfn_realizations)})"])

        m9_node = QTreeWidgetItem(root, ["M9 Parameter Field"])
        QTreeWidgetItem(m9_node, [f"P10 intervals ({len(project.m9_state.p10_intervals)})"])
        QTreeWidgetItem(m9_node, [f"P32 estimates ({len(project.m9_state.p32_estimates)})"])
        QTreeWidgetItem(m9_node, [f"Size models ({len(project.m9_state.size_models)})"])
        if project.m9_state.parameter_field_metadata is not None:
            shape = project.m9_state.parameter_field_metadata.shape
            QTreeWidgetItem(m9_node, [f"Voxel parameter field {shape[0]}x{shape[1]}x{shape[2]}"])
        QTreeWidgetItem(m9_node, [f"Validation: {project.m9_state.validation_summary.state.value}"])

        m10_node = QTreeWidgetItem(root, ["M10 Explicit DFN"])
        QTreeWidgetItem(m10_node, [f"Deterministic structures ({len(project.m10_state.deterministic_structures)})"])
        QTreeWidgetItem(m10_node, [f"Realizations ({len(project.m10_state.realizations)})"])
        QTreeWidgetItem(
            m10_node,
            [f"Fractures ({sum(item.fracture_count for item in project.m10_state.realizations)})"],
        )

        # Domains section
        domains_node = QTreeWidgetItem(root, ["Structural Domains"])
        QTreeWidgetItem(domains_node, [f"Domains ({len(project.structural_domains.domains)})"])

        # Analysis section
        analysis = QTreeWidgetItem(root, ["Analysis"])
        QTreeWidgetItem(analysis, ["Connectivity"])
        QTreeWidgetItem(analysis, ["Fragmentation"])

        for i in range(root.childCount()):
            root.child(i).setExpanded(True)

    def _ensure_database_panel(self, project) -> None:
        """Create or rebind the M8 database dock."""
        from dfn_cave_studio.ui.panels.borehole_database_panel import BoreholeDatabasePanel

        if self._database_panel is None:
            self._database_panel = BoreholeDatabasePanel(project, on_changed=self._on_database_changed, parent=self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._database_panel)
        else:
            self._database_panel.set_project(project)

    def _on_database_changed(self) -> None:
        """Apply project lifecycle effects after an accepted database change."""
        if not self._project_store.has_project:
            return
        project = self._project_store.current_project
        counts = project.borehole_database.counts()
        from dfn_cave_studio.services.workflow_controller import StepStatus

        import_step = self._workflow.get_step("import")
        clean_step = self._workflow.get_step("clean")
        if counts["raw"]:
            self._workflow.complete_step("import")
            if import_step is not None:
                import_step.set_metadata("excluded_count", counts["excluded"])
                import_step.set_metadata("pending_count", counts["pending"])
        if counts["pending"] or project.borehole_database.unresolved_error_count:
            self._workflow.mark_issues("clean")
        elif project.borehole_database.quality_confirmed_at is not None:
            self._workflow.complete_step("clean")
            if project.borehole_database.query("collars", "formal"):
                self._workflow.mark_ready("holdout")
        elif clean_step is None or clean_step.status != StepStatus.STALE:
            self._workflow.mark_ready("clean")
        self._invalidate_m9_state()
        self._project_store.mark_dirty()
        self._update_project_tree_from_project(project)

    def _update_recent_menu(self) -> None:
        """Update the recent projects submenu."""
        self._recent_menu.clear()
        recent = self._recent_manager.list()
        if not recent:
            self._recent_menu.addAction("(No recent projects)").setEnabled(False)
        else:
            for entry in recent[:10]:
                action = QAction(f"{entry['name']} — {entry['path']}", self)
                action.setData(entry["path"])
                action.triggered.connect(self._on_open_recent)
                self._recent_menu.addAction(action)
            self._recent_menu.addSeparator()
            clear_action = QAction("Clear Recent", self)
            clear_action.triggered.connect(lambda: self._recent_manager.clear() or self._update_recent_menu())
            self._recent_menu.addAction(clear_action)

    def _on_open_recent(self) -> None:
        """Open a recent project from the menu."""
        action = self.sender()
        if action and action.data():
            path = action.data()
            if Path(path).exists():
                try:
                    project = self._load_project(Path(path))
                    self._recent_manager.add(Path(path), project.metadata.name)
                    self.setWindowTitle(f"DFN Cave Studio — {project.metadata.name}")
                    self.set_status(f"Loaded: {Path(path).name}")
                    self._update_project_tree_from_project(project)
                    self._update_recent_menu()
                    self._restore_project_to_ui(project)
                    self._ensure_database_panel(project)
                    self.log_message(f"Loaded recent project: {path}")
                except Exception as e:
                    self.log_error(f"Failed to open {path}: {e}")
            else:
                self.log_warning(f"Recent project not found: {path}")
                self._recent_manager.remove(Path(path))
                self._update_recent_menu()
