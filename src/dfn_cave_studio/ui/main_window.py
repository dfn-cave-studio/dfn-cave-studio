"""
Main window for DFN Cave Studio.
"""

from pathlib import Path
from typing import Optional

from dfn_cave_studio.ui.qt_adapter import (
    Qt,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QStatusBar,
    QMenuBar,
    QMenu,
    QAction,
    QKeySequence,
    QMessageBox,
    QDockWidget,
    QTextEdit,
    QToolBar,
    QIcon,
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

from dfn_cave_studio.core import get_config, AppVersion
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
            self._file_menu, "&New Project", "Ctrl+N",
            self._on_new_project, "Create a new DFN Cave Studio project"
        )
        self._add_menu_action(
            self._file_menu, "&Open Project...", "Ctrl+O",
            self._on_open_project, "Open an existing project"
        )
        self._file_menu.addSeparator()
        self._add_menu_action(
            self._file_menu, "&Save Project", "Ctrl+S",
            self._on_save_project, "Save current project"
        )
        self._add_menu_action(
            self._file_menu, "Save Project &As...", "Ctrl+Shift+S",
            self._on_save_project_as, "Save project to a new location"
        )
        self._file_menu.addSeparator()
        self._add_menu_action(
            self._file_menu, "&Import", None,
            None, "Import data"
        )
        self._add_menu_action(
            self._file_menu, "&Export", None,
            None, "Export data"
        )
        self._file_menu.addSeparator()
        self._recent_menu = self._file_menu.addMenu("&Recent Projects")
        self._file_menu.addSeparator()
        self._add_menu_action(
            self._file_menu, "E&xit", "Alt+F4",
            self.close, "Exit DFN Cave Studio"
        )

        # === Data Menu ===
        self._data_menu = menu_bar.addMenu("&Data")
        self._add_menu_action(
            self._data_menu, "&Borehole Manager...", None,
            self._on_borehole_manager, "Import and manage borehole data"
        )

        # === Voxel Menu ===
        self._voxel_menu = menu_bar.addMenu("&Voxel")
        self._add_menu_action(
            self._voxel_menu, "Voxel &Settings...", None,
            self._on_voxel_settings, "Configure voxel grid"
        )

        # === DFN Menu ===
        self._dfn_menu = menu_bar.addMenu("D&FN")
        self._add_menu_action(
            self._dfn_menu, "&Joint Set Manager...", None,
            self._on_joint_set_manager, "Manage fracture sets"
        )
        self._add_menu_action(
            self._dfn_menu, "&Generate DFN...", None,
            self._on_generate_dfn, "Generate stochastic DFN"
        )

        # === Domains Menu ===
        self._domains_menu = menu_bar.addMenu("D&omains")
        self._add_menu_action(
            self._domains_menu, "&Domain Manager...", None,
            self._on_domain_manager, "Manage structural domains"
        )

        # === Analysis Menu ===
        self._analysis_menu = menu_bar.addMenu("&Analysis")
        self._add_menu_action(
            self._analysis_menu, "&Connectivity...", None,
            self._on_connectivity, "Analyze fracture connectivity"
        )
        self._add_menu_action(
            self._analysis_menu, "&Fragmentation...", None,
            self._on_fragmentation, "Analyze block fragmentation"
        )

        # === Visualization Menu ===
        self._vis_menu = menu_bar.addMenu("&Visualization")
        self._add_menu_action(
            self._vis_menu, "&Reset View", "R",
            self._on_reset_view, "Reset 3D camera view"
        )
        self._add_menu_action(
            self._vis_menu, "Top &View", "T",
            self._on_top_view, "Switch to top-down view"
        )
        self._add_menu_action(
            self._vis_menu, "&Front View", "F",
            self._on_front_view, "Switch to front view"
        )
        self._add_menu_action(
            self._vis_menu, "&Left View", "L",
            self._on_left_view, "Switch to left view"
        )

        # === Export Menu ===
        self._export_menu = menu_bar.addMenu("E&xport")
        self._add_menu_action(
            self._export_menu, "Export &3DEC...", None,
            self._on_export_3dec, "Export for 3DEC"
        )
        self._add_menu_action(
            self._export_menu, "Export &FLAC3D...", None,
            self._on_export_flac3d, "Export for FLAC3D"
        )
        self._add_menu_action(
            self._export_menu, "Export &VTK...", None,
            self._on_export_vtk, "Export to VTK format"
        )

        # === Tools Menu ===
        self._tools_menu = menu_bar.addMenu("&Tools")
        self._add_menu_action(
            self._tools_menu, "&Settings...", "Ctrl+,",
            self._on_settings, "Application settings"
        )

        # === Help Menu ===
        self._help_menu = menu_bar.addMenu("&Help")
        self._add_menu_action(
            self._help_menu, "&About", None,
            self._on_about, "About DFN Cave Studio"
        )
        self._add_menu_action(
            self._help_menu, "&Documentation", "F1",
            self._on_documentation, "Open documentation"
        )

    def _init_tool_bar(self) -> None:
        """Create the main toolbar."""
        self._toolbar = QToolBar("Main Toolbar", self)
        self._toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        self._add_toolbar_action("New Project", self._on_new_project)
        self._add_toolbar_action("Open Project", self._on_open_project)
        self._add_toolbar_action("Save Project", self._on_save_project)
        self._toolbar.addSeparator()
        self._add_toolbar_action("Generate DFN", self._on_generate_dfn)

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

    def _init_central_widget(self) -> None:
        """Create the central widget with 3D view or welcome page."""
        self._central_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(self._central_splitter)

        # 3D view area
        if HAS_PYVISTAQT:
            self._plotter = PyVistaQtInteractor(self)
            self._central_splitter.addWidget(self._plotter)
        else:
            # Fallback: show welcome/label
            self._plotter = None
            welcome = QLabel(
                "<h1>DFN Cave Studio</h1>"
                "<p>Discrete Fracture Network Modeling for Block Cave Mining</p>"
                "<p>Version 0.1.0-M0</p>"
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
        self._project_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
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
        self._log_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.document().setMaximumBlockCount(1000)
        self._log_dock.setWidget(self._log_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)

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
        self.log_message("DFN Cave Studio v0.1.0-M0 started")
        self.log_message(f"Python: (see about dialog)")
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
        """Create a new project."""
        self.log_message("Creating new project...")
        try:
            project = self._project_store.new_project("New Project")
            self.setWindowTitle("DFN Cave Studio — New Project [unsaved]")
            self.set_status(f"New project created")
            self._update_project_tree_from_project(project)
            self.log_message(f"New project '{project.metadata.name}' created")
        except Exception as e:
            self.log_error(f"Failed to create project: {e}")

    def _on_open_project(self) -> None:
        """Open an existing project."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "DFN Cave Studio Projects (*.dfncs);;JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        self.log_message(f"Opening project: {path}")
        try:
            project = self._project_store.open(Path(path))
            self._recent_manager.add(Path(path), project.metadata.name)
            self.setWindowTitle(f"DFN Cave Studio — {project.metadata.name}")
            self.set_status(f"Loaded: {Path(path).name}")
            self._update_project_tree_from_project(project)
            self._update_recent_menu()
            self.log_message(f"Project '{project.metadata.name}' loaded ({project.model_volume:.0f} m³)")
        except Exception as e:
            self.log_error(f"Failed to open project: {e}")
            QMessageBox.critical(self, "Open Project Error", str(e))

    def _on_save_project(self) -> None:
        """Save the current project."""
        if not self._project_store.has_project:
            self._on_save_project_as()
            return

        try:
            path = self._project_store.save()
            self._recent_manager.add(path, self._project_store.current_project.metadata.name)
            self.setWindowTitle(f"DFN Cave Studio — {self._project_store.current_project.metadata.name}")
            self.set_status(f"Saved: {path.name}")
            self.log_message(f"Project saved to {path}")
        except Exception as e:
            self.log_error(f"Failed to save project: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_save_project_as(self) -> None:
        """Save project to a new location."""
        if not self._project_store.has_project:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "untitled.dfncs",
            "DFN Cave Studio Projects (*.dfncs);;All Files (*)",
        )
        if not path:
            return

        try:
            saved_path = self._project_store.save_as(Path(path))
            self._recent_manager.add(saved_path, self._project_store.current_project.metadata.name)
            self.setWindowTitle(f"DFN Cave Studio — {self._project_store.current_project.metadata.name}")
            self.set_status(f"Saved: {saved_path.name}")
            self.log_message(f"Project saved to {saved_path}")
            self._update_recent_menu()
        except Exception as e:
            self.log_error(f"Failed to save project: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_borehole_manager(self) -> None:
        """Open borehole manager."""
        self.log_message("Borehole Manager requested")

    def _on_voxel_settings(self) -> None:
        """Open voxel settings."""
        self.log_message("Voxel Settings requested")

    def _on_joint_set_manager(self) -> None:
        """Open joint set manager."""
        self.log_message("Joint Set Manager requested")

    def _on_generate_dfn(self) -> None:
        """Generate DFN."""
        self.log_message("Generate DFN requested")

    def _on_domain_manager(self) -> None:
        """Open domain manager."""
        self.log_message("Domain Manager requested")

    def _on_connectivity(self) -> None:
        """Run connectivity analysis."""
        self.log_message("Connectivity Analysis requested")

    def _on_fragmentation(self) -> None:
        """Run fragmentation analysis."""
        self.log_message("Fragmentation Analysis requested")

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
        self.log_message("3DEC Export requested")

    def _on_export_flac3d(self) -> None:
        """Export for FLAC3D."""
        self.log_message("FLAC3D Export requested")

    def _on_export_vtk(self) -> None:
        """Export to VTK."""
        self.log_message("VTK Export requested")

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
            "<p>Version 0.1.0-M0</p>"
            "<p>Discrete Fracture Network Modeling<br>"
            "for Underground Block Cave Mining Research</p>"
            f"<p>Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}</p>"
            "<p>License: MIT</p>"
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
                self, "Unsaved Changes",
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

    def _update_project_tree_from_project(self, project) -> None:
        """Refresh the project tree to reflect the current project."""
        tree = self._project_tree
        tree.clear()

        root = QTreeWidgetItem(tree, [project.metadata.name])
        root.setExpanded(True)

        # Model section
        model = QTreeWidgetItem(root, ["Model"])
        QTreeWidgetItem(model, [f"Bounds: {project.model_bounds.width:.0f}x{project.model_bounds.depth:.0f}x{project.model_bounds.height:.0f} m"])
        QTreeWidgetItem(model, [f"Voxel: {project.voxel_config.cell_size_x:.1f}x{project.voxel_config.cell_size_y:.1f}x{project.voxel_config.cell_size_z:.1f} m"])
        if project.surface_model:
            QTreeWidgetItem(model, [f"Surface: {project.surface_model.name}"])

        # Data section
        data = QTreeWidgetItem(root, ["Data"])
        n_boreholes = len(project.borehole_collection)
        QTreeWidgetItem(data, [f"Boreholes ({n_boreholes})"])
        QTreeWidgetItem(data, [f"Deterministic Fractures ({len(project.deterministic_fractures)})"])

        # DFN section
        dfn_node = QTreeWidgetItem(root, ["DFN"])
        QTreeWidgetItem(dfn_node, [f"Joint Sets ({len(project.joint_sets)})"])
        for js in project.joint_sets:
            QTreeWidgetItem(dfn_node, [f"  {js.name} (P32={js.target_p32})"])
        QTreeWidgetItem(dfn_node, [f"Realizations ({len(project.dfn_realizations)})"])

        # Domains section
        domains_node = QTreeWidgetItem(root, ["Structural Domains"])
        QTreeWidgetItem(domains_node, [f"Domains ({len(project.structural_domains.domains)})"])

        # Analysis section
        analysis = QTreeWidgetItem(root, ["Analysis"])
        QTreeWidgetItem(analysis, ["Connectivity"])
        QTreeWidgetItem(analysis, ["Fragmentation"])

        for i in range(root.childCount()):
            root.child(i).setExpanded(True)

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
                    project = self._project_store.open(Path(path))
                    self._recent_manager.add(Path(path), project.metadata.name)
                    self.setWindowTitle(f"DFN Cave Studio — {project.metadata.name}")
                    self.set_status(f"Loaded: {Path(path).name}")
                    self._update_project_tree_from_project(project)
                    self.log_message(f"Loaded recent project: {path}")
                except Exception as e:
                    self.log_error(f"Failed to open {path}: {e}")
            else:
                self.log_warning(f"Recent project not found: {path}")
                self._recent_manager.remove(Path(path))
                self._update_recent_menu()
