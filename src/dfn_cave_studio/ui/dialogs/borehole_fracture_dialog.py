"""Transactional UI for Phase 2A along-hole fracture realizations."""

from __future__ import annotations

from pathlib import Path

from dfn_cave_studio.models.borehole_fracture_realization import (
    BoreholeFractureGenerationConfig,
    BoreholeFractureState,
    RandomComponentStatus,
)
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.ui.dialog_geometry import fit_dialog_to_screen
from dfn_cave_studio.ui.i18n import language_manager
from dfn_cave_studio.ui.qt_adapter import (
    QCheckBox,
    QColor,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSignalBlocker,
    QSize,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QThreadPool,
    Qt,
    QVBoxLayout,
    QWidget,
)
from dfn_cave_studio.workers.borehole_fracture_worker import BoreholeFractureWorker


class BoreholeFractureDialog(QDialog):
    """Fit representative orientations and transactionally generate Phase 2A results."""

    def __init__(self, project, renderer=None, archive_path: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.renderer = renderer
        self.archive_path = Path(archive_path) if archive_path else None
        self.service = BoreholeFractureService(project)
        self.repository = BoreholeRepository(project)
        self.committed_changes = False
        self._database_snapshot = project.borehole_database.model_copy(deep=True)
        self._database_changed = False
        self._pending_state = None
        self._worker = None
        self._discard_worker_result = False
        self._pending_preview = False
        self._visible_set_ids: set[int] = set()
        self._known_set_ids: set[int] = set()
        self._filters_initialized = False
        self._random_visibility_preference = True
        self.setWindowTitle("Generate Borehole Fracture Realizations — Phase 2A")
        self._build_ui()
        self._restore_config()
        self._refresh_sites()
        self._refresh_authority()
        self._refresh_realizations()
        self._refresh_estimate()
        fit_dialog_to_screen(self, QSize(1050, 760))

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        notice = QLabel(
            "Uses clustered P/Z representative orientations and P-site spacing constraints to generate independent "
            "along-hole realizations. Results are not Formal observations and are not sent to M9 automatically."
        )
        notice.setWordWrap(True)
        outer.addWidget(notice)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        settings = QGroupBox("Global sets and generation settings")
        form = QFormLayout(settings)
        self.confirmed_groups_label = QLabel()
        self.confirmed_fit_label = QLabel()
        self.confirmed_fit_label.setWordWrap(True)
        self.open_joint_sets_button = QPushButton("Open Joint Set Management…")
        self.open_joint_sets_button.clicked.connect(self._open_joint_set_management)
        self.realization_count = QSpinBox()
        self.realization_count.setRange(1, 100)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-2_147_483_648, 2_147_483_647)
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItem("Homogeneous Poisson", "HOMOGENEOUS_POISSON")
        self.power_spin = QDoubleSpinBox()
        self.power_spin.setRange(0.01, 20.0)
        self.power_spin.setDecimals(3)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.001, 1e9)
        self.radius_spin.setSuffix(" m")
        self.radius_spin.setDecimals(3)
        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItem("Within search radius", "RADIUS")
        self.search_mode_combo.addItem("All P sites (long-range extrapolation)", "ALL_WITHIN_DOMAIN")
        self.search_mode_combo.currentIndexChanged.connect(self._on_search_mode_changed)
        self.max_neighbors = QSpinBox()
        self.max_neighbors.setRange(1, 10_000)
        self.min_neighbors = QSpinBox()
        self.min_neighbors.setRange(1, 10_000)
        self.direction_combo = QComboBox()
        self.direction_combo.addItem("Local representative", "LOCAL_REPRESENTATIVE")
        self.direction_combo.addItem(
            "Spatially fitted within global set", "SPATIALLY_FITTED_WITHIN_GLOBAL_SET"
        )
        self.direction_combo.addItem("Fixed global-set mean (legacy)", "FIXED_GLOBAL_SET_MEAN")
        self.random_combo = QComboBox()
        self.random_combo.addItem("Isotropic axial hemisphere", "ISOTROPIC_AXIAL_HEMISPHERE")
        self.memory_spin = QDoubleSpinBox()
        self.memory_spin.setRange(1, 1_048_576)
        self.memory_spin.setSuffix(" MiB")
        for label, control in (
            ("Confirmed global groups", self.confirmed_groups_label),
            ("Confirmed fit", self.confirmed_fit_label),
            ("Joint-set workflow", self.open_joint_sets_button),
            ("Realizations", self.realization_count),
            ("Master seed", self.seed_spin),
            ("Position sampler", self.sampler_combo),
            ("IDW power", self.power_spin),
            ("IDW search radius", self.radius_spin),
            ("IDW search mode", self.search_mode_combo),
            ("IDW maximum neighbors", self.max_neighbors),
            ("IDW minimum neighbors", self.min_neighbors),
            ("Dominant-set direction", self.direction_combo),
            ("Random background", self.random_combo),
            ("Memory budget", self.memory_spin),
        ):
            form.addRow(label, control)
        layout.addWidget(settings)
        for control in (
            self.realization_count,
            self.seed_spin,
            self.power_spin,
            self.radius_spin,
            self.max_neighbors,
            self.min_neighbors,
            self.memory_spin,
        ):
            control.valueChanged.connect(self._refresh_estimate)

        buttons = QHBoxLayout()
        self.generate_button = QPushButton("Generate batch")
        self.cancel_computation_button = QPushButton("Cancel computation")
        self.cancel_computation_button.setEnabled(False)
        buttons.addWidget(self.generate_button)
        buttons.addWidget(self.cancel_computation_button)
        layout.addLayout(buttons)
        self.generate_button.clicked.connect(self._generate)
        self.cancel_computation_button.clicked.connect(self._cancel_computation)

        self.estimate_label = QLabel()
        self.estimate_label.setWordWrap(True)
        layout.addWidget(self.estimate_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(QLabel("Point-cloud random component status (absence must be explicit)"))
        self.site_table = QTableWidget(0, 3)
        self.site_table.setHorizontalHeaderLabels(["Point key", "Has RANDOM row", "Status"])
        layout.addWidget(self.site_table)
        status_buttons = QHBoxLayout()
        self.mark_absent_button = QPushButton("Confirm selected as REPORTED_ABSENT")
        self.mark_not_reported_button = QPushButton("Reset selected to NOT_REPORTED")
        status_buttons.addWidget(self.mark_absent_button)
        status_buttons.addWidget(self.mark_not_reported_button)
        layout.addLayout(status_buttons)
        self.mark_absent_button.clicked.connect(
            lambda: self._set_selected_random_status(RandomComponentStatus.REPORTED_ABSENT)
        )
        self.mark_not_reported_button.clicked.connect(
            lambda: self._set_selected_random_status(RandomComponentStatus.NOT_REPORTED)
        )

        layout.addWidget(QLabel("Global/local set mapping"))
        self.mapping_table = QTableWidget(0, 5)
        self.mapping_table.setHorizontalHeaderLabels(
            ["Observation", "Point", "Local set", "Global set", "Weight"]
        )
        layout.addWidget(self.mapping_table)

        layout.addWidget(QLabel("Realizations (summary only; fracture rows are not materialized in the UI)"))
        self.realization_table = QTableWidget(0, 7)
        self.realization_table.setHorizontalHeaderLabels(
            ["Realization", "Seed", "Total", "Dominant", "Random", "Blocked intervals", "Input hash"]
        )
        layout.addWidget(self.realization_table)
        self.realization_table.itemSelectionChanged.connect(self._refresh_result_details)
        self.realization_combo = QComboBox()
        self.realization_combo.currentIndexChanged.connect(self._on_realization_selected)
        layout.addWidget(QLabel("Current realization"))
        layout.addWidget(self.realization_combo)
        self.hole_summary_label = QLabel()
        self.hole_summary_label.setWordWrap(True)
        layout.addWidget(self.hole_summary_label)
        self.set_summary_table = QTableWidget(0, 4)
        self.set_summary_table.setHorizontalHeaderLabels(["Component", "Global set", "Count", "Share"])
        layout.addWidget(self.set_summary_table)
        self.interval_table = QTableWidget(0, 7)
        self.interval_table.setHorizontalHeaderLabels(
            ["Hole", "Interval", "MD range", "Expected", "Generated", "Status", "Random constraint"]
        )
        layout.addWidget(self.interval_table)
        preview_controls = QHBoxLayout()
        self.show_preview_check = QCheckBox("Show Borehole Fracture Realization")
        self.preview_button = QPushButton("Show Current Realization")
        self.hide_preview_button = QPushButton("Hide Preview")
        self.clear_preview_button = QPushButton("Clear Preview")
        self.show_preview_check.toggled.connect(self._on_preview_visibility_changed)
        self.preview_button.clicked.connect(self._preview_selected)
        self.hide_preview_button.clicked.connect(lambda: self.show_preview_check.setChecked(False))
        self.clear_preview_button.clicked.connect(self._clear_preview)
        preview_controls.addWidget(self.show_preview_check)
        preview_controls.addWidget(self.preview_button)
        preview_controls.addWidget(self.hide_preview_button)
        preview_controls.addWidget(self.clear_preview_button)
        layout.addLayout(preview_controls)

        filter_note = QLabel(
            "Visible Joint Sets (one-click display filters; retained when switching realizations; "
            "local components inherit Global Set colour)"
        )
        filter_note.setWordWrap(True)
        layout.addWidget(filter_note)
        self.visible_sets_table = QTableWidget(0, 5)
        self.visible_sets_table.setHorizontalHeaderLabels(["Visible", "Colour", "Global Set", "Name", "Count"])
        self.visible_sets_table.itemChanged.connect(self._on_set_visibility_changed)
        layout.addWidget(self.visible_sets_table)
        group_controls = QHBoxLayout()
        self.show_all_groups_button = QPushButton("Show All")
        self.hide_all_groups_button = QPushButton("Hide All")
        self.random_visible_check = QCheckBox("Random Background: 0")
        self.random_visible_check.setChecked(True)
        for control in (
            self.show_all_groups_button,
            self.hide_all_groups_button,
            self.random_visible_check,
        ):
            group_controls.addWidget(control)
        layout.addLayout(group_controls)
        self.show_all_groups_button.clicked.connect(self._show_all_groups)
        self.hide_all_groups_button.clicked.connect(self._hide_all_groups)
        self.random_visible_check.toggled.connect(self._on_random_visibility_changed)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)

    def _restore_config(self) -> None:
        config = self.project.borehole_fracture_state.config
        self.realization_count.setValue(config.realization_count)
        self.seed_spin.setValue(config.master_seed)
        self.power_spin.setValue(config.idw_power)
        self.radius_spin.setValue(config.search_radius)
        self.search_mode_combo.setCurrentIndex(self.search_mode_combo.findData(config.idw_search_mode))
        self._on_search_mode_changed()
        self.max_neighbors.setValue(config.max_neighbors)
        self.min_neighbors.setValue(config.min_neighbors)
        self.direction_combo.setCurrentIndex(self.direction_combo.findData(config.direction_mode))
        self.memory_spin.setValue(config.memory_budget_bytes / 1024**2)

    def _on_search_mode_changed(self) -> None:
        self.radius_spin.setEnabled(self.search_mode_combo.currentData() == "RADIUS")
        self._refresh_estimate()

    def _config(self) -> BoreholeFractureGenerationConfig:
        self._confirmed_fit()
        return BoreholeFractureGenerationConfig.model_validate(
            {
                "number_of_sets": len(self.project.joint_sets),
                "realization_count": self.realization_count.value(),
                "master_seed": self.seed_spin.value(),
                "position_sampler": self.sampler_combo.currentData(),
                "idw_power": self.power_spin.value(),
                "search_radius": self.radius_spin.value(),
                "idw_search_mode": self.search_mode_combo.currentData(),
                "max_neighbors": self.max_neighbors.value(),
                "min_neighbors": self.min_neighbors.value(),
                "direction_mode": self.direction_combo.currentData(),
                "random_background_strategy": self.random_combo.currentData(),
                "memory_budget_bytes": int(self.memory_spin.value() * 1024**2),
                "use_confirmed_global_fit": True,
            }
        )

    def _confirmed_fit(self):
        """Return only the mapping explicitly confirmed by Joint Set Management."""
        if not self.project.joint_sets:
            raise ValueError("No confirmed global joint sets. Open Joint Set Management and confirm a global K first.")
        saved = self.project.borehole_fracture_state.global_fit
        if saved is None:
            raise ValueError("The local-to-global mapping has not been confirmed in Joint Set Management.")
        if self.service.authoritative_confirmed_fit() != saved:
            raise ValueError("Confirmed joint sets or imported representatives changed; reconfirm their mapping.")
        return saved.model_copy(deep=True)

    def _authority_is_ready(self) -> bool:
        try:
            self._confirmed_fit()
        except ValueError:
            return False
        return True

    def _refresh_authority(self) -> None:
        """Refresh read-only K, fit version, seed, and mapping status."""
        count = len(self.project.joint_sets)
        self.confirmed_groups_label.setText(f"K={count}" if count else "Not confirmed")
        try:
            fit = self._confirmed_fit()
            self.confirmed_fit_label.setText(
                f"{fit.algorithm} | seed={fit.random_seed} | mapping=CONFIRMED ({len(fit.mappings)} rows)"
            )
            ready = True
        except ValueError as exc:
            self.confirmed_fit_label.setText(f"mapping=NOT CONFIRMED — {exc}")
            ready = False
        self.generate_button.setEnabled(ready and self._worker is None)
        self._refresh_fit()
        self._refresh_visible_sets()

    def _open_joint_set_management(self) -> None:
        """Delegate the only global-K workflow to the owning MainWindow."""
        callback = getattr(self.parent(), "_on_joint_set_manager", None)
        if callback is None:
            QMessageBox.warning(self, "Joint Set Management", "Open Joint Set Management from the DFN menu.")
            return
        before = (
            [item.model_copy(deep=True) for item in self.project.joint_sets],
            self.project.borehole_fracture_state.global_fit.model_copy(deep=True)
            if self.project.borehole_fracture_state.global_fit is not None
            else None,
        )
        callback()
        after = (self.project.joint_sets, self.project.borehole_fracture_state.global_fit)
        if before != after:
            self._pending_state = None
        self._refresh_authority()
        self._refresh_realizations()
        self._refresh_estimate()

    def _refresh_estimate(self) -> None:
        try:
            details = self.service.estimate(self._config())
            self.estimate_label.setText(
                f"Expected {details['expected_fractures']:,.1f} fractures | persistent "
                f"{details['persistent_bytes'] / 1024**2:,.2f} MiB | peak "
                f"{details['peak_bytes'] / 1024**2:,.2f} MiB"
            )
        except (ValueError, TypeError) as exc:
            self.estimate_label.setText(str(exc))

    def _refresh_fit(self) -> None:
        fit = self.project.borehole_fracture_state.global_fit
        mappings = [] if fit is None else fit.mappings
        self.mapping_table.setRowCount(len(mappings))
        for row, mapping in enumerate(mappings):
            values = (
                mapping.observation_id,
                mapping.point_key,
                mapping.local_set_id,
                mapping.global_set_id,
                mapping.sample_weight,
            )
            for column, value in enumerate(values):
                self.mapping_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _refresh_visible_sets(self) -> None:
        """Refresh display-only group choices without changing project state."""
        available = {int(item.set_id) for item in self.project.joint_sets}
        if not self._filters_initialized or available != self._known_set_ids:
            self._visible_set_ids = set(available)
            self._filters_initialized = True
        else:
            self._visible_set_ids &= available
        self._known_set_ids = set(available)
        realization = self._selected_realization()
        counts = {} if realization is None else realization.global_set_counts
        blocker = QSignalBlocker(self.visible_sets_table)
        ordered = sorted(self.project.joint_sets, key=lambda item: item.set_id)
        self.visible_sets_table.setRowCount(len(ordered))
        for row, joint_set in enumerate(ordered):
            set_id = int(joint_set.set_id)
            visible_item = QTableWidgetItem("")
            visible_item.setData(Qt.ItemDataRole.UserRole, set_id)
            visible_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
            )
            visible_item.setCheckState(
                Qt.CheckState.Checked if set_id in self._visible_set_ids else Qt.CheckState.Unchecked
            )
            colour_item = QTableWidgetItem(str(joint_set.color))
            colour_item.setBackground(QColor(joint_set.color))
            cells = (
                visible_item,
                colour_item,
                QTableWidgetItem(str(set_id)),
                QTableWidgetItem(str(joint_set.name)),
                QTableWidgetItem(str(int(counts.get(set_id, 0)))),
            )
            for column, item in enumerate(cells):
                self.visible_sets_table.setItem(row, column, item)
        del blocker

        random_count = 0 if realization is None else int(realization.random_background_count)
        random_blocker = QSignalBlocker(self.random_visible_check)
        self.random_visible_check.setText(f"Random Background: {random_count:,}")
        self.random_visible_check.setEnabled(random_count > 0)
        self.random_visible_check.setChecked(random_count > 0 and self._random_visibility_preference)
        del random_blocker

    def _render_current_filters(self) -> None:
        if self.show_preview_check.isChecked() or (
            self.renderer is not None and self.renderer.current_realization_id is not None
        ):
            self._preview_selected()

    def _on_set_visibility_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        set_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(set_id, int):
            return
        if item.checkState() == Qt.CheckState.Checked:
            self._visible_set_ids.add(set_id)
        else:
            self._visible_set_ids.discard(set_id)
        self._render_current_filters()

    def _show_all_groups(self) -> None:
        self._visible_set_ids = {int(item.set_id) for item in self.project.joint_sets}
        self._random_visibility_preference = True
        self._refresh_visible_sets()
        blocker = QSignalBlocker(self.random_visible_check)
        self.random_visible_check.setChecked(self.random_visible_check.isEnabled())
        del blocker
        self._render_current_filters()

    def _hide_all_groups(self) -> None:
        self._visible_set_ids.clear()
        self._random_visibility_preference = False
        self._refresh_visible_sets()
        blocker = QSignalBlocker(self.random_visible_check)
        self.random_visible_check.setChecked(False)
        del blocker
        self._render_current_filters()

    def _on_random_visibility_changed(self, visible: bool) -> None:
        self._random_visibility_preference = bool(visible)
        self._render_current_filters()

    def _refresh_sites(self) -> None:
        observations = self.service.observations.orientation_points()
        random_keys = {item.point_key for item in observations if item.component_type == "RANDOM_BACKGROUND"}
        summaries = self.service.observations.orientation_point_summaries()
        self.site_table.setRowCount(len(summaries))
        for row, summary in enumerate(summaries):
            for column, value in enumerate(
                (summary.point_key, "yes" if summary.point_key in random_keys else "no", summary.random_component_status)
            ):
                self.site_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _set_selected_random_status(self, status: RandomComponentStatus) -> None:
        row = self.site_table.currentRow()
        if row < 0:
            return
        point_key = self.site_table.item(row, 0).text()
        try:
            self.repository.set_random_component_status(point_key, status)
            self._database_changed = self.project.borehole_database != self._database_snapshot
            self._pending_state = None
            self._refresh_sites()
            self._refresh_fit()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid random component status", str(exc))

    def _generate(self) -> None:
        if self._worker is not None:
            return
        try:
            config = self._config()
            estimate = self.service.estimate(config)
            if estimate["peak_bytes"] > config.memory_budget_bytes:
                raise MemoryError("Estimated generation memory exceeds the configured budget")
        except (ValueError, MemoryError) as exc:
            QMessageBox.warning(self, "Cannot generate", str(exc))
            return
        self._discard_worker_result = False
        self._set_running(True)
        worker = BoreholeFractureWorker(self.project, config, self._confirmed_fit())
        self._worker = worker
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda candidate, source=worker: self._on_generated(source, candidate))
        worker.signals.cancelled.connect(lambda source=worker: self._on_cancelled(source))
        worker.signals.failed.connect(lambda message, source=worker: self._on_failed(source, message))
        QThreadPool.globalInstance().start(worker)

    def _set_running(self, running: bool) -> None:
        language_manager().set_busy(f"borehole-phase2a-{id(self)}", running)
        self.open_joint_sets_button.setEnabled(not running)
        self.generate_button.setEnabled(not running and self._authority_is_ready())
        self.cancel_computation_button.setEnabled(running)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress.setValue(int(100 * current / max(total, 1)))
        self.status_label.setText(message)

    def _cancel_computation(self) -> None:
        if self._worker is None:
            return
        self._discard_worker_result = True
        self.status_label.setText("Cancellation requested; stopping without committing partial results…")
        self._worker.cancel()

    def _on_generated(self, worker, candidate) -> None:
        if worker is not self._worker:
            return
        self._worker = None
        self._set_running(False)
        if self._discard_worker_result:
            self.status_label.setText("Generation stopped; late result discarded.")
            return
        self._pending_state = candidate
        self.status_label.setText(
            f"Generated {len(candidate.realizations)} complete realization(s); click OK to commit."
        )
        self._refresh_fit()
        self._refresh_realizations()

    def _on_cancelled(self, worker) -> None:
        if worker is not self._worker:
            return
        self._worker = None
        self._set_running(False)
        self.status_label.setText("Generation cancelled; no partial result was committed.")

    def _on_failed(self, worker, message: str) -> None:
        if worker is not self._worker:
            return
        self._worker = None
        self._set_running(False)
        QMessageBox.critical(self, "Borehole realization generation failed", message)

    def _display_state(self):
        return self._pending_state or self.project.borehole_fracture_state

    def _refresh_realizations(self) -> None:
        realizations = self._display_state().realizations
        selected_id = self.realization_combo.currentData()
        blocker = QSignalBlocker(self.realization_combo)
        self.realization_combo.clear()
        for realization in realizations:
            self.realization_combo.addItem(realization.realization_id, realization.realization_id)
        index = self.realization_combo.findData(selected_id)
        self.realization_combo.setCurrentIndex(index if index >= 0 else (0 if realizations else -1))
        del blocker
        self.realization_table.setRowCount(len(realizations))
        for row, realization in enumerate(realizations):
            blocked = sum(item.status != "GENERATED" for item in realization.interval_diagnostics)
            dominant = realization.fracture_count - realization.random_background_count
            values = (
                realization.realization_id,
                realization.derived_seed,
                realization.fracture_count,
                dominant,
                realization.random_background_count,
                blocked,
                realization.input_hash[:12],
            )
            for column, value in enumerate(values):
                self.realization_table.setItem(row, column, QTableWidgetItem(str(value)))
        if realizations and self.realization_table.currentRow() < 0:
            self.realization_table.selectRow(0)
        self._refresh_result_details()
        self._refresh_visible_sets()

    def _selected_realization(self):
        state = self._display_state()
        realization_id = self.realization_combo.currentData()
        return next((item for item in state.realizations if item.realization_id == realization_id), None)

    def _on_realization_selected(self) -> None:
        realization_id = self.realization_combo.currentData()
        for row, realization in enumerate(self._display_state().realizations):
            if realization.realization_id == realization_id:
                self.realization_table.selectRow(row)
                break
        self._refresh_visible_sets()
        if self.show_preview_check.isChecked():
            self._preview_selected()

    def _refresh_result_details(self) -> None:
        row = self.realization_table.currentRow()
        realizations = self._display_state().realizations
        if 0 <= row < len(realizations) and self.realization_combo.currentData() != realizations[row].realization_id:
            blocker = QSignalBlocker(self.realization_combo)
            self.realization_combo.setCurrentIndex(self.realization_combo.findData(realizations[row].realization_id))
            del blocker
        realization = self._selected_realization()
        if realization is None:
            self.hole_summary_label.clear()
            self.set_summary_table.setRowCount(0)
            self.interval_table.setRowCount(0)
            return
        per_hole: dict[str, int] = {}
        for diagnostic in realization.interval_diagnostics:
            per_hole[diagnostic.hole_id] = per_hole.get(diagnostic.hole_id, 0) + diagnostic.generated_count
        self.hole_summary_label.setText(
            "Per hole: " + ", ".join(f"{hole}={count:,}" for hole, count in sorted(per_hole.items()))
        )
        rows = [("DOMINANT_SET", set_id, count) for set_id, count in sorted(realization.global_set_counts.items())]
        rows.append(("RANDOM_BACKGROUND", "—", realization.random_background_count))
        self.set_summary_table.setRowCount(len(rows))
        for row, (component, set_id, count) in enumerate(rows):
            share = count / realization.fracture_count if realization.fracture_count else 0.0
            for column, value in enumerate((component, set_id, count, f"{share:.2%}")):
                self.set_summary_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.interval_table.setRowCount(len(realization.interval_diagnostics))
        for row, diagnostic in enumerate(realization.interval_diagnostics):
            values = (
                diagnostic.hole_id,
                diagnostic.interval_index,
                f"[{diagnostic.from_depth:g}, {diagnostic.to_depth:g})",
                f"{diagnostic.expected_count:.3f}",
                diagnostic.generated_count,
                diagnostic.status,
                diagnostic.random_component_diagnostic,
            )
            for column, value in enumerate(values):
                self.interval_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _preview_selected(self) -> None:
        state = self._display_state()
        realization = self._selected_realization()
        if realization is None or self.renderer is None:
            return
        try:
            if not realization.arrays and self.archive_path is not None and state is self.project.borehole_fracture_state:
                ZipProjectStore().load_borehole_fracture_arrays(self.project, self.archive_path, realization.realization_id)
            shown = self.renderer.render(
                realization,
                self.project.joint_sets,
                visible_set_ids=self._visible_set_ids,
                show_random=self._random_visibility_preference and realization.random_background_count > 0,
            )
            blocker = QSignalBlocker(self.show_preview_check)
            self.show_preview_check.setChecked(True)
            del blocker
            self._pending_preview = state is self._pending_state
            self.status_label.setText(f"Preview displays {shown:,} of {realization.fracture_count:,} fractures.")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            QMessageBox.warning(self, "Preview failed", str(exc))

    def _on_preview_visibility_changed(self, visible: bool) -> None:
        """Show or hide only the named Phase 2A session layer."""
        if self.renderer is None:
            return
        realization = self._selected_realization()
        if visible and realization is not None and self.renderer.current_realization_id != realization.realization_id:
            self._preview_selected()
            return
        self.renderer.set_visible(visible)

    def _clear_preview(self) -> None:
        if self.renderer is not None:
            self.renderer.clear()
        blocker = QSignalBlocker(self.show_preview_check)
        self.show_preview_check.setChecked(False)
        del blocker

    def accept(self) -> None:
        if self._worker is not None:
            return
        if self._pending_state is not None:
            self.service.commit(self._pending_state)
            self.committed_changes = True
        elif self._database_changed:
            fit = self._confirmed_fit()
            self.project.borehole_fracture_state = BoreholeFractureState(
                config=self._config(),
                global_fit=fit,
                random_component_status={
                    item.point_key: item.random_component_status
                    for item in self.service.observations.orientation_point_summaries()
                },
            )
            self.committed_changes = True
        super().accept()

    def reject(self) -> None:
        if self._worker is not None:
            self._discard_worker_result = True
            self._worker.cancel()
        if self._database_changed:
            self.project.borehole_database = self._database_snapshot
            self.repository = BoreholeRepository(self.project)
        if self._pending_preview and self.renderer is not None:
            self.renderer.clear()
        super().reject()
