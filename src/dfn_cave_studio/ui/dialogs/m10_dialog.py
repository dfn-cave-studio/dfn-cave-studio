"""M10 explicit DFN generation, realization, export, and layer-management UI."""

from __future__ import annotations

from pathlib import Path

from dfn_cave_studio.models.m10 import M10FractureSource, M10GenerationConfig
from dfn_cave_studio.services.m10_service import M10Service
from dfn_cave_studio.visualization.dfn_layer_manager import stable_category_hex
from dfn_cave_studio.ui.dialog_geometry import fit_dialog_to_screen
from dfn_cave_studio.ui.i18n import language_manager, tr
from dfn_cave_studio.ui.qt_adapter import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSize,
    QSpinBox,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QThreadPool,
    QTimer,
    QVBoxLayout,
    QWidget,
)
from dfn_cave_studio.workers.m10_worker import M10GenerationWorker


class _NoWheelSpinBox(QSpinBox):
    """Keep scroll-wheel gestures available to the containing scroll area."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Prevent accidental scientific-parameter changes while scrolling."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class _NoWheelComboBox(QComboBox):
    """Prevent wheel-only changes to stable business values."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class M10ExplicitDFNDialog(QDialog):
    """Transactional M10 dialog; generation is committed only when OK is accepted."""

    def __init__(self, project, workflow, layer_manager=None, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.workflow = workflow
        self.layer_manager = layer_manager
        self.service = M10Service(project)
        self.committed_changes = False
        self._pending_realizations = None
        self._pending_structures = list(project.m10_state.deterministic_structures)
        self._worker = None
        self._layer_color = "#1976d2"
        self._hidden_size_classes: set[int] = set()
        self.setWindowTitle("M10 Explicit DFN Generation")
        self._build_ui()
        self.setMinimumSize(620, 420)
        fit_dialog_to_screen(self, QSize(1050, 720))
        self._restore_config()
        self._refresh_realizations()
        self._refresh_layers()
        self._refresh_estimate()
        language_manager().language_changed.connect(self._retranslate_dynamic_content)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, 1)
        notice = QLabel(
            "Generates explicit fracture geometry from the M9 target parameter field. "
            "Size thresholds are numerical modelling resolution recommendations, not fixed geological standards. "
            "Exact fracture–voxel P32 recomputation is deferred to M11."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)
        sources = sorted({model.source.value for model in self.project.m9_state.size_models})
        layout.addWidget(QLabel(f"Size model sources: {', '.join(sources) if sources else 'none'} | STL/OBJ complex surfaces: not supported in M10"))

        config_group = QGroupBox("Generation Configuration")
        form = QFormLayout(config_group)
        self.base_seed = _NoWheelSpinBox()
        self.base_seed.setRange(-2_147_483_648, 2_147_483_647)
        self.realization_count = _NoWheelSpinBox()
        self.realization_count.setRange(1, 100)
        self.worker_count = _NoWheelComboBox()
        for worker_count in (1, 2, 4, 8):
            self.worker_count.addItem(str(worker_count), worker_count)
        self.condition_observations = QCheckBox("Condition Calibration FULL_ORIENTATION observations")
        self.deterministic_budget = QCheckBox("Deduct deterministic structures from random P32 budget when set_id is supplied")
        self.retain_outside_deterministic = QCheckBox("Still retain deterministic discs fully outside Generation Domain")
        self.experimental_confirmed = QCheckBox("I confirm use of EXPERIMENTAL size models")
        self.validation_ack = QCheckBox("I acknowledge Validation warnings and choose to continue")
        self.threshold_mode = _NoWheelComboBox()
        self.threshold_mode.addItem("Auto", "auto")
        self.threshold_mode.addItem("Manual", "manual")
        self.generate_large = QCheckBox("Generate LARGE")
        self.generate_medium = QCheckBox("Generate MEDIUM")
        self.generate_small = QCheckBox("Generate SMALL")
        self.small_area_share = _NoWheelDoubleSpinBox()
        self.small_area_share.setRange(0.001, 0.998)
        self.small_area_share.setDecimals(3)
        self.medium_large_share = _NoWheelDoubleSpinBox()
        self.medium_large_share.setRange(0.002, 0.999)
        self.medium_large_share.setDecimals(3)
        self.manual_sm = _NoWheelDoubleSpinBox()
        self.manual_sm.setRange(0.0, 1e9)
        self.manual_sm.setSuffix(" m")
        self.manual_ml = _NoWheelDoubleSpinBox()
        self.manual_ml.setRange(0.001, 1e9)
        self.manual_ml.setSuffix(" m")
        self.restore_threshold_defaults = QPushButton("Restore Recommended Defaults")
        form.addRow("Base seed", self.base_seed)
        form.addRow("Realizations", self.realization_count)
        form.addRow("CPU workers", self.worker_count)
        form.addRow(self.condition_observations)
        form.addRow(self.deterministic_budget)
        form.addRow(self.retain_outside_deterministic)
        form.addRow(self.experimental_confirmed)
        form.addRow(self.validation_ack)
        form.addRow("Threshold Mode", self.threshold_mode)
        form.addRow("Auto SMALL P32 share", self.small_area_share)
        form.addRow("Auto cumulative MEDIUM boundary", self.medium_large_share)
        form.addRow("Manual SMALL/MEDIUM", self.manual_sm)
        form.addRow("Manual MEDIUM/LARGE", self.manual_ml)
        form.addRow(self.generate_large, self.generate_medium)
        form.addRow(self.generate_small, self.restore_threshold_defaults)
        layout.addWidget(config_group)

        self.size_estimate_table = QTableWidget(0, 7)
        self.size_estimate_table.setHorizontalHeaderLabels(
            ["Size Class", "Radius Range", "Generate", "Expected Count", "Target P32", "P32 Share", "Estimated Memory"]
        )
        layout.addWidget(self.size_estimate_table)

        self.group_summary_table = QTableWidget(0, 8)
        self.group_summary_table.setHorizontalHeaderLabels(
            [
                "Domain",
                "Set ID",
                "Observations",
                "Orientation Status",
                "P32 Target",
                "Expected",
                "Actual",
                "Unresolved Reason",
            ]
        )
        layout.addWidget(QLabel("Domain / Joint Set generation summary"))
        layout.addWidget(self.group_summary_table)

        estimate_row = QHBoxLayout()
        self.estimate_label = QLabel()
        self.estimate_button = QPushButton("Refresh Estimate")
        self.import_structures_button = QPushButton("Import Deterministic CSV")
        self.generate_button = QPushButton("Generate Batch")
        estimate_row.addWidget(self.estimate_label, 1)
        estimate_row.addWidget(self.estimate_button)
        estimate_row.addWidget(self.import_structures_button)
        layout.addLayout(estimate_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status_label = QLabel("")

        self.realizations_table = QTableWidget(0, 10)
        self.realizations_table.setHorizontalHeaderLabels(
            ["ID", "Seed", "Total", "Stochastic", "Conditioned", "Deterministic", "Target P32", "Generated P32", "Warnings", "Config hash"]
        )
        self.realizations_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.realizations_table, 2)

        realization_actions = QHBoxLayout()
        self.render_all_button = QPushButton("All Fractures – LOD")
        self.color_by = _NoWheelComboBox()
        self.color_by.addItem("Joint Set", "joint_set")
        self.color_by.addItem("Domain", "domain")
        self.color_by.addItem("Source", "source")
        self.color_by.addItem("Size Class", "size_class")
        self.display_size_class = _NoWheelComboBox()
        self.display_size_class.addItem("SMALL", 1)
        self.display_size_class.addItem("MEDIUM", 2)
        self.display_size_class.addItem("LARGE", 3)
        self.toggle_size_class_button = QPushButton("Show/Hide Size Class")
        self.render_exact_button = QPushButton("Exact Geometry (selected, max 100k)")
        self.render_set_button = QPushButton("Render Selected by Set")
        self.render_conditioned_button = QPushButton("Render Conditioned")
        self.render_deterministic_button = QPushButton("Render Deterministic")
        self.export_button = QPushButton("Export Selected")
        realization_actions.addWidget(QLabel("Color By"))
        realization_actions.addWidget(self.color_by)
        realization_actions.addWidget(self.display_size_class)
        realization_actions.addWidget(self.toggle_size_class_button)
        for button in (
            self.render_all_button, self.render_exact_button, self.render_set_button, self.render_conditioned_button,
            self.render_deterministic_button, self.export_button,
        ):
            realization_actions.addWidget(button)
        layout.addLayout(realization_actions)

        self.layers_table = QTableWidget(0, 7)
        self.layers_table.setHorizontalHeaderLabels(["Visible", "Layer", "Realization", "Set", "Source", "Opacity", "Fractures"])
        self.layers_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(QLabel("Rendered DFN Layers (session-only display state)"))
        layout.addWidget(self.layers_table, 1)
        self.visible_count_label = QLabel("Visible 0 / explicit 0")
        layout.addWidget(self.visible_count_label)
        self.legend_label = QLabel("Legend: not rendered")
        layout.addWidget(self.legend_label)
        layer_actions = QHBoxLayout()
        self.toggle_button = QPushButton("Show/Hide Selected")
        self.opacity = _NoWheelDoubleSpinBox()
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setSingleStep(0.1)
        self.opacity.setValue(0.7)
        self.opacity_button = QPushButton("Apply Opacity")
        self.color_button = QPushButton("Choose Color")
        self.remove_button = QPushButton("Remove Selected")
        self.clear_current_button = QPushButton("Clear Current Realization")
        self.clear_all_button = QPushButton("Clear All DFN Layers")
        for widget in (self.toggle_button, self.opacity, self.opacity_button, self.color_button, self.remove_button, self.clear_current_button, self.clear_all_button):
            layer_actions.addWidget(widget)
        layout.addLayout(layer_actions)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        fixed_status = QVBoxLayout()
        fixed_status.addWidget(self.progress)
        self.status_label.setWordWrap(True)
        fixed_status.addWidget(self.status_label)
        fixed_actions = QHBoxLayout()
        fixed_actions.addStretch(1)
        fixed_actions.addWidget(self.generate_button)
        fixed_actions.addWidget(self.buttons)
        fixed_status.addLayout(fixed_actions)
        outer.addLayout(fixed_status)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.estimate_button.clicked.connect(self._refresh_estimate)
        self.import_structures_button.clicked.connect(self._import_structures)
        self.generate_button.clicked.connect(self._generate)
        self.render_all_button.clicked.connect(self._render_all)
        self.toggle_size_class_button.clicked.connect(self._toggle_size_class)
        self.render_exact_button.clicked.connect(self._render_exact)
        self.render_set_button.clicked.connect(self._render_by_set)
        self.render_conditioned_button.clicked.connect(
            lambda: self._render_source(M10FractureSource.CONDITIONED_OBSERVATION.value)
        )
        self.render_deterministic_button.clicked.connect(
            lambda: self._render_source(M10FractureSource.DETERMINISTIC_STRUCTURE.value)
        )
        self.export_button.clicked.connect(self._export_selected)
        self.toggle_button.clicked.connect(self._toggle_selected_layer)
        self.opacity_button.clicked.connect(self._apply_opacity)
        self.color_button.clicked.connect(self._choose_color)
        self.remove_button.clicked.connect(self._remove_selected_layer)
        self.clear_current_button.clicked.connect(self._clear_current)
        self.clear_all_button.clicked.connect(self._clear_all)
        self.realizations_table.itemSelectionChanged.connect(self._refresh_group_summary)
        self._estimate_timer = QTimer(self)
        self._estimate_timer.setSingleShot(True)
        self._estimate_timer.setInterval(180)
        self._estimate_timer.timeout.connect(self._refresh_estimate)
        for control in (
            self.threshold_mode, self.small_area_share, self.medium_large_share,
            self.manual_sm, self.manual_ml, self.generate_large, self.generate_medium, self.generate_small,
        ):
            signal = getattr(control, "currentIndexChanged", None) or getattr(control, "valueChanged", None) or control.toggled
            signal.connect(lambda *_: self._estimate_timer.start())
        self.restore_threshold_defaults.clicked.connect(self._restore_threshold_recommendations)

    def _restore_config(self) -> None:
        config = self.project.m10_state.config
        self.base_seed.setValue(config.base_seed)
        self.realization_count.setValue(config.realization_count)
        self.worker_count.setCurrentIndex(self.worker_count.findData(config.worker_count))
        self.condition_observations.setChecked(config.condition_calibration_observations)
        self.deterministic_budget.setChecked(config.deterministic_structures_reduce_budget)
        self.retain_outside_deterministic.setChecked(config.retain_outside_deterministic)
        self.experimental_confirmed.setChecked(config.experimental_size_models_confirmed)
        self.validation_ack.setChecked(config.validation_warning_acknowledged)
        self.threshold_mode.setCurrentIndex(self.threshold_mode.findData(config.size_threshold_mode))
        self.small_area_share.setValue(config.small_area_share)
        self.medium_large_share.setValue(config.medium_large_cumulative_share)
        self.manual_sm.setValue(config.manual_small_medium_radius)
        self.manual_ml.setValue(config.manual_medium_large_radius)
        enabled = set(config.enabled_size_classes)
        self.generate_small.setChecked("SMALL" in enabled)
        self.generate_medium.setChecked("MEDIUM" in enabled)
        self.generate_large.setChecked("LARGE" in enabled)
        self._update_threshold_controls()

    def _config(self) -> M10GenerationConfig:
        old = self.project.m10_state.config
        candidate = old.model_dump(mode="python")
        candidate.update(
            {
                "base_seed": self.base_seed.value(),
                "realization_count": self.realization_count.value(),
                "worker_count": int(self.worker_count.currentData()),
                "condition_calibration_observations": self.condition_observations.isChecked(),
                "deterministic_structures_reduce_budget": self.deterministic_budget.isChecked(),
                "retain_outside_deterministic": self.retain_outside_deterministic.isChecked(),
                "experimental_size_models_confirmed": self.experimental_confirmed.isChecked(),
                "validation_warning_acknowledged": self.validation_ack.isChecked(),
                "size_threshold_mode": str(self.threshold_mode.currentData()),
                "small_area_share": self.small_area_share.value(),
                "medium_large_cumulative_share": self.medium_large_share.value(),
                "manual_small_medium_radius": self.manual_sm.value(),
                "manual_medium_large_radius": self.manual_ml.value(),
                "enabled_size_classes": [
                    name for name, control in (("SMALL", self.generate_small), ("MEDIUM", self.generate_medium), ("LARGE", self.generate_large))
                    if control.isChecked()
                ],
            }
        )
        return M10GenerationConfig.model_validate(candidate)

    def _restore_threshold_recommendations(self) -> None:
        self.threshold_mode.setCurrentIndex(self.threshold_mode.findData("auto"))
        self.small_area_share.setValue(0.10)
        self.medium_large_share.setValue(0.70)
        self.generate_small.setChecked(False)
        self.generate_medium.setChecked(True)
        self.generate_large.setChecked(True)
        self._update_threshold_controls()
        self._estimate_timer.start()

    def _update_threshold_controls(self) -> None:
        auto = self.threshold_mode.currentData() == "auto"
        self.small_area_share.setEnabled(auto)
        self.medium_large_share.setEnabled(auto)
        self.manual_sm.setEnabled(not auto)
        self.manual_ml.setEnabled(not auto)

    def _refresh_estimate(self) -> None:
        try:
            old = self.project.m10_state.deterministic_structures
            self.project.m10_state.deterministic_structures = self._pending_structures
            details = self.service.estimate_details(self._config())
            self._update_threshold_controls()
            self.estimate_label.setText(
                f"Expected: {details['expected_fractures']:,.1f} | final: {details['final_storage_bytes'] / 1024**2:,.1f} MiB | "
                f"peak generation: {details['peak_generation_bytes'] / 1024**2:,.1f} MiB | "
                f"LOD: {details['preview_render_bytes'] / 1024**2:,.1f} MiB | "
                f"save temporary: {details['project_save_temporary_bytes'] / 1024**2:,.1f} MiB | "
                f"all realizations peak: {details['all_realizations_peak_bytes'] / 1024**2:,.1f} MiB"
            )
            rows = [item for target in details.get("targets", []) for item in target.get("classes", [])]
            self.size_estimate_table.setRowCount(len(rows))
            for row, item in enumerate(rows):
                lower, upper = item["radius_range"]
                radius_range = f"{lower:.4g} ≤ R < {'∞' if not isinstance(upper, (int, float)) or upper == float('inf') else f'{upper:.4g}'} m"
                values = (
                    item["size_class"], radius_range, "Yes" if item["generate"] else "No",
                    f"{item['expected_count']:,.1f}", f"{item['target_p32']:.6g}",
                    f"{100 * item['p32_share']:.2f}%", f"{item['expected_count'] * 113 / 1024**2:.2f} MiB",
                )
                for column, value in enumerate(values):
                    self.size_estimate_table.setItem(row, column, QTableWidgetItem(str(value)))
            self._refresh_group_summary(details=details)
        except Exception as exc:
            self.estimate_label.setText(f"Estimate unavailable: {exc}")
            self._refresh_group_summary()
        finally:
            self.project.m10_state.deterministic_structures = old

    def _import_structures(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Deterministic Structures", "", "CSV (*.csv)")
        if not path:
            return
        try:
            self._pending_structures = self.service.import_deterministic_csv(Path(path), commit=False)
            relations = self.service.deterministic_structure_relations(self._pending_structures)
            outside = [item for item in relations if item["relation"] == "OUTSIDE"]
            self.status_label.setText(
                f"Loaded {len(self._pending_structures)} deterministic structures (pending OK); "
                f"{len(outside)} fully outside and excluded by default."
            )
            if outside:
                details = "\n".join(
                    f"{item['structure_id']} | set={item['set_id']} | center={item['center']} | "
                    f"domain={item['generation_domain']}" for item in outside[:20]
                )
                QMessageBox.warning(
                    self, "Deterministic Structures Outside Generation Domain",
                    "The following discs do not intersect the Generation Domain and will not be rendered or counted "
                    f"unless 'retain outside' is enabled:\n{details}",
                )
            self._refresh_estimate()
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))

    def _generate(self) -> None:
        if self._worker is not None:
            return
        try:
            config = self._config()
        except Exception as exc:
            QMessageBox.warning(self, self.tr("Invalid M10 Configuration"), str(exc))
            self.status_label.setText(self.tr("Configuration is invalid; project settings were not changed."))
            return
        old = self.project.m10_state.deterministic_structures
        self.project.m10_state.deterministic_structures = self._pending_structures
        errors = self.service.validate_readiness(config)
        self.project.m10_state.deterministic_structures = old
        if errors:
            QMessageBox.warning(self, "Cannot Generate", "\n".join(errors))
            return
        try:
            details = self.service.estimate_details(config)
            expected = details["expected_fractures"]
            memory = details["peak_generation_bytes"]
        except Exception as exc:
            QMessageBox.critical(self, "Estimate Failed", str(exc))
            return
        if memory > config.memory_warning_bytes:
            answer = QMessageBox.question(
                self, "Large Generation", f"Estimated {expected:,.0f} fractures and {memory / 1024**2:,.1f} MiB. Continue?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._set_running(True)
        self._worker = M10GenerationWorker(self.project, config, self._pending_structures)
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.finished.connect(self._on_generated)
        self._worker.signals.failed.connect(self._on_failed)
        self._worker.signals.cancelled.connect(self._on_cancelled)
        QThreadPool.globalInstance().start(self._worker)

    def _set_running(self, running: bool) -> None:
        language_manager().set_busy(f"m10-generation-{id(self)}", running)
        self.progress.setVisible(running)
        self.generate_button.setEnabled(not running)
        self.import_structures_button.setEnabled(not running)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not running)
        if running:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress.setValue(int(100 * current / max(total, 1)))
        self.status_label.setText(message)

    def _on_generated(self, realizations) -> None:
        self._pending_realizations = list(realizations)
        self._worker = None
        self._set_running(False)
        self.status_label.setText(f"Generated {len(realizations)} complete realization(s); click OK to commit.")
        self._refresh_realizations()

    def _on_failed(self, message: str) -> None:
        self._worker = None
        self._set_running(False)
        QMessageBox.critical(self, "M10 Generation Failed", message)

    def _on_cancelled(self) -> None:
        self._worker = None
        self._set_running(False)
        self.status_label.setText("Generation cancelled; no partial realization was saved.")

    def _display_realizations(self):
        return self._pending_realizations if self._pending_realizations is not None else self.project.m10_state.realizations

    def _refresh_realizations(self) -> None:
        rows = self._display_realizations()
        self.realizations_table.setRowCount(len(rows))
        for row, realization in enumerate(rows):
            quality = realization.quality
            values = (
                realization.realization_id, realization.seed, quality.fracture_count, quality.stochastic_count,
                quality.conditioned_count, quality.deterministic_count, f"{quality.target_p32:.6g}",
                f"{quality.generated_clipped_p32:.6g}", len(quality.warnings), realization.config_hash[:12],
            )
            for column, value in enumerate(values):
                self.realizations_table.setItem(row, column, QTableWidgetItem(str(value)))
        if rows and self.realizations_table.currentRow() < 0:
            self.realizations_table.selectRow(0)
        self._refresh_group_summary()

    def _refresh_group_summary(self, *, details=None) -> None:
        """Show all configured groups, including zero and unresolved outputs."""
        realization = self._selected_realization()
        try:
            rows = self.service.joint_set_diagnostics(
                self._config(), realization, estimate_details=details
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            rows = []
        self.group_summary_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            observations = (
                f"{row['observations']} (cal={row['calibration_observations']}, "
                f"val={row['validation_observations']}, full={row['full_orientation']}, "
                f"dip-only={row['dip_only']})"
            )
            target = row["target_p32"]
            target_text = f"{target:.6g}" if isinstance(target, (int, float)) and target == target else "NO_DATA"
            actual = (
                f"{row['actual']} (random={row['random']}, conditioned={row['conditioned']}, "
                f"deterministic={row['deterministic']})"
            )
            values = (
                row["domain_id"] if row["domain_id"] is not None else "None",
                row["set_id"],
                observations,
                self._orientation_status_text(row["orientation_status"]),
                target_text,
                f"{row['expected']:,.1f}",
                actual,
                self._unresolved_reason_text(row["unresolved_reason"]),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 3:
                    item.setData(Qt.ItemDataRole.UserRole, row["orientation_status"])
                elif column == 7:
                    item.setData(Qt.ItemDataRole.UserRole, row["unresolved_reason"])
                self.group_summary_table.setItem(row_index, column, item)

    @staticmethod
    def _orientation_status_text(status: str) -> str:
        """Translate a stable orientation status for display without changing its code."""
        return {
            "valid": tr("Valid"),
            "invalid": tr("Invalid"),
            "not_applicable": tr("Not Applicable"),
        }.get(status, status)

    @staticmethod
    def _unresolved_reason_text(reason: str) -> str:
        """Translate a stable generation reason for display without changing its code."""
        return {
            "": "-",
            "NO_DATA": tr("No Data"),
            "TARGET_P32_ZERO": tr("Target P32 Zero"),
            "MISSING_SIZE_MODEL": tr("Missing Size Model"),
            "INSUFFICIENT_ORIENTATION_DATA": tr("Insufficient Orientation Data"),
        }.get(reason, reason)

    def _retranslate_dynamic_content(self, _language: str) -> None:
        """Refresh translated table values and legends without recomputing science."""
        for row in range(self.group_summary_table.rowCount()):
            orientation_item = self.group_summary_table.item(row, 3)
            if orientation_item is not None:
                code = orientation_item.data(Qt.ItemDataRole.UserRole)
                if code is not None:
                    orientation_item.setText(self._orientation_status_text(str(code)))
            reason_item = self.group_summary_table.item(row, 7)
            if reason_item is not None:
                code = reason_item.data(Qt.ItemDataRole.UserRole)
                if code is not None:
                    reason_item.setText(self._unresolved_reason_text(str(code)))
        legend_set_ids = self.legend_label.property("joint_set_ids")
        if isinstance(legend_set_ids, list):
            self._set_joint_set_legend([int(value) for value in legend_set_ids])

    def _set_joint_set_legend(self, set_ids: list[int]) -> None:
        """Display a translated legend while retaining stable numeric set IDs."""
        self.legend_label.setProperty("joint_set_ids", list(set_ids))
        labels = [
            tr("Joint Set {set_id}").format(set_id=set_id) + f" {stable_category_hex(set_id)}"
            for set_id in set_ids
        ]
        self.legend_label.setText(tr("Legend") + ": " + " | ".join(labels))

    def _selected_realization(self):
        row = self.realizations_table.currentRow()
        rows = self._display_realizations()
        return rows[row] if 0 <= row < len(rows) else None

    def _render_all(self) -> None:
        realization = self._selected_realization()
        if realization is None or self.layer_manager is None:
            return
        try:
            mode = {
                "joint_set": "Joint Set",
                "domain": "Domain",
                "source": "Source",
                "size_class": "Size Class",
            }[self.color_by.currentData()]
            self.layer_manager.clear_realization(realization.realization_id)
            self.layer_manager.render_realization(
                realization, opacity=self.opacity.value(), color=self._layer_color,
                color_by=mode, category="all", excluded_size_classes=self._hidden_size_classes,
            )
            labels = self._legend_categories(realization, mode)
            if mode == "Joint Set":
                set_ids = sorted(set(map(int, realization.geometry_arrays["set_id"])))
                self._set_joint_set_legend(set_ids)
            else:
                self.legend_label.setProperty("joint_set_ids", None)
                self.legend_label.setText(tr("Legend") + ": " + " | ".join(labels))
            self.status_label.setText(f"LOD color legend: {mode}; colors are stable categorical mappings.")
            self._refresh_layers()
        except Exception as exc:
            QMessageBox.critical(self, "Render Failed", str(exc))

    def _render_exact(self) -> None:
        realization = self._selected_realization()
        if realization is None or self.layer_manager is None:
            return
        try:
            self.layer_manager.render_realization(
                realization,
                opacity=self.opacity.value(),
                color=self._layer_color,
                mode="exact",
            )
            self._refresh_layers()
        except Exception as exc:
            QMessageBox.critical(self, "Exact Render Failed", str(exc))

    def _render_by_set(self) -> None:
        realization = self._selected_realization()
        if realization is None or self.layer_manager is None:
            return
        self.layer_manager.clear_realization(realization.realization_id)
        set_ids = sorted(set(int(value) for value in realization.geometry_arrays["set_id"] if int(value) > 0))
        for set_id in set_ids:
            self.layer_manager.render_realization(
                realization, set_id=set_id, opacity=self.opacity.value(), color=self._layer_color,
                excluded_size_classes=self._hidden_size_classes,
            )
        self._set_joint_set_legend(set_ids)
        self._refresh_layers()

    def _render_source(self, source: str) -> None:
        realization = self._selected_realization()
        if realization is None or self.layer_manager is None:
            return
        self.layer_manager.clear_realization(realization.realization_id)
        self.layer_manager.render_realization(
            realization, source=source, opacity=self.opacity.value(), color=self._layer_color,
            excluded_size_classes=self._hidden_size_classes,
        )
        self._refresh_layers()

    def _toggle_size_class(self) -> None:
        code = int(self.display_size_class.currentData())
        if code in self._hidden_size_classes:
            self._hidden_size_classes.remove(code)
        else:
            self._hidden_size_classes.add(code)
        self._render_all()

    def _legend_categories(self, realization, mode: str) -> list[str]:
        arrays = realization.geometry_arrays
        if mode == "Joint Set":
            return [
                tr("Joint Set {set_id}").format(set_id=value) + f" {stable_category_hex(value)}"
                for value in sorted(set(map(int, arrays["set_id"])))
            ]
        if mode == "Domain":
            palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")
            return [f"Domain {value} {palette[abs(value) % len(palette)]}" for value in sorted(set(map(int, arrays["domain_id"])))]
        if mode == "Source":
            return ["Stochastic #4e79a7", "Conditioned Observation #f28e2b", "Deterministic Structure #e15759"]
        names = {0: "UNKNOWN", 1: "SMALL", 2: "MEDIUM", 3: "LARGE"}
        colors = {0: "#7f7f7f", 1: "#9ecae1", 2: "#fdae6b", 3: "#d62728"}
        present = set(map(int, arrays.get("size_class", [])))
        values = sorted(({1, 2, 3} | ({0} if 0 in present else set())))
        return [
            f"{names[value]}{' (hidden)' if value in self._hidden_size_classes else ''} {colors[value]}"
            for value in values
        ]

    def _refresh_layers(self) -> None:
        layers = self.layer_manager.list_layers() if self.layer_manager is not None else []
        self.layers_table.setRowCount(len(layers))
        for row, layer in enumerate(layers):
            values = ("Yes" if layer.visible else "No", layer.layer_id, layer.realization_id, layer.set_id or "—", layer.source or "all", layer.opacity, layer.fracture_count)
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, layer.layer_id)
                self.layers_table.setItem(row, column, item)
        realization = self._selected_realization()
        total = realization.fracture_count if realization is not None else 0
        visible = sum(
            layer.fracture_count
            for layer in layers
            if layer.visible and realization is not None and layer.realization_id == realization.realization_id
        )
        visible = min(visible, total)
        self.visible_count_label.setText(f"Visible {visible:,} / explicit {total:,}")

    def _selected_layer_id(self) -> str | None:
        row = self.layers_table.currentRow()
        item = self.layers_table.item(row, 0) if row >= 0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _toggle_selected_layer(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is None or self.layer_manager is None:
            return
        record = next((item for item in self.layer_manager.list_layers() if item.layer_id == layer_id), None)
        if record:
            self.layer_manager.set_visible(layer_id, not record.visible)
            self._refresh_layers()

    def _apply_opacity(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is not None and self.layer_manager is not None:
            self.layer_manager.set_opacity(layer_id, self.opacity.value())
            self._refresh_layers()

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self._layer_color = color.name()
            self.color_button.setText(self._layer_color)

    def _remove_selected_layer(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is not None and self.layer_manager is not None:
            self.layer_manager.remove(layer_id)
            self._refresh_layers()

    def _clear_current(self) -> None:
        realization = self._selected_realization()
        if realization is not None and self.layer_manager is not None:
            self.layer_manager.clear_realization(realization.realization_id)
            self._refresh_layers()

    def _clear_all(self) -> None:
        if self.layer_manager is not None:
            self.layer_manager.clear_dfn_layers()
            self._refresh_layers()

    def _export_selected(self) -> None:
        realization = self._selected_realization()
        if realization is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Export M10 Realization")
        if not directory:
            return
        try:
            old_realizations = self.project.m10_state.realizations
            if self._pending_realizations is not None:
                self.project.m10_state.realizations = self._pending_realizations
            paths = self.service.export_realization(realization.realization_id, Path(directory))
            self.status_label.setText(f"Exported {len(paths)} files to {directory}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
        finally:
            self.project.m10_state.realizations = old_realizations

    def accept(self) -> None:
        if self._worker is not None:
            return
        try:
            config = self._config()
        except Exception as exc:
            QMessageBox.warning(self, self.tr("Invalid M10 Configuration"), str(exc))
            self.status_label.setText(self.tr("Configuration is invalid; project settings were not changed."))
            return
        config_changed = config != self.project.m10_state.config
        changed = (
            self._pending_realizations is not None
            or self._pending_structures != self.project.m10_state.deterministic_structures
            or config_changed
        )
        if self._pending_realizations is not None:
            self.project.m10_state.deterministic_structures = list(self._pending_structures)
            self.service.commit_realizations(config, self._pending_realizations, replace=True)
        elif self._pending_structures != self.project.m10_state.deterministic_structures or config_changed:
            self.project.m10_state.deterministic_structures = list(self._pending_structures)
            self.service.commit_config(config)
        self.committed_changes = changed
        if self.project.m10_state.realizations:
            self.workflow.complete_step("explicit_dfn")
        super().accept()

    def reject(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            return
        if self._pending_realizations is not None and self.layer_manager is not None:
            for realization in self._pending_realizations:
                self.layer_manager.clear_realization(realization.realization_id)
        super().reject()

    def closeEvent(self, event) -> None:
        """Cancel an active worker and prevent an orphaned background operation."""
        if self._worker is not None:
            self._worker.cancel()
            event.ignore()
            return
        super().closeEvent(event)
