"""M9 generic continuous scalar-parameter import, kriging, and slice UI."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, KrigingSettings, NonNegativePolicy, VariogramMode, VariogramModel
from dfn_cave_studio.services.scalar_field_service import ScalarParameterFieldService, canonical_parameter_name
from dfn_cave_studio.ui.qt_adapter import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QColor, QLabel, QMessageBox, QPainter, QPen, QProgressBar, QPushButton, QScrollArea, QSpinBox, Qt,
    QTableWidget, QTableWidgetItem, QThreadPool,
    QVBoxLayout, QWidget,
)
from dfn_cave_studio.workers.m9_worker import M9Worker
from dfn_cave_studio.voxel.resource_estimate import available_system_memory_bytes, format_resource_estimate


class _VariogramPlot(QWidget):
    """Small dependency-free plot of experimental and fitted semivariances."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lags = []
        self.setMinimumHeight(160)

    def set_diagnostics(self, diagnostics) -> None:
        self._lags = [] if diagnostics is None else list(diagnostics.lags)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        left, top, right, bottom = 46, 12, 12, 28
        width = max(1, self.width() - left - right)
        height = max(1, self.height() - top - bottom)
        painter.setPen(QPen(QColor("#404040"), 1))
        painter.drawLine(left, top, left, top + height)
        painter.drawLine(left, top + height, left + width, top + height)
        values = [
            value
            for lag in self._lags
            for value in (lag.experimental_semivariance, lag.fitted_semivariance)
            if value is not None
        ]
        if not self._lags or not values:
            painter.drawText(left + 8, top + 20, "No variogram diagnostics")
            return
        max_distance = max(lag.distance for lag in self._lags) or 1.0
        max_value = max(values) or 1.0

        def point(distance: float, value: float) -> tuple[int, int]:
            return (
                int(left + width * distance / max_distance),
                int(top + height * (1.0 - value / max_value)),
            )

        fitted_points = []
        painter.setPen(QPen(QColor("#1f77b4"), 1))
        for lag in self._lags:
            if lag.experimental_semivariance is not None:
                x, y = point(lag.distance, lag.experimental_semivariance)
                painter.drawEllipse(x - 3, y - 3, 6, 6)
            if lag.fitted_semivariance is not None:
                fitted_points.append(point(lag.distance, lag.fitted_semivariance))
        painter.setPen(QPen(QColor("#d62728"), 2))
        for first, second in zip(fitted_points, fitted_points[1:]):
            painter.drawLine(*first, *second)


class M9ScalarFieldDialog(QDialog):
    """Manage auxiliary fields without changing DFN-input workflow state."""

    def __init__(self, project, layer_manager=None, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.service = ScalarParameterFieldService(project)
        self.layer_manager = layer_manager
        self._before_samples = deepcopy(project.m9_state.scalar_samples)
        self._before_fields = list(project.m9_state.scalar_fields)
        self._before_database = deepcopy(project.borehole_database)
        self._worker = None
        self._discard_worker_result = False
        self._discarded_worker_ids: set[int] = set()
        self.committed_changes = False
        self.dfn_inputs_changed = False
        self.setWindowTitle("M9 Physical Parameter Fields / M9 通用物理参数场")
        self.resize(900, 720)
        root_layout = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); content = QWidget(); layout = QVBoxLayout(content)
        scroll.setWidget(content); root_layout.addWidget(scroll)
        top = QHBoxLayout()
        self.import_button = QPushButton("Import long-table CSV/XLSX / 导入长表")
        self.import_button.clicked.connect(self._import)
        self.parameter = QComboBox(); self.parameter.currentIndexChanged.connect(self._refresh_results)
        top.addWidget(self.import_button); top.addWidget(QLabel("Parameter / 参数")); top.addWidget(self.parameter, 1)
        layout.addLayout(top)
        form = QFormLayout()
        self.method = QComboBox()
        self.method.addItem("Global constant", DensityMethod.GLOBAL_CONSTANT.value)
        self.method.addItem("IDW", DensityMethod.IDW.value)
        self.method.addItem("Ordinary Kriging", DensityMethod.ORDINARY_KRIGING.value)
        self.mode = QComboBox(); self.mode.addItem("Auto", VariogramMode.AUTO.value); self.mode.addItem("Manual", VariogramMode.MANUAL.value)
        self.model = QComboBox()
        for item in VariogramModel: self.model.addItem(item.value.title(), item.value)
        self.nugget = QDoubleSpinBox(); self.nugget.setRange(0, 1e12); self.nugget.setDecimals(6)
        self.power = QDoubleSpinBox(); self.power.setRange(0.1, 10.0); self.power.setValue(2.0)
        self.sill = QDoubleSpinBox(); self.sill.setRange(1e-12, 1e12); self.sill.setValue(1.0); self.sill.setDecimals(6)
        self.variogram_range = QDoubleSpinBox(); self.variogram_range.setRange(1e-9, 1e12); self.variogram_range.setValue(100.0)
        self.lags = QSpinBox(); self.lags.setRange(3, 100); self.lags.setValue(12)
        self.minimum = QSpinBox(); self.minimum.setRange(1, 1000); self.minimum.setValue(3)
        self.maximum = QSpinBox(); self.maximum.setRange(2, 1000); self.maximum.setValue(24)
        self.radius = QDoubleSpinBox(); self.radius.setRange(0, 1e12); self.radius.setSpecialValueText("Unlimited")
        self.policy = QComboBox(); self.policy.addItem("Reject out-of-range predictions", NonNegativePolicy.REJECT.value)
        self.policy.addItem("Clip to physical bounds with audit", NonNegativePolicy.CLIP_WITH_AUDIT.value)
        self.memory_budget_gib = QDoubleSpinBox(); self.memory_budget_gib.setRange(0.25, 512.0)
        self.memory_budget_gib.setValue(2.0); self.memory_budget_gib.setDecimals(2)
        form.addRow("Interpolation method", self.method); form.addRow("IDW power", self.power)
        form.addRow("Variogram mode", self.mode); form.addRow("Variogram", self.model)
        form.addRow("Nugget", self.nugget); form.addRow("Sill", self.sill); form.addRow("Range (m)", self.variogram_range)
        form.addRow("Lag count", self.lags); form.addRow("Min / max neighbours", self._pair(self.minimum, self.maximum))
        form.addRow("Search radius (m)", self.radius); form.addRow("Prediction bound policy", self.policy)
        form.addRow("Safe memory budget (GiB)", self.memory_budget_gib)
        layout.addLayout(form)
        controls = QHBoxLayout()
        self.build_button = QPushButton("Build field / 建立参数场"); self.build_button.clicked.connect(self._build)
        self.cancel_button = QPushButton("Cancel computation / 取消计算"); self.cancel_button.clicked.connect(self._cancel); self.cancel_button.setEnabled(False)
        self.display = QComboBox(); self.display.addItem("Estimate / 估计值", "estimate"); self.display.addItem("Kriging Variance / 克里金方差", "kriging_variance")
        self.display_mode = QComboBox(); self.display_mode.addItem("Outer Surface Cloud / 外围云图", "outer_surface")
        self.display_mode.addItem("X/Y/Z Slice / 正交切片", "orthogonal_section"); self.display_mode.addItem("Box Cutaway / 盒式剖切", "box_cutaway")
        self.interpolation = QComboBox(); self.interpolation.addItem("Exact Cell Colours", "exact"); self.interpolation.addItem("Smooth Display", "smooth")
        self.grid_lines = QCheckBox("Show Grid Lines / 显示网格线")
        self.axis = QComboBox()
        for axis in "xyz": self.axis.addItem(axis.upper(), axis)
        self.fraction = QDoubleSpinBox(); self.fraction.setRange(0, 1); self.fraction.setSingleStep(0.05); self.fraction.setValue(0.5)
        self.render_button = QPushButton("Render Slice / 渲染切片"); self.render_button.clicked.connect(self._render)
        self.export_button = QPushButton("Export / 导出"); self.export_button.clicked.connect(self._export)
        for widget in (self.build_button, self.cancel_button, self.display, self.display_mode, self.interpolation,
                       self.axis, self.fraction, self.grid_lines, self.render_button, self.export_button): controls.addWidget(widget)
        layout.addLayout(controls)
        self.box_group = QGroupBox("Box bounds / 盒式范围"); box_row = QHBoxLayout(self.box_group); self.box_values = []
        for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            box_row.addWidget(QLabel(name)); spin = QDoubleSpinBox(); spin.setRange(-1e12, 1e12); spin.setDecimals(4); self.box_values.append(spin); box_row.addWidget(spin)
        layout.addWidget(self.box_group); self.display_mode.currentIndexChanged.connect(self._display_mode_changed)
        self.progress = QProgressBar(); self.progress.setVisible(False); layout.addWidget(self.progress)
        self.summary = QLabel("No scalar parameter selected"); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.variogram_plot = _VariogramPlot(); layout.addWidget(self.variogram_plot)
        self.variogram_table = QTableWidget(0, 4); self.variogram_table.setHorizontalHeaderLabels(["Distance", "Pairs", "Experimental", "Fitted"])
        layout.addWidget(self.variogram_table)
        self.validation_table = QTableWidget(0, 7)
        self.validation_table.setHorizontalHeaderLabels(["Borehole", "From", "To", "Observed", "Predicted", "Residual", "Variance"])
        layout.addWidget(self.validation_table)
        self.layers = QTableWidget(0, 3); self.layers.setHorizontalHeaderLabels(["Visible", "Layer", "Opacity"]); layout.addWidget(self.layers)
        layer_buttons = QHBoxLayout(); self.toggle_layer = QPushButton("Show/Hide Selected"); self.remove_layer = QPushButton("Remove Selected")
        self.clear_layers = QPushButton("Clear Scalar Layers"); self.toggle_layer.clicked.connect(self._toggle_layer); self.remove_layer.clicked.connect(self._remove_layer); self.clear_layers.clicked.connect(self._clear_layers)
        for button in (self.toggle_layer, self.remove_layer, self.clear_layers): layer_buttons.addWidget(button)
        layout.addLayout(layer_buttons)
        self.dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.dialog_buttons.accepted.connect(self.accept)
        self.dialog_buttons.rejected.connect(self.reject)
        root_layout.addWidget(self.dialog_buttons)
        self.method.currentIndexChanged.connect(self._update_controls); self.mode.currentIndexChanged.connect(self._update_controls)
        self._refresh_parameters(); self._update_controls(); self._display_mode_changed(); self._refresh_layers()

    @staticmethod
    def _pair(first, second):
        group = QGroupBox(); row = QHBoxLayout(group); row.setContentsMargins(0, 0, 0, 0); row.addWidget(first); row.addWidget(second); return group

    def _update_controls(self) -> None:
        kriging = self.method.currentData() == DensityMethod.ORDINARY_KRIGING.value
        manual = self.mode.currentData() == VariogramMode.MANUAL.value
        idw = self.method.currentData() == DensityMethod.IDW.value
        self.minimum.setMinimum(2 if kriging else 1)
        self.power.setEnabled(idw)
        for widget in (self.minimum, self.maximum, self.radius): widget.setEnabled(kriging or idw)
        for widget in (self.mode, self.model, self.lags, self.policy): widget.setEnabled(kriging)
        for widget in (self.nugget, self.sill, self.variogram_range): widget.setEnabled(kriging and manual)

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import scalar parameter table", "", "Data (*.csv *.xlsx *.xls)")
        if not path: return
        try:
            rows = self.service.import_table(path); self.committed_changes = bool(rows); self._refresh_parameters()
        except Exception as error:
            QMessageBox.critical(self, "Scalar import failed", str(error))

    def _settings(self) -> DensitySettings:
        method = DensityMethod(self.method.currentData())
        kriging = KrigingSettings()
        if method == DensityMethod.ORDINARY_KRIGING:
            kriging = KrigingSettings(
                mode=self.mode.currentData(), model=self.model.currentData(), nugget=self.nugget.value(),
                sill=self.sill.value(), range=self.variogram_range.value(), lag_count=self.lags.value(),
                minimum_neighbors=self.minimum.value(), maximum_neighbors=self.maximum.value(),
                search_radius=self.radius.value() or None, non_negative_policy=self.policy.currentData(),
            )
        return DensitySettings(
            method=method, power=self.power.value(), search_radius=self.radius.value() or None,
            min_neighbors=self.minimum.value(), max_neighbors=self.maximum.value(), kriging=kriging,
        )

    def _build(self) -> None:
        if self.parameter.currentData() is None: return
        try: settings = self._settings()
        except Exception as error: QMessageBox.critical(self, "Invalid interpolation settings", str(error)); return
        estimate = None
        if self.project.spatial_grid_config is not None:
            try:
                estimate = self.service.estimate_resources(
                    budget_bytes=int(self.memory_budget_gib.value() * 1024**3),
                    system_available_bytes=available_system_memory_bytes(),
                )
            except Exception as error:
                QMessageBox.critical(self, "Resource estimation failed", str(error))
                return
        if estimate is not None:
            self.summary.setText(format_resource_estimate(estimate))
        if estimate is not None and estimate.exceeds_budget:
            answer = QMessageBox.warning(
                self,
                "Large scalar field requires confirmation",
                format_resource_estimate(estimate)
                + "\n\nThis may exhaust system memory. Continue with the configured grid unchanged?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        # A cancellation only discards the result of the worker that was active at
        # that time.  A later, explicitly started computation owns a new result.
        self._discard_worker_result = False
        self._set_running(True)
        worker = M9Worker(
            lambda progress, cancelled: self.service.build(
                self.parameter.currentData(), settings, progress=progress, cancelled=cancelled
            )
        )
        self._worker = worker
        worker.signals.progress.connect(self._progress)
        worker.signals.finished.connect(lambda result, owner=worker: self._done(result, owner))
        worker.signals.cancelled.connect(lambda owner=worker: self._worker_cancelled(owner))
        worker.signals.failed.connect(lambda message, owner=worker: self._failed(message, owner))
        QThreadPool.globalInstance().start(worker)

    def _progress(self, current: int, total: int) -> None: self.progress.setRange(0, total); self.progress.setValue(current)
    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running); self.build_button.setEnabled(not running); self.import_button.setEnabled(not running); self.cancel_button.setEnabled(running)
        ok_button = self.dialog_buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(not running)
        if not running: self._worker = None
    def _cancel(self) -> None:
        # Set the discard guard before requesting cancellation.  The worker may
        # already have produced a candidate and deliver ``finished`` immediately.
        self._discard_worker_result = True
        if self._worker is not None:
            self._discarded_worker_ids.add(id(self._worker))
            self._worker.cancel()
    def _done(self, result, worker=None) -> None:
        discarded = id(worker) in self._discarded_worker_ids if worker is not None else self._discard_worker_result
        if discarded:
            self._discarded_worker_ids.discard(id(worker))
            if worker is None or worker is self._worker:
                self._set_running(False)
            return
        if worker is not None and worker is not self._worker:
            return
        self.service.commit(result)
        self.committed_changes = True
        self.dfn_inputs_changed = canonical_parameter_name(result.metadata.parameter_name) == "p32"
        self._set_running(False)
        self._refresh_results()
    def _worker_cancelled(self, worker) -> None:
        self._discarded_worker_ids.discard(id(worker))
        if worker is self._worker:
            self._set_running(False)

    def _failed(self, message: str, worker=None) -> None:
        discarded = id(worker) in self._discarded_worker_ids if worker is not None else self._discard_worker_result
        self._discarded_worker_ids.discard(id(worker))
        if worker is None or worker is self._worker:
            self._set_running(False)
        if not discarded:
            QMessageBox.critical(self, "Scalar field generation failed", message)

    def _refresh_parameters(self) -> None:
        selected = self.parameter.currentData(); self.parameter.clear()
        names_by_key = {}
        for item in self.project.m9_state.scalar_samples:
            names_by_key.setdefault(canonical_parameter_name(item.parameter_name), item.parameter_name)
        names = sorted(names_by_key.values(), key=str.casefold)
        for name in names: self.parameter.addItem(name, name)
        if selected is not None: self.parameter.setCurrentIndex(self.parameter.findData(selected))
        self._refresh_results()

    def _current_result(self):
        name = self.parameter.currentData()
        return next(
            (
                item
                for item in self.project.m9_state.scalar_fields
                if canonical_parameter_name(item.metadata.parameter_name) == canonical_parameter_name(name or "")
            ),
            None,
        )

    def _refresh_results(self, *_args) -> None:
        result = self._current_result(); self.variogram_table.setRowCount(0); self.validation_table.setRowCount(0)
        if result is None:
            self.variogram_plot.set_diagnostics(None)
            self.summary.setText("No generated field for the selected parameter")
            return
        meta = result.metadata
        bounds = [meta.origin[0], meta.origin[0] + meta.shape[0] * meta.spacing[0], meta.origin[1],
                  meta.origin[1] + meta.shape[1] * meta.spacing[1], meta.origin[2], meta.origin[2] + meta.shape[2] * meta.spacing[2]]
        for spin, value in zip(self.box_values, bounds): spin.setValue(value)
        self.summary.setText(f"{meta.parameter_name} [{meta.unit}] — {meta.method.value}; validation n={result.validation_summary.predicted_count}; "
                             f"rejected={meta.rejected_voxel_count}; clipped={meta.clipped_voxel_count}. "
                             "Kriging variance is conditional on the fitted variogram, not total geological uncertainty.")
        diagnostics = next(iter(meta.variograms.values()), None)
        self.variogram_plot.set_diagnostics(diagnostics)
        if diagnostics is not None:
            self.variogram_table.setRowCount(len(diagnostics.lags))
            for row, lag in enumerate(diagnostics.lags):
                for column, value in enumerate((lag.distance, lag.pair_count, lag.experimental_semivariance, lag.fitted_semivariance)):
                    self.variogram_table.setItem(row, column, QTableWidgetItem("" if value is None else f"{value:.6g}"))
        self.validation_table.setRowCount(len(result.validation_results))
        for row, item in enumerate(result.validation_results):
            values = (item.borehole_id, item.from_depth, item.to_depth, item.observed, item.predicted, item.residual, item.kriging_variance)
            for column, value in enumerate(values): self.validation_table.setItem(row, column, QTableWidgetItem("" if value is None else str(value)))

    def _render(self) -> None:
        result = self._current_result(); plotter = getattr(self.parent(), "_plotter", None)
        if result is None or plotter is None or self.layer_manager is None: return
        from dfn_cave_studio.visualization.m11_voxel_renderer import M11DisplayConfig
        from dfn_cave_studio.visualization.scalar_field_renderer import ScalarFieldRenderer
        config = M11DisplayConfig(display_mode=self.display_mode.currentData(), interpolation_mode=self.interpolation.currentData(),
            axis=self.axis.currentData(), fraction=self.fraction.value(), show_grid_lines=self.grid_lines.isChecked(),
            box_bounds=tuple(spin.value() for spin in self.box_values) if self.display_mode.currentData() == "box_cutaway" else None)
        try:
            ScalarFieldRenderer().render(self.layer_manager, result.metadata, result.arrays, result.metadata.field_id,
                                         self.display.currentData(), result.metadata.unit, config)
            plotter.render(); self._refresh_layers()
        except Exception as error: QMessageBox.critical(self, "Scalar field rendering failed", str(error))

    def _display_mode_changed(self, *_args) -> None:
        mode = self.display_mode.currentData(); self.box_group.setVisible(mode == "box_cutaway"); self.axis.setEnabled(mode == "orthogonal_section"); self.fraction.setEnabled(mode == "orthogonal_section")

    def _selected_layer_id(self):
        row = self.layers.currentRow(); item = self.layers.item(row, 1) if row >= 0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _refresh_layers(self) -> None:
        records = [] if self.layer_manager is None else [r for r in self.layer_manager.list_layers() if r.layer_id.startswith("m9_slice:scalar_")]
        self.layers.setRowCount(len(records))
        for row, record in enumerate(records):
            visible = QTableWidgetItem("Yes" if record.visible else "No"); name = QTableWidgetItem(record.layer_id); name.setData(Qt.ItemDataRole.UserRole, record.layer_id)
            self.layers.setItem(row, 0, visible); self.layers.setItem(row, 1, name); self.layers.setItem(row, 2, QTableWidgetItem(f"{record.opacity:.2f}"))

    def _toggle_layer(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id and self.layer_manager:
            record = self.layer_manager.get(layer_id); self.layer_manager.set_visible(layer_id, not record.visible); self._refresh_layers()

    def _remove_layer(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id and self.layer_manager: self.layer_manager.remove(layer_id); self._refresh_layers()

    def _clear_layers(self) -> None:
        if self.layer_manager:
            for record in list(self.layer_manager.list_layers()):
                if record.layer_id.startswith("m9_slice:scalar_"): self.layer_manager.remove(record.layer_id)
            self._refresh_layers()

    def _export(self) -> None:
        result = self._current_result()
        if result is None: return
        directory = QFileDialog.getExistingDirectory(self, "Export scalar parameter field")
        if not directory: return
        try: self.service.export(result, Path(directory))
        except Exception as error: QMessageBox.critical(self, "Scalar field export failed", str(error))

    def accept(self) -> None:
        """Close only when no background result can arrive after acceptance."""
        if self._worker is not None:
            return
        super().accept()

    def reject(self) -> None:
        self._discard_worker_result = True
        if self._worker is not None:
            self._discarded_worker_ids.add(id(self._worker))
            self._worker.cancel()
        self.project.m9_state.scalar_samples = self._before_samples; self.project.m9_state.scalar_fields = self._before_fields
        self.project.borehole_database = self._before_database
        super().reject()
